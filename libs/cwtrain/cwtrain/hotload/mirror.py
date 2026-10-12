# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""A trainer-side, byte-exact mirror of the weights a hot-load deployment serves.

slime and miles publish weight updates as XOR deltas against the previous version. The engine
applies those deltas to the chain root to rebuild the full checkpoint, so the trainer has no copy
of the exact bytes the engine serves. :class:`WeightMirror` keeps that copy. It copies the base
checkpoint once, then applies every delta the trainer writes. It verifies each changed tensor
against the Adler-32 checksum the trainer stored in the delta shard.

The mirror gives a long run two things:

* **Chain cuts.** A full checkpoint at any version, so a chain of deltas can be restarted with a
  FULL snapshot every N versions (:meth:`WeightMirror.write_full_snapshot`). The snapshot carries
  per-tensor checksums, which the engine verifies on install.
* **Verification.** A manifest of every tensor's checksum for every published version
  (:meth:`WeightMirror.manifest`). The engine's materialized checkpoint must match it.

The module needs ``numpy`` and ``zstandard``. Both are core dependencies of cwtrain.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import struct
import threading
import time
import zlib
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from cwtrain.errors import CWTrainError
from cwtrain.hotload.errors import SnapshotError
from cwtrain.hotload.snapshot import CONFIG_FILE, INDEX_FILE, SHARD_SUFFIX, load_index

log = logging.getLogger(__name__)

STATE_FILE = "mirror-state.json"
CHECKSUMS_FILE = "mirror-checksums.json"
#: Staging directory for a delta being applied. It holds the new bytes of every changed tensor,
#: then the journal that commits them once they are durable.
PENDING_DIR = ".pending"
JOURNAL_FILE = "commit.json"
CHECKSUM_FORMAT = "adler32"
_COPY_CHUNK = 64 * 2**20
#: Read size when a shard is copied and checksummed in one pass. One buffer per worker thread.
_STREAM_CHUNK = 16 * 2**20
#: Seconds between progress lines while a checkpoint is copied.
_PROGRESS_INTERVAL = 60.0
#: ``os.copy_file_range`` errors that mean "not available here", not an I/O failure: a
#: cross-file-system copy (before Linux 5.3, or between file systems that cannot share), no kernel
#: support, or a file system that refuses the call. The copy then falls back to userspace.
_COPY_RANGE_FALLBACK = frozenset(
    getattr(errno, name)
    for name in ("EXDEV", "ENOSYS", "EOPNOTSUPP", "ENOTSUP", "EINVAL")
    if hasattr(errno, name)
)

ProgressCallback = Callable[[int], None]


class MirrorError(CWTrainError):
    """The mirror could not apply a delta, or a checksum did not match."""


def adler32_hex(buf: Any) -> str:
    """Adler-32 of ``buf`` as the 8-digit hex string slime, miles and the engine agree on."""
    return f"{zlib.adler32(buf):08x}"


@dataclass(frozen=True)
class TensorLocation:
    """Where one tensor's raw bytes live inside a safetensors checkpoint directory."""

    path: Path
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


@dataclass
class DeltaReport:
    """What :meth:`WeightMirror.apply_delta` did for one version directory."""

    version: str
    base_version: str
    tensors_changed: int = 0
    bytes_changed: int = 0
    wire_bytes: int = 0
    changed: list[str] = field(default_factory=list)


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    """Parse a safetensors file header. Returns ``(header, data_start)``."""
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise SnapshotError(f"{path} is not a safetensors file")
        (header_len,) = struct.unpack("<Q", raw)
        try:
            header = json.loads(f.read(header_len))
        except ValueError as exc:
            raise SnapshotError(f"{path} has an invalid safetensors header") from exc
    if not isinstance(header, dict):
        raise SnapshotError(f"{path} has an invalid safetensors header")
    return header, 8 + header_len


def _shard_tensors(shard: Path) -> dict[str, TensorLocation]:
    header, data_start = read_safetensors_header(shard)
    tensors: dict[str, TensorLocation] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        tensors[name] = TensorLocation(
            shard, data_start + begin, end - begin, info["dtype"], tuple(info["shape"])
        )
    return tensors


def tensor_locations(root: Path) -> dict[str, TensorLocation]:
    """Index every tensor of every ``*.safetensors`` shard under ``root``."""
    locations: dict[str, TensorLocation] = {}
    for shard in sorted(root.glob(f"*{SHARD_SUFFIX}")):
        locations.update(_shard_tensors(shard))
    return locations


def _encode_header(header: dict[str, Any]) -> bytes:
    # Pad the JSON to 8-byte alignment with spaces, as the reference implementation does.
    body = json.dumps(header, separators=(",", ":")).encode()
    body += b" " * (-len(body) % 8)
    return struct.pack("<Q", len(body)) + body


def rewrite_shard_metadata(
    src: Path,
    dst: Path,
    metadata: dict[str, str],
    *,
    progress: ProgressCallback | None = None,
) -> None:
    """Copy ``src`` to ``dst`` with ``metadata`` merged into the header's ``__metadata__``.

    Tensor data is copied verbatim, so ``data_offsets`` stay valid. ``progress`` is called with
    the number of data bytes copied as the copy advances.
    """
    header, data_start = read_safetensors_header(src)
    merged = dict(header.get("__metadata__") or {})
    merged.update({k: str(v) for k, v in metadata.items()})
    header = {"__metadata__": merged, **{k: v for k, v in header.items() if k != "__metadata__"}}
    head = _encode_header(header)
    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(head)
        size = os.fstat(fin.fileno()).st_size
        _copy_range(fin, fout, data_start, len(head), max(0, size - data_start), progress)


def _workers(workers: int | None) -> int:
    return max(1, min(16, workers or (os.cpu_count() or 4)))


def _parallel(fn: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    """``[fn(item) for item in items]`` on up to ``workers`` threads, results in item order.

    When an item fails, items not yet started are cancelled and the failure is raised. Items
    already running are always waited for, so nothing is still writing when this returns.
    """
    pool = ThreadPoolExecutor(max_workers=min(workers, max(1, len(items))))
    try:
        futures = [pool.submit(fn, item) for item in items]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for future in futures:
            if future in done and future.exception() is not None:
                future.result()  # raises it
        return [future.result() for future in futures]
    finally:
        # On failure or interrupt: cancel what has not started, let the rest finish.
        pool.shutdown(wait=True, cancel_futures=True)


def _total_size(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths)


class _Progress:
    """Counts copied bytes across threads and logs at most every ``_PROGRESS_INTERVAL`` seconds."""

    def __init__(self, what: str, total: int) -> None:
        self.what, self.total, self.done = what, total, 0
        self.started = self._logged = time.monotonic()
        self._lock = threading.Lock()

    def __call__(self, nbytes: int) -> None:
        with self._lock:
            self.done += nbytes
            now = time.monotonic()
            if now - self._logged < _PROGRESS_INTERVAL:
                return
            self._logged = now
            done, elapsed = self.done, now - self.started
        log.info(
            "%s: %.1f/%.1f GiB (%.0f MiB/s)",
            self.what,
            done / 2**30,
            self.total / 2**30,
            done / 2**20 / max(elapsed, 1e-9),
        )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


def _copy_range(
    fin: BinaryIO,
    fout: BinaryIO,
    src_offset: int,
    dst_offset: int,
    length: int,
    progress: ProgressCallback | None = None,
) -> None:
    """Copy ``length`` bytes of ``fin`` at ``src_offset`` into ``fout`` at ``dst_offset``.

    On Linux, ``os.copy_file_range`` moves the bytes in the kernel. File systems that share
    extents (XFS, Btrfs) may share them instead of copying when the offsets line up. If the call
    is missing or refused, the rest goes through a read/write loop. A source that ends early is
    copied up to its end, like ``shutil.copyfileobj``.
    """
    fout.flush()  # buffered bytes (a header) must reach the file before the kernel writes after it
    copy_file_range = getattr(os, "copy_file_range", None)
    while length > 0 and copy_file_range is not None:
        try:
            n = copy_file_range(fin.fileno(), fout.fileno(), length, src_offset, dst_offset)
        except OSError as exc:
            if exc.errno not in _COPY_RANGE_FALLBACK:
                raise
            break
        if n <= 0:  # end of source, or the file system copied nothing: finish in the loop below
            break
        src_offset, dst_offset, length = src_offset + n, dst_offset + n, length - n
        if progress:
            progress(n)
    if length <= 0:
        return
    fin.seek(src_offset)
    fout.seek(dst_offset)
    while length > 0:
        chunk = fin.read(min(_COPY_CHUNK, length))
        if not chunk:
            return
        fout.write(chunk)
        length -= len(chunk)
        if progress:
            progress(len(chunk))


def _copy_shard(src: Path, dst: Path, progress: ProgressCallback | None = None) -> None:
    """Copy one shard as it is, data and file metadata, like ``shutil.copy2``."""
    with src.open("rb") as fin, dst.open("wb") as fout:
        _copy_range(fin, fout, 0, 0, os.fstat(fin.fileno()).st_size, progress)
    shutil.copystat(src, dst)


def _tensor_spans(header: dict[str, Any], data_start: int) -> list[tuple[int, int, str]] | None:
    """Every tensor's absolute ``(begin, end, name)`` in file order, or ``None`` when the ranges
    are not the well-formed, non-overlapping kind :func:`_copy_and_checksum` can stream."""
    spans: list[tuple[int, int, str]] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        if not (isinstance(begin, int) and isinstance(end, int) and 0 <= begin <= end):
            return None
        spans.append((data_start + begin, data_start + end, name))
    spans.sort()
    if any(nxt[0] < prev[1] for prev, nxt in zip(spans, spans[1:], strict=False)):
        return None
    return spans


def _copy_and_checksum(
    src: Path, dst: Path, progress: ProgressCallback | None = None
) -> dict[str, str]:
    """Copy one shard and return the Adler-32 of every tensor in it, reading the shard once.

    The checksums equal :func:`adler32_hex` of each tensor's bytes in the copy, including empty
    tensors (``00000001``) and tensors cut short by the end of the file. If the header's ranges
    overlap or run backwards, the shard is copied first and then hashed tensor by tensor.
    """
    header, data_start = read_safetensors_header(src)
    spans = _tensor_spans(header, data_start)
    if spans is None:
        _copy_shard(src, dst, progress)
        return {name: adler32_hex(_read(loc)) for name, loc in _shard_tensors(dst).items()}
    checksums: dict[str, str] = {}
    k, running, pos = 0, zlib.adler32(b""), 0
    buf = bytearray(_STREAM_CHUNK)
    view = memoryview(buf)
    with src.open("rb", buffering=0) as fin, dst.open("wb") as fout:
        while n := fin.readinto(buf):
            fout.write(view[:n])
            end_of_chunk = pos + n
            while k < len(spans):
                begin, end, name = spans[k]
                if begin >= end_of_chunk:
                    break  # starts in a later chunk (an empty tensor here starts in the next)
                lo, hi = max(begin, pos), min(end, end_of_chunk)
                if hi > lo:
                    running = zlib.adler32(view[lo - pos : hi - pos], running)
                if end > end_of_chunk:
                    break  # the tensor continues in the next chunk
                checksums[name] = f"{running:08x}"
                k, running = k + 1, zlib.adler32(b"")
            pos = end_of_chunk
            if progress:
                progress(n)
    # Tensors cut short by the end of the file: checksum the bytes that exist, like a short read.
    for _begin, _end, name in spans[k:]:
        checksums[name] = f"{running:08x}"
        running = zlib.adler32(b"")
    shutil.copystat(src, dst)
    return checksums


class WeightMirror:
    """A full HF checkpoint directory kept byte-identical to what the trainer has published.

    Create it once per run from the trainer's ``--hf-checkpoint`` with :meth:`create`, then call
    :meth:`apply_delta` for every ``weight_vNNNNNN`` the trainer writes. Reopen an existing
    mirror with ``WeightMirror(root)``.
    """

    def __init__(self, root: str | os.PathLike[str], *, workers: int | None = None) -> None:
        self.root = Path(root)
        self.workers = _workers(workers)
        if not (self.root / STATE_FILE).is_file():
            raise MirrorError(f"{self.root} is not a weight mirror; call WeightMirror.create")
        self.state: dict[str, Any] = json.loads((self.root / STATE_FILE).read_text())
        self.checksums: dict[str, str] = json.loads((self.root / CHECKSUMS_FILE).read_text())
        self._locations: dict[str, TensorLocation] | None = None
        self._recover()

    # ---------------------------------------------------------------------- lifecycle

    @classmethod
    def create(
        cls,
        base: str | os.PathLike[str],
        root: str | os.PathLike[str],
        *,
        version: str = "0",
        checksums: Mapping[str, str] | None = None,
        workers: int | None = None,
    ) -> WeightMirror:
        """Copy the full checkpoint ``base`` to ``root`` and checksum every tensor.

        Each shard is read once and is copied and checksummed in the same pass. Up to
        ``workers`` shards (default: the CPU count, at most 16) are copied at a time. The returned
        mirror uses the same number of workers.

        An existing mirror at ``root`` is replaced. A mirror is valid only for the run that
        created it, because the trainer's deltas restart from the base at every launch.

        Args:
            checksums: Every tensor's Adler-32 in ``base``, if already computed (for example with
                :func:`~cwtrain.hotload.delta.checkpoint_checksums`). Skips hashing the copy.
        """
        base, root = Path(base), Path(root)
        if not (base / INDEX_FILE).is_file() or not (base / CONFIG_FILE).is_file():
            raise SnapshotError(f"{base} is not a full HF checkpoint (needs config.json + index)")
        load_index(base)
        if root.exists():
            shutil.rmtree(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        tmp = root.with_name(root.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        workers = _workers(workers)
        shards = sorted(p for p in base.glob(f"*{SHARD_SUFFIX}") if p.is_file())
        shard_names = {p.name for p in shards}
        top = os.fspath(base)

        def not_the_shards(directory: str, names: list[str]) -> list[str]:
            return [n for n in names if n in shard_names] if directory == top else []

        progress = _Progress(f"copying {base} to the weight mirror", _total_size(shards))
        computed: dict[str, str] = {}
        try:
            # Copy everything except the top-level shards (nested directories included), then
            # copy the shards, which hold nearly all the bytes, in parallel.
            shutil.copytree(base, tmp, ignore=not_the_shards)
            if checksums is None:
                copies = _parallel(
                    lambda s: _copy_and_checksum(s, tmp / s.name, progress), shards, workers
                )
                for shard_checksums in copies:  # in shard order: a later shard wins
                    computed.update(shard_checksums)
            else:
                _parallel(lambda s: _copy_shard(s, tmp / s.name, progress), shards, workers)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        locations = tensor_locations(tmp)
        if not locations:
            shutil.rmtree(tmp)
            raise SnapshotError(f"{base} has no tensors")
        if checksums is None:
            checksums = computed
        elif set(checksums) == set(locations):
            checksums = dict(checksums)
        else:
            shutil.rmtree(tmp)
            raise SnapshotError(f"the checksums given for {base} do not name its tensors")
        (tmp / CHECKSUMS_FILE).write_text(json.dumps(checksums, indent=0, sort_keys=True))
        (tmp / STATE_FILE).write_text(
            json.dumps(
                {
                    "base": str(base.resolve()),
                    "version": version,
                    "applied": [],
                    "checksum_format": CHECKSUM_FORMAT,
                    "tensors": len(locations),
                }
            )
        )
        tmp.replace(root)
        log.info(
            "weight mirror created at %s from %s (%d tensors, %.1f GiB in %.0fs)",
            root,
            base,
            len(locations),
            progress.total / 2**30,
            progress.elapsed,
        )
        return cls(root, workers=workers)

    @property
    def version(self) -> str:
        """The trainer version the mirror currently equals (``"0"`` for the base)."""
        return str(self.state["version"])

    @property
    def locations(self) -> dict[str, TensorLocation]:
        if self._locations is None:
            self._locations = tensor_locations(self.root)
        return self._locations

    def _save(self) -> None:
        _atomic_write_text(
            self.root / CHECKSUMS_FILE, json.dumps(self.checksums, indent=0, sort_keys=True)
        )
        _atomic_write_text(self.root / STATE_FILE, json.dumps(self.state))

    # ------------------------------------------------------------------- commit protocol
    #
    # Applying a delta is a redo-logged transaction. After a crash at any point, the mirror is
    # either at the old version or recoverable to the new one, weights and metadata together:
    #
    #   1. stage: decode and verify every changed tensor, write its new bytes to .pending/ (fsync)
    #   2. commit point: write .pending/commit.json atomically (tmp file, fsync, rename, fsync dir)
    #   3. apply: copy the staged bytes into the shards (fsync), then write the checksums and the
    #      state atomically, then remove .pending/
    #
    # Opening a mirror finishes step 3 when a journal exists. Every step is idempotent, and torn
    # tensor writes are rewritten. Staging that never reached step 2 is discarded.

    def _recover(self) -> None:
        pending = self.root / PENDING_DIR
        if not pending.exists():
            return
        journal_path = pending / JOURNAL_FILE
        if not journal_path.is_file():
            log.warning("discarding uncommitted staging in %s", pending)
            shutil.rmtree(pending)
            return
        journal = json.loads(journal_path.read_text())
        if self.version not in (journal["base_version"], journal["version"]):
            raise MirrorError(
                f"{pending}: journal commits {journal['base_version']} -> {journal['version']}, "
                f"but the mirror is at {self.version}; recreate the mirror"
            )
        log.warning("completing interrupted commit of version %s", journal["version"])
        self._apply_journal(journal)

    def _apply_journal(self, journal: dict[str, Any]) -> None:
        pending = self.root / PENDING_DIR
        entries = journal["tensors"]

        def write_one(entry: dict[str, str]) -> Path:
            loc = self.locations[entry["name"]]
            _write(loc, (pending / entry["file"]).read_bytes())
            return loc.path

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            shards = set(pool.map(write_one, entries))
        for shard in shards:
            _fsync_path(shard)
        for entry in entries:
            self.checksums[entry["name"]] = entry["checksum"]
        if self.version != journal["version"]:
            self.state["version"] = journal["version"]
            self.state["applied"] = [*self.state.get("applied", []), journal["version"]]
        self._save()
        self._clear_pending()

    def _clear_pending(self) -> None:
        shutil.rmtree(self.root / PENDING_DIR)
        _fsync_path(self.root)

    # -------------------------------------------------------------------------- deltas

    def apply_delta(self, delta_dir: str | os.PathLike[str]) -> DeltaReport:
        """Apply one slime/miles delta directory in place, verifying every changed tensor.

        The delta's ``base_version`` must equal the mirror's current version. Each shard tensor
        is a zstd-compressed XOR of the new bytes against the old. The header's ``__metadata__``
        carries the Adler-32 of the new bytes.

        Every tensor is decoded and checked before any is written, so a failed delta leaves the
        mirror at its previous version. The new bytes are staged and committed through a journal.
        If a crash interrupts the write, the next open of the mirror completes it. Staging needs
        free disk space next to the mirror for the changed tensors' bytes.

        Raises:
            MirrorError: wrong base version, unsupported encoding, size or checksum mismatch.
        """
        try:
            import numpy as np
            import zstandard
        except ImportError as exc:  # pragma: no cover - exercised only in stripped-down envs
            raise MirrorError(
                "WeightMirror needs numpy and zstandard (core dependencies of cwtrain)"
            ) from exc

        delta_dir = Path(delta_dir)
        index = load_index(delta_dir)
        meta = index["metadata"]
        version, base_version = str(meta.get("version", "")), str(meta.get("base_version", ""))
        if str(meta.get("delta_encoding", "")).lower() != "xor":
            raise MirrorError(f"{delta_dir}: only xor deltas are supported")
        if str(meta.get("compression_format", "zstd")).lower() != "zstd":
            raise MirrorError(f"{delta_dir}: only zstd compression is supported")
        if str(meta.get("checksum_format", "")).lower().replace("-", "") != CHECKSUM_FORMAT:
            raise MirrorError(f"{delta_dir}: only adler32 checksums are supported")
        if not _same_version(base_version, self.version):
            raise MirrorError(
                f"{delta_dir} applies to version {base_version!r}; mirror is at {self.version!r}"
            )

        report = DeltaReport(version=version, base_version=base_version)
        weight_map: dict[str, str] = index["weight_map"]
        by_shard: dict[str, list[str]] = {}
        for name, filename in weight_map.items():
            by_shard.setdefault(filename, []).append(name)

        # (tensor name, shard path, byte range of the compressed diff, expected checksum)
        work: list[tuple[str, Path, int, int, str]] = []
        for filename, names in sorted(by_shard.items()):
            shard = delta_dir / filename
            header, data_start = read_safetensors_header(shard)
            expected = header.get("__metadata__") or {}
            for name in names:
                if name not in self.locations:
                    raise MirrorError(f"{delta_dir}: tensor {name!r} is not in the mirror")
                if name not in expected:
                    raise MirrorError(f"{delta_dir}: no checksum for {name!r} in {filename}")
                begin, end = header[name]["data_offsets"]
                work.append((name, shard, data_start + begin, end - begin, expected[name]))

        def materialize(item: tuple[str, Path, int, int, str]) -> tuple[str, Any, str]:
            """Decode one tensor's new bytes and check them against the trainer's checksum."""
            name, shard, offset, length, want = item
            loc = self.locations[name]
            with shard.open("rb") as f:
                f.seek(offset)
                compressed = f.read(length)
            # zstandard reads max_output_size=0 as "no limit", so never pass 0.
            diff = zstandard.ZstdDecompressor().decompress(
                compressed, max_output_size=max(1, loc.nbytes)
            )
            if len(diff) != loc.nbytes:
                raise MirrorError(
                    f"{delta_dir}: {name!r} delta is {len(diff)} bytes, tensor is {loc.nbytes}"
                )
            new = np.frombuffer(_read(loc), dtype=np.uint8) ^ np.frombuffer(diff, dtype=np.uint8)
            digest = adler32_hex(new)
            if digest != want:
                raise MirrorError(
                    f"{delta_dir}: checksum mismatch for {name!r}: trainer wrote {want}, "
                    f"mirror computed {digest}"
                )
            return name, new, digest

        pending = self.root / PENDING_DIR
        shutil.rmtree(pending, ignore_errors=True)
        pending.mkdir()

        def stage_one(numbered: tuple[int, tuple[str, Path, int, int, str]]) -> dict[str, Any]:
            i, item = numbered
            name, new, digest = materialize(item)
            staged = pending / f"{i:06d}.bin"
            with staged.open("wb") as f:
                f.write(new.tobytes())
                f.flush()
                os.fsync(f.fileno())
            return {"name": name, "file": staged.name, "checksum": digest, "wire": item[3]}

        # Stage and verify every tensor before the mirror changes (see "commit protocol" above).
        # numpy, zlib, zstd and file I/O release the GIL, so tensors are staged in parallel.
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                staged = list(pool.map(stage_one, enumerate(work)))
        except BaseException:
            shutil.rmtree(pending, ignore_errors=True)
            raise
        for entry in staged:
            report.changed.append(entry["name"])
            report.tensors_changed += 1
            report.bytes_changed += self.locations[entry["name"]].nbytes
            report.wire_bytes += entry.pop("wire")
        _fsync_path(pending)
        journal = {"version": version, "base_version": self.version, "tensors": staged}
        _atomic_write_text(pending / JOURNAL_FILE, json.dumps(journal))  # the commit point
        self._apply_journal(journal)
        log.info(
            "mirror now at version %s: %d tensors (%.1f MiB) changed, verified adler32",
            version,
            report.tensors_changed,
            report.bytes_changed / 2**20,
        )
        return report

    def advance(self, version: str) -> None:
        """Record that ``version`` has the mirror's current bytes (a sync that changed nothing)."""
        self.state["version"] = version
        self.state["applied"] = [*self.state.get("applied", []), version]
        self._save()

    # ------------------------------------------------------------------------ outputs

    def manifest(self, identity: str, **extra: Any) -> dict[str, Any]:
        """Per-tensor checksums of the mirror's current contents, tagged with ``identity``."""
        return {
            "identity": identity,
            "version": self.version,
            "checksum_format": CHECKSUM_FORMAT,
            "tensors": dict(sorted(self.checksums.items())),
            **extra,
        }

    def write_full_snapshot(self, dest: str | os.PathLike[str]) -> Path:
        """Write the mirror as a FULL hot-load snapshot at ``dest``.

        Each shard gets every tensor's Adler-32 in its ``__metadata__``, and the index gets
        ``checksum_format: adler32``, so the engine verifies each tensor on install. Other files
        of the base checkpoint (``config.json``, tokenizer) are copied unchanged. ``dest`` is
        replaced if it exists.
        """
        index = load_index(self.root)
        dest = Path(dest)
        tmp = dest.with_name(dest.name + ".tmp")
        for path in (dest, tmp):
            shutil.rmtree(path, ignore_errors=True)
        tmp.mkdir(parents=True)
        by_shard: dict[Path, dict[str, str]] = {}
        for name, loc in self.locations.items():
            by_shard.setdefault(loc.path, {})[name] = self.checksums[name]
        shards = sorted(by_shard)
        progress = _Progress(
            f"writing version {self.version} as a full snapshot", _total_size(shards)
        )
        try:
            _parallel(
                lambda shard: rewrite_shard_metadata(
                    shard, tmp / shard.name, by_shard[shard], progress=progress
                ),
                shards,
                self.workers,
            )
            for path in self.root.iterdir():
                if path.is_file() and path.name not in (STATE_FILE, CHECKSUMS_FILE, INDEX_FILE):
                    if not path.name.endswith(SHARD_SUFFIX):
                        shutil.copy2(path, tmp / path.name)
            metadata = index["metadata"]
            metadata.pop("delta_encoding", None)
            metadata["checksum_format"] = CHECKSUM_FORMAT
            index["metadata"] = metadata
            (tmp / INDEX_FILE).write_text(json.dumps(index, indent=1))
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)  # a partial snapshot can be model-sized
            raise
        tmp.replace(dest)
        log.info(
            "wrote full snapshot of version %s to %s (%.1f GiB in %.0fs)",
            self.version,
            dest,
            progress.total / 2**30,
            progress.elapsed,
        )
        return dest


def _read(loc: TensorLocation) -> bytes:
    with loc.path.open("rb") as f:
        f.seek(loc.offset)
        return f.read(loc.nbytes)


def _write(loc: TensorLocation, data: bytes) -> None:
    with loc.path.open("r+b") as f:
        f.seek(loc.offset)
        f.write(data)


def _same_version(a: str, b: str) -> bool:
    """Equal as strings, or as numbers when both are digit strings ("0" == "000000")."""
    if a == b:
        return True
    return a.isdigit() and b.isdigit() and int(a) == int(b)


def _fsync_path(path: Path) -> None:
    """fsync a file or directory by path (directories make renames and deletions durable)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` so that readers see either the old or the new content."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_path(path.parent)


__all__ = [
    "CHECKSUM_FORMAT",
    "DeltaReport",
    "MirrorError",
    "TensorLocation",
    "WeightMirror",
    "adler32_hex",
    "read_safetensors_header",
    "rewrite_shard_metadata",
    "tensor_locations",
]
