# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Write incremental (delta) snapshots from two full HF checkpoints.

slime and miles compute deltas in memory while they sync weights. This module gives the same wire
format to trainers that only write full HF exports (NeMo-RL, or any ``save_pretrained``-style
loop).

For every tensor whose bytes changed between ``base`` and ``new``, the delta stores the
zstd-compressed XOR of the two byte strings. Each one is a ``uint8`` tensor in a safetensors shard.
The shard header carries the Adler-32 of the *new* bytes. The index names the base version, the new
version and the encoding. :func:`cwtrain.hotload.inspect_snapshot` classifies the directory as
INCREMENTAL, and the platform applies it like a slime/miles delta.
:class:`cwtrain.hotload.WeightMirror` can apply it too.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import shutil
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cwtrain.hotload.errors import SnapshotError
from cwtrain.hotload.mirror import TensorLocation, adler32_hex, tensor_locations
from cwtrain.hotload.snapshot import INDEX_FILE, load_index

log = logging.getLogger(__name__)

SHARD_TARGET_BYTES = 2 * 2**30


@dataclass
class DeltaStats:
    """What :func:`write_delta` produced."""

    path: Path
    version: str
    base_version: str
    tensors_total: int = 0
    tensors_changed: int = 0
    bytes_changed: int = 0
    wire_bytes: int = 0
    shards: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.tensors_changed == 0


def write_safetensors(path: Path, tensors: dict[str, bytes], metadata: dict[str, str]) -> int:
    """Write ``tensors`` (raw bytes, stored as ``U8`` vectors) as one safetensors file."""
    header: dict[str, Any] = {"__metadata__": dict(metadata)}
    offset = 0
    for name, data in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    body = json.dumps(header, separators=(",", ":")).encode()
    body += b" " * (-len(body) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(body)) + body)
        for data in tensors.values():
            f.write(data)
    return 8 + len(body) + offset


def _read(loc: TensorLocation) -> bytes:
    with loc.path.open("rb") as f:
        f.seek(loc.offset)
        return f.read(loc.nbytes)


def write_delta(
    base: str | os.PathLike[str],
    new: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    version: str,
    base_version: str,
    workers: int | None = None,
    shard_bytes: int = SHARD_TARGET_BYTES,
) -> DeltaStats:
    """Write the delta that turns checkpoint ``base`` into checkpoint ``new`` under ``out``.

    Both directories must hold the same tensors with identical dtypes and shapes. ``version`` and
    ``base_version`` go into the index metadata in the slime/miles form (``000003`` /
    ``000002``). :func:`upload_snapshot` rewrites them as snapshot identities at upload time.

    If no tensor changed, the index is still written, with no shards, as slime does. Callers
    usually skip publishing an empty delta.

    Raises:
        SnapshotError: the two checkpoints do not have the same tensor layout.
    """
    import numpy as np
    import zstandard

    base_dir, new_dir, out_dir = Path(base), Path(new), Path(out)
    base_loc = tensor_locations(base_dir)
    new_loc = tensor_locations(new_dir)
    if set(base_loc) != set(new_loc):
        missing = sorted(set(base_loc) ^ set(new_loc))[:5]
        raise SnapshotError(f"{new_dir} and {base_dir} hold different tensors, e.g. {missing}")
    for name, loc in new_loc.items():
        b = base_loc[name]
        if (loc.dtype, loc.shape, loc.nbytes) != (b.dtype, b.shape, b.nbytes):
            raise SnapshotError(
                f"tensor {name!r} changed layout: {b.dtype}{list(b.shape)} -> "
                f"{loc.dtype}{list(loc.shape)}"
            )

    stats = DeltaStats(
        path=out_dir, version=version, base_version=base_version, tensors_total=len(new_loc)
    )

    def diff(name: str) -> tuple[str, bytes | None, str, int]:
        old_bytes, new_bytes = _read(base_loc[name]), _read(new_loc[name])
        if old_bytes == new_bytes:
            return name, None, "", 0
        xor = np.frombuffer(new_bytes, dtype=np.uint8) ^ np.frombuffer(old_bytes, dtype=np.uint8)
        compressed = zstandard.ZstdCompressor(level=1).compress(xor.tobytes())
        return name, compressed, adler32_hex(new_bytes), len(new_bytes)

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # Stream: at most `window` tensors are diffed at once, and a shard is written when it reaches
    # shard_bytes. Memory stays near the window's tensors plus one shard, for any update size.
    # Shard names carry the final shard count, so shards get provisional names and are renamed
    # at the end.
    parts: list[tuple[str, list[str]]] = []
    current: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    size = 0

    def flush() -> None:
        nonlocal current, digests, size
        if not current:
            return
        part = f"part-{len(parts):05d}.safetensors"
        write_safetensors(tmp / part, current, digests)
        parts.append((part, list(current)))
        current, digests, size = {}, {}, 0

    max_workers = max(1, min(16, workers or (os.cpu_count() or 4)))
    window = 2 * max_workers
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        names = iter(sorted(new_loc))
        inflight: deque = deque()
        for name in itertools.islice(names, window):
            inflight.append(pool.submit(diff, name))
        while inflight:
            name, compressed, digest, nbytes = inflight.popleft().result()
            following = next(names, None)
            if following is not None:
                inflight.append(pool.submit(diff, following))
            if compressed is None:
                continue
            stats.tensors_changed += 1
            stats.bytes_changed += nbytes
            stats.wire_bytes += len(compressed)
            if current and size + len(compressed) > shard_bytes:
                flush()
            current[name] = compressed
            digests[name] = digest
            size += len(compressed)
    flush()

    weight_map: dict[str, str] = {}
    for i, (part, tensor_names) in enumerate(parts):
        fname = f"model-{i:05d}-of-{len(parts):05d}.safetensors"
        os.replace(tmp / part, tmp / fname)
        weight_map.update({n: fname for n in tensor_names})
        stats.shards.append(fname)

    index = {
        "metadata": {
            "version": version,
            "base_version": base_version,
            "delta_encoding": "xor",
            "compression_format": "zstd",
            "checksum_format": "adler32",
        },
        "weight_map": weight_map,
    }
    (tmp / INDEX_FILE).write_text(json.dumps(index))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.replace(out_dir)
    log.info(
        "delta %s -> %s: %d/%d tensors changed, %.1f MiB on the wire (%d shards)",
        base_version,
        version,
        stats.tensors_changed,
        stats.tensors_total,
        stats.wire_bytes / 2**20,
        len(stats.shards),
    )
    return stats


def write_full_snapshot(
    src: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    checksums: dict[str, str] | None = None,
) -> Path:
    """Copy full checkpoint ``src`` to ``dest`` as a checksummed FULL snapshot.

    Each shard gets every tensor's Adler-32 in its ``__metadata__``, and the index gets
    ``checksum_format: adler32``, so the engine verifies each tensor on install. Other files
    (``config.json``, tokenizer) are copied unchanged. ``dest`` is replaced if it exists.

    Args:
        checksums: Every tensor's Adler-32, if already computed. Computed from ``src`` otherwise.
    """
    import shutil

    from cwtrain.hotload.mirror import rewrite_shard_metadata

    src_dir, dest_dir = Path(src), Path(dest)
    index = load_index(src_dir)
    checksums = checksums or checkpoint_checksums(src_dir)
    tmp = dest_dir.with_name(dest_dir.name + ".tmp")
    for path in (dest_dir, tmp):
        shutil.rmtree(path, ignore_errors=True)
    tmp.mkdir(parents=True)
    by_shard: dict[Path, dict[str, str]] = {}
    for name, loc in tensor_locations(src_dir).items():
        by_shard.setdefault(loc.path, {})[name] = checksums[name]
    for shard, sums in by_shard.items():
        rewrite_shard_metadata(shard, tmp / shard.name, sums)
    for path in src_dir.iterdir():
        if path.is_file() and path.name != INDEX_FILE and not path.name.endswith(".safetensors"):
            shutil.copy2(path, tmp / path.name)
    metadata = index["metadata"]
    metadata.pop("delta_encoding", None)
    metadata["checksum_format"] = "adler32"
    index["metadata"] = metadata
    (tmp / INDEX_FILE).write_text(json.dumps(index, indent=1))
    tmp.replace(dest_dir)
    return dest_dir


def checkpoint_checksums(path: str | os.PathLike[str]) -> dict[str, str]:
    """Adler-32 of every tensor of a full checkpoint, in the manifest format the mirror uses."""
    return {name: adler32_hex(_read(loc)) for name, loc in tensor_locations(Path(path)).items()}


__all__ = [
    "DeltaStats",
    "checkpoint_checksums",
    "write_delta",
    "write_full_snapshot",
    "write_safetensors",
]
