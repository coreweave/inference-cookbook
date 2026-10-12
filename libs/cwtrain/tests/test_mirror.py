# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import errno
import json
import logging
import os
import random
import struct
import threading
import time
import zlib
from pathlib import Path

import pytest

from cwtrain.hotload import MirrorError, SnapshotError, SnapshotType, WeightMirror, inspect_snapshot
from cwtrain.hotload.mirror import (
    read_safetensors_header,
    rewrite_shard_metadata,
    tensor_locations,
)
from tests.conftest import make_weights, perturb, write_real_checkpoint, write_real_delta


def read_all(root: Path) -> dict[str, bytes]:
    out = {}
    for name, loc in tensor_locations(root).items():
        with loc.path.open("rb") as f:
            f.seek(loc.offset)
            out[name] = f.read(loc.nbytes)
    return out


def adler(b: bytes) -> str:
    return f"{zlib.adler32(b):08x}"


@pytest.fixture
def base(tmp_path: Path):
    weights = make_weights()
    return write_real_checkpoint(tmp_path / "base", weights), weights


def test_create_copies_base_and_checksums_every_tensor(tmp_path: Path, base) -> None:
    root, weights = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    assert mirror.version == "000000"
    assert read_all(mirror.root) == {n: d for n, (_t, _s, d) in weights.items()}
    assert mirror.checksums == {n: adler(d) for n, (_t, _s, d) in weights.items()}
    assert (mirror.root / "config.json").is_file() and (
        mirror.root / "tokenizer_config.json"
    ).is_file()
    # reopen
    again = WeightMirror(mirror.root)
    assert again.checksums == mirror.checksums and again.version == "000000"
    # a second create replaces the mirror
    WeightMirror.create(root, tmp_path / "mirror")
    assert WeightMirror(tmp_path / "mirror").version == "0"


def test_create_reuses_checksums_the_caller_computed(tmp_path: Path, base, monkeypatch) -> None:
    from cwtrain.hotload import mirror as mirror_module
    from cwtrain.hotload.delta import checkpoint_checksums

    root, _weights = base
    checksums = checkpoint_checksums(root)

    def no_hashing(*args, **kwargs):
        raise AssertionError("the copy was hashed again")

    monkeypatch.setattr(mirror_module, "adler32_hex", no_hashing)
    monkeypatch.setattr(mirror_module, "_copy_and_checksum", no_hashing)
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000", checksums=checksums)
    assert mirror.checksums == checksums
    assert WeightMirror(mirror.root).checksums == checksums
    with pytest.raises(SnapshotError, match="do not name its tensors"):
        WeightMirror.create(root, tmp_path / "other", checksums={"x": "00000000"})
    assert not (tmp_path / "other").exists() and not (tmp_path / "other.tmp").exists()


def test_create_rejects_non_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "junk").mkdir()
    with pytest.raises(Exception, match="full HF checkpoint"):
        WeightMirror.create(tmp_path / "junk", tmp_path / "m")


def test_apply_delta_updates_bytes_and_verifies(tmp_path: Path, base) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    w1 = perturb(w0, ("model.layers.0.weight", "lm_head.bias"), seed=1)
    d1 = write_real_delta(tmp_path / "deltas" / "weight_v000001", w0, w1, version=1)

    report = mirror.apply_delta(d1)

    assert report.version == "000001" and report.base_version == "000000"
    assert sorted(report.changed) == ["lm_head.bias", "model.layers.0.weight"]
    assert report.tensors_changed == 2
    assert report.bytes_changed == len(w1["model.layers.0.weight"][2]) + len(w1["lm_head.bias"][2])
    assert 0 < report.wire_bytes
    assert read_all(mirror.root) == {n: d for n, (_t, _s, d) in w1.items()}
    assert mirror.checksums == {n: adler(d) for n, (_t, _s, d) in w1.items()}
    assert mirror.version == "000001" and mirror.state["applied"] == ["000001"]

    # chain a second delta, then reopening the mirror sees the same state
    w2 = perturb(w1, ("model.layers.1.weight",), seed=2)
    mirror.apply_delta(write_real_delta(tmp_path / "deltas" / "weight_v000002", w1, w2, version=2))
    assert read_all(mirror.root) == {n: d for n, (_t, _s, d) in w2.items()}
    assert WeightMirror(mirror.root).version == "000002"


def test_apply_delta_rejects_wrong_base_version(tmp_path: Path, base) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    w1 = perturb(w0, ("lm_head.bias",))
    d3 = write_real_delta(tmp_path / "weight_v000003", w0, w1, version=3)  # base 000002
    with pytest.raises(MirrorError, match="applies to version '000002'"):
        mirror.apply_delta(d3)


def test_apply_delta_checksum_mismatch_leaves_mirror_untouched(tmp_path: Path, base) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    # two changed tensors, the second with a bad checksum: neither may advance
    w1 = perturb(w0, ("model.layers.0.weight", "lm_head.bias"))
    bad = write_real_delta(
        tmp_path / "bad" / "weight_v000001", w0, w1, version=1, corrupt="lm_head.bias"
    )
    with pytest.raises(MirrorError, match="checksum mismatch for 'lm_head.bias'"):
        mirror.apply_delta(bad)
    assert read_all(mirror.root) == {n: d for n, (_t, _s, d) in w0.items()}
    assert mirror.checksums == {n: adler(d) for n, (_t, _s, d) in w0.items()}
    assert mirror.version == "000000" and mirror.state.get("applied", []) == []

    assert not (mirror.root / ".pending").exists()  # failed staging is cleaned up

    # the repaired delta then applies cleanly on the untouched base
    good = write_real_delta(tmp_path / "good" / "weight_v000001", w0, w1, version=1)
    report = mirror.apply_delta(good)
    assert report.tensors_changed == 2
    assert read_all(mirror.root) == {n: d for n, (_t, _s, d) in w1.items()}
    assert mirror.version == "000001"


def test_apply_delta_rejects_unsupported_formats(tmp_path: Path, base) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    w1 = perturb(w0, ("lm_head.bias",))
    d = write_real_delta(tmp_path / "weight_v000001", w0, w1, version=1)
    index_path = d / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"]["delta_encoding"] = "overwrite"
    index_path.write_text(json.dumps(index))
    with pytest.raises(MirrorError, match="xor"):
        mirror.apply_delta(d)


def test_advance_records_noop_versions(tmp_path: Path, base) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    mirror.advance("000001")
    assert mirror.version == "000001"
    w2 = perturb(w0, ("lm_head.bias",))
    mirror.apply_delta(write_real_delta(tmp_path / "weight_v000002", w0, w2, version=2))
    assert mirror.version == "000002" and mirror.state["applied"] == ["000001", "000002"]


def test_manifest_lists_every_tensor(tmp_path: Path, base) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    m = mirror.manifest("run-v000000", run_id="run")
    assert m["identity"] == "run-v000000" and m["run_id"] == "run"
    assert m["checksum_format"] == "adler32" and m["version"] == "000000"
    assert m["tensors"] == {n: adler(d) for n, (_t, _s, d) in w0.items()}


def test_write_full_snapshot_is_a_verified_full_checkpoint(tmp_path: Path, base) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    w1 = perturb(w0, ("model.layers.0.weight",))
    mirror.apply_delta(write_real_delta(tmp_path / "weight_v000001", w0, w1, version=1))

    snap = mirror.write_full_snapshot(tmp_path / "staging" / "run-v000001")

    info = inspect_snapshot(snap)
    assert info.type is SnapshotType.FULL
    names = {p.name for p in info.files}
    assert names == {
        "config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    # same bytes as the mirror, per-tensor checksums in every shard header, index flags them
    assert read_all(snap) == {n: d for n, (_t, _s, d) in w1.items()}
    for shard in snap.glob("*.safetensors"):
        header, data_start = read_safetensors_header(shard)
        assert (data_start - 8) % 8 == 0
        meta = header["__metadata__"]
        assert meta["format"] == "pt"
        for name in header:
            if name != "__metadata__":
                assert meta[name] == mirror.checksums[name]
                # layout unchanged
                assert header[name]["dtype"] == w1[name][0]
                assert tuple(header[name]["shape"]) == w1[name][1]
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    assert index["metadata"]["checksum_format"] == "adler32"
    assert index["metadata"]["total_size"] == sum(len(d) for _t, _s, d in w1.values())
    assert set(index["weight_map"]) == set(w1)
    # rewriting again replaces the directory
    mirror.write_full_snapshot(snap)
    assert inspect_snapshot(snap).type is SnapshotType.FULL


def test_rewrite_shard_metadata_keeps_data(tmp_path: Path, base) -> None:
    root, w0 = base
    src = root / "model-00001-of-00002.safetensors"
    dst = tmp_path / "out.safetensors"
    rewrite_shard_metadata(src, dst, {"model.layers.0.weight": "00000001", "x": "y"})
    header, _ = read_safetensors_header(dst)
    assert header["__metadata__"] == {"format": "pt", "model.layers.0.weight": "00000001", "x": "y"}
    assert read_all(tmp_path)["model.layers.0.weight"] == w0["model.layers.0.weight"][2]


# ------------------------------------------------------------------ crash safety (journal)


def _crash_on_write(monkeypatch, after: int, torn: bool = False) -> None:
    """Make the n-th tensor write into a shard die (optionally after writing half of it)."""
    import cwtrain.hotload.mirror as mirror_mod

    real = mirror_mod._write
    calls = {"n": 0}

    def flaky(loc, data):
        calls["n"] += 1
        if calls["n"] > after:
            if torn:
                real(loc, data[: len(data) // 2])
            raise OSError("simulated crash while writing shards")
        real(loc, data)

    monkeypatch.setattr(mirror_mod, "_write", flaky)


def test_crash_while_writing_shards_is_completed_on_reopen(
    tmp_path: Path, base, monkeypatch
) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    w1 = perturb(w0, ("model.layers.0.weight", "model.layers.1.weight", "lm_head.bias"), seed=3)
    d1 = write_real_delta(tmp_path / "deltas" / "weight_v000001", w0, w1, version=1)
    _crash_on_write(monkeypatch, after=1, torn=True)  # one tensor written, the next one torn
    with pytest.raises(OSError, match="simulated crash"):
        mirror.apply_delta(d1)
    assert (mirror.root / ".pending" / "commit.json").is_file()
    monkeypatch.undo()

    reopened = WeightMirror(mirror.root)  # recovery completes the committed version
    assert reopened.version == "000001" and reopened.state["applied"] == ["000001"]
    assert read_all(reopened.root) == {n: d for n, (_t, _s, d) in w1.items()}
    assert reopened.checksums == {n: adler(d) for n, (_t, _s, d) in w1.items()}
    assert not (reopened.root / ".pending").exists()
    # and the chain continues from there
    w2 = perturb(w1, ("model.layers.1.weight",), seed=4)
    reopened.apply_delta(
        write_real_delta(tmp_path / "deltas" / "weight_v000002", w1, w2, version=2)
    )
    assert read_all(reopened.root) == {n: d for n, (_t, _s, d) in w2.items()}


def test_crash_before_the_commit_point_leaves_the_old_version(tmp_path: Path, base) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    # staging written but no journal: the process died before the commit point
    pending = mirror.root / ".pending"
    pending.mkdir()
    (pending / "000000.bin").write_bytes(b"half-staged")
    reopened = WeightMirror(mirror.root)
    assert reopened.version == "000000" and not pending.exists()
    assert read_all(reopened.root) == {n: d for n, (_t, _s, d) in w0.items()}


def test_recovery_after_metadata_was_saved_is_idempotent(tmp_path: Path, base, monkeypatch) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    w1 = perturb(w0, ("lm_head.bias",), seed=5)
    d1 = write_real_delta(tmp_path / "deltas" / "weight_v000001", w0, w1, version=1)

    def die(self):
        raise OSError("simulated crash before cleanup")

    monkeypatch.setattr(WeightMirror, "_clear_pending", die)
    with pytest.raises(OSError, match="before cleanup"):
        mirror.apply_delta(d1)
    monkeypatch.undo()

    reopened = WeightMirror(mirror.root)
    assert reopened.version == "000001" and reopened.state["applied"] == ["000001"]  # not twice
    assert read_all(reopened.root) == {n: d for n, (_t, _s, d) in w1.items()}
    assert not (reopened.root / ".pending").exists()


def test_metadata_files_are_replaced_atomically(tmp_path: Path, base) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    mirror.advance("000001")
    assert not list(mirror.root.glob("*.tmp"))
    assert WeightMirror(mirror.root).version == "000001"


def test_non_numeric_versions_raise_mirror_error(tmp_path: Path, base) -> None:
    root, w0 = base
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="run-v000000")
    w1 = perturb(w0, ("lm_head.bias",))
    d1 = write_real_delta(tmp_path / "weight_v000001", w0, w1, version=1)  # base "000000"
    with pytest.raises(MirrorError, match="applies to version '000000'"):
        mirror.apply_delta(d1)


# ------------------------------------------------------- copying large checkpoints quickly


def per_tensor_checksums(root: Path) -> dict[str, str]:
    """Reference checksums: every tensor read on its own, short reads included."""
    out = {}
    for name, loc in tensor_locations(root).items():
        with loc.path.open("rb") as f:
            f.seek(loc.offset)
            out[name] = adler(f.read(loc.nbytes))
    return out


def write_layout_shard(path: Path, entries, rng: random.Random, tail: bytes = b"") -> None:
    """A shard from ``(name, gap before, payload)`` entries, header keys in random order."""
    tensors, data = {}, bytearray()
    for name, gap, payload in entries:
        data += gap
        tensors[name] = {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [len(data), len(data) + len(payload)],
        }
        data += payload
    data += tail
    names = list(tensors)
    rng.shuffle(names)
    write_raw_shard(
        path, {"__metadata__": {"format": "pt"}, **{n: tensors[n] for n in names}}, data
    )


def write_raw_shard(path: Path, header: dict, data: bytes) -> None:
    body = json.dumps(header).encode()
    body += b" " * (-len(body) % 8)
    path.write_bytes(struct.pack("<Q", len(body)) + body + bytes(data))


def write_random_checkpoint(root: Path, seed: int, shards: int | None = None) -> Path:
    """Shards with empty tensors, gaps, trailing bytes and an empty tensor at the very end, plus
    files and nested directories that are not top-level shards."""
    rng = random.Random(seed)
    root.mkdir(parents=True)
    count = shards or rng.randint(2, 5)
    weight_map = {}
    for s in range(count):
        fname = f"model-{s + 1:05d}-of-{count:05d}.safetensors"
        entries = [(f"s{s}.t0", b"", rng.randbytes(rng.randint(1, 50)))]
        for t in range(1, rng.randint(1, 7)):
            size = rng.choice([0, 0, 1, 2, 7, 64, rng.randint(0, 300)])
            entries.append(
                (f"s{s}.t{t}", rng.randbytes(rng.choice([0, 0, 1, 5])), rng.randbytes(size))
            )
        tail = b""
        if rng.random() < 0.5:
            entries.append((f"s{s}.empty_at_eof", b"", b""))
        else:
            tail = rng.randbytes(rng.choice([1, 3]))
        write_layout_shard(root / fname, entries, rng, tail)
        weight_map.update({name: fname for name, _gap, _payload in entries})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (root / "config.json").write_text("{}")
    (root / "tokenizer.json").write_bytes(rng.randbytes(100))
    (root / ".cache" / "huggingface").mkdir(parents=True)
    (root / ".cache" / "huggingface" / "model.lock").write_text("lock")
    (root / "extra").mkdir()
    (root / "extra" / "nested.safetensors").write_bytes(b"not a top-level shard")
    return root


def files_under(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 64, 2**20])
@pytest.mark.parametrize("seed", range(6))
def test_create_checksums_equal_a_per_tensor_read(tmp_path: Path, monkeypatch, chunk, seed) -> None:
    from cwtrain.hotload import mirror as mirror_module

    monkeypatch.setattr(mirror_module, "_STREAM_CHUNK", chunk)  # tensors straddle every chunk
    root = write_random_checkpoint(tmp_path / "base", seed)
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000", workers=3)

    assert mirror.checksums == per_tensor_checksums(root)
    assert WeightMirror(mirror.root).checksums == mirror.checksums
    copied = files_under(mirror.root)
    for rel, data in files_under(root).items():  # every file, nested directories included
        assert copied.pop(rel) == data, rel
    assert set(copied) == {"mirror-state.json", "mirror-checksums.json"}
    assert json.loads((mirror.root / "mirror-state.json").read_text())["tensors"] == len(
        tensor_locations(root)
    )


@pytest.mark.parametrize("chunk", [1, 16, 2**20])
def test_create_checksums_truncated_and_malformed_shards_as_before(
    tmp_path: Path, monkeypatch, chunk
) -> None:
    from cwtrain.hotload import mirror as mirror_module

    monkeypatch.setattr(mirror_module, "_STREAM_CHUNK", chunk)
    root = write_random_checkpoint(tmp_path / "base", 0, shards=1)
    rng = random.Random(1)
    u8 = {"dtype": "U8", "shape": [1]}
    # the file ends inside the first tensor; the other two start past its end
    write_raw_shard(
        root / "model-00002-of-00004.safetensors",
        {
            "cut.a": {**u8, "data_offsets": [0, 100]},
            "cut.b": {**u8, "data_offsets": [100, 120]},
            "cut.empty": {**u8, "data_offsets": [120, 120]},
        },
        rng.randbytes(40),
    )
    # overlapping ranges are hashed tensor by tensor, as in the reference
    write_raw_shard(
        root / "model-00003-of-00004.safetensors",
        {
            "odd.a": {**u8, "data_offsets": [0, 10]},
            "odd.b": {**u8, "data_offsets": [5, 15]},
            "odd.empty": {**u8, "data_offsets": [15, 15]},
        },
        rng.randbytes(20),
    )
    mirror = WeightMirror.create(root, tmp_path / "mirror")
    assert mirror.checksums == per_tensor_checksums(root)
    assert mirror.checksums["cut.b"] == mirror.checksums["cut.empty"] == "00000001"


def test_create_rejects_a_range_that_runs_backwards_as_before(tmp_path: Path) -> None:
    root = write_random_checkpoint(tmp_path / "base", 0, shards=1)
    u8 = {"dtype": "U8", "shape": [1]}
    write_raw_shard(
        root / "model-00002-of-00002.safetensors",
        {"back.a": {**u8, "data_offsets": [10, 5]}, "back.b": {**u8, "data_offsets": [0, 4]}},
        bytes(20),
    )
    with pytest.raises(ValueError, match="read length must be non-negative"):
        per_tensor_checksums(root)  # the reference fails the same way
    with pytest.raises(ValueError, match="read length must be non-negative"):
        WeightMirror.create(root, tmp_path / "mirror")
    assert not (tmp_path / "mirror").exists() and not (tmp_path / "mirror.tmp").exists()


def test_create_copies_shards_in_parallel(tmp_path: Path, base, monkeypatch) -> None:
    from cwtrain.hotload import mirror as mirror_module

    root, weights = base  # two shards
    together = threading.Barrier(2, timeout=10)
    real = mirror_module._copy_and_checksum

    def copy(src, dst, progress=None):
        together.wait()  # passes only while both shards are being copied at once
        return real(src, dst, progress)

    monkeypatch.setattr(mirror_module, "_copy_and_checksum", copy)
    mirror = WeightMirror.create(root, tmp_path / "mirror", workers=2)
    assert mirror.workers == 2
    assert mirror.checksums == {n: adler(d) for n, (_t, _s, d) in weights.items()}
    assert (
        WeightMirror.create(root, tmp_path / "again").workers
        == WeightMirror(root.parent / "mirror").workers
    )


def test_create_stops_at_the_first_failed_shard(tmp_path: Path, monkeypatch) -> None:
    from cwtrain.hotload import mirror as mirror_module

    root = write_random_checkpoint(tmp_path / "base", 3, shards=6)
    started = []

    def broken(src, dst, progress=None):
        started.append(src.name)
        time.sleep(0.2)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(mirror_module, "_copy_and_checksum", broken)
    with pytest.raises(OSError, match="No space left"):
        WeightMirror.create(root, tmp_path / "mirror", workers=1)
    assert len(started) <= 2  # the shards not yet started were cancelled
    assert not (tmp_path / "mirror").exists() and not (tmp_path / "mirror.tmp").exists()


def test_copies_log_progress(tmp_path: Path, base, monkeypatch, caplog) -> None:
    from cwtrain.hotload import mirror as mirror_module

    root, _weights = base
    monkeypatch.setattr(mirror_module, "_PROGRESS_INTERVAL", 0.0)
    caplog.set_level(logging.INFO, logger="cwtrain.hotload.mirror")
    mirror = WeightMirror.create(root, tmp_path / "mirror", version="000000")
    mirror.write_full_snapshot(tmp_path / "staging" / "run-v000000")
    lines = [r.getMessage() for r in caplog.records]
    assert any(m.startswith(f"copying {root} to the weight mirror: ") for m in lines)
    assert any(m.startswith("writing version 000000 as a full snapshot: ") for m in lines)
    assert all("GiB (" in m and "MiB/s)" in m for m in lines if ": 0.0/" in m)


def test_write_full_snapshot_rewrites_shards_in_parallel(tmp_path: Path, base, monkeypatch) -> None:
    from cwtrain.hotload import mirror as mirror_module

    root, weights = base  # two shards
    mirror = WeightMirror.create(root, tmp_path / "mirror", workers=2)
    together = threading.Barrier(2, timeout=10)
    real = mirror_module.rewrite_shard_metadata

    def rewrite(src, dst, metadata, **kwargs):
        together.wait()  # passes only while both shards are being written at once
        real(src, dst, metadata, **kwargs)

    monkeypatch.setattr(mirror_module, "rewrite_shard_metadata", rewrite)
    snap = mirror.write_full_snapshot(tmp_path / "staging" / "run-v000000")
    assert read_all(snap) == {n: d for n, (_t, _s, d) in weights.items()}


# os.copy_file_range is Linux-only; these fakes stand in for it on any OS


def copy_file_range_fake(
    calls: list, *, step: int = 5, fail_on: int | None = None, error=0, zero=False
):
    # os.copy_file_range's signature: Python 3.14's shutil.copyfile calls it too, without
    # offset_src and with offset_dst by keyword; a None offset uses and moves the file position
    def fake(src_fd, dst_fd, count, offset_src=None, offset_dst=None):
        calls.append((count, offset_src, offset_dst))
        if fail_on is not None and len(calls) > fail_on:
            raise OSError(error, os.strerror(error))
        if zero:
            return 0
        size = min(count, step)
        if offset_src is None:
            data = os.read(src_fd, size)
        else:
            data = os.pread(src_fd, size, offset_src)
        if offset_dst is None:
            return os.write(dst_fd, data)
        return os.pwrite(dst_fd, data, offset_dst)

    return fake


def expected_rewrite(src: Path, metadata: dict[str, str]) -> bytes:
    from cwtrain.hotload.mirror import _encode_header

    header, data_start = read_safetensors_header(src)
    merged = {**header["__metadata__"], **metadata}
    rest = {k: v for k, v in header.items() if k != "__metadata__"}
    return _encode_header({"__metadata__": merged, **rest}) + src.read_bytes()[data_start:]


@pytest.fixture
def big_shard(tmp_path: Path) -> Path:
    path = tmp_path / "model-00001-of-00001.safetensors"
    payload = random.Random(7).randbytes(203)
    write_layout_shard(
        path, [("a", b"", payload[:100]), ("b", b"", payload[100:])], random.Random(0)
    )
    return path


def test_rewrite_shard_metadata_lets_the_kernel_copy_the_data(
    tmp_path: Path, big_shard: Path, monkeypatch
) -> None:
    calls: list = []
    monkeypatch.setattr(os, "copy_file_range", copy_file_range_fake(calls), raising=False)
    dst = tmp_path / "out.safetensors"
    rewrite_shard_metadata(big_shard, dst, {"a": "00000002"})
    assert dst.read_bytes() == expected_rewrite(big_shard, {"a": "00000002"})
    assert len(calls) > 1  # partial copies continue where the last one stopped
    assert [c[1] for c in calls] == sorted(c[1] for c in calls)


@pytest.mark.parametrize("name", ["EXDEV", "ENOSYS", "EOPNOTSUPP", "ENOTSUP", "EINVAL"])
@pytest.mark.parametrize("fail_on", [0, 2])
def test_rewrite_shard_metadata_falls_back_when_the_kernel_refuses(
    tmp_path: Path, big_shard: Path, monkeypatch, name: str, fail_on: int
) -> None:
    if not hasattr(errno, name):
        pytest.skip(f"no errno.{name} here")
    calls: list = []
    fake = copy_file_range_fake(calls, fail_on=fail_on, error=getattr(errno, name))
    monkeypatch.setattr(os, "copy_file_range", fake, raising=False)
    dst = tmp_path / "out.safetensors"
    rewrite_shard_metadata(big_shard, dst, {"b": "00000003"})
    assert dst.read_bytes() == expected_rewrite(big_shard, {"b": "00000003"})
    assert len(calls) == fail_on + 1


def test_rewrite_shard_metadata_falls_back_when_the_kernel_copies_nothing(
    tmp_path: Path, big_shard: Path, monkeypatch
) -> None:
    calls: list = []
    monkeypatch.setattr(
        os, "copy_file_range", copy_file_range_fake(calls, zero=True), raising=False
    )
    dst = tmp_path / "out.safetensors"
    rewrite_shard_metadata(big_shard, dst, {})
    assert dst.read_bytes() == expected_rewrite(big_shard, {})
    assert len(calls) == 1


def test_rewrite_shard_metadata_without_copy_file_range(
    tmp_path: Path, big_shard: Path, monkeypatch
) -> None:
    monkeypatch.delattr(os, "copy_file_range", raising=False)
    dst = tmp_path / "out.safetensors"
    rewrite_shard_metadata(big_shard, dst, {"a": "x"})
    assert dst.read_bytes() == expected_rewrite(big_shard, {"a": "x"})


def test_rewrite_shard_metadata_raises_real_io_errors(
    tmp_path: Path, big_shard: Path, monkeypatch
) -> None:
    fake = copy_file_range_fake([], fail_on=0, error=errno.EIO)
    monkeypatch.setattr(os, "copy_file_range", fake, raising=False)
    with pytest.raises(OSError) as raised:
        rewrite_shard_metadata(big_shard, tmp_path / "out.safetensors", {})
    assert raised.value.errno == errno.EIO


def test_create_with_known_checksums_copies_through_the_kernel(
    tmp_path: Path, base, monkeypatch
) -> None:
    from cwtrain.hotload.delta import checkpoint_checksums

    root, weights = base
    calls: list = []
    monkeypatch.setattr(os, "copy_file_range", copy_file_range_fake(calls, step=64), raising=False)
    mirror = WeightMirror.create(root, tmp_path / "mirror", checksums=checkpoint_checksums(root))
    assert calls and read_all(mirror.root) == {n: d for n, (_t, _s, d) in weights.items()}
    for shard in root.glob("*.safetensors"):
        assert (mirror.root / shard.name).read_bytes() == shard.read_bytes()


def test_a_failure_cancels_the_backlog_while_earlier_items_still_run() -> None:
    from cwtrain.hotload.mirror import _parallel

    release, started = threading.Event(), []
    threading.Timer(0.5, release.set).start()

    def work(i: int) -> int:
        if i == 0:
            release.wait(10)  # a long shard that is still copying when the next one fails
        elif i == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        else:
            started.append(i)
            time.sleep(0.01)
        return i

    with pytest.raises(OSError, match="No space left"):
        _parallel(work, list(range(50)), workers=2)
    assert release.is_set()  # it waited for the running shard before returning
    assert len(started) <= 2  # the queued shards were cancelled, not copied while waiting


def test_write_full_snapshot_leaves_nothing_behind_when_a_shard_fails(
    tmp_path: Path, base, monkeypatch
) -> None:
    from cwtrain.hotload import mirror as mirror_module

    root, _weights = base
    mirror = WeightMirror.create(root, tmp_path / "mirror")

    def full_disk(src, dst, metadata, **kwargs):
        dst.write_bytes(b"partial")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(mirror_module, "rewrite_shard_metadata", full_disk)
    dest = tmp_path / "staging" / "run-v000001"
    with pytest.raises(OSError, match="No space left"):
        mirror.write_full_snapshot(dest)
    assert not dest.exists() and not dest.with_name(dest.name + ".tmp").exists()


@pytest.mark.parametrize("bad_index", [[], {"weight_map": {"w": "missing.safetensors"}}])
def test_direct_mirror_apply_rejects_bad_index_without_mutation(tmp_path, bad_index):
    base = write_real_checkpoint(tmp_path / "base")
    mirror = WeightMirror.create(base, tmp_path / "mirror")
    before = {p.name: p.read_bytes() for p in mirror.root.iterdir() if p.is_file()}
    delta = tmp_path / "delta"
    delta.mkdir()
    (delta / "model.safetensors.index.json").write_text(json.dumps(bad_index))
    with pytest.raises(SnapshotError):
        mirror.apply_delta(delta)
    assert {p.name: p.read_bytes() for p in mirror.root.iterdir() if p.is_file()} == before
    assert not (mirror.root / ".pending").exists()


@pytest.mark.parametrize("operation", ["create", "mirror_cut", "full_copy"])
def test_invalid_index_does_not_replace_existing_destination(tmp_path, operation):
    from cwtrain.hotload.delta import write_full_snapshot

    base = write_real_checkpoint(tmp_path / "base")
    mirror = WeightMirror.create(base, tmp_path / "mirror")
    source = mirror.root if operation == "mirror_cut" else base
    (source / "model.safetensors.index.json").write_text("[]")
    destination = tmp_path / "destination"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_bytes(b"previous snapshot")
    with pytest.raises(SnapshotError, match="index must be a JSON object"):
        if operation == "create":
            WeightMirror.create(base, destination)
        elif operation == "mirror_cut":
            mirror.write_full_snapshot(destination)
        else:
            write_full_snapshot(base, destination)
    assert sentinel.read_bytes() == b"previous snapshot"
    assert not destination.with_name("destination.tmp").exists()
