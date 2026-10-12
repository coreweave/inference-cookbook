# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from cwtrain.hotload import SnapshotError, SnapshotType, WeightMirror, inspect_snapshot
from cwtrain.hotload.delta import checkpoint_checksums, write_delta
from cwtrain.hotload.mirror import tensor_locations
from tests.conftest import make_weights, perturb, write_real_checkpoint


def read_all(root: Path) -> dict[str, bytes]:
    out = {}
    for name, loc in tensor_locations(root).items():
        with loc.path.open("rb") as f:
            f.seek(loc.offset)
            out[name] = f.read(loc.nbytes)
    return out


def test_write_delta_round_trips_through_the_mirror(tmp_path: Path) -> None:
    w0 = make_weights()
    w1 = perturb(w0, ("model.layers.0.weight", "lm_head.bias"), seed=3)
    base = write_real_checkpoint(tmp_path / "v0", w0)
    new = write_real_checkpoint(tmp_path / "v1", w1)

    stats = write_delta(
        base, new, tmp_path / "weight_v000001", version="000001", base_version="000000"
    )

    assert stats.tensors_total == 3 and stats.tensors_changed == 2 and not stats.is_empty
    assert stats.bytes_changed == len(w1["model.layers.0.weight"][2]) + len(w1["lm_head.bias"][2])
    assert stats.shards == ["model-00000-of-00001.safetensors"]
    info = inspect_snapshot(stats.path)
    assert info.type is SnapshotType.INCREMENTAL
    assert (
        info.incremental is not None
        and info.incremental.previous_snapshot_identity == "weight_v000000"
    )
    index = json.loads((stats.path / "model.safetensors.index.json").read_text())
    assert index["metadata"] == {
        "version": "000001",
        "base_version": "000000",
        "delta_encoding": "xor",
        "compression_format": "zstd",
        "checksum_format": "adler32",
    }
    assert set(index["weight_map"]) == {"model.layers.0.weight", "lm_head.bias"}

    # the mirror applies it like the engine does and lands on the new bytes
    mirror = WeightMirror.create(base, tmp_path / "mirror", version="000000")
    report = mirror.apply_delta(stats.path)
    assert report.tensors_changed == 2
    assert read_all(mirror.root) == {n: d for n, (_t, _s, d) in w1.items()}
    assert mirror.checksums == checkpoint_checksums(new)


def test_write_delta_empty_when_nothing_changed(tmp_path: Path) -> None:
    w0 = make_weights()
    base = write_real_checkpoint(tmp_path / "v0", w0)
    same = write_real_checkpoint(tmp_path / "v1", w0)
    stats = write_delta(
        base, same, tmp_path / "weight_v000001", version="000001", base_version="000000"
    )
    assert stats.is_empty and stats.shards == []
    assert not list(stats.path.glob("*.safetensors"))
    assert (stats.path / "model.safetensors.index.json").is_file()


def test_write_delta_shards_by_size(tmp_path: Path) -> None:
    w0 = make_weights()
    w1 = perturb(w0, tuple(w0), seed=5)  # every tensor changes
    base = write_real_checkpoint(tmp_path / "v0", w0)
    new = write_real_checkpoint(tmp_path / "v1", w1)
    stats = write_delta(
        base, new, tmp_path / "d", version="000002", base_version="000001", shard_bytes=1
    )
    assert len(stats.shards) == 3 and stats.tensors_changed == 3
    mirror = WeightMirror.create(base, tmp_path / "mirror", version="000001")
    mirror.apply_delta(stats.path)
    assert read_all(mirror.root) == {n: d for n, (_t, _s, d) in w1.items()}


def test_write_delta_rejects_layout_changes(tmp_path: Path) -> None:
    w0 = make_weights()
    base = write_real_checkpoint(tmp_path / "v0", w0)
    w_other = dict(w0)
    del w_other["lm_head.bias"]
    other = write_real_checkpoint(tmp_path / "v1", w_other)
    with pytest.raises(SnapshotError, match="different tensors"):
        write_delta(base, other, tmp_path / "d", version="000001", base_version="000000")


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_write_delta_streams_shards_instead_of_holding_the_model(
    tmp_path: Path, monkeypatch, workers: int
) -> None:
    import cwtrain.hotload.delta as delta

    # Exceed the prefetch window (2 * workers), including replacements submitted before the
    # first shard is flushed. With only three tensors, all reads can finish before the first write.
    w0 = {f"copy_{i}.{name}": tensor for i in range(8) for name, tensor in make_weights().items()}
    w1 = perturb(w0, tuple(w0), seed=9)  # every tensor changes
    base = write_real_checkpoint(tmp_path / "base", w0)
    new = write_real_checkpoint(tmp_path / "new", w1)
    last = tensor_locations(base)[max(w0)]  # write_delta diffs tensors in name order
    written = threading.Event()
    real_read, real_write = delta._read, delta.write_safetensors

    def read(loc):
        # Hold the last tensor's read until a shard is on disk. A writer that waits for every
        # tensor never gets here. Comparing read and write order instead would race the threads.
        if loc == last and not written.wait(timeout=10):
            raise AssertionError("no shard was written before the last tensor was read")
        return real_read(loc)

    def write(*args, **kwargs):
        result = real_write(*args, **kwargs)
        written.set()
        return result

    monkeypatch.setattr(delta, "_read", read)
    monkeypatch.setattr(delta, "write_safetensors", write)
    stats = write_delta(base, new, tmp_path / "d", version="000001", base_version="000000",
                        workers=workers, shard_bytes=1)  # fmt: skip
    assert len(stats.shards) == len(w0)
    assert sorted(p.name for p in (tmp_path / "d").iterdir()) == sorted(
        [*stats.shards, "model.safetensors.index.json"]
    )
    mirror = WeightMirror.create(base, tmp_path / "mirror", version="000000")
    mirror.apply_delta(tmp_path / "d")
    assert read_all(mirror.root) == {n: d for n, (_t, _s, d) in w1.items()}
