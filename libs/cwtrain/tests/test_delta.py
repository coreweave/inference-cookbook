# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import pytest

from cwtrain.hotload import SnapshotError, SnapshotType, WeightMirror, inspect_snapshot
from cwtrain.hotload.delta import checkpoint_checksums, write_delta, write_full_snapshot
from cwtrain.hotload.mirror import read_safetensors_header, tensor_locations
from tests.conftest import (
    make_weights,
    perturb,
    write_real_checkpoint,
    write_single_file_checkpoint,
)


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


# ------------------------------------------------------------------ write_full_snapshot


def verified_full_snapshot(snap: Path) -> dict:
    """The index of ``snap``, after the checks the engine makes when it installs a full snapshot:
    each shard in the index holds exactly the tensors that the index maps to it, and the
    Adler-32 that its header records for each one matches the tensor's bytes."""
    info = inspect_snapshot(snap)
    assert info.type is SnapshotType.FULL and not info.index_generated
    index = json.loads(info.index_bytes)
    assert index["metadata"]["checksum_format"] == "adler32"
    checksums = checkpoint_checksums(snap)
    by_shard: dict[str, set[str]] = {}
    for name, shard in index["weight_map"].items():
        by_shard.setdefault(shard, set()).add(name)
    for shard, names in by_shard.items():
        header, _ = read_safetensors_header(snap / shard)
        assert set(header) - {"__metadata__"} == names
        assert {name: header["__metadata__"][name] for name in names} == {
            name: checksums[name] for name in names
        }
    assert index["metadata"]["total_size"] == sum(
        loc.nbytes for loc in tensor_locations(snap).values()
    )
    return index


def with_tied_head(weights):
    """``weights`` plus a tied ``lm_head.weight`` that is a second copy of the embedding, saved
    between other tensors, as an FSDP2 full-state-dict gather saves it."""
    embedding = ("F32", (4, 8), bytes(range(128)))
    return {"model.embed_tokens.weight": embedding, "lm_head.weight": embedding, **weights}


def test_write_full_snapshot_of_a_single_file_checkpoint(tmp_path: Path) -> None:
    weights = make_weights()
    src = write_single_file_checkpoint(tmp_path / "src", weights)
    snap = write_full_snapshot(src, tmp_path / "sft-v000030")
    index = verified_full_snapshot(snap)
    assert index == {
        "metadata": {"total_size": 128 + 30 + 16, "checksum_format": "adler32"},
        "weight_map": dict.fromkeys(weights, "model.safetensors"),
    }
    assert read_all(snap) == {n: d for n, (_t, _s, d) in weights.items()}
    assert sorted(p.name for p in snap.iterdir()) == [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ]
    assert not (src / "model.safetensors.index.json").exists()  # the source is left as it is


def test_write_full_snapshot_keeps_only_the_named_tensors(tmp_path: Path) -> None:
    saved = with_tied_head(make_weights())
    src = write_single_file_checkpoint(tmp_path / "src", saved)
    base = sorted(set(saved) - {"lm_head.weight"})  # the tensors the deployment's base holds
    snap = write_full_snapshot(src, tmp_path / "sft-v000030", names=base)
    index = verified_full_snapshot(snap)
    assert sorted(index["weight_map"]) == base
    assert index["metadata"]["total_size"] == sum(len(saved[n][2]) for n in base)
    assert read_all(snap) == {n: saved[n][2] for n in base}
    header, _ = read_safetensors_header(snap / "model.safetensors")
    assert sorted(set(header) - {"__metadata__"}) == base
    assert {n: (header[n]["dtype"], tuple(header[n]["shape"])) for n in base} == {
        n: saved[n][:2] for n in base
    }
    assert sorted(set(header["__metadata__"]) - {"format"}) == base  # a checksum for each

    # A source with checksums of its own keeps none for a tensor it drops.
    fewer = [n for n in base if n != "model.layers.0.weight"]
    again = write_full_snapshot(snap, tmp_path / "again", names=fewer)
    header, _ = read_safetensors_header(again / "model.safetensors")
    assert sorted(set(header["__metadata__"]) - {"format"}) == fewer
    assert sorted(verified_full_snapshot(again)["weight_map"]) == fewer


def test_write_full_snapshot_refuses_a_name_the_checkpoint_lacks(tmp_path: Path) -> None:
    src = write_single_file_checkpoint(tmp_path / "src")
    dest = tmp_path / "sft-v000030"
    dest.mkdir()
    (dest / "keep").write_text("previous snapshot")
    with pytest.raises(SnapshotError, match=r"lacks 1 of the tensors to keep, e\.g\. \['x'\]"):
        write_full_snapshot(src, dest, names=["lm_head.bias", "x"])
    assert (dest / "keep").read_text() == "previous snapshot"


def test_write_full_snapshot_leaves_out_a_shard_that_keeps_no_tensor(tmp_path: Path) -> None:
    src = write_real_checkpoint(tmp_path / "src")  # lm_head.bias is alone in the second shard
    keep = ["model.layers.0.weight", "model.layers.1.weight"]
    snap = write_full_snapshot(src, tmp_path / "pruned", names=keep)
    first = "model-00001-of-00002.safetensors"
    assert [p.name for p in snap.glob("*.safetensors")] == [first]
    assert verified_full_snapshot(snap)["weight_map"] == dict.fromkeys(keep, first)
    # a shard that keeps all of its tensors is written as it is without names
    whole = write_full_snapshot(src, tmp_path / "whole")
    assert (snap / first).read_bytes() == (whole / first).read_bytes()


def check_safetensors_layout(shard: Path, tensors: dict) -> None:
    """``shard`` follows the safetensors format and holds ``tensors`` byte for byte."""
    header, data_start = read_safetensors_header(shard)
    assert data_start % 8 == 0  # the header is padded to 8 bytes
    assert set(header) - {"__metadata__"} == set(tensors)
    assert all(isinstance(v, str) for v in header["__metadata__"].values())
    # The data starts at offset 0 and has no gaps or overlaps up to the end of the file.
    spans = sorted(tuple(header[name]["data_offsets"]) for name in tensors)
    assert spans[0][0] == 0 and data_start + spans[-1][1] == shard.stat().st_size
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))
    sizes = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1}
    data = shard.read_bytes()
    for name, (dtype, shape, raw) in tensors.items():
        begin, end = header[name]["data_offsets"]
        assert (header[name]["dtype"], tuple(header[name]["shape"])) == (dtype, shape)
        assert end - begin == math.prod(shape) * sizes[dtype]
        assert data[data_start + begin : data_start + end] == raw


def test_written_shards_are_valid_safetensors(tmp_path: Path) -> None:
    saved = with_tied_head(make_weights())
    src = write_single_file_checkpoint(tmp_path / "src", saved)
    base = {n: saved[n] for n in saved if n != "lm_head.weight"}
    check_safetensors_layout(
        write_full_snapshot(src, tmp_path / "all") / "model.safetensors", saved
    )
    check_safetensors_layout(
        write_full_snapshot(src, tmp_path / "base", names=base) / "model.safetensors", base
    )


def test_written_shards_load_with_the_safetensors_package(tmp_path: Path) -> None:
    import numpy as np

    safetensors = pytest.importorskip("safetensors")
    rng = np.random.default_rng(7)
    embedding = ("F32", (6, 4), rng.standard_normal((6, 4), dtype=np.float32).tobytes())
    saved = {
        "model.embed_tokens.weight": embedding,
        "lm_head.weight": embedding,  # the tied duplicate, between two other tensors
        "model.layers.0.mlp.down_proj.weight": (
            "F16",
            (3, 5),
            rng.standard_normal((3, 5)).astype(np.float16).tobytes(),
        ),
        "model.norm.weight": ("I8", (7,), rng.integers(-128, 127, 7, dtype=np.int8).tobytes()),
    }
    src = write_single_file_checkpoint(tmp_path / "src", saved)
    base = [n for n in saved if n != "lm_head.weight"]
    for snap, names in (
        (write_full_snapshot(src, tmp_path / "all"), list(saved)),
        (write_full_snapshot(src, tmp_path / "base", names=base), base),
    ):
        header, _ = read_safetensors_header(snap / "model.safetensors")
        with safetensors.safe_open(str(snap / "model.safetensors"), framework="numpy") as f:
            assert sorted(f.keys()) == sorted(names)
            assert f.metadata() == header["__metadata__"]
            for name in names:
                tensor = f.get_tensor(name)
                assert (tensor.shape, tensor.tobytes()) == (saved[name][1], saved[name][2])
