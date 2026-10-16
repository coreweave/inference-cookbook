# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cwtrain.hotload import (
    ChecksumFormat,
    CompressionFormat,
    SnapshotConflictError,
    SnapshotError,
    SnapshotType,
    inspect_snapshot,
    snapshot_prefix,
    upload_snapshot,
)
from cwtrain.hotload.snapshot import (
    CONTENT_HASH_KEY,
    file_sha256,
    generated_index,
    iter_slime_versions,
    slime_identity,
    validate_identity,
)
from tests.conftest import (
    FakeS3,
    make_weights,
    write_delta_checkpoint,
    write_full_checkpoint,
    write_safetensors,
    write_single_file_checkpoint,
)


def test_inspect_full_checkpoint(tmp_path: Path) -> None:
    info = inspect_snapshot(write_full_checkpoint(tmp_path / "step_1"))
    assert info.type is SnapshotType.FULL
    assert info.incremental is None
    assert {p.name for p in info.files} == {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert info.total_bytes > 3072


def test_inspect_delta_checkpoint_derives_previous_identity(tmp_path: Path) -> None:
    info = inspect_snapshot(write_delta_checkpoint(tmp_path / "weight_v000002", version=2))
    assert info.type is SnapshotType.INCREMENTAL
    assert info.incremental is not None
    assert info.incremental.previous_snapshot_identity == "weight_v000001"
    assert info.incremental.compression_format is CompressionFormat.ZSTD
    assert info.incremental.checksum_format is ChecksumFormat.ADLER32


def test_inspect_delta_explicit_previous_identity(tmp_path: Path) -> None:
    info = inspect_snapshot(write_delta_checkpoint(tmp_path / "d"), previous_identity="step_9")
    assert info.incremental is not None
    assert info.incremental.previous_snapshot_identity == "step_9"


def test_inspect_delta_custom_identity_mapping(tmp_path: Path) -> None:
    info = inspect_snapshot(
        write_delta_checkpoint(tmp_path / "d", version=7),
        identity_for_version=lambda v: f"run1-{int(v)}",
    )
    assert info.incremental is not None
    assert info.incremental.previous_snapshot_identity == "run1-6"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoding": "overwrite"},
        {"checksum": "xxh3-128"},
        {"checksum": "blake3"},
        {"compression": "lz4"},
    ],
)
def test_inspect_rejects_unsupported_delta_formats(tmp_path: Path, kwargs) -> None:
    with pytest.raises(SnapshotError, match="not supported"):
        inspect_snapshot(write_delta_checkpoint(tmp_path / "d", **kwargs))


def test_inspect_missing_files(tmp_path: Path) -> None:
    root = write_full_checkpoint(tmp_path / "s")
    (root / "config.json").unlink()
    with pytest.raises(SnapshotError, match="config.json"):
        inspect_snapshot(root)

    root = write_full_checkpoint(tmp_path / "t")
    for shard in root.glob("*.safetensors"):
        shard.unlink()
    with pytest.raises(SnapshotError, match="existing immediate-child"):
        inspect_snapshot(root)

    root = write_full_checkpoint(tmp_path / "u")
    (root / "model.safetensors.index.json").unlink()
    with pytest.raises(SnapshotError, match="index"):
        inspect_snapshot(root)

    with pytest.raises(SnapshotError, match="not a directory"):
        inspect_snapshot(tmp_path / "nope")


def test_inspect_full_rejects_previous_identity(tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="full checkpoint"):
        inspect_snapshot(write_full_checkpoint(tmp_path / "s"), previous_identity="x")


def test_inspect_bad_index_json(tmp_path: Path) -> None:
    root = write_full_checkpoint(tmp_path / "s")
    (root / "model.safetensors.index.json").write_text("{not json")
    with pytest.raises(SnapshotError, match="valid JSON"):
        inspect_snapshot(root)


def test_validate_identity() -> None:
    assert validate_identity("step_000100") == "step_000100"
    for bad in ["", "a/b", ".", "..", "initial", "x" * 129]:
        with pytest.raises(SnapshotError):
            validate_identity(bad)


def test_snapshot_prefix_matches_server_contract() -> None:
    assert snapshot_prefix("", "v1") == "v1"
    assert snapshot_prefix("/", "v1") == "v1"
    assert snapshot_prefix("/checkpoints/", "v1") == "checkpoints/v1"
    assert snapshot_prefix("a//./b/", "v1") == "a//./b/v1"


def test_upload_snapshot_resumes_with_verified_content_and_uploads_index_last(
    tmp_path: Path,
) -> None:
    root = write_full_checkpoint(tmp_path / "step_1")
    shard = root / "model-00001-of-00002.safetensors"
    existing_key = "runs/exp1/step_1/model-00001-of-00002.safetensors"
    s3 = FakeS3()
    # an earlier attempt uploaded one shard before it was interrupted
    s3.seed("ckpts", existing_key, shard.read_bytes(), {CONTENT_HASH_KEY: file_sha256(shard)})
    seen: list[str] = []
    uri = upload_snapshot(
        root,
        bucket="ckpts",
        path_prefix="/runs/exp1/",
        identity="step_1",
        s3=s3,
        workers=2,
        on_file=lambda p, _n: seen.append(p.name),
    )
    assert uri == "s3://ckpts/runs/exp1/step_1/"
    keys = [k for _f, _b, k in s3.uploads]
    assert keys[-1] == "runs/exp1/step_1/model.safetensors.index.json"
    assert set(keys) == {
        "runs/exp1/step_1/config.json",
        "runs/exp1/step_1/model-00002-of-00002.safetensors",
        "runs/exp1/step_1/model.safetensors.index.json",
    }
    assert seen[-1] == "model.safetensors.index.json" and len(seen) == 4
    # every uploaded object carries the SHA-256 of its content
    for key in keys:
        name = key.rsplit("/", 1)[-1]
        assert s3.metadata[("ckpts", key)] == {CONTENT_HASH_KEY: file_sha256(root / name)}

    # a retry of the finished upload verifies everything and writes nothing
    s3.uploads.clear()
    upload_snapshot(root, bucket="ckpts", path_prefix="runs/exp1", identity="step_1", s3=s3)
    assert s3.uploads == []


def test_upload_snapshot_refuses_different_content_under_an_identity(tmp_path: Path) -> None:
    root = write_full_checkpoint(tmp_path / "step_1")
    s3 = FakeS3()
    s3.seed("ckpts", "step_1/config.json", b"{}", {CONTENT_HASH_KEY: "0" * 64})
    with pytest.raises(SnapshotConflictError, match="'step_1' already holds different content"):
        upload_snapshot(root, bucket="ckpts", identity="step_1", s3=s3)
    assert s3.uploads == []  # nothing written under the identity


def test_upload_snapshot_refuses_unverifiable_and_stray_objects(tmp_path: Path) -> None:
    root = write_full_checkpoint(tmp_path / "step_1")
    # same size but no content hash or ETag to compare: cannot be verified, so refused
    size = (root / "config.json").stat().st_size
    with pytest.raises(SnapshotConflictError) as exc:
        upload_snapshot(
            root,
            bucket="ckpts",
            identity="step_1",
            s3=FakeS3(existing={"step_1/config.json": size}),
        )
    assert exc.value.conflicts == ["config.json"]

    s3 = FakeS3()
    s3.seed("ckpts", "step_1/model-00003-of-00003.safetensors", b"old shard")
    with pytest.raises(SnapshotConflictError, match="not part of this snapshot"):
        upload_snapshot(root, bucket="ckpts", identity="step_1", s3=s3)


def test_upload_snapshot_accepts_objects_without_hash_metadata_by_etag(tmp_path: Path) -> None:
    root = write_full_checkpoint(tmp_path / "step_1")
    s3 = FakeS3()
    s3.seed("ckpts", "step_1/config.json", (root / "config.json").read_bytes())  # plain ETag
    upload_snapshot(root, bucket="ckpts", identity="step_1", s3=s3)
    assert "step_1/config.json" not in [k for _f, _b, k in s3.uploads]

    s3 = FakeS3()
    s3.seed("ckpts", "step_1/config.json", b"x" * (root / "config.json").stat().st_size)
    with pytest.raises(SnapshotConflictError):
        upload_snapshot(root, bucket="ckpts", identity="step_1", s3=s3)


def test_local_etag_matches_boto3_multipart_form(tmp_path: Path) -> None:
    import hashlib

    from cwtrain.hotload.snapshot import _MULTIPART_CHUNK, _local_etag

    data = bytes(range(256)) * (3 * _MULTIPART_CHUNK // 256 + 7)
    path = tmp_path / "blob"
    path.write_bytes(data)
    parts = [data[i : i + _MULTIPART_CHUNK] for i in range(0, len(data), _MULTIPART_CHUNK)]
    etag = hashlib.md5(b"".join(hashlib.md5(p).digest() for p in parts)).hexdigest()
    remote = f"{etag}-{len(parts)}"
    assert _local_etag(path, None, remote) == remote
    assert _local_etag(path, None, f"{etag}-9") is None  # other part size: not verifiable


def test_upload_snapshot_validates_identity(tmp_path: Path) -> None:
    root = write_full_checkpoint(tmp_path / "s")
    with pytest.raises(SnapshotError):
        upload_snapshot(root, bucket="b", identity="bad/id", s3=FakeS3())


def test_iter_slime_versions(tmp_path: Path) -> None:
    for v in (3, 1, 2):
        write_delta_checkpoint(tmp_path / slime_identity(v), version=v)
    (tmp_path / "weight_vabc").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    versions = list(iter_slime_versions(tmp_path))
    assert [v for v, _p in versions] == [1, 2, 3]
    assert versions[0][1].name == "weight_v000001"
    assert list(iter_slime_versions(tmp_path / "missing")) == []


def test_slime_full_checkpoint_dir_is_full_snapshot(tmp_path: Path) -> None:
    # slime's full mode writes a plain HF checkpoint; the index has no delta metadata.
    root = write_full_checkpoint(tmp_path / "weight_v000005")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    assert "delta_encoding" not in index["metadata"]
    assert inspect_snapshot(root).type is SnapshotType.FULL


def test_upload_rewrites_delta_index_with_identities(tmp_path: Path) -> None:
    root = write_delta_checkpoint(tmp_path / "weight_v000002", version=2)
    info = inspect_snapshot(root, previous_identity="run-v000000")  # v1 was an empty sync
    s3 = FakeS3()
    upload_snapshot(info, bucket="ckpts", path_prefix="hl", identity="run-v000002", s3=s3)
    body = json.loads(s3.bodies[("ckpts", "hl/run-v000002/model.safetensors.index.json")])
    assert body["metadata"] == {
        "version": "run-v000002",
        "base_version": "run-v000000",
        "delta_encoding": "xor",
        "compression_format": "zstd",
        "checksum_format": "adler32",
    }
    assert body["weight_map"] == {"layer.0.weight": "model-00001-of-00001.safetensors"}
    # the local file is untouched and the shard was uploaded as-is
    assert (
        json.loads((root / "model.safetensors.index.json").read_text())["metadata"]["version"]
        == "000002"
    )
    assert ("<bytes>", "ckpts", "hl/run-v000002/model.safetensors.index.json") == s3.uploads[-1]


def test_full_snapshot_index_uploaded_verbatim(tmp_path: Path) -> None:
    root = write_full_checkpoint(tmp_path / "s")
    s3 = FakeS3()
    upload_snapshot(root, bucket="b", identity="s", s3=s3)
    assert (
        s3.bodies[("b", "s/model.safetensors.index.json")]
        == (root / "model.safetensors.index.json").read_bytes()
    )


def test_delta_retry_verifies_the_rewritten_index(tmp_path: Path) -> None:
    root = write_delta_checkpoint(tmp_path / "weight_v000002", version=2)
    s3 = FakeS3()
    info = inspect_snapshot(root, previous_identity="run-v000001")
    upload_snapshot(info, bucket="ckpts", path_prefix="hl", identity="run-v000002", s3=s3)
    s3.uploads.clear()
    upload_snapshot(info, bucket="ckpts", path_prefix="hl", identity="run-v000002", s3=s3)
    assert s3.uploads == []  # identical retry
    # the same delta chained onto another base is different content under the same identity
    other = inspect_snapshot(root, previous_identity="run-v000000")
    with pytest.raises(SnapshotConflictError) as exc:
        upload_snapshot(other, bucket="ckpts", path_prefix="hl", identity="run-v000002", s3=s3)
    assert exc.value.conflicts == ["model.safetensors.index.json"]


def test_non_numeric_base_version_is_a_snapshot_error(tmp_path: Path) -> None:
    root = write_delta_checkpoint(tmp_path / "weight_v000002", version=2)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["metadata"]["base_version"] = "initial"
    index_path.write_text(json.dumps(index))
    with pytest.raises(SnapshotError, match="pass previous_identity"):
        inspect_snapshot(root)
    assert inspect_snapshot(root, previous_identity="step_000100").incremental is not None


def test_listing_that_makes_no_progress_is_an_error() -> None:
    from cwtrain.hotload.snapshot import _list_sizes

    class LoopingS3:
        def list_objects_v2(self, **_kw):
            return {"Contents": [], "IsTruncated": True, "NextContinuationToken": "same"}

    with pytest.raises(SnapshotError, match="did not make progress"):
        _list_sizes(LoopingS3(), "b", "p/")


def test_bad_pool_size_env_falls_back(monkeypatch) -> None:
    from cwtrain.infra.storage import _env_int

    monkeypatch.setenv("CW_S3_MAX_POOL", "lots")
    assert _env_int("CW_S3_MAX_POOL", 128) == 128
    monkeypatch.setenv("CW_S3_MAX_POOL", "64")
    assert _env_int("CW_S3_MAX_POOL", 128) == 64


@pytest.mark.parametrize("index", [[], None, True, 42, "index"])
def test_inspect_rejects_non_object_index(tmp_path, index):
    root = write_full_checkpoint(tmp_path / "s")
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(SnapshotError, match="index must be a JSON object"):
        inspect_snapshot(root)


@pytest.mark.parametrize(
    "field,value",
    [
        (field, value)
        for field in ("metadata", "weight_map")
        for value in (None, [], [["key", "value"]], "", 0, False)
        if field != "metadata" or value is not None
    ],
)
def test_inspect_rejects_non_object_fields(tmp_path, field, value):
    root = write_full_checkpoint(tmp_path / "s")
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index[field] = value
    path.write_text(json.dumps(index))
    with pytest.raises(SnapshotError, match=field + " must be a JSON object"):
        inspect_snapshot(root)


def test_inspect_requires_weight_map(tmp_path):
    root = write_full_checkpoint(tmp_path / "s")
    (root / "model.safetensors.index.json").write_text("{}")
    with pytest.raises(SnapshotError, match="weight_map must be a JSON object"):
        inspect_snapshot(root)


@pytest.mark.parametrize(
    "shard",
    [
        None,
        1,
        False,
        [],
        {},
        "",
        "config.json",
        ".",
        "..",
        "../model.safetensors",
        "/model.safetensors",
        "nested/model.safetensors",
        "./model.safetensors",
        "nested\\model.safetensors",
        "bad\0.safetensors",
    ],
)
def test_inspect_rejects_invalid_shard_references(tmp_path, shard):
    root = write_full_checkpoint(tmp_path / "s")
    (root / "nested").mkdir()
    (root / "nested" / "model.safetensors").write_bytes(b"weights")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": shard}})
    )
    with pytest.raises(SnapshotError, match="existing immediate-child") as error:
        inspect_snapshot(root)
    assert "layer.weight" in str(error.value)


@pytest.mark.parametrize("kind", ["missing", "directory", "broken_symlink"])
@pytest.mark.parametrize("delta", [False, True])
def test_inspect_rejects_missing_or_non_file_shard_before_upload(tmp_path, kind, delta):
    writer = write_delta_checkpoint if delta else write_full_checkpoint
    root = writer(tmp_path / "s")
    shard = root / "missing.safetensors"
    if kind == "directory":
        shard.mkdir()
    elif kind == "broken_symlink":
        shard.symlink_to(root / "absent")
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["weight_map"] = {"layer.weight": shard.name}
    path.write_text(json.dumps(index))
    s3 = FakeS3()
    with pytest.raises(SnapshotError, match="existing immediate-child") as error:
        upload_snapshot(root, bucket="ckpts", identity="step_1", s3=s3)
    assert str(path) in str(error.value)
    assert "layer.weight" in str(error.value)
    assert shard.name in str(error.value)
    assert not s3.objects


@pytest.mark.parametrize("metadata", ["absent", "null", "empty", "extra"])
def test_inspect_full_allows_optional_metadata_and_shared_shards(tmp_path, metadata):
    root = write_full_checkpoint(tmp_path / "s")
    shard = "model-00001-of-00002.safetensors"
    index = {"weight_map": {"a": shard, "b": shard}}
    if metadata != "absent":
        index["metadata"] = {"null": None, "empty": {}, "extra": {"custom": [1, 2]}}[metadata]
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    (root / "tokenizer.json").write_text("{}")
    # Fixture shards are opaque bytes: preflight deliberately does not parse tensor headers.
    info = inspect_snapshot(root)
    assert info.type is SnapshotType.FULL
    assert len(info.files) == 5  # includes the unreferenced shard and auxiliary file


def test_inspect_empty_delta_still_requires_shards(tmp_path):
    root = write_delta_checkpoint(tmp_path / "d")
    for shard in root.glob("*.safetensors"):
        shard.unlink()
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["weight_map"] = {}
    path.write_text(json.dumps(index))
    with pytest.raises(SnapshotError, match="weight shards"):
        inspect_snapshot(root)


def test_inspect_delta_allows_partial_weight_map_and_extra_metadata(tmp_path):
    root = write_delta_checkpoint(tmp_path / "d")
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    shard = next(iter(index["weight_map"].values()))
    index["weight_map"] = {"only.changed.tensor": shard, "another.changed.tensor": shard}
    index["metadata"]["custom"] = {"producer": "training"}
    path.write_text(json.dumps(index))
    info = inspect_snapshot(root)
    assert info.type is SnapshotType.INCREMENTAL
    assert info.index_metadata["custom"] == {"producer": "training"}
    assert info.incremental.previous_snapshot_identity == "weight_v000001"


@pytest.mark.parametrize("delta", [False, True])
def test_empty_weight_map_allowed_only_for_delta(tmp_path, delta):
    writer = write_delta_checkpoint if delta else write_full_checkpoint
    root = writer(tmp_path / "s")
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["weight_map"] = {}
    path.write_text(json.dumps(index))
    if delta:
        assert inspect_snapshot(root).empty_delta
    else:
        with pytest.raises(SnapshotError, match="full checkpoint weight_map must not be empty"):
            inspect_snapshot(root)


@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize(
    "replacement", [b"[]", b'{"weight_map":{"x":"missing.safetensors"}}', None]
)
def test_upload_uses_validated_index_even_if_disk_index_changes(tmp_path, delta, replacement):
    from cwtrain.hotload.snapshot import delta_index_for_upload

    writer = write_delta_checkpoint if delta else write_full_checkpoint
    root = writer(tmp_path / "s")
    info = inspect_snapshot(root)
    expected = delta_index_for_upload(info, "step_1") if delta else info.index_bytes
    index_path = root / "model.safetensors.index.json"
    if replacement is None:
        index_path.unlink()
    else:
        index_path.write_bytes(replacement)
    s3 = FakeS3()
    upload_snapshot(info, bucket="b", identity="step_1", s3=s3)
    assert s3.bodies[("b", "step_1/model.safetensors.index.json")] == expected
    assert s3.uploads[-1][2] == "step_1/model.safetensors.index.json"
    s3.uploads.clear()
    # The retry must compare with the captured bytes too, not the new/missing disk index.
    upload_snapshot(info, bucket="b", identity="step_1", s3=s3)
    assert not s3.uploads


def test_backslash_in_existing_shard_name_is_accepted(tmp_path):
    root = write_full_checkpoint(tmp_path / "s")
    shard = root / "a\\b.safetensors"
    shard.write_bytes(b"weights")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard.name}})
    )
    assert shard in inspect_snapshot(root).files


def test_validated_shardless_empty_delta_cannot_be_uploaded(tmp_path):
    root = write_delta_checkpoint(tmp_path / "delta")
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["weight_map"] = {}
    path.write_text(json.dumps(index))
    for shard in root.glob("*.safetensors"):
        shard.unlink()
    info = inspect_snapshot(root, allow_empty_delta=True)
    assert info.empty_delta
    s3 = FakeS3()
    with pytest.raises(SnapshotError, match="empty delta must be skipped"):
        upload_snapshot(info, bucket="b", identity="empty", s3=s3)
    assert not s3.uploads


# -------------------------------------------------------------- single-file checkpoints


def test_single_file_checkpoint_gets_an_index_from_the_header(tmp_path: Path) -> None:
    from cwtrain.hotload.mirror import read_safetensors_header

    root = write_single_file_checkpoint(tmp_path / "ckpt")
    shard = root / "model.safetensors"
    _header, data_start = read_safetensors_header(shard)
    with shard.open("r+b") as f:
        f.truncate(data_start)  # no tensor bytes left: only the header can be read
    info = inspect_snapshot(root)
    assert info.type is SnapshotType.FULL and info.index_generated
    expected = {
        "metadata": {"total_size": 128 + 30 + 16},
        "weight_map": dict.fromkeys(make_weights(), "model.safetensors"),
    }
    assert json.loads(info.index_bytes) == expected
    # formatted as save_pretrained writes an index, and the same bytes on every call
    assert info.index_bytes == (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode()
    assert generated_index(root) == info.index_bytes
    assert {p.name for p in info.files} == {
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
    }
    assert not (root / "model.safetensors.index.json").exists()  # nothing is written


def test_an_existing_index_is_used_as_it_is(tmp_path: Path) -> None:
    root = write_single_file_checkpoint(tmp_path / "ckpt")
    index = root / "model.safetensors.index.json"
    body = b'{"weight_map": {"lm_head.bias": "model.safetensors"}}'
    index.write_bytes(body)
    assert generated_index(root) is None
    info = inspect_snapshot(root)
    assert info.index_bytes == body and not info.index_generated and index in info.files
    s3 = FakeS3()
    upload_snapshot(info, bucket="b", identity="s", s3=s3)
    assert s3.bodies[("b", "s/model.safetensors.index.json")] == body == index.read_bytes()


def test_upload_sends_a_generated_index_last_and_a_retry_verifies_it(tmp_path: Path) -> None:
    root = write_single_file_checkpoint(tmp_path / "ckpt")
    info = inspect_snapshot(root)
    shard = root / "model.safetensors"
    s3 = FakeS3()
    # an earlier attempt uploaded the shard before it was interrupted
    s3.seed(
        "b",
        "hl/sft-v000030/model.safetensors",
        shard.read_bytes(),
        {CONTENT_HASH_KEY: file_sha256(shard)},
    )
    seen: list[tuple[str, int]] = []
    upload_snapshot(
        info,
        bucket="b",
        path_prefix="hl",
        identity="sft-v000030",
        s3=s3,
        on_file=lambda p, n: seen.append((p.name, n)),
    )
    assert s3.uploads[-1] == ("<bytes>", "b", "hl/sft-v000030/model.safetensors.index.json")
    assert {key for _f, _b, key in s3.uploads[:-1]} == {
        "hl/sft-v000030/config.json",
        "hl/sft-v000030/tokenizer_config.json",
    }
    key = ("b", "hl/sft-v000030/model.safetensors.index.json")
    assert s3.bodies[key] == info.index_bytes
    assert s3.metadata[key] == {CONTENT_HASH_KEY: hashlib.sha256(info.index_bytes).hexdigest()}
    assert seen[-1] == ("model.safetensors.index.json", len(info.index_bytes)) and len(seen) == 4

    # a retry generates the same index, finds every object identical and writes nothing
    s3.uploads.clear()
    upload_snapshot(root, bucket="b", path_prefix="hl", identity="sft-v000030", s3=s3)
    assert s3.uploads == []


def test_a_different_index_under_the_identity_is_a_conflict(tmp_path: Path) -> None:
    root = write_single_file_checkpoint(tmp_path / "ckpt")
    s3 = FakeS3()
    s3.seed("b", "sft-v000030/model.safetensors.index.json", b"{}", {CONTENT_HASH_KEY: "0" * 64})
    with pytest.raises(SnapshotConflictError) as exc:
        upload_snapshot(root, bucket="b", identity="sft-v000030", s3=s3)
    assert exc.value.conflicts == ["model.safetensors.index.json"]  # compared, not stray
    assert s3.uploads == []

    # the same bytes without the hash metadata are verified by ETag and kept
    s3 = FakeS3()
    s3.seed("b", "sft-v000030/model.safetensors.index.json", inspect_snapshot(root).index_bytes)
    upload_snapshot(root, bucket="b", identity="sft-v000030", s3=s3)
    assert "sft-v000030/model.safetensors.index.json" not in [k for _f, _b, k in s3.uploads]


@pytest.mark.parametrize("operation", ["inspect", "upload", "full_snapshot", "mirror"])
def test_a_lora_adapter_is_refused_with_a_merge_hint(tmp_path: Path, operation: str) -> None:
    from cwtrain.hotload import WeightMirror
    from cwtrain.hotload.delta import write_full_snapshot

    root = tmp_path / "checkpoint-30"  # what PEFT's save_pretrained writes for a LoRA run
    root.mkdir()
    lora_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    write_safetensors(root / "adapter_model.safetensors", {lora_a: ("F32", (2, 4), bytes(32))})
    (root / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 2}))
    s3 = FakeS3()
    with pytest.raises(SnapshotError, match="is a LoRA adapter; .* merge it into its base model"):
        if operation == "inspect":
            inspect_snapshot(root)
        elif operation == "upload":
            upload_snapshot(root, bucket="b", identity="sft-v000030", s3=s3)
        elif operation == "full_snapshot":
            write_full_snapshot(root, tmp_path / "sft-v000030")
        else:
            WeightMirror.create(root, tmp_path / "mirror")
    assert not s3.uploads
    assert not (tmp_path / "sft-v000030").exists() and not (tmp_path / "mirror").exists()


def _raw_safetensors(header: object) -> bytes:
    body = json.dumps(header).encode()
    return len(body).to_bytes(8, "little") + body


@pytest.mark.parametrize(
    "blob,message",
    [
        (b"abc", "is not a safetensors file"),
        (_raw_safetensors({"__metadata__": {"format": "pt"}}), "holds no tensors"),
        (_raw_safetensors({"w": {"dtype": "F32", "shape": [2]}}), "invalid safetensors header"),
        (_raw_safetensors({"w": {"data_offsets": [0]}}), "invalid safetensors header"),
        (_raw_safetensors({"w": [0, 8]}), "invalid safetensors header"),
    ],
)
def test_a_single_file_with_a_bad_header_is_refused(tmp_path: Path, blob, message) -> None:
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "model.safetensors").write_bytes(blob)
    (root / "config.json").write_text("{}")
    with pytest.raises(SnapshotError, match=message):
        inspect_snapshot(root)
