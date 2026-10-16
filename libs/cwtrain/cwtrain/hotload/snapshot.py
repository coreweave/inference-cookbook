# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Local snapshot inspection and upload to CoreWeave AI Object Storage.

A hot-load snapshot is a directory of safetensors shards plus ``model.safetensors.index.json``.
A full snapshot also has ``config.json``. :func:`inspect_snapshot` checks the local layout and
index structure before upload. It does not verify tensor contents or delta checksums. A full
checkpoint saved as one ``model.safetensors`` without an index, as ``save_pretrained`` saves a
model below its shard size, gets an index generated from that file's header, which is uploaded
with it.

Incremental snapshots use the delta layout that slime and miles write in
``--update-weight-mode delta``. The index ``metadata`` names the ``base_version``,
``delta_encoding`` (``xor``), ``compression_format`` (``zstd``) and ``checksum_format``. Only the
combination the hot-load API accepts is allowed: zstd XOR deltas with Adler-32 checksums.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cwtrain.hotload.errors import SnapshotAccessError, SnapshotConflictError, SnapshotError
from cwtrain.infra.models import (
    IncrementalSnapshotMetadata,
    SnapshotType,
)
from cwtrain.infra.storage import s3_client

log = logging.getLogger(__name__)

INDEX_FILE = "model.safetensors.index.json"
CONFIG_FILE = "config.json"
SHARD_SUFFIX = ".safetensors"
#: The one shard ``save_pretrained`` writes, without an index, for a model below its shard size.
SINGLE_FILE = "model.safetensors"
#: Marks a LoRA adapter directory (PEFT's ``save_pretrained``), which holds no full weights.
ADAPTER_CONFIG = "adapter_config.json"
#: User metadata key (``x-amz-meta-cwtrain-sha256``) carrying the SHA-256 of an uploaded object.
CONTENT_HASH_KEY = "cwtrain-sha256"
# boto3's default multipart_chunksize. Used to recompute the "<md5 of part md5s>-<parts>" ETag
# of objects uploaded without CONTENT_HASH_KEY.
_MULTIPART_CHUNK = 8 * 2**20
_HASH_CHUNK = 16 * 2**20

REQUIRED_FILES: dict[SnapshotType, tuple[str, ...]] = {
    SnapshotType.FULL: (CONFIG_FILE, INDEX_FILE),
    SnapshotType.INCREMENTAL: (INDEX_FILE,),
}

_IDENTITY_MAX_LEN = 128
_SLIME_VERSION_DIR = re.compile(r"^weight_v(\d{6})$")


def validate_identity(identity: str) -> str:
    """Check ``identity`` against the API's rules and return it unchanged.

    Raises:
        SnapshotError: ``identity`` is empty, too long, not a single path segment, or ``initial``.
    """
    if not identity or len(identity) > _IDENTITY_MAX_LEN:
        raise SnapshotError(f"identity must be 1-{_IDENTITY_MAX_LEN} characters: {identity!r}")
    if identity in (".", "..") or "/" in identity:
        raise SnapshotError(f"identity must be a single path segment: {identity!r}")
    if identity == "initial":
        raise SnapshotError("identity 'initial' is reserved for the deployment's base weights")
    return identity


def snapshot_prefix(path_prefix: str, identity: str) -> str:
    """The object key prefix for ``identity``, matching the server's literal-key contract."""
    prefix = path_prefix.strip("/")
    return identity if not prefix else f"{prefix}/{identity}"


def slime_identity(version: int | str) -> str:
    """The identity this library uses for a slime/miles ``weight_v{N:06d}`` directory."""
    return f"weight_v{int(version):06d}"


@dataclass(frozen=True)
class SnapshotInfo:
    """What :func:`inspect_snapshot` learned about a local checkpoint directory."""

    path: Path
    type: SnapshotType
    files: tuple[Path, ...]
    total_bytes: int
    index_bytes: bytes = field(repr=False)
    empty_delta: bool = False
    incremental: IncrementalSnapshotMetadata | None = None
    index_metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    #: The directory has no index file, so ``index_bytes`` comes from :func:`generated_index`.
    index_generated: bool = False

    @property
    def is_incremental(self) -> bool:
        return self.type is SnapshotType.INCREMENTAL


def generated_index(root: Path) -> bytes | None:
    """The index of a checkpoint saved as one ``model.safetensors`` without an index.

    ``save_pretrained`` writes no index for a model below its shard size, but the platform
    requires one for a full snapshot. This one is built from the shard's header alone, without
    reading a tensor: ``total_size`` in its ``metadata`` is the bytes of all tensors, and the
    ``weight_map`` maps each tensor to ``model.safetensors``. It is formatted the way
    ``save_pretrained`` writes the index of a sharded model.

    Returns:
        The index as JSON, or ``None`` when ``root`` has an index or no ``model.safetensors``.

    Raises:
        SnapshotError: ``root`` is a LoRA adapter, or the shard's header names no tensors or is
            invalid.
    """
    if (root / INDEX_FILE).exists():
        return None
    shard = root / SINGLE_FILE
    if not shard.is_file():
        if (root / ADAPTER_CONFIG).is_file():
            raise SnapshotError(
                f"{root} is a LoRA adapter; hot-load serves full checkpoints and deltas, so "
                "merge it into its base model first"
            )
        return None
    from cwtrain.hotload.mirror import read_safetensors_header  # mirror imports this module

    header, _ = read_safetensors_header(shard)
    tensors = {name: info for name, info in header.items() if name != "__metadata__"}
    try:
        total = sum(end - begin for begin, end in (t["data_offsets"] for t in tensors.values()))
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError(f"{shard} has an invalid safetensors header") from exc
    if not tensors:
        raise SnapshotError(f"{shard} holds no tensors")
    index = {"metadata": {"total_size": total}, "weight_map": dict.fromkeys(tensors, SINGLE_FILE)}
    # The same bytes for the same header, so a retried upload can compare the index by content.
    return (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()


def _read_index_bytes(root: Path) -> bytes:
    index_path = root / INDEX_FILE
    try:
        return index_path.read_bytes()
    except OSError as exc:
        raise SnapshotError(f"{index_path} is missing or unreadable: {exc}") from exc


def load_index(root: Path, *, body: bytes | None = None) -> dict[str, Any]:
    """Read the index of ``root`` and validate its structure and shard references.

    Each shard must be an existing ``*.safetensors`` file directly under ``root``. Missing or null
    metadata becomes an empty object. A delta may have an empty ``weight_map``; a full checkpoint
    may not. Tensors and delta checksums are not checked.

    Args:
        body: Index bytes to parse instead of reading the file.

    Raises:
        SnapshotError: the index is missing, not valid JSON, or structurally invalid.
    """
    index_path = root / INDEX_FILE
    try:
        index = json.loads(_read_index_bytes(root) if body is None else body)
    except (ValueError, UnicodeError) as exc:
        raise SnapshotError(f"{index_path} is not valid JSON: {exc}") from exc
    if not isinstance(index, dict):
        raise SnapshotError(f"{index_path}: index must be a JSON object")
    metadata = index.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise SnapshotError(f"{index_path}: metadata must be a JSON object when present")
    index["metadata"] = metadata
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SnapshotError(
            f"{index_path}: weight_map must be a JSON object mapping tensors to shards"
        )
    if not weight_map and "delta_encoding" not in metadata:
        raise SnapshotError(f"{index_path}: full checkpoint weight_map must not be empty")
    names = {p.name for p in root.iterdir() if p.is_file()}
    for tensor, shard in weight_map.items():
        if not isinstance(shard, str) or not shard.endswith(SHARD_SUFFIX) or shard not in names:
            raise SnapshotError(
                f"{index_path}: weight_map[{tensor!r}] references shard {shard!r}; "
                f"expected an existing immediate-child {SHARD_SUFFIX} file"
            )
    return index


def inspect_snapshot(
    path: str | os.PathLike[str],
    *,
    previous_identity: str | None = None,
    identity_for_version: Callable[[str], str] = slime_identity,
    allow_empty_delta: bool = False,
) -> SnapshotInfo:
    """Validate a checkpoint directory and classify it as a full or incremental snapshot.

    The directory is incremental when its index ``metadata`` has a ``delta_encoding`` key (the
    slime/miles delta layout). A checkpoint saved as one ``model.safetensors`` without an index
    gets the index from :func:`generated_index`, and the result has ``index_generated`` set.

    Args:
        path: The checkpoint directory.
        previous_identity: For a delta, the identity of the snapshot it was computed against. If
            omitted, derived from the index's ``base_version`` with ``identity_for_version``.
        identity_for_version: Maps a ``base_version`` to an identity. Default:
            :func:`slime_identity`.
        allow_empty_delta: Accept a delta with no changed tensors, so the caller can inspect it
            and skip it. The result has ``empty_delta`` set and cannot be uploaded.

    Raises:
        SnapshotError: the index structure or a shard reference is invalid, a required file
            is missing, no shard is present, the directory is a LoRA adapter, or the delta uses
            an encoding, compression or checksum the hot-load API does not accept.
    """
    root = Path(path)
    if not root.is_dir():
        raise SnapshotError(f"snapshot path is not a directory: {root}")

    files = tuple(sorted(p for p in root.iterdir() if p.is_file()))
    names = {p.name for p in files}
    generated = generated_index(root)
    if generated is None and INDEX_FILE not in names:
        raise SnapshotError(f"{root} is missing {INDEX_FILE}")
    index_bytes = _read_index_bytes(root) if generated is None else generated
    index = load_index(root, body=index_bytes)
    metadata = index["metadata"]
    empty_delta = "delta_encoding" in metadata and not index["weight_map"]
    if not any(n.endswith(SHARD_SUFFIX) for n in names) and not (allow_empty_delta and empty_delta):
        raise SnapshotError(f"{root} has no *{SHARD_SUFFIX} weight shards")

    incremental: IncrementalSnapshotMetadata | None = None
    if "delta_encoding" in metadata:
        snapshot_type = SnapshotType.INCREMENTAL
        incremental = _delta_metadata(root, metadata, previous_identity, identity_for_version)
    else:
        snapshot_type = SnapshotType.FULL
        if previous_identity is not None:
            raise SnapshotError(
                f"{root} is a full checkpoint; previous_identity only applies to deltas"
            )

    present = names | ({INDEX_FILE} if generated is not None else set())
    missing = [n for n in REQUIRED_FILES[snapshot_type] if n not in present]
    if missing:
        raise SnapshotError(f"{root} is missing {', '.join(missing)}")

    # The server reads only the snapshot's immediate children, so nested dirs are ignored.
    nested = [p.name for p in root.iterdir() if p.is_dir()]
    if nested:
        log.warning("%s contains subdirectories that will not be uploaded: %s", root, nested)

    return SnapshotInfo(
        path=root,
        type=snapshot_type,
        files=files,
        total_bytes=sum(p.stat().st_size for p in files),
        index_bytes=index_bytes,
        empty_delta=empty_delta,
        incremental=incremental,
        index_metadata=metadata,
        index_generated=generated is not None,
    )


def _delta_metadata(
    root: Path,
    metadata: dict[str, Any],
    previous_identity: str | None,
    identity_for_version: Callable[[str], str],
) -> IncrementalSnapshotMetadata:
    encoding = str(metadata.get("delta_encoding", "")).lower()
    if encoding != "xor":
        raise SnapshotError(
            f"{root}: delta_encoding {encoding!r} is not supported; hot-load accepts 'xor' only"
        )
    compression = str(metadata.get("compression_format", "zstd")).lower()
    if compression != "zstd":
        raise SnapshotError(
            f"{root}: compression_format {compression!r} is not supported; use 'zstd'"
        )
    checksum = str(metadata.get("checksum_format", "")).lower().replace("-", "")
    if checksum != "adler32":
        raise SnapshotError(
            f"{root}: checksum_format {checksum or 'unset'!r} is not supported; "
            "hot-load accepts 'adler32' only (slime: --update-weight-delta-checksum adler32)"
        )
    if previous_identity is None:
        base = metadata.get("base_version")
        if base is None:
            raise SnapshotError(
                f"{root}: delta index has no base_version; pass previous_identity explicitly"
            )
        try:
            previous_identity = identity_for_version(str(base))
        except (TypeError, ValueError) as exc:
            raise SnapshotError(
                f"{root}: cannot derive the previous identity from base_version {base!r} ({exc}); "
                "pass previous_identity= or an identity_for_version= that understands it"
            ) from exc
    return IncrementalSnapshotMetadata(
        previous_snapshot_identity=validate_identity(previous_identity)
    )


def delta_index_for_upload(info: SnapshotInfo, identity: str) -> bytes:
    """Return the index bytes to upload for a delta snapshot stored as ``identity``.

    The engine validates a delta's ``metadata`` against the snapshot chain. ``version`` must be
    the snapshot identity and ``base_version`` the predecessor's identity, with the ``xor`` /
    ``adler32`` / ``zstd`` markers. slime and miles write their own version counters
    (``000001``), so the index is rewritten at upload time. Shards are not changed.

    Raises:
        SnapshotError: ``info`` is not an incremental snapshot.
    """
    if info.incremental is None:
        raise SnapshotError("delta_index_for_upload needs an incremental snapshot")
    index = json.loads(info.index_bytes)
    metadata = dict(index["metadata"])
    metadata.update(
        {
            "version": identity,
            "base_version": info.incremental.previous_snapshot_identity,
            "delta_encoding": "xor",
            "compression_format": "zstd",
            "checksum_format": "adler32",
        }
    )
    index["metadata"] = metadata
    return json.dumps(index, indent=1).encode()


# ------------------------------------------------------------------------------ upload


def upload_snapshot(
    snapshot: SnapshotInfo | str | os.PathLike[str],
    *,
    bucket: str,
    identity: str,
    path_prefix: str = "",
    s3: Any = None,
    workers: int = 8,
    on_file: Callable[[Path, int], None] | None = None,
) -> str:
    """Upload every file of ``snapshot`` under ``s3://bucket/<path_prefix>/<identity>/``.

    Identities are immutable. Each object is written with the SHA-256 of its content in its
    metadata (``x-amz-meta-cwtrain-sha256``). If objects already exist under the identity, each
    one must match the local file byte for byte. Then only the missing files are uploaded, so
    you can retry an interrupted upload. Objects without the hash metadata are compared by ETag,
    which works for single-part and default boto3 multipart uploads.

    Files are uploaded concurrently. The index goes last, also when :func:`inspect_snapshot`
    generated it, so a reader that sees the index can rely on the shards being present.

    Args:
        snapshot: A :class:`SnapshotInfo`, or a directory to pass to :func:`inspect_snapshot`.
        bucket: The destination bucket.
        identity: The snapshot identity, a single path segment.
        path_prefix: Key prefix under which snapshots are stored.
        s3: A boto3 S3 client. Built from the default credential chain when omitted.
        workers: Number of concurrent uploads.
        on_file: Called with ``(path, size)`` after each file is uploaded or already present.

    Returns:
        The ``s3://`` URI of the snapshot prefix.

    Raises:
        SnapshotConflictError: an existing object differs, cannot be verified, or is not part of
            this snapshot. Nothing is written.
        SnapshotError: the snapshot is an empty delta, or ``identity`` is invalid.
    """
    info = snapshot if isinstance(snapshot, SnapshotInfo) else inspect_snapshot(snapshot)
    if info.empty_delta:
        raise SnapshotError("an empty delta must be skipped, not uploaded")
    validate_identity(identity)
    s3 = s3 or s3_client()
    prefix = snapshot_prefix(path_prefix, identity)
    index_key = f"{prefix}/{INDEX_FILE}"

    # Shards and other files are uploaded as they are; a delta's index is rewritten.
    local: dict[str, Path] = {f"{prefix}/{p.name}": p for p in info.files}
    local.setdefault(index_key, info.path / INDEX_FILE)  # a generated index has no local file
    index_body = delta_index_for_upload(info, identity) if info.is_incremental else info.index_bytes
    digests: dict[str, str] = {index_key: hashlib.sha256(index_body).hexdigest()}

    def digest(key: str) -> str:
        if key not in digests:
            digests[key] = file_sha256(local[key])
        return digests[key]

    existing = _list_sizes(s3, bucket, prefix + "/")
    if existing:
        stray = sorted(set(existing) - set(local))
        conflicts = [f"{key.rsplit('/', 1)[-1]} (not part of this snapshot)" for key in stray]

        def check(key: str) -> str | None:
            return (
                None
                if _object_matches(
                    s3,
                    bucket,
                    key,
                    local[key],
                    digest(key),
                    index_body if key == index_key else None,
                )
                else key.rsplit("/", 1)[-1]
            )

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            conflicts += [c for c in pool.map(check, sorted(set(existing) & set(local))) if c]
        if conflicts:
            raise SnapshotConflictError(identity, conflicts)
        log.info(
            "%s: %d of %d objects already uploaded with identical content; completing the upload",
            identity,
            len(existing),
            len(local),
        )

    def put(key: str) -> tuple[Path, int]:
        path = local[key]
        size = path.stat().st_size
        if key in existing:
            return path, size
        s3.upload_file(
            str(path), bucket, key, ExtraArgs={"Metadata": {CONTENT_HASH_KEY: digest(key)}}
        )
        return path, size

    shard_keys = [k for k in local if k != index_key]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for future in as_completed(pool.submit(put, k) for k in shard_keys):
            path, size = future.result()
            if on_file:
                on_file(path, size)
    if index_key not in existing:
        s3.put_object(
            Bucket=bucket,
            Key=index_key,
            Body=index_body,
            Metadata={CONTENT_HASH_KEY: digests[index_key]},
        )
    if on_file:
        on_file(local[index_key], len(index_body))

    uri = f"s3://{bucket}/{prefix}/"
    log.info(
        "uploaded %s (%d files, %.1f GiB) to %s",
        info.path,
        len(info.files),
        info.total_bytes / 2**30,
        uri,
    )
    return uri


def check_snapshot_access(s3: Any, bucket: str, path_prefix: str = "") -> str:
    """Check that ``s3``'s credentials can do what a trainer does under the snapshot prefix.

    It lists the prefix, reads a key that does not exist and starts a multipart upload, which it
    aborts. So it needs exactly the trainer's permissions (``s3:ListBucket``, ``s3:GetObject``,
    ``s3:PutObject``, ``s3:AbortMultipartUpload``) and leaves nothing behind, which matters for
    a trainer that may not delete. Run it before training to catch a bucket or organization
    policy that does not cover the trainer.

    Returns:
        The ``s3://`` URI of the prefix.

    Raises:
        SnapshotAccessError: the object store refused a call; it names the S3 permission and the
            error code. Missing credentials and transport failures propagate unchanged.
    """
    from botocore.exceptions import ClientError

    prefix = path_prefix.strip("/")
    base = f"{prefix}/" if prefix else ""
    uri = f"s3://{bucket}/{base}"
    probe = f"{base}_preflight/{uuid.uuid4().hex}"

    def refused(action: str, exc: ClientError) -> SnapshotAccessError:
        return SnapshotAccessError(uri, action, exc.response.get("Error", {}).get("Code", "?"))

    try:
        s3.list_objects_v2(Bucket=bucket, Prefix=base, MaxKeys=1)
    except ClientError as exc:
        raise refused("s3:ListBucket", exc) from exc
    try:
        s3.head_object(Bucket=bucket, Key=probe)
    except ClientError as exc:
        # The key never exists. With s3:ListBucket, a missing key reads as 404, not 403.
        if exc.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise refused("s3:GetObject", exc) from exc
    try:
        upload_id = s3.create_multipart_upload(Bucket=bucket, Key=probe)["UploadId"]
    except ClientError as exc:
        raise refused("s3:PutObject", exc) from exc
    try:
        s3.abort_multipart_upload(Bucket=bucket, Key=probe, UploadId=upload_id)
    except ClientError as exc:
        raise refused("s3:AbortMultipartUpload", exc) from exc
    return uri


def file_sha256(path: str | os.PathLike[str]) -> str:
    """SHA-256 hex digest of a file, read in large chunks (hashlib releases the GIL)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _object_matches(
    s3: Any, bucket: str, key: str, path: Path, sha256: str, body: bytes | None
) -> bool:
    """Whether the object at ``key`` holds exactly the local content (``body`` if given)."""
    head = s3.head_object(Bucket=bucket, Key=key)
    metadata = {k.lower(): v for k, v in (head.get("Metadata") or {}).items()}
    if CONTENT_HASH_KEY in metadata:
        return metadata[CONTENT_HASH_KEY] == sha256
    size = len(body) if body is not None else path.stat().st_size
    if int(head.get("ContentLength", -1)) != size:
        return False
    etag = str(head.get("ETag", "")).strip('"')
    return bool(etag) and etag == _local_etag(path, body, etag)


def _local_etag(path: Path, body: bytes | None, remote_etag: str) -> str | None:
    """The ETag S3 would report for the local content, in the remote ETag's form."""
    data_iter: Iterable[bytes]
    if body is not None:
        data_iter = (body[i : i + _MULTIPART_CHUNK] for i in range(0, len(body), _MULTIPART_CHUNK))
    else:

        def chunks() -> Iterable[bytes]:
            with open(path, "rb") as f:
                while chunk := f.read(_MULTIPART_CHUNK):
                    yield chunk

        data_iter = chunks()
    if "-" not in remote_etag:
        h = hashlib.md5(usedforsecurity=False)
        for chunk in data_iter:
            h.update(chunk)
        return h.hexdigest()
    parts = [hashlib.md5(chunk, usedforsecurity=False).digest() for chunk in data_iter]
    if str(len(parts)) != remote_etag.rsplit("-", 1)[1]:
        return None  # uploaded with another part size: cannot recompute, so treat as unverified
    return f"{hashlib.md5(b''.join(parts), usedforsecurity=False).hexdigest()}-{len(parts)}"


def _list_sizes(s3: Any, bucket: str, prefix: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        for obj in page.get("Contents") or []:
            sizes[obj["Key"]] = int(obj.get("Size", -1))
        if not page.get("IsTruncated"):
            return sizes
        token = page.get("NextContinuationToken")
        if not token or token in seen_tokens:
            raise SnapshotError(
                f"listing s3://{bucket}/{prefix} did not make progress (continuation token "
                f"{token!r}); the object store returned an inconsistent page"
            )
        seen_tokens.add(token)


def iter_slime_versions(disk_dir: str | os.PathLike[str]) -> Iterable[tuple[int, Path]]:
    """Yield ``(version, path)`` for every ``weight_v{N:06d}`` directory, ascending."""
    root = Path(disk_dir)
    found: list[tuple[int, Path]] = []
    if root.is_dir():
        for child in root.iterdir():
            match = _SLIME_VERSION_DIR.match(child.name)
            if match and child.is_dir():
                found.append((int(match.group(1)), child))
    return sorted(found)
