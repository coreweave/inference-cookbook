# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Shared fixtures: a fake CoreWeave API served through an httpx mock transport."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from cwtrain.infra import TOKEN_ENV_VARS, InferenceClient

DEPLOYMENT_ID = "11111111-1111-4111-8111-111111111111"
NOW = "2026-09-22T12:00:00.123456789Z"


def deployment_json(
    *,
    ready: bool = True,
    hot_load: bool = True,
    bucket: str = "ckpts",
    prefix: str = "runs/exp1",
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": DEPLOYMENT_ID,
        "name": "qwen3-8b-rl",
        "organizationId": "org-abc",
        "runtime": {"engine": "dynamo-vllm", "engineVersion": "0.4.0"},
        "model": {"name": "Qwen/Qwen3-8B", "bucket": "models", "path": "qwen3-8b"},
    }
    if hot_load:
        spec["hotLoad"] = {
            "bucket": bucket,
            "pathPrefix": prefix,
            "transitionMode": "TRANSITION_MODE_ASYNC",
        }
    return {
        "deployment": {
            "spec": spec,
            "status": {
                "createdAt": NOW,
                "updatedAt": NOW,
                "status": "STATUS_READY" if ready else "STATUS_UPDATING",
                "conditions": [],
            },
        }
    }


def hot_load_json(
    hot_load_id: str,
    identity: str,
    state: str,
    *,
    snapshot_type: str = "SNAPSHOT_TYPE_FULL",
    incremental: dict[str, Any] | None = None,
    chain: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": hot_load_id,
        "deploymentId": DEPLOYMENT_ID,
        "organizationId": "org-abc",
        "identity": identity,
        "type": snapshot_type,
        "promptCachePolicy": "PROMPT_CACHE_POLICY_PRESERVE",
        "snapshotChain": chain
        or [{"identity": identity, "type": snapshot_type, "hotLoadId": hot_load_id}],
    }
    if incremental:
        spec["incrementalSnapshotMetadata"] = incremental
    status: dict[str, Any] = {"createdAt": NOW, "updatedAt": NOW, "state": state}
    if state in ("HOT_LOAD_STATE_COMPLETED", "HOT_LOAD_STATE_FAILED"):
        status["finishedAt"] = NOW
    return {"spec": spec, "status": status}


def replica_json(name: str, state: str, identity: str = "", error: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "zone": "US-EAST-04A",
        "state": state,
        "currentSnapshotIdentity": identity,
        "errorReason": error,
        "updatedAt": NOW,
    }


def error_json(code: str | int, message: str) -> dict[str, Any]:
    return {"code": code, "message": message, "details": []}


@dataclass
class FakeAPI:
    """In-memory stand-in for the HotLoad, Deployment and Gateway services.

    ``script`` queues canned responses for a path. Other requests use the in-memory state, so
    the runner runs end to end.
    """

    deployments: dict[str, dict[str, Any]] = field(default_factory=dict)
    gateways: dict[str, dict[str, Any]] = field(default_factory=dict)
    hot_loads: dict[str, dict[str, Any]] = field(default_factory=dict)
    replicas: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    script: dict[tuple[str, str], list[Callable[[httpx.Request], httpx.Response]]] = field(
        default_factory=dict
    )
    requests: list[httpx.Request] = field(default_factory=list)
    #: Called before every GET of a hot-load, so tests can advance state per poll.
    on_poll: Callable[[str], None] | None = None

    def queue(self, method: str, path: str, *responses: httpx.Response | Callable) -> None:
        handlers = self.script.setdefault((method, path), [])
        for r in responses:
            handlers.append(r if callable(r) else (lambda _req, r=r: r))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        if self.script.get(key):
            return self.script[key].pop(0)(request)

        path = request.url.path
        if request.method == "GET" and path.startswith("/v1alpha1/inference/deployments/"):
            dep = self.deployments.get(path.rsplit("/", 1)[1])
            if dep is None:
                return httpx.Response(404, json=error_json("not_found", "deployment not found"))
            return httpx.Response(200, json=dep)

        if request.method == "GET" and path.startswith("/v1alpha1/inference/gateways/"):
            gateway = self.gateways.get(path.rsplit("/", 1)[1])
            if gateway is None:
                return httpx.Response(404, json=error_json("not_found", "gateway not found"))
            return httpx.Response(200, json=gateway)

        if request.method == "POST" and path == "/v1alpha1/inference/hotLoads":
            body = json.loads(request.content)
            hl_id = body.get("hotLoadId") or str(uuid.uuid4())
            if hl_id in self.hot_loads:
                return httpx.Response(409, json=error_json("already_exists", "hot-load exists"))
            if any(
                h["status"]["state"] in ("HOT_LOAD_STATE_PENDING", "HOT_LOAD_STATE_IN_PROGRESS")
                for h in self.hot_loads.values()
            ):
                return httpx.Response(
                    412,
                    json=error_json(
                        "failed_precondition",
                        "a hot-load is already in progress on this deployment",
                    ),
                )
            hl = hot_load_json(
                hl_id,
                body["identity"],
                "HOT_LOAD_STATE_PENDING",
                snapshot_type=body["type"],
                incremental=body.get("incrementalSnapshotMetadata"),
            )
            self.hot_loads[hl_id] = hl
            return httpx.Response(200, json={"hotLoad": hl})

        if request.method == "GET" and path == "/v1alpha1/inference/hotLoads":
            items = sorted(self.hot_loads.values(), key=lambda h: h["spec"]["id"], reverse=True)
            dep = request.url.params.get("parentDeploymentId")
            if dep:
                items = [h for h in items if h["spec"]["deploymentId"] == dep]
            return httpx.Response(200, json={"hotLoads": items, "nextPageToken": ""})

        if request.method == "GET" and path.endswith("/replicas"):
            hl_id = path.split("/")[-2]
            return httpx.Response(
                200, json={"replicas": self.replicas.get(hl_id, []), "nextPageToken": ""}
            )

        if request.method == "GET" and path.startswith("/v1alpha1/inference/hotLoads/"):
            hl_id = path.rsplit("/", 1)[1]
            if self.on_poll:
                self.on_poll(hl_id)
            hl = self.hot_loads.get(hl_id)
            if hl is None:
                return httpx.Response(404, json=error_json(5, "hot-load not found"))
            return httpx.Response(200, json={"hotLoad": hl})

        if request.method == "POST" and path.endswith(":cancel"):
            return httpx.Response(
                501, json=error_json("unimplemented", "CancelHotLoad is not implemented yet")
            )
        return httpx.Response(404, json=error_json("not_found", f"no route {path}"))


@pytest.fixture
def api() -> FakeAPI:
    fake = FakeAPI()
    fake.deployments[DEPLOYMENT_ID] = deployment_json()
    return fake


@pytest.fixture
def client(api: FakeAPI) -> InferenceClient:
    with InferenceClient(
        "test-token",
        base_url="https://api.test",
        transport=httpx.MockTransport(api.handle),
        max_retries=3,
    ) as c:
        yield c


@pytest.fixture(autouse=True)
def no_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a token in the developer's shell away from the tests; a test that needs one sets it."""
    for name in TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep retry and polling loops instant."""
    import cwtrain.hotload.runner as runner_mod
    import cwtrain.infra.client as client_mod

    monkeypatch.setattr(client_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(runner_mod.time, "sleep", lambda _s: None)


def write_full_checkpoint(root: Path, shards: int = 2) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    weight_map = {}
    for i in range(shards):
        name = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        (root / name).write_bytes(b"\0" * (1024 * (i + 1)))
        weight_map[f"layer.{i}.weight"] = name
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 3072}, "weight_map": weight_map})
    )
    return root


def write_delta_checkpoint(
    root: Path,
    *,
    version: int = 2,
    encoding: str = "xor",
    checksum: str = "adler32",
    compression: str = "zstd",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model-00001-of-00001.safetensors").write_bytes(b"\1" * 512)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": f"{version:06d}",
                    "base_version": f"{version - 1:06d}",
                    "delta_encoding": encoding,
                    "compression_format": compression,
                    "checksum_format": checksum,
                },
                "weight_map": {"layer.0.weight": "model-00001-of-00001.safetensors"},
            }
        )
    )
    return root


class FakeS3:
    """Records uploads; lists, heads and returns what it has been given.

    ``existing`` seeds objects by size only (no content or metadata, so they cannot be verified);
    :meth:`seed` stores real content, optionally with user metadata.
    """

    def __init__(self, existing: dict[str, int] | None = None) -> None:
        self.objects: dict[tuple[str, str], int] = {}
        self.metadata: dict[tuple[str, str], dict[str, str]] = {}
        self.etags: dict[tuple[str, str], str] = {}
        self.bodies: dict[tuple[str, str], bytes] = {}
        for key, size in (existing or {}).items():
            self.objects[("ckpts", key)] = size
        self.uploads: list[tuple[str, str, str]] = []

    def seed(self, bucket: str, key: str, body: bytes, metadata: dict[str, str] | None = None):
        import hashlib

        self.objects[(bucket, key)] = len(body)
        self.bodies[(bucket, key)] = body
        self.metadata[(bucket, key)] = dict(metadata or {})
        self.etags[(bucket, key)] = hashlib.md5(body).hexdigest()

    def upload_file(self, filename: str, bucket: str, key: str, ExtraArgs=None) -> None:  # noqa: N803
        self.uploads.append((filename, bucket, key))
        self.seed(bucket, key, Path(filename).read_bytes(), (ExtraArgs or {}).get("Metadata"))

    def put_object(self, Bucket: str, Key: str, Body: bytes, **kw: Any) -> None:  # noqa: N803
        self.uploads.append(("<bytes>", Bucket, Key))
        self.seed(Bucket, Key, Body, kw.get("Metadata"))

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        key = (Bucket, Key)
        return {
            "ContentLength": self.objects[key],
            "ETag": f'"{self.etags[key]}"' if key in self.etags else "",
            "Metadata": self.metadata.get(key, {}),
        }

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        import io

        from botocore.exceptions import ClientError

        bodies = getattr(self, "bodies", {})
        if (Bucket, Key) not in bodies:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": Key}}, "GetObject")
        return {"Body": io.BytesIO(bodies[(Bucket, Key)])}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        bucket, prefix = kwargs["Bucket"], kwargs["Prefix"]
        contents = [
            {"Key": k, "Size": s}
            for (b, k), s in self.objects.items()
            if b == bucket and k.startswith(prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}


# ------------------------------------------------------------------ real weights & deltas
# Small but genuine safetensors checkpoints and miles-style zstd XOR deltas, for the mirror tests.


def _st_header(
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]], meta: dict[str, str]
) -> bytes:
    import struct

    header: dict[str, Any] = {"__metadata__": meta}
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    body = json.dumps(header, separators=(",", ":")).encode()
    body += b" " * (-len(body) % 8)
    return struct.pack("<Q", len(body)) + body


def write_safetensors(
    path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]], meta=None
):
    """Write one safetensors shard from ``{name: (dtype, shape, raw_bytes)}``."""
    blob = _st_header(tensors, meta or {"format": "pt"})
    for _dtype, _shape, data in tensors.values():
        blob += data
    path.write_bytes(blob)


def make_weights(seed: int = 0) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """Three tensors of different dtypes; the bytes are what matters to the mirror."""
    import numpy as np

    rng = np.random.default_rng(seed)
    f32 = rng.standard_normal((4, 8), dtype=np.float32)
    bf16 = rng.integers(0, 2**16, size=(3, 5), dtype=np.uint16)  # raw BF16 payload
    i8 = rng.integers(-128, 127, size=(16,), dtype=np.int8)
    return {
        "model.layers.0.weight": ("F32", (4, 8), f32.tobytes()),
        "model.layers.1.weight": ("BF16", (3, 5), bf16.tobytes()),
        "lm_head.bias": ("I8", (16,), i8.tobytes()),
    }


def write_real_checkpoint(root: Path, weights=None) -> Path:
    """A full HF checkpoint: config.json, two shards, index; plus a tokenizer stand-in."""
    weights = weights or make_weights()
    root.mkdir(parents=True, exist_ok=True)
    names = list(weights)
    shards = {
        "model-00001-of-00002.safetensors": {n: weights[n] for n in names[:2]},
        "model-00002-of-00002.safetensors": {n: weights[n] for n in names[2:]},
    }
    weight_map = {}
    for fname, tensors in shards.items():
        write_safetensors(root / fname, tensors)
        weight_map.update({n: fname for n in tensors})
    total = sum(len(d) for _t, _s, d in weights.values())
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )
    (root / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (root / "tokenizer_config.json").write_text("{}")
    return root


def write_single_file_checkpoint(root: Path, weights=None) -> Path:
    """What ``save_pretrained`` writes below its shard size: one model.safetensors, no index."""
    weights = weights or make_weights()
    root.mkdir(parents=True, exist_ok=True)
    write_safetensors(root / "model.safetensors", weights)
    (root / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    (root / "tokenizer_config.json").write_text("{}")
    return root


def write_real_delta(
    root: Path,
    old: dict[str, tuple[str, tuple[int, ...], bytes]],
    new: dict[str, tuple[str, tuple[int, ...], bytes]],
    *,
    version: int,
    base_version: int | None = None,
    corrupt: str | None = None,
) -> Path:
    """A miles ``weight_vNNNNNN`` directory: zstd(level 1) XOR per changed tensor, adler32 of the
    new bytes in the shard ``__metadata__``, index metadata as miles writes it."""
    import zlib

    import numpy as np
    import zstandard

    root.mkdir(parents=True, exist_ok=True)
    delta: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    checksums: dict[str, str] = {}
    for name, (_dtype, _shape, new_bytes) in new.items():
        old_bytes = old[name][2]
        if new_bytes == old_bytes:
            continue
        diff = np.frombuffer(new_bytes, np.uint8) ^ np.frombuffer(old_bytes, np.uint8)
        compressed = zstandard.ZstdCompressor(level=1).compress(diff.tobytes())
        delta[name] = ("U8", (len(compressed),), compressed)
        checksums[name] = f"{zlib.adler32(new_bytes):08x}"
    if corrupt:
        checksums[corrupt] = "deadbeef"
    fname = "model-00000-of-00001.safetensors"
    if delta:
        write_safetensors(root / fname, delta, meta=checksums)
    base = (version - 1) if base_version is None else base_version
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": f"{version:06d}",
                    "base_version": f"{base:06d}",
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                },
                "weight_map": {n: fname for n in delta},
            }
        )
    )
    return root


def perturb(weights, names: tuple[str, ...], seed: int = 1):
    """Return a copy of ``weights`` with the given tensors' bytes changed."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = dict(weights)
    for name in names:
        dtype, shape, data = weights[name]
        buf = np.frombuffer(data, np.uint8).copy()
        idx = rng.choice(len(buf), size=max(1, len(buf) // 4), replace=False)
        buf[idx] ^= rng.integers(1, 256, size=len(idx), dtype=np.uint8)
        out[name] = (dtype, shape, buf.tobytes())
    return out
