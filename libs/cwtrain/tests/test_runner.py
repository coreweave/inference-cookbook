# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from cwtrain.errors import CWTrainError, HotLoadInProgressError
from cwtrain.hotload import (
    HotLoader,
    HotLoadFailedError,
    HotLoadState,
    HotLoadTimeoutError,
    HotLoadVerificationError,
    ProgressEvent,
    PromptCachePolicy,
    SnapshotError,
    SnapshotType,
    inspect_snapshot,
)
from cwtrain.infra import InferenceClient
from tests.conftest import (
    DEPLOYMENT_ID,
    FakeAPI,
    FakeS3,
    deployment_json,
    hot_load_json,
    replica_json,
    write_delta_checkpoint,
    write_full_checkpoint,
)


def advance(api: FakeAPI, plan: dict[str, list[tuple[str, list[dict]]]]) -> None:
    """Make each poll of a hot-load step through (state, replicas) tuples."""
    cursors: dict[str, int] = {}

    def on_poll(hl_id: str) -> None:
        steps = plan.get(hl_id)
        if not steps:
            return
        i = min(cursors.get(hl_id, 0), len(steps) - 1)
        cursors[hl_id] = i + 1
        state, replicas = steps[i]
        api.hot_loads[hl_id]["status"]["state"] = state
        api.replicas[hl_id] = replicas

    api.on_poll = on_poll


@pytest.fixture
def loader(client: InferenceClient) -> HotLoader:
    return HotLoader(client, DEPLOYMENT_ID, s3=FakeS3(), poll_interval=0.01, timeout=60)


def test_publish_full_checkpoint_end_to_end(
    loader: HotLoader, api: FakeAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = write_full_checkpoint(tmp_path / "step_000100")
    created: list[str] = []

    original = api.handle

    def handle(req: httpx.Request) -> httpx.Response:
        resp = original(req)
        if req.method == "POST" and req.url.path.endswith("/hotLoads") and resp.status_code == 200:
            hl_id = resp.json()["hotLoad"]["spec"]["id"]
            created.append(hl_id)
            advance(
                api,
                {
                    hl_id: [
                        ("HOT_LOAD_STATE_PENDING", []),
                        (
                            "HOT_LOAD_STATE_IN_PROGRESS",
                            [
                                replica_json("r1", "UPDATING", "initial"),
                                replica_json("r2", "PENDING_UPDATE", "initial"),
                            ],
                        ),
                        (
                            "HOT_LOAD_STATE_COMPLETED",
                            [
                                replica_json("r1", "READY", "step_000100"),
                                replica_json("r2", "READY", "step_000100"),
                            ],
                        ),
                    ]
                },
            )
        return resp

    monkeypatch.setattr(loader.client._http, "_transport", httpx.MockTransport(handle))

    events: list[ProgressEvent] = []
    hl = loader.publish(ckpt, "step_000100", on_progress=events.append)

    assert hl.completed and hl.identity == "step_000100"
    assert len(created) == 1
    # Uploaded to the deployment's configured bucket/prefix.
    s3: FakeS3 = loader._s3  # type: ignore[assignment]
    keys = sorted(k for _f, b, k in s3.uploads if b == "ckpts")
    assert keys[0].startswith("runs/exp1/step_000100/")
    assert any(k.endswith("model.safetensors.index.json") for k in keys)
    # Progress was reported and ended with every replica ready.
    assert events[-1].ready == 2 and events[-1].total == 2
    assert "completed" in events[-1].summary()
    body = json.loads(next(r for r in api.requests if r.method == "POST").content)
    assert body["type"] == "SNAPSHOT_TYPE_FULL" and "incrementalSnapshotMetadata" not in body


def test_publish_delta_sends_incremental_metadata(
    loader: HotLoader, api: FakeAPI, tmp_path: Path
) -> None:
    ckpt = write_delta_checkpoint(tmp_path / "weight_v000002", version=2)
    hl = loader.publish(ckpt, "weight_v000002", wait=False)
    assert hl.state is HotLoadState.PENDING
    body = json.loads(next(r for r in api.requests if r.method == "POST").content)
    assert body["type"] == "SNAPSHOT_TYPE_INCREMENTAL"
    assert body["incrementalSnapshotMetadata"]["previousSnapshotIdentity"] == "weight_v000001"


def test_publish_prompt_cache_policy(loader: HotLoader, api: FakeAPI, tmp_path: Path) -> None:
    ckpt = write_full_checkpoint(tmp_path / "s")
    loader.publish(ckpt, "s", wait=False, prompt_cache_policy=PromptCachePolicy.RESET_ALL)
    body = json.loads(next(r for r in api.requests if r.method == "POST").content)
    assert body["promptCachePolicy"] == "PROMPT_CACHE_POLICY_RESET_ALL"


def test_trigger_an_uploaded_delta(loader: HotLoader, api: FakeAPI) -> None:
    hl = loader.trigger("run-v000002", previous_identity="run-v000001")
    assert hl.state is HotLoadState.PENDING
    body = json.loads(next(r for r in api.requests if r.method == "POST").content)
    assert body["type"] == "SNAPSHOT_TYPE_INCREMENTAL"
    assert body["incrementalSnapshotMetadata"] == {
        "previousSnapshotIdentity": "run-v000001",
        "compressionFormat": "COMPRESSION_FORMAT_ZSTD",
        "checksumFormat": "CHECKSUM_FORMAT_ADLER32",
    }


@pytest.mark.parametrize("case", ["with info", "full type", "incremental type", "bad identity"])
def test_trigger_rejects_inconsistent_delta_arguments(
    loader: HotLoader, api: FakeAPI, tmp_path: Path, case: str
) -> None:
    kwargs = {
        "with info": {
            "info": inspect_snapshot(write_full_checkpoint(tmp_path / "s")),
            "previous_identity": "run-v1",
        },
        "full type": {"previous_identity": "run-v1", "snapshot_type": SnapshotType.FULL},
        "incremental type": {"snapshot_type": SnapshotType.INCREMENTAL},
        "bad identity": {"previous_identity": "run/v1"},
    }[case]
    with pytest.raises(SnapshotError):  # a CWTrainError, as the CLI and the hooks expect
        loader.trigger("run-v2", **kwargs)
    assert not [r for r in api.requests if r.method == "POST"]


def test_wait_raises_on_failure_with_replica_reasons(loader: HotLoader, api: FakeAPI) -> None:
    api.hot_loads["hl1"] = hot_load_json("hl1", "s1", "HOT_LOAD_STATE_IN_PROGRESS")
    advance(
        api,
        {
            "hl1": [
                ("HOT_LOAD_STATE_IN_PROGRESS", [replica_json("r1", "UPDATING")]),
                (
                    "HOT_LOAD_STATE_FAILED",
                    [
                        replica_json("r1", "ERROR", "initial", "update: weights checksum mismatch"),
                        replica_json("r2", "READY", "s1"),
                    ],
                ),
            ]
        },
    )
    with pytest.raises(HotLoadFailedError) as exc:
        loader.wait("hl1")
    assert "r1: update: weights checksum mismatch" in str(exc.value)
    assert exc.value.hot_load.state is HotLoadState.FAILED
    assert len(exc.value.replicas) == 2

    # Non-raising mode returns the terminal object instead.
    api.on_poll = None
    assert loader.wait("hl1", raise_on_failure=False).state is HotLoadState.FAILED


def test_wait_times_out(loader: HotLoader, api: FakeAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    api.hot_loads["hl1"] = hot_load_json("hl1", "s1", "HOT_LOAD_STATE_IN_PROGRESS")
    api.replicas["hl1"] = [replica_json("r1", "UPDATING")]
    import cwtrain.hotload.runner as runner_mod

    clock = iter([0.0, 0.0, 5.0, 11.0, 20.0, 30.0])
    monkeypatch.setattr(runner_mod.time, "monotonic", lambda: next(clock))
    with pytest.raises(HotLoadTimeoutError) as exc:
        loader.wait("hl1", timeout=10)
    assert "0/1 replicas ready" in str(exc.value)


def test_trigger_waits_out_active_hot_load(loader: HotLoader, api: FakeAPI) -> None:
    api.hot_loads["active"] = hot_load_json("active", "s1", "HOT_LOAD_STATE_IN_PROGRESS")
    advance(api, {"active": [("HOT_LOAD_STATE_COMPLETED", [replica_json("r1", "READY", "s1")])]})
    hl = loader.trigger("s2")
    assert hl.identity == "s2" and hl.state is HotLoadState.PENDING
    posts = [r for r in api.requests if r.method == "POST"]
    assert len(posts) == 2  # first rejected, second accepted
    # Same client-side ID on both attempts keeps the retry idempotent.
    assert {json.loads(p.content)["hotLoadId"] for p in posts}.__len__() == 1


def test_trigger_can_surface_in_progress(loader: HotLoader, api: FakeAPI) -> None:
    api.hot_loads["active"] = hot_load_json("active", "s1", "HOT_LOAD_STATE_IN_PROGRESS")
    with pytest.raises(HotLoadInProgressError):
        loader.trigger("s2", wait_for_active=False)


def test_deployment_without_hot_load_is_rejected(client: InferenceClient, api: FakeAPI) -> None:
    api.deployments[DEPLOYMENT_ID] = deployment_json(hot_load=False)
    loader = HotLoader(client, DEPLOYMENT_ID, s3=FakeS3())
    with pytest.raises(CWTrainError, match="without a hot_load section"):
        _ = loader.deployment


def test_wait_for_deployment_ready(client: InferenceClient, api: FakeAPI) -> None:
    api.deployments[DEPLOYMENT_ID] = deployment_json(ready=False)
    loader = HotLoader(client, DEPLOYMENT_ID, s3=FakeS3(), poll_interval=0.01)
    calls = {"n": 0}
    original = api.handle

    def handle(req: httpx.Request) -> httpx.Response:
        if req.url.path.startswith("/v1alpha1/inference/deployments/"):
            calls["n"] += 1
            if calls["n"] >= 3:
                api.deployments[DEPLOYMENT_ID] = deployment_json(ready=True)
        return original(req)

    client._http._transport = httpx.MockTransport(handle)
    assert loader.wait_for_deployment_ready(timeout=30).ready
    assert calls["n"] == 3


def test_latest_completed_and_current_identity(loader: HotLoader, api: FakeAPI) -> None:
    assert loader.current_identity() == "initial"
    api.hot_loads["a"] = hot_load_json("a", "s1", "HOT_LOAD_STATE_COMPLETED")
    api.hot_loads["b"] = hot_load_json("b", "s2", "HOT_LOAD_STATE_FAILED")
    latest = loader.latest_completed()
    assert latest is not None and latest.identity == "s1"
    assert loader.current_identity() == "s1"
    assert loader.active_hot_load() is None
    api.hot_loads["c"] = hot_load_json("c", "s3", "HOT_LOAD_STATE_PENDING")
    active = loader.active_hot_load()
    assert active is not None and active.id == "c"


def test_rollback_reuses_previous_snapshot_type(loader: HotLoader, api: FakeAPI) -> None:
    api.hot_loads["a"] = hot_load_json(
        "a",
        "s2",
        "HOT_LOAD_STATE_COMPLETED",
        snapshot_type="SNAPSHOT_TYPE_INCREMENTAL",
        incremental={
            "previousSnapshotIdentity": "s1",
            "compressionFormat": "COMPRESSION_FORMAT_ZSTD",
            "checksumFormat": "CHECKSUM_FORMAT_ADLER32",
        },
    )
    original = api.handle

    def handle(req: httpx.Request) -> httpx.Response:
        resp = original(req)
        if req.method == "POST" and resp.status_code == 200:
            hl_id = resp.json()["hotLoad"]["spec"]["id"]
            advance(
                api, {hl_id: [("HOT_LOAD_STATE_COMPLETED", [replica_json("r", "READY", "s2")])]}
            )
        return resp

    loader.client._http._transport = httpx.MockTransport(handle)
    hl = loader.rollback("s2")
    assert hl.completed
    body = json.loads(next(r for r in api.requests if r.method == "POST").content)
    assert body["type"] == "SNAPSHOT_TYPE_INCREMENTAL"
    assert body["incrementalSnapshotMetadata"]["previousSnapshotIdentity"] == "s1"

    with pytest.raises(CWTrainError, match="no completed hot-load"):
        loader.rollback("never")


def test_verify_served_requires_every_replica_on_identity(loader: HotLoader, api: FakeAPI) -> None:
    api.hot_loads["hl-9"] = hot_load_json("hl-9", "run-v000009", "HOT_LOAD_STATE_COMPLETED")
    api.replicas["hl-9"] = [
        replica_json("w-0", "READY", "run-v000009"),
        replica_json("w-1", "READY", "run-v000009"),
    ]
    replicas = loader.verify_served("hl-9", "run-v000009")
    assert [r.name for r in replicas] == ["w-0", "w-1"]

    api.replicas["hl-9"][1] = replica_json("w-1", "READY", "run-v000008")
    with pytest.raises(HotLoadVerificationError, match="w-1: READY on 'run-v000008'") as exc:
        loader.verify_served("hl-9", "run-v000009")
    assert exc.value.problems == ["w-1: READY on 'run-v000008'"]

    api.replicas["hl-9"][1] = replica_json("w-1", "ERROR", "", error="chain root unresolved")
    with pytest.raises(HotLoadVerificationError, match="chain root unresolved"):
        loader.verify_served("hl-9", "run-v000009")

    api.replicas["hl-9"] = []
    with pytest.raises(HotLoadVerificationError, match="no replicas"):
        loader.verify_served("hl-9", "run-v000009")

    api.hot_loads["hl-9"]["status"]["state"] = "HOT_LOAD_STATE_FAILED"
    with pytest.raises(HotLoadFailedError):
        loader.verify_served("hl-9", "run-v000009")


def test_upload_manifest_lands_next_to_the_snapshots(loader: HotLoader) -> None:
    uri = loader.upload_manifest("run-v000003", {"identity": "run-v000003", "tensors": {"a": "1"}})
    assert uri == "s3://ckpts/runs/exp1/_manifests/run-v000003.json"
    s3 = loader.s3
    body = json.loads(s3.bodies[("ckpts", "runs/exp1/_manifests/run-v000003.json")])
    assert body["tensors"] == {"a": "1"}
    assert loader.manifest_key("x") == "runs/exp1/_manifests/x.json"


def test_upload_reuses_one_s3_client(client: InferenceClient, monkeypatch, tmp_path: Path) -> None:
    import cwtrain.hotload.runner as runner_mod

    made: list[FakeS3] = []

    def fake_client() -> FakeS3:
        made.append(FakeS3())
        return made[-1]

    monkeypatch.setattr(runner_mod, "s3_client", fake_client)
    loader = HotLoader(client, DEPLOYMENT_ID)
    loader.upload(write_full_checkpoint(tmp_path / "a"), "a")
    loader.upload(write_full_checkpoint(tmp_path / "b"), "b")
    loader.upload_manifest("b", {"tensors": {}})
    assert len(made) == 1 and made[0].uploads  # one client, and it did the uploads


def test_rollback_waits_out_an_active_hot_load(loader: HotLoader, api: FakeAPI) -> None:
    api.hot_loads["done"] = hot_load_json("done", "s1", "HOT_LOAD_STATE_COMPLETED")
    api.hot_loads["active"] = hot_load_json("active", "s2", "HOT_LOAD_STATE_IN_PROGRESS")
    advance(api, {"active": [("HOT_LOAD_STATE_COMPLETED", [replica_json("r1", "READY", "s2")])]})
    hl = loader.rollback("s1", wait=False)
    assert hl.identity == "s1"
    posts = [r for r in api.requests if r.method == "POST"]
    assert len(posts) == 2  # rejected while "active" ran, then accepted
    with pytest.raises(TypeError):
        loader.rollback("s1", snapshot_type=None)  # type: ignore[call-arg]


def test_trigger_gives_up_waiting_for_active_after_its_timeout(
    client: InferenceClient, api: FakeAPI
) -> None:
    api.hot_loads["active"] = hot_load_json("active", "s1", "HOT_LOAD_STATE_IN_PROGRESS")
    loader = HotLoader(client, DEPLOYMENT_ID, s3=FakeS3(), poll_interval=0.01, timeout=0.2)
    with pytest.raises((HotLoadTimeoutError, HotLoadInProgressError)):
        loader.trigger("s2")


def test_zero_poll_interval_is_honoured(client: InferenceClient, api: FakeAPI) -> None:
    api.hot_loads["hl1"] = hot_load_json("hl1", "s1", "HOT_LOAD_STATE_COMPLETED")
    loader = HotLoader(client, DEPLOYMENT_ID, s3=FakeS3(), poll_interval=30)
    assert loader.wait("hl1", poll_interval=0).completed


def test_read_manifest_round_trips_and_is_none_when_missing(loader: HotLoader) -> None:
    assert loader.read_manifest("run-v000009") is None
    manifest = {"identity": "run-v000009", "tensors": {"a": "1"}}
    loader.upload_manifest("run-v000009", manifest)
    assert loader.read_manifest("run-v000009") == manifest


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    from cwtrain.hotload import runner

    fake = FakeClock()
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=fake.monotonic, sleep=fake.sleep))
    return fake


def wait_lines(caplog: pytest.LogCaptureFixture, hot_load_id: str) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "cwtrain.hotload.runner"
        and r.getMessage().startswith(f"hot-load {hot_load_id} ")
    ]


UPDATING = [
    replica_json("r1", "UPDATING", "initial"),
    replica_json("r2", "PENDING_UPDATE", "initial"),
]
READY = [replica_json("r1", "READY", "s1"), replica_json("r2", "READY", "s1")]


def test_wait_logs_what_changed_not_every_poll(
    client: InferenceClient, api: FakeAPI, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="cwtrain.hotload.runner")
    api.hot_loads["hl1"] = hot_load_json("hl1", "s1", "HOT_LOAD_STATE_PENDING")
    advance(
        api,
        {
            "hl1": [("HOT_LOAD_STATE_PENDING", UPDATING)] * 10
            + [("HOT_LOAD_STATE_COMPLETED", READY)]
        },
    )
    loader = HotLoader(client, DEPLOYMENT_ID, s3=FakeS3(), poll_interval=5, timeout=3600)
    events: list[ProgressEvent] = []
    assert loader.wait("hl1", on_progress=events.append).completed

    assert len(events) == 11  # on_progress still sees every poll
    assert wait_lines(caplog, "hl1") == [
        "hot-load hl1 (s1): pending after 0s (pending_update=1, updating=1)",
        "hot-load hl1 (s1): completed after 50s (ready=2)",
    ]


def test_wait_logs_a_heartbeat_while_nothing_changes(
    client: InferenceClient, api: FakeAPI, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="cwtrain.hotload.runner")
    api.hot_loads["hl1"] = hot_load_json("hl1", "s1", "HOT_LOAD_STATE_PENDING")
    advance(
        api,
        {
            "hl1": [("HOT_LOAD_STATE_PENDING", UPDATING)] * 30
            + [("HOT_LOAD_STATE_COMPLETED", READY)]
        },
    )
    loader = HotLoader(client, DEPLOYMENT_ID, s3=FakeS3(), poll_interval=5, timeout=3600)
    assert loader.wait("hl1").completed

    lines = wait_lines(caplog, "hl1")
    assert [line.split(": ", 1)[1].split(" (")[0] for line in lines] == [
        "pending after 0s",
        "pending after 60s",
        "pending after 120s",
        "completed after 150s",
    ]


def test_wait_logs_a_replica_that_starts_serving_a_new_version(
    client: InferenceClient, api: FakeAPI, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="cwtrain.hotload.runner")
    api.hot_loads["hl1"] = hot_load_json("hl1", "s1", "HOT_LOAD_STATE_PENDING")
    first_done = [replica_json("r1", "READY", "s1"), replica_json("r2", "UPDATING", "s0")]
    second_serving = [replica_json("r1", "READY", "s1"), replica_json("r2", "UPDATING", "s1")]
    advance(
        api,
        {
            "hl1": [("HOT_LOAD_STATE_PENDING", first_done)] * 3
            + [("HOT_LOAD_STATE_PENDING", second_serving)] * 3
            + [("HOT_LOAD_STATE_COMPLETED", READY)]
        },
    )
    loader = HotLoader(client, DEPLOYMENT_ID, s3=FakeS3(), poll_interval=5, timeout=3600)
    assert loader.wait("hl1").completed
    assert len(wait_lines(caplog, "hl1")) == 3


@pytest.mark.parametrize("index", [[], {"weight_map": {"w": "missing.safetensors"}}])
def test_publish_rejects_invalid_index_before_writes(loader, api, tmp_path, index):
    root = write_full_checkpoint(tmp_path / "s")
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(SnapshotError):
        loader.publish(root, "step_1", wait=False)
    assert not loader.s3.objects
    assert not any(request.method != "GET" for request in api.requests)
