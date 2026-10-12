# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

from datetime import UTC

from cwtrain.infra import (
    Deployment,
    HotLoad,
    HotLoadState,
    IncrementalSnapshotMetadata,
    PromptCachePolicy,
    ReplicaState,
    ResourceStatus,
    SnapshotType,
)
from cwtrain.infra.models import parse_time
from tests.conftest import deployment_json, hot_load_json


def test_enum_parses_wire_name_int_and_bare_name() -> None:
    assert HotLoadState.parse("HOT_LOAD_STATE_COMPLETED") is HotLoadState.COMPLETED
    assert HotLoadState.parse(3) is HotLoadState.COMPLETED
    assert HotLoadState.parse("completed") is HotLoadState.COMPLETED
    assert HotLoadState.parse(None) is HotLoadState.UNSPECIFIED
    assert HotLoadState.parse("SOMETHING_NEW") is HotLoadState.UNSPECIFIED
    # ReplicaState values have no prefix on the wire.
    assert ReplicaState.parse("PENDING_UPDATE") is ReplicaState.PENDING_UPDATE
    assert ReplicaState.parse(3) is ReplicaState.READY


def test_enum_wire_names() -> None:
    assert SnapshotType.FULL.wire == "SNAPSHOT_TYPE_FULL"
    assert PromptCachePolicy.RESET_ALL.wire == "PROMPT_CACHE_POLICY_RESET_ALL"
    assert HotLoadState.FAILED.terminal and not HotLoadState.IN_PROGRESS.terminal


def test_parse_time_handles_nanoseconds_and_z() -> None:
    dt = parse_time("2026-09-22T12:00:00.123456789Z")
    assert dt is not None
    assert dt.tzinfo == UTC
    assert dt.microsecond == 123456
    assert parse_time("") is None
    assert parse_time("2026-09-22T12:00:00Z") is not None


def test_hot_load_from_json() -> None:
    hl = HotLoad.from_json(
        hot_load_json(
            "aaaa",
            "step_2",
            "HOT_LOAD_STATE_IN_PROGRESS",
            snapshot_type="SNAPSHOT_TYPE_INCREMENTAL",
            incremental={
                "previousSnapshotIdentity": "step_1",
                "compressionFormat": "COMPRESSION_FORMAT_ZSTD",
                "checksumFormat": "CHECKSUM_FORMAT_ADLER32",
            },
            chain=[
                {"identity": "step_1", "type": "SNAPSHOT_TYPE_FULL", "hotLoadId": "0000"},
                {"identity": "step_2", "type": 2, "hotLoadId": "aaaa", "compressionFormat": 1},
            ],
        )
    )
    assert hl.id == "aaaa" and hl.identity == "step_2"
    assert hl.state is HotLoadState.IN_PROGRESS and not hl.terminal
    assert hl.spec.type is SnapshotType.INCREMENTAL
    assert hl.spec.incremental_snapshot_metadata == IncrementalSnapshotMetadata("step_1")
    assert [r.identity for r in hl.spec.snapshot_chain] == ["step_1", "step_2"]
    assert hl.spec.snapshot_chain[1].type is SnapshotType.INCREMENTAL


def test_deployment_from_json() -> None:
    dep = Deployment.from_json(deployment_json()["deployment"])
    assert dep.ready and dep.status is ResourceStatus.READY
    assert dep.hot_load_enabled
    assert dep.hot_load is not None
    assert dep.hot_load.bucket == "ckpts" and dep.hot_load.path_prefix == "runs/exp1"
    assert dep.engine == "dynamo-vllm"
    assert dep.model_bucket == "models"

    plain = Deployment.from_json(deployment_json(hot_load=False, ready=False)["deployment"])
    assert plain.hot_load is None and not plain.ready
    assert plain.gateway_ids == ()

    wire = deployment_json()["deployment"]
    wire["spec"]["gatewayIds"] = ["gw-a", "gw-b"]
    assert Deployment.from_json(wire).gateway_ids == ("gw-a", "gw-b")


def test_condition_status_uses_status_prefix_on_the_wire() -> None:
    # The API sends Condition.status as "STATUS_TRUE", not "CONDITION_STATUS_TRUE".
    dep = Deployment.from_json(
        {
            "spec": {"id": "d", "name": "n"},
            "status": {
                "status": "STATUS_READY",
                "conditions": [
                    {
                        "type": "Ready",
                        "status": "STATUS_TRUE",
                        "lastUpdateTime": "2026-09-22T19:54:37Z",
                        "reason": "Ready",
                        "message": "Ready replicas: 4/4. All resources are ready",
                        "zone": "EU-NORTH-04A",
                    }
                ],
            },
        }
    )
    assert dep.conditions[0].status.name == "TRUE"
    assert dep.conditions[0].zone == "EU-NORTH-04A"


def test_parse_time_tolerates_malformed_timestamps() -> None:
    from cwtrain.infra.models import parse_time

    assert parse_time("not a timestamp") is None
    assert parse_time("2026-09-24T04:30:40.499989Z") is not None
