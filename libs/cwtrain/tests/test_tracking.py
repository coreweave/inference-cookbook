# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import logging
import math
import sys
from types import SimpleNamespace

import pytest

from cwtrain.infra.models import Deployment
from cwtrain.integrations import tracking
from tests.conftest import FakeRun, deployment_json


def test_without_wandb_nothing_is_logged_or_imported(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    assert tracking.active_run() is None
    assert not tracking.log_metrics({"hotload/x": 1.0}, step_metric="s", step=1, prefixes=("h",))
    assert not tracking.update_config({"a": 1})
    assert not tracking.bind(("coreweave/",), "rollout/step")
    assert "wandb" not in sys.modules


def test_wandb_imported_without_a_run_is_a_no_op(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=None))
    assert not tracking.log_metrics({"hotload/x": 1.0}, step_metric="s", step=1, prefixes=("h",))


def test_a_row_carries_its_step_metric_and_binds_the_prefix_once(wandb_run: FakeRun) -> None:
    for step in (0, 1):
        assert tracking.log_metrics(
            {"hotload/upload_seconds": 2.0},
            step_metric="rollout/step",
            step=step,
            prefixes=("hotload/",),
        )
    assert wandb_run.rows == [
        {"hotload/upload_seconds": 2.0, "rollout/step": 0},
        {"hotload/upload_seconds": 2.0, "rollout/step": 1},
    ]
    assert wandb_run.defined == [("hotload/*", "rollout/step")]


def test_config_updates_merge_into_the_cwtrain_section(wandb_run: FakeRun) -> None:
    tracking.update_config({"version": "1.2.3"})
    tracking.update_config({"rollout": {"concurrency": 16}})
    assert wandb_run.config == {"cwtrain": {"version": "1.2.3", "rollout": {"concurrency": 16}}}


def test_a_failing_run_warns_once_and_never_raises(wandb_run: FakeRun, monkeypatch, caplog) -> None:
    def broken(*_args, **_kwargs):
        raise RuntimeError("backend gone")

    monkeypatch.setattr(wandb_run, "log", broken)
    with caplog.at_level(logging.WARNING, logger="cwtrain.integrations.tracking"):
        for _ in range(3):
            assert not tracking.log_metrics({"a/b": 1.0}, step_metric="s", step=0, prefixes=("a/",))
    assert sum("backend gone" in r.getMessage() for r in caplog.records) == 1


def test_deployment_config_keeps_the_shape_and_leaves_out_where_it_runs() -> None:
    data = deployment_json()["deployment"]
    data["spec"].update(
        {
            "gatewayIds": ["gw-1"],
            "runtime": {"engine": "dynamo-vllm", "version": "0.30.0+crwv.5",
                        "engineConfig": {"max-model-len": "8192"}},
            "resources": {"instanceType": "gd-8xh100ib-i128", "gpuCount": 1},
            "autoscaling": {"min": 2, "max": 2, "concurrency": 256},
        }
    )  # fmt: skip
    out = tracking.deployment_config(Deployment.from_json(data))
    assert out == {
        "id": data["spec"]["id"],
        "name": "qwen3-8b-rl",
        "model": "Qwen/Qwen3-8B",
        "engine": "dynamo-vllm",
        "engine_version": "0.30.0+crwv.5",
        "engine_config": {"max-model-len": "8192"},
        "instance_type": "gd-8xh100ib-i128",
        "gpus_per_replica": 1,
        "replicas_min": 2,
        "replicas_max": 2,
        "autoscaling_concurrency": 256,
    }
    assert "gw-1" not in str(out) and "ckpts" not in str(out) and "models" not in out.values()


def test_a_span_is_two_timeline_keys_on_a_known_lane() -> None:
    assert tracking.span("upload", 10.12345, 12.5) == {
        "timeline/upload/start": 10.123,
        "timeline/upload/end": 12.5,
    }
    with pytest.raises(ValueError, match="unknown timeline lane"):
        tracking.span("uploads", 0, 1)
    keys = tracking.timeline_keys()
    assert keys[:2] == ["timeline/rollout/start", "timeline/rollout/end"]
    assert keys[-1] == "timeline/step" and len(keys) == 2 * len(tracking.LANES) + 1


def test_wandb_run_env_tags_the_run_unless_set() -> None:
    from cwtrain.integrations.launch import wandb_run_env

    assert wandb_run_env("slime", "grpo", {}) == {
        "WANDB_JOB_TYPE": "train",
        "WANDB_TAGS": "slime,grpo",
    }
    assert wandb_run_env("slime", "grpo", {"WANDB_TAGS": "mine", "WANDB_JOB_TYPE": "x"}) == {}


def test_a_timeline_row_carries_every_timeline_key(wandb_run: FakeRun) -> None:
    # W&B's keyed history returns only rows with every key the chart asks for
    row = tracking.timeline_row({"timeline/upload/start": 1.0, "timeline/upload/end": 2.0, "a": 3})
    assert set(tracking.timeline_keys()) <= set(row) and row["a"] == 3
    assert row["timeline/upload/end"] == 2.0 and math.isnan(row["timeline/rollout/start"])
    assert tracking.timeline_row({"a": 1}) == {"a": 1}  # no timeline key: left alone
    tracking.log_metrics({"timeline/upload/start": 1.0}, step_metric="s", step=0, prefixes=("t",))
    assert set(tracking.timeline_keys()) <= set(wandb_run.rows[-1])
