# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cwtrain.errors import NotFoundError
from cwtrain.infra import Gateway


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"gym_recipe_{name}", Path(__file__).resolve().parents[1] / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch, resources = load("launch"), load("resources")


def test_key_file_and_normalized_endpoint(tmp_path):
    key = tmp_path / "key"
    key.write_text("test-key\n")
    env = launch.start_environment(
        {
            "CWTRAIN_ROLLOUT_BASE_URL": " https://example.invalid/v1/ ",
            "CWTRAIN_ROLLOUT_MODEL": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            "CWTRAIN_ROLLOUT_API_KEY_FILE": str(key),
        }
    )
    assert env["CWTRAIN_ROLLOUT_API_KEY"] == "test-key"
    assert env["CWTRAIN_ROLLOUT_BASE_URL"] == "https://example.invalid/v1"
    cmd = launch.command(argparse.Namespace(action="start"), env)
    assert "test-key" not in " ".join(cmd)
    key.unlink()
    env.pop("CWTRAIN_ROLLOUT_API_KEY")
    with pytest.raises(ValueError, match="nonempty key"):
        launch.start_environment(env)


def test_collection_is_bounded_and_uses_shared_concurrency(tmp_path):
    cmd = launch.command(
        argparse.Namespace(action="collect", output=tmp_path / "results.jsonl"),
        {
            "CWTRAIN_ROLLOUT_BASE_URL": "https://example.invalid/v1",
            "CWTRAIN_ROLLOUT_MODEL": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            "CWTRAIN_ROLLOUT_CONCURRENCY": "2",
        },
    )
    assert cmd[cmd.index("--limit") + 1] == "5"
    assert cmd[cmd.index("--concurrency") + 1] == "2"
    assert "--no-serve" in cmd


def test_journal_survives_ambiguous_creation_and_cannot_be_overwritten(tmp_path):
    state_path = tmp_path / "state.json"

    def fail(spec):
        state = json.loads(state_path.read_text())
        assert state["gateway_id"] == spec["id"]
        raise RuntimeError("connection lost after POST")

    client = SimpleNamespace(
        get_gateway_parameters=lambda: {"zones": ["US-WEST-04A"]},
        list_gateways=lambda: [Gateway.from_json({"spec": {"organizationId": "org1"}})],
        create_gateway=fail,
    )
    with pytest.raises(RuntimeError, match="connection lost"):
        resources.provision(
            client,
            state_path,
            {"model": {"name": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"}},
            "org1",
            "us-west-04a",
        )
    before = state_path.read_text()
    with pytest.raises(FileExistsError):
        resources.provision(
            client,
            state_path,
            {"model": {"name": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"}},
            "org1",
            "US-WEST-04A",
        )
    assert state_path.read_text() == before


def test_cleanup_order_and_repeat(monkeypatch):
    events, present = [], {"deployment", "gateway"}

    def get(kind):
        if kind not in present:
            raise NotFoundError("not_found", "gone")
        return SimpleNamespace(name="run1")

    def delete(kind):
        events.append(kind)
        present.remove(kind)

    client = SimpleNamespace(
        get_deployment=lambda _: get("deployment"),
        get_gateway=lambda _: get("gateway"),
        delete_deployment=lambda _: delete("deployment"),
        delete_gateway=lambda _: delete("gateway"),
    )
    state = {"name": "run1", "deployment_id": "d", "gateway_id": "g"}
    resources.cleanup(client, state)
    resources.cleanup(client, state)
    assert events == ["deployment", "gateway"]


def test_cleanup_refuses_name_mismatch():
    client = SimpleNamespace(get_deployment=lambda _: SimpleNamespace(name="unrelated"))
    with pytest.raises(ValueError, match="refusing"):
        resources.cleanup(client, {"name": "run1", "deployment_id": "d"})
