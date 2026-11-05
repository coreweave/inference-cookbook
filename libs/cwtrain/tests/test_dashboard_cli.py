# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Dashboard entry points remain credential-independent and never sample timeline rows."""

import builtins
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from cwtrain import cli
from cwtrain.dashboard import cli as dashboard_cli

SETUP = ["dashboard", "setup", "--entity", "team", "--project", "project", "--framework", "miles"]


def test_offline_dry_run_needs_neither_optional_sdk_nor_inference_credentials(
    monkeypatch, capsys, tmp_path
):
    real_import = builtins.__import__

    def without_wandb(name, *args, **kwargs):
        if name == "wandb" or name.startswith("wandb_workspaces"):
            pytest.fail(f"offline command imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_wandb)
    monkeypatch.setattr(cli, "InferenceClient", Mock(side_effect=AssertionError("inference API")))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    state = tmp_path / "must-not-exist.json"
    assert cli.main([*SETUP, "--dry-run", "--state-file", str(state)]) == 0
    output = capsys.readouterr()
    plan = json.loads(output.out)
    assert output.err == ""
    assert plan["name"] == "Post-training — Miles"
    assert plan["entity"] == "team" and plan["project"] == "project"
    assert plan["panels"] and plan["omitted_metrics"] == []
    assert not state.exists() and not list(tmp_path.iterdir())


@pytest.fixture
def wandb_api(monkeypatch):
    try:
        from wandb_workspaces.workspaces import errors
    except ImportError:
        errors = ModuleType("wandb_workspaces.workspaces.errors")
        errors.SpecVersionError = type("SpecVersionError", (Exception,), {})
        errors.UnsupportedViewError = type("UnsupportedViewError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "wandb_workspaces.workspaces.errors", errors)
    module = ModuleType("wandb")
    module.Error = type("WandbError", (Exception,), {})
    module.workspace_errors = errors
    api = Mock()
    api.settings = {"base_url": "https://api.wandb.example"}
    api.viewer = SimpleNamespace(username="researcher")
    module.Api = Mock(return_value=api)
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setattr(cli, "InferenceClient", Mock(side_effect=AssertionError("inference API")))
    return module, api


def test_setup_discovers_at_most_ten_matching_runs_and_forwards_only_available_keys(
    wandb_api, monkeypatch, capsys, tmp_path
):
    module, api = wandb_api
    sources = [
        SimpleNamespace(
            id=f"example-{index}",
            url=f"https://wandb.example/proxy/team/project/runs/example-{index}",
            tags=["grpo", "miles"],
            summary={"train/loss": 1, "train/pg_clipfrac": 0.1},
            history=Mock(return_value=[{"system.gpu.0.gpu/l:writer": 90, "_timestamp": 1}]),
        )
        for index in range(10)
    ]

    def bounded_runs():
        yield SimpleNamespace(tags=["slime"])
        yield from sources
        pytest.fail("setup consumed more than ten runs")

    api.runs.return_value = bounded_runs()
    workspace = object()
    build = Mock(return_value=workspace)
    save = Mock(
        return_value={
            "url": "https://wandb.example/team/project?nw=view123456",
            "action": "created",
        }
    )
    monkeypatch.setattr(dashboard_cli, "build_workspace", build)
    monkeypatch.setattr(dashboard_cli, "save_managed", save)
    state = tmp_path / "dashboards.json"
    assert cli.main([*SETUP, "--state-file", str(state)]) == 0
    module.Api.assert_called_once_with(timeout=30)
    api.runs.assert_called_once_with(
        "team/project",
        filters={"jobType": "train"},
        order="-created_at",
        per_page=50,
    )
    for source in sources:
        source.history.assert_called_once_with(stream="system", samples=20, pandas=False)
    entity, project, plan = build.call_args.args
    assert (entity, project) == ("team", "project")
    assert {metric for panel in plan["panels"] for metric in panel["metrics"]} == {
        "train/loss",
        "train/pg_clipfrac",
    }
    assert any(panel.get("metric_regex") for panel in plan["panels"])
    assert plan["included_runs"] == [source.id for source in sources]
    save.assert_called_once_with(
        workspace,
        plan,
        path=state,
        api_host="https://api.wandb.example",
        username="researcher",
        app_url=sources[0].url,
    )
    result = json.loads(capsys.readouterr().out)
    assert result["discovered_runs"] == [source.id for source in sources]
    assert result["url"] == "https://wandb.example/proxy/team/project?nw=view123456"
    assert "train/tis_clipfrac" in result["omitted_metrics"]


def test_discovery_bounds_nonmatching_candidates_and_stops_before_creating_a_view(
    wandb_api, monkeypatch, capsys
):
    _, api = wandb_api

    def candidates():
        for _ in range(200):
            yield SimpleNamespace(tags=["slime"])
        pytest.fail("setup inspected more than 200 candidate runs")

    api.runs.return_value = candidates()
    save = Mock()
    monkeypatch.setattr(dashboard_cli, "save_managed", save)
    assert cli.main(SETUP) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "no runs tagged 'miles' among the latest 200 train runs" in output.err
    save.assert_not_called()


def test_missing_dashboard_extra_has_actionable_error(monkeypatch, capsys):
    real_import = builtins.__import__

    def without_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_wandb)
    assert cli.main(SETUP) == 1
    output = capsys.readouterr()
    assert "pip install 'cwtrain[dashboard]'" in output.err and output.out == ""


def test_wandb_failure_does_not_print_request_details(wandb_api, capsys):
    module, api = wandb_api
    api.runs.side_effect = module.Error("sensitive request authorization details")
    assert cli.main(SETUP) == 1
    output = capsys.readouterr()
    assert "dashboard operation failed (WandbError)" in output.err
    assert "WANDB_BASE_URL" in output.err
    assert "sensitive" not in output.err and "Traceback" not in output.err
    assert output.out == ""


@pytest.mark.parametrize("name", ["SpecVersionError", "UnsupportedViewError"])
def test_workspace_sdk_failures_use_concise_cli_error(wandb_api, capsys, name):
    module, api = wandb_api
    api.runs.side_effect = getattr(module.workspace_errors, name)("private workspace payload")
    assert cli.main(SETUP) == 1
    output = capsys.readouterr()
    assert f"dashboard operation failed ({name})" in output.err
    assert "--verbose" in output.err
    assert "private" not in output.err and "Traceback" not in output.err
    assert output.out == ""


def test_timeline_exports_unsampled_sparse_history_to_local_file(
    wandb_api, monkeypatch, capsys, tmp_path
):
    from cwtrain.dashboard import timeline

    _, api = wandb_api
    rows = iter(
        [
            {"rollout/step": 1, "timeline/rollout/start": 1},
            {"rollout/step": 1, "timeline/reward/start": 2},
        ]
    )
    source = SimpleNamespace(name="Example training", scan_history=Mock(return_value=rows))
    api.run.return_value = source
    export = Mock(return_value="<!doctype html><title>Local timeline</title>")
    monkeypatch.setattr(timeline, "export_html", export)
    output_path = tmp_path / "nested" / "timeline.html"
    assert (
        cli.main(
            [
                "dashboard",
                "timeline",
                "--run",
                "team/project/example",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    api.run.assert_called_once_with("team/project/example")
    source.scan_history.assert_called_once_with(page_size=1000)
    export.assert_called_once_with(rows, title="Post-training · Example training")
    assert output_path.read_text() == export.return_value
    assert json.loads(capsys.readouterr().out) == {
        "output": str(output_path),
        "run": "team/project/example",
        "prototype": True,
    }


def test_invalid_timeline_run_path_does_not_fetch_a_run(wandb_api, capsys, tmp_path):
    _, api = wandb_api
    assert (
        cli.main(
            [
                "dashboard",
                "timeline",
                "--run",
                "project/example",
                "--output",
                str(tmp_path / "out.html"),
            ]
        )
        == 1
    )
    api.run.assert_not_called()
    assert "--run must be entity/project/run-id" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
