# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Native schema, save-time panel preservation and private preset lifecycle."""

import copy
import json
from unittest.mock import Mock

import pytest

from cwtrain.dashboard.workspace import build_workspace, definition, save_managed
from cwtrain.errors import CWTrainError


def test_native_panels_use_sparse_history_and_selected_run_metadata():
    plan = definition(
        "miles",
        metric_keys={
            "timeline/update/start",
            "timeline/update/end",
            "train/loss",
        },
    )
    workspace = build_workspace("team", "project", plan)
    model = workspace._to_model().model_dump(mode="json", by_alias=True, exclude_none=True)
    assert model["displayName"] == "Post-training"
    section = model["spec"]["section"]
    groups = section["panelBankConfig"]["sections"]
    assert [group["name"] for group in groups] == ["Overview", "Timeline", "Training"]
    assert [group["flowConfig"]["boxHeight"] for group in groups[:2]] == [365, 556]
    for group in groups:
        assert group["isPanelsAuto"] is False
        assert all(panel["isAuto"] is False for panel in group["panels"])
    assert section["workspaceSettings"]["linePlot"]["coordinateZooming"] is True
    overview, timeline = [group["panels"][0]["config"] for group in groups[:2]]
    for panel in (overview, timeline):
        root = panel["userQuery"]["queryFields"][0]
        assert root["args"] == [
            {"name": "runSets", "value": "${runSets}"},
            {"name": "limit", "value": 5},
        ]
    assert [field["name"] for field in overview["userQuery"]["queryFields"][0]["fields"]] == [
        "config",
        "summary",
        "id",
        "name",
        "createdAt",
    ]
    history = timeline["userQuery"]["queryFields"][0]["fields"][0]
    assert history["name"] == "historyFold"
    args = {arg["name"]: arg["value"] for arg in history["args"]}
    assert args["keys"] == ["_step"]
    assert {"timeline/step", "rollout/step", "rollout/raw_reward"} <= set(args["foldKeys"])
    assert {"timeline/update/start", "timeline/update/end"} <= set(args["foldKeys"])


def test_discovery_does_not_create_timeline_without_complete_span_keys():
    plan = definition("slime", metric_keys={"timeline/update/start"})
    assert [panel["kind"] for panel in plan["native_panels"]] == ["overview"]
    assert plan["native_panels"][0]["strings"] == {"framework": "Slime"}
    assert definition("nemo-rl")["native_panels"] == []
    assert definition("axolotl")["native_panels"] == []


@pytest.mark.parametrize("fail_at", [None, 1, 2])
def test_presets_registered_once_and_ambiguous_creation_blocks_retry(
    tmp_path, monkeypatch, fail_at
):
    plan = definition("miles")
    workspace = build_workspace("team", "project", plan)
    cls = type(workspace)
    state_path = tmp_path / "state.json"
    saved = []
    remote = []

    def save(self):
        saved.append(True)
        remote.append(copy.deepcopy(self))

    monkeypatch.setattr(cls, "save", save)
    monkeypatch.setattr(
        cls, "url", property(lambda self: "https://wandb.example/team/project?nw=view123456")
    )
    monkeypatch.setattr(cls, "from_url", classmethod(lambda cls, url: copy.deepcopy(remote[0])))
    api = Mock()
    created = []

    def create(**kwargs):
        state = json.loads(state_path.read_text())
        record = next(iter(state["views"].values()))
        assert record["pending"] is True
        assert any(
            p["name"] == kwargs["name"] and p["status"] == "pending"
            for p in record["presets"].values()
        )
        assert kwargs["access"] == "private" and kwargs["entity"] == "team"
        created.append(kwargs["name"])
        if len(created) == fail_at:
            raise TimeoutError("server may have committed preset creation")
        return f"team/{kwargs['name']}"

    api.create_custom_chart.side_effect = create
    options = dict(
        path=state_path,
        api_host="https://api.wandb.example",
        username="owner",
        app_url="https://wandb.example/team/project/runs/run",
        api=api,
    )
    if fail_at:
        with pytest.raises(TimeoutError):
            save_managed(workspace, plan, **options)
        with pytest.raises(CWTrainError, match="unknown outcome"):
            save_managed(workspace, plan, **options)
        assert not saved and len(created) == fail_at
    else:
        assert save_managed(workspace, plan, **options)["action"] == "created"
        assert save_managed(workspace, plan, **options)["action"] == "unchanged"
        assert len(saved) == 1 and len(created) == 2
        assert [s.panels[0].chart_name for s in remote[0].sections[:2]] == [
            f"team/{name}" for name in created
        ]
