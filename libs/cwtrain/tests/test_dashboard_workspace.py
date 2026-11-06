# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Serialization and ownership checks without creating any remote W&B views."""

import copy
import json
import re
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from cwtrain.dashboard.workspace import (
    build_workspace,
    default_state_file,
    definition,
    fingerprint,
    save_managed,
    state_file,
)
from cwtrain.errors import CWTrainError


def test_discovery_keeps_available_members_and_never_invents_zero_series():
    plan = definition("miles", metric_keys={"train/pg_clipfrac", "train/loss"})
    panels = {panel["title"]: panel for panel in plan["panels"]}
    assert set(panels) == {"Training loss", "Clipped fractions"}
    assert panels["Clipped fractions"]["metrics"] == ("train/pg_clipfrac",)
    assert panels["Clipped fractions"]["unit"] == "fraction"
    assert "train/tis_clipfrac" in plan["omitted_metrics"]
    assert "train/loss" not in plan["omitted_metrics"]
    assert definition("miles", metric_keys=set())["panels"] == []


def test_reference_kl_is_present_only_when_logged_and_follows_update_kl():
    keys = {"train/loss", "train/ppo_kl", "train/train_rollout_kl"}
    plan = definition("miles", metric_keys=keys)
    assert plan["panels"][0]["metrics"] == ("train/ppo_kl",)
    assert "train/kl_loss" in plan["omitted_metrics"]
    assert not any("train/kl_loss" in panel["metrics"] for panel in plan["panels"])

    plan = definition("miles", metric_keys=keys | {"train/kl_loss"})
    assert [panel["metrics"] for panel in plan["panels"][:2]] == [
        ("train/ppo_kl",),
        ("train/kl_loss",),
    ]
    assert "train/kl_loss" not in plan["omitted_metrics"]


def test_native_sdk_serializes_axes_filters_order_and_full_fidelity():
    pytest.importorskip("wandb_workspaces.workspaces")
    plan = definition(
        "miles",
        metric_keys={"train/loss", "rollout/raw_reward", "hotload/snapshot_bytes"},
        system_keys={
            "system.gpu.0.gpu/l:writer-a",
            "system.gpu.1.gpu/l:writer-b",
            "system.gpu.0.powerWatts/l:writer-a",
            "system.cpu/l:writer-a",
        },
    )
    plan["included_runs"] = ["example-0", "example-1"]
    workspace = build_workspace("team", "project", plan)
    model = workspace._to_model().model_dump(mode="json", by_alias=True, exclude_none=True)
    section = model["spec"]["section"]
    sections = section["panelBankConfig"]["sections"]
    assert [item["name"] for item in sections] == ["Overview", "Training", "Hot-load", "System"]
    panels = [panel["config"] for item in sections[1:] for panel in item["panels"]]
    by_title = {panel["chartTitle"]: panel for panel in panels}
    assert by_title["Training loss"]["xAxis"] == "train/step"
    assert by_title["Policy reward"]["xAxis"] == "rollout/step"
    snapshot = next(panel for panel in panels if panel["metrics"] == ["hotload/snapshot_bytes"])
    assert snapshot["yAxisTitle"] == "bytes" and snapshot["xAxis"] == "rollout/step"
    for panel in panels:
        assert panel["aggregate"] is False
        assert panel["smoothingType"] == "none"
        assert panel["smoothingWeight"] == 0
        assert panel["ignoreOutliers"] is False
        assert panel["pointVisualizationMethod"] == "bucketing-gorilla"
    gpu = by_title["GPU utilization"]
    assert gpu["xAxis"] == "_timestamp"
    assert gpu["useMetricRegex"] is True and gpu["metrics"] == []
    assert re.fullmatch(gpu["metricRegex"], "system/gpu.12.gpu/l:new-writer")
    assert not re.fullmatch(gpu["metricRegex"], "system.gpu.process.0.gpu")
    assert not re.fullmatch(gpu["metricRegex"], "system.gpu.0.memory")
    assert [panel["config"]["chartTitle"] for panel in sections[-1]["panels"]] == [
        "GPU utilization",
        "GPU power",
        "Reported process CPU",
    ]
    runset = section["runSets"][0]
    assert runset["filters"]["filters"] == [
        {
            "key": {"section": "run", "name": "jobType"},
            "op": "=",
            "value": "train",
            "disabled": False,
        },
        {
            "filters": [
                {
                    "key": {"section": "run", "name": "tags"},
                    "op": "IN",
                    "value": ["miles"],
                    "disabled": False,
                },
                {
                    "key": {"section": "run", "name": "name"},
                    "op": "IN",
                    "value": ["example-0", "example-1"],
                    "disabled": False,
                    "connector": "OR",
                },
            ],
            "connector": "AND",
        },
    ]
    assert runset["sort"]["keys"] == [
        {"key": {"section": "run", "name": "createdAt"}, "ascending": False}
    ]
    assert section["settings"]["pointVisualizationMethod"] == "bucketing-gorilla"
    assert section["settings"]["shouldAutoGeneratePanels"] is False


@dataclass
class RemoteWorkspace:
    """The SDK's save mutates identity; reload returns a separate dataclass."""

    entity: str = "team"
    project: str = "project"
    name: str = "Post-training"
    sections: list = field(default_factory=lambda: ["original panels"])
    settings: dict = field(default_factory=lambda: {"smoothing": 0})
    runset_settings: dict = field(default_factory=lambda: {"filter": "train"})
    url: str = ""
    _stashed_unknown: dict = field(default_factory=lambda: {"future_setting": {"enabled": True}})

    remote: ClassVar[dict] = {}
    saves: ClassVar[list] = []
    reads: ClassVar[list] = []
    fail_reload: ClassVar[bool] = False

    def save(self):
        if not self.url:
            self.url = (
                f"https://wandb.example/{self.entity}/{self.project}?nw=view{len(self.remote):08}"
            )
        self.saves.append(self.url)
        self.remote[self.url] = copy.deepcopy(self)

    @classmethod
    def from_url(cls, url):
        cls.reads.append(url)
        if cls.fail_reload:
            raise OSError("verification unavailable")
        return copy.deepcopy(cls.remote[url])


@pytest.fixture
def managed(tmp_path):
    RemoteWorkspace.remote = {}
    RemoteWorkspace.saves = []
    RemoteWorkspace.reads = []
    RemoteWorkspace.fail_reload = False
    plan = definition("miles", metric_keys={"train/loss"})
    # These lifecycle fixtures exercise the view independently of chart registration.
    plan["native_panels"] = []
    options = {
        "path": tmp_path / "state" / "dashboards.json",
        "api_host": "https://api.wandb.example",
        "username": "researcher",
        "app_url": "https://wandb.example/team/project/runs/example",
    }
    return plan, options


def test_creation_and_unchanged_rerun_keep_one_managed_view_but_spec_update_stops(managed):
    plan, options = managed
    created = save_managed(RemoteWorkspace(), plan, **options)
    assert created["action"] == "created"
    state = json.loads(options["path"].read_text())
    record = next(iter(state["views"].values()))
    assert record["remote_hash"] == fingerprint(RemoteWorkspace.remote[created["url"]])
    assert options["path"].stat().st_mode & 0o777 == 0o600
    original_state = options["path"].read_bytes()
    assert save_managed(RemoteWorkspace(), plan, **options)["action"] == "unchanged"
    assert options["path"].read_bytes() == original_state
    assert len(RemoteWorkspace.saves) == 1

    revised = {**plan, "spec_version": plan["spec_version"] + 1}
    desired = RemoteWorkspace(sections=["new panel"], settings={"smoothing": 0, "limit": 5})
    with pytest.raises(CWTrainError, match="cannot preserve all native UI settings"):
        save_managed(desired, revised, **options)
    assert len(RemoteWorkspace.remote) == 1 and len(RemoteWorkspace.saves) == 1
    assert RemoteWorkspace.remote[created["url"]].sections == ["original panels"]
    assert RemoteWorkspace.remote[created["url"]].settings == {"smoothing": 0}
    assert options["path"].read_bytes() == original_state


def test_native_settings_discarded_by_sdk_cannot_be_silently_overwritten(managed, monkeypatch):
    ws = pytest.importorskip("wandb_workspaces.workspaces")
    plan, options = managed
    created = save_managed(RemoteWorkspace(), plan, **options)
    desired = build_workspace("team", "project", plan)
    raw = desired._to_model()
    raw.spec.section.workspace_settings = {
        "shouldAutoGeneratePanels": False,
        "futureNativeSetting": {"enabled": False},
    }
    before = ws.Workspace._from_model(raw)
    raw.spec.section.workspace_settings["futureNativeSetting"]["enabled"] = True
    after = ws.Workspace._from_model(raw)
    # A real SDK load discards these settings; a dataclass fingerprint alone
    # cannot establish that replacing the saved view would be safe.
    assert after._to_model().spec.section.workspace_settings is None
    assert fingerprint(before) == fingerprint(after)
    state = json.loads(options["path"].read_text())
    next(iter(state["views"].values()))["remote_hash"] = fingerprint(before)
    options["path"].write_text(json.dumps(state))
    original_state = options["path"].read_bytes()
    monkeypatch.setattr(ws.Workspace, "from_url", classmethod(lambda cls, url: after))
    monkeypatch.setattr(ws.Workspace, "save", lambda self: pytest.fail("rewrote the saved view"))
    assert save_managed(desired, plan, **options)["url"] == created["url"]
    with pytest.raises(CWTrainError, match="cannot preserve all native UI settings"):
        save_managed(desired, {**plan, "spec_version": plan["spec_version"] + 1}, **options)
    assert options["path"].read_bytes() == original_state


def test_new_indexed_runs_and_recovered_fallbacks_reuse_without_writing(managed):
    plan, options = managed
    plan = {**plan, "included_runs": ["index-miss"]}
    created = save_managed(RemoteWorkspace(), plan, **options)
    before = options["path"].read_bytes()
    # A new tagged run is handled by the native tag filter. The previous index
    # miss may also disappear from the discovered cohort or become indexed.
    refreshed = {**plan, "included_runs": []}
    result = save_managed(RemoteWorkspace(), refreshed, **options)
    assert result["action"] == "unchanged" and result["url"] == created["url"]
    assert options["path"].read_bytes() == before
    assert len(RemoteWorkspace.saves) == 1
    # Ignoring ID changes indiscriminately would hide this new required run.
    with pytest.raises(CWTrainError, match="dashboard definition changed"):
        save_managed(RemoteWorkspace(), {**plan, "included_runs": ["new-index-miss"]}, **options)
    assert options["path"].read_bytes() == before and len(RemoteWorkspace.saves) == 1


def test_legacy_state_without_fallback_coverage_stays_fail_closed(managed):
    plan, options = managed
    plan = {**plan, "included_runs": ["index-miss"]}
    save_managed(RemoteWorkspace(), plan, **options)
    state = json.loads(options["path"].read_text())
    record = next(iter(state["views"].values()))
    del record["template_hash"]
    del record["included_runs"]
    options["path"].write_text(json.dumps(state))
    before = options["path"].read_bytes()
    assert save_managed(RemoteWorkspace(), plan, **options)["action"] == "unchanged"
    with pytest.raises(CWTrainError, match="dashboard definition changed"):
        save_managed(RemoteWorkspace(), {**plan, "included_runs": []}, **options)
    assert options["path"].read_bytes() == before and len(RemoteWorkspace.saves) == 1


@pytest.mark.parametrize("change", ["name", "sections", "stashed"])
def test_remote_edits_including_unrecognized_sdk_fields_block_overwrites(managed, change):
    plan, options = managed
    created = save_managed(RemoteWorkspace(), plan, **options)
    remote = RemoteWorkspace.remote[created["url"]]
    if change == "name":
        remote.name = "My custom view"
    elif change == "sections":
        remote.sections.append("personal panel")
    else:
        remote._stashed_unknown["future_setting"]["enabled"] = False
    before = options["path"].read_bytes()
    with pytest.raises(CWTrainError, match="managed view was edited"):
        save_managed(RemoteWorkspace(), {**plan, "spec_version": 2}, **options)
    assert len(RemoteWorkspace.saves) == 1
    assert options["path"].read_bytes() == before
    assert not options["path"].with_suffix(".json.lock").exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://other.example/team/project?nw=validview123",
        "https://wandb.example/team/other?nw=validview123",
        "http://wandb.example/team/project?nw=validview123",
        "https://wandb.example/team/project?nw=userresearcher",
        "https://wandb.example/team/project?nw=nwuserresearcher",
        "https://wandb.example/team/project?nw=nw-user-researcher",
        "https://wandb.example/team/project",
    ],
)
def test_record_cannot_redirect_updates_to_another_host_project_or_personal_view(managed, url):
    plan, options = managed
    save_managed(RemoteWorkspace(), plan, **options)
    state = json.loads(options["path"].read_text())
    next(iter(state["views"].values()))["url"] = url
    options["path"].write_text(json.dumps(state))
    with pytest.raises(CWTrainError, match="saved view in this W&B host/project"):
        save_managed(RemoteWorkspace(), plan, **options)
    assert len(RemoteWorkspace.saves) == 1 and len(RemoteWorkspace.reads) == 1


def test_failed_verification_records_identity_and_prevents_duplicate_creation(managed):
    plan, options = managed
    RemoteWorkspace.fail_reload = True
    with pytest.raises(OSError, match="verification unavailable"):
        save_managed(RemoteWorkspace(), plan, **options)
    record = next(iter(json.loads(options["path"].read_text())["views"].values()))
    assert record["url"] in RemoteWorkspace.remote and record["remote_hash"] is None
    RemoteWorkspace.fail_reload = False
    with pytest.raises(CWTrainError, match="save was not verified"):
        save_managed(RemoteWorkspace(), plan, **options)
    assert len(RemoteWorkspace.saves) == 1


def test_unknown_sdk_panel_config_is_fingerprinted_and_blocks_overwrite(managed):
    interface = pytest.importorskip("wandb_workspaces.reports.v2.interface")
    panel = interface.UnknownPanel(config={"custom": {"metric": "train/loss"}})
    before = fingerprint(panel)
    panel.config["custom"]["metric"] = "train/grad_norm"
    assert fingerprint(panel) != before

    plan, options = managed
    created = save_managed(RemoteWorkspace(sections=[panel]), plan, **options)
    remote = RemoteWorkspace.remote[created["url"]]
    remote.sections[0].config["custom"]["metric"] = "personal/metric"
    with pytest.raises(CWTrainError, match="managed view was edited"):
        save_managed(RemoteWorkspace(), {**plan, "spec_version": 2}, **options)
    assert len(RemoteWorkspace.saves) == 1


def test_ambiguous_committed_creation_blocks_retry_without_creating_a_duplicate(
    managed, monkeypatch
):
    plan, options = managed
    original_save = RemoteWorkspace.save

    def committed_but_response_lost(workspace):
        record = next(iter(json.loads(options["path"].read_text())["views"].values()))
        assert record["pending"] is True
        assert record["previous"] is None
        original_save(workspace)
        raise TimeoutError("response lost after server commit")

    monkeypatch.setattr(RemoteWorkspace, "save", committed_but_response_lost)
    with pytest.raises(TimeoutError, match="response lost"):
        save_managed(RemoteWorkspace(), plan, **options)
    saves, reads = len(RemoteWorkspace.saves), len(RemoteWorkspace.reads)
    assert len(RemoteWorkspace.remote) == 1
    with pytest.raises(CWTrainError, match="previous dashboard save has an unknown outcome"):
        save_managed(RemoteWorkspace(), plan, **options)
    assert len(RemoteWorkspace.saves) == saves and len(RemoteWorkspace.reads) == reads
    assert len(RemoteWorkspace.remote) == 1


@pytest.mark.parametrize("scope", ["username", "api_host"])
def test_account_or_host_change_creates_a_separate_managed_record(managed, scope):
    plan, options = managed
    first = save_managed(RemoteWorkspace(), plan, **options)
    options[scope] += "-other"
    second = save_managed(RemoteWorkspace(), plan, **options)
    assert second["action"] == "created" and first["url"] != second["url"]
    assert len(json.loads(options["path"].read_text())["views"]) == 2


def test_state_lock_blocks_concurrent_installer_and_survives_its_rejection(tmp_path):
    path = tmp_path / "dashboards.json"
    lock = tmp_path / "dashboards.json.lock"
    with state_file(path):
        assert lock.exists()
        with pytest.raises(CWTrainError, match="installer is locked"):
            with state_file(path):
                pytest.fail("a concurrent installer entered the critical section")
        assert lock.exists()
    assert not lock.exists()


def test_default_state_is_user_scoped(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert default_state_file() == tmp_path / "cwtrain" / "dashboards.json"
