# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Build native W&B views and safely maintain only a recorded managed view."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from cwtrain.dashboard.catalog import SPEC_VERSION, Framework, catalog, system_panels
from cwtrain.errors import CWTrainError

_LABELS = {"miles": "Miles", "slime": "Slime", "nemo-rl": "NeMo-RL", "axolotl": "Axolotl"}


def definition(
    framework: Framework,
    *,
    metric_keys: set[str] | None = None,
    system_keys: set[str] | None = None,
) -> dict[str, Any]:
    """An offline, versioned plan. Missing data never becomes a zero series."""
    panels = catalog(framework)
    omitted = []
    if metric_keys is not None:
        selected = []
        for panel in panels:
            metrics = tuple(key for key in panel.metrics if key in metric_keys)
            omitted.extend(key for key in panel.metrics if key not in metric_keys)
            if metrics:
                selected.append(replace(panel, metrics=metrics))
        panels = tuple(selected)
    panels += system_panels(sorted(system_keys or ()))
    panel_specs = []
    system_fields = set()
    for panel in panels:
        spec = asdict(panel)
        if panel.section == "System":
            # W&B exposes event keys with a dot, but some UI versions use a slash
            # after 'system'. Match either, and retain distinct device/writer lines.
            base_key = panel.metrics[0].split("/l:", 1)[0]
            field = base_key.rsplit(".", 1)[-1]
            if field in system_fields:
                continue
            system_fields.add(field)
            prefix = r"^system[./]gpu\.[0-9]+\." if ".gpu." in base_key else r"^system[./]"
            spec["metric_regex"] = prefix + field + r"(/l:.*)?$"
            spec["metrics"] = ()
            spec["title"] = panel.title.split(" · ", 1)[0]
        panel_specs.append(spec)
    return {
        "spec_version": SPEC_VERSION,
        "framework": framework,
        "name": f"Post-training — {_LABELS[framework]}",
        "panels": panel_specs,
        "omitted_metrics": sorted(set(omitted)),
        "system_discovery": "recent matching runs; unavailable offline",
        "run_filter": {"tags": {"$in": [framework]}, "jobType": "train"},
        "run_order": "-created_at",
    }


def build_workspace(entity: str, project: str, plan: dict[str, Any]) -> Any:
    """Load optional dependencies only when constructing a native workspace."""
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as exc:
        raise CWTrainError("install the dashboard extra: pip install 'cwtrain[dashboard]'") from exc

    sections = {}
    for panel in plan["panels"]:
        sections.setdefault(panel["section"], []).append(
            wr.LinePlot(
                title=panel["title"],
                x=panel["x"],
                y=list(panel["metrics"]),
                metric_regex=panel.get("metric_regex"),
                title_x=panel["x"],
                title_y=panel["unit"],
                aggregate=False,
                smoothing_type="none",
                smoothing_factor=0,
                ignore_outliers=False,
                max_runs_to_show=5,
                point_visualization_method="bucketing-gorilla",
            )
        )
    framework_filter = ws.Tags().isin([plan["framework"]])
    if plan.get("included_runs"):
        framework_filter = ws.Or(framework_filter, ws.Metric("ID").isin(plan["included_runs"]))
    return ws.Workspace(
        entity=entity,
        project=project,
        name=plan["name"],
        auto_generate_panels=False,
        sections=[
            ws.Section(
                name=name,
                panels=panels,
                is_open=True,
                layout_settings=ws.SectionLayoutSettings(columns=2, rows=2),
            )
            for name, panels in sections.items()
        ],
        runset_settings=ws.RunsetSettings(
            filters=ws.And(ws.Metric("JobType") == "train", framework_filter),
            order=[ws.Ordering(ws.Metric("CreatedTimestamp"), ascending=False)],
        ),
        settings=ws.WorkspaceSettings(
            smoothing_type="none",
            smoothing_weight=0,
            ignore_outliers=False,
            max_runs=5,
            point_visualization_method="bucketing",
        ),
    )


def _serializable(value: Any) -> Any:
    # Include stashed unknown panel settings as well as public dataclass fields:
    # editing a panel that this SDK cannot interpret must also block an overwrite.
    if is_dataclass(value) and not isinstance(value, type):
        contents = dict(vars(value)) if hasattr(value, "__dict__") else {}
        contents.update({field.name: getattr(value, field.name) for field in fields(value)})
        return {key: _serializable(item) for key, item in contents.items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_serializable(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def default_state_file() -> Path:
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "cwtrain" / "dashboards.json"


@contextmanager
def state_file(path: Path) -> Iterator[dict[str, Any]]:
    """Serialize installers; do not let concurrent invocations create duplicate views."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise CWTrainError(
            f"dashboard installer is locked by {lock}; check for another process"
        ) from exc
    try:
        os.close(fd)
        data = json.loads(path.read_text()) if path.exists() else {"version": 1, "views": {}}
        if data.get("version") != 1 or not isinstance(data.get("views"), dict):
            raise CWTrainError(f"unsupported dashboard state file: {path}")
        yield data
    finally:
        lock.unlink(missing_ok=True)


def write_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(state, stream, indent=2)
        stream.write("\n")
    tmp.replace(path)


def _managed_url(url: str, entity: str, project: str, app_url: str) -> str:
    parsed = urlsplit(url)
    expected = urlsplit(app_url)
    view = parse_qs(parsed.query).get("nw", [""])[0]
    if (
        (parsed.scheme, parsed.netloc) != (expected.scheme, expected.netloc)
        or parsed.path.rstrip("/") not in {f"/{entity}/{project}", f"/{entity}/{project}/workspace"}
        or not re.fullmatch(r"[A-Za-z0-9_-]{8,}", view)
        or view.startswith(("user", "nwuser", "nw-user"))
    ):
        raise CWTrainError("state does not identify a saved view in this W&B host/project")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"/{entity}/{project}", urlencode({"nw": view}), "")
    )


def save_managed(
    workspace: Any,
    plan: dict[str, Any],
    *,
    path: Path,
    api_host: str,
    username: str,
    app_url: str,
) -> dict[str, Any]:
    """Reuse a saved URL, reject drift, and never overwrite a personal/default view."""
    scope = fingerprint(
        [api_host.rstrip("/"), username, workspace.entity, workspace.project, plan["framework"]]
    )
    plan_hash = fingerprint(plan)
    with state_file(path) as state:
        record = state["views"].get(scope)
        action = "created"
        if record:
            if record.get("pending"):
                raise CWTrainError(
                    f"a previous dashboard save has an unknown outcome; inspect W&B and {path} "
                    "before retrying. No new view was created by this invocation."
                )
            url = _managed_url(record["url"], workspace.entity, workspace.project, app_url)
            current = type(workspace).from_url(url)
            if current.name != plan["name"] or fingerprint(current) != record.get("remote_hash"):
                raise CWTrainError(
                    "managed view was edited or its save was not verified; "
                    f"leaving it unchanged: {url}. "
                    "Use a separate --state-file to create a new managed view; keep the original."
                )
            if record.get("plan_hash") == plan_hash:
                return {"url": url, "action": "unchanged", "state_file": str(path)}
            current.sections = workspace.sections
            current.settings = workspace.settings
            current.runset_settings = workspace.runset_settings
            workspace = current
            action = "updated"
        # A server may commit a creation even if its response is lost. Record intent
        # first and fail closed on the next invocation rather than creating a duplicate.
        state["views"][scope] = {"pending": True, "previous": record}
        write_state(path, state)
        workspace.save()
        # Persist the URL before reloading. A failed verification must not create another view.
        url = workspace.url
        record = {"url": url, "plan_hash": plan_hash, "remote_hash": None}
        state["views"][scope] = record
        write_state(path, state)
        url = _managed_url(url, workspace.entity, workspace.project, app_url)
        current = type(workspace).from_url(url)
        record["remote_hash"] = fingerprint(current)
        write_state(path, state)
        return {"url": url, "action": action, "state_file": str(path)}
