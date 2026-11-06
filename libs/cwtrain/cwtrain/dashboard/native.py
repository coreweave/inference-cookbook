# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Packaged native W&B panels. No run data, credentials, or hosted app is bundled."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any

from cwtrain.dashboard.catalog import Framework

PHASES = (
    "mirror",
    "upload",
    "hotload",
    "hotload_first_replica",
    "hotload_peers",
    "rollout",
    "reward",
    "logprobs",
    "update",
)
OVERVIEW_FIELDS = {
    "run_name": "name",
    "run_id": "id",
    "created": "createdAt",
    "model": "hf_checkpoint",
    "algorithm": "advantage_estimator",
    "engine": "engine",
    "engine_version": "engine_version",
    "replicas_min": "replicas_min",
    "replicas_max": "replicas_max",
    "batch": "rollout_batch_size",
    "samples": "n_samples_per_prompt",
    "train_batch": "global_batch_size",
    "kl_enabled": "use_kl_loss",
    "kl_coef": "kl_loss_coef",
    "rollout_step": "rollout/step",
    "train_step": "train/step",
    "reward": "rollout/raw_reward",
}


def spec(kind: str) -> dict[str, Any]:
    if kind not in {"overview", "timeline"}:
        raise ValueError(f"unknown native panel: {kind}")
    return json.loads(files("cwtrain.dashboard").joinpath("specs", f"{kind}.json").read_text())


def panels(framework: Framework, metric_keys: set[str] | None) -> list[dict[str, Any]]:
    """Miles and Slime share the logged schema used by the approved native panels.

    Other frameworks retain their line charts until their native metadata and
    phase/reward-axis mappings have been validated. Missing phases stay absent.
    """
    if framework not in {"miles", "slime"}:
        return []
    result = [
        {
            "kind": "overview",
            "section": "Overview",
            "height": 365,
            "fields": dict(OVERVIEW_FIELDS),
            "strings": {"framework": framework.title()},
            # Root name comes after config so a nested deployment name cannot win.
            "query": {
                "config": {
                    "keys": [
                        "cwtrain",
                        "hf_checkpoint",
                        "advantage_estimator",
                        "rollout_batch_size",
                        "n_samples_per_prompt",
                        "global_batch_size",
                        "use_kl_loss",
                        "kl_loss_coef",
                    ]
                },
                "summary": {"keys": ["rollout/step", "train/step", "rollout/raw_reward"]},
                "id": None,
                "name": None,
                "createdAt": None,
            },
        }
    ]
    if metric_keys is None or any(
        {f"timeline/{phase}/start", f"timeline/{phase}/end"} <= metric_keys for phase in PHASES
    ):
        fields = {
            "run_name": "name",
            "timeline_step": "timeline/step",
            "history_step": "_step",
            "reward_step": "rollout/step",
            "reward_value": "rollout/raw_reward",
        }
        fields.update(
            {
                f"{edge}_{phase}": f"timeline/{phase}/{edge}"
                for phase in PHASES
                for edge in ("start", "end")
            }
        )
        result.append(
            {
                "kind": "timeline",
                "section": "Timeline",
                "height": 556,
                "fields": fields,
                "strings": {},
                # Sparse reward and phase rows cannot use history(keys=[all keys]):
                # that query requires every key on each row and returns no data.
                "query": {
                    "historyFold": {
                        "keys": ["_step"],
                        "foldKeys": [
                            value
                            for key, value in fields.items()
                            if key not in {"run_name", "history_step"}
                        ],
                    }
                },
            }
        )
    for panel in result:
        panel["spec_sha256"] = hashlib.sha256(
            json.dumps(spec(panel["kind"]), sort_keys=True).encode()
        ).hexdigest()
    return result


def workspace_type() -> type:
    """Isolate the SDK adapter needed for modern native settings on new views.

    Existing views are only read, never saved through this adapter. The SDK's
    dataclass load discards settings and is not a lossless editor for saved views.
    """
    import wandb_workspaces.workspaces as ws

    class NativeWorkspace(ws.Workspace):
        def _to_model(self):
            model = super()._to_model()
            section = model.spec.section
            section.workspace_settings = {
                "shouldAutoGeneratePanels": False,
                "ignoreOutliers": False,
                "linePlot": {
                    "coordinateZooming": True,
                    "maxRuns": 5,
                    "smoothingType": "none",
                    "smoothingWeight": 0,
                    "pointVisualizationMethod": "bucketing-gorilla",
                },
            }
            for group in section.panel_bank_config.sections:
                group.is_panels_auto = False
                for panel in group.panels:
                    # Native UI saves may compress auto panels after a reorder.
                    panel.is_auto = False
                if group.name in {"Overview", "Timeline"}:
                    group.flow_config.box_height = 365 if group.name == "Overview" else 556
                    for panel in group.panels:
                        # CustomChart currently hardcodes a 500-run query limit.
                        for arg in panel.config.user_query.query_fields[0].args:
                            if arg.name == "limit":
                                arg.value = 5
            return model

    return NativeWorkspace
