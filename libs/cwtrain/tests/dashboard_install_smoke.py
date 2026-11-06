# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Exercise the installed wheel and real W&B SDK without remote writes."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import wandb

import cwtrain
from cwtrain.dashboard import native
from cwtrain.dashboard.workspace import build_workspace, definition
from cwtrain.integrations import tracking

assert Path(cwtrain.__file__).is_relative_to(Path(sys.prefix)), cwtrain.__file__
with wandb.init(project="cwtrain-install-check", mode="offline", dir=".") as run:
    assert tracking.update_config({"framework": "miles"})
    assert tracking.log_metrics(
        {
            **tracking.span("rollout", 100, 102),
            tracking.TIMELINE_STEP: 0,
            "rollout/raw_reward": 0.5,
        },
        step_metric="rollout/step",
        step=0,
        prefixes=("rollout/", "timeline/"),
    )
    assert run.config["cwtrain"]["framework"] == "miles"
    assert run.summary["rollout/raw_reward"] == 0.5
    assert run.summary["timeline/rollout/start"] == 100

for framework in ("miles", "slime", "nemo-rl", "axolotl"):
    workspace = build_workspace("compatibility", "offline", definition(framework))
    model = workspace._to_model().model_dump(mode="json", by_alias=True, exclude_none=True)
    assert model["spec"]["section"]["panelBankConfig"]["sections"]

# Exercise the real preset method and its protobuf serialization, replacing only transport.
api = object.__new__(wandb.Api)
api._service_api = Mock()
api._service_api.send_api_request.return_value = SimpleNamespace(
    create_custom_chart_response=SimpleNamespace(chart_id="compatibility/test")
)
for kind in ("overview", "timeline"):
    spec = native.spec(kind)
    assert (
        api.create_custom_chart(
            entity="compatibility",
            name="test",
            display_name="Post-training",
            spec_type="vega2",
            access="private",
            spec=spec,
        )
        == "compatibility/test"
    )
    request = api._service_api.send_api_request.call_args.args[0].create_custom_chart_request
    assert request.access == "PRIVATE"
    assert json.loads(request.spec) == spec
