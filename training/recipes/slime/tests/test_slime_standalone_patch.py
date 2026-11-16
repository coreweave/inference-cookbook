# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from cwtrain.integrations.launch import apply_patches


@pytest.mark.parametrize(
    "standalone,gpus,expected",
    [(True, 0, ({}, [])), (False, 0, "normal"), (True, 2, "normal")],
)
def test_serving_owner_skips_router_only_for_standalone(
    tmp_path, standalone, gpus, expected
):
    # Both the serving owner and legacy manager use this startup entry point in 0.4.0.
    name = "slime/backends/sglang_utils/deployment.py"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text('''from typing import Any
import logging
logger = logging.getLogger(__name__)


def start_rollout_servers(args, pg) -> tuple[dict[str, Any], list[Any]]:
    """Start configured rollout servers without waiting for final engine initialization."""
    if args.rollout_external:
        return start_external_rollout_servers(args, start_router=_start_router)

    return "normal"
''')
    original = (
        Path(__file__).resolve().parents[1]
        / "patches/0001-standalone-delta-publish.patch"
    )
    header = "--- a/" + name
    patch = tmp_path / "deployment.patch"
    patch.write_text(header + original.read_text().split(header, 1)[1])
    apply_patches(tmp_path, [patch], tested="slime v0.4.0")
    start = runpy.run_path(str(path))["start_rollout_servers"]
    assert (
        start(
            SimpleNamespace(
                update_weight_standalone_publish=standalone,
                rollout_num_gpus=gpus,
                rollout_external=False,
            ),
            None,
        )
        == expected
    )
