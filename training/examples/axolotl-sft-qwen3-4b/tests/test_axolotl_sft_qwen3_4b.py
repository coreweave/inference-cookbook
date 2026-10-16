# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The example's settings fit the Axolotl SFT recipe: the image that run.env.example pins, the
telemetry switch, and a deployment of the staged model under a name of its own."""

from __future__ import annotations

import json
import re
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1]
ROOT = EXAMPLE.parents[2]
RECIPE = ROOT / "training" / "recipes" / "axolotl-sft"


def run_env_example() -> dict[str, str]:
    values = {}
    for line in (EXAMPLE / "k8s" / "run.env.example").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def test_the_example_deployment_is_the_staged_model_under_a_name_of_its_own() -> None:
    ours = json.loads((EXAMPLE / "deployment.example.json").read_text())
    others = [
        json.loads(path.read_text())
        for units in ("training/examples/*", "evaluation/recipes/*")
        for path in ROOT.glob(f"{units}/deployment*.example.json")
        if path.parent != EXAMPLE
    ]
    assert others, "no other deployment to compare with"
    assert ours["name"] not in {other.get("name") for other in others}
    env = run_env_example()
    assert ours["model"]["path"] == f"{env['MODEL_PREFIX']}/{env['MODEL_SLUG']}"


def test_the_example_turns_off_axolotl_telemetry() -> None:
    # the recipe's Job passes AXOLOTL_DO_NOT_TRACK on from run.env
    assert run_env_example()["AXOLOTL_DO_NOT_TRACK"] == "1"


def test_the_trainer_image_is_pinned_by_digest_and_named_in_the_recipe_readme() -> None:
    image = run_env_example()["TRAINER_IMAGE"]
    assert re.fullmatch(r"axolotlai/axolotl:[\w.-]+@sha256:[0-9a-f]{64}", image)
    readme = (RECIPE / "README.md").read_text()
    versions = readme[readme.index("## Versions") :].split("\n## ")[0]
    assert f"`{image}`" in versions
