# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The recipe's files agree with what the launcher and the snapshot depend on."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
ROOT = RECIPE.parents[2]


def run_env_example() -> dict[str, str]:
    values = {}
    for line in (RECIPE / "k8s" / "run.env.example").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


@pytest.mark.parametrize(
    "setting",
    [
        r"strict: false",  # the launcher's options and EXTRA_TRAIN_ARGS override any key
        r"chat_template: tokenizer_default",  # the deployment keeps the base model's template
        r"chat_template_kwargs: \{enable_thinking: false\}",
        # the dataset's bytes are pinned: one file, by URL, at a commit
        r"\s+- path: https://huggingface\.co/datasets/\S+/resolve/[0-9a-f]{40}/\S+\.parquet",
        r"\s+ds_type: parquet",
        r"attn_implementation: sdpa",
        r"sample_packing: false",  # sdpa does not pack
        r"fsdp_version: 2",
        r"\s+state_dict_type: FULL_STATE_DICT",  # the final save is one HF checkpoint
        r'save_strategy: "no"',
    ],
)
def test_the_config_keeps_what_the_snapshot_depends_on(setting: str) -> None:
    text = (RECIPE / "sft_coreweave.yaml").read_text()
    assert re.search(rf"^{setting}\s*(#.*)?$", text, re.M), setting


@pytest.mark.parametrize("key", ["plugins", "hub_model_id", "wandb_project"])
def test_the_config_loads_no_plugin_and_sends_nothing_elsewhere(key: str) -> None:
    text = (RECIPE / "sft_coreweave.yaml").read_text()
    assert not re.search(rf"^\s*{key}:", text, re.M)


def test_the_example_deployment_is_the_staged_model_under_a_name_of_its_own() -> None:
    ours = json.loads((RECIPE / "deployment.example.json").read_text())
    others = [
        json.loads(path.read_text())
        for path in ROOT.glob("recipes/*/*/deployment*.example.json")
        if path.parent != RECIPE
    ]
    assert ours["name"] not in {other.get("name") for other in others}
    env = run_env_example()
    assert ours["model"]["path"] == f"{env['MODEL_PREFIX']}/{env['MODEL_SLUG']}"


def test_the_trainer_job_runs_the_image_entrypoint_without_an_inference_key() -> None:
    job = (RECIPE / "k8s" / "trainer-job.yaml").read_text()
    # args, not command: the entrypoint sets up the CUDA environment first
    assert re.search(r'^\s+args: \["bash", "/opt/cwtrain/run-trainer.sh"\]$', job, re.M)
    assert not re.search(r"^\s+command:", job, re.M)
    assert "INFERENCE_KEY_SECRET" not in job and "CWTRAIN_ROLLOUT_" not in job
    assert '{name: AXOLOTL_DO_NOT_TRACK, value: "${AXOLOTL_DO_NOT_TRACK}"}' in job
    assert run_env_example()["AXOLOTL_DO_NOT_TRACK"] == "1"


def test_the_trainer_image_is_pinned_by_digest_and_named_in_the_readme() -> None:
    image = run_env_example()["TRAINER_IMAGE"]
    assert re.fullmatch(r"axolotlai/axolotl:[\w.-]+@sha256:[0-9a-f]{64}", image)
    readme = (RECIPE / "README.md").read_text()
    versions = readme[readme.index("## Versions") :].split("\n## ")[0]
    assert f"`{image}`" in versions
