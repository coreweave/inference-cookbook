# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The example's settings fit the slime recipe: the image, slime commit and Megatron settings that
run.env.example pins, and rollout concurrency from run.env to the Job, sized against the example's
deployment."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]
ROOT = EXAMPLE.parents[2]
RECIPE = ROOT / "training" / "recipes" / "slime"


def run_env_example() -> dict[str, str]:
    values = {}
    for line in (EXAMPLE / "k8s" / "run.env.example").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("'\"")
    return values


def samples_per_request() -> int:
    """launch.py's default for --n-samples-per-prompt, the completions per group request."""
    source = (RECIPE / "launch.py").read_text()
    match = re.search(r'env_int\(env, "N_SAMPLES_PER_PROMPT", (\d+)\)', source)
    assert match, "launch.py no longer reads N_SAMPLES_PER_PROMPT"
    return int(match.group(1))


def test_example_deployment_runs_everything_the_trainer_keeps_in_flight() -> None:
    autoscaling = json.loads((EXAMPLE / "deployment.example.json").read_text())["autoscaling"]
    # autoscaling.concurrency becomes vLLM's --max-num-seqs on every replica
    capacity = autoscaling["min"] * autoscaling["concurrency"]
    in_flight = int(run_env_example()["ROLLOUT_CONCURRENCY"]) * samples_per_request()
    assert capacity >= in_flight


@pytest.mark.skipif(shutil.which("envsubst") is None, reason="render.sh needs envsubst")
def test_the_job_passes_rollout_concurrency_from_run_env(tmp_path: Path) -> None:
    # render.sh refuses unset values and placeholders, so start from the example, filled in
    env_file = tmp_path / "run.env"
    example = (EXAMPLE / "k8s" / "run.env.example").read_text()
    env_file.write_text(re.sub(r"<[a-z][a-z0-9 -]*>", "filled", example) + "ROLLOUT_CONCURRENCY=40\n")
    rendered = subprocess.run(
        [
            "bash",
            str(ROOT / "k8s" / "render.sh"),  # the shared k8s/, at the repository root
            str(env_file),
            str(RECIPE / "k8s" / "trainer-job.yaml"),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert '{name: CWTRAIN_ROLLOUT_CONCURRENCY, value: "40"}' in rendered


def test_run_env_defaults_fit_the_image_they_name() -> None:
    source = (RECIPE / "launch.py").read_text()
    tested = re.search(r'^TESTED_SLIME_COMMIT = "([0-9a-f]{40})"$', source, re.M)
    assert tested, "launch.py no longer names the slime commit it was tested with"
    env = run_env_example()
    image = env["TRAINER_IMAGE"]
    assert not image.endswith(":latest"), "pin the trainer image"
    assert env["SLIME_COMMIT"] == tested.group(1)
    if image.startswith("radixark/miles:"):
        # numpy 2 and a Megatron that is not slime's: lift the assertion, bring slime's Megatron
        assert env["SLIME_RELAX_NUMPY_CHECK"] == "1"
        assert len(env["MEGATRON_COMMIT"]) == 40
    elif image.startswith("slimerl/slime:"):
        assert env["SLIME_RELAX_NUMPY_CHECK"] == "0" and env["MEGATRON_COMMIT"] == ""
        assert not env["INSTANCE_TYPE"].startswith("gb"), "slimerl/slime images are amd64 only"
    else:
        pytest.fail(f"no rule for {image}")
