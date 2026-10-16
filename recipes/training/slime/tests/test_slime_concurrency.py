# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Rollout concurrency: run.env to the Job to the trainer, sized against the example deployment."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]


def run_env_example() -> dict[str, str]:
    values = {}
    for line in (RECIPE / "k8s" / "run.env.example").read_text().splitlines():
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
    autoscaling = json.loads((RECIPE / "deployment.example.json").read_text())["autoscaling"]
    # autoscaling.concurrency becomes vLLM's --max-num-seqs on every replica
    capacity = autoscaling["min"] * autoscaling["concurrency"]
    in_flight = int(run_env_example()["ROLLOUT_CONCURRENCY"]) * samples_per_request()
    assert capacity >= in_flight


@pytest.mark.skipif(shutil.which("envsubst") is None, reason="render.sh needs envsubst")
def test_the_job_passes_rollout_concurrency_from_run_env(tmp_path: Path) -> None:
    # render.sh refuses unset values and placeholders, so start from the example, filled in
    env_file = tmp_path / "run.env"
    example = (RECIPE / "k8s" / "run.env.example").read_text()
    env_file.write_text(re.sub(r"<[a-z][a-z0-9 -]*>", "filled", example) + "ROLLOUT_CONCURRENCY=40\n")
    rendered = subprocess.run(
        [
            "bash",
            str(RECIPE.parents[2] / "k8s" / "render.sh"),  # the shared k8s/, at the repository root
            str(env_file),
            str(RECIPE / "k8s" / "trainer-job.yaml"),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert '{name: CWTRAIN_ROLLOUT_CONCURRENCY, value: "40"}' in rendered
