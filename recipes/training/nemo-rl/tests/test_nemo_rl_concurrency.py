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
    """grpo.num_generations_per_prompt, the completions per request (one request per prompt)."""
    config = (RECIPE / "grpo_coreweave.yaml").read_text()
    match = re.search(r"^\s*num_generations_per_prompt:\s*(\d+)", config, re.MULTILINE)
    assert match, "grpo_coreweave.yaml no longer sets num_generations_per_prompt"
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
            str(RECIPE / "k8s" / "render.sh"),
            str(env_file),
            str(RECIPE / "k8s" / "trainer-job.yaml"),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert '{name: CWTRAIN_ROLLOUT_CONCURRENCY, value: "40"}' in rendered


def test_the_config_defaults_to_the_run_env_value() -> None:
    config = (RECIPE / "grpo_coreweave.yaml").read_text()
    match = re.search(r"concurrency: \$\{oc\.env:CWTRAIN_ROLLOUT_CONCURRENCY,(\d+)\}", config)
    assert match, "coreweave_cfg.concurrency no longer reads CWTRAIN_ROLLOUT_CONCURRENCY"
    assert match.group(1) == run_env_example()["ROLLOUT_CONCURRENCY"]


@pytest.mark.parametrize(("value", "seen"), [("", "unset"), (" ", "unset"), ("48", "48")])
def test_run_trainer_leaves_an_empty_value_to_the_config(tmp_path: Path, value, seen) -> None:
    # a setting missing from run.env renders as ""; oc.env would pass that on instead of 32
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    python = tmp_path / "python"
    # the first call is the snapshot access check (-m cwtrain.integrations.preflight)
    python.write_text('#!/bin/sh\n[ "$1" = -m ] && exit 0\nprintf %s "${CWTRAIN_ROLLOUT_CONCURRENCY-unset}"\n')
    python.chmod(0o755)
    out = subprocess.run(
        ["bash", str(RECIPE / "k8s" / "run-trainer.sh")],
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "NEMO_RL_PYTHON": str(python),
            "CWTRAIN_ROLLOUT_CONCURRENCY": value,
        },
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert out == seen
