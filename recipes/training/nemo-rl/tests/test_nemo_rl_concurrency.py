# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Rollout concurrency: run.env to the Job to the trainer, and the README's sizing checked against
the Nemotron deployment.

The Nemotron deployment is smaller than a default step on purpose (README, "Concurrency"), so,
unlike slime and miles, nothing tests that it runs everything the trainer keeps in flight.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
# the deployment that the README creates; deployment.example.json is only the copy that
# check_shared_recipe_files.py requires in every training recipe
DEPLOYMENT = RECIPE / "deployment.nemotron.example.json"


def run_env_example() -> dict[str, str]:
    values = {}
    for line in (RECIPE / "k8s" / "run.env.example").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("'\"")
    return values


def grpo_setting(key: str) -> int:
    """An integer that grpo_coreweave.yaml sets, such as num_generations_per_prompt."""
    config = (RECIPE / "grpo_coreweave.yaml").read_text()
    match = re.search(rf"^\s*{key}:\s*(\d+)", config, re.MULTILINE)
    assert match, f"grpo_coreweave.yaml no longer sets {key}"
    return int(match.group(1))


def stated(section: str, pattern: str) -> set[int]:
    """The numbers in section where pattern has (N): in digits, or as a word up to ten."""
    words = "zero one two three four five six seven eight nine ten".split()
    found = re.findall(pattern.replace("(N)", rf"\b(\d+|{'|'.join(words)})\b"), section)
    return {int(n) if n.isdigit() else words.index(n) for n in found}


def test_the_readme_concurrency_numbers_are_the_defaults() -> None:
    readme = (RECIPE / "README.md").read_text()
    heading = re.search(r"^## .*concurrency.*$", readme, re.IGNORECASE | re.MULTILINE)
    assert heading, "the README has no section about concurrency"
    section = " ".join(readme[heading.end() :].split("\n## ")[0].split())
    deployment = json.loads(DEPLOYMENT.read_text())
    autoscaling = deployment["autoscaling"]
    replicas, per_replica = autoscaling["min"], autoscaling["concurrency"]
    assert autoscaling["max"] == replicas, "the README counts a fixed number of replicas"
    # autoscaling.concurrency becomes vLLM's --max-num-seqs; an engineConfig max-num-seqs must agree
    max_num_seqs = int(deployment["runtime"]["engineConfig"].get("max-num-seqs", per_replica))
    assert max_num_seqs == per_replica, "engineConfig's max-num-seqs is not autoscaling.concurrency"
    default = int(run_env_example()["ROLLOUT_CONCURRENCY"])
    prompts = grpo_setting("num_prompts_per_step")
    generations = grpo_setting("num_generations_per_prompt")
    # each number goes by the words next to it, so that rewording passes and a wrong number fails
    step = {
        "(N) prompts": prompts,
        "(N) generations": generations,
        "(N) replicas? at": replicas,
        "replicas? at (N)": per_replica,
        "(N) sequences": prompts * generations,
        # a request per prompt, ROLLOUT_CONCURRENCY at a time; the replicas' slots bound each group
        "at most (N)": min(min(default, prompts) * generations, replicas * per_replica),
    }
    assert {pattern: stated(section, pattern) for pattern in step} == {
        pattern: {value} for pattern, value in step.items()
    }, "the README's default step (left) differs from the config and the deployment (right)"
    # the section may leave these out, but not state another number
    elsewhere = {"sets (N)": per_replica, "default (N)": default, "config's (N)": default}
    for pattern, value in elsewhere.items():
        assert stated(section, pattern) <= {value}, f"the README's {pattern!r} should be {value}"


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
