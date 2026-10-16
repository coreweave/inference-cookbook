# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Rollout concurrency in the recipe: the config's default, the README sentence that names it, and
run-trainer.sh, which leaves an empty value to the config."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]


def config_default() -> str:
    config = (RECIPE / "grpo_coreweave.yaml").read_text()
    match = re.search(r"concurrency: \$\{oc\.env:CWTRAIN_ROLLOUT_CONCURRENCY,(\d+)\}", config)
    assert match, "coreweave_cfg.concurrency no longer reads CWTRAIN_ROLLOUT_CONCURRENCY"
    return match.group(1)


def test_the_readme_names_the_config_default() -> None:
    readme = " ".join((RECIPE / "README.md").read_text().split())
    section = readme[readme.index("## Concurrency") :].split(" ## ")[0]
    assert f"An empty value keeps the config's {config_default()}." in section


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
