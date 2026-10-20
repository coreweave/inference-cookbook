# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "reproduce.sh"
SETTINGS = {
    "CWTRAIN_ROLLOUT_BASE_URL": "https://gw.invalid/v1",
    "CWTRAIN_ROLLOUT_API_KEY": "key",
    "COREWEAVE_API_TOKEN": "token",
    "DEPLOYMENT_ID": "deployment",
}


def commands() -> list[str]:
    """The script's commands, with continued lines joined."""
    return SCRIPT.read_text().replace("\\\n", " ").splitlines()


def run(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    base = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "WORK": str(tmp_path / "work")}
    return subprocess.run(
        ["bash", str(SCRIPT)], env=base | env, capture_output=True, text=True, timeout=60
    )


def test_the_script_parses_and_is_executable() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert os.access(SCRIPT, os.X_OK)


def test_the_script_stops_before_any_step_without_its_settings(tmp_path: Path) -> None:
    result = run(tmp_path)
    assert result.returncode != 0 and "CWTRAIN_ROLLOUT_BASE_URL" in result.stderr
    assert not (tmp_path / "work").exists()


def test_a_second_run_in_the_same_work_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "work" / "recording").mkdir(parents=True)
    result = run(tmp_path, **SETTINGS)
    assert result.returncode != 0 and "set WORK to a fresh directory" in result.stderr
    assert "==" not in result.stdout  # stopped before the first step, its settings check included
    assert sorted(p.name for p in (tmp_path / "work").iterdir()) == ["recording"]


def test_the_delta_names_its_base_snapshot() -> None:
    # router_noise.py writes base_version "v1", which cwtrain can't turn into a slime identity
    [publish] = [c for c in commands() if "hotload publish" in c and "v2-delta" in c]
    assert "--previous-identity v1" in publish
