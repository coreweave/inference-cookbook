# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "reproduce.sh"


def test_the_script_parses_and_is_executable() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert os.access(SCRIPT, os.X_OK)


def test_the_script_stops_before_any_step_without_its_settings(tmp_path: Path) -> None:
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "WORK": str(tmp_path / "work")}
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "CWTRAIN_ROLLOUT_BASE_URL" in result.stderr
    assert not (tmp_path / "work").exists()
