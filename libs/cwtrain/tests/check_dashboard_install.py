# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Check first-time wheel installs beside existing trainer W&B dependencies.

Run after ``uv build``: ``python tests/check_dashboard_install.py dist/cwtrain-*.whl``.
Requires uv and package-index access; isolated from the ordinary unit-test suite.
These are packaging and offline tracking checks, not GPU trainer-image tests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

VERSIONS = """
import importlib.metadata, json
print(json.dumps({d.metadata['Name'].lower().replace('_', '-'): d.version
                  for d in importlib.metadata.distributions()}))
"""


def run_command(root: Path, env: dict[str, str], *command: str, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout or ""


def package_command(python: Path, installer: str, action: str, *args: str) -> list[str]:
    if installer == "pip":
        return [str(python), "-m", "pip", action, *args]
    return ["uv", "pip", action, "--python", str(python), *args]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    smoke = Path(__file__).with_name("dashboard_install_smoke.py").resolve()
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VIRTUAL_ENV"}}
    env["WANDB_MODE"] = "offline"

    for installer in ("pip", "uv"):
        for version in ("0.28.2", "0.30.0"):
            label = f"{installer}, wandb {version}"
            with tempfile.TemporaryDirectory(prefix="cwtrain-install-") as directory:
                root = Path(directory)
                python = root / "venv" / "bin" / "python"
                run = partial(run_command, root, env)
                packages = partial(package_command, python, installer)
                run("uv", "venv", "--seed", "--python", sys.executable, str(root / "venv"))
                # Older valid transitive versions reveal unnecessary upgrades too.
                run(
                    *packages(
                        "install",
                        f"wandb=={version}",
                        "pydantic==2.10.6",
                        "protobuf==5.29.5",
                    )
                )
                before = json.loads(run(str(python), "-c", VERSIONS, capture=True))
                assert "cwtrain" not in before, before
                run(*packages("install", str(wheel)))  # No extra or constraints.
                after = json.loads(run(str(python), "-c", VERSIONS, capture=True))
                changed = {k: (v, after.get(k)) for k, v in before.items() if after.get(k) != v}
                assert not changed, (label, changed)
                run(*packages("check"))
                # Axolotl reinstalls the image's W&B version after installing cwtrain.
                run(*packages("install", f"wandb=={version}"))
                run(*packages("check"))
                run(str(python), str(smoke))
                result = json.loads(
                    run(
                        str(python.parent / "cwtrain"),
                        "dashboard",
                        "setup",
                        "--entity",
                        "compatibility",
                        "--project",
                        "offline",
                        "--framework",
                        "miles",
                        "--dry-run",
                        capture=True,
                    )
                )
                assert result["native_panels"], result
                print(f"PASS: {label}; {len(before)} existing packages preserved", flush=True)


if __name__ == "__main__":
    main()
