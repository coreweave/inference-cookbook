# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Launch Gym with cwtrain's shared endpoint, credential and concurrency settings."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from cwtrain.integrations.rollout import RolloutConfig

RECIPE = Path(__file__).resolve().parent


def start_environment(environ: Mapping[str, str]) -> dict[str, str]:
    config = RolloutConfig.from_env(environ)
    if environ.get("CWTRAIN_ROLLOUT_API_KEY_FILE") and not config.api_key:
        raise ValueError(
            "CWTRAIN_ROLLOUT_API_KEY_FILE must contain a readable, nonempty key"
        )
    return dict(
        environ,
        CWTRAIN_ROLLOUT_BASE_URL=config.base_url,
        CWTRAIN_ROLLOUT_MODEL=config.model,
        CWTRAIN_ROLLOUT_API_KEY=config.api_key or "unused",
    )


def command(args: argparse.Namespace, environ: Mapping[str, str]) -> list[str]:
    gym = str(Path(sys.executable).parent / "gym")
    if args.action == "start":
        return [
            gym,
            "env",
            "start",
            "--config",
            "resources_servers/mcqa/configs/mcqa.yaml",
            "--config",
            str(RECIPE / "coreweave.yaml"),
        ]
    config = RolloutConfig.from_env(environ)
    return [
        gym,
        "eval",
        "run",
        "--no-serve",
        "--agent",
        "mcqa_simple_agent",
        "--input",
        "resources_servers/mcqa/data/example.jsonl",
        "--output",
        str(args.output.resolve()),
        "--limit",
        "5",
        "--num-repeats",
        "1",
        "--concurrency",
        str(config.concurrency),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "collect"))
    parser.add_argument("--gym-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/coreweave_mcqa.jsonl")
    )
    args = parser.parse_args()
    gym_dir = args.gym_dir.resolve()
    if not (gym_dir / "resources_servers/mcqa/data/example.jsonl").is_file():
        parser.error("--gym-dir must point to the pinned NeMo Gym checkout")
    env = start_environment(os.environ)
    cmd = command(args, env)
    os.chdir(gym_dir)
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
