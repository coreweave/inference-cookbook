# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Launch Gym against the NeMo-RL recipe's hot-load deployment, with cwtrain's shared settings.

The recipe creates no inference resources: Gym runs on CPU and sends every request to the existing
hot-load deployment that ``CWTRAIN_ROLLOUT_BASE_URL`` names. ``--benchmark`` picks the evaluation
(BrowseComp by default), ``--model`` the deployment's served model (Nemotron 3.5 Lightning, as
``nemotron-lightning-hotload``), and ``--judge-model`` / ``--judge-base-url`` the LLM judge (GLM-5.2
on W&B Inference by default). BrowseComp's search and browse tools use You.com.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cwtrain.integrations import gym_tool_calls
from cwtrain.integrations.rollout import RolloutConfig

RECIPE = Path(__file__).resolve().parent


def _you_search():
    """This recipe's You.com client, ``you_search.py`` next to the launcher."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("you_search", RECIPE / "you_search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
DEFAULT_BENCHMARK = "browsecomp"
# The served name of Nemotron 3.5 Lightning on the NeMo-RL recipe's hot-load deployment
# (ROLLOUT_MODEL in its run.env).
DEFAULT_MODEL = "nemotron-lightning-hotload"
# The hot-load deployment's max-model-len (training/recipes/nemo-rl/deployment.nemotron.example.json).
DEFAULT_MAX_CONTEXT_TOKENS = 32768
# Generated tokens per BrowseComp turn (browsecomp.yaml's max_tokens).
TURN_TOKENS = 4096
DEFAULT_JUDGE_MODEL = "zai-org/GLM-5.2"
DEFAULT_JUDGE_BASE_URL = "https://api.inference.wandb.ai/v1"


@dataclass(frozen=True)
class Benchmark:
    gym_configs: tuple[str, ...]  # relative to the Gym checkout
    recipe_config: str  # relative to this recipe
    agent: str
    input: str
    judge: bool = False
    search: bool = False
    prepare: bool = False  # the input is downloaded by `gym eval prepare`


BENCHMARKS = {
    "browsecomp": Benchmark(
        gym_configs=("benchmarks/browsecomp/config.yaml",),
        recipe_config="browsecomp.yaml",
        agent="browsecomp_benchmark_agent",
        input="benchmarks/browsecomp/data/browsecomp_benchmark.jsonl",
        judge=True,
        search=True,
        prepare=True,
    ),
}


def _key(environ: Mapping[str, str], file: Path | None, names: tuple[str, ...]) -> str:
    if file is not None:
        return file.read_text().strip()
    return next((environ[n].strip() for n in names if environ.get(n, "").strip()), "")


def context_reset_tokens(max_context: int, reset: int | None) -> int:
    """The prompt size at which the BrowseComp agent resets its context.

    Gym's benchmark config resets at 50,000 tokens, which a smaller deployment never reaches before
    a request overflows. The default resets at half of ``max_context``, which leaves room for the
    next tool result and turn.

    Raises:
        ValueError: The reset point leaves no room for a turn of ``TURN_TOKENS``.
    """
    reset = max_context // 2 if reset is None else reset
    if not 0 < reset <= max_context - TURN_TOKENS:
        raise ValueError(
            f"--context-reset-tokens must be positive and at most --max-context-tokens minus "
            f"{TURN_TOKENS}, got {reset} with --max-context-tokens {max_context}"
        )
    return reset


def start_environment(
    environ: Mapping[str, str], args: argparse.Namespace | None = None
) -> dict[str, str]:
    """The environment for Gym: endpoint, target model, judge and search settings.

    Raises:
        ValueError: A key file is empty or unreadable, or the benchmark needs a judge or search
            key that is not set.
    """
    environ = dict(environ)
    model = getattr(args, "model", None)
    if model:
        environ["CWTRAIN_ROLLOUT_MODEL"] = model
    environ.setdefault("CWTRAIN_ROLLOUT_MODEL", DEFAULT_MODEL)
    config = RolloutConfig.from_env(environ)
    if environ.get("CWTRAIN_ROLLOUT_API_KEY_FILE") and not config.api_key:
        raise ValueError(
            "CWTRAIN_ROLLOUT_API_KEY_FILE must contain a readable, nonempty key"
        )
    env = dict(
        environ,
        CWTRAIN_ROLLOUT_BASE_URL=config.base_url,
        CWTRAIN_ROLLOUT_MODEL=config.model,
        CWTRAIN_ROLLOUT_API_KEY=config.api_key or "unused",
    )
    if args is None or args.action != "start":
        return env
    benchmark = BENCHMARKS[args.benchmark]
    if benchmark.judge:
        key = _key(environ, args.judge_api_key_file, ("CWTRAIN_JUDGE_API_KEY", "WANDB_API_KEY"))
        if not key:
            raise ValueError(
                "the judge needs a key: set WANDB_API_KEY or pass --judge-api-key-file"
            )
        env.update(
            CWTRAIN_JUDGE_API_KEY=key,
            CWTRAIN_JUDGE_MODEL=args.judge_model,
            CWTRAIN_JUDGE_BASE_URL=args.judge_base_url.rstrip("/"),
        )
    if benchmark.search:
        key = _key(environ, args.search_api_key_file, ("CWTRAIN_SEARCH_API_KEY", "YOU_API_KEY"))
        if not key:
            raise ValueError(
                "You.com search needs a key: set YOU_API_KEY or pass --search-api-key-file"
            )
        env.update(
            CWTRAIN_SEARCH_API_KEY=key,
            CWTRAIN_EXCLUDE_DOMAINS_FILE=str(RECIPE / "exclude_domains.json"),
            CWTRAIN_MAX_CONTEXT_TOKENS=str(args.max_context_tokens),
            CWTRAIN_CONTEXT_RESET_TOKENS=str(
                context_reset_tokens(args.max_context_tokens, args.context_reset_tokens)
            ),
        )
    return env


def command(args: argparse.Namespace, environ: Mapping[str, str]) -> list[str]:
    gym = str(Path(sys.executable).parent / "gym")
    benchmark = BENCHMARKS[getattr(args, "benchmark", DEFAULT_BENCHMARK)]
    configs = [arg for path in benchmark.gym_configs for arg in ("--config", path)]
    if args.action == "prepare":
        return [gym, "eval", "prepare", *configs]
    if args.action == "start":
        return [
            gym,
            "env",
            "start",
            *configs,
            "--config",
            str(RECIPE / "coreweave.yaml"),
            "--config",
            str(RECIPE / benchmark.recipe_config),
        ]
    config = RolloutConfig.from_env(environ)
    return [
        gym,
        "eval",
        "run",
        "--no-serve",
        "--agent",
        benchmark.agent,
        "--input",
        benchmark.input,
        "--output",
        str(args.output.resolve()),
        "--limit",
        str(getattr(args, "limit", 5)),
        "--num-repeats",
        "1",
        "--concurrency",
        str(config.concurrency),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start", "collect"))
    parser.add_argument("--gym-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARKS),
        default=DEFAULT_BENCHMARK,
        help=f"Evaluation to run (default: {DEFAULT_BENCHMARK}). Use the same one for every action.",
    )
    parser.add_argument(
        "--model",
        help="Served name of the target model (default: $CWTRAIN_ROLLOUT_MODEL, "
        f"else {DEFAULT_MODEL})",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument(
        "--judge-api-key-file", type=Path, help="Judge key file (default: $WANDB_API_KEY)"
    )
    parser.add_argument(
        "--search-api-key-file", type=Path, help="You.com key file (default: $YOU_API_KEY)"
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help="BrowseComp agent context limit: the deployment's max-model-len "
        f"(default: {DEFAULT_MAX_CONTEXT_TOKENS})",
    )
    parser.add_argument(
        "--context-reset-tokens",
        type=int,
        help="Prompt size at which the BrowseComp agent resets its context "
        "(default: half of --max-context-tokens)",
    )
    parser.add_argument("--limit", type=int, default=5, help="Tasks to collect (default: 5)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"results/coreweave_{args.benchmark}.jsonl")
    gym_dir = args.gym_dir.resolve()
    benchmark = BENCHMARKS[args.benchmark]
    if not all((gym_dir / path).is_file() for path in benchmark.gym_configs):
        parser.error("--gym-dir must point to the pinned NeMo Gym checkout")
    if args.action == "collect" and not (gym_dir / benchmark.input).is_file():
        parser.error(f"no {benchmark.input} in --gym-dir; run the prepare action first")
    try:
        # prepare only downloads data, so it needs no endpoint or keys
        env = dict(os.environ) if args.action == "prepare" else start_environment(os.environ, args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.action == "start":
        gym_tool_calls.install(gym_dir)
        if benchmark.search:
            _you_search().install(gym_dir)
    cmd = command(args, env)
    os.chdir(gym_dir)
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
