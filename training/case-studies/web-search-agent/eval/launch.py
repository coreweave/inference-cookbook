# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Evaluate the case study's policy with Gym on the NeMo-RL hot-load deployment, with cwtrain's
shared settings.

It creates no inference resources: Gym runs on CPU and sends every request to the existing
hot-load deployment that ``CWTRAIN_ROLLOUT_BASE_URL`` names. ``--benchmark`` picks the evaluation
(BrowseComp by default), ``--model`` the deployment's served model (Nemotron 3.5 Lightning, as
``nemotron-lightning-hotload``), and ``--judge-model`` / ``--judge-base-url`` the LLM judge (GLM-5.2
on W&B Inference by default). BrowseComp's search and browse tools use You.com.

``report`` prints each task's reward and tool calls and the mean reward. With ``WANDB_PROJECT``
set, it also logs them as a W&B run (job type ``eval``): the scores as the run's summary, and a
table with one row per task.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cwtrain.integrations import gym_tool_calls
from cwtrain.integrations.rollout import RolloutConfig

RECIPE = Path(__file__).resolve().parent


def _you_search():
    """The You.com client, ``you_search.py`` next to the launcher."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("you_search", RECIPE / "you_search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEFAULT_BENCHMARK = "browsecomp"
# The served name of Nemotron 3.5 Lightning on the NeMo-RL recipe's hot-load deployment
# (ROLLOUT_MODEL in its run.env).
DEFAULT_MODEL = "nemotron-lightning-hotload"
# The hot-load deployment's max-model-len (the case study's deployment.example.json).
DEFAULT_MAX_CONTEXT_TOKENS = 32768
# Tool-calling turns per BrowseComp task. Gym's benchmark allows 400; 50 bounds each task's time.
DEFAULT_MAX_STEPS = 50
# Generated tokens per BrowseComp turn (browsecomp.yaml's max_tokens).
TURN_TOKENS = 4096
DEFAULT_JUDGE_MODEL = "zai-org/GLM-5.2"
DEFAULT_JUDGE_BASE_URL = "https://api.inference.wandb.ai/v1"
# Characters of each task's final answer that the W&B table keeps.
ANSWER_CHARS = 500
# Gym's row keys for the task (and its repeat) a trajectory answers.
TASK_INDEX = "_ng_task_index"


@dataclass(frozen=True)
class Benchmark:
    gym_configs: tuple[str, ...]  # relative to the Gym checkout
    recipe_config: str  # relative to this directory
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
            CWTRAIN_MAX_STEPS=str(args.max_steps),
            CWTRAIN_CONTEXT_RESET_TOKENS=str(
                context_reset_tokens(args.max_context_tokens, args.context_reset_tokens)
            ),
        )
    return env


def wandb_settings(environ: Mapping[str, str]) -> dict[str, str] | None:
    """The W&B run to log the scores to, or ``None`` when ``WANDB_PROJECT`` is unset.

    Raises:
        ValueError: ``WANDB_PROJECT`` is set, but ``WANDB_API_KEY`` is not and W&B is not in
            offline mode.
    """
    project = environ.get("WANDB_PROJECT", "").strip()
    if not project:
        return None
    offline = environ.get("WANDB_MODE", "").strip() in ("offline", "disabled")
    if not offline and not environ.get("WANDB_API_KEY", "").strip():
        raise ValueError(
            "WANDB_PROJECT is set but WANDB_API_KEY is not: create the Secret that "
            "WANDB_API_KEY_SECRET names, or leave WANDB_PROJECT empty"
        )
    settings = {
        "project": project,
        "entity": environ.get("WANDB_ENTITY", "").strip(),
        "group": environ.get("WANDB_RUN_GROUP", "").strip(),
        "name": environ.get("EVAL_NAME", "").strip(),
    }
    return {k: v for k, v in settings.items() if v}


def final_answer(output: Sequence[dict[str, Any]]) -> str:
    """The text of the trajectory's last message, which holds BrowseComp's ``Exact Answer:``."""
    for item in reversed(output):
        if item.get("type") == "message":
            parts = item.get("content") or []
            if isinstance(parts, str):
                return parts
            return "".join(p.get("text") or "" for p in parts if isinstance(p, dict))
    return ""


def tool_turns(output: Sequence[dict[str, Any]]) -> int:
    """The model turns in a trajectory that called tools.

    The agent runs all the calls of a model response before it asks the model again, so a turn's
    calls are the ``function_call`` items before the next ``function_call_output`` items.
    """
    turns, calling = 0, False
    for item in output:
        kind = item.get("type")
        if kind == "function_call_output":
            calling = False
        elif kind == "function_call" and not calling:
            turns, calling = turns + 1, True
    return turns


def scores(rows: Sequence[dict[str, Any]], max_steps: int) -> tuple[list[dict[str, Any]], dict]:
    """One record per task, and the summary of all of them.

    A task reached the turn cap when it called tools in ``max_steps`` model turns; it is graded on
    the answer it gave then. The agent's cap also counts turns without a tool call, so a task that
    had such turns can reach it uncounted.
    """
    tasks = []
    for position, row in enumerate(rows):
        output = (row.get("response") or {}).get("output") or []
        calls = sum(item.get("type") == "function_call" for item in output)
        turns = tool_turns(output)
        tasks.append(
            {
                "task": row.get(TASK_INDEX, position),
                "reward": float(row.get("reward") or 0.0),
                "tool_calls": calls,
                "turns": turns,
                "reached_turn_cap": turns >= max_steps,
                "answer": final_answer(output)[:ANSWER_CHARS],
            }
        )
    count = len(tasks)
    summary = {
        "tasks": count,
        "mean_reward": sum(t["reward"] for t in tasks) / max(count, 1),
        "correct": sum(t["reward"] >= 1.0 for t in tasks),
        "tasks_at_turn_cap": sum(t["reached_turn_cap"] for t in tasks),
        "mean_tool_calls": sum(t["tool_calls"] for t in tasks) / max(count, 1),
    }
    return tasks, summary


def report(
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    environ: Mapping[str, str],
    *,
    out: Any = None,
) -> dict:
    """Print the scores, and log them to W&B when ``WANDB_PROJECT`` is set. Returns the summary."""
    out = out or sys.stdout
    tasks, summary = scores(rows, args.max_steps)
    for task in tasks:
        print(f"reward={task['reward']} tool_calls={task['tool_calls']}", file=out)
    print(f"tasks={summary['tasks']} mean_reward={summary['mean_reward']:.3f}", file=out)
    settings = wandb_settings(environ)
    if settings is None:
        return summary
    # Empty W&B variables from an unfilled eval.env would override the run's own settings.
    for name in ("WANDB_ENTITY", "WANDB_RUN_GROUP", "WANDB_NAME", "WANDB_PROJECT"):
        if name in os.environ and not os.environ[name].strip():
            del os.environ[name]
    import wandb

    run = wandb.init(
        job_type="eval",
        config={
            "benchmark": args.benchmark,
            "model": environ.get("CWTRAIN_ROLLOUT_MODEL") or args.model or DEFAULT_MODEL,
            "identity": environ.get("EVAL_IDENTITY", "").strip() or None,
            "limit": args.limit,
            "max_steps": args.max_steps,
            "max_context_tokens": args.max_context_tokens,
            "judge_model": args.judge_model,
            "gym_ref": environ.get("GYM_REF", "").strip() or None,
        },
        **settings,
    )
    try:
        columns = ["task", "reward", "tool_calls", "turns", "reached_turn_cap", "answer"]
        table = wandb.Table(columns=columns, data=[[t[c] for c in columns] for t in tasks])
        run.log({**{f"eval/{k}": v for k, v in summary.items()}, "eval/task_results": table})
        run.summary.update({f"eval/{k}": v for k, v in summary.items()})
    finally:
        run.finish()
    return summary


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
    parser.add_argument("action", choices=("prepare", "start", "collect", "report"))
    parser.add_argument("--gym-dir", type=Path, help="The pinned NeMo Gym checkout (not report)")
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Tool-calling turns per BrowseComp task (default: {DEFAULT_MAX_STEPS}; Gym's is 400)",
    )
    parser.add_argument("--limit", type=int, default=5, help="Tasks to collect (default: 5)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"results/coreweave_{args.benchmark}.jsonl")
    if args.action == "report":
        try:
            rows = [json.loads(line) for line in args.output.read_text().splitlines() if line]
            report(rows, args, os.environ)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return
    if args.gym_dir is None:
        parser.error(f"{args.action} needs --gym-dir")
    gym_dir = args.gym_dir.resolve()
    benchmark = BENCHMARKS[args.benchmark]
    if not all((gym_dir / path).is_file() for path in benchmark.gym_configs):
        parser.error("--gym-dir must point to the pinned NeMo Gym checkout")
    if args.action == "collect" and not (gym_dir / benchmark.input).is_file():
        parser.error(f"no {benchmark.input} in --gym-dir; run the prepare action first")
    try:
        # prepare only downloads data, so it needs no endpoint or keys
        env = dict(os.environ) if args.action == "prepare" else start_environment(os.environ, args)
        wandb_settings(os.environ)  # a W&B run without its key fails now, not after the eval
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
