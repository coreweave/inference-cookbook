# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Record rollouts with their exact tokens, log-probs and routed experts.

Every request goes through cwtrain's sampler with routed experts on, so it is the request a trainer
sends: a token-id prompt, one choice, the sampled ids back as ``token_id:N`` strings and
``routed_experts`` in ``nvext.extra_fields``, at top-p 1. Each set goes to
``<out>/<version>/<set>.jsonl``, one sample per line, and its routes to ``<set>.npz``, keyed by
the sample id. Sets:

- ``single``: DAPO-Math-17k prompts, one response each, at temperature 1.
- ``two_turn``: a DAPO-Math-17k prompt, then a fixed follow-up. The second turn's prompt is the
  first turn's prompt and sampled ids, then only the follow-up rendered by the chat template. A
  stop token that ended the first turn is left out, because the follow-up starts the next turn.
  First turns cut off at their token budget are kept on purpose: the study measures how a text
  round trip handles them, and each sample keeps its ``finish_reason``.
- ``P``, ``Q``, ``R`` and so on: prompts longer than a prefix-cache block, with short greedy
  responses. Sending one set again after a hot-load shows whether the deployment serves the new
  weights' routes for a cached prompt.

A request that fails for good does not discard its set: the samples that came back are written,
the failures go to ``summary.json``, and the command exits with status 1. The endpoint comes from
cwtrain's ``CWTRAIN_ROLLOUT_*`` environment variables::

    python record.py --model-path zai-org/GLM-5.3-Flash-BF16 --out recording --version v0 \
        --sets single,two_turn,P
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import sys
import time
from collections.abc import Awaitable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from cwtrain.integrations.rollout import CompletionSampler, RolloutConfig, RolloutError
from probe import PASSAGE

DATASET = "zhuzilin/dapo-math-17k"
DATASET_REVISION = "2e65612930298bde4c5d58fd97b3f23a483aaff9"
FOLLOW_UP = "Check your answer and state it again."
#: Every sequence stays within this many tokens, so that a trainer with dense attention computes
#: what GLM-5.3-Flash's sparse attention layers compute: up to 2,048 tokens they select every one.
MAX_LEN = 2048
#: The second turn gets at least this many tokens to answer in, or the conversation is skipped.
TURN2_MIN = 64
#: Attempts for a response whose routes are missing or don't line up. The sampler itself retries
#: busy and failed requests.
ATTEMPTS = 3


def follow_up_suffix(tokenizer: Any, messages: list[dict], follow_up: str) -> list[int]:
    """The ids the chat template puts after an assistant turn: the follow-up and the next prompt."""
    turn = [*messages, {"role": "assistant", "content": "PLACEHOLDER"}]
    head = render(tokenizer, turn, generation_prompt=False)
    full = render(tokenizer, [*turn, {"role": "user", "content": follow_up}])
    if full[: len(head)] != head:
        raise ValueError(
            "the chat template renders an assistant turn differently before a user turn"
        )
    return full[len(head) :]


def render(tokenizer: Any, messages: list[dict], *, generation_prompt: bool = True) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation_prompt, return_dict=False
    )
    return [int(i) for i in ids]


def second_turn_prompt(
    first_prompt: list[int], response: list[int], suffix: list[int], stop_ids: Collection[int]
) -> list[int]:
    """The first turn's exact ids, then the suffix, which starts the next turn.

    A stop token that ended the first turn is left out, whichever it is: GLM's ``<|user|>`` would
    otherwise come twice, and its ``<|endoftext|>`` would sit in front of the next user turn.
    """
    if response and response[-1] in stop_ids:
        response = response[:-1]
    return first_prompt + response + suffix


def model_stop_ids(model_path: str, revision: str | None, tokenizer: Any) -> set[int]:
    """The model's stop token ids: its generation config's, else the tokenizer's end of text."""
    from transformers import GenerationConfig

    try:
        ids = GenerationConfig.from_pretrained(model_path, revision=revision).eos_token_id
    except OSError:  # no generation_config.json
        ids = None
    if ids is None:
        ids = tokenizer.eos_token_id
    if ids is None:
        return set()
    return {int(ids)} if isinstance(ids, int) else {int(i) for i in ids}


def long_prompt(index: int, tokens: int, tokenizer: Any) -> list[dict]:
    """A prompt of at least ``tokens`` tokens. Other indexes' prompts differ in the first line."""
    text = f"Document {index:04d}. Summarize it in one sentence.\n"
    while len(render(tokenizer, [{"role": "user", "content": text}])) < tokens:
        text += PASSAGE
    return [{"role": "user", "content": text}]


def route_problem(error: RolloutError) -> str | None:
    """Why a response's routes were unusable, if that is what ``error`` is about."""
    message = str(error)
    if "returned no routed experts" in message:
        return "no routed experts"
    if "routed experts have" in message:
        return message
    return None


class Recorder:
    """cwtrain's sampler with routed experts on. Each response must carry routes that line up."""

    def __init__(
        self,
        config: RolloutConfig,
        *,
        concurrency: int = 16,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        config = dataclasses.replace(config, routed_experts=True, concurrency=concurrency)
        self.sampler = CompletionSampler(config, transport=transport)
        #: Why a response was asked again, one entry per retry.
        self.retries: list[str] = []

    async def aclose(self) -> None:
        await self.sampler.aclose()

    async def complete(
        self, prompt: list[int], max_tokens: int, *, temperature: float = 1.0
    ) -> dict[str, Any]:
        for attempt in range(1, ATTEMPTS + 1):
            try:
                [done] = await self.sampler.sample(
                    prompt, 1, max_tokens=max_tokens, temperature=temperature, top_p=1.0
                )
            except RolloutError as exc:
                problem = route_problem(exc)
                if problem is None or attempt == ATTEMPTS:
                    raise
                self.retries.append(problem)
                continue
            assert done.token_ids is not None and done.token_logprobs is not None
            return {
                "prompt_ids": list(prompt),
                "response_ids": list(done.token_ids),
                "logprobs": [float(x) for x in done.token_logprobs],
                "text": done.text,
                "finish_reason": done.finish_reason,
                "cached_tokens": done.cached_tokens,
                "routes": done.routed_experts,
            }
        raise AssertionError("unreachable")


@dataclass
class SetResult:
    """The samples a set got, and why the rest are missing."""

    samples: list[dict] = field(default_factory=list)
    #: One entry per request that failed for good: ``"<sample id>: <error>"``.
    failures: list[str] = field(default_factory=list)
    #: Prompts or conversations left out because they don't fit in ``MAX_LEN``.
    skipped: int = 0


async def settle(jobs: dict[str, Awaitable[list[dict] | None]]) -> SetResult:
    """Run every job; keep what succeeded, and count what was skipped or failed."""
    result = SetResult()
    outcomes = await asyncio.gather(*jobs.values(), return_exceptions=True)
    for sid, outcome in zip(jobs, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            if not isinstance(outcome, Exception):  # cancelled or interrupted: stop for real
                raise outcome
            result.failures.append(f"{sid}: {type(outcome).__name__}: {outcome}")
        elif outcome is None:
            result.skipped += 1
        else:
            result.samples.extend(outcome)
    return result


def load_prompts(path: str | None, seed: int) -> list[dict]:
    """DAPO-Math-17k rows (``prompt``, ``label``) in a seeded order."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            DATASET, "dapo-math-17k.jsonl", repo_type="dataset", revision=DATASET_REVISION
        )
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)
    return rows


def messages_of(row: dict) -> list[dict]:
    prompt = row["prompt"]
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [{"role": m["role"], "content": m["content"]} for m in prompt]


async def record_single(
    recorder: Recorder, tokenizer: Any, rows: list[dict], first: int, n: int
) -> SetResult:
    async def one(i: int) -> list[dict] | None:
        messages = messages_of(rows[i])
        prompt = render(tokenizer, messages)
        if len(prompt) >= MAX_LEN:
            return None
        done = await recorder.complete(prompt, MAX_LEN - len(prompt))
        sample = {"row": i, "messages": messages, "label": rows[i].get("label"), "turn": 1}
        return [{"id": f"single-{i:04d}", **sample, **done}]

    return await settle({f"single-{i:04d}": one(i) for i in range(first, first + n)})


async def record_two_turn(
    recorder: Recorder,
    tokenizer: Any,
    rows: list[dict],
    first: int,
    n: int,
    follow_up: str,
    stop_ids: Collection[int],
) -> SetResult:
    async def conversation(i: int) -> list[dict] | None:
        messages = messages_of(rows[i])
        prompt = render(tokenizer, messages)
        suffix = follow_up_suffix(tokenizer, messages, follow_up)
        # Turn 1 leaves room for the follow-up and for an answer of at least TURN2_MIN tokens.
        budget = min(MAX_LEN // 2, MAX_LEN - len(suffix) - TURN2_MIN) - len(prompt)
        if budget <= 0:
            return None
        turn1 = await recorder.complete(prompt, budget)
        prompt2 = second_turn_prompt(prompt, turn1["response_ids"], suffix, stop_ids)
        if MAX_LEN - len(prompt2) < TURN2_MIN:
            return None
        turn2 = await recorder.complete(prompt2, MAX_LEN - len(prompt2))
        common = {"row": i, "messages": messages, "label": rows[i].get("label")}
        return [
            {"id": f"two_turn-{i:04d}-1", **common, "turn": 1, **turn1},
            {"id": f"two_turn-{i:04d}-2", **common, "turn": 2, "follow_up": follow_up, **turn2},
        ]

    return await settle({f"two_turn-{i:04d}": conversation(i) for i in range(first, first + n)})


async def record_cache_set(
    recorder: Recorder, tokenizer: Any, name: str, n: int, tokens: int
) -> SetResult:
    offset = 1000 * (ord(name) - ord("A"))  # each set's prompts differ from every other set's

    async def one(i: int) -> list[dict]:
        messages = long_prompt(offset + i, tokens, tokenizer)
        prompt = render(tokenizer, messages)
        done = await recorder.complete(prompt, 16, temperature=0.0)
        return [{"id": f"{name}-{i:04d}", "messages": messages, "turn": 1, **done}]

    return await settle({f"{name}-{i:04d}": one(i) for i in range(n)})


def write_set(directory: Path, name: str, result: SetResult) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    samples = result.samples
    routes = {s["id"]: s.pop("routes") for s in samples}
    with (directory / f"{name}.jsonl").open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    np.savez_compressed(directory / f"{name}.npz", **routes)
    tokens = sum(len(s["prompt_ids"]) + len(s["response_ids"]) for s in samples)
    return {
        "samples": len(samples),
        "tokens": tokens,
        "truncated": sum(s["finish_reason"] == "length" for s in samples),
        "cached_tokens": sum(int(s["cached_tokens"] or 0) for s in samples),
        "route_bytes": sum(r.nbytes for r in routes.values()),
        "skipped": result.skipped,
        "failed": len(result.failures),
    }


async def record(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.revision)
    stop_ids = model_stop_ids(args.model_path, args.revision, tokenizer)
    rows = load_prompts(args.data, args.seed)
    recorder = Recorder(RolloutConfig.from_env(), concurrency=args.concurrency)
    summary: dict[str, Any] = {"version": args.version, "sets": {}, "failures": {}}
    try:
        for name in args.sets.split(","):
            began = time.monotonic()
            if name == "single":
                result = await record_single(recorder, tokenizer, rows, 0, args.single)
            elif name == "two_turn":
                result = await record_two_turn(
                    recorder, tokenizer, rows, args.single, args.two_turn, args.follow_up, stop_ids
                )
            elif len(name) == 1 and name.isupper():
                result = await record_cache_set(
                    recorder, tokenizer, name, args.cache_set, args.cache_tokens
                )
            else:
                raise SystemExit(f"unknown set {name!r}")
            stats = write_set(args.out / args.version, name, result)
            stats["seconds"] = round(time.monotonic() - began, 1)
            summary["sets"][name] = stats
            if result.failures:
                summary["failures"][name] = result.failures
            print(json.dumps({name: stats}), flush=True)
    finally:
        await recorder.aclose()
    summary["retries"] = recorder.retries
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record rollouts with tokens and routes.")
    parser.add_argument("--model-path", required=True, help="Hugging Face id or local directory")
    parser.add_argument("--revision", help="the model's Hugging Face revision, for its tokenizer")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True, help="the weights the deployment serves")
    parser.add_argument("--sets", default="single,two_turn,P", help="comma-separated sets")
    parser.add_argument("--data", help="a DAPO-Math-17k jsonl; downloaded when omitted")
    parser.add_argument("--seed", type=int, default=0, help="the order of the prompts")
    parser.add_argument("--single", type=int, default=128, help="single-turn samples")
    parser.add_argument("--two-turn", type=int, default=32, help="two-turn conversations")
    parser.add_argument("--cache-set", type=int, default=16, help="prompts per cache set")
    parser.add_argument("--cache-tokens", type=int, default=1536, help="tokens per cache prompt")
    parser.add_argument("--follow-up", default=FOLLOW_UP)
    parser.add_argument("--concurrency", type=int, default=16, help="requests in flight")
    args = parser.parse_args(argv)
    summary = asyncio.run(record(args))
    failed = sum(len(reasons) for reasons in summary["failures"].values())
    path = args.out / args.version / "summary.json"
    if path.exists():  # an earlier run recorded other sets of this version
        earlier = json.loads(path.read_text())
        retries = earlier.get("retries")
        summary["sets"] = earlier.get("sets", {}) | summary["sets"]
        summary["failures"] = earlier.get("failures", {}) | summary["failures"]
        summary["retries"] = (retries if isinstance(retries, list) else []) + summary["retries"]
    path.write_text(json.dumps(summary, indent=2) + "\n")
    if failed:
        print(f"{failed} requests failed; see {path}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
