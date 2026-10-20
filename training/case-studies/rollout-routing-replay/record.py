# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Record rollouts with their exact tokens, log-probs and routed experts.

The requests are the ones cwtrain's sampler sends with ``CWTRAIN_ROLLOUT_ROUTED_EXPERTS=1``: a
token-id prompt, ``logprobs: 1``, the sampled ids as ``token_id:N`` strings, ``n=1``, and
``routed_experts`` in ``nvext.extra_fields``, at temperature 1 and top-p 1. Each set goes to
``<out>/<version>/<set>.jsonl``, one sample per line, and its routes to ``<set>.npz``, keyed by
the sample id. Sets:

- ``single``: DAPO-Math-17k prompts, one response each.
- ``two_turn``: a DAPO-Math-17k prompt, then a fixed follow-up. The second turn's prompt is the
  first turn's prompt and sampled ids, then only the follow-up rendered by the chat template.
- ``P``, ``Q``, ``R`` and so on: prompts longer than a prefix-cache block, with short greedy
  responses. Sending one set again after a hot-load shows whether the deployment serves the new
  weights' routes for a cached prompt.

The endpoint comes from cwtrain's ``CWTRAIN_ROLLOUT_*`` environment variables::

    python record.py --model-path zai-org/GLM-5.3-Flash-BF16 --out recording --version v0 \
        --sets single,two_turn,P
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from cwtrain.integrations.rollout import RolloutConfig, RolloutError, decode_routed_experts
from probe import PASSAGE, ROUTES, find, token_ids

DATASET = "zhuzilin/dapo-math-17k"
DATASET_REVISION = "2e65612930298bde4c5d58fd97b3f23a483aaff9"
FOLLOW_UP = "Check your answer and state it again."
#: Every sequence stays within this many tokens, so that a trainer with dense attention computes
#: what GLM-5.3-Flash's sparse attention layers compute: below 2,049 tokens they select every one.
MAX_LEN = 2048
ATTEMPTS = 3
RETRYABLE = {429, 500, 502, 503, 504}


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
    first_prompt: list[int], response: list[int], suffix: list[int]
) -> list[int]:
    """The first turn's exact ids, then the suffix, without doubling a stop token that opens it."""
    if response and suffix and response[-1] == suffix[0]:
        suffix = suffix[1:]
    return first_prompt + response + suffix


def long_prompt(index: int, tokens: int, tokenizer: Any) -> list[dict]:
    """A prompt of at least ``tokens`` tokens. Other indexes' prompts differ in the first line."""
    text = f"Document {index:04d}. Summarize it in one sentence.\n"
    while len(render(tokenizer, [{"role": "user", "content": text}])) < tokens:
        text += PASSAGE
    return [{"role": "user", "content": text}]


class Recorder:
    """Requests against one deployment. Each response must carry routes that line up."""

    def __init__(
        self,
        config: RolloutConfig,
        *,
        concurrency: int = 16,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = config
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self.http = httpx.AsyncClient(
            base_url=config.base_url, headers=headers, timeout=config.timeout, transport=transport
        )
        self.slots = asyncio.Semaphore(concurrency)
        #: Why a response was asked again, one entry per retry.
        self.retries: list[str] = []

    async def aclose(self) -> None:
        await self.http.aclose()

    async def complete(
        self, prompt: list[int], max_tokens: int, *, temperature: float = 1.0
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
            "nvext": {"extra_fields": ROUTES},
        }
        problem = ""
        for attempt in range(ATTEMPTS):
            if attempt:
                await asyncio.sleep(2**attempt)
            async with self.slots:
                began = time.monotonic()
                try:
                    response = await self.http.post("/completions", json=payload)
                except httpx.TransportError as exc:
                    problem = f"{type(exc).__name__}: {exc}"
                    self.retries.append(problem)
                    continue
                seconds = time.monotonic() - began
            if response.status_code in RETRYABLE:
                problem = f"HTTP {response.status_code}"
                self.retries.append(problem)
                continue
            if response.status_code != 200:
                raise RolloutError(f"HTTP {response.status_code}: {response.text[:300]}")
            body = response.json()
            choice = body["choices"][0]
            ids = token_ids(choice)
            value = find(body, "routed_experts")
            routes = decode_routed_experts(value) if value is not None else None
            if routes is None:
                problem = "no routed experts"
            elif routes.shape[0] != len(prompt) + len(ids) - 1:
                problem = f"{routes.shape[0]} rows for {len(prompt)} + {len(ids)} tokens"
            else:
                details = (body.get("usage") or {}).get("prompt_tokens_details") or {}
                return {
                    "prompt_ids": prompt,
                    "response_ids": ids,
                    "logprobs": [float(x) for x in choice["logprobs"]["token_logprobs"]],
                    "text": choice.get("text", ""),
                    "finish_reason": choice.get("finish_reason"),
                    "cached_tokens": details.get("cached_tokens"),
                    "seconds": round(seconds, 3),
                    "routes": routes,
                }
            self.retries.append(problem)
        raise RolloutError(f"{ATTEMPTS} responses without usable routes, the last: {problem}")


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


async def record_single(recorder: Recorder, tokenizer: Any, rows: list[dict], first: int, n: int):
    async def one(i: int) -> dict:
        messages = messages_of(rows[i])
        prompt = render(tokenizer, messages)
        done = await recorder.complete(prompt, MAX_LEN - len(prompt))
        return {
            "id": f"single-{i:04d}",
            "row": i,
            "messages": messages,
            "label": rows[i].get("label"),
            "turn": 1,
            **done,
        }

    return await asyncio.gather(*(one(i) for i in range(first, first + n)))


async def record_two_turn(
    recorder: Recorder, tokenizer: Any, rows: list[dict], first: int, n: int, follow_up: str
):
    async def conversation(i: int) -> list[dict]:
        messages = messages_of(rows[i])
        prompt = render(tokenizer, messages)
        turn1 = await recorder.complete(prompt, max(1, MAX_LEN // 2 - len(prompt)))
        suffix = follow_up_suffix(tokenizer, messages, follow_up)
        prompt2 = second_turn_prompt(prompt, turn1["response_ids"], suffix)
        turn2 = await recorder.complete(prompt2, max(1, MAX_LEN - len(prompt2)))
        common = {"row": i, "messages": messages, "label": rows[i].get("label")}
        return [
            {"id": f"two_turn-{i:04d}-1", **common, "turn": 1, **turn1},
            {"id": f"two_turn-{i:04d}-2", **common, "turn": 2, "follow_up": follow_up, **turn2},
        ]

    pairs = await asyncio.gather(*(conversation(i) for i in range(first, first + n)))
    return [sample for pair in pairs for sample in pair]


async def record_cache_set(recorder: Recorder, tokenizer: Any, name: str, n: int, tokens: int):
    offset = 1000 * (ord(name) - ord("A"))  # each set's prompts differ from every other set's

    async def one(i: int) -> dict:
        messages = long_prompt(offset + i, tokens, tokenizer)
        prompt = render(tokenizer, messages)
        done = await recorder.complete(prompt, 16, temperature=0.0)
        return {"id": f"{name}-{i:04d}", "messages": messages, "turn": 1, **done}

    return await asyncio.gather(*(one(i) for i in range(n)))


def write_set(directory: Path, name: str, samples: list[dict]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
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
    }


async def record(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.revision)
    rows = load_prompts(args.data, args.seed)
    recorder = Recorder(RolloutConfig.from_env(), concurrency=args.concurrency)
    summary: dict[str, Any] = {"version": args.version, "sets": {}}
    try:
        for name in args.sets.split(","):
            began = time.monotonic()
            if name == "single":
                samples = await record_single(recorder, tokenizer, rows, 0, args.single)
            elif name == "two_turn":
                samples = await record_two_turn(
                    recorder, tokenizer, rows, args.single, args.two_turn, args.follow_up
                )
            elif len(name) == 1 and name.isupper():
                samples = await record_cache_set(
                    recorder, tokenizer, name, args.cache_set, args.cache_tokens
                )
            else:
                raise SystemExit(f"unknown set {name!r}")
            stats = write_set(args.out / args.version, name, samples)
            stats["seconds"] = round(time.monotonic() - began, 1)
            summary["sets"][name] = stats
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
    path = args.out / args.version / "summary.json"
    if path.exists():  # an earlier run recorded other sets of this version
        earlier = json.loads(path.read_text())
        summary["sets"] = earlier["sets"] | summary["sets"]
        summary["retries"] = earlier["retries"] + summary["retries"]
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
