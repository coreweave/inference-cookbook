# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Check that a deployment's routed experts line up with its tokens.

Run it against a Mixture-of-Experts deployment started with the ``enable-return-routed-experts``
engine option, before recording rollouts or training with routing replay. It sends a few requests,
most of them the way cwtrain's sampler does, and checks what comes back:

- ``routes``: a sampled response has one row of routes for each token but the last. The rows have
  the model's layer count and top-k, zeros in dense layers, and distinct expert ids in range in
  every other layer.
- ``repeat``: the same prompt, sent twice with greedy decoding, returns rows that line up with the
  tokens: in the first MoE layer, most rows agree for the prompt and for the response tokens that
  both answers share. Deeper layers may drift, because decoding on many GPUs is not bit-exact from
  run to run and sensitive routers then change some choices.
- ``cache``: a long prompt sent twice. Did the second request reuse the prefix cache, and did it
  return the same rows?
- ``burst``: many requests at once, as a GRPO rollout sends them. Every response must carry routes
  that line up with its tokens. Some engine builds drop routes once enough sequences run at once.
- ``n2``, ``stream`` and ``chat``: what an ``n=2`` request, a streamed request and a chat request
  return.
- ``long``: the size of a long response's routes, and whether the gateway delivers them.

The endpoint comes from cwtrain's ``CWTRAIN_ROLLOUT_*`` environment variables. ``--model-path`` is
the served model's Hugging Face id or local directory, for its tokenizer and ``config.json``::

    python probe.py --model-path zai-org/GLM-5.3-Flash --out probe.json

The exit status is 1 when a check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from cwtrain.integrations.rollout import (
    CompletionSampler,
    RolloutConfig,
    RolloutError,
    decode_routed_experts,
)

CHECKS = ("routes", "repeat", "cache", "burst", "n2", "stream", "chat", "long")
ROUTES = ["routed_experts"]
#: Two cold prefills of the same prompt return rows that agree at least this much.
AGREEMENT = 0.9
#: Rows of the same tokens agree at least this much in the first MoE layer, which numerical noise
#: barely reaches. Rows one position off, as a misaligned engine returns them, agree on almost none.
ALIGNED = 0.5

MATH = "Find every real x with x^2 - 5x + 6 = 0. Reason step by step, then state the answer."
SHORT = "What is 17 times 23? Reply with the number only."
PASSAGE = (
    "The committee met on a grey morning to review the harbor survey. Each section of the report "
    "listed the depth soundings, the condition of the piers and the repairs that the engineers "
    "proposed for the coming season. "
)
LONG_ASK = "Write the integers from 1 to 50000 in order, separated by commas."


@dataclass(frozen=True)
class MoeLayout:
    """The shape of a model's routes: its decoder layers, top-k and experts."""

    layers: int
    top_k: int
    experts: int
    dense: frozenset[int]

    @property
    def dtype(self) -> np.dtype:
        """The type the engine sends: one byte for up to 256 experts, else two."""
        return np.dtype(np.uint8 if self.experts <= 256 else np.uint16)

    @property
    def moe(self) -> list[int]:
        return [i for i in range(self.layers) if i not in self.dense]


def moe_layout(config: dict[str, Any]) -> MoeLayout:
    """Read the routing layout from a Hugging Face ``config.json`` (its text config if nested)."""
    text = config.get("text_config") or config
    layers = int(text["num_hidden_layers"])
    experts = int(text.get("n_routed_experts") or text["num_experts"])
    if isinstance(text.get("mlp_layer_types"), list):  # GLM-5: "dense" or "sparse" per layer
        dense = {i for i, kind in enumerate(text["mlp_layer_types"]) if kind == "dense"}
    elif "first_k_dense_replace" in text:  # DeepSeek and GLM-4.5
        dense = set(range(int(text["first_k_dense_replace"])))
    else:  # Qwen3-MoE
        step = int(text.get("decoder_sparse_step") or 1)
        only = set(text.get("mlp_only_layers") or [])
        dense = {i for i in range(layers) if i in only or (i + 1) % step}
    return MoeLayout(layers, int(text["num_experts_per_tok"]), experts, frozenset(dense))


def structure_problems(routes: np.ndarray, layout: MoeLayout) -> list[str]:
    """What is wrong with a route array's shape, type or contents. Empty when nothing is."""
    if routes.ndim != 3 or routes.shape[1:] != (layout.layers, layout.top_k):
        return [f"shape {list(routes.shape)} is not [rows, {layout.layers}, {layout.top_k}]"]
    problems = []
    if routes.dtype != layout.dtype:
        problems.append(f"dtype {routes.dtype}, not {layout.dtype} for {layout.experts} experts")
    dense = sorted(layout.dense)
    if dense and routes[:, dense].any():
        problems.append(f"dense layers {dense} have nonzero expert ids")
    sparse = routes[:, layout.moe].astype(np.int64)
    if sparse.size:
        if int(sparse.max()) >= layout.experts:
            problems.append(f"expert id {int(sparse.max())} is not below {layout.experts}")
        ordered = np.sort(sparse, axis=-1)
        repeated = int((ordered[..., 1:] == ordered[..., :-1]).any(axis=-1).sum())
        if repeated:
            problems.append(f"{repeated} (token, layer) rows name an expert twice")
    return problems


def layer_agreement(a: np.ndarray, b: np.ndarray, layout: MoeLayout) -> np.ndarray:
    """For each MoE layer, the share of tokens that route to the same set of experts."""
    if a.shape != b.shape:
        raise ValueError(f"route shapes differ: {list(a.shape)} and {list(b.shape)}")
    a, b = a[:, layout.moe], b[:, layout.moe]
    if not a.shape[0]:
        return np.ones(len(layout.moe))
    return (np.sort(a, axis=-1) == np.sort(b, axis=-1)).all(axis=-1).mean(axis=0)


def agreement(a: np.ndarray, b: np.ndarray, layout: MoeLayout) -> float:
    """The share of (token, MoE layer) rows that route to the same set of experts."""
    per_layer = layer_agreement(a, b, layout)
    return float(per_layer.mean()) if per_layer.size else 1.0


def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def token_ids(choice: dict[str, Any]) -> list[int]:
    """The sampled token ids in a choice's ``token_id:N`` log-prob strings."""
    tokens = (choice.get("logprobs") or {}).get("tokens") or []
    return [int(t.removeprefix("token_id:")) for t in tokens]


def find(body: dict[str, Any], key: str) -> Any:
    """``key`` from wherever the response put it: its ``nvext``, its first choice, or the body."""
    choice = (body.get("choices") or [{}])[0]
    for place in (body.get("nvext"), choice.get("nvext"), choice, body):
        if isinstance(place, dict) and place.get(key) is not None:
            return place[key]
    return None


def sse_events(lines: Sequence[str]) -> list[dict[str, Any]]:
    """The JSON events of a server-sent event stream, without the closing ``[DONE]``."""
    events = []
    for line in lines:
        data = line.removeprefix("data:").strip() if line.startswith("data:") else ""
        if data and data != "[DONE]":
            events.append(json.loads(data))
    return events


@dataclass
class Check:
    name: str
    status: str  # "pass", "fail" or "info"
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


class Probe:
    """Requests against one deployment, and the checks on what they return."""

    def __init__(
        self,
        config: RolloutConfig,
        tokenizer: Any,
        layout: MoeLayout,
        *,
        long_tokens: int = 16384,
        burst: int = 64,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.config = dataclasses.replace(config, routed_experts=True)
        self.sampler = CompletionSampler(self.config, transport=transport)
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self.http = httpx.AsyncClient(
            base_url=config.base_url,
            headers=headers,
            timeout=config.timeout,
            transport=transport,
            limits=httpx.Limits(max_connections=max(100, burst)),
        )
        self.tokenizer = tokenizer
        self.layout = layout
        self.long_tokens = long_tokens
        self.burst_size = burst

    async def aclose(self) -> None:
        await self.sampler.aclose()
        await self.http.aclose()

    def prompt(self, text: str) -> list[int]:
        message = [{"role": "user", "content": text}]
        ids = self.tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        return [int(i) for i in ids]

    def payload(self, prompt: list[int], max_tokens: int, **extra: Any) -> dict[str, Any]:
        """A completions request like cwtrain's, with routes."""
        return {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
            "nvext": {"extra_fields": ROUTES},
            **extra,
        }

    async def post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        response = await self.http.post(path, json=payload)
        if response.status_code != 200:
            raise RolloutError(f"HTTP {response.status_code}: {response.text[:300]}")
        return response

    def rows_check(self, name: str, routes: np.ndarray, prompt: int, response: int) -> Check:
        rows = prompt + response - 1
        problems = structure_problems(routes, self.layout)
        if routes.shape[0] != rows:
            problems.insert(0, f"{routes.shape[0]} rows, not {prompt} + {response} - 1 = {rows}")
        details = {
            "prompt_tokens": prompt,
            "response_tokens": response,
            "shape": list(routes.shape),
        }
        if problems:
            return Check(name, "fail", "; ".join(problems), details)
        return Check(name, "pass", f"{rows} rows of {routes.dtype}, shape as expected", details)

    async def routes(self) -> Check:
        prompt = self.prompt(MATH)
        [done] = await self.sampler.sample(prompt, 1, max_tokens=512, temperature=1.0)
        return self.rows_check("routes", done.routed_experts, len(prompt), len(done.token_ids))

    async def repeat(self) -> Check:
        prompt = self.prompt(SHORT)
        first, second = [
            (await self.sampler.sample(prompt, 1, max_tokens=64, temperature=0.0))[0]
            for _ in range(2)
        ]
        p = len(prompt)
        # Row p + j routes response token j; the last response token has no row.
        shared = min(
            common_prefix(first.token_ids, second.token_ids),
            len(first.token_ids) - 1,
            len(second.token_ids) - 1,
        )
        a, b = first.routed_experts, second.routed_experts
        prompt_layers = layer_agreement(a[:p], b[:p], self.layout)
        response_layers = layer_agreement(a[p : p + shared], b[p : p + shared], self.layout)
        # Rows one position apart, as a misaligned engine would return them: about chance level.
        shifted = agreement(a[p + 1 : p + shared], b[p : p + shared - 1], self.layout)
        details = {
            "prompt_rows_agree": float(prompt_layers.mean()),
            "shared_response_tokens": shared,
            "response_rows_agree": float(response_layers.mean()),
            "response_rows_agree_by_moe_layer": [round(float(x), 4) for x in response_layers],
            "response_rows_agree_one_position_off": shifted,
            "same_response": first.token_ids == second.token_ids,
        }
        status = "pass" if min(prompt_layers[0], response_layers[0]) >= ALIGNED else "fail"
        summary = (
            f"prompt rows agree {prompt_layers.mean():.1%}; rows of {shared} shared response "
            f"tokens agree {response_layers.mean():.1%} (first MoE layer {response_layers[0]:.1%}, "
            f"last {response_layers[-1]:.1%}; one position off {shifted:.1%})"
        )
        return Check("repeat", status, summary, details)

    async def cache(self) -> Check:
        # A unique first line, so the first request cannot hit an earlier run's cache.
        text = uuid.uuid4().hex + "\n" + PASSAGE
        while len(self.prompt(text)) < 6000:
            text += PASSAGE
        prompt = self.prompt(text)
        seconds, cached, routes = [], [], []
        for _ in range(2):
            began = time.monotonic()
            body = (await self.post("/completions", self.payload(prompt, 1))).json()
            seconds.append(round(time.monotonic() - began, 3))
            details = (body.get("usage") or {}).get("prompt_tokens_details") or {}
            cached.append(details.get("cached_tokens"))
            routes.append(decode_routed_experts(find(body, "routed_experts")))
        # A cache hit returns the rows stored with the cached blocks, so those must be identical.
        # The engine recomputes the rest from the cached state, which can route differently.
        hit = int(cached[1] or 0)
        details = {"prompt_tokens": len(prompt), "seconds": seconds, "cached_tokens": cached}
        if not hit:
            same = agreement(routes[0], routes[1], self.layout)
            details["rows_agree"] = same
            summary = f"no cache hit ({seconds[0]} s, then {seconds[1]} s); rows agree {same:.1%}"
            return Check("cache", "pass" if same >= AGREEMENT else "fail", summary, details)
        kept = agreement(routes[0][:hit], routes[1][:hit], self.layout)
        rest = agreement(routes[0][hit:], routes[1][hit:], self.layout)
        details |= {"cached_rows_agree": kept, "recomputed_rows_agree": rest}
        summary = (
            f"{len(prompt)}-token prompt: {seconds[0]} s, then {seconds[1]} s with {hit} cached "
            f"tokens; cached rows agree {kept:.1%}, recomputed rows {rest:.1%}"
        )
        return Check("cache", "pass" if kept == 1.0 else "fail", summary, details)

    async def burst(self) -> Check:
        async def one(prompt: list[int]) -> str:
            body = (await self.post("/completions", self.payload(prompt, 64))).json()
            value = find(body, "routed_experts")
            if value is None:
                return "missing"
            rows = decode_routed_experts(value).shape[0]
            response = len(token_ids((body.get("choices") or [{}])[0]))
            return "ok" if rows == len(prompt) + response - 1 else "misaligned"

        # Distinct prompts, all sent at once: the engine runs them as one large batch.
        prompts = [self.prompt(f"{SHORT} ({i})") for i in range(self.burst_size)]
        results = await asyncio.gather(*(one(p) for p in prompts))
        missing, misaligned = results.count("missing"), results.count("misaligned")
        details = {"requests": len(results), "missing": missing, "misaligned": misaligned}
        if missing or misaligned:
            summary = (
                f"{missing} of {len(results)} concurrent responses had no routes, and "
                f"{misaligned} had rows that do not line up with their tokens"
            )
            return Check("burst", "fail", summary, details)
        return Check(
            "burst", "pass", f"all {len(results)} concurrent responses had routes", details
        )

    async def n2(self) -> Check:
        prompt = self.prompt(MATH)
        body = (await self.post("/completions", self.payload(prompt, 256, n=2))).json()
        choices = sorted(body.get("choices") or [], key=lambda c: c.get("index", 0))
        value = find(body, "routed_experts")
        if value is None:
            return Check("n2", "info", f"{len(choices)} choices and no routes")
        rows = decode_routed_experts(value).shape[0]
        lengths = [len(token_ids(c)) for c in choices]
        matches = [i for i, n in enumerate(lengths) if len(prompt) + n - 1 == rows]
        summary = f"one set of routes ({rows} rows) for {len(choices)} choices; matches {matches}"
        return Check("n2", "info", summary, {"rows": rows, "lengths": lengths, "matches": matches})

    async def stream(self) -> Check:
        prompt = self.prompt(MATH)
        payload = self.payload(prompt, 256, stream=True)
        async with self.http.stream("POST", "/completions", json=payload) as response:
            if response.status_code != 200:
                await response.aread()
                raise RolloutError(f"HTTP {response.status_code}: {response.text[:300]}")
            events = sse_events([line async for line in response.aiter_lines()])
        ids = [t for e in events for c in e.get("choices") or [] for t in token_ids(c)]
        carrying = [i for i, e in enumerate(events) if find(e, "routed_experts") is not None]
        if not carrying:
            return Check("stream", "fail", f"no routes in {len(events)} events")
        check = self.rows_check(
            "stream",
            decode_routed_experts(find(events[carrying[-1]], "routed_experts")),
            len(prompt),
            len(ids),
        )
        check.details |= {"events": len(events), "events_with_routes": carrying}
        check.summary += f"; routes in event(s) {carrying} of {len(events)}"
        return check

    async def chat(self) -> Check:
        messages = [{"role": "user", "content": MATH}]
        fields = [*ROUTES, "prompt_token_ids", "completion_token_ids"]
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": 256,
            "temperature": 1.0,
            "nvext": {"extra_fields": fields},
        }
        body = (await self.post("/chat/completions", payload)).json()
        value, prompt, completion = (find(body, key) for key in fields)
        if value is None or completion is None:
            missing = [k for k, v in ((ROUTES[0], value), (fields[2], completion)) if v is None]
            return Check("chat", "fail", f"the response has no {', '.join(missing)}")
        if prompt is None:
            # Some Dynamo versions return no prompt ids on chat; the usage still counts them.
            prompt_tokens = int((body.get("usage") or {}).get("prompt_tokens") or 0)
        else:
            prompt_tokens = len(prompt)
        check = self.rows_check(
            "chat", decode_routed_experts(value), prompt_tokens, len(completion)
        )
        if prompt is None:
            check.details["prompt_token_ids"] = "not returned"
            check.summary += "; no prompt_token_ids (rows checked against usage.prompt_tokens)"
            return check
        ours = self.prompt(MATH)
        same = list(prompt) == ours
        check.details["prompt_ids_match_the_trainers_render"] = same
        if not same:
            check.details["first_difference"] = common_prefix(prompt, ours)
            check.summary += "; the engine's prompt ids differ from the trainer's render"
        return check

    async def long(self) -> Check:
        prompt = self.prompt(LONG_ASK)
        payload = self.payload(prompt, self.long_tokens)
        began = time.monotonic()
        try:
            # Generate to max_tokens, so the response has a known size. ignore_eos alone leaves
            # the generation config's other stop tokens (such as an end-of-turn token) in force.
            forced = {"ignore_eos": True, "min_tokens": self.long_tokens}
            response = await self.post("/completions", {**payload, **forced})
        except RolloutError as exc:
            if "HTTP 4" not in str(exc):
                raise
            response = await self.post("/completions", payload)
        seconds = round(time.monotonic() - began, 1)
        body = response.json()
        value = find(body, "routed_experts")
        if value is None:
            return Check("long", "fail", "the response has no routes")
        routes = decode_routed_experts(value)
        response_tokens = len(token_ids(body["choices"][0]))
        details = {
            "response_tokens": response_tokens,
            "response_bytes": len(response.content),
            "route_bytes": routes.nbytes,
            "seconds": seconds,
        }
        summary = (
            f"{response_tokens} tokens: a {len(response.content) / 2**20:.1f} MiB response with "
            f"{routes.nbytes / 2**20:.1f} MiB of routes, in {seconds} s"
        )
        return Check("long", "info", summary, details)


async def run(probe: Probe, checks: Sequence[str]) -> list[Check]:
    results = []
    try:
        for name in checks:
            try:
                results.append(await getattr(probe, name)())
            except Exception as exc:  # noqa: BLE001 - report every check, even one that breaks
                results.append(Check(name, "fail", f"{type(exc).__name__}: {exc}"))
    finally:
        await probe.aclose()
    return results


def load_config(model_path: str) -> dict[str, Any]:
    local = Path(model_path) / "config.json"
    if local.is_file():
        return json.loads(local.read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(model_path, "config.json")).read_text())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a deployment's routed experts.")
    parser.add_argument("--model-path", required=True, help="Hugging Face id or local directory")
    parser.add_argument("--checks", default=",".join(CHECKS), help="comma-separated checks to run")
    parser.add_argument("--long-tokens", type=int, default=16384, help="the long check's tokens")
    parser.add_argument("--burst", type=int, default=64, help="the burst check's requests at once")
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    unknown = set(args.checks.split(",")) - set(CHECKS)
    if unknown:
        parser.error(f"unknown checks: {sorted(unknown)}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    layout = moe_layout(load_config(args.model_path))
    probe = Probe(
        RolloutConfig.from_env(),
        tokenizer,
        layout,
        long_tokens=args.long_tokens,
        burst=args.burst,
    )
    results = asyncio.run(run(probe, args.checks.split(",")))
    for result in results:
        print(f"{result.status.upper():4}  {result.name:6}  {result.summary}")
    if args.out:
        args.out.write_text(json.dumps([dataclasses.asdict(r) for r in results], indent=2) + "\n")
    return 1 if any(r.status == "fail" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
