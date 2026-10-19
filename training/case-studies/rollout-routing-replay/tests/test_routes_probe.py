# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import asyncio
import base64
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from cwtrain.integrations.rollout import RolloutConfig


def load():
    spec = importlib.util.spec_from_file_location(
        "routes_probe", Path(__file__).resolve().parents[1] / "probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = load()

# GLM-5's layout in miniature: one dense layer, then MoE layers with more than 256 experts.
GLM = {
    "text_config": {
        "num_hidden_layers": 5,
        "num_experts_per_tok": 2,
        "n_routed_experts": 288,
        "first_k_dense_replace": 1,
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse", "sparse"],
    }
}
QWEN = {"num_hidden_layers": 4, "num_experts_per_tok": 2, "num_experts": 128}
LAYOUT = probe.moe_layout(GLM)
OUT = [7, 8, 9]  # the fake engine's response to everything


class Tokenizer:
    """A chat template that wraps the characters of each message between ids 1 and 2."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and not return_dict
        return [1] + [10 + ord(c) % 1000 for m in messages for c in m["content"]] + [2]


def routes_for(tokens: list[int], dense_value: int = 0) -> np.ndarray:
    """Valid routes whose row r depends only on tokens[: r + 1], like a real engine's."""
    routes = np.full((len(tokens) - 1, LAYOUT.layers, LAYOUT.top_k), dense_value, np.uint16)
    for row in range(len(tokens) - 1):
        rng = np.random.default_rng(hash(tuple(tokens[: row + 1])) % 2**32)
        for layer in LAYOUT.moe:
            routes[row, layer] = rng.choice(LAYOUT.experts, size=LAYOUT.top_k, replace=False)
    return routes


def encode(routes: np.ndarray) -> dict:
    data = base64.b64encode(routes.tobytes()).decode("ascii")
    return {"data": data, "shape": list(routes.shape), "start": 0, "dtype": str(routes.dtype)}


class Engine:
    """Answers like Dynamo with routes.

    ``extra_rows`` and ``dense_value`` break the routes. ``cache_hit`` makes a repeated prompt
    report that many cached tokens and route the recomputed rest differently, like a hybrid model
    resuming from its cached state; ``stale_cache`` changes the cached rows too.
    """

    def __init__(
        self,
        extra_rows: int = 0,
        dense_value: int = 0,
        chat_prompt_ids: bool = True,
        cache_hit: int = 0,
        stale_cache: bool = False,
    ):
        self.extra_rows, self.dense_value = extra_rows, dense_value
        self.chat_prompt_ids, self.cache_hit, self.stale_cache = (
            chat_prompt_ids,
            cache_hit,
            stale_cache,
        )
        self.requests: list[dict] = []
        self.seen: set[tuple[int, ...]] = set()

    def routes(self, tokens: list[int], cached: int = 0) -> dict:
        tokens = tokens + list(range(100, 100 + self.extra_rows))
        routes = routes_for(tokens, self.dense_value)
        if cached:
            moe = LAYOUT.moe
            start = 0 if self.stale_cache else cached
            routes[start:, moe] = (routes[start:, moe] + 1) % LAYOUT.experts
        return encode(routes)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if request.url.path.endswith("/chat/completions"):
            prompt = Tokenizer().apply_chat_template(body["messages"], True, True, False)
            nvext = {"routed_experts": self.routes(prompt + OUT), "completion_token_ids": OUT}
            if self.chat_prompt_ids:
                nvext["prompt_token_ids"] = prompt
            message = {"role": "assistant", "content": "x"}
            usage = {"prompt_tokens": len(prompt)}
            choices = [{"index": 0, "message": message}]
            return httpx.Response(200, json={"choices": choices, "nvext": nvext, "usage": usage})
        prompt = body["prompt"]
        cached = self.cache_hit if tuple(prompt) in self.seen else 0
        self.seen.add(tuple(prompt))
        tokens = [f"token_id:{t}" for t in OUT]
        choice = {
            "index": 0,
            "text": "x",
            "finish_reason": "length",
            "logprobs": {"tokens": tokens, "token_logprobs": [-0.5] * len(OUT)},
        }
        nvext = {"routed_experts": self.routes(prompt + OUT, cached)}
        if body.get("stream"):
            events = [
                {"choices": [{"index": 0, "logprobs": {"tokens": [t], "token_logprobs": [-0.5]}}]}
                for t in tokens
            ]
            events[-1]["nvext"] = nvext  # routes come on the last event only
            text = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        choices = [{**choice, "index": i} for i in range(body.get("n", 1))]
        usage = {"prompt_tokens": len(prompt), "prompt_tokens_details": {"cached_tokens": cached}}
        return httpx.Response(200, json={"choices": choices, "nvext": nvext, "usage": usage})


def run_probe(engine: Engine, checks=probe.CHECKS) -> dict:
    config = RolloutConfig(base_url="https://gw.invalid/v1", model="m")
    p = probe.Probe(config, Tokenizer(), LAYOUT, transport=httpx.MockTransport(engine))
    return {check.name: check for check in asyncio.run(probe.run(p, checks))}


def test_layouts_of_glm_deepseek_and_qwen_configs() -> None:
    assert LAYOUT.dense == {0} and LAYOUT.moe == [1, 2, 3, 4]
    assert LAYOUT.experts == 288 and LAYOUT.dtype == np.uint16
    deepseek = probe.moe_layout({**QWEN, "n_routed_experts": 64, "first_k_dense_replace": 2})
    assert deepseek.dense == {0, 1} and deepseek.dtype == np.uint8
    assert probe.moe_layout(QWEN).dense == frozenset()
    every_other = probe.moe_layout({**QWEN, "decoder_sparse_step": 2, "mlp_only_layers": [3]})
    assert every_other.dense == {0, 2, 3}


def test_a_healthy_deployment_passes_every_check() -> None:
    engine = Engine()
    results = run_probe(engine)
    assert {name: r.status for name, r in results.items()} == {
        "routes": "pass",
        "repeat": "pass",
        "cache": "pass",
        "n2": "info",
        "stream": "pass",
        "chat": "pass",
        "long": "info",
    }
    assert results["n2"].details["matches"] == [0, 1]
    assert results["chat"].details["prompt_ids_match_the_trainers_render"] is True
    assert results["stream"].details["events_with_routes"] == [2]
    # requests go out like cwtrain's, with routes, and the long one runs to max_tokens
    assert all(r["nvext"]["extra_fields"][0] == "routed_experts" for r in engine.requests)
    assert engine.requests[-1]["ignore_eos"] is True
    assert engine.requests[-1]["min_tokens"] == engine.requests[-1]["max_tokens"] == 16384
    assert results["cache"].details["prompt_tokens"] >= 6000


def test_cached_rows_must_match_while_recomputed_rows_may_differ() -> None:
    results = run_probe(Engine(cache_hit=512), ("cache",))
    cache = results["cache"]
    assert cache.status == "pass" and cache.details["cached_rows_agree"] == 1.0
    assert cache.details["recomputed_rows_agree"] == 0.0  # every recomputed row moved
    stale = run_probe(Engine(cache_hit=512, stale_cache=True), ("cache",))["cache"]
    assert stale.status == "fail" and stale.details["cached_rows_agree"] == 0.0


def test_chat_without_prompt_ids_checks_rows_against_the_usage() -> None:
    chat = run_probe(Engine(chat_prompt_ids=False), ("chat",))["chat"]
    assert chat.status == "pass" and "no prompt_token_ids" in chat.summary
    broken = run_probe(Engine(chat_prompt_ids=False, extra_rows=1), ("chat",))["chat"]
    assert broken.status == "fail"


def test_rows_that_do_not_line_up_with_the_tokens_fail() -> None:
    results = run_probe(Engine(extra_rows=1), ("routes", "stream", "chat"))
    assert all(r.status == "fail" for r in results.values())
    assert "rows" in results["routes"].summary and "not" in results["stream"].summary


def test_nonzero_dense_layers_fail() -> None:
    results = run_probe(Engine(dense_value=5), ("routes",))
    assert results["routes"].status == "fail" and "dense layers [0]" in results["routes"].summary


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda r: r.astype(np.uint8), "dtype uint8"),
        (lambda r: r[:, :3], "shape"),
        (lambda r: np.where(r == r[0, 1, 0], 300, r).astype(np.uint16), "not below 288"),
        (lambda r: np.repeat(r[..., :1], 2, axis=-1), "name an expert twice"),
    ],
)
def test_structure_problems(change, problem) -> None:
    routes = routes_for(list(range(12)))
    assert probe.structure_problems(routes, LAYOUT) == []
    assert problem in "; ".join(probe.structure_problems(change(routes), LAYOUT))


def test_agreement_ignores_expert_order_and_dense_layers() -> None:
    a = routes_for(list(range(11)))
    b = a[..., ::-1].copy()  # the same experts in the other order
    assert probe.agreement(a, b, LAYOUT) == 1.0
    b[0, 1] = (b[0, 1] + 1) % LAYOUT.experts
    assert probe.agreement(a, b, LAYOUT) == pytest.approx(1 - 1 / (10 * 4))
    with pytest.raises(ValueError, match="shapes differ"):
        probe.agreement(a, a[:5], LAYOUT)


def test_find_looks_in_nvext_then_the_first_choice_then_the_body() -> None:
    body = {"nvext": {"a": 1}, "choices": [{"b": 2, "nvext": {"c": 3}}], "d": 4}
    assert [probe.find(body, k) for k in "abcde"] == [1, 2, 3, 4, None]


def test_sse_events_skip_comments_blank_lines_and_done() -> None:
    lines = [": keep-alive", "", 'data: {"x": 1}', 'data:{"y": 2}', "data: [DONE]"]
    assert probe.sse_events(lines) == [{"x": 1}, {"y": 2}]
