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
OUT = [7, 8, 9]  # the fake engine's answer, unless a request forces its length


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
    """Answers like Dynamo with routes, in the response's top-level ``nvext``.

    ``extra_rows`` and ``dense_value`` break the routes. ``cache_unit`` turns on a prefix cache
    that matches prompts in units of that many tokens; rows after a hit are recomputed and route
    differently, like a hybrid model resuming from its cached state. ``stale_cache`` changes the
    cached rows too, ``zero_recomputed`` zeroes the recomputed ones, and ``cache_block`` zeroes the
    cached rows of a block that a hit ends inside, as older vLLM builds did. ``drop_every`` leaves
    the routes out of every n-th completion, like an engine that loses them under load.
    ``drift_from`` changes the response rows of every completion from that MoE layer on, like
    decoding that is not bit-exact. ``shift_decode`` returns each response row one position late.
    ``routes_in_choice`` puts ``nvext`` in the choice. ``unavailable`` answers that many first
    requests with 503, and ``refuse_every`` answers every n-th with 400.
    """

    def __init__(
        self,
        extra_rows: int = 0,
        dense_value: int = 0,
        chat_prompt_ids: bool = True,
        cache_unit: int = 0,
        cache_block: int = 0,
        stale_cache: bool = False,
        zero_recomputed: bool = False,
        drop_every: int = 0,
        drift_from: int | None = None,
        shift_decode: bool = False,
        routes_in_choice: bool = False,
        unavailable: int = 0,
        refuse_every: int = 0,
    ):
        self.extra_rows, self.dense_value = extra_rows, dense_value
        self.chat_prompt_ids = chat_prompt_ids
        self.cache_unit, self.cache_block = cache_unit, cache_block
        self.stale_cache, self.zero_recomputed = stale_cache, zero_recomputed
        self.drop_every, self.drift_from = drop_every, drift_from
        self.shift_decode, self.routes_in_choice = shift_decode, routes_in_choice
        self.unavailable, self.refuse_every = unavailable, refuse_every
        self.requests: list[dict] = []
        self.seen: list[list[int]] = []

    def cached(self, prompt: list[int]) -> int:
        if not self.cache_unit:
            return 0
        shared = max((probe.common_prefix(prompt, seen) for seen in self.seen), default=0)
        shared = min(shared, len(prompt) - 1)  # the last prompt token is always computed
        return shared // self.cache_unit * self.cache_unit

    def routes(self, tokens: list[int], cached: int = 0, prompt: int | None = None) -> dict:
        tokens = tokens + list(range(100, 100 + self.extra_rows))
        routes = routes_for(tokens, self.dense_value)
        moe = LAYOUT.moe
        if self.shift_decode and prompt is not None:
            routes[prompt:] = routes_for(tokens + [5], self.dense_value)[prompt + 1 :]
        if cached:
            start = 0 if self.stale_cache else cached
            routes[start:, moe] = (routes[start:, moe] + 1) % LAYOUT.experts
            if self.zero_recomputed:
                routes[cached:, moe] = 0
            if self.cache_block and cached % self.cache_block:
                routes[cached - cached % self.cache_block : cached, moe] = 0
        if self.drift_from is not None and prompt is not None:
            drifting, step = moe[self.drift_from :], len(self.requests)  # new on each request
            routes[prompt:, drifting] = (routes[prompt:, drifting] + step) % LAYOUT.experts
        return encode(routes)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if len(self.requests) <= self.unavailable:
            return httpx.Response(503, text="busy")
        if self.refuse_every and len(self.requests) % self.refuse_every == 0:
            return httpx.Response(400, text="bad request")
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
        cached = self.cached(prompt)
        self.seen.append(prompt)
        out = OUT if "min_tokens" not in body else list(range(7, 7 + min(body["min_tokens"], 64)))
        tokens = [f"token_id:{t}" for t in out]
        choice = {
            "index": 0,
            "text": "x",
            "finish_reason": "length",
            "logprobs": {"tokens": tokens, "token_logprobs": [-0.5] * len(out)},
        }
        nvext = {"routed_experts": self.routes(prompt + out, cached, len(prompt))}
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
        if self.drop_every and len(self.requests) % self.drop_every == 0:
            return httpx.Response(200, json={"choices": choices, "usage": usage})
        if self.routes_in_choice:
            choices = [{**c, "nvext": nvext} for c in choices]
            return httpx.Response(200, json={"choices": choices, "usage": usage})
        return httpx.Response(200, json={"choices": choices, "nvext": nvext, "usage": usage})


def run_probe(engine: Engine, checks=probe.CHECKS) -> dict:
    config = RolloutConfig(base_url="https://gw.invalid/v1", model="m")
    p = probe.Probe(config, Tokenizer(), LAYOUT, backoff=0, transport=httpx.MockTransport(engine))
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
    engine = Engine(cache_unit=128)
    results = run_probe(engine)
    assert {name: r.status for name, r in results.items()} == {
        "routes": "pass",
        "align": "pass",
        "cache": "pass",
        "burst": "pass",
        "n2": "info",
        "stream": "pass",
        "chat": "pass",
        "long": "info",
    }
    assert results["burst"].details == {"requests": 64, "missing": 0, "misaligned": 0, "errors": 0}
    assert results["align"].details["response_rows"] == 63
    assert results["align"].details["rows_agree"] == 1.0
    assert results["n2"].details["matches"] == [0, 1]
    assert results["chat"].details["prompt_ids_match_the_trainers_render"] is True
    assert results["stream"].details["events_with_routes"] == [2]
    # requests go out like cwtrain's, with routes, and the long one runs to max_tokens
    assert all(r["nvext"]["extra_fields"][0] == "routed_experts" for r in engine.requests)
    assert engine.requests[-1]["ignore_eos"] is True
    assert engine.requests[-1]["min_tokens"] == engine.requests[-1]["max_tokens"] == 16384
    assert results["cache"].details["prompt_tokens"] >= 6000


def test_a_burst_fails_when_some_responses_lose_their_routes() -> None:
    burst = run_probe(Engine(drop_every=4), ("burst",))["burst"]
    assert burst.status == "fail" and burst.details["missing"] == 16
    assert "16 of 64 concurrent responses had no routes" in burst.summary
    misaligned = run_probe(Engine(extra_rows=1), ("burst",))["burst"]
    assert misaligned.status == "fail" and misaligned.details["misaligned"] == 64


def test_busy_answers_are_asked_again_and_failed_requests_are_counted() -> None:
    busy = run_probe(Engine(unavailable=3), ("burst",))["burst"]
    assert busy.status == "pass" and busy.details["errors"] == 0
    refused = run_probe(Engine(refuse_every=8), ("burst",))["burst"]
    assert refused.status == "fail" and refused.details["errors"] == 8
    assert refused.details["missing"] == 0 and "HTTP 400" in refused.details["first_error"]


def test_align_allows_drift_in_deeper_layers_but_not_in_the_first() -> None:
    drift = run_probe(Engine(drift_from=1), ("align",))["align"]
    assert drift.status == "pass", drift.summary
    assert drift.details["rows_agree_by_moe_layer"][0] == 1.0
    everywhere = run_probe(Engine(drift_from=0), ("align",))["align"]
    assert everywhere.status == "fail" and "first MoE layer 0.0%" in everywhere.summary


def test_align_fails_rows_that_are_one_position_late_though_their_count_is_right() -> None:
    results = run_probe(Engine(shift_decode=True), ("routes", "align"))
    assert results["routes"].status == "pass"
    align = results["align"]
    assert align.status == "fail", align.summary
    assert align.details["rows_agree"] < 0.1 and align.details["rows_agree_one_position_off"] > 0.9


def test_align_judges_nothing_from_a_short_answer() -> None:
    class Short(Engine):
        def __call__(self, request):  # an endpoint that ignores min_tokens
            body = json.loads(request.content)
            body.pop("min_tokens", None)
            return super().__call__(httpx.Request("POST", request.url, json=body))

    align = run_probe(Short(), ("align",))["align"]
    assert align.status == "info" and "only 2 response rows" in align.summary


def test_cached_rows_must_match_while_recomputed_rows_may_differ() -> None:
    cache = run_probe(Engine(cache_unit=512), ("cache",))["cache"]
    assert cache.status == "pass" and cache.details["cached_rows_agree"] == 1.0
    assert cache.details["recomputed_rows_agree"] == 0.0  # every recomputed row moved
    # the prompt that shares half the hit got its own, shorter hit, with the same rows
    assert 0 < cache.details["partial_cached_tokens"] < cache.details["cached_tokens"][1]
    assert cache.details["partial_cached_rows_agree"] == 1.0
    stale = run_probe(Engine(cache_unit=512, stale_cache=True), ("cache",))["cache"]
    assert stale.status == "fail" and stale.details["cached_rows_agree"] == 0.0


def test_cache_fails_malformed_rows_even_where_they_were_recomputed() -> None:
    zeroed = run_probe(Engine(cache_unit=512, zero_recomputed=True), ("cache",))["cache"]
    assert zeroed.status == "fail" and "name an expert twice" in zeroed.summary


def test_cache_fails_zero_rows_from_a_hit_that_ends_inside_a_block() -> None:
    # 128-token matching over 640-token blocks: hits that end inside a block lose their rows
    partial = run_probe(Engine(cache_unit=128, cache_block=640), ("cache",))["cache"]
    assert partial.status == "fail" and "name an expert twice" in partial.summary


def test_no_cache_hit_is_reported_not_passed() -> None:
    cache = run_probe(Engine(), ("cache",))["cache"]
    assert cache.status == "info" and "no cache hit" in cache.summary


def test_routes_outside_the_top_level_nvext_fail_as_they_would_fail_the_trainer() -> None:
    results = run_probe(Engine(routes_in_choice=True, cache_unit=512), ("routes", "burst", "cache"))
    assert {name: r.status for name, r in results.items()} == {
        "routes": "fail",
        "burst": "fail",
        "cache": "fail",
    }
    assert results["burst"].details["missing"] == 64


def test_chat_without_prompt_ids_checks_rows_against_the_usage() -> None:
    chat = run_probe(Engine(chat_prompt_ids=False), ("chat",))["chat"]
    assert chat.status == "pass" and "no prompt_token_ids" in chat.summary
    broken = run_probe(Engine(chat_prompt_ids=False, extra_rows=1), ("chat",))["chat"]
    assert broken.status == "fail"


def test_rows_that_do_not_line_up_with_the_tokens_fail() -> None:
    results = run_probe(Engine(extra_rows=1), ("routes", "align", "stream", "chat"))
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


def test_find_reads_only_the_top_level_nvext() -> None:
    body = {"nvext": {"a": 1}, "choices": [{"b": 2, "nvext": {"c": 3}}], "d": 4}
    assert [probe.find(body, k) for k in "abcd"] == [1, None, None, None]
    assert probe.find({"nvext": None}, "a") is None


def test_sse_events_skip_comments_blank_lines_and_done() -> None:
    lines = [": keep-alive", "", 'data: {"x": 1}', 'data:{"y": 2}', "data: [DONE]"]
    assert probe.sse_events(lines) == [{"x": 1}, {"y": 2}]
