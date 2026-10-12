# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx
import pytest

from cwtrain.errors import CWTrainError
from cwtrain.integrations.rollout import (
    CompletionSampler,
    RolloutConfig,
    RolloutError,
    sample_groups,
)

ROLLOUT_ENV = {"CWTRAIN_ROLLOUT_BASE_URL": "u", "CWTRAIN_ROLLOUT_MODEL": "m"}


class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"


@dataclass
class Sample:  # the fields slime's and miles' Sample share
    prompt: str = ""
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    status: Status = Status.PENDING
    rollout_log_probs: list[float] | None = None
    Status = Status


class Tokenizer:
    """One token per character; the endpoint's token strings are the characters themselves."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def make_server(responses: list[httpx.Response | Exception]):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(
            {"path": request.url.path, "auth": request.headers.get("authorization"), **body}
        )
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.MockTransport(handler), seen


def completions_response(
    texts: list[str],
    finish: list[str] | None = None,
    logprobs: list[list[float] | None] | None = None,
) -> httpx.Response:
    finish = finish or ["stop"] * len(texts)
    choices = []
    for i, (t, f) in enumerate(zip(texts, finish, strict=True)):
        choice: dict = {"index": i, "text": t, "finish_reason": f}
        if logprobs and logprobs[i] is not None:
            # vLLM reports one string per sampled token; a stop token has no text
            tokens = list(t)[: len(logprobs[i])]
            tokens += ["<|endoftext|>"] * (len(logprobs[i]) - len(tokens))
            choice["logprobs"] = {"token_logprobs": logprobs[i], "tokens": tokens}
        choices.append(choice)
    return httpx.Response(200, json={"choices": list(reversed(choices))})  # out of order on purpose


def test_config_from_env_reads_key_file(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret\n")
    cfg = RolloutConfig.from_env(
        {
            "CWTRAIN_ROLLOUT_BASE_URL": "https://gw.example/v1/",
            "CWTRAIN_ROLLOUT_MODEL": "m",
            "CWTRAIN_ROLLOUT_API_KEY_FILE": str(key),
            "CWTRAIN_ROLLOUT_CONCURRENCY": "4",
        }
    )
    assert cfg.base_url == "https://gw.example/v1" and cfg.api_key == "secret"
    assert cfg.concurrency == 4 and cfg.timeout == 600.0
    with pytest.raises(CWTrainError, match="required"):
        RolloutConfig.from_env({})
    assert RolloutConfig.from_env(ROLLOUT_ENV).api_key is None


def test_sample_groups_fills_samples_and_orders_choices() -> None:
    transport, seen = make_server(
        [completions_response(["ab", "cde"], ["stop", "length"]), completions_response(["x"])]
    )
    cfg = RolloutConfig(base_url="https://gw/v1", model="m", api_key="k", concurrency=2)
    sampler = CompletionSampler(cfg, transport=transport)
    groups = [[Sample(prompt="p1"), Sample(prompt="p1")], [Sample(prompt="p2")]]

    metrics = asyncio.run(
        sample_groups(sampler, groups, tokenizer=Tokenizer(), max_tokens=8, temperature=1.0)
    )

    g1, g2 = groups
    assert g1[0].response == "ab" and g1[0].tokens == [ord("p"), ord("1"), ord("a"), ord("b")]
    assert g1[0].response_length == 2 and g1[0].status is Status.COMPLETED
    assert g1[1].response == "cde" and g1[1].status is Status.TRUNCATED
    assert g2[0].response == "x" and g2[0].status is Status.COMPLETED
    assert metrics["coreweave/samples"] == 3.0 and metrics["coreweave/truncated_fraction"] == 1 / 3
    # prompts go out as the token ids the trainer scores, so the endpoint never re-tokenizes them
    assert all(isinstance(s["prompt"], list) and "add_special_tokens" not in s for s in seen)
    by_prompt = {Tokenizer().decode(s["prompt"]): s for s in seen}
    assert by_prompt["p1"]["n"] == 2 and by_prompt["p2"]["n"] == 1
    assert by_prompt["p1"]["max_tokens"] == 8 and by_prompt["p1"]["model"] == "m"
    assert by_prompt["p1"]["auth"] == "Bearer k" and by_prompt["p1"]["path"] == "/v1/completions"


def test_retries_transient_failures_then_succeeds(monkeypatch) -> None:
    async def no_sleep(_s):
        return None

    monkeypatch.setattr("cwtrain.integrations.rollout.asyncio.sleep", no_sleep)
    transport, seen = make_server(
        [
            httpx.Response(503, text="busy"),
            httpx.ConnectError("boom"),
            completions_response(["ok"]),
        ]
    )
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    out = asyncio.run(sampler.sample("p", 1, max_tokens=4, temperature=0.7))
    assert [c.text for c in out] == ["ok"] and len(seen) == 3
    assert seen[0]["auth"] is None  # no key -> no Authorization header


def test_non_retryable_status_and_wrong_choice_count_raise(monkeypatch) -> None:
    async def no_sleep(_s):
        return None

    monkeypatch.setattr("cwtrain.integrations.rollout.asyncio.sleep", no_sleep)
    transport, _ = make_server([httpx.Response(400, text="bad request")])
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    with pytest.raises(RolloutError, match="HTTP 400"):
        asyncio.run(sampler.sample("p", 1, max_tokens=4, temperature=1.0))

    transport, _ = make_server([completions_response(["only one"])])
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    with pytest.raises(RolloutError, match="1 choices for n=2"):
        asyncio.run(sampler.sample("p", 2, max_tokens=4, temperature=1.0))

    transport, _ = make_server([httpx.Response(503, text="a"), httpx.Response(503, text="b")])
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m", max_attempts=2), transport=transport
    )
    with pytest.raises(RolloutError, match="after 2 attempts"):
        asyncio.run(sampler.sample("p", 1, max_tokens=4, temperature=1.0))


def test_unrendered_prompt_is_rejected() -> None:
    transport, _ = make_server([])
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    groups = [[Sample(prompt=[{"role": "user", "content": "hi"}])]]  # type: ignore[arg-type]
    with pytest.raises(RolloutError, match="apply-chat-template"):
        asyncio.run(
            sample_groups(sampler, groups, tokenizer=Tokenizer(), max_tokens=4, temperature=1.0)
        )


def test_logprobs_requested_and_aligned_with_retokenized_response() -> None:
    # "ab" -> 2 tokens with 3 log-probs and finish_reason=stop: the EOS entry is dropped.
    # "cde" -> 3 tokens with 3 log-probs: kept as is. "x" -> 1 token with 4 log-probs: discarded.
    transport, seen = make_server(
        [
            completions_response(
                ["ab", "cde"], ["stop", "length"], [[-0.1, -0.2, -0.3], [-1.0, -2.0, -3.0]]
            ),
            completions_response(["x"], ["stop"], [[-0.5, -0.5, -0.5, -0.5]]),
        ]
    )
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    groups = [[Sample(prompt="p1"), Sample(prompt="p1")], [Sample(prompt="p2")]]
    metrics = asyncio.run(
        sample_groups(sampler, groups, tokenizer=Tokenizer(), max_tokens=8, temperature=1.0)
    )
    assert all(req["logprobs"] == 1 for req in seen)
    # one sample did not align, so the whole batch goes without rollout log-probs
    assert all(s.rollout_log_probs is None for g in groups for s in g)
    assert metrics["coreweave/logprobs_aligned_fraction"] == 2 / 3
    assert metrics["coreweave/logprobs_kept"] == 0.0


def test_logprobs_kept_when_every_sample_aligns() -> None:
    transport, _ = make_server(
        [
            completions_response(
                ["ab", "cde"], ["stop", "length"], [[-0.1, -0.2, -0.3], [-1.0, -2.0, -3.0]]
            )
        ]
    )
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    groups = [[Sample(prompt="p1"), Sample(prompt="p1")]]
    metrics = asyncio.run(
        sample_groups(sampler, groups, tokenizer=Tokenizer(), max_tokens=8, temperature=1.0)
    )
    assert groups[0][0].rollout_log_probs == [-0.1, -0.2]  # EOS entry dropped
    assert groups[0][1].rollout_log_probs == [-1.0, -2.0, -3.0]
    assert metrics["coreweave/logprobs_kept"] == 1.0


def test_logprobs_can_be_disabled() -> None:
    transport, seen = make_server([completions_response(["ab"], ["stop"], [[-0.1, -0.2, -0.3]])])
    cfg = RolloutConfig.from_env(
        {
            "CWTRAIN_ROLLOUT_BASE_URL": "https://gw/v1",
            "CWTRAIN_ROLLOUT_MODEL": "m",
            "CWTRAIN_ROLLOUT_LOGPROBS": "0",
        }
    )
    assert cfg.logprobs is False
    sampler = CompletionSampler(cfg, transport=transport)
    groups = [[Sample(prompt="p")]]
    asyncio.run(
        sample_groups(sampler, groups, tokenizer=Tokenizer(), max_tokens=8, temperature=1.0)
    )
    assert "logprobs" not in seen[0]
    assert groups[0][0].rollout_log_probs is None


class WordTokenizer:
    """Re-tokenizes "equation" as equ+ation, unlike the eq+uation the endpoint sampled."""

    vocab = {1: "equ", 2: "ation", 3: "eq", 4: "uation"}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert text == "equation"
        return [1, 2]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.vocab[token_id]


def test_logprobs_discarded_when_token_identities_differ() -> None:
    def response(tokens: list[str]) -> httpx.Response:
        choice = {
            "index": 0,
            "text": "equation",
            "finish_reason": "length",
            "logprobs": {"token_logprobs": [-0.3, -0.7], "tokens": tokens},
        }
        return httpx.Response(200, json={"choices": [choice]})

    # same text, same count, different tokens: the log-probs belong to eq+uation, not equ+ation
    transport, _ = make_server([response(["eq", "uation"])])
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    groups = [[Sample(prompt="equation")]]
    metrics = asyncio.run(
        sample_groups(sampler, groups, tokenizer=WordTokenizer(), max_tokens=8, temperature=1.0)
    )
    assert groups[0][0].tokens[-2:] == [1, 2] and groups[0][0].rollout_log_probs is None
    assert metrics["coreweave/logprobs_kept"] == 0.0

    # the endpoint's strings name the re-tokenized tokens: kept
    transport, _ = make_server([response(["equ", "ation"])])
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    groups = [[Sample(prompt="equation")]]
    metrics = asyncio.run(
        sample_groups(sampler, groups, tokenizer=WordTokenizer(), max_tokens=8, temperature=1.0)
    )
    assert groups[0][0].rollout_log_probs == [-0.3, -0.7]
    assert metrics["coreweave/logprobs_kept"] == 1.0


def test_same_tokens_looks_up_the_vocabulary_once() -> None:
    from cwtrain.integrations.rollout import same_tokens

    calls: list[object] = []

    class Tok(WordTokenizer):
        def convert_ids_to_tokens(self, token_id):
            calls.append(token_id)
            if isinstance(token_id, list):
                return [self.vocab[t] for t in token_id]
            return self.vocab[token_id]

    tok = Tok()
    assert same_tokens(tok, [1, 2], ["equ", "ation"])
    assert calls == [[1, 2]]  # one batched lookup, no per-token calls
    assert not same_tokens(tok, [1, 2], ["eq", "uation"])
    assert not same_tokens(tok, [1], ["equ", "ation"])


def test_same_tokens_handles_tokenizers_that_convert_one_id_at_a_time() -> None:
    from cwtrain.integrations.rollout import same_tokens

    class OneAtATime(WordTokenizer):
        def convert_ids_to_tokens(self, token_id):
            return {1: "equ", 2: "ation"}[token_id]  # a list raises TypeError

    assert same_tokens(OneAtATime(), [1, 2], ["equ", "ation"])
    assert not same_tokens(OneAtATime(), [1, 2], ["eq", "uation"])


class EosTokenizer(Tokenizer):
    eos_token_id = 0

    def convert_ids_to_tokens(self, token_id):
        if isinstance(token_id, list):
            return [self.convert_ids_to_tokens(t) for t in token_id]
        return "<eos>" if token_id == 0 else chr(token_id)


def _choice(text, tokens, logprobs, finish="stop", stop_reason=None):
    return {
        "index": 0,
        "text": text,
        "finish_reason": finish,
        "stop_reason": stop_reason,
        "logprobs": {"tokens": tokens, "token_logprobs": logprobs},
    }


def test_end_of_sequence_stays_in_the_trained_sequence() -> None:
    transport, seen = make_server(
        [
            httpx.Response(
                200, json={"choices": [_choice("ab", ["a", "b", "<eos>"], [-0.1, -0.2, -0.3])]}
            ),
            # an EOS-only completion is one token, not an empty response
            httpx.Response(200, json={"choices": [_choice("", ["<eos>"], [-0.4])]}),
        ]
    )
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    groups = [[Sample(prompt="p1")], [Sample(prompt="p2")]]
    metrics = asyncio.run(
        sample_groups(sampler, groups, tokenizer=EosTokenizer(), max_tokens=8, temperature=1.0)
    )
    first, second = groups[0][0], groups[1][0]
    assert first.tokens[-3:] == [ord("a"), ord("b"), 0] and first.response_length == 3
    assert first.rollout_log_probs == [-0.1, -0.2, -0.3]  # the stop token's log-prob is kept
    assert second.response_length == 1 and second.rollout_log_probs == [-0.4]
    assert metrics["coreweave/logprobs_kept"] == 1.0
    assert [req["prompt"] for req in seen] == [[ord("p"), ord("1")], [ord("p"), ord("2")]]


def test_text_prompts_when_token_id_prompts_are_off() -> None:
    transport, seen = make_server([completions_response(["ab"])])
    cfg = RolloutConfig(base_url="https://gw/v1", model="m", token_id_prompts=False)
    sampler = CompletionSampler(cfg, transport=transport)
    groups = [[Sample(prompt="p1")]]
    asyncio.run(
        sample_groups(sampler, groups, tokenizer=Tokenizer(), max_tokens=8, temperature=1.0)
    )
    assert seen[0]["prompt"] == "p1" and seen[0]["add_special_tokens"] is False
    assert groups[0][0].tokens == [ord("p"), ord("1"), ord("a"), ord("b")]


def test_rejected_token_id_prompts_fall_back_to_text_once_text_works() -> None:
    transport, seen = make_server(
        [
            httpx.Response(400, text="prompt must be a string"),
            completions_response(["ab"]),
            completions_response(["cd"]),
        ]
    )
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    tok = Tokenizer()
    first = [[Sample(prompt="p1")]]
    asyncio.run(sample_groups(sampler, first, tokenizer=tok, max_tokens=8, temperature=1.0))
    assert [type(s["prompt"]) for s in seen] == [list, str] and not sampler.token_id_prompts
    assert first[0][0].response == "ab"
    # the rest of the run sends text straight away
    second = [[Sample(prompt="p2")]]
    asyncio.run(sample_groups(sampler, second, tokenizer=tok, max_tokens=8, temperature=1.0))
    assert seen[2]["prompt"] == "p2" and second[0][0].response == "cd"


def test_a_4xx_that_text_prompts_get_too_keeps_token_id_prompts() -> None:
    transport, seen = make_server(
        [
            httpx.Response(400, text="max_tokens too large"),
            httpx.Response(400, text="max_tokens too large"),
        ]
    )
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    with pytest.raises(RolloutError, match="HTTP 400: max_tokens too large"):
        asyncio.run(
            sample_groups(
                sampler,
                [[Sample(prompt="p1")]],
                tokenizer=Tokenizer(),
                max_tokens=8,
                temperature=1.0,
            )
        )
    assert [type(s["prompt"]) for s in seen] == [list, str] and sampler.token_id_prompts


@pytest.mark.parametrize("field", ["stop", "stop_token_ids"])
def test_more_than_four_stops_fail_before_anything_is_sent(field) -> None:
    transport, seen = make_server([completions_response(["ab"])])
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    stops = ["a", "b", "c", "d", "e"] if field == "stop" else [1, 2, 3, 4, 5]
    with pytest.raises(RolloutError, match="at most 4"):
        asyncio.run(sampler.sample([1], 1, max_tokens=4, temperature=1.0, **{field: stops}))
    assert seen == []
    # four is fine
    asyncio.run(sampler.sample([1], 1, max_tokens=4, temperature=1.0, **{field: stops[:4]}))
    assert len(seen[0][field]) == 4


def test_token_id_prompts_flag_comes_from_the_environment() -> None:
    assert RolloutConfig.from_env(ROLLOUT_ENV).token_id_prompts is True
    off = {**ROLLOUT_ENV, "CWTRAIN_ROLLOUT_TOKEN_ID_PROMPTS": "0"}
    assert RolloutConfig.from_env(off).token_id_prompts is False


def test_stop_string_match_appends_no_token() -> None:
    transport, _ = make_server(
        [
            httpx.Response(
                200, json={"choices": [_choice("ab", ["a", "b"], [-0.1, -0.2], stop_reason="</a>")]}
            )
        ]
    )
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    groups = [[Sample(prompt="p")]]
    asyncio.run(
        sample_groups(sampler, groups, tokenizer=EosTokenizer(), max_tokens=8, temperature=1.0)
    )
    assert groups[0][0].tokens[-2:] == [ord("a"), ord("b")]
    assert groups[0][0].rollout_log_probs == [-0.1, -0.2]


def test_exact_token_ids_include_the_sampled_stop_token() -> None:
    choice = _choice("ab", ["token_id:97", "token_id:98", "token_id:0"], [-0.1, -0.2, -0.3])
    transport, seen = make_server([httpx.Response(200, json={"choices": [choice]})])
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    groups = [[Sample(prompt="p")]]
    asyncio.run(
        sample_groups(sampler, groups, tokenizer=EosTokenizer(), max_tokens=8, temperature=1.0)
    )
    assert seen[0]["return_tokens_as_token_ids"] is True
    assert groups[0][0].tokens[-3:] == [97, 98, 0]
    assert groups[0][0].rollout_log_probs == [-0.1, -0.2, -0.3]


def test_non_json_200_is_retried(monkeypatch) -> None:
    async def no_sleep(_s):
        return None

    monkeypatch.setattr("cwtrain.integrations.rollout.asyncio.sleep", no_sleep)
    transport, seen = make_server(
        [httpx.Response(200, text="<html>rate limited</html>"), completions_response(["ok"])]
    )
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    out = asyncio.run(sampler.sample("p", 1, max_tokens=4, temperature=1.0))
    assert [c.text for c in out] == ["ok"] and len(seen) == 2


def test_rollout_flags_use_the_shared_parser() -> None:
    assert RolloutConfig.from_env({**ROLLOUT_ENV, "CWTRAIN_ROLLOUT_LOGPROBS": ""}).logprobs is True
    with pytest.raises(CWTrainError, match="must be a boolean"):
        RolloutConfig.from_env({**ROLLOUT_ENV, "CWTRAIN_ROLLOUT_TOKEN_IDS": "sometimes"})


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, 16), ("", 16), ("  ", 16), ("64", 64), (" 8 ", 8)]
)
def test_concurrency_unset_or_empty_means_the_default(raw, expected) -> None:
    # a recipe's Job renders a setting missing from run.env as ""
    env = dict(ROLLOUT_ENV)
    if raw is not None:
        env["CWTRAIN_ROLLOUT_CONCURRENCY"] = raw
    assert RolloutConfig.from_env(env).concurrency == expected


@pytest.mark.parametrize("raw", ["0", "-4", "1.5", "lots"])
def test_concurrency_must_be_a_whole_number_of_at_least_one(raw) -> None:
    with pytest.raises(CWTrainError, match="CWTRAIN_ROLLOUT_CONCURRENCY"):
        RolloutConfig.from_env({**ROLLOUT_ENV, "CWTRAIN_ROLLOUT_CONCURRENCY": raw})


@pytest.mark.parametrize(
    ("name", "raw", "field", "expected"),
    [
        ("TIMEOUT", "", "timeout", 600.0),
        ("TIMEOUT", "90.5", "timeout", 90.5),
        ("MAX_ATTEMPTS", " ", "max_attempts", 6),
        ("MAX_ATTEMPTS", "3", "max_attempts", 3),
    ],
)
def test_timeout_and_attempts_take_an_empty_value_as_the_default(
    name, raw, field, expected
) -> None:
    cfg = RolloutConfig.from_env({**ROLLOUT_ENV, f"CWTRAIN_ROLLOUT_{name}": raw})
    assert getattr(cfg, field) == expected


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("TIMEOUT", "0"),
        ("TIMEOUT", "soon"),
        ("TIMEOUT", "nan"),
        ("TIMEOUT", "inf"),
        ("TIMEOUT", "-Infinity"),
        ("MAX_ATTEMPTS", "0"),
    ],
)
def test_timeout_and_attempts_reject_bad_values(name, raw) -> None:
    with pytest.raises(CWTrainError, match=f"CWTRAIN_ROLLOUT_{name}"):
        RolloutConfig.from_env({**ROLLOUT_ENV, f"CWTRAIN_ROLLOUT_{name}": raw})


@pytest.mark.parametrize(
    "bad", [{"concurrency": 0}, {"max_attempts": 0}, {"timeout": 0.0}, {"timeout": float("inf")}]
)
def test_rollout_config_rejects_values_the_sampler_cannot_use(bad) -> None:
    with pytest.raises(CWTrainError, match=f"RolloutConfig.{next(iter(bad))}"):
        RolloutConfig(base_url="u", model="m", **bad)


@pytest.mark.parametrize(
    ("variable", "raw", "message"),
    [
        ("CONCURRENCY", "0", "must be at least 1, got '0'"),
        ("MAX_ATTEMPTS", "many", "must be a whole number, got 'many'"),
        ("TIMEOUT", "0", "must be a finite number above 0, got '0'"),
        ("TIMEOUT", "soon", "must be a number, got 'soon'"),
        ("TOKEN_IDS", "sometimes", "must be a boolean, got 'sometimes'"),
        ("BASE_URL", " ", "is required"),
    ],
)
def test_an_error_names_the_label_the_caller_gave_the_variable(variable, raw, message) -> None:
    env = {**ROLLOUT_ENV, f"CWTRAIN_ROLLOUT_{variable}": raw}
    with pytest.raises(CWTrainError) as excinfo:
        RolloutConfig.from_env(env, labels={f"CWTRAIN_ROLLOUT_{variable}": "cfg.setting"})
    assert str(excinfo.value) == f"cfg.setting {message}"
    # a variable without a label is named as itself
    with pytest.raises(CWTrainError) as excinfo:
        RolloutConfig.from_env(env, labels={"CWTRAIN_ROLLOUT_OTHER": "cfg.other"})
    assert str(excinfo.value) == f"CWTRAIN_ROLLOUT_{variable} {message}"


def test_a_group_that_kept_token_ids_does_not_undo_another_groups_fallback() -> None:
    async def run() -> CompletionSampler:
        held: dict = {}
        b_sent = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            prompt = json.loads(request.content)["prompt"]
            if prompt == [ord("a")]:  # group "a": rejected once "b" is in flight with token ids
                await b_sent.wait()
                return httpx.Response(400, text="prompt must be a string")
            if prompt == "a":  # "a"'s text retry succeeds
                return completions_response(["x"])
            assert prompt == [ord("b")], "group b must have started with token ids"
            b_sent.set()  # "b": token ids accepted, answered once "a" has recorded its fallback
            for _ in range(10_000):
                if not held["sampler"].token_id_prompts:
                    break
                await asyncio.sleep(0)
            return completions_response(["y"])

        sampler = CompletionSampler(
            RolloutConfig(base_url="https://gw/v1", model="m"),
            transport=httpx.MockTransport(handler),
        )
        held["sampler"] = sampler
        groups = [[Sample(prompt="a")], [Sample(prompt="b")]]
        await sample_groups(sampler, groups, tokenizer=Tokenizer(), max_tokens=8, temperature=1.0)
        return sampler

    assert asyncio.run(run()).token_id_prompts is False


def test_both_prompt_forms_failing_report_the_token_id_error_with_the_text_one(caplog) -> None:
    from cwtrain.integrations.rollout import sample_prompt

    transport, _seen = make_server(
        [
            httpx.Response(400, text="token ids not accepted"),
            httpx.Response(400, text="max_tokens too large"),
        ]
    )
    sampler = CompletionSampler(
        RolloutConfig(base_url="https://gw/v1", model="m"), transport=transport
    )
    sampling = sample_prompt(
        sampler, [1], 1, text=lambda: "p", token_id_prompts=True, max_tokens=8, temperature=1.0
    )
    with (
        caplog.at_level(logging.WARNING, logger="cwtrain.integrations.rollout"),
        pytest.raises(RolloutError, match="token ids not accepted") as excinfo,
    ):
        asyncio.run(sampling)
    # the text failure is shown with it in the traceback, and logged
    assert "max_tokens too large" in str(excinfo.value.__context__)
    assert not excinfo.value.__suppress_context__
    assert "text retry failed too" in caplog.text


def test_shared_sampler_is_reused_on_one_loop_and_not_across_loops() -> None:
    from cwtrain.integrations.rollout import shared_sampler

    cfg = RolloutConfig(base_url="https://gw/v1", model="m")

    async def twice():
        return shared_sampler(cfg), shared_sampler(cfg)

    loop = asyncio.new_event_loop()
    try:
        a, b = loop.run_until_complete(twice())
        c, _ = loop.run_until_complete(twice())  # a later rollout on the same loop
    finally:
        loop.close()
    assert a is b is c
    d, _ = asyncio.run(twice())  # another loop gets its own client
    assert d is not a
