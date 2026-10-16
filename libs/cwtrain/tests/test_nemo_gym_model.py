# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import copy
import json
import re
from types import SimpleNamespace

import pytest

from cwtrain.integrations.nemo_rl.gym_model import (
    chat_response,
    completion_request,
    parse_tool_calls,
)


def test_sends_exact_prompt_ids_and_sampling_parameters():
    calls = []

    def template(messages, **kwargs):
        calls.append((messages, kwargs))
        return [10, 20]

    request = completion_request(
        {
            "model": "model",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            "max_tokens": 5,
            "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        SimpleNamespace(apply_chat_template=template),
    )
    assert request["prompt"] == [10, 20]
    assert request["max_tokens"] == 5 and request["temperature"] == 0.7
    assert request["return_tokens_as_token_ids"] and request["logprobs"] == 0
    assert calls[0][0] == [{"role": "user", "content": "hello"}]
    assert calls[0][1]["enable_thinking"] is False


@pytest.fixture
def response():
    return {
        "id": "x",
        "created": 1,
        "model": "m",
        "usage": {"prompt_tokens": 2},
        "choices": [
            {
                "text": "answer",
                "finish_reason": "stop",
                "logprobs": {
                    "tokens": ["token_id:12", "token_id:2"],
                    "token_logprobs": [-0.1, -0.2],
                },
            }
        ],
    }


def test_keeps_stop_token_and_its_logprob(response):
    original = copy.deepcopy(response)
    message = chat_response(response, [10, 20])["choices"][0]["message"]
    assert message["prompt_token_ids"] == [10, 20]
    assert message["generation_token_ids"] == [12, 2]
    assert message["generation_log_probs"] == [-0.1, -0.2]
    assert response == original


@pytest.mark.parametrize(
    "change", ["text_tokens", "missing_logprobs", "nonfinite", "wrong_prompt_count"]
)
def test_rejects_unverifiable_training_metadata(response, change):
    lp = response["choices"][0]["logprobs"]
    if change == "text_tokens":
        lp["tokens"][0] = "answer"
    if change == "missing_logprobs":
        lp["token_logprobs"] = []
    if change == "nonfinite":
        lp["token_logprobs"][0] = float("nan")
    if change == "wrong_prompt_count":
        response["usage"]["prompt_tokens"] = 3
    with pytest.raises(ValueError):
        chat_response(response, [10, 20])


class ChatML:
    """A character-level ChatML tokenizer whose special tokens are single IDs."""

    eos_token = "<|im_end|>"
    SPECIAL = {"<|im_end|>": 1, "<|im_start|>": 2}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in re.split(r"(<\|im_end\|>|<\|im_start\|>)", text):
            ids.extend([self.SPECIAL[piece]] if piece in self.SPECIAL else [ord(c) for c in piece])
        return ids

    def decode(self, ids):
        names = {v: k for k, v in self.SPECIAL.items()}
        return "".join(names.get(i, chr(i)) if i < 3 else chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, tools=None, **_):
        text = "<|im_start|>system\n" + json.dumps(tools) + "<|im_end|>\n" if tools else ""
        for m in messages:
            body = m["content"]
            for call in m.get("tool_calls") or []:
                body += "<tool_call>" + json.dumps(call["function"]["arguments"]) + "</tool_call>"
            text += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "properties": {"queries": {"type": "array"}, "note": {"type": "string"}}
            },
        },
    }
]


def test_later_turn_extends_the_previous_turns_exact_tokens():
    tokenizer = ChatML()
    first = completion_request(
        {"messages": [{"role": "user", "content": "q"}], "tools": TOOLS}, tokenizer
    )
    # The model's own bytes for the tool call: re-rendering them would give different text.
    generated = tokenizer.encode('calling <tool_call>{"queries": ["x"]}</tool_call>') + [1]
    second = completion_request(
        {
            "tools": TOOLS,
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "calling",
                    "tool_calls": [
                        {"function": {"name": "search", "arguments": '{"queries": ["x"]}'}}
                    ],
                    "prompt_token_ids": first["prompt"],
                    "generation_token_ids": generated,
                },
                {"role": "tool", "tool_call_id": "c1", "content": "result"},
            ],
        },
        tokenizer,
    )
    prefix = first["prompt"] + generated
    assert second["prompt"][: len(prefix)] == prefix
    assert tokenizer.decode(second["prompt"][len(prefix) :]) == (
        "\n<|im_start|>tool\nresult<|im_end|>\n<|im_start|>assistant\n"
    )


def test_a_turn_cut_off_by_max_tokens_is_closed_in_the_next_prompt():
    tokenizer = ChatML()
    prompt = tokenizer.encode("<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n")
    request = completion_request(
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "par",
                    "prompt_token_ids": prompt,
                    "generation_token_ids": tokenizer.encode("par"),
                },
                {"role": "user", "content": "go on"},
            ]
        },
        tokenizer,
    )
    tail = tokenizer.decode(request["prompt"][len(prompt) + 3 :])
    assert tail.startswith("<|im_end|>\n<|im_start|>user\ngo on")


def test_rejects_a_template_that_rewrites_earlier_turns():
    class Rewriting(ChatML):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            text = super().apply_chat_template(messages, False, add_generation_prompt, **kwargs)
            text = text.replace("calling", "CALLING") if len(messages) > 2 else text
            return self.encode(text) if tokenize else text

    with pytest.raises(ValueError, match="token prefix"):
        completion_request(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": "calling",
                        "prompt_token_ids": [2],
                        "generation_token_ids": [1],
                    },
                    {"role": "tool", "content": "r"},
                ]
            },
            Rewriting(),
        )


def test_parses_qwen3_coder_tool_calls_by_schema():
    text = (
        "Let me look.\n<tool_call>\n<function=search>\n<parameter=queries>\n"
        '["a b", "c"]\n</parameter>\n<parameter=note>\n[not json]\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == "Let me look."
    assert calls[0]["function"]["name"] == "search"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "queries": ["a b", "c"],
        "note": "[not json]",  # a string parameter stays text
    }
    assert parse_tool_calls("Exact Answer: 42", TOOLS) == ("Exact Answer: 42", [])


def test_tool_calls_reach_the_chat_response(response):
    response["choices"][0]["text"] = (
        '<tool_call>\n<function=search>\n<parameter=queries>\n["x"]\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    choice = chat_response(response, [10, 20], TOOLS)["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "queries": ["x"]
    }
    assert choice["message"]["generation_token_ids"] == [12, 2]


def test_max_tokens_is_capped_to_the_room_the_prompt_leaves():
    tokenizer = ChatML()
    body = {"messages": [{"role": "user", "content": "q"}], "max_tokens": 2048}
    prompt = completion_request(body, tokenizer)["prompt"]
    assert completion_request(body, tokenizer, len(prompt) + 100)["max_tokens"] == 100
    assert completion_request(body, tokenizer, len(prompt) + 5000)["max_tokens"] == 2048
    assert (
        completion_request(dict(body, max_tokens=None), tokenizer, len(prompt) + 7)["max_tokens"]
        == 7
    )


def test_a_full_context_is_reported_as_one_gym_treats_as_a_length_stop():
    from cwtrain.integrations.nemo_rl.gym_model import (
        CONTEXT_FULL,
        ContextFull,
        is_context_overflow,
    )

    tokenizer = ChatML()
    body = {"messages": [{"role": "user", "content": "q"}], "max_tokens": 16}
    prompt = completion_request(body, tokenizer)["prompt"]
    with pytest.raises(ContextFull):
        completion_request(body, tokenizer, len(prompt))
    assert b"context length" in CONTEXT_FULL  # the phrase Gym's vllm_model matches
    # the gateway's two messages for prompt plus max_tokens over the limit
    assert is_context_overflow(
        400,
        "This model configuration accepts at most 32768 combined input and output tokens. "
        "However, your request has 31000 input tokens and asks for 2048 output tokens",
    )
    assert is_context_overflow(400, "This model's maximum context length is 32768 tokens.")
    assert not is_context_overflow(400, "invalid model name")
    assert not is_context_overflow(500, "context length 32768 tokens")
