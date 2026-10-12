# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import copy
from types import SimpleNamespace

import pytest

from cwtrain.integrations.nemo_rl.gym_model import chat_response, completion_request


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
