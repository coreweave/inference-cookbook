# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Gym model server that serves chat requests from CoreWeave completions with exact token IDs.

Gym sends chat completions. The server renders each one with the trainer's tokenizer and sends it
as a token-ID ``/completions`` request. It returns the generated token IDs and log-probs that
training needs.
"""

import math
from typing import Any


def completion_request(body: dict, tokenizer: Any) -> dict:
    """Convert a text-only chat request into a token-ID completions request.

    Raises:
        ValueError: The request has tools, a response format, ``n`` other than 1, or non-text
            content.
    """
    if body.get("tools") or body.get("response_format") or body.get("n", 1) != 1:
        raise ValueError("CoreWeave Gym training supports one text-only completion per request")
    messages = []
    for item in body["messages"]:
        content = item.get("content")
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("CoreWeave Gym training requires text-only messages")
            content = "".join(part["text"] for part in content)
        if not isinstance(content, str) or item.get("tool_calls"):
            raise ValueError("CoreWeave Gym training requires text-only messages")
        messages.append({"role": item["role"], "content": content})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        **(body.get("chat_template_kwargs") or {}),
    )
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("The trainer tokenizer must return nonempty integer prompt IDs")
    result = {
        key: body[key]
        for key in (
            "model",
            "temperature",
            "top_p",
            "seed",
            "stop",
            "presence_penalty",
            "frequency_penalty",
        )
        if body.get(key) is not None
    }
    result.update(
        prompt=prompt,
        max_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
        logprobs=0,
        return_tokens_as_token_ids=True,
    )
    return result


def chat_response(response: dict, prompt: list[int]) -> dict:
    """Convert a completions response into a chat response with token IDs and log-probs.

    Raises:
        ValueError: The response does not hold exactly one completion, lacks exact token IDs or
            finite, aligned log-probs, or has a prompt token count that differs from ``prompt``.
    """
    choices = response.get("choices") or []
    if len(choices) != 1 or response.get("usage", {}).get("prompt_tokens") != len(prompt):
        raise ValueError("CoreWeave must return one completion with the exact prompt token count")
    choice = choices[0]
    lp = choice.get("logprobs") or {}
    pieces, probs = lp.get("tokens"), lp.get("token_logprobs")
    if (
        not isinstance(pieces, list)
        or not isinstance(probs, list)
        or not pieces
        or len(pieces) != len(probs)
    ):
        raise ValueError("CoreWeave Gym requires generated token IDs and aligned log-probs")
    ids = []
    for piece, prob in zip(pieces, probs, strict=True):
        if (
            not isinstance(piece, str)
            or not piece.startswith("token_id:")
            or not piece[9:].isdigit()
        ):
            raise ValueError("CoreWeave omitted exact generated token IDs")
        if type(prob) not in (int, float) or not math.isfinite(prob):
            raise ValueError("CoreWeave returned a non-finite generated-token log-prob")
        ids.append(int(piece[9:]))
    return {
        **{k: response[k] for k in ("id", "created", "model", "usage")},
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": choice["finish_reason"],
                "message": {
                    "role": "assistant",
                    "content": choice["text"],
                    "prompt_token_ids": prompt,
                    "generation_token_ids": ids,
                    "generation_log_probs": probs,
                },
            }
        ],
    }


def run_server() -> None:
    # These dependencies come from NeMo-RL's Gym actor environment, not base cwtrain.
    import nemo_gym  # noqa: F401 - adds the Gym install root to sys.path
    from nemo_gym.openai_utils import NeMoGymAsyncOpenAI, get_response_json
    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
    from transformers import AutoTokenizer

    class Config(VLLMModelConfig):
        tokenizer_path: str

    class Client(NeMoGymAsyncOpenAI):
        tokenizer: Any

        async def create_chat_completion(self, **kwargs):
            body = completion_request(kwargs, self.tokenizer)
            request = {"url": self.base_url + "/completions", "json": body}
            response = await self._request(method="POST", **request)
            await self._raise_for_status(response, request)
            return chat_response(await get_response_json(response), body["prompt"])

    class Model(VLLMModel):
        config: Config

        def _post_init(self):
            super()._post_init()
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_path, local_files_only=True
            )
            self._clients = [
                Client(
                    base_url=url,
                    api_key=self.config.api_key,
                    default_headers=self.config.default_headers,
                    tokenizer=tokenizer,
                )
                for url in self.config.base_url
            ]

    Model.run_webserver()


if __name__ == "__main__":
    run_server()
