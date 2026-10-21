# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Gym model server that serves chat requests from CoreWeave completions with exact token IDs.

Gym sends chat completions. The server renders each one with the trainer's tokenizer and sends it
as a token-ID ``/completions`` request. It returns the generated token IDs and log-probs that
training needs.

Tool calls use the Qwen3-coder XML format of the Nemotron 3 chat templates
(``<tool_call><function=name><parameter=key>value</parameter></function></tool_call>``). A later
turn of a multi-turn trajectory starts from the previous turn's exact prompt and generated token
IDs. Only the messages after that turn are rendered, so the trajectory's token prefix never
changes.

Each request carries the identity that the deployment serves as its cache salt, read from the
file that the weight synchronizer writes (:mod:`cwtrain.integrations.nvext`).
"""

import json
import logging
import math
from typing import Any

from cwtrain.integrations.gym_tool_calls import parse_tool_calls
from cwtrain.integrations.nvext import served_salt, with_nvext

log = logging.getLogger(__name__)

__all__ = ["chat_response", "completion_request", "parse_tool_calls"]

_TRAINING_KEYS = ("prompt_token_ids", "generation_token_ids")
# Gym's vllm_model turns a 400 whose body names the "context length" into an empty completion that
# ends the trajectory with finish_reason "length", instead of failing the whole rollout batch.
CONTEXT_FULL = b"context length: the prompt leaves no room in the deployment's max_model_len"


class ContextFull(ValueError):
    """The prompt already fills the deployment's context, so no token can be generated."""


def is_context_overflow(status: int, body: str) -> bool:
    """Whether a CoreWeave error response says that prompt plus max_tokens exceeds the context."""
    text = body.lower()
    return status == 400 and "tokens" in text and ("context length" in text or "at most" in text)


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        if any(part.get("type") != "text" for part in content):
            raise ValueError("CoreWeave Gym training requires text-only messages")
        content = "".join(part["text"] for part in content)
    if not isinstance(content, str):
        raise ValueError("CoreWeave Gym training requires text-only messages")
    return content


def _template_message(item: dict) -> dict:
    """The chat-template form of one Gym message: text content, tool calls with dict arguments."""
    message = {"role": item["role"], "content": _text(item.get("content"))}
    if item.get("tool_calls"):
        if item["role"] != "assistant":
            raise ValueError("Only assistant messages can carry tool calls")
        calls = []
        for call in item["tool_calls"]:
            function = call["function"]
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
            calls.append(
                {"type": "function", "function": {"name": function["name"], "arguments": arguments}}
            )
        message["tool_calls"] = calls
    if item["role"] == "tool" and item.get("tool_call_id"):
        message["tool_call_id"] = item["tool_call_id"]
    return message


def _render(tokenizer: Any, messages: list, tools: Any, kwargs: dict, **options: Any):
    if tools:
        options["tools"] = tools
    return tokenizer.apply_chat_template(messages, return_dict=False, **options, **kwargs)


def _previous_turn(items: list) -> int | None:
    """The index of the last assistant message that carries its exact training token IDs."""
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if item.get("role") == "assistant" and all(
            isinstance(item.get(key), list) for key in _TRAINING_KEYS
        ):
            return index
    return None


def _prompt_ids(body: dict, tokenizer: Any) -> list[int]:
    items = body["messages"]
    messages = [_template_message(item) for item in items]
    tools = body.get("tools") or None
    kwargs = dict(body.get("chat_template_kwargs") or {})
    index = _previous_turn(items)
    if index is None:
        return _render(
            tokenizer, messages, tools, kwargs, tokenize=True, add_generation_prompt=True
        )
    # Keep earlier turns' text as generated; the Nemotron templates otherwise rewrite them.
    kwargs.setdefault("truncate_history_thinking", False)
    head = _render(
        tokenizer, messages[: index + 1], tools, kwargs, tokenize=False, add_generation_prompt=False
    )
    full = _render(tokenizer, messages, tools, kwargs, tokenize=False, add_generation_prompt=True)
    end = getattr(tokenizer, "eos_token", None)
    if not full.startswith(head) or not end or end not in head:
        raise ValueError(
            "The chat template renders the earlier turns differently in a later turn, so the "
            "trajectory's token prefix cannot be kept"
        )
    previous = items[index]
    generated = list(previous["generation_token_ids"])
    # A turn cut off by max_tokens has no end-of-turn token; close it in the next prompt.
    closed = bool(generated) and tokenizer.decode(generated[-1:]) == end
    suffix = ("" if closed else end) + head[head.rindex(end) + len(end) :] + full[len(head) :]
    added = tokenizer.encode(suffix, add_special_tokens=False)
    return list(previous["prompt_token_ids"]) + generated + list(added)


def completion_request(
    body: dict, tokenizer: Any, max_model_len: int | None = None, *, cache_salt: str | None = None
) -> dict:
    """Convert a text chat request, with optional tools, into a token-ID completions request.

    With ``max_model_len``, ``max_tokens`` is capped to the room the prompt leaves, so a long
    trajectory generates until its context is full instead of failing the request. A
    ``cache_salt`` goes into ``nvext``.

    Raises:
        ContextFull: The prompt leaves no room for a generated token.
        ValueError: The request has a response format, ``n`` other than 1 or non-text content, or
            a later turn cannot keep the earlier turns' exact token IDs.
    """
    if body.get("response_format") or body.get("n", 1) != 1:
        raise ValueError("CoreWeave Gym training supports one text completion per request")
    prompt = _prompt_ids(body, tokenizer)
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
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    if max_model_len:
        room = int(max_model_len) - len(prompt)
        if room < 1:
            raise ContextFull(f"prompt of {len(prompt)} tokens fills max_model_len {max_model_len}")
        max_tokens = min(max_tokens or room, room)
    result.update(
        prompt=prompt,
        max_tokens=max_tokens,
        logprobs=0,
        return_tokens_as_token_ids=True,
    )
    if cache_salt:
        result = with_nvext(result, cache_salt=cache_salt)
    return result


def chat_response(response: dict, prompt: list[int], tools: list | None = None) -> dict:
    """Convert a completions response into a chat response with token IDs and log-probs.

    Tool calls in the generated text become the message's ``tool_calls``.

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
    content, tool_calls = parse_tool_calls(choice["text"], tools)
    message = {
        "role": "assistant",
        "content": content,
        "prompt_token_ids": prompt,
        "generation_token_ids": ids,
        "generation_log_probs": probs,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        **{k: response[k] for k in ("id", "created", "model", "usage")},
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else choice["finish_reason"],
                "message": message,
            }
        ],
    }


def run_server() -> None:
    # These dependencies come from NeMo-RL's Gym actor environment, not base cwtrain.
    import nemo_gym  # noqa: F401 - adds the Gym install root to sys.path
    from aiohttp import ClientResponseError, RequestInfo
    from multidict import CIMultiDict, CIMultiDictProxy
    from nemo_gym.openai_utils import NeMoGymAsyncOpenAI, get_response_json
    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
    from transformers import AutoTokenizer
    from yarl import URL

    class Config(VLLMModelConfig):
        tokenizer_path: str
        max_model_len: int | None = None
        cache_salt_file: str | None = None

    def bad_request(url: str, content: bytes) -> ClientResponseError:
        info = RequestInfo(URL(url), "POST", CIMultiDictProxy(CIMultiDict()), URL(url))
        error = ClientResponseError(info, (), status=400, message="Bad Request")
        error.response_content = content  # what Gym's raise_for_status attaches
        return error

    class Client(NeMoGymAsyncOpenAI):
        tokenizer: Any
        max_model_len: int | None = None
        cache_salt_file: str | None = None

        async def create_chat_completion(self, **kwargs):
            url = self.base_url + "/completions"
            salt = served_salt(self.cache_salt_file) if self.cache_salt_file else None
            if self.cache_salt_file and salt is None:
                log.warning(
                    "no served identity in %s yet; sending no cache salt", self.cache_salt_file
                )
            try:
                body = completion_request(
                    kwargs, self.tokenizer, self.max_model_len, cache_salt=salt
                )
            except ContextFull:
                raise bad_request(url, CONTEXT_FULL) from None
            request = {"url": url, "json": body}
            response = await self._request(method="POST", **request)
            if response.status == 400:
                content = await response.read()
                overflow = is_context_overflow(400, content.decode(errors="replace"))
                raise bad_request(url, CONTEXT_FULL if overflow else content)
            await self._raise_for_status(response, request)
            return chat_response(
                await get_response_json(response), body["prompt"], kwargs.get("tools")
            )

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
                    max_model_len=self.config.max_model_len,
                    cache_salt_file=self.config.cache_salt_file,
                )
                for url in self.config.base_url
            ]

    Model.run_webserver()


if __name__ == "__main__":
    run_server()
