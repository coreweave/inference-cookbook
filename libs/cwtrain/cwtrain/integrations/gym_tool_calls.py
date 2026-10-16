# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Tool calls for Gym's ``vllm_model`` from the Qwen3-coder XML text the Nemotron 3 models write.

A deployment without a tool-call parser returns the model's
``<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`` text as the
message content, so Gym's agents see no tool call. ``install`` copies this file into Gym's
``vllm_model`` directory as ``cwtrain_tool_app.py``. A model server whose ``entrypoint`` is that
file parses the text into the message's ``tool_calls``. The copy runs in the model server's own
venv, which has no cwtrain, so this module imports only the standard library at module scope.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

MODEL_DIR = Path("responses_api_models") / "vllm_model"
ENTRYPOINT = "cwtrain_tool_app.py"
_TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _argument(value: str, schema: dict | None) -> Any:
    """A tool argument as its schema type: strings stay text, other types are parsed as JSON."""
    if (schema or {}).get("type", "string") == "string":
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def parse_tool_calls(text: str, tools: list | None = None) -> tuple[str, list[dict]]:
    """Split generated text into its content and its Qwen3-coder XML tool calls."""
    schemas = {}
    for tool in tools or []:
        function = tool.get("function", tool)
        schemas[function.get("name")] = (function.get("parameters") or {}).get("properties") or {}
    calls = []
    for match in _TOOL_CALL.finditer(text):
        name = match.group(1).strip()
        properties = schemas.get(name, {})
        arguments = {
            key.strip(): _argument(value, properties.get(key.strip()))
            for key, value in _PARAMETER.findall(match.group(2))
        }
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        )
    if not calls:
        return text, []
    return text[: text.index("<tool_call>")].rstrip(), calls


def with_tool_calls(response: dict, tools: list | None = None) -> dict:
    """Move tool-call text in a chat completion's messages into their ``tool_calls``.

    Messages that already have ``tool_calls``, or that hold no tool-call text, are unchanged.
    """
    if not tools:
        return response
    for choice in response.get("choices") or []:
        message = choice.get("message") or {}
        content = message.get("content")
        if (
            message.get("tool_calls")
            or not isinstance(content, str)
            or "<tool_call>" not in content
        ):
            continue
        content, calls = parse_tool_calls(content, tools)
        if calls:
            message.update(content=content, tool_calls=calls)
            choice["finish_reason"] = "tool_calls"
    return response


def install(gym_root: Path) -> Path:
    """Copy this module into Gym's ``vllm_model`` server as its tool-call entrypoint.

    Raises:
        FileNotFoundError: ``gym_root`` has no ``vllm_model`` server.
    """
    model_dir = Path(gym_root) / MODEL_DIR
    if not (model_dir / "app.py").is_file():
        raise FileNotFoundError(f"no vllm_model server at {model_dir}")
    target = model_dir / ENTRYPOINT
    target.write_text(Path(__file__).read_text())
    return target


def run_server() -> None:
    """Start Gym's ``vllm_model`` server, with tool calls parsed from the generated text."""
    import nemo_gym  # noqa: F401 - adds the Gym install root to sys.path
    from responses_api_models.vllm_model.app import VLLMModel

    class ToolCallModel(VLLMModel):
        def _resolve_client(self, request):
            client = super()._resolve_client(request)
            if not getattr(client, "_cwtrain_tool_calls", False):
                create = client.create_chat_completion

                async def create_chat_completion(**kwargs):
                    return with_tool_calls(await create(**kwargs), kwargs.get("tools"))

                # The client is a pydantic model, so set the wrapper past its field checks.
                object.__setattr__(client, "create_chat_completion", create_chat_completion)
                object.__setattr__(client, "_cwtrain_tool_calls", True)
            return client

    ToolCallModel.run_webserver()


if __name__ == "__main__":
    run_server()
