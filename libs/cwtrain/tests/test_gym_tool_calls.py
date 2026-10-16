# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import pytest


def test_tool_call_text_becomes_tool_calls():
    import json

    from cwtrain.integrations.gym_tool_calls import with_tool_calls

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {"properties": {"queries": {"type": "array"}}},
            },
        }
    ]
    text = (
        "Searching.\n<tool_call>\n<function=search>\n<parameter=queries>\n"
        '["a"]\n</parameter>\n</function>\n</tool_call>\n'
    )
    response = {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
    choice = with_tool_calls(response, tools)["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Searching."
    call = choice["message"]["tool_calls"][0]["function"]
    assert call["name"] == "search" and json.loads(call["arguments"]) == {"queries": ["a"]}
    # without tools, or with tool calls already parsed, the response is unchanged
    plain = {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
    assert with_tool_calls(plain, None)["choices"][0]["message"]["content"] == text
    parsed = {"choices": [{"message": {"content": "", "tool_calls": [{"id": "x"}]}}]}
    assert with_tool_calls(parsed, tools)["choices"][0]["message"]["tool_calls"] == [{"id": "x"}]


def test_tool_call_entrypoint_is_self_contained(tmp_path):
    from cwtrain.integrations import gym_tool_calls

    with pytest.raises(FileNotFoundError):
        gym_tool_calls.install(tmp_path)
    model = tmp_path / gym_tool_calls.MODEL_DIR
    model.mkdir(parents=True)
    (model / "app.py").write_text("")
    source = gym_tool_calls.install(tmp_path).read_text()
    assert "import cwtrain" not in source and "from cwtrain" not in source
