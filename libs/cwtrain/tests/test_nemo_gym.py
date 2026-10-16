# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import copy
import json

import pytest

from cwtrain.integrations.nemo_rl.gym import actor_config, prepare_config


@pytest.fixture
def config(tmp_path):
    return {
        "policy": {
            "tokenizer": {"name": "/model", "chat_template_kwargs": {"enable_thinking": False}},
            "generation": {
                "backend": "coreweave",
                "coreweave_cfg": {"state_dir": str(tmp_path), "max_model_len": 3072},
            },
        },
        "env": {"should_use_nemo_gym": True},
        "data": {"train": {"dataset_name": "DAPOMath17K", "split_validation_size": 0}},
        "grpo": {"max_val_samples": 10, "max_rollout_turns": 1},
    }


def test_flag_converts_dapo_without_mutating_source(config):
    original = copy.deepcopy(config)
    result = prepare_config(
        config,
        rows=[
            {
                "prompt": [{"role": "user", "content": "Compute 2 + 2."}],
                "reward_model": {"ground_truth": "4"},
            }
        ],
    )
    assert config == original
    from pathlib import Path

    row = json.loads(Path(result["data"]["train"]["data_path"]).read_text())
    assert row["expected_answer"] == "4"
    assert row["question"] == row["responses_create_params"]["input"][0]["content"]
    assert row["agent_ref"]["name"] == "math_with_judge_simple_agent"
    assert result["grpo"]["max_val_samples"] is None
    assert result["grpo"]["max_rollout_turns"] == 1
    assert result["policy"]["generation"]["coreweave_cfg"]["require_logprobs"] is True
    assert result["data"]["default"]["processor"] == "nemo_gym_data_processor"


def test_disabled_flag_does_not_load_data(config):
    config["env"]["should_use_nemo_gym"] = False
    assert prepare_config(config) == config


@pytest.mark.parametrize("wait", [False, "false", "0", "off"])
def test_rejects_nonblocking_hotload(config, wait):
    config["policy"]["generation"]["coreweave_cfg"]["wait"] = wait
    with pytest.raises(ValueError, match="waiting"):
        prepare_config(config, rows=[])


def test_rejects_env_nonblocking_hotload(config, monkeypatch):
    monkeypatch.setenv("CWTRAIN_HOTLOAD_WAIT", "false")
    with pytest.raises(ValueError, match="waiting"):
        prepare_config(config, rows=[])
    config["policy"]["generation"]["coreweave_cfg"]["wait"] = True
    prepare_config(config, rows=[])


def test_rejects_inline_key(config):
    config["policy"]["generation"]["coreweave_cfg"]["api_key"] = "secret"
    with pytest.raises(ValueError, match="api_key_file"):
        prepare_config(config, rows=[])


def test_custom_gym_dataset_is_preserved(config):
    config["data"]["train"] = {"dataset_name": "NemoGymDataset", "data_path": "/data/tasks.jsonl"}
    config["env"]["nemo_gym"] = {"config_paths": ["custom.yaml"]}
    result = prepare_config(config)
    assert result["data"] == config["data"]
    assert result["env"]["nemo_gym"]["config_paths"] == ["custom.yaml"]


def test_rejects_non_gym_dataset(config):
    config["data"]["train"]["dataset_name"] = "Other"
    with pytest.raises(ValueError, match="NemoGymDataset"):
        prepare_config(config)


def test_actor_resolves_file_only_after_transport(tmp_path):
    key = tmp_path / "key"
    key.write_text("test-key\n")
    initial = {
        "coreweave_training": True,
        "coreweave_api_key_file": str(key),
        "policy_base_url": ["https://example.test/v1"],
        "policy_model_name": "qwen",
    }
    result = actor_config(initial, {})
    assert result["policy_api_key"] == "test-key"
    assert "policy_api_key" not in initial
    assert "coreweave_api_key_file" not in result
    assert "coreweave_training" not in result


def test_non_coreweave_actor_unchanged():
    initial = {"policy_api_key": "dummy"}
    assert actor_config(initial, {}) == initial


def test_rejects_async_training(config):
    config["grpo"]["async_grpo"] = {"enabled": True}
    with pytest.raises(ValueError, match="synchronous GRPO"):
        prepare_config(config, rows=[])


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"generation_token_ids": [1]},
        {"prompt_token_ids": [0], "generation_token_ids": [1, 2], "generation_log_probs": [-0.1]},
        {
            "prompt_token_ids": [0],
            "generation_token_ids": [1],
            "generation_log_probs": [float("nan")],
        },
        {"prompt_token_ids": [0], "generation_token_ids": ["1"], "generation_log_probs": [-0.1]},
    ],
)
def test_rejects_untrainable_metadata(item):
    from cwtrain.integrations.nemo_rl.gym import validate_training_output

    with pytest.raises(ValueError):
        validate_training_output({"response": {"output": [item]}})


def test_metadata_allows_tool_items_without_training_tokens():
    from cwtrain.integrations.nemo_rl.gym import validate_training_output

    validate_training_output(
        {
            "response": {
                "output": [
                    {"type": "function_call_output", "output": "4"},
                    {
                        "prompt_token_ids": [0],
                        "generation_token_ids": [1],
                        "generation_log_probs": [-0.1],
                    },
                ]
            }
        }
    )


def test_metadata_rejects_retokenized_multiturn_prefix():
    from cwtrain.integrations.nemo_rl.gym import validate_training_output

    result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [0],
                    "generation_token_ids": [1],
                    "generation_log_probs": [-0.1],
                },
                {
                    "prompt_token_ids": [0, 9],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.2],
                },
            ]
        }
    }
    with pytest.raises(ValueError, match="prefix preservation"):
        validate_training_output(result)
    result["response"]["output"][1]["prompt_token_ids"] = [0, 1, 9]
    validate_training_output(result)


def test_smoke_sample_limit(config):
    from pathlib import Path

    config["data"]["train"]["max_samples"] = 2
    rows = [
        {"prompt": [{"content": str(i)}], "reward_model": {"ground_truth": str(i)}}
        for i in range(5)
    ]
    result = prepare_config(config, rows=rows)
    assert len(Path(result["data"]["train"]["data_path"]).read_text().splitlines()) == 2


@pytest.mark.parametrize("limit", [0, -1, "3", True])
def test_invalid_smoke_sample_limit(config, limit):
    config["data"]["train"]["max_samples"] = limit
    with pytest.raises(ValueError, match="positive integer"):
        prepare_config(config, rows=[])


def test_actor_installs_exact_token_transport(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    server = tmp_path / "responses_api_models" / "vllm_model"
    server.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "nemo_gym", SimpleNamespace(PARENT_DIR=tmp_path))
    initial = {
        "coreweave_training": True,
        "coreweave_tokenizer": "/model",
        "policy_base_url": ["https://example.test/v1"],
        "policy_model_name": "nemotron",
        "policy_model": {"responses_api_models": {"vllm_model": {"model": "nemotron"}}},
    }
    original = copy.deepcopy(initial)
    result = actor_config(initial, {})
    assert initial == original
    assert "coreweave_tokenizer" not in result
    model = result["policy_model"]["responses_api_models"]["vllm_model"]
    assert model["tokenizer_path"] == "/model"
    assert model["entrypoint"] == "cwtrain_app.py"
    assert model["num_workers"] == 1
    assert "cwtrain.integrations.nemo_rl.gym_model" in (server / "cwtrain_app.py").read_text()


def encrypted(text: str, canary: str) -> str:
    import base64
    import hashlib

    data = text.encode()
    digest = hashlib.sha256(canary.encode()).digest()
    key = (digest * (len(data) // len(digest) + 1))[: len(data)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key, strict=True))).decode()


@pytest.fixture
def browsecomp(config, tmp_path):
    judge, search = tmp_path / "judge", tmp_path / "search"
    judge.write_text("judge-key\n")
    search.write_text("you-key\n")
    entrypoint = tmp_path / "you_search.py"
    entrypoint.write_text("# a search backend\n")
    config["data"]["train"]["dataset_name"] = "BrowseComp"
    config["env"]["nemo_gym"] = {
        "search": {
            "judge_model": "zai-org/GLM-5.2",
            "judge_base_url": "https://api.inference.wandb.ai/v1",
            "judge_api_key_file": str(judge),
            "search_api_key_file": str(search),
            "search_entrypoint": str(entrypoint),
            "search_env": {"YOU_SEARCH_MAX_CHARS": 2000},
            "max_steps": 6,
        }
    }
    rows = [
        {"problem": encrypted(f"Who wrote book {i}?", "c"), "answer": encrypted(f"A{i}", "c"),
         "canary": "c"}
        for i in range(3)
    ]  # fmt: skip
    return config, rows


def test_browsecomp_preset_writes_tasks_and_servers(browsecomp):
    from pathlib import Path

    config, rows = browsecomp
    result = prepare_config(config, rows=rows)
    lines = Path(result["data"]["train"]["data_path"]).read_text().splitlines()
    row = json.loads(lines[0])
    assert len(lines) == 3
    assert row["agent_ref"]["name"] == "browsecomp_agent"
    assert row["question"] == "Who wrote book 0?" and row["ground_truth"] == "A0"
    assert row["responses_create_params"]["input"][1]["content"].startswith("Who wrote book 0?")
    assert [t["name"] for t in row["responses_create_params"]["tools"]] == ["search", "browse"]
    gym = result["env"]["nemo_gym"]
    assert "search" not in gym  # consumed, not passed on to Gym
    judge = gym["judge_model"]["responses_api_models"]["vllm_model"]
    assert judge["model"] == "zai-org/GLM-5.2" and judge["api_key"] == "unused"
    assert judge["extra_body"]["max_tokens"] == 8192
    harness = gym["browsecomp_resources_server"]["resources_servers"]["browsecomp_advanced_harness"]
    assert harness["entrypoint"] == "you_search.py"
    assert json.loads(Path(harness["exclude_domains_file_path"]).read_text())["notices"]
    agent = gym["browsecomp_agent"]["responses_api_agents"]["browsecomp_agent"]
    assert agent["max_steps"] == 6 and agent["max_reset_count"] == 0
    assert agent["max_context_tokens"] == 3072
    assert gym["config_paths"] == [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
    ]
    # Keys travel as file paths; the actor reads them.
    assert "judge-key" not in json.dumps(result) and "you-key" not in json.dumps(result)


def test_browsecomp_sample_limit(browsecomp):
    from pathlib import Path

    config, rows = browsecomp
    config["data"]["train"]["max_samples"] = 1
    result = prepare_config(config, rows=rows)
    assert len(Path(result["data"]["train"]["data_path"]).read_text().splitlines()) == 1


@pytest.mark.parametrize("missing", ["judge_api_key_file", "search_api_key_file", "judge_model"])
def test_browsecomp_requires_judge_and_search_settings(browsecomp, missing):
    config, rows = browsecomp
    config["env"]["nemo_gym"]["search"].pop(missing)
    with pytest.raises(ValueError, match=missing):
        prepare_config(config, rows=rows)


def test_browsecomp_rejects_an_unmounted_key_file(browsecomp, tmp_path):
    config, rows = browsecomp
    config["env"]["nemo_gym"]["search"]["search_api_key_file"] = str(tmp_path / "absent")
    with pytest.raises(ValueError, match="mount its Secret"):
        prepare_config(config, rows=rows)


def test_actor_reads_judge_and_search_keys_and_installs_you_com(browsecomp, tmp_path, monkeypatch):
    import os
    import sys

    monkeypatch.delenv("YOU_SEARCH_MAX_CHARS", raising=False)
    from types import SimpleNamespace

    from cwtrain.integrations.nemo_rl.gym import HARNESS_DIR

    config, rows = browsecomp
    gym = prepare_config(config, rows=rows)["env"]["nemo_gym"]
    root = tmp_path / "gym"
    (root / HARNESS_DIR).mkdir(parents=True)
    (root / HARNESS_DIR / "app.py").write_text("")
    monkeypatch.setitem(sys.modules, "nemo_gym", SimpleNamespace(PARENT_DIR=root))
    gym.pop("coreweave_tokenizer")
    initial = dict(gym, policy_base_url=["https://example.test/v1"], policy_model_name="m")
    result = actor_config(initial, {})
    judge = result["judge_model"]["responses_api_models"]["vllm_model"]
    harness = result["browsecomp_resources_server"]["resources_servers"][
        "browsecomp_advanced_harness"
    ]
    assert judge["api_key"] == "judge-key"
    assert harness["tavily_api_key"] == "you-key"
    assert os.environ["YOU_SEARCH_MAX_CHARS"] == "2000"  # Gym's servers inherit it
    assert (root / HARNESS_DIR / "you_search.py").read_text() == "# a search backend\n"
    assert not any(key.startswith("coreweave_") for key in result)
    Path_ = type(tmp_path)
    Path_(gym["coreweave_search_api_key_file"]).write_text("")
    with pytest.raises(ValueError, match="search API key"):
        actor_config(dict(initial), {})


def test_browsecomp_decrypts_the_published_format():
    from cwtrain.integrations.nemo_rl.gym import browsecomp_decrypt

    assert browsecomp_decrypt(encrypted("a long answer " * 5, "canary"), "canary") == (
        "a long answer " * 5
    )


def test_browsecomp_uses_the_harness_app_without_a_search_entrypoint(browsecomp):
    config, rows = browsecomp
    config["env"]["nemo_gym"]["search"].pop("search_entrypoint")
    gym = prepare_config(config, rows=rows)["env"]["nemo_gym"]
    harness = gym["browsecomp_resources_server"]["resources_servers"]["browsecomp_advanced_harness"]
    assert harness["entrypoint"] == "app.py" and "coreweave_search_entrypoint" not in gym


def test_browsecomp_rejects_a_missing_search_entrypoint(browsecomp, tmp_path):
    config, rows = browsecomp
    config["env"]["nemo_gym"]["search"]["search_entrypoint"] = str(tmp_path / "absent.py")
    with pytest.raises(ValueError, match="search_entrypoint"):
        prepare_config(config, rows=rows)


def test_simpleqa_preset_uses_the_search_agent(browsecomp):
    from pathlib import Path

    config, _ = browsecomp
    config["data"]["train"]["dataset_name"] = "SimpleQA"
    rows = [{"metadata": "{}", "problem": "Who wrote Hamlet?", "answer": "William Shakespeare"}]
    result = prepare_config(config, rows=rows)
    path = Path(result["data"]["train"]["data_path"])
    row = json.loads(path.read_text())
    assert path.name == "simpleqa.jsonl"
    assert row["agent_ref"]["name"] == "browsecomp_agent"
    assert row["question"] == "Who wrote Hamlet?" and row["ground_truth"] == "William Shakespeare"
    assert row["responses_create_params"]["input"][1]["content"].startswith("Who wrote Hamlet?")
    assert "judge_model" in result["env"]["nemo_gym"]
