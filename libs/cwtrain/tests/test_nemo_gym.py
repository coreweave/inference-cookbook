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
    model = result["env"]["nemo_gym"]["policy_model"]["responses_api_models"]["vllm_model"]
    # the Gym model server salts with what the weight synchronizer records in the state dir
    state_dir = config["policy"]["generation"]["coreweave_cfg"]["state_dir"]
    assert model["cache_salt_file"] == str(Path(state_dir).resolve() / "served_identity")


def test_the_served_identity_path_is_absolute(config, monkeypatch, tmp_path):
    # Gym's servers do not start in the driver's working directory
    monkeypatch.chdir(tmp_path)
    config["policy"]["generation"]["coreweave_cfg"]["state_dir"] = "relative/state"
    rows = [
        {"prompt": [{"role": "user", "content": "1 + 1?"}], "reward_model": {"ground_truth": "2"}}
    ]
    model = prepare_config(config, rows=rows)["env"]["nemo_gym"]["policy_model"][
        "responses_api_models"
    ]["vllm_model"]
    assert model["cache_salt_file"] == str(tmp_path.resolve() / "relative/state/served_identity")


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


@pytest.fixture
def custom(config, tmp_path):
    """A custom Gym environment that asks the actor for a key, a file and an env variable."""
    key, entry = tmp_path / "judge_key", tmp_path / "search_app.py"
    key.write_text("judge-key\n")
    entry.write_text("# a search backend\n")
    config["data"]["train"] = {"dataset_name": "NemoGymDataset", "data_path": "/data/tasks.jsonl"}
    config["env"]["nemo_gym"] = {
        "config_paths": ["responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"],
        "judge_model": {"responses_api_models": {"vllm_model": {"api_key": "unused"}}},
        "coreweave_secrets": {"judge_model.responses_api_models.vllm_model.api_key": str(key)},
        "coreweave_files": {"resources_servers/search/search_app.py": str(entry)},
        "coreweave_env": {"SEARCH_MAX_CHARS": 2000},
    }
    return config


def test_the_policy_model_knows_the_context_limit(custom):
    gym = prepare_config(custom)["env"]["nemo_gym"]
    assert gym["policy_model"]["responses_api_models"]["vllm_model"]["max_model_len"] == 3072


def test_actor_applies_secrets_files_and_env_after_transport(custom, tmp_path, monkeypatch):
    import os
    import sys
    from types import SimpleNamespace

    monkeypatch.delenv("SEARCH_MAX_CHARS", raising=False)
    gym = prepare_config(custom)["env"]["nemo_gym"]
    assert "judge-key" not in json.dumps(gym)  # only the key file's path travels
    root = tmp_path / "gym"
    (root / "resources_servers" / "search").mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "nemo_gym", SimpleNamespace(PARENT_DIR=root))
    gym.pop("coreweave_tokenizer")
    initial = dict(gym, policy_base_url=["https://example.test/v1"], policy_model_name="m")
    result = actor_config(initial, {})
    assert result["judge_model"]["responses_api_models"]["vllm_model"]["api_key"] == "judge-key"
    copied = root / "resources_servers" / "search" / "search_app.py"
    assert copied.read_text() == "# a search backend\n"
    assert os.environ["SEARCH_MAX_CHARS"] == "2000"
    assert not any(key.startswith("coreweave_") for key in result)


@pytest.mark.parametrize(
    ("directive", "value", "error"),
    [
        ("coreweave_secrets", {"judge.api_key": "/absent/key"}, "does not exist"),
        ("coreweave_files", {"../outside.py": __file__}, "inside the Gym checkout"),
        ("coreweave_files", {"resources_servers/x/app.py": "/absent.py"}, "does not exist"),
    ],
)
def test_rejects_unusable_directives(custom, directive, value, error):
    custom["env"]["nemo_gym"][directive] = value
    with pytest.raises(ValueError, match=error):
        prepare_config(custom)


def test_actor_rejects_a_secret_for_a_missing_field(custom, tmp_path):
    custom["env"]["nemo_gym"].pop("judge_model")
    gym = prepare_config(custom)["env"]["nemo_gym"]
    gym.pop("coreweave_tokenizer")
    gym.pop("coreweave_files")
    initial = dict(gym, policy_base_url=["https://example.test/v1"], policy_model_name="m")
    with pytest.raises(ValueError, match="judge_model is missing"):
        actor_config(initial, {})
