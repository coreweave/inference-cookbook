# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Configuration and data bridge for NeMo-RL's Gym rollout path.

This module also runs in Gym's separate CPU actor environment. So it does not import Gym,
NeMo-RL, torch or Ray at module scope. Credentials are resolved inside that actor.

A Gym environment that a recipe or example supplies can ask the actor for three things in
``env.nemo_gym``, which stay out of the logged config until the actor applies them:

- ``coreweave_secrets``: config field (a dotted path in the Gym config) -> key file. The actor
  reads each file and sets the field, such as a judge model's ``api_key``.
- ``coreweave_files``: path in the Gym checkout -> file to copy there, such as a resources
  server's entrypoint.
- ``coreweave_env``: environment variables for Gym's servers, which inherit the actor's.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Iterable, Mapping
from itertools import islice
from pathlib import Path
from typing import Any

from cwtrain.integrations._env import flag
from cwtrain.integrations.nvext import SERVED_IDENTITY_FILE
from cwtrain.integrations.rollout import RolloutConfig


def dapo_example(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a DAPO row to a Gym example with the question and ground truth.

    Gym does the sampling and verification.
    """
    question = row["prompt"][0]["content"]
    return {
        "agent_ref": {"name": "math_with_judge_simple_agent"},
        "responses_create_params": {"input": [{"role": "user", "content": question}]},
        "question": question,
        "expected_answer": str(row["reward_model"]["ground_truth"]),
    }


def write_dapo(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("w") as output:
            for row in rows:
                output.write(json.dumps(dapo_example(row)) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_config(config: dict[str, Any], *, rows=None) -> dict[str, Any]:
    """Check a CoreWeave Gym training config, and fill in the DAPO preset when it is selected.

    The config is returned as an unchanged copy unless the backend is ``coreweave`` and
    ``env.should_use_nemo_gym`` is set. With ``data.train.dataset_name: DAPOMath17K``, the DAPO
    dataset is written as Gym data and the Gym configs are set. Otherwise the config must name an
    explicit ``NemoGymDataset`` and ``env.nemo_gym.config_paths``.

    Args:
        config: The NeMo-RL config. It is not modified; a changed copy is returned.
        rows: DAPO rows to use instead of downloading the dataset.

    Raises:
        ValueError: The config does not meet a CoreWeave Gym training requirement, such as
            synchronous GRPO, ``api_key_file``, hot-load waiting or ``max_model_len``.
    """
    config = copy.deepcopy(config)
    generation = config["policy"]["generation"]
    if generation["backend"] != "coreweave" or not config["env"].get("should_use_nemo_gym"):
        return config
    if config["grpo"].get("async_grpo", {}).get("enabled"):
        raise ValueError("CoreWeave Gym requires synchronous GRPO to preserve the refit barrier")
    cw = generation["coreweave_cfg"]
    if cw.get("api_key"):
        raise ValueError("Gym training requires api_key_file, not a key in the logged config")
    wait = cw.get("wait")
    if wait is None:
        wait = os.environ.get("CWTRAIN_HOTLOAD_WAIT", "true")
    if str(wait).lower() not in {"1", "true", "yes", "on"}:
        raise ValueError("Gym on-policy training requires waiting for hot-load completion")
    if not cw.get("max_model_len") or int(cw["max_model_len"]) <= 0:
        raise ValueError("Gym requires a positive coreweave_cfg.max_model_len")
    generation["stop_strings"] = None
    generation["stop_token_ids"] = None
    if generation.get("top_k"):
        raise ValueError("Gym does not support top_k; set policy.generation.top_k=null")
    # Gym reads the metadata directly, not through CoreWeaveGeneration.generate().
    # Training must never substitute missing log-probs or token IDs without notice.
    cw["require_logprobs"] = True
    gym = config["env"].setdefault("nemo_gym", {})
    gym["coreweave_api_key_file"] = cw.get("api_key_file")
    gym["coreweave_training"] = True
    gym["coreweave_tokenizer"] = config["policy"]["tokenizer"]["name"]
    # where the weight synchronizer records the identity the deployment serves
    state_dir = cw.get("state_dir") or os.environ.get("CWTRAIN_HOTLOAD_STATE_DIR", "").strip()
    # Absolute: Gym's servers do not start in the driver's working directory.
    served = Path(state_dir or "/work/hotload").resolve() / SERVED_IDENTITY_FILE
    gym.setdefault("policy_model", {}).setdefault("responses_api_models", {}).setdefault(
        "vllm_model", {}
    ).update(
        {
            "return_token_id_information": True,
            "chat_template_kwargs": config["policy"]["tokenizer"].get("chat_template_kwargs"),
            # cwtrain's Gym model caps max_tokens to the room the prompt leaves.
            "max_model_len": int(cw["max_model_len"]),
            # and salts each request with the served identity (cwtrain.integrations.nvext)
            "cache_salt_file": str(served),
            # and seeds each one, unless coreweave_cfg.seeds or CWTRAIN_ROLLOUT_SEEDS turns it off
            "seeds": _seeds(cw),
        }
    )
    for field, path in (gym.get("coreweave_secrets") or {}).items():
        if not path or not Path(path).is_file():
            raise ValueError(f"Gym secret for {field}: key file {path} does not exist; mount it")
    for target, source in (gym.get("coreweave_files") or {}).items():
        if Path(target).is_absolute() or ".." in Path(target).parts:
            raise ValueError(f"coreweave_files target {target} must be inside the Gym checkout")
        if not Path(source).is_file():
            raise ValueError(f"coreweave_files source {source} does not exist")
    data = config["data"]
    train = data["train"]
    if isinstance(train, dict) and train.get("dataset_name") == "DAPOMath17K":
        if data.get("validation") or train.get("split_validation_size", 0):
            raise ValueError(
                "The DAPO Gym preset requires validation disabled; use explicit Gym data"
            )
        if gym.get("config_paths"):
            raise ValueError("Custom Gym environments require an explicit NemoGymDataset")
        if rows is None:
            from datasets import load_dataset

            rows = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        max_samples = train.get("max_samples")
        if max_samples is not None:
            if type(max_samples) is not int or max_samples <= 0:
                raise ValueError("data.train.max_samples must be a positive integer")
            rows = islice(rows, max_samples)
        path = Path(cw.get("state_dir", "/work/hotload")) / "nemo-gym" / "dapo.jsonl"
        write_dapo(rows, path)
        data["train"] = {"dataset_name": "NemoGymDataset", "data_path": str(path)}
        data["validation"] = None
        data["max_input_seq_length"] = None
        data["default"] = {
            "dataset_name": "NemoGymDataset",
            "env_name": "nemo_gym",
            "processor": "nemo_gym_data_processor",
            "prompt_file": None,
            "system_prompt_file": None,
        }
        gym["config_paths"] = [
            "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
            "resources_servers/math_with_judge/configs/math_with_judge.yaml",
        ]
        # This verifier config uses math-verify, with its LLM judge disabled.
        config["grpo"]["max_val_samples"] = None
    else:
        datasets = train if isinstance(train, list) else [train]
        if any(item.get("dataset_name") != "NemoGymDataset" for item in datasets):
            raise ValueError("Gym requires DAPOMath17K or an explicit NemoGymDataset")
        if not gym.get("config_paths"):
            raise ValueError("Custom Gym data requires env.nemo_gym.config_paths")
    return config


def _seeds(cw: Mapping[str, Any]) -> bool:
    """``coreweave_cfg.seeds``, else ``CWTRAIN_ROLLOUT_SEEDS``, as the generation backend does."""
    if cw.get("seeds") is None:
        return flag(os.environ, "CWTRAIN_ROLLOUT_SEEDS", True)
    return flag({"coreweave_cfg.seeds": str(cw["seeds"])}, "coreweave_cfg.seeds", True)


def actor_config(initial: dict[str, Any], environ=None) -> dict[str, Any]:
    """Resolve the inference key inside the Gym actor.

    The key is read here, after the config is serialized and logged, so it never appears in that
    config.

    Raises:
        ValueError: ``api_key_file`` is set but no key can be read from it.
    """
    initial = copy.deepcopy(initial)
    if not initial.pop("coreweave_training", False):
        return initial
    env = dict(os.environ if environ is None else environ)
    key_file = initial.pop("coreweave_api_key_file", None)
    if key_file:
        env["CWTRAIN_ROLLOUT_API_KEY_FILE"] = key_file
    urls = initial["policy_base_url"]
    env.update(
        CWTRAIN_ROLLOUT_BASE_URL=urls[0] if isinstance(urls, list) else urls,
        CWTRAIN_ROLLOUT_MODEL=initial["policy_model_name"],
    )
    rollout = RolloutConfig.from_env(env).without_routed_experts("NeMo-RL")
    if key_file and not rollout.api_key:
        raise ValueError("Gym actor cannot read a nonempty inference key from api_key_file")
    tokenizer = initial.pop("coreweave_tokenizer", None)
    if tokenizer:
        import sys

        import nemo_gym

        # Run in the actor's environment, which already has Gym, transformers and cwtrain.
        # Gym's model-server sub-venv then does not need its own copy of those packages.
        target = nemo_gym.PARENT_DIR / "responses_api_models" / "vllm_model" / "cwtrain_app.py"
        target.write_text(
            "import os\n"
            f"os.execv({sys.executable!r}, [{sys.executable!r}, "
            "'-m', 'cwtrain.integrations.nemo_rl.gym_model'])\n"
        )
        model = initial["policy_model"]["responses_api_models"]["vllm_model"]
        model.update(entrypoint="cwtrain_app.py", tokenizer_path=tokenizer, num_workers=1)
    initial["policy_api_key"] = rollout.api_key or "unused"
    # Gym's servers inherit the actor's environment.
    os.environ.update(
        {str(k): str(v) for k, v in (initial.pop("coreweave_env", None) or {}).items()}
    )
    files = initial.pop("coreweave_files", None) or {}
    if files:
        import shutil

        import nemo_gym

        for target, source in files.items():
            shutil.copyfile(source, nemo_gym.PARENT_DIR / target)
    for field, path in (initial.pop("coreweave_secrets", None) or {}).items():
        _set_field(initial, field, _read_key(path, field))
    return initial


def _read_key(path: str | None, field: str) -> str:
    try:
        key = Path(path).read_text().strip() if path else ""
    except OSError:
        key = ""
    if not key:
        raise ValueError(f"Gym actor cannot read a nonempty key for {field} from {path}")
    return key


def _set_field(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``config[a][b][c] = value`` for ``dotted == "a.b.c"``; every parent must exist."""
    *parents, leaf = dotted.split(".")
    node = config
    for name in parents:
        if not isinstance(node.get(name), dict):
            raise ValueError(f"Gym config has no {dotted}: {name} is missing")
        node = node[name]
    node[leaf] = value


def validate_training_output(result: Mapping[str, Any]) -> None:
    """Check Gym's training metadata before loss masks are built.

    Raises:
        ValueError: Token IDs or log-probs are missing or misaligned, a turn does not keep the
            previous turn's tokens as its prefix, or there are no generated tokens.
    """
    generated = 0
    previous: list[int] = []
    for item in result["response"]["output"]:
        fields = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")
        if not any(field in item for field in fields):
            continue
        if not all(isinstance(item.get(field), list) for field in fields):
            raise ValueError("Gym training requires prompt IDs, generated IDs and log-probs")
        prompt, tokens, logprobs = (item[field] for field in fields)
        if any(type(token) is not int or token < 0 for token in prompt + tokens):
            raise ValueError("Gym training requires exact integer token IDs")
        if len(tokens) != len(logprobs) or any(not math.isfinite(lp) for lp in logprobs):
            raise ValueError("Gym training requires one finite log-prob per generated token")
        if prompt[: len(previous)] != previous:
            raise ValueError("Gym training requires exact token prefix preservation across turns")
        previous = prompt + tokens
        generated += len(tokens)
    if not generated:
        raise ValueError(
            "Gym returned no trainable tokens; check endpoint training metadata support"
        )
