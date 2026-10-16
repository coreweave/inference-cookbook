# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Configuration and data bridge for NeMo-RL's Gym rollout path.

This module also runs in Gym's separate CPU actor environment. So it does not import Gym,
NeMo-RL, torch or Ray at module scope. Credentials are resolved inside that actor.
"""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import json
import math
import os
from collections.abc import Iterable, Mapping
from datetime import date
from itertools import islice
from pathlib import Path
from typing import Any

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


def write_examples(examples: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("w") as output:
            for example in examples:
                output.write(json.dumps(example) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_dapo(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    write_examples((dapo_example(row) for row in rows), path)


# BrowseComp: OpenAI's published test set, in the format of Gym's benchmarks/browsecomp/prepare.py
# (Gym v0.4.0, Apache-2.0), so that training prompts match the Gym benchmark's.
BROWSECOMP_CSV_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
)
BROWSECOMP_ROWS = 1266
# SimpleQA: OpenAI's short fact-seeking questions, in plain text. A search agent answers many of
# them in a few tool calls, so a training group gets mixed rewards.
SIMPLEQA_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
SIMPLEQA_ROWS = 4326
BROWSECOMP_AGENT = "browsecomp_agent"
BROWSECOMP_SERVER = "browsecomp_resources_server"
JUDGE_SERVER = "judge_model"
HARNESS_DIR = Path("resources_servers") / "browsecomp_advanced_harness"
BROWSECOMP_SYSTEM_PROMPT = (
    "You are a General Agent. Today's date: {date}. "
    "Your mission is to leverage a diverse set of tools to help the user conduct "
    "an in-depth investigation of their question, continuously reflect, and "
    "ultimately deliver a precise answer.\n\n"
    "Throughout the investigation, strictly observe the following principles:\n"
    "1. Whenever you encounter uncertain information, proactively invoke search "
    "tools to verify it.\n"
    "2. You can only invoke one tool in each round.\n"
    "3. Prioritize high-credibility sources (authoritative websites, academic "
    "databases, professional media) and maintain a critical stance toward "
    "low-credibility ones. Cite the source of any information you use with "
    "a format [^index^].\n"
    "4. You should not respond to the user with a counter-question, but instead "
    "do your best to provide an accurate answer.\n"
    "5. When providing the final answer, begin by explaining the reasoning "
    "process. Avoid presenting only the final answer, as this makes it "
    "difficult to understand."
)
BROWSECOMP_QUERY_SUFFIX = (
    "\n\nYour response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
BROWSECOMP_TOOLS = [
    {
        "type": "function",
        "name": "search",
        "description": (
            "Web Search API, works like Google Search. All queries will be searched in "
            "parallel. If you want to search with multiple keywords, put them in a single query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Search queries. All queries are executed in parallel.",
                }
            },
            "required": ["queries"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "browse",
        "description": (
            "Visit specific webpage(s) and return their full text content. Use this to read "
            "the complete content of web pages found during search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URL(s) of webpage(s) to visit.",
                },
                "goal": {
                    "type": "string",
                    "description": "What specific information you are looking for.",
                },
            },
            "required": ["urls"],
        },
        "strict": False,
    },
]
# Sites that publish BrowseComp questions with their answers.
BROWSECOMP_EXCLUDED_DOMAINS = ("huggingface.co",)


def browsecomp_decrypt(ciphertext: str, password: str) -> str:
    """Decrypt one BrowseComp field: base64 XORed with the SHA-256 of the row's canary."""
    data = base64.b64decode(ciphertext)
    digest = hashlib.sha256(password.encode()).digest()
    key = digest * (len(data) // len(digest)) + digest[: len(data) % len(digest)]
    return bytes(a ^ b for a, b in zip(data, key, strict=True)).decode()


def _csv_rows(url: str, expected: int, name: str) -> list[dict[str, str]]:
    import httpx

    response = httpx.get(url, timeout=120, follow_redirects=True)
    response.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(response.text)))
    if len(rows) != expected:
        raise ValueError(f"expected {expected} {name} rows, got {len(rows)}")
    return rows


def browsecomp_rows(url: str = BROWSECOMP_CSV_URL) -> list[dict[str, str]]:
    """Download the encrypted BrowseComp test set.

    Raises:
        ValueError: The download does not have the published number of rows.
    """
    return _csv_rows(url, BROWSECOMP_ROWS, "BrowseComp")


def simpleqa_rows(url: str = SIMPLEQA_CSV_URL) -> list[dict[str, str]]:
    """Download the SimpleQA test set.

    Raises:
        ValueError: The download does not have the published number of rows.
    """
    return _csv_rows(url, SIMPLEQA_ROWS, "SimpleQA")


def browsecomp_example(row: Mapping[str, str], today: date | None = None) -> dict[str, Any]:
    """Convert an encrypted BrowseComp row to a Gym example for the search agent."""
    problem = browsecomp_decrypt(row["problem"], row["canary"])
    answer = browsecomp_decrypt(row["answer"], row["canary"])
    return search_example(problem, answer, today)


def simpleqa_example(row: Mapping[str, str], today: date | None = None) -> dict[str, Any]:
    """Convert a SimpleQA row to a Gym example for the search agent."""
    return search_example(row["problem"], row["answer"], today)


def search_example(question: str, answer: str, today: date | None = None) -> dict[str, Any]:
    """A Gym example for the BrowseComp agent: its prompt and tools, the question and answer."""
    problem = question
    system = BROWSECOMP_SYSTEM_PROMPT.format(date=(today or date.today()).isoformat())
    return {
        "agent_ref": {"name": BROWSECOMP_AGENT},
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": problem + BROWSECOMP_QUERY_SUFFIX},
            ],
            "tools": copy.deepcopy(BROWSECOMP_TOOLS),
        },
        "question": problem,
        "ground_truth": answer,
    }


# The search-agent presets: data.train.dataset_name -> (download, example builder, file name).
SEARCH_PRESETS = {
    "BrowseComp": (browsecomp_rows, browsecomp_example, "browsecomp.jsonl"),
    "SimpleQA": (simpleqa_rows, simpleqa_example, "simpleqa.jsonl"),
}


def search_servers(settings: Mapping[str, Any], max_model_len: int, exclude: Path) -> dict:
    """Gym server instances for search-agent training: the search tools, a judge and the agent.

    Keys are placeholders here. ``actor_config`` fills them in inside the Gym actor.
    """
    return {
        JUDGE_SERVER: {
            "responses_api_models": {
                "vllm_model": {
                    "entrypoint": "app.py",
                    "base_url": settings["judge_base_url"],
                    "api_key": "unused",
                    "model": settings["judge_model"],
                    "return_token_id_information": False,
                    "uses_reasoning_parser": True,
                    # A reasoning judge needs room for its answer after the reasoning.
                    "extra_body": {"max_tokens": 8192},
                }
            }
        },
        BROWSECOMP_SERVER: {
            "resources_servers": {
                "browsecomp_advanced_harness": {
                    # The search backend: the harness's own app.py (Tavily), or a file that
                    # actor_config copies into the harness, such as a You.com client.
                    "entrypoint": Path(settings["search_entrypoint"]).name
                    if settings.get("search_entrypoint")
                    else "app.py",
                    "domain": "agent",
                    "verified": False,
                    "debug": False,
                    "tavily_api_key": "unused",
                    "exclude_domains_file_path": str(exclude),
                    "use_judge": True,
                    "judge_model_server": {"type": "responses_api_models", "name": JUDGE_SERVER},
                    "judge_responses_create_params": {"input": []},
                }
            }
        },
        BROWSECOMP_AGENT: {
            "responses_api_agents": {
                "browsecomp_agent": {
                    "entrypoint": "app.py",
                    "resources_server": {"type": "resources_servers", "name": BROWSECOMP_SERVER},
                    "model_server": {"type": "responses_api_models", "name": "policy_model"},
                    "max_steps": int(settings.get("max_steps") or 12),
                    "nudge_steps": True,
                    "max_context_tokens": max_model_len,
                    # A context reset drops earlier turns, which breaks the trajectory's token
                    # prefix. Training runs each trajectory in one context instead.
                    "max_reset_count": 0,
                    "resume_from_trajectory": False,
                    "save_model_call_using_vllm_tokenize_endpoint": False,
                }
            }
        },
    }


def _required_key_file(settings: Mapping[str, Any], name: str, what: str) -> str:
    path = settings.get(name)
    if not path:
        raise ValueError(f"Search-agent training requires env.nemo_gym.search.{name} ({what})")
    if not Path(path).is_file():
        raise ValueError(f"The {what} file {path} does not exist; mount its Secret")
    return str(path)


def prepare_config(config: dict[str, Any], *, rows=None) -> dict[str, Any]:
    """Check a CoreWeave Gym training config, and fill in a preset when one is selected.

    The config is returned as an unchanged copy unless the backend is ``coreweave`` and
    ``env.should_use_nemo_gym`` is set. With ``data.train.dataset_name`` ``DAPOMath17K``,
    ``SimpleQA`` or ``BrowseComp``, that dataset is written as Gym data and the Gym configs are set.
    SimpleQA and BrowseComp run Gym's BrowseComp agent, with the judge and search settings in
    ``env.nemo_gym.search``. Otherwise the config must name an explicit ``NemoGymDataset`` and
    ``env.nemo_gym.config_paths``.

    Args:
        config: The NeMo-RL config. It is not modified; a changed copy is returned.
        rows: Dataset rows (DAPO, SimpleQA or encrypted BrowseComp) to use instead of downloading.

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
    gym.setdefault("policy_model", {}).setdefault("responses_api_models", {}).setdefault(
        "vllm_model", {}
    ).update(
        {
            "return_token_id_information": True,
            "chat_template_kwargs": config["policy"]["tokenizer"].get("chat_template_kwargs"),
            # cwtrain's Gym model caps max_tokens to the room the prompt leaves.
            "max_model_len": int(cw["max_model_len"]),
        }
    )
    data = config["data"]
    train = data["train"]
    preset = train.get("dataset_name") if isinstance(train, dict) else None
    search = gym.pop("search", None)
    presets = ("DAPOMath17K", *SEARCH_PRESETS)
    if preset in presets:
        if data.get("validation") or train.get("split_validation_size", 0):
            raise ValueError(
                f"The {preset} Gym preset requires validation disabled; use explicit Gym data"
            )
        if gym.get("config_paths"):
            raise ValueError("Custom Gym environments require an explicit NemoGymDataset")
        max_samples = train.get("max_samples")
        if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
            raise ValueError("data.train.max_samples must be a positive integer")
        directory = Path(cw.get("state_dir", "/work/hotload")) / "nemo-gym"
    if preset == "DAPOMath17K":
        if rows is None:
            from datasets import load_dataset

            rows = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        if max_samples is not None:
            rows = islice(rows, max_samples)
        path = directory / "dapo.jsonl"
        write_dapo(rows, path)
    elif preset in SEARCH_PRESETS:
        download, example, filename = SEARCH_PRESETS[preset]
        settings = dict(search or {})
        for name in ("judge_model", "judge_base_url"):
            if not settings.get(name):
                raise ValueError(f"Search-agent training requires env.nemo_gym.search.{name}")
        gym["coreweave_judge_api_key_file"] = _required_key_file(
            settings, "judge_api_key_file", "judge API key"
        )
        gym["coreweave_search_api_key_file"] = _required_key_file(
            settings, "search_api_key_file", "search API key"
        )
        search_env = settings.get("search_env") or {}
        if search_env:
            # Environment for the search backend, set in the Gym actor before its servers start.
            gym["coreweave_search_env"] = {str(k): str(v) for k, v in search_env.items()}
        entrypoint = settings.get("search_entrypoint")
        if entrypoint:
            if not Path(entrypoint).is_file():
                raise ValueError(f"search_entrypoint {entrypoint} does not exist")
            gym["coreweave_search_entrypoint"] = str(entrypoint)
        if rows is None:
            rows = download()
        if max_samples is not None:
            rows = islice(rows, max_samples)
        path = directory / filename
        write_examples((example(row) for row in rows), path)
        exclude = directory / "exclude_domains.json"
        domains = settings.get("exclude_domains")
        domains = BROWSECOMP_EXCLUDED_DOMAINS if domains is None else domains
        # The harness's exclusion-list format: notices whose domain properties are excluded.
        exclude.write_text(
            json.dumps(
                {"notices": [{"properties": [{"type": "domain", "value": d} for d in domains]}]}
            )
        )
        gym.update(search_servers(settings, int(cw["max_model_len"]), exclude))
    if preset in presets:
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
            "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
        ]
        if preset == "DAPOMath17K":
            # This verifier config uses math-verify, with its LLM judge disabled.
            gym["config_paths"].append(
                "resources_servers/math_with_judge/configs/math_with_judge.yaml"
            )
        config["grpo"]["max_val_samples"] = None
    else:
        datasets = train if isinstance(train, list) else [train]
        if any(item.get("dataset_name") != "NemoGymDataset" for item in datasets):
            raise ValueError(
                "Gym requires DAPOMath17K, SimpleQA, BrowseComp or an explicit NemoGymDataset"
            )
        if not gym.get("config_paths"):
            raise ValueError("Custom Gym data requires env.nemo_gym.config_paths")
    return config


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
    rollout = RolloutConfig.from_env(env)
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
    judge_file = initial.pop("coreweave_judge_api_key_file", None)
    search_file = initial.pop("coreweave_search_api_key_file", None)
    search_entrypoint = initial.pop("coreweave_search_entrypoint", None)
    # Gym's servers inherit the actor's environment.
    os.environ.update(initial.pop("coreweave_search_env", None) or {})
    if judge_file or search_file:
        if search_entrypoint:
            import shutil

            import nemo_gym

            harness_dir = nemo_gym.PARENT_DIR / HARNESS_DIR
            shutil.copyfile(search_entrypoint, harness_dir / Path(search_entrypoint).name)
        judge = initial[JUDGE_SERVER]["responses_api_models"]["vllm_model"]
        judge["api_key"] = _read_key(judge_file, "judge API key")
        harness = initial[BROWSECOMP_SERVER]["resources_servers"]["browsecomp_advanced_harness"]
        harness["tavily_api_key"] = _read_key(search_file, "search API key")
    return initial


def _read_key(path: str | None, what: str) -> str:
    try:
        key = Path(path).read_text().strip() if path else ""
    except OSError:
        key = ""
    if not key:
        raise ValueError(f"Gym actor cannot read a nonempty {what} from {path}")
    return key


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
