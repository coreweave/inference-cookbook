# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The example's settings fit the NeMo-RL recipe: rollout concurrency from run.env to the Job to the
trainer, the README's sizing checked against the Nemotron deployment, and the base model fetched
from where the recipe's staging Job puts it.

The Nemotron deployment is smaller than a default step on purpose (README, "Concurrency"), so,
unlike the miles and slime examples, nothing tests that it runs everything the trainer keeps in
flight.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]
ROOT = EXAMPLE.parents[2]
RECIPE = ROOT / "training" / "recipes" / "nemo-rl"
# the deployment that the README creates
DEPLOYMENT = EXAMPLE / "deployment.example.json"


def run_env_example() -> dict[str, str]:
    values = {}
    for line in (EXAMPLE / "k8s" / "run.env.example").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("'\"")
    return values


def grpo_setting(key: str) -> int:
    """An integer that grpo_coreweave.yaml sets, such as num_generations_per_prompt."""
    config = (RECIPE / "grpo_coreweave.yaml").read_text()
    match = re.search(rf"^\s*{key}:\s*(\d+)", config, re.MULTILINE)
    assert match, f"grpo_coreweave.yaml no longer sets {key}"
    return int(match.group(1))


def stated(section: str, pattern: str) -> set[int]:
    """The numbers in section where pattern has (N): in digits, or as a word up to ten."""
    words = "zero one two three four five six seven eight nine ten".split()
    found = re.findall(pattern.replace("(N)", rf"\b(\d+|{'|'.join(words)})\b"), section)
    return {int(n) if n.isdigit() else words.index(n) for n in found}


def test_the_readme_concurrency_numbers_are_the_defaults() -> None:
    readme = (EXAMPLE / "README.md").read_text()
    heading = re.search(r"^## .*concurrency.*$", readme, re.IGNORECASE | re.MULTILINE)
    assert heading, "the README has no section about concurrency"
    section = " ".join(readme[heading.end() :].split("\n## ")[0].split())
    deployment = json.loads(DEPLOYMENT.read_text())
    autoscaling = deployment["autoscaling"]
    replicas, per_replica = autoscaling["min"], autoscaling["concurrency"]
    assert autoscaling["max"] == replicas, "the README counts a fixed number of replicas"
    # autoscaling.concurrency becomes vLLM's --max-num-seqs; an engineConfig max-num-seqs must agree
    max_num_seqs = int(deployment["runtime"]["engineConfig"].get("max-num-seqs", per_replica))
    assert max_num_seqs == per_replica, "engineConfig's max-num-seqs is not autoscaling.concurrency"
    default = int(run_env_example()["ROLLOUT_CONCURRENCY"])
    prompts = grpo_setting("num_prompts_per_step")
    generations = grpo_setting("num_generations_per_prompt")
    # each number goes by the words next to it, so that rewording passes and a wrong number fails
    step = {
        "(N) prompts": prompts,
        "(N) generations": generations,
        "(N) replicas? at": replicas,
        "replicas? at (N)": per_replica,
        "(N) sequences": prompts * generations,
        # a request per prompt, ROLLOUT_CONCURRENCY at a time; the replicas' slots bound each group
        "at most (N)": min(min(default, prompts) * generations, replicas * per_replica),
    }
    assert {pattern: stated(section, pattern) for pattern in step} == {
        pattern: {value} for pattern, value in step.items()
    }, "the README's default step (left) differs from the config and the deployment (right)"
    # the section may leave these out, but not state another number
    elsewhere = {"sets (N)": per_replica, "default (N)": default, "config's (N)": default}
    for pattern, value in elsewhere.items():
        assert stated(section, pattern) <= {value}, f"the README's {pattern!r} should be {value}"


@pytest.mark.skipif(shutil.which("envsubst") is None, reason="render.sh needs envsubst")
def test_the_job_passes_rollout_concurrency_from_run_env(tmp_path: Path) -> None:
    # render.sh refuses unset values and placeholders, so start from the example, filled in
    env_file = tmp_path / "run.env"
    example = (EXAMPLE / "k8s" / "run.env.example").read_text()
    env_file.write_text(re.sub(r"<[a-z][a-z0-9 -]*>", "filled", example) + "ROLLOUT_CONCURRENCY=40\n")
    rendered = subprocess.run(
        [
            "bash",
            str(ROOT / "k8s" / "render.sh"),  # the shared k8s/, at the repository root
            str(env_file),
            str(RECIPE / "k8s" / "trainer-job.yaml"),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert '{name: CWTRAIN_ROLLOUT_CONCURRENCY, value: "40"}' in rendered


def test_the_config_defaults_to_the_run_env_value() -> None:
    config = (RECIPE / "grpo_coreweave.yaml").read_text()
    match = re.search(r"concurrency: \$\{oc\.env:CWTRAIN_ROLLOUT_CONCURRENCY,(\d+)\}", config)
    assert match, "coreweave_cfg.concurrency no longer reads CWTRAIN_ROLLOUT_CONCURRENCY"
    assert match.group(1) == run_env_example()["ROLLOUT_CONCURRENCY"]


def test_run_env_fetches_the_base_model_from_where_the_stage_job_puts_it() -> None:
    # NeMo-RL has no Hugging Face fallback, so an empty BASE_S3_URI cannot train
    wanted = ("DEPLOYMENT_RUN", "BUCKET", "MODEL_PREFIX", "MODEL_SLUG", "BASE_S3_URI")
    lines = [line for line in (EXAMPLE / "k8s" / "run.env.example").read_text().splitlines()
             if line.split("=", 1)[0] in wanted]  # fmt: skip
    show = 'printf "%s\\n" "$BUCKET" "$MODEL_PREFIX" "$MODEL_SLUG" "$BASE_S3_URI"'
    bucket, prefix, slug, uri = subprocess.run(
        ["bash", "-c", "; ".join(["set -eu", *lines, show])],
        check=True, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
    ).stdout.splitlines()
    assert uri == f"s3://{bucket}/{prefix}/{slug}"
    # ...which is where stage-nemotron.sh uploads the checkpoint: s3://$BUCKET/$MODEL_PREFIX/$MODEL_SLUG/
    stage = (RECIPE / "k8s" / "stage-nemotron.sh").read_text()
    upload = re.search(r"upload_snapshot\((.*?)\)\n", stage, re.DOTALL).group(1)
    for arg in ('bucket=os.environ["BUCKET"]', 'path_prefix=os.environ["MODEL_PREFIX"]',
                'identity=os.environ["MODEL_SLUG"]'):  # fmt: skip
        assert arg in upload


def test_the_deployment_runs_a_whole_step_at_once_on_eight_h100s() -> None:
    deployment = json.loads(DEPLOYMENT.read_text())
    autoscaling = deployment["autoscaling"]
    assert autoscaling["min"] == autoscaling["max"] == 4  # fixed replicas, so hot-loads reach all
    assert deployment["resources"]["gpuCount"] * autoscaling["min"] == 8  # eight H100s
    sequences = grpo_setting("num_prompts_per_step") * grpo_setting("num_generations_per_prompt")
    assert autoscaling["min"] * autoscaling["concurrency"] >= sequences


def kit_files() -> dict[str, Path]:
    """The files that run.env's KIT_EXTRA_FILES adds to the kit, by their name there."""
    paths = run_env_example()["KIT_EXTRA_FILES"].split()
    return {Path(p).name: (EXAMPLE / "k8s" / p).resolve() for p in paths}


def search_config() -> dict:
    """grpo_search_agent.yaml over grpo_coreweave.yaml, as NeMo-RL merges them (without its base)."""
    from omegaconf import OmegaConf

    recipe = OmegaConf.load(RECIPE / "grpo_coreweave.yaml")
    recipe.pop("defaults")
    example = OmegaConf.load(EXAMPLE / "grpo_search_agent.yaml")
    assert example.pop("defaults") == "grpo_coreweave.yaml"
    merged = OmegaConf.merge(recipe, example)
    return OmegaConf.to_container(merged, resolve=False)


def test_the_kit_carries_what_run_env_and_the_config_name() -> None:
    files = kit_files()
    assert all(path.is_file() for path in files.values()), files
    settings = run_env_example()
    assert settings["GRPO_CONFIG"] in files
    script = settings["EXTRA_PREPARE_DATA"].split()[0]
    assert script.startswith("/opt/cwtrain/") and Path(script).name in files
    # every kit path that the config reads is in the kit
    text = (EXAMPLE / "grpo_search_agent.yaml").read_text()
    for name in re.findall(r"/opt/cwtrain/([\w.-]+)", text):
        assert name in files, f"grpo_search_agent.yaml reads /opt/cwtrain/{name}, not in the kit"


def test_the_config_and_the_data_script_agree() -> None:
    prepare = load_prepare()
    config = search_config()
    assert config["data"]["train"]["data_path"] == str(prepare.OUTPUT)
    assert prepare.AGENT in config["env"]["nemo_gym"]
    assert config["policy"]["max_total_sequence_length"] == 16384
    # within the deployment's context
    deployment = json.loads(DEPLOYMENT.read_text())
    assert 16384 <= int(deployment["runtime"]["engineConfig"]["max-model-len"])


def test_cwtrain_accepts_the_config_and_the_actor_gets_the_keys(tmp_path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    gym_module = pytest.importorskip("cwtrain.integrations.nemo_rl.gym")
    config = search_config()
    # the mounted Secret and kit, here under tmp_path
    secrets = {"judge_api_key": "judge-key", "search_api_key": "you-key"}
    for name, value in secrets.items():
        (tmp_path / name).write_text(value + "\n")
    gym = config["env"]["nemo_gym"]
    gym["coreweave_secrets"] = {
        field: str(tmp_path / Path(path).name) for field, path in gym["coreweave_secrets"].items()
    }
    gym["coreweave_files"] = {
        target: str(kit_files()[Path(source).name]) for target, source in gym["coreweave_files"].items()
    }
    coreweave = config["policy"]["generation"]["coreweave_cfg"]
    coreweave.update(api_key_file=str(tmp_path / "judge_api_key"), max_model_len=16384)
    config["policy"]["max_total_sequence_length"] = 16384
    gym["browsecomp_agent"]["responses_api_agents"]["browsecomp_agent"]["max_context_tokens"] = 16384
    monkeypatch.setenv("CWTRAIN_HOTLOAD_WAIT", "1")
    prepared = gym_module.prepare_config(config)["env"]["nemo_gym"]
    assert "judge-key" not in json.dumps(prepared) and "you-key" not in json.dumps(prepared)
    root = tmp_path / "gym"
    (root / "resources_servers" / "browsecomp_advanced_harness").mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "nemo_gym", SimpleNamespace(PARENT_DIR=root))
    monkeypatch.delenv("YOU_SEARCH_MAX_CHARS", raising=False)
    prepared.pop("coreweave_tokenizer")
    initial = dict(prepared, policy_base_url=["https://example.test/v1"], policy_model_name="m")
    result = gym_module.actor_config(initial, {})
    judge = result["judge_model"]["responses_api_models"]["vllm_model"]
    harness = result["browsecomp_resources_server"]["resources_servers"]["browsecomp_advanced_harness"]
    assert (judge["api_key"], harness["tavily_api_key"]) == ("judge-key", "you-key")
    assert (root / "resources_servers" / "browsecomp_advanced_harness" / harness["entrypoint"]).is_file()
    assert __import__("os").environ["YOU_SEARCH_MAX_CHARS"] == "2000"


def load_prepare():
    import importlib.util

    spec = importlib.util.spec_from_file_location("prepare_search_data", EXAMPLE / "prepare_search_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encrypt(plaintext: str, password: str) -> str:
    """BrowseComp's encryption, the inverse of prepare_search_data.decrypt."""
    import base64
    import hashlib

    data = plaintext.encode()
    digest = hashlib.sha256(password.encode()).digest()
    key = digest * (len(data) // len(digest)) + digest[: len(data) % len(digest)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key, strict=True))).decode()


@pytest.mark.parametrize("dataset", ["simpleqa", "browsecomp"])
def test_the_data_script_writes_the_benchmark_prompt(tmp_path, dataset, capsys) -> None:
    prepare = load_prepare()
    question, answer = "Who wrote the first draft?", "Ada Lovelace"
    if dataset == "browsecomp":
        rows = [{"problem": encrypt(question, "c"), "answer": encrypt(answer, "c"), "canary": "c"}] * 3
    else:
        rows = [{"problem": question, "answer": answer}] * 3
    output = tmp_path / "search.jsonl"
    prepare.main(["--dataset", dataset, "--max-samples", "2", "--output", str(output)], rows=rows)
    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(lines) == 2 and "wrote 2" in capsys.readouterr().out
    item = lines[0]
    assert (item["question"], item["ground_truth"]) == (question, answer)
    assert item["agent_ref"] == {"name": "browsecomp_agent"}
    system, user = item["responses_create_params"]["input"]
    assert system["role"] == "system" and "Today's date" in system["content"]
    assert user["content"].startswith(question) and "Exact Answer:" in user["content"]
    assert [tool["name"] for tool in item["responses_create_params"]["tools"]] == ["search", "browse"]


def test_the_data_script_rejects_a_short_download(monkeypatch) -> None:
    prepare = load_prepare()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"problem,answer\nq,a\n"

    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda url, timeout: Response())
    with pytest.raises(ValueError, match="expected 4326 simpleqa rows, got 1"):
        prepare.download("simpleqa")
