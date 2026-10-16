# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"gym_recipe_{name}", Path(__file__).resolve().parents[1] / "eval" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launch = load("launch")


def test_key_file_and_normalized_endpoint(tmp_path):
    key = tmp_path / "key"
    key.write_text("test-key\n")
    env = launch.start_environment(
        {
            "CWTRAIN_ROLLOUT_BASE_URL": " https://example.invalid/v1/ ",
            "CWTRAIN_ROLLOUT_MODEL": "nemotron-lightning-hotload",
            "CWTRAIN_ROLLOUT_API_KEY_FILE": str(key),
        }
    )
    assert env["CWTRAIN_ROLLOUT_API_KEY"] == "test-key"
    assert env["CWTRAIN_ROLLOUT_BASE_URL"] == "https://example.invalid/v1"
    cmd = launch.command(argparse.Namespace(action="start"), env)
    assert "test-key" not in " ".join(cmd)
    key.unlink()
    env.pop("CWTRAIN_ROLLOUT_API_KEY")
    with pytest.raises(ValueError, match="nonempty key"):
        launch.start_environment(env)


def browsecomp_args(**overrides):
    values = dict(
        action="start",
        benchmark="browsecomp",
        model=None,
        judge_model=launch.DEFAULT_JUDGE_MODEL,
        judge_base_url=launch.DEFAULT_JUDGE_BASE_URL + "/",
        judge_api_key_file=None,
        search_api_key_file=None,
        max_context_tokens=32768,
        context_reset_tokens=None,
        max_steps=50,
    )
    return argparse.Namespace(**(values | overrides))


ENDPOINT = {"CWTRAIN_ROLLOUT_BASE_URL": "https://example.invalid/v1"}


def test_browsecomp_is_the_default_with_glm_judge_and_nemotron_target():
    env = launch.start_environment(
        ENDPOINT | {"WANDB_API_KEY": "judge-key", "YOU_API_KEY": "you-key"},
        browsecomp_args(),
    )
    assert launch.DEFAULT_BENCHMARK == "browsecomp"
    # Nemotron 3.5 Lightning as the NeMo-RL hot-load deployment serves it
    assert env["CWTRAIN_ROLLOUT_MODEL"] == "nemotron-lightning-hotload"
    assert env["CWTRAIN_MAX_CONTEXT_TOKENS"] == "32768"
    assert env["CWTRAIN_CONTEXT_RESET_TOKENS"] == "16384"
    assert env["CWTRAIN_MAX_STEPS"] == "50"  # bounds each task, not Gym's 400
    assert env["CWTRAIN_JUDGE_MODEL"] == "zai-org/GLM-5.2"
    assert env["CWTRAIN_JUDGE_BASE_URL"] == "https://api.inference.wandb.ai/v1"
    assert env["CWTRAIN_JUDGE_API_KEY"] == "judge-key"
    assert env["CWTRAIN_SEARCH_API_KEY"] == "you-key"
    assert json.loads(Path(env["CWTRAIN_EXCLUDE_DOMAINS_FILE"]).read_text())["notices"]
    cmd = launch.command(browsecomp_args(), env)
    configs = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--config"]
    assert configs[0] == "benchmarks/browsecomp/config.yaml"
    assert configs[-1].endswith("browsecomp.yaml")
    assert "judge-key" not in " ".join(cmd) and "you-key" not in " ".join(cmd)


def test_target_and_judge_models_are_inputs(tmp_path):
    key = tmp_path / "judge"
    key.write_text("file-key\n")
    env = launch.start_environment(
        ENDPOINT | {"CWTRAIN_ROLLOUT_MODEL": "ignored", "YOU_API_KEY": "you-key"},
        browsecomp_args(
            model="my-model",
            judge_model="other/judge",
            judge_base_url="https://judge.invalid/v1",
            judge_api_key_file=key,
        ),
    )
    assert env["CWTRAIN_ROLLOUT_MODEL"] == "my-model"
    assert env["CWTRAIN_JUDGE_MODEL"] == "other/judge"
    assert env["CWTRAIN_JUDGE_BASE_URL"] == "https://judge.invalid/v1"
    assert env["CWTRAIN_JUDGE_API_KEY"] == "file-key"


@pytest.mark.parametrize("missing", ["WANDB_API_KEY", "YOU_API_KEY"])
def test_browsecomp_requires_judge_and_search_keys(missing):
    environ = ENDPOINT | {"WANDB_API_KEY": "j", "YOU_API_KEY": "y"}
    environ.pop(missing)
    with pytest.raises(ValueError, match=missing):
        launch.start_environment(environ, browsecomp_args())


@pytest.mark.parametrize(("max_context", "reset"), [(32768, 0), (32768, 28673), (8192, 8192)])
def test_context_reset_must_leave_room_for_a_turn(max_context, reset):
    with pytest.raises(ValueError, match="context-reset-tokens"):
        launch.context_reset_tokens(max_context, reset)


def test_context_reset_follows_the_context_limit():
    assert launch.context_reset_tokens(65536, None) == 32768
    assert launch.context_reset_tokens(32768, 20000) == 20000


def test_prepare_downloads_the_benchmark_data():
    cmd = launch.command(argparse.Namespace(action="prepare", benchmark="browsecomp"), {})
    assert cmd[1:] == ["eval", "prepare", "--config", "benchmarks/browsecomp/config.yaml"]


def test_recipe_configs_override_the_gym_benchmark():
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf

    config = OmegaConf.load(Path(launch.RECIPE) / "browsecomp.yaml")
    harness = config.browsecomp_benchmark_resources_server.resources_servers
    assert harness.browsecomp_advanced_harness.entrypoint == "you_search.py"
    policy = OmegaConf.load(Path(launch.RECIPE) / "coreweave.yaml").policy_model
    assert policy.responses_api_models.vllm_model.entrypoint == "cwtrain_tool_app.py"
    agent = config.browsecomp_benchmark_agent.responses_api_agents.browsecomp_agent
    assert agent.save_model_call_using_vllm_tokenize_endpoint is False
    raw = OmegaConf.to_container(agent, resolve=False)
    assert raw["context_reset_tokens"] == "${oc.env:CWTRAIN_CONTEXT_RESET_TOKENS}"
    assert raw["max_steps"] == "${oc.env:CWTRAIN_MAX_STEPS}"
    assert "judge_model" in config


def test_collection_is_bounded_and_uses_shared_concurrency(tmp_path):
    cmd = launch.command(
        argparse.Namespace(action="collect", output=tmp_path / "results.jsonl"),
        {
            "CWTRAIN_ROLLOUT_BASE_URL": "https://example.invalid/v1",
            "CWTRAIN_ROLLOUT_MODEL": "nemotron-lightning-hotload",
            "CWTRAIN_ROLLOUT_CONCURRENCY": "2",
        },
    )
    assert cmd[cmd.index("--limit") + 1] == "5"
    assert cmd[cmd.index("--concurrency") + 1] == "2"
    assert cmd[cmd.index("--agent") + 1] == "browsecomp_benchmark_agent"
    assert "--no-serve" in cmd


K8S = Path(launch.RECIPE).parent / "k8s"


def test_the_kit_carries_every_file_the_launcher_reads():
    kit = (K8S / "make-eval-kit.sh").read_text()
    for name in ("launch.py", "you_search.py", "coreweave.yaml", "browsecomp.yaml",
                 "exclude_domains.json", "run-eval.sh"):  # fmt: skip
        assert f"--from-file={name}=" in kit, name
    assert (Path(launch.RECIPE) / "browsecomp.yaml").is_file()
    assert "mcqa" not in kit and list(launch.BENCHMARKS) == ["browsecomp"]


@pytest.mark.parametrize("script", ["make-eval-kit.sh", "run-eval.sh"])
def test_the_eval_scripts_parse(script):
    import subprocess

    subprocess.run(["bash", "-n", str(K8S / script)], check=True)


def test_the_eval_job_runs_on_cpu_and_reuses_the_trainer_secrets():
    job = (K8S / "eval-job.yaml").read_text()
    assert "nvidia.com/gpu.present, operator: DoesNotExist" in job
    assert "nvidia.com/gpu:" not in job  # requests no GPU
    for secret in ("INFERENCE_KEY_SECRET", "GYM_SECRET"):
        assert "${" + secret + "}" in job
    example = (K8S / "eval.env.example").read_text()
    assert "BENCHMARK=browsecomp" in example and "JUDGE_MODEL=zai-org/GLM-5.2" in example
    assert "ROLLOUT_MODEL=nemotron-lightning-hotload" in example


@pytest.mark.skipif(
    __import__("shutil").which("envsubst") is None, reason="render.sh needs envsubst"
)
def test_the_eval_job_renders_from_a_filled_in_env(tmp_path):
    import re
    import subprocess

    env = tmp_path / "eval.env"
    text = (K8S / "eval.env.example").read_text()
    env.write_text(re.sub(r"<[A-Za-z][A-Za-z0-9 _-]*>", "filled", text))
    render = Path(launch.RECIPE).parents[3] / "k8s" / "render.sh"  # the shared k8s/, at the root
    out = subprocess.run(
        ["bash", str(render), str(env), str(K8S / "eval-job.yaml")],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert "${" not in out and "image: python:3.13-slim" in out
