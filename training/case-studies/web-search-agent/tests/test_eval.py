# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import argparse
import importlib.util
import json
import re
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


ROWS = [
    {
        "_ng_task_index": 7,
        "reward": 1.0,
        "response": {
            "output": [
                {"type": "function_call", "name": "search"},
                {"type": "function_call_output"},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Exact Answer: 42"}],
                },
            ]
        },
    },
    {
        "_ng_task_index": 9,
        "reward": 0.0,
        "response": {
            "output": [{"type": "function_call", "name": "search"}, {"type": "function_call_output"}]
            * 3
        },
    },
]


def report_args(**overrides):
    values = dict(
        benchmark="browsecomp", limit=5, max_steps=3, max_context_tokens=32768,
        judge_model=launch.DEFAULT_JUDGE_MODEL, model=None,
    )  # fmt: skip
    return argparse.Namespace(**(values | overrides))


def test_scores_count_tool_calls_and_the_turn_cap():
    tasks, summary = launch.scores(ROWS, max_steps=3)
    assert [t["task"] for t in tasks] == [7, 9]
    assert tasks[0]["answer"] == "Exact Answer: 42" and tasks[1]["answer"] == ""
    assert [t["turns"] for t in tasks] == [1, 3]
    assert [t["reached_turn_cap"] for t in tasks] == [False, True]
    assert summary == {
        "tasks": 2, "mean_reward": 0.5, "correct": 1, "tasks_at_turn_cap": 1,
        "mean_tool_calls": 2.0,
    }  # fmt: skip


def test_the_turn_cap_counts_model_turns_not_tool_calls():
    # two calls in one model turn, as the agent runs them, then a third turn that answers
    call, result = {"type": "function_call", "name": "search"}, {"type": "function_call_output"}
    answer = {"type": "message", "content": [{"type": "output_text", "text": "Exact Answer: 1"}]}
    output = [{"type": "reasoning"}, call, call, result, result, call, call, result, result, answer]
    (task,), _ = launch.scores([{"reward": 1.0, "response": {"output": output}}], max_steps=3)
    assert task["tool_calls"] == 4 and task["turns"] == 2 and not task["reached_turn_cap"]


class _NoWandb:
    def __getattr__(self, name):
        raise AssertionError(f"W&B used without WANDB_PROJECT: wandb.{name}")


def test_the_report_prints_the_scores_and_leaves_wandb_alone(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb", _NoWandb())
    summary = launch.report(ROWS, report_args(), {"WANDB_API_KEY": "k"})
    assert summary["mean_reward"] == 0.5
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "reward=1.0 tool_calls=1",
        "reward=0.0 tool_calls=3",
        "tasks=2 mean_reward=0.500",
    ]


class _FakeRun:
    def __init__(self, **kwargs):
        self.init = kwargs
        self.logged = []
        self.summary = {}
        self.finished = False

    def log(self, row):
        self.logged.append(row)

    def finish(self):
        self.finished = True


class _FakeTable:
    def __init__(self, columns, data):
        self.columns, self.data = columns, data


def test_the_report_logs_an_eval_run_with_a_task_table(monkeypatch, capsys):
    runs = []

    def init(**kwargs):
        runs.append(_FakeRun(**kwargs))
        return runs[-1]

    fake = argparse.Namespace(init=init, Table=_FakeTable)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_ENTITY", "")  # an unfilled eval.env value must not reach W&B
    environ = {
        "WANDB_PROJECT": "cookbook", "WANDB_API_KEY": "sk-test-123", "WANDB_RUN_GROUP": "run-1",
        "EVAL_NAME": "browsecomp-1", "CWTRAIN_ROLLOUT_MODEL": "nemotron-lightning-hotload",
        "EVAL_IDENTITY": "run-1-v000003", "WANDB_ENTITY": " ",
    }  # fmt: skip
    launch.report(ROWS, report_args(), environ)
    (run,) = runs
    assert "WANDB_ENTITY" not in __import__("os").environ
    assert run.init["job_type"] == "eval"
    assert {k: run.init[k] for k in ("project", "group", "name")} == {
        "project": "cookbook", "group": "run-1", "name": "browsecomp-1",
    }  # fmt: skip
    assert "entity" not in run.init  # the key's default entity
    assert run.init["config"]["identity"] == "run-1-v000003"
    assert run.init["config"]["model"] == "nemotron-lightning-hotload"
    assert "sk-test-123" not in json.dumps(run.init)
    (row,) = run.logged
    table = row["eval/task_results"]
    assert table.columns[:3] == ["task", "reward", "tool_calls"]
    assert [r[0] for r in table.data] == [7, 9]
    assert row["eval/mean_reward"] == run.summary["eval/mean_reward"] == 0.5
    assert run.summary["eval/tasks_at_turn_cap"] == 1
    assert run.finished


def test_a_wandb_project_without_a_key_stops_before_the_eval():
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        launch.wandb_settings({"WANDB_PROJECT": "cookbook"})
    assert launch.wandb_settings({}) is None
    offline = {"WANDB_PROJECT": "p", "WANDB_MODE": "offline"}
    assert launch.wandb_settings(offline) == {"project": "p"}


def test_the_eval_job_takes_the_wandb_key_from_an_optional_secret():
    job = (K8S / "eval-job.yaml").read_text()
    assert re.search(
        r'secretKeyRef: \{name: "\$\{WANDB_API_KEY_SECRET\}", key: api_key, optional: true\}', job
    )
    example = (K8S / "eval.env.example").read_text()
    assert re.search(r"^WANDB_PROJECT=$", example, re.M)  # off by default
    assert "WANDB_GROUP=${RUN_ID:-}" in example
    script = (K8S / "run-eval.sh").read_text()
    assert "launch.py report" in script and "WANDB_API_KEY" in script
