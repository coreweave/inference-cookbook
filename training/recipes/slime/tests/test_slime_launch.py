# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Unit tests for the slime launcher (no slime, Ray or GPUs needed)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cwtrain.errors import CWTrainError
from cwtrain.integrations.launch import LaunchError

spec = importlib.util.spec_from_file_location(
    "slime_launch", Path(__file__).resolve().parents[1] / "launch.py"
)
launch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launch
spec.loader.exec_module(launch)

BASE = {
    "HF_MODEL": "Qwen/Qwen3-4B",
    "MODEL_SLUG": "Qwen3-4B",
    "HF_REVISION": "1cfa9a7208912126459214e8b04321603b3df60c",
}


def flag_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_settings_come_from_the_environment() -> None:
    job = launch.SlimeJob.from_env({**BASE, "TP_SIZE": "4", "WORK": "/data", "NUM_ROLLOUT": "30"})
    assert job.tp_size == 4 and job.num_rollout == 30
    assert job.model_dir == Path("/data/models/Qwen3-4B")
    assert job.slime_commit == launch.TESTED_SLIME_COMMIT  # pinned unless overridden
    with pytest.raises(LaunchError, match="HF_MODEL is required"):
        launch.SlimeJob.from_env({"MODEL_SLUG": "x"})


def test_model_download_uses_the_selected_commit_and_rejects_mutable_refs(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(launch, "run", lambda args, **kwargs: calls.append(args))
    job = launch.SlimeJob.from_env({**BASE, "WORK": str(tmp_path)})
    launch.stage(job)
    assert calls[0][:6] == ["hf", "download", BASE["HF_MODEL"], "--revision", BASE["HF_REVISION"], "--local-dir"]

    unpinned = dataclasses.replace(job, hf_revision="main")
    calls.clear()
    with pytest.raises(LaunchError, match="HF_REVISION"):
        launch.stage(unpinned)
    assert calls == []


def test_sequence_parallel_only_with_tensor_parallelism() -> None:
    one = launch.train_args(launch.SlimeJob.from_env({**BASE, "TP_SIZE": "1"}))
    two = launch.train_args(launch.SlimeJob.from_env({**BASE, "TP_SIZE": "2"}))
    assert "--sequence-parallel" not in one and flag_value(one, "--tensor-model-parallel-size") == "1"
    assert "--sequence-parallel" in two


def test_train_args_wire_the_hot_load_hook_and_keep_extras_intact() -> None:
    extra = '--rollout-stop \'["</answer>"]\'\n--save-interval 50'
    args = launch.train_args(launch.SlimeJob.from_env({**BASE, "EXTRA_TRAIN_ARGS": extra}))
    assert flag_value(args, "--rollout-num-gpus") == "0"
    assert flag_value(args, "--custom-update-weight-post-write-path") == (
        "cwtrain.integrations.slime.hook"
    )
    assert "--update-weight-standalone-publish" in args
    assert args[-4:] == ["--rollout-stop", '["</answer>"]', "--save-interval", "50"]
    with pytest.raises(LaunchError, match="EXTRA_TRAIN_ARGS"):
        launch.SlimeJob.from_env({**BASE, "EXTRA_TRAIN_ARGS": "--x 'unterminated"})


def test_a_generate_function_stops_the_launcher_before_anything_is_staged() -> None:
    retool = "--custom-generate-function-path examples.retool.generate_with_retool.generate"
    with pytest.raises(CWTrainError, match="--custom-generate-function-path is not supported"):
        launch.SlimeJob.from_env({**BASE, "EXTRA_TRAIN_ARGS": f"--save-interval 50 {retool}"})


@pytest.mark.parametrize(("given", "on"), [(None, True), ("", True), ("1", True), ("off", False)])
def test_truncated_importance_sampling_is_on_unless_turned_off(given, on) -> None:
    env = {**BASE} if given is None else {**BASE, "USE_TIS": given}
    assert ("--use-tis" in launch.train_args(launch.SlimeJob.from_env(env))) is on
    with pytest.raises(LaunchError, match="USE_TIS must be a boolean"):
        launch.SlimeJob.from_env({**BASE, "USE_TIS": "sometimes"})


WANDB = {"WANDB_PROJECT": "cookbook", "WANDB_API_KEY": "key", "CWTRAIN_HOTLOAD_RUN_ID": "run-1"}


def test_wandb_flags_only_with_a_project_and_never_the_key() -> None:
    assert "--use-wandb" not in launch.train_args(launch.SlimeJob.from_env(BASE))
    extra = "--save-interval 50"
    args = launch.train_args(launch.SlimeJob.from_env({**BASE, **WANDB, "EXTRA_TRAIN_ARGS": extra}))
    i = args.index("--use-wandb")
    assert args[i : i + 6] == [
        "--use-wandb", "--wandb-project", "cookbook",
        "--wandb-group", "run-1", "--disable-wandb-random-suffix",
    ]  # fmt: skip
    assert args[-2:] == ["--save-interval", "50"]  # extras still come last, and win
    assert "key" not in args and "--wandb-key" not in args
    team = launch.train_args(launch.SlimeJob.from_env({**BASE, **WANDB, "WANDB_ENTITY": "team"}))
    assert flag_value(team, "--wandb-team") == "team"


def test_wandb_brings_the_trainer_timeline_hooks_and_yours_win() -> None:
    from cwtrain.integrations.slime import TRAINER_TIMELINE_ARGS

    flag = "--custom-megatron-before-train-step-hook-path"
    assert flag in TRAINER_TIMELINE_ARGS
    assert flag not in launch.train_args(launch.SlimeJob.from_env(BASE))
    args = launch.train_args(launch.SlimeJob.from_env({**BASE, **WANDB}))
    i = args.index(TRAINER_TIMELINE_ARGS[0])
    assert args[i : i + len(TRAINER_TIMELINE_ARGS)] == list(TRAINER_TIMELINE_ARGS)
    # a hook of your own in EXTRA_TRAIN_ARGS comes later, and argparse keeps the last value
    mine = launch.train_args(
        launch.SlimeJob.from_env({**BASE, **WANDB, "EXTRA_TRAIN_ARGS": f"{flag} my.hook"})
    )
    assert mine.count(flag) == 2 and mine[-2:] == [flag, "my.hook"]


def test_runtime_env_tags_the_wandb_run_only_with_a_project() -> None:
    assert "WANDB_TAGS" not in launch.runtime_env("/py", {})["env_vars"]
    vars_ = launch.runtime_env("/py", WANDB)["env_vars"]
    assert vars_["WANDB_TAGS"] == "slime,grpo" and vars_["WANDB_JOB_TYPE"] == "train"
    assert "WANDB_API_KEY" not in vars_  # a secret never travels in the runtime env
    # tags you set are kept: the workers inherit them from the pod
    mine = launch.runtime_env("/py", {**WANDB, "WANDB_TAGS": "mine"})["env_vars"]
    assert "WANDB_TAGS" not in mine


def test_a_wandb_project_without_a_key_or_run_id_stops_the_launcher() -> None:
    with pytest.raises(LaunchError, match="WANDB_API_KEY_SECRET"):
        launch.SlimeJob.from_env({**BASE, **WANDB, "WANDB_API_KEY": " "})
    with pytest.raises(LaunchError, match="CWTRAIN_HOTLOAD_RUN_ID is required"):
        launch.SlimeJob.from_env({**BASE, **WANDB, "CWTRAIN_HOTLOAD_RUN_ID": ""})


@pytest.mark.parametrize("flag", ["--wandb-key k", "--wandb-key=k"])
def test_the_wandb_key_as_an_argument_stops_the_launcher(flag) -> None:
    with pytest.raises(LaunchError, match="--wandb-key is not supported"):
        launch.SlimeJob.from_env({**BASE, "EXTRA_TRAIN_ARGS": f"--save-interval 50 {flag}"})


def test_runtime_env_forwards_settings_to_the_workers() -> None:
    env = {"CWTRAIN_HOTLOAD_RUN_ID": "r", "SLIME_RELAX_NUMPY_CHECK": "1", "HOME": "/root"}
    vars_ = launch.runtime_env("/work/py:/root/slime", env)["env_vars"]
    assert vars_["CWTRAIN_HOTLOAD_RUN_ID"] == "r" and vars_["SLIME_RELAX_NUMPY_CHECK"] == "1"
    assert vars_["PYTHONPATH"] == "/work/py:/root/slime" and "HOME" not in vars_


@pytest.mark.parametrize("given,expected", [(None, "reset_all"), ("", "reset_all"), ("preserve", "preserve")])
def test_runtime_env_resets_the_prompt_cache_unless_told_otherwise(given, expected) -> None:
    env = {} if given is None else {"CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY": given}
    vars_ = launch.runtime_env("/py", env)["env_vars"]
    assert vars_["CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY"] == expected


def test_main_runs_the_steps_in_order(tmp_path: Path, monkeypatch) -> None:
    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "coreweave_rollout.py").write_text("# rollout\n")
    (kit / "coreweave_train.py").write_text("# driver\n")
    (kit / "0001-a.patch").write_text("")
    slime = tmp_path / "slime"
    (slime / "scripts" / "models").mkdir(parents=True)
    calls: list[tuple] = []
    monkeypatch.setattr(launch, "apply_patches", lambda repo, patches, tested: calls.append(("patch", repo, [p.name for p in patches], tested)))  # noqa: E501
    monkeypatch.setattr(launch, "stage_from_s3", lambda uri, dest: calls.append(("stage", uri, dest)))
    monkeypatch.setattr(launch, "bash_array", lambda script, name, cwd: ["--num-layers", "36"])
    monkeypatch.setattr(launch, "run", lambda cmd, **kw: calls.append(("run", [str(c) for c in cmd], kw)))
    monkeypatch.setattr(launch, "git_describe", lambda repo: "abc 2026-09-23")
    execs: list[list[str]] = []
    monkeypatch.setattr(launch.os, "execvp", lambda file, args: execs.append(args))
    monkeypatch.setattr(launch.os, "chdir", lambda path: calls.append(("chdir", Path(path))))
    monkeypatch.setenv("PYTHONPATH", "")
    env = {
        **BASE,
        "WORK": str(tmp_path / "work"),
        "CWTRAIN_KIT_DIR": str(kit),
        "SLIME_DIR": str(slime),
        "BASE_S3_URI": "s3://b/models/Qwen3-4B",
        "CWTRAIN_HOTLOAD_RUN_ID": "r",
    }
    # the process environment holds none of the job's settings: they must come from main(env)
    monkeypatch.setattr(launch.os, "environ", {"PYTHONPATH": ""})
    launch.main(env)

    assert (tmp_path / "work" / "py" / "coreweave_rollout.py").is_file()
    assert (tmp_path / "work" / "py" / "coreweave_train.py").is_file()
    kinds = [c[0] for c in calls]
    assert kinds.index("patch") < kinds.index("stage") < kinds.index("chdir")
    assert calls[kinds.index("patch")][2] == ["0001-a.patch"]
    runs = [c[1] for c in calls if c[0] == "run"]
    assert runs[0][:2] == ["hf", "download"]  # prompt data
    assert runs[1][0] == "torchrun" and "--num-layers" in runs[1]
    assert runs[-1][:2] == ["ray", "start"]
    # Ray runs the job's entrypoint where `ray start` ran: the slime checkout, with train.py's full path
    ray_start = next(c for c in calls if c[0] == "run" and c[1][:2] == ["ray", "start"])
    assert Path(ray_start[2]["cwd"]) == slime
    cmd = execs[0]
    assert cmd[:3] == ["ray", "job", "submit"] and cmd[5:8] == ["--", "python3", str(slime / "train.py")]
    runtime = json.loads(cmd[cmd.index("--runtime-env-json") + 1])
    assert runtime["env_vars"]["CWTRAIN_HOTLOAD_RUN_ID"] == "r"
    assert runtime["env_vars"]["PYTHONPATH"].split(":")[:2] == [str(tmp_path / "work" / "py"), str(slime)]
    assert cmd[8:10] == ["--num-layers", "36"]


def test_with_wandb_the_driver_runs_train_py_through_coreweave_train() -> None:
    script, py = Path("/root/slime/train.py"), Path("/work/py")
    plain = launch.SlimeJob.from_env(BASE)
    assert launch.train_entrypoint(plain, script, py) == (["python3", str(script)], {})
    # W&B keeps only the driver's metric definitions, so the driver binds cwtrain's metrics
    traced = launch.SlimeJob.from_env({**BASE, **WANDB})
    assert launch.train_entrypoint(traced, script, py) == (
        ["python3", "/work/py/coreweave_train.py"],
        {"CWTRAIN_TRAIN_SCRIPT": str(script)},
    )


def test_megatron_patches_are_reapplied_on_every_start(tmp_path: Path, monkeypatch) -> None:
    slime = tmp_path / "slime"
    patch_dir = slime / "docker" / "patch" / "latest"
    patch_dir.mkdir(parents=True)
    (patch_dir / "megatron.patch").write_text("")
    megatron = tmp_path / "work" / "Megatron-LM"
    (megatron / ".git").mkdir(parents=True)  # fetched by an earlier, interrupted start
    fetched: list[str] = []
    applied: list[tuple[Path, list[str]]] = []
    monkeypatch.setattr(launch, "fetch_commit", lambda *a: fetched.append("x"))
    monkeypatch.setattr(launch, "apply_patches", lambda repo, patches, tested: applied.append((repo, [p.name for p in patches])))  # noqa: E501
    monkeypatch.setattr(launch, "git_describe", lambda repo: "abc")
    job = launch.SlimeJob.from_env({**BASE, "WORK": str(tmp_path / "work"), "MEGATRON_COMMIT": "abc"})
    assert launch.prepare_megatron(job, slime) == megatron
    assert fetched == []  # the checkout is reused
    assert applied == [(megatron, ["megatron.patch"])]  # but its patches are completed again


def test_a_kit_without_patches_is_refused(tmp_path: Path) -> None:
    kit = tmp_path / "kit"
    kit.mkdir()
    job = launch.SlimeJob.from_env({**BASE, "CWTRAIN_KIT_DIR": str(kit), "SLIME_DIR": str(tmp_path)})
    with pytest.raises(LaunchError, match="rebuild the kit"):
        launch.prepare_slime(job)



def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def commit(repo: Path, message: str) -> str:
    git(repo, "commit", "--quiet", "--allow-empty", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def slime_job(tmp_path: Path, **overrides) -> launch.SlimeJob:
    kit = tmp_path / "kit"
    kit.mkdir(exist_ok=True)
    (kit / "0001-a.patch").write_text("")
    job = launch.SlimeJob.from_env({**BASE, "WORK": str(tmp_path / "work"), "CWTRAIN_KIT_DIR": str(kit)})
    return dataclasses.replace(job, **overrides)


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "--quiet")
    return repo


@pytest.fixture
def patched(monkeypatch) -> list[Path]:
    applied: list[Path] = []
    monkeypatch.setattr(launch, "apply_patches", lambda repo, patches, tested: applied.append(repo))
    return applied


def test_the_images_clone_of_slime_main_is_moved_to_the_pinned_commit(tmp_path, upstream, patched) -> None:
    pinned = commit(upstream, "tested")
    commit(upstream, "main moved on")
    image = tmp_path / "image-slime"
    git(tmp_path, "clone", "--quiet", str(upstream), str(image))  # like the official images: main
    job = slime_job(tmp_path, slime_repo=str(upstream), slime_commit=pinned, image_slime_dir=str(image))
    assert launch.prepare_slime(job) == image
    assert git(image, "rev-parse", "HEAD") == pinned
    assert patched == [image]


def test_an_image_built_before_the_pinned_commit_fetches_it(tmp_path, upstream, patched) -> None:
    commit(upstream, "older")
    image = tmp_path / "image-slime"
    git(tmp_path, "clone", "--quiet", str(upstream), str(image))
    pinned = commit(upstream, "tested, newer than the image")
    job = slime_job(tmp_path, slime_repo=str(upstream), slime_commit=pinned, image_slime_dir=str(image))
    assert launch.prepare_slime(job) == image
    assert git(image, "rev-parse", "HEAD") == pinned


def test_an_image_without_slime_gets_a_clone_at_the_pinned_commit(tmp_path, upstream, patched) -> None:
    pinned = commit(upstream, "tested")
    commit(upstream, "main moved on")
    job = slime_job(tmp_path, slime_repo=str(upstream), slime_commit=pinned,
                    image_slime_dir=str(tmp_path / "no-slime-here"))
    clone = launch.prepare_slime(job)
    assert clone == tmp_path / "work" / "slime"
    assert git(clone, "rev-parse", "HEAD") == pinned


def test_slime_dir_is_used_as_it_is(tmp_path, upstream, patched) -> None:
    commit(upstream, "tested")
    head = commit(upstream, "someone's own checkout")
    job = slime_job(tmp_path, slime_dir=str(upstream), slime_commit="0" * 40)
    assert launch.prepare_slime(job) == upstream
    assert git(upstream, "rev-parse", "HEAD") == head
