# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Unit tests for the NeMo-RL launcher (no NeMo-RL, Ray or GPUs needed)."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest
from cwtrain.integrations.launch import LaunchError

spec = importlib.util.spec_from_file_location(
    "nemo_launch", Path(__file__).resolve().parents[1] / "launch.py"
)
launch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launch
spec.loader.exec_module(launch)


def make_kit(tmp_path: Path) -> Path:
    kit = tmp_path / "kit"
    kit.mkdir()
    with zipfile.ZipFile(kit / "cwtrain-0.1.0-py3-none-any.whl", "w") as whl:
        whl.writestr("cwtrain/__init__.py", "")
    (kit / "0001-backend.patch").write_text("")
    (kit / "grpo_coreweave.yaml").write_text("defaults: x\n")
    return kit


def test_grpo_command_applies_settings_and_overrides(tmp_path: Path) -> None:
    job = launch.NemoJob.from_env(
        {
            "MODEL_SLUG": "Qwen3-4B",
            "WORK": "/w",
            "CWTRAIN_KIT_DIR": "/kit",
            "NUM_STEPS": "30",
            "NEMO_RL_PYTHON": "/venv/bin/python",
            "EXTRA_OVERRIDES": 'policy.generation.stop_strings=["</answer>"]\ngrpo.val_period=0',
        }
    )
    assert launch.grpo_command(job) == [
        "/venv/bin/python",
        "examples/run_grpo.py",
        "--config",
        "/kit/grpo_coreweave.yaml",
        "policy.model_name=/w/models/Qwen3-4B",
        "grpo.max_num_steps=30",
        "cluster.gpus_per_node=8",
        'policy.generation.stop_strings=["</answer>"]',  # quotes reach Hydra
        "grpo.val_period=0",
    ]
    with pytest.raises(LaunchError, match="MODEL_SLUG is required"):
        launch.NemoJob.from_env({})


def test_install_into_workers_covers_every_prebuilt_venv(tmp_path: Path, monkeypatch) -> None:
    kit = make_kit(tmp_path)
    venvs = tmp_path / "ray_venvs"
    for name in ("a.Worker", "b.Worker", "c.Broken"):
        (venvs / name / "bin").mkdir(parents=True)
        (venvs / name / "bin" / "python").write_text("")
        if name != "c.Broken":  # a dangling or non-executable interpreter is skipped
            (venvs / name / "bin" / "python").chmod(0o755)
    runs: list[list[str]] = []
    monkeypatch.setattr(launch, "run", lambda cmd, **kw: runs.append([str(c) for c in cmd]))
    job = launch.NemoJob.from_env(
        {"MODEL_SLUG": "m", "WORK": str(tmp_path / "w"), "CWTRAIN_KIT_DIR": str(kit),
         "NEMO_RL_VENV_DIR": str(venvs)}
    )  # fmt: skip
    installed = launch.install_into_workers(job, launch.wheel(job))
    assert len(installed) == 2 and all("--no-deps" in cmd for cmd in runs)
    assert (tmp_path / "w" / "py" / "cwtrain" / "__init__.py").is_file()


@pytest.mark.parametrize("given,expected", [(None, "reset_all"), ("preserve", "preserve")])
def test_main_patches_installs_stages_then_hands_over(
    tmp_path: Path, monkeypatch, given, expected
) -> None:
    if given is None:
        monkeypatch.delenv("CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY", raising=False)
    else:
        monkeypatch.setenv("CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY", given)
    kit = make_kit(tmp_path)
    calls: list[tuple] = []
    monkeypatch.setattr(launch, "git_describe", lambda repo: "v0.7.0")
    monkeypatch.setattr(launch, "apply_patches", lambda repo, patches, tested: calls.append(("patch", [p.name for p in patches], tested)))
    monkeypatch.setattr(launch, "install_into_workers", lambda job, whl: calls.append(("install", whl.name)))

    def stage(uri, dest):
        calls.append(("stage", uri))
        dest.mkdir(parents=True)
        (dest / "config.json").write_text("{}")

    monkeypatch.setattr(launch, "stage_from_s3", stage)
    execs: list[list[str]] = []
    monkeypatch.setattr(launch.os, "execvp", lambda path, args: execs.append(args))
    monkeypatch.setattr(launch.os, "chdir", lambda path: calls.append(("chdir", str(path))))
    env = {"MODEL_SLUG": "Qwen3-4B", "WORK": str(tmp_path / "w"), "CWTRAIN_KIT_DIR": str(kit),
           "BASE_S3_URI": "s3://b/models/Qwen3-4B", "NEMO_RL_PYTHON": "/venv/bin/python"}  # fmt: skip
    launch.main(env)
    assert [c[0] for c in calls] == ["patch", "install", "stage", "chdir"]
    assert calls[0][1] == ["0001-backend.patch"] and "v0.7.0" in calls[0][2]
    assert execs[0][:2] == ["/venv/bin/python", "examples/run_grpo.py"]
    # GRPO here is synchronous, so the trainer asks for a clean prompt cache unless told otherwise
    assert launch.os.environ["CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY"] == expected


def test_main_prepares_data_then_trains_with_the_example_config(tmp_path: Path, monkeypatch) -> None:
    kit = make_kit(tmp_path)
    (kit / "grpo_search.yaml").write_text("defaults: grpo_coreweave.yaml\n")
    model = tmp_path / "w" / "models" / "m"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    monkeypatch.setattr(launch, "git_describe", lambda repo: "v0.7.0")
    monkeypatch.setattr(launch, "apply_patches", lambda *a, **kw: None)
    monkeypatch.setattr(launch, "install_into_workers", lambda *a, **kw: None)
    runs: list[list[str]] = []
    monkeypatch.setattr(launch, "run", lambda cmd, **kw: runs.append([str(c) for c in cmd]))
    execs: list[list[str]] = []
    monkeypatch.setattr(launch.os, "execvp", lambda path, args: execs.append(args))
    monkeypatch.setattr(launch.os, "chdir", lambda path: None)
    launch.main(
        {"MODEL_SLUG": "m", "WORK": str(tmp_path / "w"), "CWTRAIN_KIT_DIR": str(kit),
         "NEMO_RL_PYTHON": "/venv/bin/python", "GRPO_CONFIG": "grpo_search.yaml",
         # render.sh flattens the Job's block scalar to one line; shell quoting groups words
         "PREPARE_DATA": "/kit/prepare.py --dataset simpleqa --note 'two words'"}
    )  # fmt: skip
    assert runs == [["/venv/bin/python", "/kit/prepare.py", "--dataset", "simpleqa", "--note", "two words"]]
    assert execs[0][2:4] == ["--config", str(kit / "grpo_search.yaml")]


def test_main_runs_no_preparation_by_default(tmp_path: Path) -> None:
    job = launch.NemoJob.from_env({"MODEL_SLUG": "m", "PREPARE_DATA": "  \n"})
    assert job.prepare_data == () and job.grpo_config == "grpo_coreweave.yaml"


def test_a_grpo_config_missing_from_the_kit_stops_before_patching(tmp_path: Path, monkeypatch) -> None:
    kit = make_kit(tmp_path)
    patched: list[str] = []
    monkeypatch.setattr(launch, "apply_patches", lambda *a, **kw: patched.append("x"))
    with pytest.raises(LaunchError, match="GRPO_CONFIG grpo_search.yaml is not in the kit.*KIT_EXTRA_FILES"):
        launch.main({"MODEL_SLUG": "m", "CWTRAIN_KIT_DIR": str(kit), "GRPO_CONFIG": "grpo_search.yaml"})
    assert patched == []


def test_make_kit_bundles_extra_files_under_their_names(tmp_path: Path) -> None:
    import os
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "kubectl.log"
    # a kubectl that records its arguments, one per line, instead of reaching a cluster
    (bin_dir / "kubectl").write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> {log}\n')
    (bin_dir / "kubectl").chmod(0o755)
    wheel = tmp_path / "cwtrain-0.1.0-py3-none-any.whl"
    wheel.write_text("")
    extra = tmp_path / "example"
    extra.mkdir()
    (extra / "grpo_search.yaml").write_text("")
    (extra / "prepare.py").write_text("")
    script = Path(__file__).resolve().parents[1] / "k8s" / "make-kit.sh"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "CWTRAIN_WHEEL": str(wheel),
           "KIT_EXTRA_FILES": f"{extra / 'grpo_search.yaml'}\n{extra / 'prepare.py'}"}  # fmt: skip
    subprocess.run(["bash", str(script), "ctx", "ns"], check=True, env=env, capture_output=True)
    args = log.read_text().splitlines()
    assert f"--from-file=grpo_search.yaml={extra / 'grpo_search.yaml'}" in args
    assert f"--from-file=prepare.py={extra / 'prepare.py'}" in args
    env["KIT_EXTRA_FILES"] = str(extra / "absent.py")
    failed = subprocess.run(["bash", str(script), "ctx", "ns"], env=env, capture_output=True, text=True)
    assert failed.returncode != 0 and "absent.py is not a file" in failed.stderr


def test_main_refuses_to_train_without_a_model(tmp_path: Path, monkeypatch) -> None:
    kit = make_kit(tmp_path)
    monkeypatch.setattr(launch, "git_describe", lambda repo: "v0.7.0")
    monkeypatch.setattr(launch, "apply_patches", lambda *a, **kw: None)
    monkeypatch.setattr(launch, "install_into_workers", lambda *a, **kw: None)
    with pytest.raises(LaunchError, match=r"no model at .*set BASE_S3_URI.*stage-nemotron-job\.yaml"):
        launch.main({"MODEL_SLUG": "m", "WORK": str(tmp_path / "w"), "CWTRAIN_KIT_DIR": str(kit)})


def test_no_worker_venvs_is_a_configuration_error(tmp_path: Path, monkeypatch) -> None:
    kit = make_kit(tmp_path)
    monkeypatch.setattr(launch, "run", lambda cmd, **kw: None)
    job = launch.NemoJob.from_env(
        {"MODEL_SLUG": "m", "WORK": str(tmp_path / "w"), "CWTRAIN_KIT_DIR": str(kit),
         "NEMO_RL_VENV_DIR": str(tmp_path / "typo")}
    )  # fmt: skip
    with pytest.raises(LaunchError, match="NEMO_RL_VENV_DIR"):
        launch.install_into_workers(job, launch.wheel(job))


def test_a_kit_without_patches_stops_before_anything_else(tmp_path: Path, monkeypatch) -> None:
    kit = make_kit(tmp_path)
    (kit / "0001-backend.patch").unlink()
    monkeypatch.setattr(launch, "git_describe", lambda repo: "v0.7.0")
    installs: list[str] = []
    monkeypatch.setattr(launch, "install_into_workers", lambda *a, **kw: installs.append("x"))
    with pytest.raises(LaunchError, match="no \\*.patch files"):
        launch.main({"MODEL_SLUG": "m", "WORK": str(tmp_path / "w"), "CWTRAIN_KIT_DIR": str(kit)})
    assert installs == []


def test_python_defaults_agree() -> None:
    import sys

    assert launch.NemoJob(model_slug="m").python == sys.executable
    assert launch.NemoJob.from_env({"MODEL_SLUG": "m"}).python == sys.executable
