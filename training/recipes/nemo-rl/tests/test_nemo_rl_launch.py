# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Unit tests for the NeMo-RL launcher (no NeMo-RL, Ray or GPUs needed)."""

from __future__ import annotations

import importlib.util
import re
import subprocess
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


K8S = Path(__file__).resolve().parents[1] / "k8s"


def test_run_env_fetches_the_base_model_from_where_the_stage_job_puts_it() -> None:
    # NeMo-RL has no Hugging Face fallback, so an empty BASE_S3_URI cannot train
    wanted = ("DEPLOYMENT_RUN", "BUCKET", "MODEL_PREFIX", "MODEL_SLUG", "BASE_S3_URI")
    lines = [line for line in (K8S / "run.env.example").read_text().splitlines()
             if line.split("=", 1)[0] in wanted]  # fmt: skip
    show = 'printf "%s\\n" "$BUCKET" "$MODEL_PREFIX" "$MODEL_SLUG" "$BASE_S3_URI"'
    bucket, prefix, slug, uri = subprocess.run(
        ["bash", "-c", "; ".join(["set -eu", *lines, show])],
        check=True, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
    ).stdout.splitlines()
    assert uri == f"s3://{bucket}/{prefix}/{slug}"
    # ...which is where stage-nemotron.sh uploads the checkpoint: s3://$BUCKET/$MODEL_PREFIX/$MODEL_SLUG/
    stage = (K8S / "stage-nemotron.sh").read_text()
    upload = re.search(r"upload_snapshot\((.*?)\)\n", stage, re.DOTALL).group(1)
    for arg in ('bucket=os.environ["BUCKET"]', 'path_prefix=os.environ["MODEL_PREFIX"]',
                'identity=os.environ["MODEL_SLUG"]'):  # fmt: skip
        assert arg in upload
