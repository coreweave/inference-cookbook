# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""NeMo-RL trainer launcher. ``k8s/run-trainer.sh`` runs it with the NeMo-RL venv's interpreter.

The launcher patches NeMo-RL for the ``coreweave`` generation backend, makes ``cwtrain``
importable in the policy workers' environments and stages the base model from the deployment's
bucket. Then it replaces itself with ``examples/run_grpo.py`` and the recipe's config. It starts no
vLLM or SGLang workers: the deployment serves the rollouts and receives every refit by hot-load.

Settings are the environment variables that an example's ``k8s/run.env.example`` documents.
``TRAIN_ENV`` picks the training environment: ``browsecomp`` (the default) runs Gym's BrowseComp
agent, with You.com search and an LLM judge; ``dapo`` runs Gym's DAPO math; ``math`` runs NeMo-RL's
own math environment.
``EXTRA_OVERRIDES`` holds Hydra-style overrides. Each whitespace-separated word is one override,
and quotes reach Hydra unchanged (``policy.generation.stop_strings=["</answer>"]``). So write lists
without spaces (``["</answer>","<|end|>"]``).
"""

from __future__ import annotations

import os
import sys
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from cwtrain.integrations.launch import (
    LaunchError,
    apply_patches,
    default_prompt_cache_policy,
    env_int,
    env_text,
    git_describe,
    require,
    require_patches,
    run,
    say,
    stage_from_s3,
)

TESTED = "NeMo-RL v0.7.0 (nvcr.io/nvidia/nemo-rl:v0.7.0)"
TRAIN_ENVS = ("browsecomp", "dapo", "math")
# Trajectory token limit for each environment when MAX_SEQUENCE_LENGTH is unset. A BrowseComp
# trajectory holds every search result and turn; the math prompts are short. 32768 runs the
# 8-GPU trainer out of memory: one sequence's fp32 logits need another 16 GiB on top of 60 GiB.
SEQUENCE_LENGTHS = {"browsecomp": 16384, "dapo": 3072, "math": 3072}


@dataclass(frozen=True)
class NemoJob:
    model_slug: str
    work: Path = Path("/work")
    kit: Path = Path("/opt/cwtrain")
    num_steps: int = 6
    actor_gpus: int = 8
    nemo_rl_dir: Path = Path("/opt/nemo-rl")
    # the NeMo-RL venv interpreter that run-trainer.sh uses to start the launcher
    python: str = field(default_factory=lambda: sys.executable)
    worker_venvs: Path = Path("/opt/ray_venvs")
    base_s3_uri: str = ""
    train_env: str = "browsecomp"
    max_sequence_length: int = 0  # 0: SEQUENCE_LENGTHS[train_env]
    extra_overrides: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> NemoJob:
        env = os.environ if environ is None else environ

        def text(name: str, default: str) -> str:
            return env_text(env, name, default)

        train_env = text("TRAIN_ENV", "browsecomp").lower()
        if train_env not in TRAIN_ENVS:
            raise LaunchError(f"TRAIN_ENV must be one of {', '.join(TRAIN_ENVS)}, got {train_env!r}")
        return cls(
            model_slug=require(env, "MODEL_SLUG"),
            work=Path(text("WORK", "/work")),
            kit=Path(text("CWTRAIN_KIT_DIR", "/opt/cwtrain")),
            num_steps=env_int(env, "NUM_STEPS", 6),
            actor_gpus=env_int(env, "ACTOR_GPUS", 8),
            nemo_rl_dir=Path(text("NEMO_RL_DIR", "/opt/nemo-rl")),
            python=text("NEMO_RL_PYTHON", sys.executable),
            worker_venvs=Path(text("NEMO_RL_VENV_DIR", "/opt/ray_venvs")),
            base_s3_uri=text("BASE_S3_URI", ""),
            train_env=train_env,
            max_sequence_length=env_int(env, "MAX_SEQUENCE_LENGTH", 0),
            # whitespace-separated, quotes kept for Hydra's own parser
            extra_overrides=tuple((env.get("EXTRA_OVERRIDES") or "").split()),
        )

    @property
    def model_dir(self) -> Path:
        return self.work / "models" / self.model_slug


def wheel(job: NemoJob) -> Path:
    found = sorted(job.kit.glob("cwtrain-*.whl"))
    if not found:
        raise LaunchError(f"no cwtrain wheel in {job.kit}; rebuild the kit with make-kit.sh")
    return found[-1]


def environment_overrides(job: NemoJob) -> list[str]:
    """The overrides that select ``job.train_env`` and its trajectory token limit."""
    overrides = {
        "browsecomp": ["env.should_use_nemo_gym=true", "data.train.dataset_name=BrowseComp"],
        "dapo": ["env.should_use_nemo_gym=true"],
        "math": [],
    }[job.train_env]
    length = job.max_sequence_length or SEQUENCE_LENGTHS[job.train_env]
    return [*overrides, f"policy.max_total_sequence_length={length}"]


def grpo_command(job: NemoJob) -> list[str]:
    """``run_grpo.py`` with the recipe's config and the job's overrides."""
    return [
        job.python,
        "examples/run_grpo.py",
        "--config",
        str(job.kit / "grpo_coreweave.yaml"),
        f"policy.model_name={job.model_dir}",
        f"grpo.max_num_steps={job.num_steps}",
        f"cluster.gpus_per_node={job.actor_gpus}",
        *environment_overrides(job),
        *job.extra_overrides,
    ]


def install_into_workers(job: NemoJob, whl: Path) -> list[Path]:
    """Make cwtrain importable where Ray imports the worker extension.

    Ray imports the policy worker class in NeMo-RL's prebuilt per-class venvs before any
    configured env vars apply, so PYTHONPATH does not reach that import. So the wheel goes into
    each prebuilt venv without dependencies (the extension needs only torch and safetensors). It
    is also unpacked to ``<work>/py`` for venvs that NeMo-RL builds fresh. The recipe config puts
    that directory on their PYTHONPATH.
    """
    installed = []
    for python in sorted(job.worker_venvs.glob("*/bin/python")):
        if not os.access(python, os.X_OK):
            say(f"skipping {python}: not executable")
            continue
        run(["uv", "pip", "install", "--quiet", "--no-deps", "--python", python, whl])
        installed.append(python)
    if not installed:
        # Fail now: otherwise the Job fails late in worker startup, after staging the model.
        raise LaunchError(
            f"no executable */bin/python under {job.worker_venvs}; set NEMO_RL_VENV_DIR to the "
            "image's prebuilt Ray worker venvs"
        )
    py_dir = job.work / "py"
    py_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(whl) as archive:
        archive.extractall(py_dir)
    say(f"cwtrain installed into {len(installed)} worker venvs and unpacked to {py_dir}")
    return installed


def main(environ: Mapping[str, str] | None = None) -> None:
    job = NemoJob.from_env(environ)

    say(f"== patch NeMo-RL ({git_describe(job.nemo_rl_dir)})")
    apply_patches(job.nemo_rl_dir, require_patches(job.kit), tested=TESTED)

    say("== install cwtrain into the worker venvs")
    install_into_workers(job, wheel(job))

    say("== stage model")
    if job.base_s3_uri:
        stage_from_s3(job.base_s3_uri, job.model_dir)
    if not (job.model_dir / "config.json").is_file():
        raise LaunchError(
            f"no model at {job.model_dir}: set BASE_S3_URI to the staged checkpoint "
            "(s3://<bucket>/<model prefix>/<model slug>, where stage-nemotron-job.yaml uploads it), "
            "or mount the checkpoint there"
        )

    say("== train")
    # Use the venv's interpreter, not uv run: uv run re-syncs the venv and drops cwtrain.
    cmd = grpo_command(job)
    os.chdir(job.nemo_rl_dir)
    os.environ["PYTHONUNBUFFERED"] = "1"
    # GRPO is synchronous, so each rollout round must not reuse prefix cache from the previous
    # weights. The policy reaches the synchronizer through os.environ, which execvp passes to
    # run_grpo.py. hook_config_from reads CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY when coreweave_cfg
    # sets no prompt_cache_policy.
    default_prompt_cache_policy(os.environ)
    sys.stdout.flush()
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
