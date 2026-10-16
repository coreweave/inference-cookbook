# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""slime trainer launcher: run by ``k8s/run-trainer.sh`` inside the trainer Job.

Stages the base model and data, prepares slime (pinned commit, the recipe's patches, and slime's
Megatron-LM when the image's differs), converts the checkpoint to Megatron ``torch_dist``, starts
Ray and submits ``train.py`` with disk-delta publication into the ``cwtrain`` hot-load hook. No
SGLang engines are started: ``--rollout-num-gpus 0``, and rollouts come from the deployment.

Settings are environment variables, the same ones ``k8s/run.env.example`` documents. Every
argument is built as a list, so values are never re-split or re-quoted by a shell;
``EXTRA_TRAIN_ARGS`` is split shell-style (quotes respected, nothing expanded).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cwtrain.integrations.launch import (
    FORWARDED_PREFIXES,
    LaunchError,
    apply_patches,
    bash_array,
    checkout,
    default_prompt_cache_policy,
    env_int,
    env_text,
    fetch_commit,
    forwarded_env,
    git_describe,
    require,
    require_patches,
    run,
    say,
    split_args,
    stage_from_s3,
)

#: The slime commit this recipe and its patches were validated against (README, "Tested versions").
TESTED_SLIME_COMMIT = "8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b"
DATASET = "zhuzilin/dapo-math-17k"


@dataclass(frozen=True)
class SlimeJob:
    hf_model: str
    model_slug: str
    hf_revision: str = ""
    work: Path = Path("/work")
    kit: Path = Path("/opt/cwtrain")
    num_rollout: int = 6
    actor_gpus: int = 2
    tp_size: int = 2
    model_script: str = "qwen3-4B"
    slime_repo: str = "https://github.com/THUDM/slime"
    slime_commit: str = TESTED_SLIME_COMMIT
    slime_dir: str = ""
    #: Where the official slime images keep their clone of slime main.
    image_slime_dir: str = "/root/slime"
    megatron_dir: str = "/root/Megatron-LM"
    megatron_commit: str = ""
    megatron_repo: str = "https://github.com/NVIDIA/Megatron-LM"
    slime_patch_version: str = "latest"
    base_s3_uri: str = ""
    rm_type: str = "math"
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 4
    rollout_max_response_len: int = 2048
    global_batch_size: int = 32
    max_tokens_per_gpu: int = 9216
    extra_train_args: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SlimeJob:
        env = os.environ if environ is None else environ

        def text(name: str, default: str) -> str:
            return env_text(env, name, default)

        return cls(
            hf_model=require(env, "HF_MODEL"),
            hf_revision=text("HF_REVISION", ""),
            model_slug=require(env, "MODEL_SLUG"),
            work=Path(text("WORK", "/work")),
            kit=Path(text("CWTRAIN_KIT_DIR", "/opt/cwtrain")),
            num_rollout=env_int(env, "NUM_ROLLOUT", 6),
            actor_gpus=env_int(env, "ACTOR_GPUS", 2),
            tp_size=env_int(env, "TP_SIZE", 2),
            model_script=text("SLIME_MODEL_SCRIPT", "qwen3-4B"),
            slime_repo=text("SLIME_REPO", "https://github.com/THUDM/slime"),
            slime_commit=text("SLIME_COMMIT", TESTED_SLIME_COMMIT),
            slime_dir=text("SLIME_DIR", ""),
            megatron_dir=text("MEGATRON_DIR", "/root/Megatron-LM"),
            megatron_commit=text("MEGATRON_COMMIT", ""),
            megatron_repo=text("MEGATRON_REPO", "https://github.com/NVIDIA/Megatron-LM"),
            slime_patch_version=text("SLIME_PATCH_VERSION", "latest"),
            base_s3_uri=text("BASE_S3_URI", ""),
            rm_type=text("RM_TYPE", "math"),
            rollout_batch_size=env_int(env, "ROLLOUT_BATCH_SIZE", 8),
            n_samples_per_prompt=env_int(env, "N_SAMPLES_PER_PROMPT", 4),
            rollout_max_response_len=env_int(env, "ROLLOUT_MAX_RESPONSE_LEN", 2048),
            global_batch_size=env_int(env, "GLOBAL_BATCH_SIZE", 32),
            max_tokens_per_gpu=env_int(env, "MAX_TOKENS_PER_GPU", 9216),
            extra_train_args=tuple(
                split_args(env.get("EXTRA_TRAIN_ARGS"), name="EXTRA_TRAIN_ARGS")
            ),
        )

    @property
    def model_dir(self) -> Path:
        return self.work / "models" / self.model_slug

    @property
    def prompt_data(self) -> Path:
        return self.work / "datasets" / "dapo-math-17k" / "dapo-math-17k.jsonl"


def train_args(job: SlimeJob) -> list[str]:
    """``train.py``'s arguments after the model architecture flags."""
    return [
        "--hf-checkpoint", str(job.model_dir),
        "--ref-load", f"{job.model_dir}_torch_dist",
        "--prompt-data", str(job.prompt_data), "--input-key", "prompt", "--label-key", "label",
        "--apply-chat-template", "--rollout-shuffle", "--rm-type", job.rm_type,
        "--num-rollout", str(job.num_rollout),
        "--rollout-batch-size", str(job.rollout_batch_size),
        "--n-samples-per-prompt", str(job.n_samples_per_prompt),
        "--rollout-max-response-len", str(job.rollout_max_response_len),
        "--rollout-temperature", "1",
        "--global-batch-size", str(job.global_batch_size), "--balance-data",
        "--optimizer", "adam", "--lr", "1e-6", "--lr-decay-style", "constant",
        "--weight-decay", "0.1", "--adam-beta1", "0.9", "--adam-beta2", "0.98",
        "--advantage-estimator", "grpo", "--kl-loss-coef", "0.00", "--kl-loss-type", "low_var_kl",
        "--entropy-coef", "0.00", "--eps-clip", "0.2", "--eps-clip-high", "0.28",
        "--tensor-model-parallel-size", str(job.tp_size),
        # sequence parallelism splits activations across tensor-parallel ranks; it needs TP > 1
        *(["--sequence-parallel"] if job.tp_size > 1 else []),
        "--pipeline-model-parallel-size", "1", "--context-parallel-size", "1",
        "--expert-model-parallel-size", "1", "--expert-tensor-parallel-size", "1",
        "--recompute-granularity", "full", "--recompute-method", "uniform",
        "--recompute-num-layers", "1",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", str(job.max_tokens_per_gpu),
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", str(job.actor_gpus),
        "--num-gpus-per-node", str(job.actor_gpus),
        # the deployment is the only rollout engine
        "--rollout-num-gpus", "0",
        "--rollout-function-path", "coreweave_rollout.generate_rollout",
        # publish zstd XOR deltas with adler32 checksums to disk; the hook hot-loads each version
        "--update-weight-mode", "delta", "--update-weight-transport", "disk",
        "--update-weight-standalone-publish",
        "--update-weight-disk-dir", str(job.work / "deltas"),
        "--update-weight-delta-encoding", "xor", "--update-weight-delta-checksum", "adler32",
        "--custom-update-weight-post-write-path", "cwtrain.integrations.slime.hook",
        "--save", str(job.work / "ckpt"), "--save-interval", "100000",
        *job.extra_train_args,
    ]  # fmt: skip


def runtime_env(pythonpath: str, environ: Mapping[str, str] | None = None) -> dict:
    """Ray runtime env: everything the hook and the rollout function read inside the workers."""
    env = forwarded_env(environ, (*FORWARDED_PREFIXES, "SLIME_", "MEGATRON_"))
    # train.py is synchronous: each rollout round starts from the weights just hot-loaded, without
    # prefix-cache entries computed under the previous ones
    default_prompt_cache_policy(env)
    env.update(
        {
            "PYTHONPATH": pythonpath,
            "PYTHONUNBUFFERED": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "no_proxy": "127.0.0.1",
            "MASTER_ADDR": "127.0.0.1",
        }
    )
    return {"env_vars": env}


def at_commit(url: str, commit: str, dest: Path) -> Path:
    """``dest`` checked out at ``commit``: cloned when absent, fetched when the clone lacks it."""
    if not (dest / ".git").is_dir():
        return checkout(url, commit, dest)
    if run(["git", "-C", dest, "checkout", "--quiet", commit], check=False, capture=True).returncode:
        # the image's clone is older than the commit
        run(["git", "-C", dest, "fetch", "--quiet", url, commit])
        run(["git", "-C", dest, "checkout", "--quiet", "FETCH_HEAD"])
    return dest


def prepare_slime(job: SlimeJob) -> Path:
    """slime at ``SLIME_COMMIT``, patched; ``SLIME_DIR`` names a checkout to use as it is.

    The official slime images carry a clone of slime main, which the recipe's patches may not fit.
    That clone is moved to ``SLIME_COMMIT``; other images get a fresh clone.
    """
    if job.slime_dir:
        slime = Path(job.slime_dir)
    else:
        image = Path(job.image_slime_dir)
        dest = image if (image / ".git").is_dir() else job.work / "slime"
        slime = at_commit(job.slime_repo, job.slime_commit, dest)
    say(f"slime at {slime} ({git_describe(slime)})")
    # 0001 standalone publish (always needed); 0002 relaxes slime's numpy-1 assertion only with
    # SLIME_RELAX_NUMPY_CHECK=1; 0003 is the zero-KL fallback without rollout log-probs
    apply_patches(slime, require_patches(job.kit), tested=f"slime {job.slime_commit[:12]}")
    return slime


def prepare_megatron(job: SlimeJob, slime: Path) -> Path:
    """slime's own Megatron-LM commit and patches, when the image's Megatron differs."""
    if not job.megatron_commit:
        return Path(job.megatron_dir)
    megatron = job.work / "Megatron-LM"
    if not (megatron / ".git").is_dir():
        fetch_commit(job.megatron_repo, job.megatron_commit, megatron)
    # Patch on every start, not only after a fetch: a run that died while patching leaves a
    # half-patched tree, which apply_patches completes, or reverts and reports.
    patch_dir = slime / "docker" / "patch" / job.slime_patch_version
    apply_patches(
        megatron,
        require_patches(patch_dir, "megatron*.patch"),
        tested=f"slime {job.slime_commit[:12]} ({job.slime_patch_version} Megatron patches)",
    )
    say(f"Megatron-LM at {megatron} ({git_describe(megatron)})")
    return megatron


def stage(job: SlimeJob) -> None:
    """Base model (the deployment's own bytes when BASE_S3_URI is set) and the prompt data."""
    if not (job.model_dir / "config.json").is_file():
        if job.base_s3_uri:
            stage_from_s3(job.base_s3_uri, job.model_dir)
        else:
            if not re.fullmatch(r"[0-9a-f]{40}", job.hf_revision):
                raise LaunchError("HF_REVISION must be a full 40-character Hugging Face commit SHA")
            run(["hf", "download", job.hf_model, "--revision", job.hf_revision,
                 "--local-dir", job.model_dir,
                 "--exclude", "original/*"])  # fmt: skip
    if not job.prompt_data.is_file():
        run(["hf", "download", "--repo-type", "dataset", DATASET,
             "--local-dir", job.prompt_data.parent])  # fmt: skip


def main(environ: Mapping[str, str] | None = None) -> None:
    job = SlimeJob.from_env(environ)

    say("== rollout module")
    py_dir = job.work / "py"
    py_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(job.kit / "coreweave_rollout.py", py_dir / "coreweave_rollout.py")

    say("== slime source")
    slime = prepare_slime(job)
    megatron = prepare_megatron(job, slime)
    pythonpath = os.pathsep.join(
        p for p in (str(py_dir), str(slime), str(megatron), os.environ.get("PYTHONPATH", "")) if p
    )
    os.environ["PYTHONPATH"] = pythonpath

    say("== stage model and data")
    stage(job)
    model_args = bash_array(
        slime / "scripts" / "models" / f"{job.model_script}.sh", "MODEL_ARGS", cwd=str(slime)
    )

    say("== convert to Megatron torch_dist")
    torch_dist = Path(f"{job.model_dir}_torch_dist")
    if not (torch_dist / "latest_checkpointed_iteration.txt").is_file():
        run(["torchrun", "--nproc-per-node", str(job.actor_gpus),
             "tools/convert_hf_to_torch_dist.py", *model_args,
             "--hf-checkpoint", job.model_dir, "--save", torch_dist], cwd=slime)  # fmt: skip

    say("== start ray")
    run(["ray", "stop", "--force"], check=False, capture=True)
    # Ray runs a submitted job's entrypoint in the directory of `ray start`, so start Ray in the
    # slime checkout. Name train.py by its full path as well.
    run(["ray", "start", "--head", "--node-ip-address", "127.0.0.1",
         "--num-gpus", str(job.actor_gpus), "--disable-usage-stats"], cwd=slime)  # fmt: skip

    say("== train")
    cmd = ["ray", "job", "submit", "--runtime-env-json", json.dumps(runtime_env(pythonpath, environ)),
           "--", "python3", str(slime / "train.py"), *model_args, *train_args(job)]  # fmt: skip
    os.chdir(slime)
    sys.stdout.flush()
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
