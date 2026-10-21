# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Launch miles RL against a CoreWeave hot-load deployment, in the style of miles' scripts/run_*.py.

miles ships one launch script per model under ``scripts/`` (``run_qwen3_4b.py``, ...). Each is a
typer CLI that composes argument groups and passes them to miles' launcher helpers
(``convert_checkpoint``, ``execute_train``). The model architecture flags come from
``scripts/models/<type>.py``. The helpers are in ``miles.utils.external_utils.command_utils`` up to
miles v0.1.1, and in its ``legacy`` module after that; this script takes them from either.

The stock scripts own SGLang engines, so they hardcode ``--colocate`` and the ``--sglang-*`` flags.
This topology has no engines, and ``--extra-args`` cannot remove a store_true flag. So this script
uses the same helpers and model definitions, but replaces the engine-side arguments with the
CoreWeave ones:

* ``--rollout-num-gpus 0`` and the recipe's rollout function (the deployment is the rollout engine);
* the disk-delta publisher plus the ``cwtrain`` post-write hook that hot-loads every version;
* the ``CWTRAIN_*`` / ``COREWEAVE_*`` / ``AWS_*`` / ``CW_S3_*`` environment forwarded to the Ray workers.

Any model with a ``scripts/models/<type>.py`` works. ``--model-name`` takes a Hugging Face repo id
(``org/name``) or a bare name in the Qwen org; the checkpoint lands in ``--model-dir/<name>``::

    python3 run_hotload.py --model-name Qwen3-8B --tp 2 --num-gpus 2 --num-rollout 30
    python3 run_hotload.py --model-name meta-llama/Llama-3.1-8B --tp 2 --num-gpus 2

``--tis`` (default ``1``) passes miles' ``--use-tis`` in both loops: truncated importance sampling
against the deployment's log-probs, which every sample then needs.

``--train-script train_async.py`` selects miles' one-step-ahead loop. It generates rollout ``i+1``
while step ``i`` trains, and drains the in-flight generation before the weight update; then the
hook hot-loads. Samples are one policy step stale, so ``--async-correction`` picks the ratio's
denominator:

* ``keep_old_actor`` recomputes it with the weights that generated the rollout;
* ``none`` recomputes it with the current weights, which leaves the stale step to TIS, or, with
  ``--tis 0``, trains plain GRPO on stale data.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

from cwtrain.integrations.launch import (
    LaunchError,
    default_prompt_cache_policy,
    env_flag,
    forwarded_env,
    stage_from_s3,
)

DAPO = "zhuzilin/dapo-math-17k"
PINNED_IMAGE = "radixark/miles:v0.1.1"
HELPERS = (
    "exec_command_cpu", "hf_download_dataset", "convert_checkpoint", "execute_train", "ExecuteTrainConfig",
)


COMMAND_UTILS = "miles.utils.external_utils.command_utils"


def command_utils() -> tuple[ModuleType, str]:
    """Return miles' launcher helpers and, if they may be missing, the reason.

    The helpers are in ``command_utils.legacy`` after miles v0.1.1, and in ``command_utils`` up to
    v0.1.1. This never raises. If an import fails, it returns a module without the helpers, and
    :func:`check_helpers` reports it with the pinned image.
    """
    try:
        from miles.utils.external_utils.command_utils import legacy

        if not isinstance(legacy, ModuleType):  # an attribute of that name, not the module
            raise ImportError(f"command_utils.legacy is a {type(legacy).__name__}, not a module")
        return legacy, ""
    except ImportError as legacy_error:
        try:
            import miles.utils.external_utils.command_utils as utils
        except ImportError as exc:
            return ModuleType(COMMAND_UTILS), f" ({COMMAND_UTILS} cannot be imported: {exc})"
        # v0.1.x: command_utils is a plain module with the helpers and no legacy submodule.
        # Report the failed legacy import only for the package layout, where it is expected.
        if hasattr(utils, "__path__"):
            return utils, f" (importing command_utils.legacy failed: {legacy_error})"
        return utils, ""


U, U_PROBLEM = command_utils()


def check_helpers(utils: ModuleType, problem: str = "") -> None:
    """Stop before any download if this miles has the helpers in an unknown module."""
    missing = [name for name in HELPERS if not hasattr(utils, name)]
    if missing:
        raise SystemExit(
            f"{utils.__name__} has no {', '.join(missing)}{problem}: this miles moved its launcher "
            f"helpers. Use the image that the example's run.env.example pins ({PINNED_IMAGE}), "
            "or point run_hotload.py's command_utils() at their new module."
        )


def checkpoint_paths(model_name_arg: str, model_dir: str) -> tuple[str, str, str]:
    """``(HuggingFace repo, model name, local checkpoint dir)`` for a ``--model-name`` value.

    ``org/name`` is a Hugging Face repo id; a bare name is a model in the Qwen org.
    """
    model_name_arg = model_name_arg.strip().rstrip("/")
    hf_repo = model_name_arg if "/" in model_name_arg else f"Qwen/{model_name_arg}"
    model_name = hf_repo.rsplit("/", 1)[1]
    return hf_repo, model_name, f"{model_dir}/{model_name}"


def megatron_model_type(model_name: str) -> str:
    """``scripts/models/<type>.py`` for a Hugging Face model name (``Qwen3-4B`` -> ``qwen3-4B``)."""
    try:
        from miles.true_on_policy.model_profiles import get_megatron_model_type

        return get_megatron_model_type(model_name)
    except Exception:  # noqa: BLE001 - profile tables only cover some models
        return model_name[:1].lower() + model_name[1:]


def on_off(raw: str) -> bool:
    """``--tis`` takes the values of cwtrain's boolean settings: 1/0, true/false, yes/no, on/off."""
    try:
        return env_flag({"--tis": raw}, "--tis", True)
    except LaunchError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def async_correction(raw: str) -> str:
    """``--async-correction``: ``keep_old_actor`` or ``none``. ``tis`` is now ``--tis``."""
    if raw == "tis":
        raise argparse.ArgumentTypeError(
            "tis is no longer an async correction: TIS is on by default in both loops (USE_TIS=1, "
            "--tis 1). For what tis did, TIS without the old actor, set ASYNC_CORRECTION=none"
        )
    if raw not in ("keep_old_actor", "none"):
        raise argparse.ArgumentTypeError(f"choose keep_old_actor or none, got {raw!r}")
    return raw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default="Qwen3-4B",
                   help="Hugging Face repo id (org/name), or a bare name in the Qwen org")
    p.add_argument("--megatron-model-type", default=None, help="scripts/models/<type>.py; derived from --model-name")
    p.add_argument("--model-dir", default="/work/models")
    p.add_argument("--data-dir", default="/work/datasets")
    p.add_argument("--work-dir", default="/work", help="deltas, mirror, checkpoints and outputs live here")
    p.add_argument("--megatron-path", default="/root/Megatron-LM")
    p.add_argument("--num-gpus", type=int, default=2, help="actor GPUs on this node")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--num-rollout", type=int, default=6)
    p.add_argument("--rollout-batch-size", type=int, default=8)
    p.add_argument("--n-samples-per-prompt", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--rollout-max-response-len", type=int, default=2048)
    p.add_argument("--max-tokens-per-gpu", type=int, default=9216)
    p.add_argument("--rm-type", default="math")
    p.add_argument("--prompt-data", default=None, help="jsonl with prompt/label keys; default dapo-math-17k")
    p.add_argument("--train-script", default="train.py", choices=["train.py", "train_async.py"])
    p.add_argument("--tis", type=on_off, default=True, help="truncated importance sampling: 1 or 0")
    p.add_argument("--async-correction", default="keep_old_actor", type=async_correction,
                   help="train_async.py's ratio denominator: keep_old_actor or none")
    p.add_argument("--rollout-function", default="coreweave_rollout.CoreWeaveRolloutFn")
    p.add_argument("--rollout-module-dir", default=str(Path(__file__).resolve().parent),
                   help="directory holding coreweave_rollout.py (added to the workers' PYTHONPATH)")
    p.add_argument("--skip-download", action="store_true", help="the model is already under --model-dir")
    p.add_argument("--base-s3-uri", default="",
                   help="s3:// prefix of the staged base model, fetched instead of Hugging Face")
    p.add_argument("--extra-args", default="", help="appended to train.py's arguments, shell-quoted as given")
    return p.parse_args()


def worker_env(train_script: str, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """What the hook and the rollout function read inside the Ray workers."""
    env = forwarded_env(environ)
    if train_script == "train.py":
        # The synchronous loop starts every rollout round from the weights just hot-loaded;
        # train_async.py generates one step ahead anyway and keeps the platform default.
        default_prompt_cache_policy(env)
    return env


def main() -> None:
    args = parse_args()
    check_helpers(U, U_PROBLEM)
    hf_repo, model_name, hf_checkpoint = checkpoint_paths(args.model_name, args.model_dir)
    model_type = args.megatron_model_type or megatron_model_type(model_name)
    prompt_data = args.prompt_data or f"{args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl"

    # ---- prepare: model, data, Megatron torch_dist conversion (same helpers as scripts/run_*.py)
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir} {args.work_dir}")
    if args.base_s3_uri:
        # Fetch the copy the deployment loaded into the directory that every later step reads.
        # The hook publishes it as the run's chain root. Skipped if a checkpoint is already there.
        stage_from_s3(args.base_s3_uri, hf_checkpoint)
    if not args.skip_download and not Path(hf_checkpoint, "config.json").exists():
        revision = os.environ.get("HF_REVISION", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise SystemExit("HF_REVISION must be a full 40-character Hugging Face commit SHA")
        U.exec_command_cpu(
            f"hf download {shlex.quote(hf_repo)} --revision {revision} "
            f"--local-dir {shlex.quote(hf_checkpoint)}"
        )
    if args.prompt_data is None and not Path(prompt_data).exists():
        U.hf_download_dataset(DAPO, data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=model_name,
        megatron_model_type=model_type,
        num_gpus_per_node=args.num_gpus,
        dir_dst=args.model_dir,
        hf_checkpoint=hf_checkpoint,
        megatron_path=args.megatron_path,
    )

    # ---- train arguments: the stock groups minus --colocate and --sglang-*, plus CoreWeave's
    ckpt_args = (
        f"--hf-checkpoint {hf_checkpoint} "
        f"--ref-load {args.model_dir}/{model_name}_torch_dist "
        f"--save {args.work_dir}/ckpt --save-interval 100000 "
    )
    rollout_args = (
        f"--prompt-data {prompt_data} --input-key prompt --label-key label "
        f"--apply-chat-template --rollout-shuffle --rm-type {args.rm_type} "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} --rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} --balance-data "
    )
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )
    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )
    perf_args = f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
    train_backend_args = (
        f"--tensor-model-parallel-size {args.tp} {'--sequence-parallel ' if args.tp > 1 else ''}"
        f"--pipeline-model-parallel-size {args.pp} --context-parallel-size {args.cp} "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash "
    )
    misc_args = f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.num_gpus} --num-gpus-per-node {args.num_gpus} "
    coreweave_args = (
        # the deployment is the only rollout engine
        f"--rollout-num-gpus 0 --rollout-function-path {args.rollout_function} "
        # zstd XOR deltas with adler32 checksums on local disk; the hook hot-loads each version
        "--update-weight-transfer-mode disk-delta "
        f"--update-weight-disk-dir {args.work_dir}/deltas "
        f"--update-weight-local-checkpoint-dir {args.work_dir}/local-ckpt "
        "--update-weight-delta-encoding xor --update-weight-delta-checksum adler32 "
        "--custom-update-weight-post-write-path cwtrain.integrations.slime.hook "
    )
    # truncated importance sampling against the deployment's log-probs (README)
    tis_args = "--use-tis " if args.tis else ""
    async_args = ""
    if args.train_script == "train_async.py":
        async_args = {"keep_old_actor": "--keep-old-actor ", "none": ""}[args.async_correction]
    # Split and re-join: unbalanced quotes fail here, before any GPU work, and every token reaches
    # train.py quoted exactly once.
    extra = shlex.join(shlex.split(args.extra_args)) + " " if args.extra_args.strip() else ""

    train_args = (
        ckpt_args + rollout_args + optimizer_args + grpo_args + tis_args + perf_args
        + train_backend_args + misc_args + coreweave_args + async_args + extra
    )

    # ---- environment: the hook and the rollout function read these inside the Ray workers
    env = worker_env(args.train_script)
    # miles builds the workers' PYTHONPATH from this process's PYTHONPATH, plus its own repo and
    # Megatron. So prepend the rollout module here; PYTHONPATH in extra_env_vars would replace the
    # paths that miles adds.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (args.rollout_module_dir, os.environ.get("PYTHONPATH", "")) if p
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus,
        megatron_model_type=model_type,
        train_script=args.train_script,
        extra_env_vars=env,
        config=U.ExecuteTrainConfig(output_dir=f"{args.work_dir}/output"),
        megatron_path=args.megatron_path,
    )


if __name__ == "__main__":
    main()
