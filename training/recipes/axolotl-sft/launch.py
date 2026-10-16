# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Axolotl SFT launcher: run by ``k8s/run-trainer.sh`` inside the trainer Job.

Stages the base model, runs stock ``axolotl train`` with the recipe's config (and ``axolotl
merge-lora`` after a LoRA run), checks that the result holds the base model's tensors, and
publishes it into the deployment as one full hot-load snapshot, ``<RUN_ID>-v<MAX_STEPS:06d>``.
Nothing runs inside Axolotl's processes, and they do not get the CoreWeave API credentials.

Settings are environment variables, the ones that an example's ``k8s/run.env.example``
documents.
``AXOLOTL_CONFIG`` names another Axolotl config than the kit's ``sft_coreweave.yaml``. Every
argument is built as a list, so values are never re-split or re-quoted by a shell;
``EXTRA_TRAIN_ARGS`` is split shell-style (quotes respected, nothing expanded) and goes to
``axolotl train`` and ``axolotl merge-lora`` as config overrides.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cwtrain.errors import CWTrainError
from cwtrain.hotload import HotLoad, HotLoader, PromptCachePolicy
from cwtrain.hotload.delta import write_full_snapshot
from cwtrain.hotload.mirror import TensorLocation, tensor_locations
from cwtrain.hotload.snapshot import INDEX_FILE, validate_identity
from cwtrain.infra import DEFAULT_BASE_URL, TOKEN_ENV_VARS, InferenceClient, token_from_env
from cwtrain.integrations.launch import (
    LaunchError,
    env_int,
    env_text,
    require,
    run,
    say,
    split_args,
    stage_from_s3,
)

TIED_HEAD, EMBEDDING = "lm_head.weight", "model.embed_tokens.weight"
#: Axolotl never needs the CoreWeave API credentials, so its processes do not get them.
API_CREDENTIALS = (*TOKEN_ENV_VARS, "CWTRAIN_API_HEADERS")
_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}
_COMPARE_CHUNK = 64 * 2**20


def env_flag(environ: Mapping[str, str], name: str, default: bool) -> bool:
    """A boolean setting; unset or blank means ``default``."""
    raw = environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw not in _TRUE | _FALSE:
        raise LaunchError(f"{name} must be 1 or 0, got {raw!r}")
    return raw in _TRUE


@dataclass(frozen=True)
class SftJob:
    hf_model: str
    model_slug: str
    deployment_id: str
    run_id: str
    hf_revision: str = ""
    work: Path = Path("/work")
    config: Path = Path("/opt/cwtrain/sft_coreweave.yaml")
    gpus: int = 2
    max_steps: int = 30
    base_s3_uri: str = ""
    wait: bool = True
    timeout: float = 1800.0
    verify: bool = False
    policy: PromptCachePolicy = PromptCachePolicy.RESET_ALL
    extra_train_args: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SftJob:
        env = os.environ if environ is None else environ

        def text(name: str, default: str) -> str:
            return env_text(env, name, default)

        extra = tuple(split_args(env.get("EXTRA_TRAIN_ARGS"), name="EXTRA_TRAIN_ARGS"))
        if any(re.fullmatch(r"--max[-_]steps(=.*)?", arg) for arg in extra):
            raise LaunchError(
                "set MAX_STEPS instead of --max-steps in EXTRA_TRAIN_ARGS: the snapshot identity "
                "<RUN_ID>-v<MAX_STEPS> names the step that the run ends on"
            )
        # The fine-tuned weights are evaluated right after the swap, so by default the platform
        # is asked to drop the KV cache computed under the base model's weights.
        policy_raw = text("CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY", "reset_all")
        policy = PromptCachePolicy.parse(policy_raw)
        if policy is PromptCachePolicy.UNSPECIFIED:
            raise LaunchError(
                "CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY must be preserve or reset_all, "
                f"got {policy_raw!r}"
            )
        timeout_raw = text("CWTRAIN_HOTLOAD_TIMEOUT", "1800")
        try:
            timeout = float(timeout_raw)
        except ValueError:
            timeout = 0.0
        if not 0 < timeout < float("inf"):
            raise LaunchError(f"CWTRAIN_HOTLOAD_TIMEOUT must be seconds, got {timeout_raw!r}")
        kit = Path(text("CWTRAIN_KIT_DIR", "/opt/cwtrain"))
        job = cls(
            hf_model=require(env, "HF_MODEL"),
            hf_revision=text("HF_REVISION", ""),
            model_slug=require(env, "MODEL_SLUG"),
            deployment_id=require(env, "CWTRAIN_HOTLOAD_DEPLOYMENT_ID"),
            run_id=require(env, "CWTRAIN_HOTLOAD_RUN_ID"),
            work=Path(text("WORK", "/work")),
            config=Path(text("AXOLOTL_CONFIG", str(kit / "sft_coreweave.yaml"))),
            gpus=env_int(env, "TRAINER_GPUS", 2),
            max_steps=env_int(env, "MAX_STEPS", 30),
            base_s3_uri=text("BASE_S3_URI", ""),
            wait=env_flag(env, "CWTRAIN_HOTLOAD_WAIT", True),
            timeout=timeout,
            verify=env_flag(env, "CWTRAIN_HOTLOAD_VERIFY", False),
            policy=policy,
            extra_train_args=extra,
        )
        if job.gpus < 1 or job.max_steps < 1:
            raise LaunchError("TRAINER_GPUS and MAX_STEPS must be at least 1")
        validate_identity(job.identity)  # before training, not at the upload
        return job

    @property
    def model_dir(self) -> Path:
        return self.work / "models" / self.model_slug

    @property
    def output_dir(self) -> Path:
        return self.work / "out"

    @property
    def identity(self) -> str:
        """The snapshot identity, ``<run>-vNNNNNN`` as in every recipe, so cleanup finds it."""
        return f"{self.run_id}-v{self.max_steps:06d}"


def api_client(environ: Mapping[str, str]) -> InferenceClient:
    """The Dedicated Inference API client, from the settings that the preflight check reads."""
    raw = environ.get("CWTRAIN_API_HEADERS", "").strip()
    try:
        headers = json.loads(raw) if raw else {}
    except ValueError:
        headers = None
    if not isinstance(headers, dict):
        raise LaunchError("CWTRAIN_API_HEADERS must be a JSON object")
    try:
        return InferenceClient(
            token_from_env(environ),
            base_url=env_text(environ, "COREWEAVE_API_URL", DEFAULT_BASE_URL),
            headers={str(k): str(v) for k, v in headers.items()} or None,
            verify=not env_flag(environ, "CWTRAIN_API_INSECURE", False),
        )
    except ValueError as exc:  # no token and no headers
        raise LaunchError(str(exc)) from exc


def train_command(job: SftJob) -> list[str]:
    """``axolotl train`` with the recipe's config, on the staged base model."""
    return [
        "axolotl", "train", str(job.config),
        "--base-model", str(job.model_dir), "--output-dir", str(job.output_dir),
        "--max-steps", str(job.max_steps),
        *job.extra_train_args,
        # torchrun starts one process per GPU, and FSDP2 shards the model across them
        "--launcher", "torchrun", "--", "--nproc_per_node", str(job.gpus),
    ]  # fmt: skip


def merge_command(job: SftJob) -> list[str]:
    """``axolotl merge-lora``: the adapter in the output directory, merged into ``<out>/merged``."""
    return [
        "axolotl", "merge-lora", str(job.config),
        "--base-model", str(job.model_dir), "--output-dir", str(job.output_dir),
        *job.extra_train_args,  # the adapter settings that the run trained with
        "--lora-model-dir", str(job.output_dir),
    ]  # fmt: skip


def axolotl(args: Sequence[str], *, cwd: Path) -> None:
    """Run an ``axolotl`` command, echoed like :func:`run`, without the API credentials."""
    env = {k: v for k, v in os.environ.items() if k not in API_CREDENTIALS}
    say(f"$ {shlex.join(args)}")
    # relative paths, such as the prepared dataset, land in cwd
    code = subprocess.run(list(args), cwd=cwd, env=env).returncode
    if code:
        raise LaunchError(f"{shlex.join(args)} exited with {code}")


def stage_base(job: SftJob) -> None:
    """The base model: the deployment's own bytes when BASE_S3_URI is set, else HF_MODEL."""
    if (job.model_dir / "config.json").is_file():
        return
    if job.base_s3_uri:
        stage_from_s3(job.base_s3_uri, job.model_dir)
        return
    if not re.fullmatch(r"[0-9a-f]{40}", job.hf_revision):
        raise LaunchError("HF_REVISION must be a full 40-character Hugging Face commit SHA")
    run(["hf", "download", job.hf_model, "--revision", job.hf_revision,
         "--local-dir", job.model_dir, "--exclude", "original/*"])  # fmt: skip


def same_bytes(a: TensorLocation, b: TensorLocation) -> bool:
    """Whether two tensors have the same dtype, shape and bytes, read 64 MiB at a time."""
    if (a.dtype, a.shape, a.nbytes) != (b.dtype, b.shape, b.nbytes):
        return False
    with a.path.open("rb") as fa, b.path.open("rb") as fb:
        fa.seek(a.offset)
        fb.seek(b.offset)
        for start in range(0, a.nbytes, _COMPARE_CHUNK):
            size = min(_COMPARE_CHUNK, a.nbytes - start)
            if fa.read(size) != fb.read(size):
                return False
    return True


def same_weights(result: Path, base: Path) -> bool:
    """Whether every tensor in ``result`` has the base model's bytes, as after a merge that
    applied nothing. The first tensor that differs ends the comparison."""
    got, want = tensor_locations(result), tensor_locations(base)
    return set(got) <= set(want) and all(same_bytes(loc, want[name]) for name, loc in got.items())


def added_tokens(directory: Path) -> set[str]:
    """The ``added_tokens`` of the tokenizer in ``directory``, if it holds a ``tokenizer.json``."""
    path = directory / "tokenizer.json"
    if not path.is_file():
        return set()
    return {token["content"] for token in json.loads(path.read_text()).get("added_tokens") or []}


def served_names(result: Path, base: Path, tokenizer_dir: Path) -> set[str]:
    """The base model's tensor names, once ``result`` is shown to hold the same tensors.

    A hot-load replaces the weights of the model that the deployment serves, so the result must
    have exactly the base model's tensor names, dtypes and shapes. One extra is allowed: a tied
    ``lm_head.weight`` that equals the embedding byte for byte. The FSDP2 final save gathers it as
    a tensor of its own, so it is saved twice; the base holds it once.

    Raises:
        LaunchError: ``result`` holds no safetensors weights, or other tensors than the base, or
            the tokenizer that the run saved in ``tokenizer_dir`` gained tokens.
    """
    got, want = tensor_locations(result), tensor_locations(base)
    if not got:
        raise LaunchError(
            f"{result} holds no safetensors weights; save the final model with FULL_STATE_DICT, "
            "or merge a LoRA adapter first"
        )
    extra = set(got) - set(want)
    if extra == {TIED_HEAD} and EMBEDDING in got and same_bytes(got[TIED_HEAD], got[EMBEDDING]):
        extra = set()  # an FSDP2 gather saved the tied head again; the base holds it once
        say(f"left out {TIED_HEAD}: a copy of {EMBEDDING}, which the base model ties it to")
    problems = [
        *(f"missing {name}" for name in sorted(set(want) - set(got))),
        *(f"extra {name}" for name in sorted(extra)),
        *(
            f"{name} is {got[name].dtype} {list(got[name].shape)}, the base's "
            f"{want[name].dtype} {list(want[name].shape)}"
            for name in sorted(set(got) & set(want))
            if (got[name].dtype, got[name].shape) != (want[name].dtype, want[name].shape)
        ),
    ]
    if problems:
        more = f"; {len(problems) - 5} more" if len(problems) > 5 else ""
        raise LaunchError(
            f"{result} does not match the deployment's base model ({'; '.join(problems[:5])}"
            f"{more}); did the run add tokens, resize the embeddings or change the dtype?"
        )
    added = added_tokens(tokenizer_dir) - added_tokens(base)
    if added:
        raise LaunchError(
            f"the run added the tokens {sorted(added)[:5]}, but the deployment keeps the base "
            "model's tokenizer"
        )
    return set(want)


def _link(src: Path, dest: Path) -> None:
    try:
        os.link(src, dest)
    except OSError:  # another filesystem
        shutil.copy2(src, dest)


def stage(result: Path, base: Path, dest: Path) -> Path:
    """The snapshot's files in ``dest``: the result's weights and the base model's other files.

    A hot-load swaps the weights only. So the snapshot carries the base model's ``config.json``,
    generation config and tokenizer, not the run's copies: those are written by a newer
    transformers, in a layout that older readers misread (``rope_theta``). Files are hard-linked.
    """
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)

    def weights(path: Path) -> bool:
        return path.name.endswith(".safetensors") or path.name == INDEX_FILE

    for path in sorted(result.iterdir()):
        if path.is_file() and weights(path):
            _link(path, dest / path.name)
    for path in sorted(base.iterdir()):
        if path.is_file() and not weights(path):
            _link(path, dest / path.name)
    return dest


def recovery_command(job: SftJob) -> str:
    """The command that hot-loads the uploaded snapshot by hand."""
    cmd = ["cwtrain", "hotload", "trigger", "--deployment", job.deployment_id,
           "--identity", job.identity]  # fmt: skip
    if job.policy is PromptCachePolicy.RESET_ALL:
        cmd.append("--reset-prompt-cache")
    return shlex.join(cmd)


def publish(loader: HotLoader, snapshot: Path, job: SftJob) -> HotLoad:
    """Upload ``snapshot`` as ``job.identity``, then hot-load it once the deployment is READY."""
    # the weights sit on the pod's disk until this returns, so upload before any wait
    info = loader.upload(snapshot, job.identity)
    # HOTLOAD_TIMEOUT bounds what follows the upload, as in the other recipes' hooks: waiting for
    # the deployment, for a hot-load already running, and for this one share one budget
    deadline = time.monotonic() + job.timeout

    def remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    try:
        loader.wait_for_deployment_ready(timeout=remaining())
        hot_load = loader.trigger(
            job.identity, info, prompt_cache_policy=job.policy, timeout=remaining()
        )
        if job.wait:
            hot_load = loader.wait(hot_load.id, timeout=remaining())
            if job.verify:
                loader.verify_served(hot_load.id, job.identity)
    except CWTrainError:
        say(
            f"{job.identity} is in the bucket, but not hot-loaded. Once `cwtrain deployment get "
            f"{job.deployment_id}` shows READY, hot-load it with: {recovery_command(job)}"
        )
        raise
    say(f"hot-load {hot_load.id} of {job.identity}: {hot_load.state.name}")
    return hot_load


def main(environ: Mapping[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    job = SftJob.from_env(env)
    loader = HotLoader(api_client(env), job.deployment_id)
    if loader.completed_hot_load(job.identity):
        raise LaunchError(
            f"{job.identity} is already hot-loaded on {job.deployment_id}; render the trainer Job "
            "again, which picks a new RUN_ID"
        )

    say("== stage the base model")
    stage_base(job)
    job.work.mkdir(parents=True, exist_ok=True)

    say("== train")
    axolotl(train_command(job), cwd=job.work)
    result = job.output_dir
    if (result / "adapter_config.json").is_file():  # hot-load serves full weights only
        say("== merge the LoRA adapter")
        axolotl(merge_command(job), cwd=job.work)
        result = result / "merged"
        if same_weights(result, job.model_dir):
            raise LaunchError(
                f"{result} is the base model: axolotl merge-lora applied no LoRA weight (its "
                "'Applied LoRA to' line says how many tensors it changed). With FSDP2, train "
                'LoRA with --no-fsdp-config.activation-checkpointing (README, "LoRA")'
            )

    say(f"== snapshot {job.identity}")
    names = served_names(result, job.model_dir, tokenizer_dir=job.output_dir)
    staged = stage(result, job.model_dir, job.work / f"{job.identity}.files")
    snapshot = write_full_snapshot(staged, job.work / job.identity, names=names)

    say("== publish")
    publish(loader, snapshot, job)


if __name__ == "__main__":
    # cwtrain logs the upload and the hot-load's progress; httpx would log every status poll
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
