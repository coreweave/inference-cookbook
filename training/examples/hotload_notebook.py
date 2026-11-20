# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25",
#     "cwtrain>=1.0.3,<2",
#     "torch>=2.8",
#     "transformers>=4.56,<6",
#     "safetensors>=0.5",
#     "huggingface-hub>=0.34",
#     "altair>=5",
#     "pandas>=2",
#     "wandb>=0.19,<1",
# ]
# ///

"""GPU training worker; Dedicated Inference stays on the laptop.

Run hotload_laptop.py on the laptop and use the same bucket, zone and fresh run ID here.
Molab uses CAIOS/S3 credentials and a W&B key for rollout access and inline visualizations.
Verified snapshots and control acknowledgments travel through the run's S3 prefix.
See hotload-notebook.md for the two-process walkthrough.

    uvx marimo edit --sandbox training/examples/hotload_notebook.py
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="cwtrain CAIOS training worker")


@app.cell(hide_code=True)
def _():
    import asyncio
    import json
    import os
    import random
    import re
    import shutil
    import time
    import uuid
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo

    return (
        Path,
        SimpleNamespace,
        asyncio,
        json,
        mo,
        os,
        random,
        re,
        shutil,
        time,
        uuid,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train in Molab, manage inference on your laptop

    The laptop creates and deletes Dedicated Inference resources and runs **all hot-load control**:
    trigger, status polling, and replica verification. This notebook samples through a
    **W&B-authenticated gateway**, trains Qwen3-0.6B on its GPU, verifies compressed deltas,
    uploads them to **CoreWeave AI Object Storage (CAIOS)**, and displays W&B charts.

    | Laptop | CAIOS handoff | Molab GPU |
    | --- | --- | --- |
    | Create W&B gateway and deployment; install root | Base + full snapshot ← | Load and upload |
    | Authorize next step | Endpoint + predecessor → | Sample with W&B key; train |
    | Trigger hot-load; poll; verify every replica | Verified delta ← | Encode, checksum, upload |
    | Confirm the served identity | Acknowledgment → | Advance; log and display charts |

    The CoreWeave platform key stays on the laptop. Set `WANDB_API_KEY` here for gateway
    rollout access and notebook charts. S3 credentials need **GetObject, PutObject and
    prefix-scoped ListBucket** access under this run's prefix. Public model downloads
    and package installation also need network access.

    Choose **Local** to sample on this GPU too, without S3.
    """)
    return


@app.cell(hide_code=True)
def _(mo, os):
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _gpu = (
        torch.cuda.get_device_properties(0).name
        if DEVICE == "cuda"
        else "none (CPU training is slow)"
    )
    mo.md(f"GPU: **{_gpu}** · CPU cores: {os.cpu_count()}")
    return DEVICE, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Settings

    Enable a GPU from Molab's Compute menu. In the Secrets panel, set `AWS_ACCESS_KEY_ID` and
    `AWS_SECRET_ACCESS_KEY`; add `AWS_SESSION_TOKEN` if the credentials include one. Existing
    AWS credential providers are also supported. Credential values are never written into messages.

    Use the **same fresh run ID** in the laptop command and here. Restarting the model or notebook
    kernel requires a new run ID; optimizer state is not restored from the uploaded deltas.
    Temporary credentials must remain valid for the entire run.
    """)
    return


@app.cell(hide_code=True)
def _(mo, os):
    _env = os.environ.get
    settings = (
        mo.md("""
    {mode}

    {bucket} {zone}

    {run_id}

    {access}

    {secret}

    {session_token}
    """)
        .batch(
            mode=mo.ui.radio(
                ["Laptop-managed Dedicated Inference", "Local"],
                value="Laptop-managed Dedicated Inference",
                label="Mode",
            ),
            bucket=mo.ui.text(
                value=_env("CWTRAIN_BUCKET", "gradient-hotloading"), label="CAIOS bucket"
            ),
            zone=mo.ui.text(value=_env("AWS_REGION", "us-east-02a"), label="Bucket zone"),
            run_id=mo.ui.text(
                value=_env("CWTRAIN_RUN_ID", ""), label="Fresh run ID (same as laptop --run-id)"
            ),
            access=mo.ui.text(
                value=_env("AWS_ACCESS_KEY_ID", ""), kind="password", label="S3 access key ID"
            ),
            secret=mo.ui.text(
                value=_env("AWS_SECRET_ACCESS_KEY", ""),
                kind="password",
                label="S3 secret access key",
            ),
            session_token=mo.ui.text(
                value=_env("AWS_SESSION_TOKEN", ""),
                kind="password",
                label="S3 session token (if issued)",
            ),
        )
        .form(submit_button_label="Use these settings", bordered=True)
    )
    settings
    return (settings,)


@app.cell(hide_code=True)
def _(Path, json, mo, re, settings, uuid):
    mo.stop(settings.value is None, mo.md("_Submit settings to continue._"))
    cfg = dict(settings.value)
    REMOTE = cfg["mode"] != "Local"
    if REMOTE:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", cfg["run_id"]):
            raise ValueError("Set a fresh run ID: 1–63 lowercase letters, digits or hyphens.")
        if not cfg["bucket"] or not cfg["zone"]:
            raise ValueError("Bucket and zone are required.")
        if bool(cfg["access"]) != bool(cfg["secret"]):
            raise ValueError(
                "Supply both S3 access and secret keys, or leave both blank "
                "for the AWS credential chain."
            )
        import boto3
        from botocore.config import Config

        _keys = {}
        if cfg["access"]:
            _keys = dict(aws_access_key_id=cfg["access"], aws_secret_access_key=cfg["secret"])
            if cfg["session_token"]:
                _keys["aws_session_token"] = cfg["session_token"]
        s3 = boto3.Session(**_keys).client(
            "s3",
            endpoint_url="https://cwobject.com",
            region_name=cfg["zone"].upper(),
            config=Config(
                s3={"addressing_style": "virtual"},
                connect_timeout=15,
                read_timeout=120,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
    else:
        s3 = None
    PREFIX = f"cwtrain-molab/{cfg['run_id']}" if REMOTE else ""
    WORK = Path.home() / "cwtrain-notebook" / (cfg["run_id"] or "local") / uuid.uuid4().hex
    WORK.mkdir(parents=True, exist_ok=True)

    def read_message(relative):
        try:
            response = s3.get_object(Bucket=cfg["bucket"], Key=f"{PREFIX}/{relative}")
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        with response["Body"] as body:
            return json.load(body)

    def put_message(relative, session, **fields):
        message = {"schema": 1, "run_id": cfg["run_id"], "session": session, **fields}
        s3.put_object(
            Bucket=cfg["bucket"],
            Key=f"{PREFIX}/{relative}",
            Body=json.dumps(message, allow_nan=False).encode(),
            ContentType="application/json",
        )

    def check_message(message, session, step=None):
        if (message.get("schema"), message.get("run_id"), message.get("session")) != (
            1,
            cfg["run_id"],
            session,
        ):
            raise ValueError("S3 message belongs to a different run/session or protocol version.")
        if step is not None and message.get("step") != step:
            raise ValueError("S3 message has an unexpected training step.")

    mo.md(
        f"Run prefix: `s3://{cfg['bucket']}/{PREFIX}/`"
        if REMOTE
        else "Local mode: sampling and training on this GPU."
    )
    return (
        PREFIX,
        REMOTE,
        WORK,
        cfg,
        check_message,
        put_message,
        read_message,
        s3,
    )


@app.cell(hide_code=True)
def _(os):
    import logging
    import math

    LOG = logging.getLogger("hotload-notebook")

    WANDB_METRICS = {
        "reward": "train/reward",
        "loss": "train/loss",
        "grad_norm": "train/grad_norm",
        "logprob_gap": "train/logprob_gap",
        "trained": "train/updated",
        "tensors_changed": "delta/tensors_changed",
        "tensors_total": "delta/tensors_total",
        "changed_mb": "delta/changed_mib",
        "wire_mb": "delta/wire_mib",
        "verified": "delta/tensors_verified",
        "sample_s": "timing/sample_s",
        "train_s": "timing/train_s",
        "save_s": "timing/save_s",
        "delta_s": "timing/encode_s",
        "verify_s": "timing/verify_s",
        "upload_s": "timing/upload_s",
        "hotload_s": "timing/hotload_s",
        "step_s": "timing/step_s",
    }

    class NotebookTracker:
        """Notebook telemetry. Logging failures cannot interrupt the served-version barrier."""

        def __init__(self, sdk, run):
            (self.sdk, self.run) = (sdk, run)
            self.rows = []
            run.define_metric("step")
            for prefix in ("train", "delta", "timing", "hotload"):
                run.define_metric(f"{prefix}/*", step_metric="step")

        def log_step(self, step, identity, metrics, *, empty):
            try:
                row = {"step": step, "hotload/skipped": int(empty)}
                for source, target in WANDB_METRICS.items():
                    value = metrics.get(source)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        row[target] = int(value) if isinstance(value, bool) else value
                (changed, wire) = (row.get("delta/changed_mib", 0), row.get("delta/wire_mib", 0))
                if wire > 0:
                    row["delta/compression_ratio"] = changed / wire
                self.rows.append(row)
                payload = dict(row)
                xs = [r["step"] for r in self.rows]
                if all(("delta/changed_mib" in r and "delta/wire_mib" in r for r in self.rows)):
                    payload["charts/delta_size"] = self.sdk.plot.line_series(
                        xs=xs,
                        ys=[
                            [r[key] for r in self.rows]
                            for key in ("delta/changed_mib", "delta/wire_mib")
                        ],
                        keys=["Changed bf16 tensors (MiB)", "Compressed delta (MiB)"],
                        title="Delta size before and after compression",
                        xname="Training step",
                    )
                phases = [
                    key
                    for key in ("sample", "train", "save", "encode", "verify", "upload", "hotload")
                    if all((f"timing/{key}_s" in r for r in self.rows))
                ]
                if phases:
                    payload["charts/stage_seconds"] = self.sdk.plot.line_series(
                        xs=xs,
                        ys=[[r[f"timing/{key}_s"] for r in self.rows] for key in phases],
                        keys=phases,
                        title="Seconds per stage",
                        xname="Training step",
                    )
                self.run.log(payload, step=step)
                self.run.summary.update({"completed_steps": step, "last_served_identity": identity})
            except Exception:
                LOG.warning("W&B step logging failed; training continues")

        def finish(self, exit_code):
            try:
                if self.rows:
                    columns = [
                        "step",
                        *WANDB_METRICS.values(),
                        "delta/compression_ratio",
                        "hotload/skipped",
                    ]
                    table = self.sdk.Table(
                        columns=columns,
                        data=[[row.get(key) for key in columns] for row in self.rows],
                    )
                    self.run.log({"tables/verified_steps": table})
            except Exception:
                LOG.warning("W&B metrics table could not be saved")
            finally:
                try:
                    self.run.finish(exit_code=exit_code)
                except Exception:
                    LOG.warning("W&B run finalization failed")

    def start_notebook_tracking(options, work_dir, name):
        import wandb

        key = options["api_key"].strip() or os.environ.get("WANDB_API_KEY", "")
        if options["mode"] == "online" and not key:
            raise ValueError("Set WANDB_API_KEY in Molab Secrets or the masked W&B settings field.")
        if key:
            os.environ["WANDB_API_KEY"] = key
        if wandb.run is not None:
            wandb.run.finish()
        wb_run = wandb.init(
            project=options["project"],
            entity=options["entity"] or None,
            name=name,
            mode=options["mode"],
            dir=str(work_dir),
            job_type="molab-grpo",
            save_code=False,
            settings=wandb.Settings(disable_git=True, disable_code=True, console="off"),
            config={
                "model": "Qwen/Qwen3-0.6B",
                "learning_rate": 1e-5,
                "training_host": "molab",
                "management_host": "laptop",
            },
        )
        try:
            return NotebookTracker(wandb, wb_run)
        except BaseException:
            wb_run.finish(exit_code=1)
            raise

    return (start_notebook_tracking,)


@app.cell(hide_code=True)
def _(mo, os):
    wandb_settings = (
        mo.md("""
    ## W&B visualization

    Set `WANDB_API_KEY` in **Molab Secrets** or enter it below. W&B logging runs on this GPU
    notebook, and its live dashboard is embedded here. The laptop sends hot-load timing and
    served-version status back through CAIOS; it needs no W&B key.
    This key also authenticates rollout sampling through the gateway. Start W&B before
    training, or set the key in Secrets when logging is disabled. Restarting W&B starts a new run.

    {enabled} {mode}

    {project} {entity}

    {api_key}
    """)
        .batch(
            enabled=mo.ui.checkbox(value=True, label="Enable W&B"),
            mode=mo.ui.dropdown(["online", "offline"], value="online", label="W&B mode"),
            project=mo.ui.text(
                value=os.environ.get("WANDB_PROJECT", "cwtrain-molab"), label="W&B project"
            ),
            entity=mo.ui.text(
                value=os.environ.get("WANDB_ENTITY", ""), label="W&B entity (optional)"
            ),
            api_key=mo.ui.text(
                value=os.environ.get("WANDB_API_KEY", ""), kind="password", label="W&B API key"
            ),
        )
        .form(submit_button_label="Start or restart W&B run", bordered=True)
    )
    wandb_settings
    return (wandb_settings,)


@app.cell(hide_code=True)
def _(WORK, cfg, start_notebook_tracking, wandb_settings):
    wandb_tracker = None
    if wandb_settings.value is not None and wandb_settings.value["enabled"]:
        wandb_tracker = start_notebook_tracking(
            wandb_settings.value, WORK, cfg["run_id"] or "molab-local"
        )
    return (wandb_tracker,)


@app.cell(hide_code=True)
def _(mo, wandb_tracker):
    if wandb_tracker is None:
        _wb_display = mo.md(
            "_Start a W&B run above to show its dashboard here. Live charts remain below._"
        )
    elif wandb_tracker.run.url:
        _wb_display = mo.vstack(
            [
                mo.md(f"[Open W&B run in a new tab]({wandb_tracker.run.url})"),
                mo.Html(wandb_tracker.run.to_html(height=720)),
            ]
        )
    else:
        _wb_display = mo.md(
            "**W&B offline recording is active.** Use the live charts below; sync the run later."
        )
    _wb_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Load the model and build its weight mirror

    This keeps fp32 training weights and AdamW on the GPU, plus a byte-exact bf16 mirror on disk.
    The mirror verifies every delta before it can be uploaded.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    load_model = mo.ui.run_button(label="Download Qwen3-0.6B and build the mirror")
    load_model
    return (load_model,)


@app.cell(hide_code=True)
def _(DEVICE, SimpleNamespace, WORK, cfg, load_model, mo, time, torch, uuid):
    mo.stop(not load_model.value)
    from cwtrain.hotload import WeightMirror, inspect_snapshot, upload_snapshot
    from cwtrain.hotload.mirror import tensor_locations
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "Qwen/Qwen3-0.6B"
    REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
    _t = time.monotonic()
    # WORK is unique to this model initialization; do not reuse an optimizer from another session.
    _base = snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=WORK / "base",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"],
    )
    from pathlib import Path as _Path

    _base = _Path(_base)
    _tok = AutoTokenizer.from_pretrained(_base)
    _model = AutoModelForCausalLM.from_pretrained(_base, dtype=torch.float32).to(DEVICE)
    _model.gradient_checkpointing_enable()
    _model.config.use_cache = False
    _mirror = WeightMirror.create(_base, WORK / "mirror", version="000000")
    run = SimpleNamespace(
        session=uuid.uuid4().hex,
        name=cfg["run_id"] or "local",
        work=WORK,
        base=_base,
        tok=_tok,
        model=_model,
        opt=torch.optim.AdamW(_model.parameters(), lr=1e-5, weight_decay=0.0),
        mirror=_mirror,
        names=sorted(tensor_locations(_base)),
        step=0,
        pending=None,
        failed=False,
        bootstrapped=False,
        last_identity=f"{cfg['run_id']}-v000000",
    )
    mo.md(f"Loaded {MODEL} and mirrored {len(run.names)} tensors in {time.monotonic() - _t:.1f}s.")
    return MODEL, REVISION, inspect_snapshot, run, upload_snapshot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Upload the base model and chain root

    This writes the base checkpoint and a checksummed full snapshot, then writes `ready.json`
    **last**. The laptop waits for that marker before creating the deployment. Upload retries are
    safe. Once ready, start the laptop driver and click **Serve laptop requests** below.
    """)
    return


@app.cell(hide_code=True)
def _(REMOTE, mo):
    mo.stop(not REMOTE)
    bootstrap = mo.ui.run_button(label="Upload base model and chain root")
    bootstrap
    return (bootstrap,)


@app.cell(hide_code=True)
def _(
    MODEL,
    PREFIX,
    REMOTE,
    REVISION,
    bootstrap,
    cfg,
    inspect_snapshot,
    mo,
    put_message,
    read_message,
    run,
    s3,
    upload_snapshot,
):
    mo.stop(not REMOTE)
    mo.stop(not bootstrap.value)
    _existing = read_message("ready.json")
    if _existing is not None and _existing.get("session") != run.session:
        raise ValueError(
            "This run ID already has a trainer. Use a fresh run ID after any model/kernel restart."
        )
    upload_snapshot(
        inspect_snapshot(run.base),
        bucket=cfg["bucket"],
        path_prefix=f"{PREFIX}/models",
        identity="base",
        s3=s3,
    )
    _root = run.work / "root"
    if not (_root / "model.safetensors.index.json").exists():
        run.mirror.write_full_snapshot(_root)
    upload_snapshot(
        inspect_snapshot(_root),
        bucket=cfg["bucket"],
        path_prefix=f"{PREFIX}/hotload",
        identity=run.last_identity,
        s3=s3,
    )
    put_message(
        "ready.json", run.session, model=MODEL, revision=REVISION, root_identity=run.last_identity
    )
    run.bootstrapped = True
    mo.md("**Ready for the laptop driver.** The base model and chain root are uploaded.")
    return


@app.cell(hide_code=True)
def _(DEVICE, asyncio, random, re, run, shutil, time, torch):
    PROMPTS, GROUP, MAX_TOKENS, TEMPERATURE = 8, 4, 384, 1.0

    async def sample_remote(tasks, *, base_url, model, token):
        from cwtrain.integrations.rollout import CompletionSampler, RolloutConfig

        sampler = CompletionSampler(
            RolloutConfig(
                base_url=base_url,
                model=model,
                api_key=token,
                concurrency=8,
                require_logprobs=True,
            )
        )
        try:
            groups = await asyncio.gather(
                *(
                    sampler.sample(t["prompt_ids"], 4, max_tokens=384, temperature=1.0)
                    for t in tasks
                )
            )
        finally:
            await sampler.aclose()
        result = []
        for group in groups:
            if len(group) != 4:
                raise ValueError("deployment did not return four completions per prompt")
            row = []
            for completion in group:
                if (
                    not completion.token_ids
                    or not completion.token_logprobs
                    or len(completion.token_ids) != len(completion.token_logprobs)
                ):
                    raise ValueError("deployment must return exact token IDs and aligned log-probs")
                row.append(
                    dict(
                        ids=list(completion.token_ids),
                        logprobs=list(completion.token_logprobs),
                        text=completion.text,
                        cached=completion.cached_tokens,
                    )
                )
            result.append(row)
        return result

    def make_prompts(step):
        rng = random.Random(step)
        tasks = []
        for _ in range(PROMPTS):
            (a, b, c) = (rng.randint(101, 999), rng.randint(13, 99), rng.randint(100, 999))
            question = (
                f"What is {a} * {b} + {c}? Think briefly, then put only "
                "the final number in \\boxed{}."
            )
            ids = run.tok.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=False,
            )
            tasks.append({"question": question, "answer": a * b + c, "prompt_ids": list(ids)})
        return tasks

    def reward(text, answer):
        found = re.findall("\\\\boxed\\{\\s*([-\\d,\\.]+)\\s*\\}", text)
        try:
            return float(bool(found) and float(found[-1].replace(",", "")) == answer)
        except ValueError:
            return 0.0

    @torch.no_grad()
    def sample_local(tasks):
        """The same groups, sampled with this GPU: what the deployment does in remote mode."""
        (tok, model) = (run.tok, run.model)
        model.eval()
        width = max((len(t["prompt_ids"]) for t in tasks))
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.tensor(
            [[pad] * (width - len(t["prompt_ids"])) + t["prompt_ids"] for t in tasks], device=DEVICE
        )
        mask = (
            torch.arange(width, device=DEVICE)[None]
            >= (width - torch.tensor([len(t["prompt_ids"]) for t in tasks], device=DEVICE))[:, None]
        ).long()
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            out = model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=1.0,
                top_k=0,
                max_new_tokens=MAX_TOKENS,
                num_return_sequences=GROUP,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=pad,
                use_cache=True,
            )
        model.train()
        new = out.sequences[:, width:]
        logp = torch.stack(out.logits, 1).float().log_softmax(-1).gather(-1, new[..., None])[..., 0]
        eos = model.generation_config.eos_token_id
        stops = {tok.eos_token_id, *(eos if isinstance(eos, list) else [eos])}
        groups = []
        for i in range(len(tasks)):
            group = []
            for j in range(GROUP):
                row = new[i * GROUP + j].tolist()
                end = next((k + 1 for (k, t) in enumerate(row) if t in stops), len(row))
                group.append(
                    {
                        "ids": row[:end],
                        "logprobs": logp[i * GROUP + j, :end].tolist(),
                        "text": tok.decode(row[:end], skip_special_tokens=True),
                        "cached": None,
                    }
                )
            groups.append(group)
        return groups

    def train_step(batch):
        """GRPO with group-normalized advantages. Returns the step's metrics."""
        (tok, model, opt) = (run.tok, run.model, run.opt)
        samples = []
        for task, group in zip(batch["tasks"], batch["groups"], strict=True):
            rewards = torch.tensor([reward(c["text"], task["answer"]) for c in group])
            adv = (rewards - rewards.mean()) / (rewards.std() + 0.0001)
            samples += [
                (task["prompt_ids"], c, float(r), float(a))
                for (c, r, a) in zip(group, rewards, adv, strict=True)
            ]
        mean_reward = sum((s[2] for s in samples)) / len(samples)
        if all((a == 0 for (*_, a) in samples)):
            return {"reward": mean_reward, "loss": 0.0, "logprob_gap": None, "trained": False}
        tokens = sum((len(c["ids"]) for (_, c, _, _) in samples))
        (loss_total, gap_sum, gap_n) = (0.0, 0.0, 0)
        opt.zero_grad(set_to_none=True)
        for k in range(0, len(samples), 8):
            chunk = samples[k : k + 8]
            seqs = [p + c["ids"] for (p, c, _, _) in chunk]
            width = max(map(len, seqs))
            ids = torch.tensor(
                [s + [tok.eos_token_id] * (width - len(s)) for s in seqs], device=DEVICE
            )
            attn = torch.tensor(
                [[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=DEVICE
            )
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]
            logp = logits.float().log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0]
            loss = 0.0
            for row, (p, c, _, a) in enumerate(chunk):
                span = slice(len(p) - 1, len(p) - 1 + len(c["ids"]))
                loss = loss - a * logp[row, span].sum()
                if c["logprobs"]:
                    engine = torch.tensor(c["logprobs"], device=DEVICE)
                    gap_sum += (logp[row, span].detach() - engine).abs().sum().item()
                    gap_n += len(c["logprobs"])
            (loss / tokens).backward()
            loss_total += loss.item() / tokens
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return {
            "reward": mean_reward,
            "loss": loss_total,
            "grad_norm": float(grad_norm),
            "logprob_gap": gap_sum / gap_n if gap_n else None,
            "trained": True,
        }

    def export_delta(version):
        """Save bf16 weights, encode a delta and verify it against the mirror."""
        from cwtrain.hotload.delta import write_delta
        from safetensors.torch import save_file

        timings = {}
        t = time.monotonic()
        new_dir = run.work / "ckpt"
        shutil.rmtree(new_dir, ignore_errors=True)
        new_dir.mkdir(parents=True)
        state = run.model.state_dict()
        save_file(
            {n: state[n].detach().to(torch.bfloat16).contiguous().cpu() for n in run.names},
            new_dir / "model.safetensors",
            metadata={"format": "pt"},
        )
        shutil.copy(run.base / "config.json", new_dir / "config.json")
        timings["save_s"] = time.monotonic() - t
        t = time.monotonic()
        delta_dir = run.work / "deltas" / f"v{version:06d}"
        shutil.rmtree(delta_dir, ignore_errors=True)
        stats = write_delta(
            run.mirror.root,
            new_dir,
            delta_dir,
            version=f"{version:06d}",
            base_version=f"{version - 1:06d}",
        )
        timings["delta_s"] = time.monotonic() - t
        t = time.monotonic()
        if stats.is_empty:
            run.mirror.advance(f"{version:06d}")
            verified = 0
        else:
            verified = run.mirror.apply_delta(delta_dir).tensors_changed
        timings["verify_s"] = time.monotonic() - t
        shutil.rmtree(new_dir)
        return (
            delta_dir,
            stats,
            {
                "tensors_changed": stats.tensors_changed,
                "tensors_total": stats.tensors_total,
                "changed_mb": stats.bytes_changed / 2**20,
                "wire_mb": stats.wire_bytes / 2**20,
                "verified": verified,
                **timings,
            },
        )

    return export_delta, make_prompts, sample_local, sample_remote, train_step


@app.cell(hide_code=True)
def _(mo):
    get_history, set_history = mo.state({})
    return get_history, set_history


@app.cell(hide_code=True)
def _(
    PREFIX,
    cfg,
    check_message,
    export_delta,
    inspect_snapshot,
    make_prompts,
    os,
    put_message,
    read_message,
    run,
    s3,
    sample_remote,
    time,
    train_step,
    upload_snapshot,
):
    def wait_message(relative, step, timeout=1800):
        deadline = time.monotonic() + timeout
        while True:
            message = read_message(relative)
            if message is not None:
                check_message(message, run.session, step)
                return message
            error = read_message("control/error.json")
            if error is not None:
                check_message(error, run.session)
                raise RuntimeError("Laptop driver failed. Check its terminal before retrying.")
            done = read_message("control/done.json")
            if done is not None:
                check_message(done, run.session)
                if done["step"] <= run.step:
                    return None
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Waiting for laptop S3 message timed out; "
                    "retry while this kernel is still alive."
                )
            time.sleep(2)

    async def process_request():
        if run.failed:
            raise RuntimeError(
                "A previous training/export failed. Restart the model with a fresh run ID."
            )
        if run.pending is None:
            step = run.step + 1
            request = wait_message(f"requests/{step:06d}.json", step)
            if request is None:
                return None
            if request.get("previous_identity") != run.last_identity:
                raise ValueError("Request refers to a different served snapshot.")
            from urllib.parse import urlsplit

            endpoint = request.get("rollout_base_url", "")
            parsed = urlsplit(endpoint)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or parsed.path.rstrip("/") != "/v1"
                or request.get("model") != run.name
            ):
                raise ValueError("Invalid rollout endpoint or model in laptop request.")
            token = os.environ.get("WANDB_API_KEY", "").strip()
            if not token:
                raise ValueError("Set WANDB_API_KEY in Molab for gateway rollout access.")
            started = time.monotonic()
            tasks = make_prompts(step)
            groups = await sample_remote(tasks, base_url=endpoint, model=run.name, token=token)
            batch = dict(tasks=tasks, groups=groups, sample_s=time.monotonic() - started)
            if (
                len(batch.get("tasks", [])) != 8
                or any(len(g) != 4 for g in batch.get("groups", []))
                or len(batch.get("groups", [])) != 8
            ):
                raise ValueError("Expected 8 prompts with 4 completions each.")
            for task, group in zip(batch["tasks"], batch["groups"], strict=True):
                if not task["prompt_ids"] or len(task["prompt_ids"]) > 4096:
                    raise ValueError("Invalid prompt length.")
                for completion in group:
                    if (
                        not completion["ids"]
                        or len(completion["ids"]) > 384
                        or len(completion["logprobs"]) != len(completion["ids"])
                    ):
                        raise ValueError("Completion tokens and log-probs must align.")
            try:
                started = time.monotonic()
                metrics = train_step(batch)
                metrics.update(train_s=time.monotonic() - started, sample_s=batch["sample_s"])
                delta, stats, delta_metrics = export_delta(step)
                metrics.update(delta_metrics)
            except Exception as exc:
                run.failed = True
                put_message(
                    "control/trainer-error.json", run.session, step=step, error=type(exc).__name__
                )
                raise
            run.step = step
            # Retain this transaction before any upload. A transient S3 failure must not retrain.
            run.pending = dict(
                step=step,
                delta=delta,
                empty=stats.is_empty,
                metrics=metrics,
                identity=run.last_identity if stats.is_empty else f"{run.name}-v{step:06d}",
                previous_identity=run.last_identity,
                upload_s=0.0,
            )
        pending = run.pending
        if not pending["empty"]:
            info = inspect_snapshot(
                pending["delta"], previous_identity=pending["previous_identity"]
            )
            upload_started = time.monotonic()
            try:
                upload_snapshot(
                    info,
                    bucket=cfg["bucket"],
                    path_prefix=f"{PREFIX}/hotload",
                    identity=pending["identity"],
                    s3=s3,
                )
            finally:
                pending["upload_s"] += time.monotonic() - upload_started
        pending["metrics"]["upload_s"] = pending["upload_s"]
        put_message(
            f"results/{pending['step']:06d}.json",
            run.session,
            **{
                k: pending[k] for k in ("step", "empty", "metrics", "identity", "previous_identity")
            },
        )
        ack = wait_message(f"acks/{pending['step']:06d}.json", pending["step"])
        if ack is None or ack.get("identity") != pending["identity"]:
            raise ValueError("Laptop did not acknowledge the expected served identity.")
        run.last_identity = pending["identity"]
        metrics = pending["metrics"]
        for key in ("hotload_s", "step_s"):
            value = ack.get("metrics", {}).get(key)
            if isinstance(value, (int, float)) and 0 <= value < 86400:
                metrics[key] = value
        run.pending = None
        return metrics

    return (process_request,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Run training steps

    In laptop-managed mode, this waits for permission in CAIOS, samples with W&B gateway auth,
    trains, uploads each verified delta,
    and waits for the laptop's served-version acknowledgment. Start the laptop driver with the same
    step count. A retry after an upload timeout reuses the pending delta without training twice.

    In Local mode, the same button samples and trains here, then verifies deltas without uploading.
    """)
    return


@app.cell(hide_code=True)
def _(REMOTE, mo):
    steps = mo.ui.slider(1, 10, value=3, label="Maximum steps this click", show_value=True)
    run_steps = mo.ui.run_button(label="Serve laptop requests" if REMOTE else "Run local steps")
    mo.hstack([steps, run_steps])
    return run_steps, steps


@app.cell(hide_code=True)
async def _(
    REMOTE,
    export_delta,
    make_prompts,
    mo,
    process_request,
    run,
    run_steps,
    sample_local,
    set_history,
    steps,
    time,
    train_step,
    wandb_tracker,
):
    mo.stop(not run_steps.value)
    mo.stop(
        REMOTE and not run.bootstrapped,
        mo.callout("Upload the base model and chain root first.", kind="warn"),
    )
    for _ in mo.status.progress_bar(range(steps.value), title="Training", remove_on_exit=True):
        _step_started = time.monotonic()
        if REMOTE:
            _metrics = await process_request()
            if _metrics is None:
                break
        else:
            _tasks = make_prompts(run.step + 1)
            _t = time.monotonic()
            _groups = sample_local(_tasks)
            _sample_s = time.monotonic() - _t
            _t = time.monotonic()
            _metrics = train_step({"tasks": _tasks, "groups": _groups})
            _metrics.update(sample_s=_sample_s, train_s=time.monotonic() - _t)
            _delta, _stats, _delta_metrics = export_delta(run.step + 1)
            run.step += 1
            _metrics.update(_delta_metrics)
            _metrics["step_s"] = time.monotonic() - _step_started
        set_history(lambda h, s=run.step, m=_metrics: {**h, s: m})
        if wandb_tracker is not None:
            wandb_tracker.log_step(
                run.step,
                run.last_identity if REMOTE else f"local-v{run.step:06d}",
                _metrics,
                empty=_metrics["tensors_changed"] == 0,
            )
    mo.md(f"Done: now at step {run.step}.")
    return


@app.cell(hide_code=True)
def _(get_history, mo):
    import altair as alt
    import pandas as pd

    _h = get_history()
    mo.stop(not _h, mo.md("_Training metrics appear here._"))
    history = pd.DataFrame([{"step": s, **m} for s, m in sorted(_h.items())])
    _charts = [
        alt.Chart(history)
        .mark_line(point=True)
        .encode(x="step:O", y=alt.Y("reward:Q", scale=alt.Scale(domain=[0, 1])))
        .properties(title="Reward"),
        alt.Chart(
            history.melt("step", ["changed_mb", "wire_mb"], var_name="size", value_name="MiB")
        )
        .mark_bar()
        .encode(x="step:O", xOffset="size:N", y="MiB:Q", color="size:N")
        .properties(title="Verified delta size"),
    ]
    for _metric, _title in [
        ("loss", "Training loss"),
        ("grad_norm", "Gradient norm"),
        ("logprob_gap", "Trainer versus engine log-probability gap"),
    ]:
        if _metric in history:
            _charts.append(
                alt.Chart(history)
                .mark_line(point=True)
                .encode(x="step:O", y=alt.Y(f"{_metric}:Q", title=_title))
                .properties(title=_title)
            )
    _phases = [
        k
        for k in ("sample_s", "train_s", "save_s", "delta_s", "verify_s", "upload_s", "hotload_s")
        if k in history
    ]
    if _phases:
        _charts.append(
            alt.Chart(history.melt("step", _phases, var_name="stage", value_name="seconds"))
            .mark_bar()
            .encode(x="step:O", y="seconds:Q", color="stage:N")
            .properties(title="Seconds per stage")
        )
    mo.vstack(
        [
            mo.md("### Live metrics — also logged to W&B when enabled"),
            *_charts,
            mo.ui.table(history.round(3), selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    finish_tracking = mo.ui.run_button(label="Finish W&B run")
    finish_tracking
    return (finish_tracking,)


@app.cell(hide_code=True)
def _(finish_tracking, mo, wandb_tracker):
    mo.stop(not finish_tracking.value or wandb_tracker is None)
    wandb_tracker.finish(0)
    mo.md("W&B run finished. Start a new W&B run before logging more training steps.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Cleanup

    The laptop driver deletes the deployment and any gateway it created when it exits, including
    on failure. Molab never creates or deletes Dedicated Inference resources. If the laptop process
    was killed, run its `--cleanup` command using the saved local state. Uploaded run data remains
    in CAIOS for inspection; remove that run prefix when you no longer need it. Release Molab's GPU
    from the Compute menu when finished.
    """)
    return


if __name__ == "__main__":
    app.run()
