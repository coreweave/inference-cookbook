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
app = marimo.App(
    width="medium",
    app_title="Reinforcement Learning with Hot-Load APIs on CoreWeave Forge",
)


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
    # Reinforcement Learning with Hot-Load APIs on CoreWeave Forge

    **Train in Molab and hot-load each weight update into Dedicated Inference so the next
    rollouts use the latest policy.**

    You may already know the RL loop: sample answers, score them, and update the policy.
    This tutorial adds one new idea: the trainer and the service that samples answers can
    run on different GPUs. A **hot-load** lets that service use updated weights without
    creating a new deployment for every training step.

    We will train Qwen3-0.6B on small arithmetic tasks with GRPO. The goal is to understand
    the weight-update loop, rather than to build a strong math model in a few steps.

    **Your route through the tutorial**

    1. Explore weights and serving versions with two experiments. No credentials or GPU required.
    2. Set up Molab, Dedicated Inference, and CAIOS for your run.
    3. Load the model, sample answers, train, and inspect the loss and reward charts.
    4. Follow one delta from the trainer to the serving replicas. Then clean up.

    Open code with the pencil icon in Molab. Read each visible Python cell, change the line
    marked **Try it**, and use that cell's Run control. Use the forms to enter settings and
    secrets. The Python cells below control when you load weights, upload files, and train.
    On a fresh copy, the cloud action cells start disabled so you can read and experiment first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The three building blocks

    **Dedicated Inference** is a model-serving deployment with GPUs assigned to it.
    You send prompts to its gateway and receive completions. A **gateway** is the front door:
    it checks authentication and routes requests to the deployment. A **replica** is one
    running copy of the model. This tutorial uses one replica to keep the system small.

    **cwtrain** is the Python library that connects training to Dedicated Inference.
    It helps sample exact tokens and their log-probabilities, prepare weight snapshots,
    upload them, and call the hot-load API.

    **The hot-load API** tells an existing deployment to load a weight snapshot already
    stored in object storage. Uploading a snapshot does **not** change the serving model.
    You must trigger the load, wait for it, and check which version the replicas serve.

    In this tutorial, the trainer runs in **Molab**. A small control program runs on your
    **laptop** and makes the management API calls. That program can also run on a devbox.
    The CoreWeave platform key stays with that program.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Where does each part run?"),
            mo.mermaid("""
        flowchart TB
            M["Molab - training GPU<br/>Sample, score, train<br/>Native training charts"]
            L["Laptop - CPU<br/>Create and delete deployment<br/>Trigger, wait, verify"]
            S[("CAIOS / S3<br/>Weights and step messages")]
            F["CoreWeave Forge - inference GPU<br/>Gateway and serving replicas"]
            M <-->|Rollouts: W and B key| F
            M -->|Upload weights| S
            S -->|Read weights| F
            L -->|CoreWeave API key| F
            L <-->|Step messages| S
        """),
            mo.md("""
        **One iteration:** sample from version A → train → upload delta B → hot-load B →
        verify B on every replica → sample the next batch.

        CAIOS means **CoreWeave AI Object Storage**. It provides an S3-compatible API.
        Here it stores both model files and small JSON messages between the two programs.
        W&B records training metrics; the charts inside this notebook are native plots.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it · Rebuild new weights from a delta

    A **snapshot** is a versioned set of weight files. The first snapshot contains full weights.
    Later snapshots can hold a **delta**: the byte changes relative to the previous snapshot.

    cwtrain uses XOR to encode those changes and Zstandard to compress them. It keeps a
    **weight mirror**, a local copy of the serving weights, and checks that each delta rebuilds
    the expected bytes. The real serving weights below use bf16.

    Run this tiny example first. It uses ordinary bytes so you can see the XOR operation.
    **Before running:** what will happen if you change the base used to rebuild the weights?
    Set `demo_wrong_base = demo_base` and run the cell again. Compare both results.
    """)
    return


@app.cell
def _():
    # Try it: change a byte, then run this cell. This does not touch the real model.
    demo_base = bytes([10, 20, 30, 40])
    demo_updated = bytes([10, 21, 30, 42])
    demo_delta = bytes(old ^ new for old, new in zip(demo_base, demo_updated, strict=True))
    demo_rebuilt = bytes(old ^ change for old, change in zip(demo_base, demo_delta, strict=True))
    assert demo_rebuilt == demo_updated
    print("Base:   ", list(demo_base))
    print("Delta:  ", list(demo_delta))
    print("Rebuilt:", list(demo_rebuilt))
    # Try a wrong base. The same delta no longer rebuilds the intended weights.
    demo_wrong_base = bytes([11, 20, 30, 40])
    print(
        "Wrong base matches:",
        bytes(a ^ b for a, b in zip(demo_wrong_base, demo_delta)) == demo_updated,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it · When is it safe to sample again?

    GRPO compares answers from the policy that produced the batch. If the next batch comes
    from old serving weights, it no longer reflects the latest policy.

    A hot-load request can be accepted before all replicas finish loading. The control program
    therefore waits and verifies every replica before it allows the next batch.
    Run the cell below. Then change `replica_versions` so both replicas serve `B`, and run it again.
    These are simulated versions; no API requests are sent.
    """)
    return


@app.cell
def _():
    # Try it: one replica is still loading. Should the next rollout batch start?
    requested_version = "B"
    replica_versions = ["B", "A"]
    safe_to_sample = all(version == requested_version for version in replica_versions)
    print("Replica versions:", replica_versions)
    print("Next action:", "Sample from B" if safe_to_sample else "Wait; a replica still serves A")
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
    ## 1 · Set up your Dedicated Inference run

    You need a training GPU in Molab, capacity for a serving GPU in Dedicated Inference,
    and a CAIOS bucket both programs can access.

    | Credential | Put it where? | What is it for? |
    | --- | --- | --- |
    | CoreWeave platform key | Laptop only | Create/delete resources and control hot-loads |
    | S3 access and secret keys | Molab and laptop | Read/write weights and step messages |
    | W&B API key | Molab | Authenticate gateway rollouts and optionally log metrics |

    A **bucket** holds objects. A **prefix** is the run's folder-like path inside that bucket.
    The bucket zone is its storage location; it can differ from the inference deployment region.
    S3 keys need GetObject, PutObject, and prefix-scoped ListBucket access, plus upload permissions.


    Enable a GPU from Molab's Compute menu. In the Secrets panel, set `AWS_ACCESS_KEY_ID` and
    `AWS_SECRET_ACCESS_KEY`; add `AWS_SESSION_TOKEN` if the credentials include one. Existing
    AWS credential providers are also supported. Credential values are never written into messages.

    Enter a **fresh run ID** such as `forge-demo-001`. Start with a lowercase letter or digit,
    then use lowercase letters, digits, or hyphens. The notebook uses this ID in the laptop
    command below. Use a new ID after a model or kernel restart because the optimizer state
    is lost. Temporary credentials must remain valid for the entire run.
    """)
    return


@app.cell(hide_code=True)
def _(mo, os):
    _env = os.environ.get
    settings = (
        mo.md("""
    {bucket} {zone}

    {run_id}

    {access}

    {secret}

    {session_token}
    """)
        .batch(
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
    PREFIX = f"cwtrain-molab/{cfg['run_id']}"
    WORK = Path.home() / "cwtrain-notebook" / cfg["run_id"] / uuid.uuid4().hex
    WORK.mkdir(parents=True, exist_ok=True)
    trainer_cache = {}

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

    mo.md(f"Run prefix: `s3://{cfg['bucket']}/{PREFIX}/`")
    return (
        PREFIX,
        WORK,
        cfg,
        check_message,
        put_message,
        read_message,
        s3,
        trainer_cache,
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
            self.finished = False
            run.define_metric("step")
            for prefix in ("train", "delta", "timing", "hotload"):
                run.define_metric(f"{prefix}/*", step_metric="step")

        def log_step(self, step, identity, metrics, *, empty):
            if self.finished:
                return
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
            if self.finished:
                return
            self.finished = True
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
    ## W&B logging

    Set `WANDB_API_KEY` in **Molab Secrets** or enter it below. W&B logging runs on this GPU
    notebook. Native charts below show training loss, reward, and gradient norm.
    The laptop returns hot-load timing and served-version status through CAIOS.
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
        wandb_tracker = start_notebook_tracking(wandb_settings.value, WORK, cfg["run_id"])
    return (wandb_tracker,)


@app.cell(hide_code=True)
def _(mo, wandb_tracker):
    if wandb_tracker is None:
        _wb_display = mo.md("W&B logging is optional. Native training charts work without it.")
    elif wandb_tracker.run.url:
        _wb_display = mo.md(f"W&B logging is active. [Open run]({wandb_tracker.run.url}).")
    else:
        _wb_display = mo.md("W&B offline recording is active. Sync the saved run later.")
    _wb_display
    return


@app.cell(hide_code=True)
def _(get_history, mo):
    import altair as alt
    import pandas as pd

    _h = get_history()
    history = (
        pd.DataFrame([{"step": s, **m} for s, m in sorted(_h.items())])
        if _h
        else pd.DataFrame(columns=["step", "loss", "reward"])
    )
    _charts = []
    for _metric, _title in [
        ("loss", "Training loss"),
        ("reward", "Mean reward"),
        ("grad_norm", "Gradient norm"),
        ("logprob_gap", "Trainer versus engine log-probability gap"),
    ]:
        if _metric in history:
            _charts.append(
                alt.Chart(history)
                .mark_line(point=True)
                .encode(
                    x=alt.X("step:Q", title="Training step", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y(f"{_metric}:Q", title=_title, scale=alt.Scale(zero=False)),
                    tooltip=["step:Q", alt.Tooltip(f"{_metric}:Q", title=_title)],
                )
                .properties(title=_title, width=560, height=180)
            )
    _details = []
    if _h:
        _details.append(
            alt.Chart(
                history.melt("step", ["changed_mb", "wire_mb"], var_name="size", value_name="MiB")
            )
            .mark_bar()
            .encode(x="step:O", xOffset="size:N", y="MiB:Q", color="size:N")
            .properties(title="Verified delta size", width=560, height=180)
        )
        _phases = [
            k
            for k in (
                "sample_s",
                "train_s",
                "save_s",
                "delta_s",
                "verify_s",
                "upload_s",
                "hotload_s",
            )
            if k in history
        ]
        if _phases:
            _details.append(
                alt.Chart(history.melt("step", _phases, var_name="stage", value_name="seconds"))
                .mark_bar()
                .encode(x="step:O", y="seconds:Q", color="stage:N")
                .properties(title="Seconds per stage", width=560, height=180)
            )
    mo.vstack(
        [
            mo.md("## Training progress"),
            mo.md(
                f"**{len(_h)} completed steps.** Metrics match those logged to W&B when enabled."
                if _h
                else "Run training steps below to populate these native notebook charts."
            ),
            *_charts,
            *(
                [
                    mo.accordion(
                        {
                            "Weight sync and timing details": mo.vstack(_details),
                            "Step metrics": mo.ui.table(history.round(3), selection=None),
                        }
                    )
                ]
                if _h
                else []
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Meet the trainer and its serving-weight mirror

    The model starts from a fixed Qwen3-0.6B revision so both sides have the same base.
    Training uses fp32 weights and AdamW. The serving copy uses bf16 weights to reduce memory.
    The mirror tracks those bf16 bytes, rather than the optimizer state.

    Read the cell below and find `WeightMirror.create`. Its version starts at `000000`.
    Set `run_id_to_load` to the run ID you entered above, then run the cell.
    The output shows the model name and how many weight tensors are in the mirror.
    Rerunning this cell with the same settings reuses the trainer and its optimizer.
    Start with a fresh run ID if you restart the kernel.
    """)
    return


@app.cell
def _(DEVICE, SimpleNamespace, WORK, cfg, mo, time, torch, trainer_cache, uuid):
    # Try it: enter the same run ID you submitted in the settings form.
    run_id_to_load = ""
    mo.stop(
        run_id_to_load != cfg["run_id"],
        mo.md("Set `run_id_to_load` to your run ID to load the model."),
    )
    from cwtrain.hotload import WeightMirror, inspect_snapshot, upload_snapshot
    from cwtrain.hotload.mirror import tensor_locations
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "Qwen/Qwen3-0.6B"
    REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
    if "run" not in trainer_cache:
        _t = time.monotonic()
        # Each settings submission gets its own work directory and trainer.
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
        trainer_cache["run"] = SimpleNamespace(
            session=uuid.uuid4().hex,
            name=cfg["run_id"],
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
        _load_time = time.monotonic() - _t
        mo.md(f"Loaded {MODEL} in {_load_time:.1f}s.")
    run = trainer_cache["run"]
    mo.md(f"Trainer ready with **{len(run.names)} mirrored tensors**.")
    return MODEL, REVISION, inspect_snapshot, run, upload_snapshot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Give the serving service its starting weights

    A serving deployment must first read a full model. We upload the base model and a full
    snapshot called the **chain root**. Every later delta refers to an exact predecessor.
    The chain is valid only when the trainer and service agree on that starting version.

    Submit your run settings above to generate a command for your bucket, zone, and run ID.
    Copy the command below and run it from this repository on the **laptop**.
    Run `source ~/.profile` separately first if your CoreWeave key is set there.
    The command uses `cwic` for temporary S3 credentials. Change to `--storage-auth aws`
    if you already configured S3 credentials.

    The command waits for `ready.json`, creates a gateway and deployment, and installs the root.
    Resource creation and later hot-load calls stay on the laptop. Molab only sends rollout
    requests to the gateway and exchanges files and messages through S3.


    Read the two calls to `upload_snapshot`. The first puts the base model in S3. The second
    puts the full chain root where the serving deployment can load it.
    Set `publish_weights = True` and run the cell. Look for the ready message. The cell writes
    `ready.json` only after both uploads finish. If an upload fails, run the same cell again.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, mo):
    import shlex

    _command = shlex.join(
        [
            "uv",
            "run",
            "training/examples/hotload_laptop.py",
            "--bucket",
            cfg["bucket"],
            "--zone",
            cfg["zone"].upper(),
            "--run-id",
            cfg["run_id"],
            "--steps",
            "3",
            "--storage-auth",
            "cwic",
        ]
    )
    mo.ui.code_editor(value=_command, language="shell", disabled=True, label="Laptop command")
    return


@app.cell
def _(
    MODEL,
    PREFIX,
    REVISION,
    cfg,
    inspect_snapshot,
    mo,
    put_message,
    read_message,
    run,
    s3,
    upload_snapshot,
):
    # Try it: read the upload order, then change False to True.
    publish_weights = False
    mo.stop(not publish_weights, mo.md("Set `publish_weights = True` to upload the weights."))
    mo.stop(run.bootstrapped, mo.md("The starting weights are already in S3."))
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
def _(DEVICE, asyncio, random, reward, run, shutil, time, torch):
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
                    sampler.sample(
                        task["prompt_ids"], GROUP, max_tokens=MAX_TOKENS, temperature=TEMPERATURE
                    )
                    for task in tasks
                )
            )
        finally:
            await sampler.aclose()
        result = []
        for group in groups:
            if len(group) != GROUP:
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

    return export_delta, make_prompts, sample_remote, train_step


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
    ### Try it · Inspect a real rollout prompt

    The prompt includes token IDs, not just text. Exact token IDs and per-token log-probabilities
    let the trainer compare its policy with the policy that sampled the answers.
    After loading the model, set `prompt_seed = 1` and run the cell. Try seed 2 to get a
    different prompt. This cell builds prompts using the model's tokenizer.
    """)
    return


@app.cell
def _(make_prompts, mo):
    # Try it: set 1 after loading the model. Then try 2 for a different problem.
    prompt_seed = 0
    mo.stop(prompt_seed == 0, mo.md("Load the model, then set `prompt_seed = 1`."))
    example_task = make_prompts(prompt_seed)[0]
    mo.md(
        f"**Prompt:** {example_task['question']}\n\n"
        f"**Target:** {example_task['answer']} · "
        f"**Prompt tokens:** {len(example_task['prompt_ids'])}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Try it · Score a sample answer

    This is the reward function used by the training loop. It reads the last number inside
    `\boxed{}` and compares it with the expected answer. Predict the score, then run the
    cell. Try changing the number or removing the box. This exercise sends no requests.
    """)
    return


@app.cell
def _(re):
    def reward(text, answer):
        found = re.findall("\\\\boxed\\{\\s*([-\\d,\\.]+)\\s*\\}", text)
        try:
            return float(bool(found) and float(found[-1].replace(",", "")) == answer)
        except ValueError:
            return 0.0

    # Try it: change the sample answer and see how the score changes.
    sample_answer = r"The result is \boxed{42}."
    print("Reward:", reward(sample_answer, answer=42))
    return (reward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What does the laptop do after an upload?

    Read this small part of the laptop program. It runs there because these are management
    API calls that need the CoreWeave platform key. It does not run in this notebook.

    ```python
    # Laptop: identity names the uploaded snapshot; previous_identity names its base.
    hot_load = loader.trigger(identity, previous_identity=previous_identity)
    loader.wait(hot_load.id)
    loader.verify_served(hot_load.id, identity)
    # Only now: send the acknowledgment that allows Molab to advance.
    ```

    `loader` is a cwtrain `HotLoader` connected to the deployment. Trigger asks for the load;
    wait checks request progress; verify checks the replicas' actual serving identities.
    The real program also resets the prompt cache for a changed model. Open
    `hotload_laptop.py` to follow resource creation, the step loop, and cleanup.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Read one iteration, then run it

    The implementation follows the same order as an ordinary GRPO loop:

    1. Build eight arithmetic prompts and sample four answers for each prompt.
    2. Score each answer: 1 for a correct boxed number, 0 otherwise.
    3. Normalize rewards within each group and update the model.
    4. Save serving weights, encode a delta, and check it against the mirror.
    5. Upload the delta and wait for the laptop to verify that the replicas serve it.

    If all rewards within every group are equal, all advantages are zero and the optimizer
    update is skipped. An empty delta needs no hot-load. A zero loss alone does not prove
    that training succeeded: check reward and the `trained` field in the metric table too.

    **Predict first:** should each step upload the full model again? Should it sample again
    as soon as the upload ends? Use the two opening experiments to explain your answers.

    Read the cell below. `process_request()` waits for a request from the laptop, runs one
    training step, and returns its metrics after the laptop checks the new serving version.

    **Try it.** Set `target_step = 1` and run the cell. Check the first point in **Training
    progress** and the serving version in the laptop log. Then change the target to 2 and 3.
    The target is the total number of steps to finish. Match the final target to the laptop's
    `--steps` argument. To retry an interrupted upload, run the cell with the same target.
    """)
    return


@app.cell
async def _(mo, os, process_request, run, set_history, wandb_tracker):
    # Try it: set 1 first, then 2 and 3 as you inspect each result.
    target_step = 0
    mo.stop(target_step == 0, mo.md("Set `target_step = 1` to run the first training step."))
    if type(target_step) is not int or not 1 <= target_step <= 10:
        raise ValueError("Set target_step to a whole number from 1 to 10.")
    mo.stop(
        not run.bootstrapped,
        mo.callout("Upload the base model and chain root first.", kind="warn"),
    )
    mo.stop(
        not os.environ.get("WANDB_API_KEY", "").strip(),
        mo.callout("Set WANDB_API_KEY in Molab for gateway access.", kind="warn"),
    )
    # While a delta is pending, its step still needs a serving acknowledgment.
    completed_steps = run.step - int(run.pending is not None)
    for _ in mo.status.progress_bar(
        range(completed_steps, target_step), title="Training", remove_on_exit=True
    ):
        # Sample, train, upload the delta, and wait for the laptop to verify the serving version.
        _metrics = await process_request()
        if _metrics is None:
            break
        set_history(lambda h, s=run.step, m=_metrics: {**h, s: m})
        if wandb_tracker is not None:
            wandb_tracker.log_step(
                run.step,
                run.last_identity,
                _metrics,
                empty=_metrics["tensors_changed"] == 0,
            )
    _completed = run.step - int(run.pending is not None)
    mo.md(f"**Completed {_completed} training steps.** Serving version: `{run.last_identity}`.")
    return


@app.cell(hide_code=True)
def _(get_history, mo):
    _observed = get_history()
    mo.stop(not _observed)
    _last_step = max(_observed)
    _last = _observed[_last_step]
    mo.md(f"""
    ### Pause and explain step {_last_step}

    Your measured loss is **{_last["loss"]:.4f}** and mean reward is **{_last["reward"]:.3f}**.
    The optimizer update was **{"applied" if _last.get("trained") else "skipped"}**.

    - Open **Step metrics** in Training progress. Did the model update when you expected it to?
    - Compare the changed-weight size with the uploaded delta size. Is compression useful here?
    - Compare the served identity in the laptop log with this step.
    - Run one more step. Does reward improve? A few stochastic batches can fluctuate;
      loss and reward do not have to move together on every step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Finish the experiment

    You have now connected a familiar RL trainer to a separate inference service.
    Before leaving, explain the loop in your own words: **upload → trigger → wait → verify**.
    Which step would be missing if you only copied files to S3?

    If you started W&B logging, set `finish_wandb = True` and run the cell below to save the
    metric table and finish the run. Reset the flag before opening another W&B run.
    When you are done, stop the Molab GPU in the Compute menu.


    The laptop driver deletes the deployment and any gateway it created when it exits, including
    on failure. Molab never creates or deletes Dedicated Inference resources. If the laptop process
    was killed, run its `--cleanup` command using the saved local state. Uploaded run data remains
    in CAIOS for inspection. Remove the run prefix when you no longer need those files.
    """)
    return


@app.cell
def _(mo, wandb_tracker):
    finish_wandb = False
    mo.stop(not finish_wandb or wandb_tracker is None)
    wandb_tracker.finish(0)
    mo.md("W&B run finished.")
    return


if __name__ == "__main__":
    app.run()
