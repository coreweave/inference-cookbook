# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25",
#     "cwtrain>=1.0.2",
#     "torch>=2.8",
#     "transformers>=4.56",
#     "safetensors>=0.5",
#     "huggingface-hub>=0.34",
#     "altair>=5",
#     "pandas>=2",
# ]
# ///

"""Train against a Dedicated Inference deployment from one GPU, with cwtrain.

Sized for a molab notebook (https://molab.marimo.io/notebooks): 4 CPU cores, 32 GiB of memory and
one RTX PRO 6000. Qwen3-0.6B trains on the notebook's GPU; its rollouts come from a hot-load
deployment, and every step's new weights go back to it as a verified, compressed delta.

    uvx marimo edit --sandbox training/examples/hotload_notebook.py

Without a CoreWeave account, pick "Local" mode: the notebook samples on its own GPU instead and
still writes, verifies and measures every delta, but publishes nothing.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="cwtrain hot-load notebook")


@app.cell(hide_code=True)
def _():
    import asyncio
    import json
    import os
    import random
    import re
    import secrets
    import shutil
    import time
    from datetime import UTC, datetime
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo

    return (
        Path,
        SimpleNamespace,
        UTC,
        asyncio,
        datetime,
        json,
        mo,
        os,
        random,
        re,
        secrets,
        shutil,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train against a hot-load deployment with `cwtrain`

    An RL trainer that uses [CoreWeave Dedicated Inference](https://docs.coreweave.com/products/inference/rl-rollouts)
    runs no inference engine of its own. Rollouts come from a deployment's OpenAI-compatible
    endpoint, and after every training step the trainer **hot-loads** its new weights into that
    deployment. This notebook runs that loop on one GPU with GRPO on Qwen3-0.6B and shows what
    `cwtrain` does at each step:

    | Step | `cwtrain` | What you see |
    | --- | --- | --- |
    | Mirror the served weights | `WeightMirror` | a byte-exact copy of what the deployment serves, checksummed per tensor |
    | Stage and deploy | `upload_snapshot`, `InferenceClient` | the base model in your bucket, a deployment with `hotLoad` |
    | Publish the chain root | `HotLoader.publish` | a full snapshot, the root every delta chains from |
    | Sample | `CompletionSampler` | exact token ids and log-probs from the endpoint |
    | Train | (your loop, here 40 lines of torch) | reward, and how far the trainer's log-probs are from the engine's |
    | Write the delta | `write_delta`, `WeightMirror.apply_delta` | only the changed tensors, XOR-encoded and zstd-compressed, verified before upload |
    | Hot-load it | `HotLoader.publish`, `get_hot_load` | each replica switching to the new version |

    Each button does one of these, and most take seconds. Creating the deployment is the slow
    one: it takes several minutes, and this notebook polls it instead of blocking.
    """)
    return


@app.cell(hide_code=True)
def _(mo, os):
    import torch

    if torch.cuda.is_available():
        _p = torch.cuda.get_device_properties(0)
        _gpu = f"**{_p.name}**, {_p.total_memory / 2**30:.0f} GiB"
        DEVICE = "cuda"
    else:
        _gpu = "**none**: training runs on the CPU, slowly"
        DEVICE = "cpu"
    mo.md(f"GPU: {_gpu} · CPU cores: {os.cpu_count()}")
    return DEVICE, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Settings

    **Dedicated Inference** needs one secret, `COREWEAVE_API_TOKEN`: an API token with the
    Inference Admin role in an organization where hot-load is enabled. The same token reaches
    the AI Object Storage bucket. `cwic auth accesskey api-token` exchanges it for temporary
    storage keys, which boto3 fetches through `credential_process`, so no long-lived key is
    involved. The bucket must let the inference service read it (see the
    [getting-started guide](https://docs.coreweave.com/products/inference/getting-started)).
    In molab, set the token as a secret so it stays out of the notebook. The other values
    default to `CWTRAIN_BUCKET`, `AWS_REGION` (the bucket's zone), `CWTRAIN_REGION`,
    `CWTRAIN_GATEWAY_ID`, `CWTRAIN_INSTANCE_TYPE` and `CWTRAIN_RUNTIME_VERSION` when they are set.

    Without a gateway id, the notebook uses a ready gateway in the deployment's region, or
    creates one there.

    **Local** needs nothing: the notebook samples with its own GPU.
    """)
    return


@app.cell(hide_code=True)
def _(mo, os):
    _env = os.environ.get
    settings = (
        mo.md(
            """
            {mode}

            {token}

            {bucket} {zone}

            {region} {gateway}

            {instance_type} {runtime}

            {deployment_id}
            """
        )
        .batch(
            mode=mo.ui.radio(
                ["Dedicated Inference", "Local"],
                value="Dedicated Inference" if _env("COREWEAVE_API_TOKEN") else "Local",
                label="Rollouts from",
                inline=True,
            ),
            token=mo.ui.text(
                value=_env("COREWEAVE_API_TOKEN", _env("CW_API_TOKEN", "")),
                kind="password",
                label="API token (also used for the bucket)",
                full_width=True,
            ),
            bucket=mo.ui.text(value=_env("CWTRAIN_BUCKET", "gradient-hotloading"), label="Bucket"),
            zone=mo.ui.text(value=_env("AWS_REGION", "us-east-02a"), label="Bucket zone"),
            region=mo.ui.text(
                value=_env("CWTRAIN_REGION", "us-west-04a"), label="Deployment region"
            ),
            gateway=mo.ui.text(value=_env("CWTRAIN_GATEWAY_ID", ""), label="Gateway id (optional)"),
            instance_type=mo.ui.text(
                value=_env("CWTRAIN_INSTANCE_TYPE", "h100-2x"), label="Deployment instance type"
            ),
            runtime=mo.ui.text(
                value=_env("CWTRAIN_RUNTIME_VERSION", "0.32.0+crwv.5"), label="dynamo-vllm version"
            ),
            deployment_id=mo.ui.text(
                value=_env("CWTRAIN_DEPLOYMENT_ID", ""),
                label="Existing deployment id (optional: reuse one that serves Qwen3-0.6B with hot-load)",
                full_width=True,
            ),
        )
        .form(submit_button_label="Use these settings", bordered=True)
    )
    settings
    return (settings,)


@app.cell(hide_code=True)
def _(os, shutil):
    import hashlib
    import platform
    import tarfile

    # cwic exchanges COREWEAVE_API_TOKEN for temporary AI Object Storage keys. The version and
    # checksums are those of the release that the hot-load E2E workflow pins.
    CWIC_VERSION = "v1.44.0"
    CWIC_SHA256 = {
        "Linux_x86_64": "a6d41b550df3b2e4a223c775e574519d5d0f25855f26d036af8c2f629949131d",
        "Linux_arm64": "ce12a1e09560b392048f3b307de50fcf5f099e9299f410ea480b99d536d25586",
        "Darwin_x86_64": "a48bc1f1faa20b5ac06f7780c4d8fcb2d466db54fe45dff6f51b42c49fb073ce",
        "Darwin_arm64": "436251143b063bde2f680f8eb004fc2f610603fe84b6c905459159275e3c311d",
    }

    def cwic_binary(work):
        """cwic from PATH, or the pinned release, downloaded once and checked against its hash."""
        found = shutil.which("cwic")
        if found:
            return found
        target = work / "bin" / "cwic"
        if target.exists():
            return str(target)
        machine = platform.machine().lower()
        machine = {"aarch64": "arm64", "amd64": "x86_64"}.get(machine, machine)
        asset = f"{platform.system()}_{machine}"
        if asset not in CWIC_SHA256:
            raise RuntimeError(f"no cwic release for {asset}; install cwic and put it on PATH")
        import io

        import httpx

        url = f"https://github.com/coreweave/cwic/releases/download/{CWIC_VERSION}/cwic_{asset}.tar.gz"
        data = httpx.get(url, follow_redirects=True, timeout=120).raise_for_status().content
        if hashlib.sha256(data).hexdigest() != CWIC_SHA256[asset]:
            raise RuntimeError(f"{url} does not match its pinned SHA-256")
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            binary = tar.extractfile("cwic").read()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(binary)
        target.chmod(0o755)
        return str(target)

    def storage_from_api_token(work, zone):
        """Point boto3, and so cwtrain, at the bucket with keys that cwic derives from the token."""
        config = work / "aws-config"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            f"[default]\ncredential_process = {cwic_binary(work)} auth accesskey api-token\n"
        )
        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
        ):
            os.environ.pop(name, None)  # boto3 would take these over credential_process
        os.environ.update(
            AWS_CONFIG_FILE=str(config),
            AWS_REGION=zone.upper(),
            CW_S3_ENDPOINT="https://cwobject.com",
            CWIC_CACHE_DIR=str(work / "cwic-cache"),
            AWS_EC2_METADATA_DISABLED="true",
        )

    return (storage_from_api_token,)


@app.cell(hide_code=True)
def _(Path, SimpleNamespace, mo, os, settings, storage_from_api_token):
    from cwtrain.infra import lifecycle
    from cwtrain.infra.storage import s3_client

    mo.stop(settings.value is None, mo.md("_Submit the settings to continue._"))
    cfg = dict(settings.value)
    REMOTE = cfg["mode"] == "Dedicated Inference"
    WORK = Path.home() / "cwtrain-notebook"
    infra = SimpleNamespace(gateway_id=cfg["gateway"] or None, created_gateway=False)
    if REMOTE:
        _missing = [k for k in ("token", "bucket", "zone", "region") if not cfg[k]]
        mo.stop(_missing, mo.callout(f"Fill in: {', '.join(_missing)}", kind="warn"))
        os.environ["COREWEAVE_API_TOKEN"] = cfg["token"]
        from cwtrain import InferenceClient

        client = InferenceClient()
        storage_from_api_token(WORK, cfg["zone"])
        s3_client().head_bucket(Bucket=cfg["bucket"])  # the token's storage keys reach the bucket

        _zone = cfg["region"].upper()
        if infra.gateway_id:
            _gw = client.get_gateway(infra.gateway_id)
        else:
            _gw = next(
                (
                    g
                    for g in client.list_gateways()
                    if g.status.name == "READY" and _zone in (z.upper() for z in g.zones)
                ),
                None,
            )
            infra.gateway_id = _gw.id if _gw else None
        mo.output.replace(
            mo.callout(
                mo.md(
                    f"Connected. Bucket **{cfg['bucket']}** ({cfg['zone'].upper()}) is reachable "
                    "with storage keys from the API token. "
                    + (
                        f"Gateway **{_gw.name}** ({_gw.status.name}) in {', '.join(_gw.zones)}."
                        if _gw
                        else f"No ready gateway in {_zone}: **Create the deployment** creates one."
                    )
                ),
                kind="success",
            )
        )
    else:
        client = None
        mo.output.replace(
            mo.callout("Local mode: rollouts on this GPU; nothing is published.", kind="info")
        )
    return REMOTE, WORK, cfg, client, infra, lifecycle, s3_client


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The model and the weight mirror

    The trainer keeps fp32 master weights and an AdamW optimizer on the GPU. What it publishes is
    bf16, the dtype the deployment serves. `WeightMirror` keeps a byte-exact copy of the
    **served** weights with an Adler-32 checksum per tensor. Every delta is computed against
    it and applied to it, so the trainer always knows exactly what the deployment serves, and can
    cut the chain with a full snapshot at any version.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    load_model = mo.ui.run_button(label="Download Qwen3-0.6B and build the mirror")
    load_model
    return (load_model,)


@app.cell(hide_code=True)
def _(
    DEVICE,
    Path,
    SimpleNamespace,
    UTC,
    WORK,
    cfg,
    datetime,
    load_model,
    mo,
    secrets,
    time,
    torch,
):
    mo.stop(not load_model.value)
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cwtrain.hotload import WeightMirror
    from cwtrain.hotload.mirror import tensor_locations

    MODEL = "Qwen/Qwen3-0.6B"
    REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

    _t = time.monotonic()
    _base = Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            local_dir=WORK / "base",
            allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"],
        )
    )
    _download = time.monotonic() - _t

    _t = time.monotonic()
    _tok = AutoTokenizer.from_pretrained(_base)
    _model = AutoModelForCausalLM.from_pretrained(_base, dtype=torch.float32).to(DEVICE)
    _model.gradient_checkpointing_enable()
    _model.config.use_cache = False
    _load = time.monotonic() - _t

    _t = time.monotonic()
    _mirror = WeightMirror.create(_base, WORK / "mirror", version="000000")
    _mirror_s = time.monotonic() - _t

    _name = f"molab-{datetime.now(UTC):%m%d-%H%M}-{secrets.token_hex(2)}"
    run = SimpleNamespace(
        name=_name,
        prefix=f"cwtrain-molab/{_name}",
        work=WORK,
        base=_base,
        tok=_tok,
        model=_model,
        # lr 1e-5: Adam moves each fp32 weight by about lr per step, so a few percent of the bf16
        # values the deployment serves cross a rounding step, and the delta stays small
        opt=torch.optim.AdamW(_model.parameters(), lr=1e-5, weight_decay=0.0),
        mirror=_mirror,
        names=sorted(tensor_locations(_base)),  # the tensors the deployment serves
        step=0,
        published=[],  # identities, oldest first
        pending=None,  # the hot-load in flight
        deployment_id=cfg["deployment_id"] or None,
        rollout_base_url=None,
        served_model=None,
        batch=None,
        delta=None,
    )
    mo.md(
        f"""
        Run **`{run.name}`** · download {_download:.1f}s · load {_load:.1f}s · mirror {_mirror_s:.1f}s

        The mirror holds {len(run.names)} tensors at version `{run.mirror.version}`, each with its
        Adler-32 (`{run.names[0]}` → `{run.mirror.checksums[run.names[0]]}`).
        """
    )
    return (run,)


@app.cell(hide_code=True)
def _(REMOTE, mo):
    mo.stop(not REMOTE)
    mo.md(r"""
    ## 3 · Stage the base model and create the deployment

    The deployment loads its base model from your bucket, under this run's prefix. It enables
    hot-load at creation: the `hotLoad` section names the bucket and the prefix that the trainer
    publishes snapshots under. Skip both buttons if you entered an existing deployment.
    """)
    return


@app.cell(hide_code=True)
def _(REMOTE, mo, run):
    mo.stop(not REMOTE)
    stage = mo.ui.run_button(label="Stage the base model", disabled=bool(run.deployment_id))
    create = mo.ui.run_button(
        label="Create the deployment", kind="warn", disabled=bool(run.deployment_id)
    )
    mo.hstack([stage, create], justify="start")
    return create, stage


@app.cell(hide_code=True)
def _(cfg, mo, run, s3_client, stage, time):
    mo.stop(not stage.value)
    from cwtrain.hotload import inspect_snapshot, upload_snapshot

    _t = time.monotonic()
    _info = inspect_snapshot(run.base)
    upload_snapshot(
        _info,
        bucket=cfg["bucket"],
        path_prefix=f"{run.prefix}/models",
        identity="base",
        s3=s3_client(),
    )
    mo.md(
        f"Uploaded {_info.total_bytes / 2**30:.2f} GiB to `s3://{cfg['bucket']}/{run.prefix}/models/base/` "
        f"in {time.monotonic() - _t:.1f}s. Each object carries its SHA-256, so a retry uploads only "
        "what is missing."
    )
    return


@app.cell(hide_code=True)
def _(cfg, client, create, infra, json, lifecycle, mo, run):
    mo.stop(not create.value)
    if infra.gateway_id is None:
        with mo.status.spinner(f"Creating a gateway in {cfg['region'].upper()}"):
            _gw = client.create_gateway(
                {
                    "name": run.name,
                    "zones": [cfg["region"].upper()],
                    "coreWeaveAuth": {},
                    "bodyBasedRouting": {"apiType": "API_TYPE_OPENAI"},
                }
            )
            infra.gateway_id, infra.created_gateway = _gw.id, True
            lifecycle.wait_for_gateway(client, _gw.id)
    deployment_spec = {
        "name": run.name,
        "gatewayIds": [infra.gateway_id],
        "runtime": {
            "engine": "dynamo-vllm",
            "version": cfg["runtime"],
            "engineConfig": {"max-model-len": "4096", "gpu-memory-utilization": "0.80"},
        },
        "resources": {"instanceType": cfg["instance_type"], "gpuCount": 1},
        "model": {"name": run.name, "bucket": cfg["bucket"], "path": f"{run.prefix}/models/base"},
        "autoscaling": {"min": 1, "max": 1, "concurrency": 64},
        "traffic": {"weight": 100},
        "hotLoad": {
            "bucket": cfg["bucket"],
            "pathPrefix": f"{run.prefix}/hotload",
            "transitionMode": "TRANSITION_MODE_ASYNC",
        },
    }
    _created = client.create_deployment(deployment_spec)
    run.deployment_id = _created.id
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"Created **{_created.name}** `{_created.id}`. It holds its GPU until **Clean up**."
                ),
                kind="warn",
            ),
            mo.accordion(
                {"Request body": mo.md(f"```json\n{json.dumps(deployment_spec, indent=2)}\n```")}
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(REMOTE, mo):
    mo.stop(not REMOTE)
    deployment_refresh = mo.ui.refresh(
        options=["5s", "15s", "30s"], default_interval="15s", label="Poll"
    )
    deployment_refresh
    return (deployment_refresh,)


@app.cell(hide_code=True)
def _(REMOTE, client, create, deployment_refresh, lifecycle, mo, run):
    mo.stop(not REMOTE)
    deployment_refresh.value, create.value
    mo.stop(not run.deployment_id, mo.md("_No deployment yet._"))
    from cwtrain import HotLoader

    deployment = client.get_deployment(run.deployment_id)
    if deployment.ready and run.rollout_base_url is None:
        _endpoints = lifecycle.endpoints(client, deployment)
        run.rollout_base_url = lifecycle.rollout_base_url(_endpoints[0]) if _endpoints else None
        run.served_model = deployment.model_name
        run.loader = HotLoader(client, deployment.id, poll_interval=2.0, timeout=900)
    _waiting = [
        f"- {c.type}: {c.message or c.reason}"
        for c in deployment.conditions
        if c.status.name != "TRUE"
    ]
    mo.vstack(
        [
            mo.md(
                f"Deployment **{deployment.name}**: **{deployment.status.name}**"
                + (
                    f" · rollouts at `{run.rollout_base_url}` · model `{run.served_model}`"
                    if run.rollout_base_url
                    else ""
                )
            ),
            *([mo.md("\n".join(_waiting))] if _waiting and not deployment.ready else []),
        ]
    )
    return (deployment,)


@app.cell(hide_code=True)
def _(REMOTE, mo):
    mo.stop(not REMOTE)
    mo.md(r"""
    ### Publish the chain root

    A delta is an XOR against the version before it, so the chain needs a full snapshot to start
    from: the trainer's own copy of the base weights, `<run>-v000000`. The deployment's base
    model cannot be the root. `write_full_snapshot` adds each tensor's checksum to the shard
    metadata, so the engine verifies every tensor when it installs the snapshot.
    """)
    return


@app.cell(hide_code=True)
def _(REMOTE, deployment, mo):
    mo.stop(not REMOTE)
    publish_root = mo.ui.run_button(label="Publish v000000", disabled=not deployment.ready)
    publish_root
    return (publish_root,)


@app.cell(hide_code=True)
def _(mo, publish_root, run, shutil, time):
    mo.stop(not publish_root.value)
    _identity = f"{run.name}-v000000"
    _t = time.monotonic()
    _root = run.mirror.write_full_snapshot(run.work / "root")
    _staged = time.monotonic() - _t
    _t = time.monotonic()
    _hl = run.loader.publish(_root, identity=_identity, wait=False)
    _uploaded = time.monotonic() - _t
    shutil.rmtree(_root)
    run.published.append(_identity)
    run.pending = _hl.id
    mo.md(
        f"Staged the full snapshot in {_staged:.1f}s, uploaded it in {_uploaded:.1f}s, and "
        f"created hot-load `{_hl.id}`. Watch it below."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · One training step

    Four buttons, one per stage. **Run steps** below does all four in a loop.

    The task is arithmetic with a checkable answer: the reward is 1 when the last `\boxed{}`
    holds the right number. Each step samples 8 prompts × 4 completions, and GRPO pushes up the
    completions that beat their group's mean.
    """)
    return


@app.cell(hide_code=True)
def _(DEVICE, asyncio, os, random, re, run, shutil, time, torch):
    PROMPTS, GROUP, MAX_TOKENS, TEMPERATURE = 8, 4, 384, 1.0

    def make_prompts(step):
        rng = random.Random(step)
        tasks = []
        for _ in range(PROMPTS):
            # hard enough that Qwen3-0.6B gets some wrong, so groups disagree and GRPO has signal
            a, b, c = rng.randint(101, 999), rng.randint(13, 99), rng.randint(100, 999)
            question = f"What is {a} * {b} + {c}? Think briefly, then put only the final number in \\boxed{{}}."
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
        found = re.findall(r"\\boxed\{\s*([-\d,\.]+)\s*\}", text)
        try:
            return float(bool(found) and float(found[-1].replace(",", "")) == answer)
        except ValueError:
            return 0.0

    async def sample_remote(tasks):
        """One /v1/completions request per group: exact token ids and log-probs come back."""
        from cwtrain.integrations.rollout import CompletionSampler, RolloutConfig

        sampler = CompletionSampler(
            RolloutConfig(
                base_url=run.rollout_base_url,
                model=run.served_model,
                api_key=os.environ["COREWEAVE_API_TOKEN"],
                concurrency=PROMPTS,
            )
        )
        try:
            groups = await asyncio.gather(
                *(
                    sampler.sample(
                        t["prompt_ids"], GROUP, max_tokens=MAX_TOKENS, temperature=TEMPERATURE
                    )
                    for t in tasks
                )
            )
        finally:
            await sampler.aclose()
        eos = run.tok.eos_token_id
        return [
            [
                {
                    "ids": list(c.token_ids) if c.token_ids else run.tok.encode(c.text) + [eos],
                    "logprobs": list(c.token_logprobs)
                    if c.token_ids and c.token_logprobs
                    else None,
                    "text": c.text,
                    "cached": c.cached_tokens,
                }
                for c in completions
            ]
            for completions in groups
        ]

    @torch.no_grad()
    def sample_local(tasks):
        """The same groups, sampled with this GPU: what the deployment does in remote mode."""
        tok, model = run.tok, run.model
        model.eval()
        width = max(len(t["prompt_ids"]) for t in tasks)
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
                end = next((k + 1 for k, t in enumerate(row) if t in stops), len(row))
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
        tok, model, opt = run.tok, run.model, run.opt
        samples = []
        for task, group in zip(batch["tasks"], batch["groups"], strict=True):
            rewards = torch.tensor([reward(c["text"], task["answer"]) for c in group])
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
            samples += [
                (task["prompt_ids"], c, float(r), float(a))
                for c, r, a in zip(group, rewards, adv, strict=True)
            ]
        mean_reward = sum(s[2] for s in samples) / len(samples)
        if all(a == 0 for *_, a in samples):
            return {"reward": mean_reward, "loss": 0.0, "logprob_gap": None, "trained": False}

        tokens = sum(len(c["ids"]) for _, c, _, _ in samples)
        loss_total, gap_sum, gap_n = 0.0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        for k in range(0, len(samples), 8):  # micro-batches keep the vocab-sized logits small
            chunk = samples[k : k + 8]
            seqs = [p + c["ids"] for p, c, _, _ in chunk]
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
        """Save the bf16 weights, write the delta against the mirror, and verify it by applying it."""
        from safetensors.torch import save_file

        from cwtrain.hotload.delta import write_delta

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
            verified = run.mirror.apply_delta(delta_dir).tensors_changed  # checks every Adler-32
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

    def wait_for_pending():
        """Block until the last hot-load is served everywhere, so the next rollouts are on-policy."""
        if run.pending is None:
            return None
        t = time.monotonic()
        done = run.loader.wait(run.pending)
        run.pending = None
        return done, time.monotonic() - t

    return (
        GROUP,
        MAX_TOKENS,
        PROMPTS,
        export_delta,
        make_prompts,
        reward,
        sample_local,
        sample_remote,
        train_step,
        wait_for_pending,
    )


@app.cell(hide_code=True)
def _(mo):
    get_history, set_history = mo.state({})
    return get_history, set_history


@app.cell(hide_code=True)
def _(REMOTE, mo, run):
    run  # the buttons appear once the model is loaded
    sample_button = mo.ui.run_button(label="a · Sample")
    train_button = mo.ui.run_button(label="b · Train")
    delta_button = mo.ui.run_button(label="c · Write the delta")
    publish_button = mo.ui.run_button(label="d · Hot-load it", disabled=not REMOTE)
    mo.hstack([sample_button, train_button, delta_button, publish_button], justify="start")
    return delta_button, publish_button, sample_button, train_button


@app.cell(hide_code=True)
async def _(
    GROUP,
    MAX_TOKENS,
    PROMPTS,
    REMOTE,
    make_prompts,
    mo,
    reward,
    run,
    sample_button,
    sample_local,
    sample_remote,
    time,
    wait_for_pending,
):
    mo.stop(not sample_button.value)
    mo.stop(REMOTE and not run.published, mo.callout("Publish v000000 first.", kind="warn"))
    _waited = wait_for_pending() if REMOTE else None
    _tasks = make_prompts(run.step + 1)
    _t = time.monotonic()
    _groups = await sample_remote(_tasks) if REMOTE else sample_local(_tasks)
    run.batch = {"tasks": _tasks, "groups": _groups, "seconds": time.monotonic() - _t}
    _rows = [
        {
            "question": t["question"].split("?")[0] + "?",
            "answer": t["answer"],
            "reward": reward(c["text"], t["answer"]),
            "tokens": len(c["ids"]),
            "engine log-probs": c["logprobs"] is not None,
            "completion": c["text"][-160:],
        }
        for t, g in zip(_tasks, _groups, strict=True)
        for c in g
    ]
    mo.vstack(
        [
            mo.md(
                (
                    f"Hot-load `{_waited[0].identity}` was served after another {_waited[1]:.1f}s. "
                    if _waited
                    else ""
                )
                + f"Sampled {PROMPTS} × {GROUP} completions (≤ {MAX_TOKENS} tokens) "
                f"{'from the deployment' if REMOTE else 'on this GPU'} in {run.batch['seconds']:.1f}s · "
                f"mean reward **{sum(r['reward'] for r in _rows) / len(_rows):.2f}**"
            ),
            mo.ui.table(_rows, selection=None, page_size=8),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, run, set_history, time, train_button, train_step):
    mo.stop(not train_button.value)
    mo.stop(run.batch is None, mo.callout("Sample first.", kind="warn"))
    # each delta is against the version before it, so write this step's delta before the next step
    mo.stop(
        run.mirror.version != f"{run.step:06d}", mo.callout("Write the delta first.", kind="warn")
    )
    _t = time.monotonic()
    _m = train_step(run.batch)
    _m["train_s"] = time.monotonic() - _t
    _m["sample_s"] = run.batch["seconds"]
    run.batch = None
    run.step += 1
    set_history(lambda h, s=run.step: {**h, s: {**h.get(s, {}), **_m}})
    mo.md(
        f"Step **{run.step}** in {_m['train_s']:.1f}s · reward {_m['reward']:.2f} · loss {_m['loss']:.4f}"
        + (
            f" · mean \\|Δ log p\\| trainer vs engine **{_m['logprob_gap']:.4f}**"
            if _m["logprob_gap"] is not None
            else ""
        )
        + ("" if _m["trained"] else " · every group had equal rewards, so nothing changed")
    )
    return


@app.cell(hide_code=True)
def _(delta_button, export_delta, mo, run, set_history):
    mo.stop(not delta_button.value)
    mo.stop(
        run.step == 0 or run.mirror.version == f"{run.step:06d}",
        mo.callout("Train a step first.", kind="warn"),
    )
    _dir, _stats, _m = export_delta(run.step)
    run.delta = (_dir, _stats)
    set_history(lambda h, s=run.step: {**h, s: {**h.get(s, {}), **_m}})
    mo.md(
        f"""
        Version `{run.step:06d}`: **{_m["tensors_changed"]} of {_m["tensors_total"]}** tensors changed
        ({_m["changed_mb"]:.0f} MiB of bf16 in the changed tensors), **{_m["wire_mb"]:.1f} MiB** on the wire after XOR + zstd.
        Saved in {_m["save_s"]:.1f}s, encoded in {_m["delta_s"]:.1f}s, and applied to the mirror in
        {_m["verify_s"]:.1f}s, where every changed tensor matched its checksum
        ({_m["verified"]} verified). The mirror is now at `{run.mirror.version}`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, publish_button, run, set_history, time):
    mo.stop(not publish_button.value)
    mo.stop(run.delta is None, mo.callout("Write the delta first.", kind="warn"))
    _dir, _stats = run.delta
    run.delta = None
    _identity = f"{run.name}-v{run.step:06d}"
    if _stats.is_empty:
        mo.stop(
            True,
            mo.md(
                "The delta is empty: the deployment already serves these weights. Nothing to publish."
            ),
        )
    from cwtrain.hotload import PromptCachePolicy

    _t = time.monotonic()
    _hl = run.loader.publish(
        _dir,
        identity=_identity,
        previous_identity=run.published[-1],
        prompt_cache_policy=PromptCachePolicy.RESET_ALL,
        wait=False,
    )
    _upload = time.monotonic() - _t
    run.published.append(_identity)
    run.pending = _hl.id
    set_history(
        lambda h, s=run.step: {**h, s: {**h.get(s, {}), "upload_s": _upload, "hot_load": _hl.id}}
    )
    mo.md(
        f"Uploaded `{_identity}` in {_upload:.1f}s and created hot-load `{_hl.id}`, chained on `{run.published[-2]}`."
    )
    return


@app.cell(hide_code=True)
def _(REMOTE, mo):
    mo.stop(not REMOTE)
    hot_load_refresh = mo.ui.refresh(
        options=["2s", "5s", "15s"], default_interval="5s", label="Hot-loads"
    )
    hot_load_refresh
    return (hot_load_refresh,)


@app.cell(hide_code=True)
def _(REMOTE, client, hot_load_refresh, mo, run):
    mo.stop(not REMOTE)
    hot_load_refresh.value
    mo.stop(not run.deployment_id or not run.published, mo.md("_No hot-loads yet._"))
    from itertools import islice

    _loads = list(islice(client.list_hot_loads(run.deployment_id), 50))
    _latest = _loads[0] if _loads else None
    _replicas = client.list_hot_load_replicas(_latest.id) if _latest else []
    mo.vstack(
        [
            mo.md(
                f"Latest **`{_latest.identity}`** ({_latest.spec.type.name}): **{_latest.state.name}**"
                if _latest
                else "_None yet._"
            ),
            mo.ui.table(
                [
                    {
                        "replica": r.name,
                        "zone": r.zone,
                        "state": r.state.name,
                        "serving": r.current_snapshot_identity,
                        "error": r.error_reason,
                    }
                    for r in _replicas
                ],
                selection=None,
            )
            if _replicas
            else mo.md(""),
            mo.accordion(
                {
                    "History": mo.ui.table(
                        [
                            {
                                "identity": h.identity,
                                "type": h.spec.type.name,
                                "state": h.state.name,
                                "seconds": round(
                                    (h.status.finished_at - h.status.created_at).total_seconds(), 1
                                )
                                if h.status.finished_at and h.status.created_at
                                else None,
                            }
                            for h in _loads
                        ],
                        selection=None,
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Run steps

    The same four stages in a loop. In Dedicated Inference mode, each step waits until the
    previous version is served before it samples, so every rollout comes from the weights it
    trains.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    steps = mo.ui.slider(1, 10, value=3, label="Steps", show_value=True)
    run_steps = mo.ui.run_button(label="Run steps")
    mo.hstack([steps, run_steps], justify="start")
    return run_steps, steps


@app.cell(hide_code=True)
async def _(
    REMOTE,
    export_delta,
    make_prompts,
    mo,
    run,
    run_steps,
    sample_local,
    sample_remote,
    set_history,
    steps,
    time,
    train_step,
    wait_for_pending,
):
    mo.stop(not run_steps.value)
    mo.stop(REMOTE and not run.published, mo.callout("Publish v000000 first.", kind="warn"))
    from cwtrain.hotload import PromptCachePolicy as _Policy

    for _ in mo.status.progress_bar(range(steps.value), title="Training", remove_on_exit=True):
        _m = {}
        if REMOTE:
            _waited = wait_for_pending()
            if _waited:
                set_history(
                    lambda h, s=run.step, w=_waited: {
                        **h,
                        s: {**h.get(s, {}), "served_after_s": w[1]},
                    }
                )
        _tasks = make_prompts(run.step + 1)
        _t = time.monotonic()
        _groups = await sample_remote(_tasks) if REMOTE else sample_local(_tasks)
        _m["sample_s"] = time.monotonic() - _t
        _t = time.monotonic()
        _m.update(train_step({"tasks": _tasks, "groups": _groups}))
        _m["train_s"] = time.monotonic() - _t
        run.step += 1
        _dir, _stats, _d = export_delta(run.step)
        _m.update(_d)
        if REMOTE and not _stats.is_empty:
            _identity = f"{run.name}-v{run.step:06d}"
            _t = time.monotonic()
            _hl = run.loader.publish(
                _dir,
                identity=_identity,
                previous_identity=run.published[-1],
                prompt_cache_policy=_Policy.RESET_ALL,
                wait=False,
            )
            _m["upload_s"] = time.monotonic() - _t
            run.published.append(_identity)
            run.pending = _hl.id
        set_history(lambda h, s=run.step, m=_m: {**h, s: {**h.get(s, {}), **m}})
    mo.md(f"Done: now at step {run.step}.")
    return


@app.cell(hide_code=True)
def _(get_history, mo):
    import altair as alt
    import pandas as pd

    _h = get_history()
    mo.stop(not _h, mo.md("_The charts fill in as you train._"))
    history = pd.DataFrame([{"step": s, **m} for s, m in sorted(_h.items())])
    _base = alt.Chart(history).encode(x=alt.X("step:O", title="step"))
    _charts = [
        _base.mark_line(point=True)
        .encode(y=alt.Y("reward:Q", title="mean reward", scale=alt.Scale(domain=[0, 1])))
        .properties(title="Reward", height=180),
    ]
    if "wire_mb" in history:
        _charts.append(
            alt.Chart(
                history.melt("step", ["changed_mb", "wire_mb"], var_name="size", value_name="MiB")
            )
            .mark_bar()
            .encode(x="step:O", xOffset="size:N", y="MiB:Q", color=alt.Color("size:N", title=None))
            .properties(title="Delta: changed tensors vs on the wire", height=180)
        )
    if "logprob_gap" in history and history["logprob_gap"].notna().any():
        _charts.append(
            _base.mark_line(point=True)
            .encode(y=alt.Y("logprob_gap:Q", title="mean |Δ log p|"))
            .properties(title="Trainer vs engine log-probs", height=180)
        )
    _timing_cols = [
        c
        for c in (
            "sample_s",
            "train_s",
            "save_s",
            "delta_s",
            "verify_s",
            "upload_s",
            "served_after_s",
        )
        if c in history
    ]
    _charts.append(
        alt.Chart(
            history.melt("step", _timing_cols, var_name="stage", value_name="seconds").dropna()
        )
        .mark_bar()
        .encode(x="step:O", y="seconds:Q", color=alt.Color("stage:N", sort=_timing_cols))
        .properties(title="Seconds per stage", height=180)
    )
    mo.vstack(
        [mo.hstack(_charts[i : i + 2], widths="equal") for i in range(0, len(_charts), 2)]
        + [mo.ui.table(history.round(3), selection=None)]
    )
    return


@app.cell(hide_code=True)
def _(REMOTE, mo):
    mo.stop(not REMOTE)
    mo.md(r"""
    ## 6 · Clean up

    Deletes the deployment, which stops its GPU billing, the gateway if this notebook created
    it, and every object under this run's prefix in the bucket: the staged base model and the
    snapshots. Nothing outside the prefix is touched.
    """)
    return


@app.cell(hide_code=True)
def _(REMOTE, mo):
    mo.stop(not REMOTE)
    confirm = mo.ui.checkbox(label="Delete this run's deployment and objects")
    confirm
    return (confirm,)


@app.cell(hide_code=True)
def _(confirm, mo):
    cleanup = mo.ui.run_button(label="Clean up", kind="danger", disabled=not confirm.value)
    cleanup
    return (cleanup,)


@app.cell(hide_code=True)
def _(cfg, cleanup, client, infra, lifecycle, mo, run, s3_client):
    mo.stop(not cleanup.value)

    _done = []
    # never delete a deployment or gateway that this run did not create
    if run.deployment_id and not cfg["deployment_id"]:
        if infra.created_gateway:
            # a gateway can go only once its deployments are gone, so wait for this one
            with mo.status.spinner("Deleting the deployment, then the gateway"):
                lifecycle.delete_deployment(client, run.deployment_id)
                lifecycle.delete_gateway(client, infra.gateway_id)
            _done.append(
                f"Deleted deployment `{run.deployment_id}` and gateway `{infra.gateway_id}`."
            )
        else:
            client.delete_deployment(run.deployment_id)
            _done.append(
                f"Deleting deployment `{run.deployment_id}`; it is gone within a few minutes."
            )
    _prefix = f"{run.prefix}/"
    assert _prefix.startswith("cwtrain-molab/molab-") and _prefix.count("/") == 2
    _s3 = s3_client()
    _deleted = 0
    for _page in _s3.get_paginator("list_objects_v2").paginate(
        Bucket=cfg["bucket"], Prefix=_prefix
    ):
        _keys = [{"Key": o["Key"]} for o in _page.get("Contents", [])]
        if _keys:
            _s3.delete_objects(Bucket=cfg["bucket"], Delete={"Objects": _keys, "Quiet": True})
            _deleted += len(_keys)
    _done.append(f"Deleted {_deleted} objects under `s3://{cfg['bucket']}/{_prefix}`.")
    mo.callout(mo.md("\n\n".join(_done)), kind="success")
    return


if __name__ == "__main__":
    app.run()
