# Molab training with laptop-managed Dedicated Inference

Run the training GPU in Molab while keeping Dedicated Inference management and all hot-load control on your laptop.
The two processes communicate through CoreWeave AI Object Storage (CAIOS), using one fresh
prefix per run. Molab uses S3 credentials and a W&B key for gateway rollout access and inline visualization;
the CoreWeave platform API token stays on the laptop. Local mode remains available for a demo without either CAIOS or Dedicated Inference.

| Component | Runs where | Access it needs |
| --- | --- | --- |
| Gateway/deployment creation, status and deletion | Laptop | CoreWeave API token |
| Rollout sampling | Molab → W&B-authenticated Dedicated Inference gateway | W&B API key |
| AdamW/GRPO training, bf16 mirror, XOR/zstd encoding and verification | Molab GPU | Public model download; CAIOS access |
| Snapshot upload and control/acknowledgment exchange | Molab ↔ CAIOS ↔ laptop | S3 credentials |
| Triggering hot-load and verifying every replica | Laptop | CoreWeave API token |

The laptop creates a dedicated gateway with `weightsAndBiasesAuth` and sends only its endpoint,
model name, step and served predecessor identity through S3. Molab samples exact token IDs and
aligned log-probabilities directly using `WANDB_API_KEY`, trains, verifies its delta, uploads it,
and commits a result marker. The laptop alone triggers the snapshot with `HotLoader.trigger`,
polls its status and verifies every replica before acknowledging the step. Molab cannot sample
its next batch before that acknowledgment, keeping the loop on-policy. Neither the CoreWeave
key nor the W&B key is included in S3 messages.

## Credentials and bucket

Use a bucket that Dedicated Inference can read, in its own CAIOS zone. Bucket zone and deployment
region are separate settings. Choose a fresh run ID, such as `molab-demo-001`; the shared prefix
will be `cwtrain-molab/molab-demo-001/`.

Molab's S3 credentials need:

- `s3:GetObject` and `s3:PutObject` for objects under that prefix, including control messages,
  step requests, base weights and snapshots.
- `s3:ListBucket` restricted to that prefix, used by resumable snapshot uploads. Multipart-upload
  permissions must also be available for large weight files.
- No Dedicated Inference permissions or ability to mint credentials. Write-only credentials are
  insufficient: the worker must read step requests and served-version acknowledgments.

Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in Molab Secrets. Set `AWS_SESSION_TOKEN`
**only if the issuer supplied one**. Alternatively enter those values in the notebook's masked
settings fields. Credentials issued with an expiry must remain valid for the whole run; this
worker does not refresh them through the CoreWeave management API. Existing boto3 credential
providers are supported when the form's credential fields are blank.

On the laptop, source your profile and map its existing key name if needed:

```bash
source ~/.profile
export COREWEAVE_API_TOKEN="${COREWEAVE_API_TOKEN:-$COREWEAVE_API_KEY}"
```

The driver accepts either name. Its default `--storage-auth aws` uses the laptop's usual AWS
credential chain. With `--storage-auth cwic`, the driver uses the locally installed `cwic auth
accesskey api-token` credential process instead. That exchange happens only on the laptop.
Credentials are never included in the S3 messages or local resource-state JSON. The cwic cache
is under the ignored `.cwtrain-laptop/` directory; treat it as local credential material.

## Run both sides

1. On the laptop, start the driver from the repository. It waits for Molab's ready marker:

   ```bash
   uv run training/examples/hotload_laptop.py \
     --bucket YOUR_BUCKET --zone US-EAST-02A \
     --run-id molab-demo-001 --steps 3 --storage-auth cwic \
     --region US-WEST-04A --instance-type h100-2x \
     --runtime 0.32.0+crwv.5
   ```

   Choose runtime, instance type and region that are available to your organization. The example
   defaults mirror the original notebook; they are not an availability guarantee. The laptop
   needs `uv`, network access to Dedicated Inference and CAIOS, and the platform token. It does
   not load model weights or the tokenizer, and needs no GPU.

2. Import `training/examples/hotload_notebook.py` into Molab and attach a GPU. The file is
   self-contained, so uploading the rest of this repository is unnecessary.
3. Choose **Laptop-managed Dedicated Inference**, enter the same bucket, zone and run ID, and
   submit the settings. Do not set a CoreWeave API token in Molab; remove any old one you added
   while testing the previous notebook.
4. Click **Download Qwen3-0.6B and build the mirror**.
5. Click **Upload base model and chain root**. Molab uploads both checkpoints and writes
   `ready.json` last. The laptop then creates the deployment and installs the chain root.
6. Set the notebook's maximum steps to 3 and click **Serve laptop requests**. It waits for the
   laptop to authorize the first step, then repeats sample → train → verify → upload → await served ack.
   Set `WANDB_API_KEY` in Molab before this step, through Secrets or the W&B form.
7. Check the notebook's metrics and the laptop's verified served identities. By default the
   laptop deletes the deployment and any gateway it created when it finishes or fails. Stop
   Molab's GPU from the Compute menu when finished.

Marimo's lazy execution mode may require **Run stale cells** after a button or form submission.
Both sides time out after 30 minutes per mailbox wait by default; the laptop timeout can be
changed with `--timeout`.

## W&B visualizations inside Molab

W&B logging runs **in the notebook**. Set `WANDB_API_KEY` in Molab Secrets or enter it in the
masked W&B API key field. This is separate from the CoreWeave platform token, which stays on
the laptop. The laptop driver needs no W&B key, SDK or tracking flags.

After submitting the notebook's main settings:

1. Expand **W&B visualization**, set the project and optional entity/team, and click
   **Start or restart W&B run** before training.
2. The run's W&B dashboard is embedded directly in the notebook, with an external link too.
   Private runs may require signing into W&B in the browser. The native live charts below
   remain available if the browser blocks the embedded dashboard.
3. Train normally. Molab logs reward, loss, gradient norm, log-probability gap, delta compression,
   verified tensor counts, GPU/machine stats and stage timings. The laptop returns hot-load and
   full-step timings in its CAIOS acknowledgments, so those are visible here too.
4. Click **Finish W&B run** after the last training step to save the final verified-step table
   and flush the run. Multiple training-button clicks can share the same W&B run. To log further
   steps after finishing, start a new W&B run first.

`WANDB_PROJECT` defaults to `cwtrain-molab`; `WANDB_ENTITY` is optional. The form can override
both. Logging begins when its form is submitted. Gateway rollout sampling still requires a W&B key
when tracking is disabled; Local mode can run without one.
Choose **offline** in the form to record in the Molab filesystem without uploading, and sync
that saved run before the notebook's ephemeral storage is lost.

| Panel group | Metrics |
| --- | --- |
| `train/*` | Reward, loss, gradient norm, trainer/engine log-probability gap, optimizer-update flag |
| `delta/*` | Changed/total/verified tensors, changed bf16 MiB, wire MiB, compression ratio |
| `timing/*` | Sampling, training, save, encoding, checksum verification, upload, hot-load, full step |
| `hotload/skipped` | Empty deltas that required no hot-load |
| `charts/delta_size` | Changed bf16 tensors versus compressed delta, in MiB |
| `charts/stage_seconds` | Seconds per pipeline stage |
| `tables/verified_steps` | Numeric metrics for every successfully served step |

Automatic W&B machine/GPU statistics now describe the Molab worker. Full-step timing includes
CAIOS transfer and waiting; upload timing accumulates retry attempts. In Local mode there are
no upload or hot-load timings. The run summary records the completed step count and last served
snapshot identity. Credentials, rollout text and model weights are excluded from this
integration's logged data; automatic code/git/console capture is disabled. Logging failures do
not interrupt training or the CAIOS acknowledgment barrier.

## Ordering, retries and restarts

The wire protocol uses JSON only, with a schema version, run ID, random trainer session ID,
step number and exact predecessor identity. No pickles or executable control messages are used.

```text
cwtrain-molab/<run-id>/
  models/base/                 # uploaded before ready.json
  hotload/<run-id>-v000000/     # full snapshot, uploaded before ready.json
  ready.json                   # trainer session + pinned model/revision
  requests/000001.json          # laptop: gateway endpoint/model + served predecessor
  hotload/<run-id>-v000001/     # Molab: verified snapshot shards and index
  results/000001.json           # Molab: committed after snapshot upload
  acks/000001.json              # laptop: served identity + timings, after replica verification
  control/done.json            # laptop finished
  control/error.json           # laptop failed; details stay in its terminal
  control/trainer-error.json   # training/export failed; restart with a fresh run
```

An empty delta is acknowledged without triggering a hot-load; its predecessor identity remains
unchanged. If upload or acknowledgment waiting fails, keep the kernel alive and retry **Serve
laptop requests**: it retains the pending delta and never trains that batch twice. The upload
helper checks existing files for matching content before reusing them. Retrying after a failed
training/export is deliberately refused because optimizer/model mutation may already have
happened.

A kernel restart loses the optimizer and model state. Start both sides with a **new run ID**;
reusing a prior ready marker is rejected. A driver restart is also a new run, after cleaning up
any old resources. Versioned step requests are not replayed against a newly initialized model.

## Cleanup and recovery

The laptop saves planned resource IDs before creating them, so an interrupted creation is
recoverable. Its local state is `.cwtrain-laptop/<run-id>/state.json`. Each run creates its own W&B-authenticated gateway. Cleanup waits until the created deployment
is gone before deleting that gateway; recovery also preserves any unowned gateway in older state.

If the laptop was killed, or you used `--keep-deployment` to retain resources for inspection:

```bash
uv run training/examples/hotload_laptop.py --run-id molab-demo-001 --cleanup
```

Preserve the local state until cleanup succeeds. CAIOS run objects are kept for inspection and
can be deleted separately under that run's prefix. Molab never deletes deployments, gateways,
or bucket objects. Starting or stopping the notebook alone does not stop Dedicated Inference
billing; the laptop driver's cleanup does.

## Checks without cloud resources

```bash
uv run --with pytest --with wandb --with-editable ./libs/cwtrain \
  pytest training/examples/test_hotload_bridge.py -q
uvx marimo check --strict training/examples/hotload_notebook.py
```
