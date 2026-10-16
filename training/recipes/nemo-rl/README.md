# Recipe: NeMo-RL GRPO with rollouts on a CoreWeave hot-load deployment

Train with [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) and sample rollouts from a CoreWeave
Dedicated Inference deployment. The deployment receives every refit by checkpoint hot-load.
NeMo-RL starts no inference engines:

- A generation backend samples from the deployment's OpenAI-compatible endpoint.
- A weight synchronizer exports the policy and hot-loads it before each rollout round, including
  the update from the preceding optimizer step.

Training pauses until every replica serves the new version, so the next rollout is on-policy.

![One training step: the trainer exports weights, the cwtrain synchronizer uploads a full or delta snapshot to Object Storage, creates a hot-load and polls until every replica is READY; the next rollout batch samples fresh weights from the OpenAI-compatible endpoint.](../../../docs/hotload-flow.drawio.svg)

The [slime](../slime/README.md) and [miles](../miles/README.md) recipes run the same loop. Those
frameworks expose a post-write hook. NeMo-RL has two seams instead, and this recipe uses both.

## How it plugs into NeMo-RL

| NeMo-RL seam | What the recipe provides |
| --- | --- |
| `policy.generation.backend` | `coreweave`: `cwtrain.integrations.nemo_rl.CoreWeaveGeneration`, a `GenerationInterface`. It turns NeMo-RL's token-id batches into `/v1/completions` requests (prompts as token ids, `n` per distinct prompt). It returns `output_ids`, `logprobs`, lengths and `truncated` in the same shape as the vLLM backend. It needs no GPUs and no Ray workers. |
| `policy_generation.weight_synchronizer` | `HotLoadWeightSynchronizer`. On every refit it exports the weights in HF layout and publishes a full snapshot (first refit and every `full_every`) or a zstd XOR delta against the previous export. Then it hot-loads the snapshot and waits until every replica is READY on the new identity. Per-version checksum manifests go next to the snapshots. |
| `policy.worker_extension_cls_fqn` | `CoreWeaveDTensorPolicyWorker` (and a Megatron counterpart): the stock policy worker plus an `export_hf_checkpoint` RPC. It reuses the HF-layout weight generator of NeMo-RL's NCCL refit, so TP/PP shards are gathered the same way. The refit casts every tensor to the policy dtype. The DTensor worker instead keeps each tensor in the dtype that the deployment's checkpoint stores. The Megatron worker writes the dtypes that Megatron-Bridge exports. Rank 0 writes the safetensors shards, the index and the checkpoint's non-weight files. |

NeMo-RL's registries are hardcoded. So
[`patches/0001-coreweave-generation-backend.patch`](patches/0001-coreweave-generation-backend.patch)
adds these changes to the `v0.7.0` release (the newest release with an arm64 container):

- A `coreweave` branch in `grpo.setup()`. It builds only the policy's Ray cluster (no inference
  cluster, no NCCL collective). It creates the policy with the cwtrain worker extension for the
  configured training backend, or with the `worker_extension_cls_fqn` you set. It registers that
  class's Ray Python environment (reusing the stock worker's prebuilt venv) and attaches the
  synchronizer.
- A short-circuit at the top of `refit_policy_generation`. When the generation object has a
  `weight_synchronizer`, it calls `sync_weights()` and returns. Newer NeMo-RL releases have this
  seam.

Nothing else in NeMo-RL changes. The loop, data, loss and checkpointing are stock.

### About the patch

The patch (about 120 changed lines) is necessary until NeMo-RL can load a generation backend and a
weight synchronizer from configuration. All of the work happens in the `cwtrain` library, which
has no patches. The launcher ([`launch.py`](launch.py)) checks the patch against the checkout in
the container before it applies it. If a patch fails, the launcher reverts what it applied and
stops the job with a message that names the tested version. So a different NeMo-RL never starts
half-patched.

### Tested versions

| Component | Version |
| --- | --- |
| NeMo-RL | [`v0.7.0`](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.7.0); the patch applies to the tag and to the checkout inside the container |
| Trainer image | `nvcr.io/nvidia/nemo-rl:v0.7.0` (arm64 and amd64) |
| Bundled training Gym | `0.4.0` |
| Inference engine | `dynamo-vllm` `0.30.0+crwv.5` |

## What you need

Follow the Dedicated Inference
[getting-started guide](https://docs.coreweave.com/products/inference/getting-started) for the
account prerequisites, the gateway and the bucket policy. Then get these:

- **A hot-load-enabled deployment** (engine `dynamo-vllm`, with a `hotLoad` section). Use
  [`deployment.nemotron.example.json`](deployment.nemotron.example.json) and see
  [Creating the deployment](#creating-the-deployment). The prerequisites are in the
  [library README](../../../libs/cwtrain/README.md#prerequisites). CoreWeave enables hot-load per
  organization. To enable it, ask CoreWeave support and give your organization ID. Until then, the
  create fails with `hot-load is not enabled for this organization`. The
  [RL Rollouts docs](https://docs.coreweave.com/products/inference/rl-rollouts) describe the
  feature. The deployment's `autoscaling.concurrency` limits how many sequences each replica
  decodes at once. See [Concurrency](#concurrency).
- **A bucket that the inference service can read and the trainer can write.**
  [`bucket-policy.example.json`](../../../k8s/bucket-policy.example.json) has four statements. The
  first two are from the docs'
  [Grant inference access to your bucket](https://docs.coreweave.com/products/inference/models#grant-inference-access-to-your-bucket):
  `AllowInference` lets the canonical inference service account read, and `AllowOrgUsers` lets
  your org read. **Any attached policy makes the bucket explicit-allow only, so do not remove the
  org statement.** The other two statements name the trainer and the admin:
  - `AllowTrainer` lets the trainer add objects under the deployment's hot-load prefix, and nowhere
    else. It cannot delete objects or change the policy.
  - `AllowAdmin` lets your admin user stage the base model, and delete objects when you clean up.

  To apply the policy:
  1. Copy it to `k8s/bucket-policy.run.json` in this recipe's directory (git-ignored, like
     `deployment.run.json`).
  2. Fill in `[BUCKET-NAME]`, `[ORG-ID]`, `[HOT-LOAD-PREFIX]` (the deployment's
     `hotLoad.pathPrefix`, `hotload` in every recipe) and `[ADMIN-USER-ID]`, which is on the
     admin's Cloud Console Settings page. Fill in the trainer's principal as the next item
     describes.
  3. From this directory, with the admin's credentials, run `aws s3api put-bucket-policy --bucket
     <bucket> --policy file://k8s/bucket-policy.run.json --endpoint-url https://cwobject.com
     --region <bucket zone>`.

  If your credentials are in the cluster, [`make-bucket-job.yaml`](../../../k8s/make-bucket-job.yaml)
  can create the bucket and attach the same policy instead. It fills in the template from the kit
  ConfigMap with the values in `run.env`. Running it again keeps the policy's other statements, and
  keeps the trainers and admins that an earlier run added.
- **A principal of its own for the trainer.** A request needs an allow from both your org access
  policies and the bucket policy. But `s3:PutBucketPolicy` is checked against org access policies
  only, so if the trainer shared the admin's principal, it could replace the bucket policy. We
  recommend that the trainer Job run as a Kubernetes service account with
  [Workload Identity Federation](https://docs.coreweave.com/security/tutorials/cks-object-storage-authentication/manual).
  The pod exchanges the service account's token for temporary credentials, so no long-lived key
  sits in the cluster. This needs:
  1. The cluster's OIDC issuer, registered on the Cloud Console's **Workload Federation** page with
     the audience `https://coreweave.com/iam`. This needs the IAM Admin role. The issuer URL is on
     the cluster's page in the Cloud Console.
  2. An org access policy for the service account. Copy
     [`trainer-access-policy.example.json`](../../../k8s/trainer-access-policy.example.json) to
     `k8s/trainer-access-policy.run.json` in this recipe's directory (git-ignored), fill it in, and
     apply it from this directory with
     `cwic cwobject policy create --file k8s/trainer-access-policy.run.json`. It allows the token
     exchange, reading the bucket, and `s3:PutObject` and `s3:AbortMultipartUpload` under the
     hot-load prefix, and nothing else.
  3. In `AllowTrainer`, the issuer URL for `[OIDC-ISSUER-URL]`, and `NAMESPACE` and
     `TRAINER_SERVICE_ACCOUNT` from `run.env` for `[NAMESPACE]` and `[SERVICE-ACCOUNT]`.

  To use a Cloud Console user's key instead, give the trainer a user of its own, and set
  `AllowTrainer`'s principal to `{"CW": ["arn:aws:iam::[ORG-ID]:coreweave/[TRAINER-USER-ID]"]}`
  (see `run.env.example`). To try the recipe on your own, you can name your own user in both
  statements, but the trainer's key then has all of your rights; the bucket Job warns about this.
  At start, the trainer Job checks that its credentials can list, read and write under the
  hot-load prefix, and it stops if they cannot.
- **Object storage credentials.** The trainer Job mounts `STORAGE_SECRET`: with Workload Identity
  Federation, it holds no key; otherwise, the trainer's key. The Jobs that create the bucket and
  stage the base model mount the admin's key from `ADMIN_STORAGE_SECRET`. The trainer also needs
  the API token in its environment.
- **The NeMo-RL container**, `nvcr.io/nvidia/nemo-rl:v0.7.0` (amd64 and arm64 on NGC). The launch
  script patches `/opt/nemo-rl` in place and installs the `cwtrain` wheel into the project venv.
- **The base model, staged separately for each deployment.** Stage it into the bucket with
  `k8s/stage-nemotron-job.yaml`. The deployment loads it from there. The trainer fetches the same
  objects from `BASE_S3_URI`, which `run.env.example` sets to that location
  (`s3://<bucket>/<model prefix>/<model slug>`). The launcher does not download from Hugging Face,
  so the trainer stops at start if the model is not there. The synchronizer publishes the first
  refit as a full snapshot, the root of the run's chain. So the deployment's copy does not have to
  stay byte-identical to the policy. But it sets the layout: the synchronizer refuses an export
  whose tensor names, dtypes or shapes differ from that checkpoint.

## Configure

[`grpo_coreweave.yaml`](grpo_coreweave.yaml) inherits NeMo-RL's `grpo_math_1B.yaml` and overrides
what this topology needs. The model is `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`. The
trainer uses eight GPUs and eight-way expert parallelism, with sequence packing off. The hot-load
deployment requests two H100s, for the BF16 model and memory headroom during refits. The trainer
and the deployment must load the same prepared BF16 checkpoint.

The pinned NeMo-RL AutoModel ignores the optional `mtp.*` speculative-decoding heads.
[`prepare_model.py`](prepare_model.py) makes a separate checkpoint without those heads:

- It turns off MTP in the config.
- It keeps every main-model shard byte-identical.
- It rejects checkpoints that mix MTP and main-model tensors in one shard.

The staging Job runs this step before it uploads. Use the prepared copy for both the trainer and
the deployment, and keep speculative decoding off. Otherwise cwtrain's tensor-layout check rejects
the first export. Use `k8s/stage-nemotron-job.yaml`, which gives 80 GiB for the raw checkpoint.
The shared [`stage-model-job.yaml`](../../../k8s/stage-model-job.yaml) keeps its 60 GiB default.

Set a unique `DEPLOYMENT_RUN` in `run.env` for every new inference deployment, and stage its base
model under that prefix. Two deployments in one organization that use the same model path can
collide in the weight UID registry of the model exchange server. A new served-model name does not
prevent this. Keep the value the same while you stage, create and train one deployment.

The backend configuration is:

```yaml
policy:
  worker_extension_cls_fqn: cwtrain.integrations.nemo_rl.worker.CoreWeaveDTensorPolicyWorker  # optional; the patch defaults to it
  dtensor_cfg:
    env_vars: {PYTHONPATH: /work/py}   # the worker venv sees the extension without cwtrain's deps
  generation:
    backend: coreweave
    colocated: {enabled: false, resources: {gpus_per_node: null, num_nodes: null}}
    coreweave_cfg:
      base_url: ${oc.env:CWTRAIN_ROLLOUT_BASE_URL}     # the deployment's gateway endpoint, /v1
      model: ${oc.env:CWTRAIN_ROLLOUT_MODEL}
      api_key_file: ${oc.env:CWTRAIN_ROLLOUT_API_KEY_FILE,null}
      max_model_len: ${policy.max_total_sequence_length}
      deployment_id: ${oc.env:CWTRAIN_HOTLOAD_DEPLOYMENT_ID}
      run_id: ${oc.env:CWTRAIN_HOTLOAD_RUN_ID}          # unique per run; identities are <run>-vNNNNNN
      full_every: ${oc.env:CWTRAIN_HOTLOAD_FULL_EVERY,null}  # null: cwtrain's default, 20
      state_dir: /work/hotload                          # exports, deltas, manifests
```

Every `coreweave_cfg` key except `max_model_len` and `require_logprobs` can also come from the
`CWTRAIN_ROLLOUT_*` and `CWTRAIN_HOTLOAD_*` environment variables that the other recipes use. The
API token (`COREWEAVE_API_TOKEN`, from a Secret in the Job) and the object storage credentials come
only from the environment.

`coreweave_cfg.timeout` sets two timeouts: the rollout requests and the hot-load wait. To set them
separately, leave it unset and use `CWTRAIN_ROLLOUT_TIMEOUT` and `CWTRAIN_HOTLOAD_TIMEOUT`. In the
Kubernetes run, use `HOTLOAD_TIMEOUT` in `run.env`, next to `HOTLOAD_WAIT`, `HOTLOAD_VERIFY` and
`HOTLOAD_PROMPT_CACHE_POLICY`.

GRPO here is synchronous. So, unless you set it, the launcher sets
`CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY=reset_all`: each rollout round then starts without KV cache
from the previous weights.

The recipe runs the DTensor (Automodel) policy backend. The Megatron worker extension exists but is
not tested on a live deployment.

## Sequence of one training step

| Step | Who | What |
| --- | --- | --- |
| refit | synchronizer | `export_hf_checkpoint` on every policy worker (rank 0 writes `<state_dir>/exports/vN`, each tensor in the base checkpoint's dtype), verify the tensor names, dtypes and shapes equal the base checkpoint's, write the checksum manifest |
| refit | synchronizer | v0 and every `full_every`: checksummed **full** snapshot of the export; otherwise a zstd XOR **delta** against the previous export, chained to the last published identity |
| refit | synchronizer | upload, create the hot-load, and wait until it completes (every replica READY on the new identity) |
| rollout | generation backend | one request per distinct prompt with `n` = its repeats; token ids in, token ids out (or re-tokenized text plus the stop token), sampled-token log-probs when they align |
| train | NeMo-RL | stock GRPO step; the next refit publishes the result |

If a refit's export is byte-identical to the previous one, the synchronizer publishes nothing.

A restarted run that keeps its run id finds versions that it already published. Before the
synchronizer records or uploads anything under an identity, full or delta, it checks the export
against that identity's stored checksum manifest:

- Matching weights are reused without an upload. If the deployment has moved on to a later
  version, the synchronizer hot-loads them again first.
- Different weights stop the run with a request for a new run id. So a restart never overwrites a
  published version.

## Running it on Kubernetes

This recipe's `k8s/` directory, with the files that the training recipes share in
[`k8s/`](../../../k8s/README.md) at the repository root, runs this topology on a Kubernetes cluster
with a GPU node of `INSTANCE_TYPE`, such as CoreWeave Kubernetes Service (CKS). You need these
tools:

- `kubectl`
- `envsubst` (GNU gettext; `brew install gettext` on macOS)
- `uv`
- the `aws` CLI
- the `cwtrain` CLI. To install it, run `uv tool install ./libs/cwtrain` from the repository root.

Work in this recipe's `k8s/` directory, with a filled-in copy of `run.env.example` loaded into your
shell:

```bash
cd training/recipes/nemo-rl/k8s
K=../../../../k8s                # the shared files: k8s/ at the repository root
[ -e run.env ] || cp run.env.example run.env   # once; fill in every <placeholder>, render.sh refuses one
set -a; . ./run.env; set +a      # the commands below use its values
CTX=$(kubectl config current-context)
```

| Step | Command |
| --- | --- |
| 1. Namespace | `"$K/render.sh" run.env "$K/namespace.yaml" \| kubectl apply -f -` |
| 2. Trainer service account and object storage Secrets (AWS-style `config` + `credentials` files) | `kubectl -n "$NAMESPACE" create serviceaccount "$TRAINER_SERVICE_ACCOUNT"`, then `WORKLOAD_IDENTITY=1 AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$STORAGE_SECRET"` (with a trainer key instead, export it as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` and leave out `WORKLOAD_IDENTITY=1`); then, with the admin's key exported, `AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$ADMIN_STORAGE_SECRET"` |
| 3. Wheel + config + patch ConfigMap | `(cd ../../../../libs/cwtrain && uv build --wheel) && ./make-kit.sh "$CTX" "$NAMESPACE" "$KIT_NAME"` |
| 4. Bucket + policy (skip if done with `aws s3api`) | `"$K/render.sh" run.env "$K/make-bucket-job.yaml" \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m job/cwtrain-make-bucket`; it is done when `kubectl -n "$NAMESPACE" get job cwtrain-make-bucket` shows COMPLETIONS `1/1` |
| 5. Stage the base model into the bucket (required: the trainer reads it from there) | `"$K/render.sh" run.env stage-nemotron-job.yaml \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m "job/cwtrain-stage-$JOB_NAME"`; it is done when `kubectl -n "$NAMESPACE" get job "cwtrain-stage-$JOB_NAME"` shows COMPLETIONS `1/1` (it retries once, so follow the logs again while it shows `0/1`) |
| 6. Create the deployment | [below](#creating-the-deployment); put its `id` and `rollout_base_url` in `run.env` as `DEPLOYMENT_ID` and `ROLLOUT_BASE_URL` |
| 7. Inference API key Secret | `kubectl -n "$NAMESPACE" create secret generic "$INFERENCE_KEY_SECRET" --from-file=api_key=<file>` |
| 8. CoreWeave API token Secret | `kubectl -n "$NAMESPACE" create secret generic "$API_TOKEN_SECRET" --from-file=token=<file>` |
| 9. Trainer | `"$K/render.sh" run.env trainer-job.yaml \| kubectl apply -f -` |

Notes on the steps:

- **Step 7** uses the bearer token that the deployment's gateway accepts for inference requests.
  When the gateway uses CoreWeave authentication, this is a CoreWeave API token.
- **Step 8** uses a token with the Inference Admin role, because it creates hot-loads. One token
  can serve both steps. The trainer pod does not start until both Secrets exist. Reading the
  tokens from files keeps them out of your shell history.
- **Step 2**'s storage key must stay valid through staging, node provisioning and training. Use an
  object storage key that lasts for the whole run. If you copy temporary `credential_process`
  output into a static Secret, it does not refresh.

`run-trainer.sh` installs the wheel, with its dependencies, into `/opt/nemo_rl_venv`. Then it
starts [`launch.py`](launch.py), which:

1. applies the patch,
2. installs the wheel into the prebuilt worker venvs (without dependencies),
3. fetches the base model,
4. runs `python examples/run_grpo.py --config grpo_coreweave.yaml` with the venv's interpreter.

Neither script uses `uv run`, because it re-syncs the venv and removes the wheel.

`EXTRA_OVERRIDES` is split on whitespace, one override per word. Quotes reach Hydra unchanged. So
write list values without spaces: `policy.generation.stop_strings=["</answer>","<|end|>"]`. The
unit tests for the launcher's argument building are in `tests/`.

### Creating the deployment

1. Copy [`deployment.nemotron.example.json`](deployment.nemotron.example.json) to
   `deployment.run.json` in this recipe's `k8s/` (git-ignored).
2. Fill in the gateway id, instance type, bucket and unique deployment prefix. Set `model.path` to
   `$MODEL_PREFIX/$MODEL_SLUG` from the same `run.env` that you used for staging.
3. The example pins `dynamo-vllm` to `0.30.0+crwv.5`. Make sure that the gateway zone has this
   version with hot-load support. Put your API token in `COREWEAVE_API_TOKEN` (or `CW_API_TOKEN`,
   as the CoreWeave docs set it). Then `cwtrain deployment parameters` lists the runtime versions,
   instance types and engine options for your organization. Add `--gateway <gateway id>` to show
   only that gateway's zone.

```bash
cwtrain deployment parameters
[ -e deployment.run.json ] || cp ../deployment.nemotron.example.json deployment.run.json   # then fill it in
```

4. Create the deployment from the filled-in file, and wait until it serves:

```bash
cwtrain deployment create --file deployment.run.json --wait
```

`create` refuses a file that still has a `<placeholder>`. If the name is taken, it names the
deployment that has it. With `--wait`, it returns when the deployment is READY. This takes several
minutes, and longer for large models. Without `--wait`, it returns at once, and
`cwtrain deployment get "$DEPLOYMENT_ID"` shows the status later.

In both cases it prints the deployment as JSON, also when the wait fails. **The deployment then
still exists and holds its GPUs until you delete it** ([Cleaning up](#cleaning-up)).

5. Put the deployment's `id` in `run.env` as `DEPLOYMENT_ID` and its `rollout_base_url` as
   `ROLLOUT_BASE_URL`. A `null` URL means that the gateway was not readable yet;
   `cwtrain deployment get "$DEPLOYMENT_ID"` shows it later. Make sure that `ROLLOUT_MODEL` matches
   the deployment's `model_name`. Then load `run.env` again (`set -a; . ./run.env; set +a`).

The trainer also waits for READY before its first hot-load, within `HOTLOAD_TIMEOUT`.

The example sets `gpu-memory-utilization`, `max-model-len`, `max-num-seqs` and Nemotron's
`reasoning-parser`. Add other options only from the engine options that
`cwtrain deployment parameters` lists.

`deployment.run.json` is the request body of `POST /v1alpha1/inference/deployments`. So the `curl`
request in the [RL Rollouts guide](https://docs.coreweave.com/products/inference/rl-rollouts) with
`-d @deployment.run.json` creates the same deployment. You can also create it in the Cloud Console,
in the **Hot loading** section of the create drawer.

### Watching and re-running

In a new shell, run the setup block in [Running it on Kubernetes](#running-it-on-kubernetes) again
first: these commands and step 9 use its `K` and the values in `run.env`. The block keeps your
filled-in `run.env`.

```bash
kubectl -n "$NAMESPACE" logs -f "job/cwtrain-$JOB_NAME"   # the trainer, including cwtrain's lines
cwtrain hotload list --deployment "$DEPLOYMENT_ID"         # one hot-load per published version
```

You cannot change a Job while it exists. Before the next run, delete the finished Job with
`kubectl -n "$NAMESPACE" delete job "cwtrain-$JOB_NAME"`. Then render and apply step 9 again.

The `RUN_ID` in `run.env.example` uses the time of each render plus a random number. So each new
run publishes under new identities, as it must. For what a restart under the same id does, see
[Sequence of one training step](#sequence-of-one-training-step). Your shell's `$RUN_ID` comes from
when you loaded `run.env`, not from the render. `cwtrain hotload list` shows the identities that
the Job uses.

### Cleaning up

The deployment holds its GPUs until you delete it. Run this block with the kubectl context that you
used for the steps above (`kubectl config current-context`), and with the admin's object-storage
credentials. Only the admin may delete from the bucket:

```bash
( set -eu   # run from anywhere in the repository; stops at the first unset value or failed step
  cd "$(git rev-parse --show-toplevel)/training/recipes/nemo-rl/k8s"
  set -a; . ./run.env; set +a
  # deployment.run.json, when present, decides; a file without hotLoad.pathPrefix stops the block.
  # Like the API, ignore slashes at either end of the prefix ("/hotload/" is "hotload").
  HOTLOAD_PREFIX=$(HOTLOAD_PREFIX=${HOTLOAD_PREFIX:-} python3 -c 'import json, os, sys
if os.path.exists("deployment.run.json"):
    p = json.load(open("deployment.run.json")).get("hotLoad", {}).get("pathPrefix") or ""
    p.strip("/") or sys.exit("deployment.run.json has no hotLoad.pathPrefix, so the snapshots sit at the top of the bucket; remove them by hand")
else:
    p = os.environ["HOTLOAD_PREFIX"]
print(p.strip("/"))')
  : "${DEPLOYMENT_ID:?set DEPLOYMENT_ID in run.env}" \
    "${HOTLOAD_PREFIX:?set HOTLOAD_PREFIX in run.env to the hotLoad.pathPrefix of the deployment}" "${NAMESPACE:?}" \
    "${BUCKET:?}" "${BUCKET_ZONE:?}" "${MODEL_PREFIX:?}" "${MODEL_SLUG:?}" "${JOB_NAME:?}" "${KIT_NAME:?}" \
    "${STORAGE_SECRET:?}" "${ADMIN_STORAGE_SECRET:?}" "${INFERENCE_KEY_SECRET:?}" "${API_TOKEN_SECRET:?}" \
    "${TRAINER_SERVICE_ACCOUNT?set TRAINER_SERVICE_ACCOUNT in run.env, empty for "default"}"
  case "$DEPLOYMENT_ID $NAMESPACE $BUCKET $BUCKET_ZONE $MODEL_PREFIX $MODEL_SLUG $HOTLOAD_PREFIX $JOB_NAME
        $KIT_NAME $STORAGE_SECRET $ADMIN_STORAGE_SECRET $INFERENCE_KEY_SECRET $API_TOKEN_SECRET
        $TRAINER_SERVICE_ACCOUNT" in
    *'<'[A-Za-z]*'>'*) echo "fill in run.env before cleaning up" >&2; exit 1;;
  esac
  # Every deployment shares the hot-load prefix, so remove only the runs that this deployment
  # hot-loaded: <prefix>/<run>-vNNNNNN/ and <prefix>/_manifests/<run>-vNNNNNN.json. List them
  # while the deployment exists.
  runs=$(cwtrain hotload list --deployment "$DEPLOYMENT_ID" --limit 100000 | python3 -c 'import json, re, sys
print("\n".join(sorted({re.sub(r"-v[0-9]+$", "", h["spec"]["identity"]) for h in json.load(sys.stdin)})))')
  cwtrain deployment delete "$DEPLOYMENT_ID" --ignore-not-found
  kubectl -n "$NAMESPACE" delete job "cwtrain-$JOB_NAME" "cwtrain-stage-$JOB_NAME" cwtrain-make-bucket --ignore-not-found
  kubectl -n "$NAMESPACE" delete secret "$STORAGE_SECRET" "$ADMIN_STORAGE_SECRET" "$INFERENCE_KEY_SECRET" \
    "$API_TOKEN_SECRET" --ignore-not-found
  kubectl -n "$NAMESPACE" delete configmap "$KIT_NAME" --ignore-not-found
  [ -z "$TRAINER_SERVICE_ACCOUNT" ] || kubectl -n "$NAMESPACE" delete serviceaccount "$TRAINER_SERVICE_ACCOUNT" --ignore-not-found
  [ -n "$runs" ] || echo "found no hot-loads of $DEPLOYMENT_ID, so nothing under $HOTLOAD_PREFIX/ is removed" >&2
  # read line by line: zsh, unlike bash, does not split an unquoted $runs into words
  printf '%s\n' "$runs" | while read -r run; do
    [ -z "$run" ] || aws s3 rm --recursive "s3://$BUCKET/$HOTLOAD_PREFIX/" --exclude '*' \
      --include "$run-v*" --include "_manifests/$run-v*" --endpoint-url https://cwobject.com \
      --region "$BUCKET_ZONE"
  done
  for prefix in "$MODEL_PREFIX/$MODEL_SLUG" "$MODEL_PREFIX/_probe"; do
    aws s3 rm --recursive "s3://$BUCKET/$prefix/" --endpoint-url https://cwobject.com --region "$BUCKET_ZONE"
  done
)
```

The block removes from the bucket:

- The runs that the deployment hot-loaded: each `<run>-vNNNNNN/` snapshot under the hot-load
  prefix, and its entry in `_manifests/`. Other deployments' runs under the same prefix stay. The
  prefix comes from `deployment.run.json`, or from `HOTLOAD_PREFIX` in `run.env` if you created the
  deployment another way.
- The staged base model.
- The bucket Job's write check (`_probe`).

It also deletes the Jobs, the Secrets, the kit ConfigMap and the trainer's service account that
`run.env` names, and the bucket Job. It does not delete the namespace or the trainer's org access
policy. You can run it again after a partial cleanup. But a deleted deployment lists no hot-loads,
so if the deployment is already gone, remove its runs by hand (`aws s3 ls` the hot-load prefix).

Recipes that share a namespace also share these objects, unless their `run.env` names differ. The
defaults are the same in every recipe, so clean up after the last recipe. If this recipe created
the namespace and nothing else runs there, delete it too: `kubectl delete namespace "$NAMESPACE"`.

## Concurrency

Two settings control how many sequences are sampled at once:

- **The deployment's `autoscaling.concurrency`**, a per-replica limit. The platform passes it to
  vLLM as `--max-num-seqs`. So a replica decodes at most that many sequences at once, and later
  requests wait in a queue. vLLM never schedules more than its KV cache holds, so a higher value
  only moves the wait into the engine.
  [`deployment.nemotron.example.json`](deployment.nemotron.example.json) sets 8.
- **`ROLLOUT_CONCURRENCY`** in `run.env` (default 32), the number of requests that the generation
  backend keeps in flight. Each request is one prompt that asks for
  `grpo.num_generations_per_prompt` completions. The Job passes it as
  `CWTRAIN_ROLLOUT_CONCURRENCY`, which `coreweave_cfg.concurrency` reads. An empty value keeps the
  config's 32.

When requests × completions is more than replicas × `autoscaling.concurrency`, the extra sequences
wait for a free slot. With the defaults (8 prompts × 4 generations per step, one replica at 8), the
32 sequences run in groups of at most eight. Gym sends one completion per request.

## Pitfalls

- **`snapshot access check failed` at the start of the trainer Job means that the trainer may not
  use the hot-load prefix.** It needs `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` and
  `s3:AbortMultipartUpload` there, from `AllowTrainer` and from its own org access policy. Check
  that `AllowTrainer` names the trainer's principal (`TRAINER_PRINCIPAL_ARN`), and that
  `HOTLOAD_PREFIX` is the deployment's `hotLoad.pathPrefix`. With Workload Identity Federation,
  also check that the cluster's issuer is registered and that the Job runs as
  `TRAINER_SERVICE_ACCOUNT`. `cwtrain hotload check --deployment <id>` runs the same check with the
  credentials of your shell.
- **Token ids in, token ids out.** NeMo-RL trains on exact ids and uses generation log-probs for
  its importance-sampling correction. The backend sends prompts as token ids and asks for
  `return_tokens_as_token_ids`. Fallbacks:
  - If the endpoint rejects a token-id prompt, the backend retries it as text. After a text
    request succeeds, the rest of the run sends text.
  - If the endpoint drops the flag, the backend re-tokenizes the response. It restores the
    end-of-sequence or custom stop token that vLLM leaves out of the text. It keeps log-probs only
    when the endpoint's token strings match the re-tokenized tokens one for one.

  When `coreweave_cfg.require_logprobs` is on, a response without aligned log-probs stops the step,
  instead of feeding zeros to the correction. The config ties this key to
  `loss_fn.use_importance_sampling_correction`.
- **Stop criteria come from the config and the batch.** Every request sends `stop_strings` and
  `stop_token_ids` from `policy.generation`, merged with the batch's per-row stop strings. A
  matched stop string stays in the output, as with NeMo-RL's own backends. The gateway accepts at
  most four stop strings and four stop token ids per request. More configured stops fail at setup.
- **The export must match the deployment's tensor set.** The platform rejects a full snapshot with
  a tied `lm_head.weight` or an adapter tensor that the base checkpoint does not have. So the
  export is filtered to the base checkpoint's names, and the synchronizer stops if a name is
  missing. Dtypes must also match:
  - NeMo-RL's refit generator casts every tensor to the policy dtype. This would turn FP32 tensors
    (for example, MoE router biases) into bf16 and round their values. The export keeps each tensor
    in the base checkpoint's dtype instead. The synchronizer stops before it publishes if a dtype
    or shape differs.
  - The export casts only between F16, BF16, F32 and F64. A quantized base checkpoint (FP8 or
    integer weights with their scales) stops the export, because a plain cast ignores the scales
    and serves wrong weights.
  - The Megatron worker does not apply dtypes. It writes the tensors as Megatron-Bridge exports
    them. If the bridge rounds FP32 tensors, the synchronizer's check stops the refit, so no cast
    hides the rounding.
- **The worker extension must be importable in the worker venv.** Ray imports the worker class in
  NeMo-RL's per-class venv before any configured env vars apply. So `PYTHONPATH` does not reach
  that import. The launch script installs the wheel without dependencies into the container's
  prebuilt worker venvs (`$NEMO_RL_VENV_DIR/*`). A venv that NeMo-RL builds fresh needs the same
  step.
- **Rank 0 and the driver share `state_dir`.** On one node, an emptyDir is enough. Across nodes, use
  a shared filesystem, or pin the driver to rank 0's node.
- **Do not `uv run` after you install extra packages.** `uv run` syncs the environment back to the
  lockfile. Call the venv's interpreter directly, as the launch script does.
- **Turn off Nemotron thinking** (`tokenizer.chat_template_kwargs: {enable_thinking: false}`).
  Otherwise the responses fill the token budget before the boxed answer.
- **A traceback at exit is not a failed run.** NeMo-RL v0.7.0 shuts down Ray before the policy
  object's destructor runs. So the last lines of a successful run can be `Exception ignored in
  Policy.__del__ ... PythonFinalizationError`. The process still exits 0. Judge the run by the last
  `[cwtrain] hot-load` line and the Job status.

## Failure handling

See [Failure handling](../miles/README.md#failure-handling) in the miles recipe, with two
differences. NeMo-RL keeps no weight mirror, so there is no `MirrorError`. Its synchronizer chains
each delta onto the last version it published.

One NeMo-RL-specific check: if the log never shows `Using CoreWeave hot-load deployment for
generation`, the patch was not applied to the container's checkout. NeMo-RL then fell through to a
GPU backend.

## NeMo Gym rollouts

To use Gym for the Nemotron 3.5 Lightning DAPO recipe:

1. Add this Hydra override to `EXTRA_OVERRIDES`:

   ```sh
   EXTRA_OVERRIDES="env.should_use_nemo_gym=true"
   ```

2. Rebuild the kit, so that it includes both patches.

The launcher then selects NeMo-RL's Gym runner. `cwtrain` converts the DAPO questions and answers
into a `NemoGymDataset` under `/work/hotload/nemo-gym`. It starts the
`math_with_judge_simple_agent` with math-verify scoring (the LLM judge is off). Gym uses the
configured CoreWeave rollout URL, served model and mounted API key file. The default, `false`,
keeps NeMo-RL's math environment.

Training uses NeMo-RL's bundled Gym revision and actor environments. Gym owns sampling and rewards.
NeMo-RL owns optimization. cwtrain keeps the blocking checkpoint hot-load between batches.
Asynchronous GRPO and nonblocking hot-load are rejected. The Gym actor resolves the credentials
itself, so they are not in its serialized config or Ray runtime environment.

The cwtrain Gym adapter formats prompts with the staged trainer tokenizer. It sends integer token
IDs to CoreWeave's `/v1/completions` endpoint. It requires exact generated token IDs and
sampled-token log-probs, including the sampled stop token, and it checks the returned prompt count.
It does not need a `/tokenize` endpoint or text re-tokenization.

Limits of the adapter:

- It supports one text-only completion per request. It rejects tool calls, structured outputs and
  multiple choices per request.
- The flag-only preset is single-turn DAPO math. For custom text environments, provide an explicit
  `NemoGymDataset`, a Gym data processor and `env.nemo_gym.config_paths`.
- Multi-turn trajectories also require exact preservation of the token prefix.

### Validation

An end-to-end run with Nemotron 3.5 Lightning BF16 completed two GRPO steps of 32 Gym trajectories
each. It used eight trainer H100s and a two-H100 inference replica. The first full checkpoint and
the trained delta between steps both hot-loaded. The serving replica's identity was verified before
sampling. The Job exited successfully, and generation still worked after the trained delta was
applied. This is a smoke test of training and weight propagation, not a model-quality benchmark.

For a short validation run, add `++data.train.max_samples=32` and `grpo.max_num_steps=2` to
`EXTRA_OVERRIDES`. The sample limit is optional. Without it, the launcher converts the complete
published DAPO train split. On clusters with spot nodes, the Job also needs a toleration for
`qos.coreweave.cloud/interruptable` with effect `NoExecute`.

For a Gym validation run, use your hot-load bucket and a unique prefix under `hotload/`, such as
`hotload/nemo-gym-validation/<run-id>`. The bucket policy lets the trainer write only under
`hotload/`:

1. Set the temporary deployment's `hotLoad.bucket` and `hotLoad.pathPrefix` to match. The base
   model can stay in its existing source bucket.
2. Before you allocate compute, run `cwtrain hotload check --deployment <id>` with the trainer's
   credentials. It lists, reads and starts a multipart upload under the prefix, then aborts it, so
   it leaves no object behind. `AllowTrainer` must cover the prefix.
3. Make sure that the inference service can read the prefix. The example policy lets it read the
   whole bucket.
4. After the run, use the admin's credentials to remove only that run's objects and any incomplete
   multipart uploads. Then remove its temporary deployment and gateway.

Do not create or delete a bucket for validation.
