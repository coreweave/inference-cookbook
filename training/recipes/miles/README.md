# Recipe: miles RL with rollouts on a CoreWeave hot-load deployment

Train with [miles](https://github.com/radixark/miles) and serve rollouts from a CoreWeave
Dedicated Inference deployment. The deployment receives every weight update by checkpoint
hot-load. The trainer owns no inference engines:

1. miles writes a versioned weight directory after each sync.
2. The `cwtrain` post-write hook uploads it and triggers a hot-load.
3. The deployment swaps the weights in place on all replicas.
4. The next rollout batch samples the new weights through the deployment's OpenAI-compatible
   endpoint.

![One training step: the trainer writes weight_vNNNNNN, the cwtrain hook uploads it to Object Storage, creates a hot-load and polls until every replica is READY; the next rollout batch samples fresh weights from the OpenAI-compatible endpoint.](../../../docs/hotload-flow.drawio.svg)

The same loop with slime is in [`../slime/`](../slime/README.md).

## What you need

Follow the Dedicated Inference [getting-started guide](https://docs.coreweave.com/products/inference/getting-started)
for the account prerequisites (Inference access, GPU quota, an API token with the Inference Admin
role), the gateway, and the bucket policy. Then, specific to this recipe:

- **A hot-load-enabled deployment.** You enable hot-load when you create the deployment: engine
  `dynamo-vllm` and a `hotLoad` section that names the snapshot bucket and path prefix. CoreWeave
  enables hot-load per organization. Until it is enabled for yours, the create fails with
  `hot-load is not enabled for this organization`; ask CoreWeave support and give your
  organization ID. The [RL Rollouts docs](https://docs.coreweave.com/products/inference/rl-rollouts)
  describe the feature. [`deployment.example.json`](deployment.example.json) is a complete request
  body for `POST /v1alpha1/inference/deployments` (see
  [Creating the deployment](#creating-the-deployment)). The
  [library README](../../../libs/cwtrain/README.md#prerequisites) describes the prerequisites and
  the API. The deployment's `autoscaling.concurrency` sets how many sequences each replica decodes
  at once (see [Concurrency](#concurrency)).
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
- **The trainer image with `cwtrain` installed.** `k8s/make-kit.sh` bundles the wheel from
  `libs/cwtrain/dist` and the scripts into a ConfigMap. `run-trainer.sh` installs the wheel at
  start.
- **The base model, staged once.** Stage it into the bucket with
  [`stage-model-job.yaml`](../../../k8s/stage-model-job.yaml). The
  deployment loads it from there. When `BASE_S3_URI` is set, the trainer fetches the same objects,
  so the run starts from exactly the weights the deployment serves. Otherwise the trainer downloads
  `HF_MODEL` from Hugging Face. In both cases the hook publishes the trainer's copy as the run's
  chain root (`<run>-v000000`) before the first delta. So the two copies must have the same
  architecture and tensor layout.

## Configure the hook

Set these in the trainer workers' environment (rank 0 does the publishing):

```bash
export CWTRAIN_HOTLOAD_DEPLOYMENT_ID=<deployment uuid>
export CWTRAIN_HOTLOAD_RUN_ID=qwen3-4b-grpo-$(date -u +%Y%m%dT%H%M%SZ)   # unique per run
export CWTRAIN_HOTLOAD_FULL_EVERY=20                                    # the default; at most 64
export COREWEAVE_API_TOKEN=...
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...                   # AI Object Storage
export AWS_REGION=<bucket zone>                                         # e.g. US-EAST-04A
```

Optional settings:

- `CWTRAIN_HOTLOAD_WAIT=0`: do not block until the hot-load completes.
- `CWTRAIN_HOTLOAD_TIMEOUT`: seconds one hot-load may take (default 1800).
- `CWTRAIN_HOTLOAD_VERIFY=1`: re-check every replica after each hot-load.

The hook's module docstring lists all its settings (`cwtrain/integrations/slime.py`; miles uses
slime's hook contract). In the Kubernetes run, `run.env` sets these and the prompt-cache policy
(`HOTLOAD_WAIT`, `HOTLOAD_TIMEOUT`, `HOTLOAD_VERIFY`, `HOTLOAD_PROMPT_CACHE_POLICY`).

With `train.py` (the default), the loop is synchronous: each rollout round samples the weights
that the hook just hot-loaded. So, unless you set it, the launcher sets
`CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY=reset_all`. This asks the platform to drop the KV cache
computed under the previous weights, instead of reusing cached prompt prefixes after the swap.
`train_async.py` generates one step ahead and keeps the platform default, `preserve`.

Snapshot identities are `<RUN_ID>-vNNNNNN`. The platform keys weight versions by identity, so
never reuse a run ID for different weights.

## miles flags

miles uses slime's post-write hook contract, with different names for the disk-delta flags. miles
also runs the weight sync when it owns no engines. So this topology has **no SGLang engines at
all**:

- The deployment is the rollout engine.
- The disk-delta publisher runs on the trainer.
- The hook hot-loads each version.

miles never pauses, reloads or resumes an engine. The hot-load API runs the swap on all replicas.

```bash
python train.py ... \
  --rollout-num-gpus 0 \
  --rollout-function-path coreweave_rollout.CoreWeaveRolloutFn \
  --update-weight-transfer-mode disk-delta \
  --update-weight-disk-dir /work/deltas \
  --update-weight-local-checkpoint-dir /work/local-ckpt \
  --update-weight-delta-encoding xor \
  --update-weight-delta-checksum adler32 \
  --custom-update-weight-post-write-path cwtrain.integrations.slime.hook
```

Delta mode publishes zstd XOR deltas with Adler-32 checksums. This is the only delta format the
hot-load API accepts. The hook rejects any other encoding or checksum before it uploads.
`disk-delta` does not work with `--colocate`.

[`coreweave_rollout.py`](coreweave_rollout.py) is the rollout function. It is a thin miles adapter
over `cwtrain.integrations.rollout`:

- It sends one `/v1/completions` request per GRPO group (`n` choices) to the deployment, through
  the deployment's gateway.
- The prompt goes out as the token ids that the trainer scores.
- Responses come back as the exact sampled token ids. Only when the endpoint does not report them
  does it re-tokenize the text.
- Rewards come from miles' own `--rm-type`.
- When the endpoint's sampled-token log-probs align with the response tokens, including the token
  that ended the response, it attaches them as `rollout_log_probs`. Otherwise it leaves them unset.
  The trainer recomputes log-probs itself in both cases.

Configure it with `CWTRAIN_ROLLOUT_BASE_URL`, `CWTRAIN_ROLLOUT_MODEL` and, optionally,
`CWTRAIN_ROLLOUT_API_KEY_FILE`. The hook blocks until every replica serves the new version, so the
next rollout is on-policy without an engine handshake.

## Hooking into miles' per-model scripts

miles ships one launch script per model under `scripts/` (`run_qwen3_4b.py`,
`run_qwen3_30b_a3b.py`, ...). Each is a small typer CLI. It composes argument groups (checkpoint,
rollout, optimizer, GRPO, performance, parallelism) and passes them to
`miles.utils.external_utils.command_utils`. That module's `convert_checkpoint` and `execute_train`
do the Megatron conversion, start Ray and submit `train.py`. The architecture flags come from
`scripts/models/<type>.py`.

The scripts take `--extra-args`. But they own SGLang engines, so they also hardcode `--colocate`
and the `--sglang-*` flags, and `--extra-args` cannot remove a `store_true` flag. So the recipe
does not wrap a stock script. [`run_hotload.py`](run_hotload.py) is a script in the same style. It
uses the same helpers and model definitions, and replaces only the engine-side group:

| Stock `scripts/run_<model>.py` | `run_hotload.py` |
| --- | --- |
| `--colocate`, `--sglang-*`, `--rollout-num-gpus-per-engine` | `--rollout-num-gpus 0 --rollout-function-path coreweave_rollout.CoreWeaveRolloutFn` |
| NCCL weight sync into its own engines | `--update-weight-transfer-mode disk-delta ... --custom-update-weight-post-write-path cwtrain.integrations.slime.hook` |
| model, data, conversion via `command_utils` | same calls |
| `U.execute_train(train_args, ...)` | same call, plus the `CWTRAIN_*` / `COREWEAVE_*` / `AWS_*` / `CW_S3_*` environment forwarded to the Ray workers |

Any model with a `scripts/models/<type>.py` works by name (`--model-name Qwen3-8B --tp 2`).
`--extra-args` appends any other flags. To edit a stock script instead, change three things:

1. Remove `--colocate` and the `sglang_args` group.
2. Add the CoreWeave group above.
3. Forward the environment through `execute_train`'s `extra_env_vars`.

## Versions

| Component | Version |
| --- | --- |
| miles | [`v0.1.1`](https://github.com/radixark/miles/releases/tag/v0.1.1): the trainer image `radixark/miles:v0.1.1` (amd64 and arm64), pinned by digest in `run.env.example`. `run-trainer.sh` runs the miles checkout inside that image. |

Releases after v0.1.1 keep the launcher helpers that `run_hotload.py` calls (`convert_checkpoint`,
`execute_train`, `hf_download_dataset`, `exec_command_cpu`) in `command_utils.legacy`, not in
`command_utils`. The script takes them from either module. When neither has them, it stops before
any download and names the pinned image. A newer image works if it accepts the flags above.

## Synchronous or one-step-ahead: `train.py` vs `train_async.py`

`--train-script train_async.py` (the recipe's `TRAIN_SCRIPT`) selects miles' asynchronous loop:

1. It generates rollout `i+1` while step `i` trains.
2. Before the weight update, it drains the in-flight generation, so no request spans two weight
   versions.
3. It calls the same `update_weights` path, so the hook hot-loads as in the synchronous loop.

The data is different: the policy that sampled each rollout is one update behind the policy being
trained. miles requires an explicit behaviour-policy correction, which `--async-correction`
selects:

| Value | Flag | Notes |
| --- | --- | --- |
| `keep_old_actor` (default) | `--keep-old-actor` | Recomputes the ratio denominator with the weights that generated the rollout. Costs one extra copy of the actor weights on the trainer. Needs nothing from the endpoint. |
| `tis` | `--use-tis` | Truncated importance sampling against the endpoint's sampled-token log-probs. Every sample of every batch must carry aligned `rollout_log_probs`; batches where the response tokens could not be matched to the endpoint's at every position lose them. |
| `none` | | Plain GRPO on one-step-stale samples. |

On a hot-load deployment, the gain is that generation overlaps the training step. The hot-load
itself still runs between steps. slime's upstream tree does not ship a `train_async.py`. Its
asynchronous mode is the fully-async rollout worker, which this cookbook does not cover.

## Running it on Kubernetes

This recipe's `k8s/` directory, with the files that the training recipes share in
[`k8s/`](../../../k8s/README.md) at the repository root, is a complete run of this topology on a
Kubernetes cluster with a GPU node of `INSTANCE_TYPE`, such as CoreWeave Kubernetes Service (CKS).
You need these tools:

- `kubectl`
- `envsubst` (GNU gettext; `brew install gettext` on macOS)
- `uv`
- the `aws` CLI
- the `cwtrain` CLI (from the repository root, run `uv tool install ./libs/cwtrain`)

Work in this recipe's `k8s/` directory, with a filled-in copy of `run.env.example` loaded into your
shell:

```bash
cd training/recipes/miles/k8s
K=../../../../k8s                # the shared files: k8s/ at the repository root
[ -e run.env ] || cp run.env.example run.env   # once; fill in every <placeholder>, render.sh refuses one
set -a; . ./run.env; set +a      # the commands below use its values
CTX=$(kubectl config current-context)
```

| Step | Command |
| --- | --- |
| 1. Namespace | `"$K/render.sh" run.env "$K/namespace.yaml" \| kubectl apply -f -` |
| 2. Trainer service account and object storage Secrets (AWS-style `config` + `credentials` files) | `kubectl -n "$NAMESPACE" create serviceaccount "$TRAINER_SERVICE_ACCOUNT"`, then `WORKLOAD_IDENTITY=1 AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$STORAGE_SECRET"` (with a trainer key instead, export it as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` and leave out `WORKLOAD_IDENTITY=1`); then, with the admin's key exported, `AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$ADMIN_STORAGE_SECRET"` |
| 3. Wheel + scripts ConfigMap | `(cd ../../../../libs/cwtrain && uv build --wheel) && ./make-kit.sh "$CTX" "$NAMESPACE" "$KIT_NAME"` |
| 4. Bucket + policy (skip if done with `aws s3api`) | `"$K/render.sh" run.env "$K/make-bucket-job.yaml" \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m job/cwtrain-make-bucket`; it is done when `kubectl -n "$NAMESPACE" get job cwtrain-make-bucket` shows COMPLETIONS `1/1` |
| 5. Stage the base model into the bucket | `"$K/render.sh" run.env "$K/stage-model-job.yaml" \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m "job/cwtrain-stage-$JOB_NAME"`; it is done when `kubectl -n "$NAMESPACE" get job "cwtrain-stage-$JOB_NAME"` shows COMPLETIONS `1/1` (it retries once, so follow the logs again while it shows `0/1`) |
| 6. Create the deployment | [below](#creating-the-deployment); put its `id` and `rollout_base_url` in `run.env` as `DEPLOYMENT_ID` and `ROLLOUT_BASE_URL` |
| 7. Inference API key Secret | `kubectl -n "$NAMESPACE" create secret generic "$INFERENCE_KEY_SECRET" --from-file=api_key=<file>` |
| 8. CoreWeave API token Secret | `kubectl -n "$NAMESPACE" create secret generic "$API_TOKEN_SECRET" --from-file=token=<file>` |
| 9. Trainer | `"$K/render.sh" run.env trainer-job.yaml \| kubectl apply -f -` |

Step 7's key is the bearer token that the deployment's gateway accepts for inference requests.
When the gateway uses CoreWeave authentication, this is a CoreWeave API token. Step 8's token
creates hot-loads, so it needs the Inference Admin role. One token can do both. The trainer pod
does not start until both Secrets exist. Reading the tokens from files keeps them out of your shell
history.

`trainer-job.yaml` runs `run-trainer.sh`, which does these steps:

1. Install the wheel.
2. Fetch the base model from the bucket, or from Hugging Face when `BASE_S3_URI` is empty.
3. Start `run_hotload.py`. It fetches the dapo-math-17k prompts, converts the checkpoint to
   Megatron `torch_dist`, starts Ray, and submits `train.py` or `train_async.py` with the flags
   above. The defaults are GRPO, Qwen3-4B, and TP=2 on two GPUs.

### Creating the deployment

Copy [`deployment.example.json`](deployment.example.json) to `deployment.run.json` in this recipe's
`k8s/` (git-ignored). Fill in the gateway id and the bucket. The example pins `dynamo-vllm` to
`0.30.0+crwv.5`, which has hot-load support; its ModelExpress client is built into the engine. Put your API token in `COREWEAVE_API_TOKEN` (or `CW_API_TOKEN`, as the
CoreWeave docs set it). Then `cwtrain deployment parameters` lists the runtime versions, instance
types and engine options that your organization can use. `--gateway <gateway id>` limits the list
to that gateway's zone:

```bash
cwtrain deployment parameters
[ -e deployment.run.json ] || cp ../deployment.example.json deployment.run.json   # then fill it in
```

Then create it from the filled-in file, and wait until it serves:

```bash
cwtrain deployment create --file deployment.run.json --wait
```

`create` refuses a file that still holds a `<placeholder>`. When the name is taken, it names the
deployment that has it. With `--wait`, it returns when the deployment is READY. This takes several
minutes, and longer for large models. Without `--wait`, it returns at once, and
`cwtrain deployment get "$DEPLOYMENT_ID"` shows the status later.

In both cases it prints the deployment as JSON, also when the wait fails. **If the wait fails, the
deployment still exists and holds its GPUs until you delete it** (see
[Cleaning up](#cleaning-up)).

Then:

1. Put its `id` in `run.env` as `DEPLOYMENT_ID`.
2. Put its `rollout_base_url` in `run.env` as `ROLLOUT_BASE_URL`. A `null` value means that the
   gateway could not be read yet; `cwtrain deployment get "$DEPLOYMENT_ID"` shows it later.
3. Check that `ROLLOUT_MODEL` matches its `model_name`.
4. Load `run.env` again: `set -a; . ./run.env; set +a`.

The trainer also waits for READY before its first hot-load, within `HOTLOAD_TIMEOUT`.

The example sets two engine options, `gpu-memory-utilization` and `max-model-len`. The example in
the [RL Rollouts guide](https://docs.coreweave.com/products/inference/rl-rollouts) sets none. Add
only engine options that `cwtrain deployment parameters` lists.

`deployment.run.json` is the request body of `POST /v1alpha1/inference/deployments`. So you can
also create the same deployment with the `curl` request in the
[RL Rollouts guide](https://docs.coreweave.com/products/inference/rl-rollouts) and
`-d @deployment.run.json`, or in the **Hot loading** section of the Cloud Console's create drawer.

### Watching and re-running

In a new shell, run the setup block in [Running it on Kubernetes](#running-it-on-kubernetes) again
first: these commands and step 9 use its `K` and the values in `run.env`. The block keeps your
filled-in `run.env`.

```bash
kubectl -n "$NAMESPACE" logs -f "job/cwtrain-$JOB_NAME"   # the trainer, including cwtrain's lines
cwtrain hotload list --deployment "$DEPLOYMENT_ID"         # one hot-load per published version
```

You cannot change a Job while it exists. Before the next run, delete the finished Job with
`kubectl -n "$NAMESPACE" delete job "cwtrain-$JOB_NAME"`, then render and apply step 9 again.
The `RUN_ID` in `run.env.example` uses the time of each render plus a random number. So the new run
publishes under new identities, as it must (see [Resuming a run](#resuming-a-run)). Your shell's
`$RUN_ID` is from when you loaded `run.env`, not from the render. `cwtrain hotload list` shows the
identities that the Job uses.

### Cleaning up

The deployment holds its GPUs until you delete it. Run this block with the kubectl context that you
used for the steps above (`kubectl config current-context`), and with the admin's object-storage
credentials. Only the admin may delete from the bucket:

```bash
( set -eu   # run from anywhere in the repository; stops at the first unset value or failed step
  cd "$(git rev-parse --show-toplevel)/training/recipes/miles/k8s"
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
print(" ".join(sorted({re.sub(r"-v[0-9]+$", "", h["spec"]["identity"]) for h in json.load(sys.stdin)})))')
  cwtrain deployment delete "$DEPLOYMENT_ID" --ignore-not-found
  kubectl -n "$NAMESPACE" delete job "cwtrain-$JOB_NAME" "cwtrain-stage-$JOB_NAME" cwtrain-make-bucket --ignore-not-found
  kubectl -n "$NAMESPACE" delete secret "$STORAGE_SECRET" "$ADMIN_STORAGE_SECRET" "$INFERENCE_KEY_SECRET" \
    "$API_TOKEN_SECRET" --ignore-not-found
  kubectl -n "$NAMESPACE" delete configmap "$KIT_NAME" --ignore-not-found
  [ -z "$TRAINER_SERVICE_ACCOUNT" ] || kubectl -n "$NAMESPACE" delete serviceaccount "$TRAINER_SERVICE_ACCOUNT" --ignore-not-found
  [ -n "$runs" ] || echo "found no hot-loads of $DEPLOYMENT_ID, so nothing under $HOTLOAD_PREFIX/ is removed" >&2
  for run in $runs; do
    aws s3 rm --recursive "s3://$BUCKET/$HOTLOAD_PREFIX/" --exclude '*' --include "$run-v*" \
      --include "_manifests/$run-v*" --endpoint-url https://cwobject.com --region "$BUCKET_ZONE"
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
defaults are the same in every recipe, so clean up after the last recipe. If the recipe created the
namespace and nothing else runs there, delete it too: `kubectl delete namespace "$NAMESPACE"`.

## Concurrency

Two settings decide how many sequences are sampled at once:

- **The deployment's `autoscaling.concurrency`**, a per-replica limit. The platform passes it to
  vLLM as `--max-num-seqs`. So a replica decodes at most that many sequences at once, and later
  requests wait in a queue. vLLM never schedules more than its KV cache holds, so a higher value
  only moves the wait into the engine. [`deployment.example.json`](deployment.example.json) sets
  256.
- **`ROLLOUT_CONCURRENCY`** in `run.env` (default 16): the number of GRPO group requests that the
  trainer keeps in flight. The Job passes it as `CWTRAIN_ROLLOUT_CONCURRENCY`. Each request asks
  for `--n-samples-per-prompt` completions. So the trainer has up to `ROLLOUT_CONCURRENCY` times
  that many sequences in flight.

When the trainer sends more sequences than replicas × `autoscaling.concurrency`, the extra
sequences wait for a free slot. With the defaults (8 prompts × 4 samples per step, two replicas at
256), a whole step runs at once.

## Sequence of one training step (delta mode)

| Step | Who | What |
| --- | --- | --- |
| baseline | miles rank 0 | Captures `--hf-checkpoint` bytes as the delta baseline and calls the hook with the delta root |
| baseline | hook | Publishes `--hf-checkpoint` as full snapshot `<run>-v000000`, unless a completed hot-load with that identity already exists; creates the weight mirror |
| sync N | miles ranks | Diff against the previous sync, write `weight_v00000N/` (shards + index with `base_version`) |
| sync N | hook (rank 0) | Apply the delta to the mirror and check every changed tensor's Adler-32; upload to `<prefix>/<run>-v00000N/`, create an incremental hot-load with predecessor `<run>-v00000(N-1)`, and wait until the hot-load completes (every replica READY on `<run>-v00000N`) |
| sync N, N % FULL_EVERY == 0 | hook (rank 0) | Same, but publish a **full** snapshot rebuilt from the mirror (with per-tensor checksums) so the chain restarts |
| sync N | miles | Barrier, then continues with the next rollout batch |

The hook blocks rank 0 before the barrier, so training pauses until the deployment serves the new
version. Set `CWTRAIN_HOTLOAD_WAIT=0` to give up that guarantee for more throughput.

## Full snapshots every N versions, and what is verified

A delta chain must not grow without limit, for three reasons:

- A replica that starts with no peer to copy the weights from replays the whole chain from object
  storage. It can replay at most 63 deltas.
- The API rejects a chain longer than 128 deltas.
- One bad delta corrupts every version after it.

So, by default, the hook publishes a full checkpoint every 20th version
(`CWTRAIN_HOTLOAD_FULL_EVERY`, the recipe's `FULL_EVERY`, at most 64):

- At the baseline call, the hook copies `--hf-checkpoint` to `<delta dir>-hotload/mirror/` and
  computes the checksum of every tensor. This **mirror** is a byte-exact copy of what the
  deployment serves. The trainer itself has no such copy, because miles writes only XOR deltas.
- Every sync applies the new `weight_vNNNNNN` delta to the mirror in place. For each tensor, the
  hook decompresses the zstd data, XORs it onto the tensor's bytes, and compares the Adler-32 of
  the result with the checksum in the shard header. It writes a tensor only when the checksums
  match. A mismatch raises an error before anything is uploaded, and stops the run.
- Every `FULL_EVERY`-th version, the hook writes the mirror out as a full HF checkpoint and
  publishes it as a **full** snapshot under the same identity scheme. The shards carry each
  tensor's checksum in `__metadata__`, and the index says `checksum_format: adler32`. The next
  delta chains onto this snapshot.
- Every version also gets a manifest of all tensor checksums, at
  `<delta dir>-hotload/manifests/<identity>.json` and
  `s3://<bucket>/<prefix>/_manifests/<identity>.json`.

What checks the weights that end up in the engine:

| Where | Check |
| --- | --- |
| Trainer, mirror | Adler-32 of every changed tensor after applying the delta equals the trainer's checksum. Proves the uploaded delta reconstructs the trainer's exact bytes. |
| Engine, install from object storage | The engine applies the delta the same way and verifies the same per-tensor Adler-32; for a full snapshot with `checksum_format` it verifies every tensor. Mismatch = failed hot-load, never a silent load. |
| Engine, install from a peer | Replicas that receive the version from another replica over NIXL check each tensor's name, size and dtype against the source. They do not recompute checksums. |
| API, the hot-load itself | The platform marks a replica READY only after reading the engine's own weight version back, and completes the hot-load only when every replica is READY. `CWTRAIN_HOTLOAD_VERIFY=1` re-reads the replicas of the completed hot-load anyway (all must be READY on the new identity, or `HotLoadVerificationError` stops the run); it is off by default. |

Cost:

- **Disk.** The mirror is a second copy of the checkpoint on the trainer's local disk. Each chain
  cut stages a third copy before the upload. Plan for about twice the model's size in free disk.
- **Launch time.** The hook builds the mirror (copy and checksums) at every launch.
- **Sync time.** Every sync reads and rewrites the changed tensors, in parallel across tensors.
  Every N-th version adds a full upload and a full hot-load.

On large models, the copies and a full hot-load each take minutes. With `CWTRAIN_HOTLOAD_WAIT=1`,
training waits for every hot-load. `CWTRAIN_HOTLOAD_MIRROR=0` turns off the mirror. It requires
`FULL_EVERY=0`, which never cuts the chain.

## Resuming a run

On restart, miles captures its baseline again, so it calls the hook with the delta root again. The
hook looks for a completed `<run>-v000000` and skips the upload when it finds one. Version numbers
also restart at 1: miles clears the delta directory, and the deltas are again relative to
`--hf-checkpoint`. So a resumed run must publish under a *new* run ID. With the same run ID, the
hook would try to reuse identities that already exist with different bytes. Always start a
resumed run with a new `CWTRAIN_HOTLOAD_RUN_ID`.

## Pitfalls

- **`snapshot access check failed` at the start of the trainer Job means that the trainer may not
  use the hot-load prefix.** It needs `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` and
  `s3:AbortMultipartUpload` there, from `AllowTrainer` and from its own org access policy. Check
  that `AllowTrainer` names the trainer's principal (`TRAINER_PRINCIPAL_ARN`), and that
  `HOTLOAD_PREFIX` is the deployment's `hotLoad.pathPrefix`. With Workload Identity Federation,
  also check that the cluster's issuer is registered and that the Job runs as
  `TRAINER_SERVICE_ACCOUNT`. `cwtrain hotload check --deployment <id>` runs the same check with the
  credentials of your shell.
- **Turn off thinking for Qwen3 rollouts.** With the default chat template, every response fills
  the token budget with `<think>` reasoning. All rewards are 0, the gradient is 0, and the first
  delta is empty. `--apply-chat-template-kwargs '{"enable_thinking":false}'` fixes this.
  `run.env.example` sets it in `EXTRA_TRAIN_ARGS`.
- **Pick a verifier that can see the answer.** miles' `deepscaler` reward grades only the text
  after a `</think>` marker. So it returns 0 for every response when thinking is off, or when
  thinking is on but truncated. `--rm-type math` grades the boxed answer wherever it is.
- **The uploaded index names identities, not counters.** miles writes `000001` / `000000` as the
  index's `metadata.version` and `metadata.base_version`. The platform takes the delta's identity
  and its predecessor from the hot-load request, not from these fields. `upload_snapshot` still
  rewrites them to the snapshot identities during the upload, so the index in the bucket agrees
  with the request. It uploads the shards and their per-tensor checksums unchanged.
- **A sync that changes nothing publishes nothing.** miles still writes `weight_vNNNNNN/`, with an
  index and no shards. The hook logs it and reports `hotload/skipped_empty`. The skipped version is
  byte-identical to the last one, so the hook chains the next delta onto the last version that the
  deployment received.
- **A replica that a failed replay left on `unknown` recovers on the next hot-load.** After a
  rejected delta, a replica can report weight version `unknown`. It stays paused until a later
  hot-load succeeds. Fix the cause, then create the hot-load again (see
  [Failure handling](#failure-handling)). `cwtrain hotload status <id>` shows each replica's state
  and error. If hot-loads still stall with `chain root unresolved`, the platform cannot tell which
  version a replica serves. The deployment's pods run on CoreWeave's side, so send CoreWeave
  support the deployment id and the hot-load id.
- **A missing bucket policy is the most common reason that weights fail to load.** The model
  download at deployment start and every hot-load read the bucket as the inference service account.

## Failure handling

- A snapshot that fails validation (missing files, unsupported checksum) raises an error before
  the upload.
- A hot-load that ends `FAILED` raises `HotLoadFailedError` with per-replica reasons. Nothing
  retries automatically: the run stops so that it does not continue off-policy. The snapshot is
  already in the bucket. When you have fixed the cause, create its hot-load again. For a delta (the
  default):
  `cwtrain hotload trigger --deployment <id> --identity <run>-vNNNNNN --previous-identity <run>-v(NNNNNN-1)`.
  For a full snapshot, leave out `--previous-identity`.
- Delta chains have limits. The API rejects a chain longer than 128 deltas (`resolved snapshot
  chain is too long`). A replica that starts with no peer to copy from can replay only 63. The
  default `FULL_EVERY=20` stays below both limits. If you set it to 0, set it back so that the hook
  restarts the chain with a full snapshot on schedule.
- With `CWTRAIN_HOTLOAD_VERIFY=1`, a `HotLoadVerificationError` means that the API marked the
  hot-load COMPLETED, but a replica in its list does not report the new identity. Check
  `cwtrain hotload status <id>`. A replica stuck on `unknown` recovers on the next successful
  hot-load (see [Pitfalls](#pitfalls)).
- A `MirrorError` means that a delta the trainer wrote does not reconstruct to the trainer's
  checksums. Causes include disk corruption, or a delta written against a different base. Nothing
  was uploaded, and the deployment still serves the previous version.

## Verifying by hand

```bash
cwtrain hotload list --deployment <uuid>                 # newest first, with state
cwtrain hotload status <hot-load uuid>                   # replicas and their current identity
cwtrain hotload inspect /work/deltas/weight_v000003      # what the hook would publish
```

With the mirror on, `<delta dir>-hotload/manifests/<identity>.json` lists the Adler-32 of every
tensor for that version. The bucket has a copy under `<prefix>/_manifests/`. These are the
checksums of the engine's copy of that version. The engine checks them when it installs a version
from object storage. On every hot-load, the platform checks that each replica reports the new
version.
