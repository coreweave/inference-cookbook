# Recipe: Axolotl SFT, published to a CoreWeave hot-load deployment

Fine-tune with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), then publish the result into
a CoreWeave Dedicated Inference deployment of the same base model by checkpoint hot-load. The
deployment serves the fine-tuned weights without a redeploy, so you can query and evaluate them
right away:

1. The launcher stages the base model and runs stock `axolotl train`, and `axolotl merge-lora`
   after a LoRA run.
2. It checks that the result holds the base model's tensors, and writes it as one full snapshot
   with a checksum per tensor.
3. `cwtrain` uploads the snapshot and creates a hot-load.
4. The deployment swaps the weights in place on all replicas.

![The hot-load flow, which this recipe runs once, after training: cwtrain uploads the weights to Object Storage, creates a hot-load and polls until every replica is READY; requests to the OpenAI-compatible endpoint then get the new weights.](../../../docs/hotload-flow.drawio.svg)

The snapshot is also a plain Hugging Face checkpoint in your bucket, so an RL recipe can start
from it (see [SFT, then RL](#sft-then-rl)).

## What the deployment can serve

A hot-load replaces the weights of the model that a deployment serves. So the snapshot must fit
the deployment, and the launcher checks this before it uploads anything:

1. **The base model's tensors.** The result must have exactly the base checkpoint's tensor names,
   dtypes and shapes. One extra is allowed and left out of the snapshot: a tied `lm_head.weight`
   that equals `model.embed_tokens.weight` byte for byte. The FSDP2 final save can write it a
   second time; the base holds it once. A run that adds tokens, resizes the embeddings or changes the
   dtype is refused.
2. **Full weights.** A LoRA run is merged into the base model first (see [LoRA](#lora)).
3. **The base model's tokenizer and chat template.** Everything but the weights stays as the
   deployment started with it: the snapshot carries the base model's `config.json`, generation
   config and tokenizer files, training uses the base model's chat template, and a run whose saved
   tokenizer gained tokens is refused. The run's own `config.json` is written by a newer
   transformers, in a layout that older readers misread.
4. **A new identity for every run.** The snapshot is `<RUN_ID>-v<MAX_STEPS, six digits>`,
   `<RUN_ID>-v000030` with the defaults. Every recipe names its snapshots `<run>-vNNNNNN`, so any
   recipe's cleanup block finds this one.
5. **No prompt cache from the base weights.** Unless you set `HOTLOAD_PROMPT_CACHE_POLICY`, the
   launcher asks the platform to drop the KV cache computed under the previous weights
   (`reset_all`), so that a request right after the swap does not reuse cached prompt prefixes.
6. **RL Rollouts for your organization**, and a deployment created with engine `dynamo-vllm` and a
   `hotLoad` section (see [What you need](#what-you-need)).

The snapshot is a full checkpoint, never a delta: the first snapshot that a deployment receives
must be full, because no delta can chain onto its base weights.

## Versions

| Component | Version |
| --- | --- |
| Axolotl | [`v0.20.0`](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.20.0): the trainer image `axolotlai/axolotl:0.20.0-py3.12-cu130-2.13.0@sha256:80ab8e2eb7a6f1987656a5b8c09e4c317e7481bcfb543674fd7414ef61a73c4b` (amd64 and arm64), `TRAINER_IMAGE` in `run.env.example` |
| Inference engine | `dynamo-vllm` `0.30.0+crwv.5` |

This configuration is not yet validated against a live deployment.

## What you need

Follow the Dedicated Inference [getting-started guide](https://docs.coreweave.com/products/inference/getting-started)
for the account prerequisites (Inference access, GPU quota, an API token with the Inference Admin
role), the gateway, and the bucket policy. Then, specific to this recipe:

- **A hot-load-enabled deployment of the base model.** You enable hot-load when you create the
  deployment: engine `dynamo-vllm` and a `hotLoad` section that names the snapshot bucket and path
  prefix. CoreWeave enables hot-load per organization. Until it is enabled for yours, the create
  fails with `hot-load is not enabled for this organization`; ask CoreWeave support and give your
  organization ID. The [RL Rollouts docs](https://docs.coreweave.com/products/inference/rl-rollouts)
  describe the feature. [`deployment.example.json`](deployment.example.json) is a complete request
  body for `POST /v1alpha1/inference/deployments` (see
  [Creating the deployment](#creating-the-deployment)). The
  [library README](../../../libs/cwtrain/README.md#prerequisites) describes the prerequisites and
  the API.
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
- **Axolotl's release image.** It runs unchanged. `k8s/make-kit.sh` bundles the `cwtrain` wheel
  from `libs/cwtrain/dist`, the launcher and the Axolotl config into a ConfigMap. `run-trainer.sh`
  installs the wheel into a venv of its own, so Axolotl's environment stays as the image built it.
  The install needs access to PyPI for `cwtrain`'s dependencies.
- **The base model, staged once.** Stage it into the bucket with
  [`stage-model-job.yaml`](../../../k8s/stage-model-job.yaml). The deployment loads it from there.
  When `BASE_S3_URI` is set, the trainer fetches the same objects, so the run starts from exactly
  the weights the deployment serves. Otherwise the trainer downloads `HF_MODEL` at `HF_REVISION`
  from Hugging Face. In both cases, the result must have the tensors of the model that the
  deployment serves (see [What the deployment can serve](#what-the-deployment-can-serve)).

## How the launcher runs Axolotl

Axolotl runs as released, with no patch and no plugin. Nothing of the recipe runs inside its
processes, and they do not get the CoreWeave API credentials (`COREWEAVE_API_TOKEN`,
`CW_API_TOKEN`, `CWTRAIN_API_HEADERS`). [`launch.py`](launch.py) does these steps:

1. It stops if the deployment already hot-loaded the run's identity, `<RUN_ID>-v<MAX_STEPS>`.
2. It stages the base model in `/work/models/<MODEL_SLUG>`.
3. It trains: `axolotl train sft_coreweave.yaml --base-model <staged copy> --output-dir /work/out
   --max-steps <MAX_STEPS> <EXTRA_TRAIN_ARGS> --launcher torchrun -- --nproc_per_node
   <TRAINER_GPUS>`.
4. After a LoRA run, which leaves an `adapter_config.json` in the output, it runs
   `axolotl merge-lora` with the same options and continues with `/work/out/merged`.
5. It checks the result against the base model (see
   [What the deployment can serve](#what-the-deployment-can-serve)).
6. It writes the snapshot: the result's weights, with each tensor's Adler-32 checksum, which the
   engine verifies on install, and the base model's other files.
7. It uploads the snapshot, waits until the deployment is READY, creates a full hot-load, and waits
   until every replica serves the new weights.

The deployment and the hot-load share one budget after the upload, `HOTLOAD_TIMEOUT` (default 1800
seconds). With `HOTLOAD_WAIT=0`, the launcher returns once the hot-load is created, and
`cwtrain hotload list --deployment "$DEPLOYMENT_ID"` shows when it completes.
`HOTLOAD_VERIFY=1` re-checks every replica after the hot-load completes.

[`sft_coreweave.yaml`](sft_coreweave.yaml) is the smoke run: a full fine-tune of Qwen3-4B on
2,000 rows of [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), one shard
at a pinned revision, with FSDP2 on two GPUs for 30 steps. Its solutions box their final
answers, as the RL recipes' `--rm-type math` reward expects. When you change the config, keep
these settings:

- `strict: false`, so that the launcher's options and `EXTRA_TRAIN_ARGS` can override any key.
  They are config keys as dashed options; `axolotl train --help` lists them. An option that
  Axolotl does not know goes on to `torchrun`.
- `chat_template: tokenizer_default`: training uses the base model's chat template, which the
  deployment keeps. `chat_template_kwargs: {enable_thinking: false}` renders Qwen3's template the
  way the RL recipes request it, with thinking off.
- `state_dict_type: FULL_STATE_DICT`: the final save is one Hugging Face checkpoint, gathered on
  rank 0. A `SHARDED_STATE_DICT` final save holds none, and the launcher stops.
- `save_strategy: "no"`: only the final model is published, so the run saves no checkpoints
  before it.

Axolotl sends telemetry unless `AXOLOTL_DO_NOT_TRACK` is set; `run.env.example` sets it to `1`.

## LoRA

A hot-load serves full weights only. So after a LoRA run, the launcher merges the adapter into the
base model with `axolotl merge-lora`, and publishes the merged model. Turn LoRA on in
`EXTRA_TRAIN_ARGS`:

```bash
EXTRA_TRAIN_ARGS='--adapter lora --lora-r 32 --lora-alpha 64 --lora-target-linear'
```

The merge gets the same options as the training run and writes `/work/out/merged`. With FSDP2,
Axolotl saves the adapter only, so the merge is the step that produces full weights.

## Running it on Kubernetes

This recipe's `k8s/` directory, with the files that the training recipes share in
[`k8s/`](../../../k8s/README.md) at the repository root, is a complete run on a Kubernetes cluster
with a GPU node of `INSTANCE_TYPE`, such as CoreWeave Kubernetes Service (CKS). You need these
tools:

- `kubectl`
- `envsubst` (GNU gettext; `brew install gettext` on macOS)
- `uv`
- the `aws` CLI
- the `cwtrain` CLI (from the repository root, run `uv tool install ./libs/cwtrain`)

Work in this recipe's `k8s/` directory, with a filled-in copy of `run.env.example` loaded into your
shell:

```bash
cd training/recipes/axolotl-sft/k8s
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
| 6. Create the deployment | [below](#creating-the-deployment); put its `id` in `run.env` as `DEPLOYMENT_ID` |
| 7. CoreWeave API token Secret | `kubectl -n "$NAMESPACE" create secret generic "$API_TOKEN_SECRET" --from-file=token=<file>` |
| 8. Trainer | `"$K/render.sh" run.env trainer-job.yaml \| kubectl apply -f -` |

Step 7's token creates the hot-load, so it needs the Inference Admin role. The trainer pod does not
start until the Secret exists. Reading the token from a file keeps it out of your shell history.
The trainer sends no inference requests, so it needs no inference API key.

`trainer-job.yaml` runs `run-trainer.sh` through the image's entrypoint, which sets up the CUDA
environment. `run-trainer.sh` does these steps:

1. Install the wheel into a venv of its own.
2. Check that the trainer's credentials can publish under the hot-load prefix.
3. Start `launch.py`, which runs the steps in
   [How the launcher runs Axolotl](#how-the-launcher-runs-axolotl). The defaults are a full
   fine-tune of Qwen3-4B with FSDP2 on two GPUs, for 30 steps.

### Creating the deployment

Copy [`deployment.example.json`](deployment.example.json) to `deployment.run.json` in this recipe's
`k8s/` (git-ignored). Fill in the gateway id and the bucket. The example pins `dynamo-vllm` to
`0.30.0+crwv.5`, which has hot-load support; its ModelExpress client is built into the engine. Put
your API token in `COREWEAVE_API_TOKEN` (or `CW_API_TOKEN`, as the CoreWeave docs set it). Then
`cwtrain deployment parameters` lists the runtime versions, instance types and engine options that
your organization can use. `--gateway <gateway id>` limits the list to that gateway's zone:

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
2. Load `run.env` again: `set -a; . ./run.env; set +a`.

Its `rollout_base_url` is the OpenAI-compatible endpoint, and its `model_name` the model name to
send requests for. The trainer also waits for READY before the hot-load, within `HOTLOAD_TIMEOUT`.

The example runs two replicas with one GPU each. One is enough to serve the fine-tuned model; two
keep the deployment in the shape of the RL recipes', so that an RL run can continue on it (see
[SFT, then RL](#sft-then-rl)).

The example sets two engine options, `gpu-memory-utilization` and `max-model-len`. The example in
the [RL Rollouts guide](https://docs.coreweave.com/products/inference/rl-rollouts) sets none. Add
only engine options that `cwtrain deployment parameters` lists.

`deployment.run.json` is the request body of `POST /v1alpha1/inference/deployments`. So you can
also create the same deployment with the `curl` request in the
[RL Rollouts guide](https://docs.coreweave.com/products/inference/rl-rollouts) and
`-d @deployment.run.json`, or in the **Hot loading** section of the Cloud Console's create drawer.

### Watching and re-running

In a new shell, run the setup block in [Running it on Kubernetes](#running-it-on-kubernetes) again
first: these commands and step 8 use its `K` and the values in `run.env`. The block keeps your
filled-in `run.env`.

```bash
kubectl -n "$NAMESPACE" logs -f "job/cwtrain-$JOB_NAME"   # the trainer, including cwtrain's lines
cwtrain hotload list --deployment "$DEPLOYMENT_ID"         # one hot-load per run
```

You cannot change a Job while it exists. Before the next run, delete the finished Job with
`kubectl -n "$NAMESPACE" delete job "cwtrain-$JOB_NAME"`, then render and apply step 8 again.
The `RUN_ID` in `run.env.example` uses the time of each render plus a random number. So the new run
publishes under a new identity, as it must: the launcher stops before training when the deployment
already hot-loaded the identity. Your shell's `$RUN_ID` is from when you loaded `run.env`, not from
the render. `cwtrain hotload list` shows the identities that the Jobs used.

### Cleaning up

The deployment holds its GPUs until you delete it. Run this block with the kubectl context that you
used for the steps above (`kubectl config current-context`), and with the admin's object-storage
credentials. Only the admin may delete from the bucket:

```bash
( set -eu   # run from anywhere in the repository; stops at the first unset value or failed step
  cd "$(git rev-parse --show-toplevel)/training/recipes/axolotl-sft/k8s"
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
    "${STORAGE_SECRET:?}" "${ADMIN_STORAGE_SECRET:?}" "${API_TOKEN_SECRET:?}" \
    "${TRAINER_SERVICE_ACCOUNT?set TRAINER_SERVICE_ACCOUNT in run.env, empty for "default"}"
  case "$DEPLOYMENT_ID $NAMESPACE $BUCKET $BUCKET_ZONE $MODEL_PREFIX $MODEL_SLUG $HOTLOAD_PREFIX $JOB_NAME
        $KIT_NAME $STORAGE_SECRET $ADMIN_STORAGE_SECRET $API_TOKEN_SECRET $TRAINER_SERVICE_ACCOUNT" in
    *'<'[A-Za-z]*'>'*) echo "fill in run.env before cleaning up" >&2; exit 1;;
  esac
  # Every deployment shares the hot-load prefix, so remove only the runs that this deployment
  # hot-loaded: <prefix>/<run>-vNNNNNN/ and <prefix>/_manifests/<run>-vNNNNNN.json. List them
  # while the deployment exists.
  runs=$(cwtrain hotload list --deployment "$DEPLOYMENT_ID" --limit 100000 | python3 -c 'import json, re, sys
print(" ".join(sorted({re.sub(r"-v[0-9]+$", "", h["spec"]["identity"]) for h in json.load(sys.stdin)})))')
  cwtrain deployment delete "$DEPLOYMENT_ID" --ignore-not-found
  kubectl -n "$NAMESPACE" delete job "cwtrain-$JOB_NAME" "cwtrain-stage-$JOB_NAME" cwtrain-make-bucket --ignore-not-found
  kubectl -n "$NAMESPACE" delete secret "$STORAGE_SECRET" "$ADMIN_STORAGE_SECRET" "$API_TOKEN_SECRET" \
    --ignore-not-found
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

The snapshot goes with the deployment's other runs. So if an RL run starts from it, clean up after
that run. Recipes that share a namespace also share the Secrets and the service account, unless
their `run.env` names differ; the defaults are the same in every recipe, so clean up after the last
recipe. If the recipe created the namespace and nothing else runs there, delete it too:
`kubectl delete namespace "$NAMESPACE"`.

## SFT, then RL

The snapshot is a plain Hugging Face checkpoint, so an RL recipe can start from it. Set it as
`BASE_S3_URI` in the RL recipe's `run.env`, such as [miles'](../miles/README.md):

```bash
BASE_S3_URI=s3://<bucket>/hotload/<RUN_ID>-v000030
```

The RL trainer then starts from exactly the fine-tuned weights. To continue on this deployment,
which already serves them, set its `DEPLOYMENT_ID` there too. The RL hook publishes the weights
again as its run's chain root, `<rl-run>-v000000`, before the first delta. The RL recipe's cleanup
block then removes both runs' snapshots.

To evaluate the fine-tuned model, send requests to the deployment's endpoint for its
`model_name`. The [NeMo Gym recipe](../../../evaluation/recipes/nemo-gym/README.md) describes how
to evaluate an existing deployment.

## Pitfalls

- **`snapshot access check failed` at the start of the trainer Job means that the trainer may not
  use the hot-load prefix.** It needs `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` and
  `s3:AbortMultipartUpload` there, from `AllowTrainer` and from its own org access policy. Check
  that `AllowTrainer` names the trainer's principal (`TRAINER_PRINCIPAL_ARN`), and that
  `HOTLOAD_PREFIX` is the deployment's `hotLoad.pathPrefix`. With Workload Identity Federation,
  also check that the cluster's issuer is registered and that the Job runs as
  `TRAINER_SERVICE_ACCOUNT`. `cwtrain hotload check --deployment <id>` runs the same check with the
  credentials of your shell.
- **`does not match the deployment's base model` means that the run changed the model's shape.**
  The error names up to five tensors. Added tokens resize the embeddings; a different dtype comes
  from the precision settings. Train the model that the deployment serves, with the same
  tokenizer and dtype.
- **`holds no safetensors weights` means that the output has no full checkpoint.** Keep
  `state_dict_type: FULL_STATE_DICT`. A LoRA adapter is merged first only when its
  `adapter_config.json` is in the output directory.
- **`is already hot-loaded` means that the run reused an identity.** Render the trainer Job
  again, which picks a new `RUN_ID`.
- **An option that Axolotl does not know goes to `torchrun`, not into the config.** Take the
  option names from `axolotl train --help`.
- **Large models need host memory for the final save.** `FULL_STATE_DICT` gathers the whole model
  on rank 0 before it writes it. The Job requests 256Gi.
- **A missing bucket policy is the most common reason that weights fail to load.** The model
  download at deployment start and every hot-load read the bucket as the inference service account.

## Failure handling

- A failure before the upload (training, the merge, the check, the snapshot) leaves nothing in the
  bucket. Fix the cause, delete the Job, and render and apply step 8 again.
- A failure after the upload leaves the snapshot in the bucket: the deployment was not READY
  within `HOTLOAD_TIMEOUT`, the hot-load ended `FAILED` (`HotLoadFailedError`, with per-replica
  reasons), or it did not complete in time. The launcher prints the command that hot-loads the
  snapshot without training again:
  `cwtrain hotload trigger --deployment <id> --identity <RUN_ID>-v000030 --reset-prompt-cache`.
  Run it once `cwtrain deployment get <id>` shows READY.
- The Job does not retry (`backoffLimit: 0`). The fine-tuned weights live on the pod's disk until
  they are uploaded, and a retry would train again.
- With `HOTLOAD_VERIFY=1`, a `HotLoadVerificationError` means that the API marked the hot-load
  COMPLETED, but a replica in its list does not report the new identity. Check
  `cwtrain hotload status <id>`.

## Verifying by hand

```bash
cwtrain hotload list --deployment <uuid>                 # newest first, with state
cwtrain hotload status <hot-load uuid>                   # replicas and their current identity
```

To see the swap, send the same greedy request (`"temperature": 0`) to the deployment's
`/v1/completions` before and after the hot-load, and compare the answers.
