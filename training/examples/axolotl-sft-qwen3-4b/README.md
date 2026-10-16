# Example: SFT of Qwen3-4B with Axolotl

This example runs the [Axolotl SFT recipe](../../recipes/axolotl-sft/README.md) on Kubernetes: a
full fine-tune of Qwen3-4B on 2,000 rows of NuminaMath-CoT, published as one full snapshot into a
Dedicated Inference deployment of the same model with two replicas. The deployment then serves the
fine-tuned weights. The recipe's README explains the launcher, what the deployment can serve, LoRA
and its failure handling.

| | |
| --- | --- |
| Recipe | [Axolotl SFT](../../recipes/axolotl-sft/README.md): stock `axolotl train`, a full fine-tune by default, or LoRA (merged) with `EXTRA_TRAIN_ARGS` |
| Model | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), at the commit that `HF_REVISION` pins |
| Data | 2,000 rows of [`AI-MO/NuminaMath-CoT`](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), one shard at a pinned revision, in the recipe's [`sft_coreweave.yaml`](../../recipes/axolotl-sft/sft_coreweave.yaml), which the example runs unchanged |
| Trainer | 2 GPUs of a `gb300-4x` node (`TRAINER_GPUS`), with FSDP2; 30 optimizer steps (`MAX_STEPS`) |
| Deployment | [`deployment.example.json`](deployment.example.json): `dynamo-vllm` on `gb300-4x`, 1 GPU per replica, 2 replicas |

The recipe was validated on an amd64 node with two B200 GPUs. This example's `INSTANCE_TYPE`,
`gb300-4x` (arm64), is not yet validated; see the recipe's
[Versions](../../recipes/axolotl-sft/README.md#versions).

## What you need

Follow the Dedicated Inference [getting-started guide](https://docs.coreweave.com/products/inference/getting-started)
for the account prerequisites (Inference access, GPU quota, an API token with the Inference Admin
role), the gateway, and the bucket policy. Then:

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
  1. Copy it to `k8s/bucket-policy.run.json` in this example's directory (git-ignored, like
     `deployment.run.json`).
  2. Fill in `[BUCKET-NAME]`, `[ORG-ID]`, `[HOT-LOAD-PREFIX]` (the deployment's
     `hotLoad.pathPrefix`, `hotload` in every example) and `[ADMIN-USER-ID]`, which is on the
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
     `k8s/trainer-access-policy.run.json` in this example's directory (git-ignored), fill it in, and
     apply it from this directory with
     `cwic cwobject policy create --file k8s/trainer-access-policy.run.json`. It allows the token
     exchange, reading the bucket, and `s3:PutObject` and `s3:AbortMultipartUpload` under the
     hot-load prefix, and nothing else.
  3. In `AllowTrainer`, the issuer URL for `[OIDC-ISSUER-URL]`, and `NAMESPACE` and
     `TRAINER_SERVICE_ACCOUNT` from `run.env` for `[NAMESPACE]` and `[SERVICE-ACCOUNT]`.

  To use a Cloud Console user's key instead, give the trainer a user of its own, and set
  `AllowTrainer`'s principal to `{"CW": ["arn:aws:iam::[ORG-ID]:coreweave/[TRAINER-USER-ID]"]}`
  (see `run.env.example`). To try the example on your own, you can name your own user in both
  statements, but the trainer's key then has all of your rights; the bucket Job warns about this.
  At start, the trainer Job checks that its credentials can list, read and write under the
  hot-load prefix, and it stops if they cannot.
- **Object storage credentials.** The trainer Job mounts `STORAGE_SECRET`: with Workload Identity
  Federation, it holds no key; otherwise, the trainer's key. The Jobs that create the bucket and
  stage the base model mount the admin's key from `ADMIN_STORAGE_SECRET`. The trainer also needs
  the API token in its environment.
- **Axolotl's release image.** It runs unchanged. The recipe's `k8s/make-kit.sh` bundles the
  `cwtrain` wheel from `libs/cwtrain/dist`, the launcher and the Axolotl config into a ConfigMap.
  Its `run-trainer.sh` installs the wheel into a venv of its own, so Axolotl's environment stays as
  the image built it. The install needs access to PyPI for `cwtrain`'s dependencies.
- **The base model, staged once.** Stage it into the bucket with
  [`stage-model-job.yaml`](../../../k8s/stage-model-job.yaml). The deployment loads it from there.
  When `BASE_S3_URI` is set, the trainer fetches the same objects, so the run starts from exactly
  the weights the deployment serves. Otherwise the trainer downloads `HF_MODEL` at `HF_REVISION`
  from Hugging Face. In both cases, the result must have the tensors of the model that the
  deployment serves (see the recipe's
  [What the deployment can serve](../../recipes/axolotl-sft/README.md#what-the-deployment-can-serve)).

## Running it on Kubernetes

This example's `k8s/run.env.example`, with the recipe's `k8s/` directory and the files that the
training recipes share in [`k8s/`](../../../k8s/README.md) at the repository root, is a complete
run on a Kubernetes cluster with a GPU node of `INSTANCE_TYPE`, such as CoreWeave Kubernetes
Service (CKS). You need these tools:

- `kubectl`
- `envsubst` (GNU gettext; `brew install gettext` on macOS)
- `uv`
- the `aws` CLI
- the `cwtrain` CLI (from the repository root, run `uv tool install ./libs/cwtrain`)

Work in this example's `k8s/` directory, with a filled-in copy of `run.env.example` loaded into your
shell. In zsh, the default shell on macOS, the first line turns on `interactivecomments`, so
that the `#` comments in the blocks you paste are comments, as in bash:

```bash
[ -z "${ZSH_VERSION-}" ] || setopt interactivecomments
cd training/examples/axolotl-sft-qwen3-4b/k8s
K=../../../../k8s                # the shared files: k8s/ at the repository root
R=../../../recipes/axolotl-sft/k8s   # the recipe: its Job, scripts and kit
[ -e run.env ] || cp run.env.example run.env   # once; fill in every <placeholder>, render.sh refuses one
set -a; . ./run.env; set +a      # the commands below use its values
CTX=$(kubectl config current-context)
```

| Step | Command |
| --- | --- |
| 1. Namespace | `"$K/render.sh" run.env "$K/namespace.yaml" \| kubectl apply -f -` |
| 2. Trainer service account and object storage Secrets (AWS-style `config` + `credentials` files) | `kubectl -n "$NAMESPACE" create serviceaccount "$TRAINER_SERVICE_ACCOUNT"`, then `WORKLOAD_IDENTITY=1 AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$STORAGE_SECRET"` (with a trainer key instead, export it as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` and leave out `WORKLOAD_IDENTITY=1`); then, with the admin's key exported, `AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$ADMIN_STORAGE_SECRET"` |
| 3. Wheel + scripts ConfigMap | `(cd ../../../../libs/cwtrain && uv build --wheel) && "$R/make-kit.sh" "$CTX" "$NAMESPACE" "$KIT_NAME"` |
| 4. Bucket + policy (skip if done with `aws s3api`) | `"$K/render.sh" run.env "$K/make-bucket-job.yaml" \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m job/cwtrain-make-bucket`; it is done when `kubectl -n "$NAMESPACE" get job cwtrain-make-bucket` shows COMPLETIONS `1/1` |
| 5. Stage the base model into the bucket | `"$K/render.sh" run.env "$K/stage-model-job.yaml" \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m "job/cwtrain-stage-$JOB_NAME"`; it is done when `kubectl -n "$NAMESPACE" get job "cwtrain-stage-$JOB_NAME"` shows COMPLETIONS `1/1` (it retries once, so follow the logs again while it shows `0/1`) |
| 6. Create the deployment | [below](#creating-the-deployment); put its `id` in `run.env` as `DEPLOYMENT_ID` |
| 7. CoreWeave API token Secret | `kubectl -n "$NAMESPACE" create secret generic "$API_TOKEN_SECRET" --from-file=token=<file>` |
| 8. Trainer | `"$K/render.sh" run.env "$R/trainer-job.yaml" \| kubectl apply -f -` |

Step 7's token creates the hot-load, so it needs the Inference Admin role. The trainer pod does not
start until the Secret exists. Reading the token from a file keeps it out of your shell history.
The trainer sends no inference requests, so it needs no inference API key.

The recipe's `trainer-job.yaml` runs its `run-trainer.sh` through the image's entrypoint, which
sets up the CUDA environment. `run-trainer.sh` does these steps:

1. Install the wheel into a venv of its own.
2. Check that the trainer's credentials can publish under the hot-load prefix.
3. Start `launch.py`, which runs the steps in the recipe's
   [How the launcher runs Axolotl](../../recipes/axolotl-sft/README.md#how-the-launcher-runs-axolotl).
   This example's `run.env.example` runs the recipe's `sft_coreweave.yaml`, a full fine-tune of
   Qwen3-4B with FSDP2, on two GPUs for 30 steps.

### Creating the deployment

Copy [`deployment.example.json`](deployment.example.json) to `deployment.run.json` in this example's
`k8s/` (git-ignored). Fill in the gateway id and the bucket. If you changed `MODEL_PREFIX`,
`MODEL_SLUG` or `HOTLOAD_PREFIX` in `run.env`, change `model.path` (`<MODEL_PREFIX>/<MODEL_SLUG>`)
and `hotLoad.pathPrefix` (`HOTLOAD_PREFIX`) to match. The example pins `dynamo-vllm` to
`0.30.0+crwv.5`, which has hot-load support; its ModelExpress client is built into the engine. Put
your API token in `COREWEAVE_API_TOKEN` (or `CW_API_TOKEN`, as the CoreWeave docs set it). Then
`cwtrain deployment parameters` lists the runtime versions, instance types and engine options that
your organization can use. `--gateway <gateway id>` limits the list to that gateway's zone. Set
`resources.instanceType` to one of the instance types that it lists:

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
keep the deployment in the shape of the RL examples', so that an RL run can continue on it (see
the recipe's [SFT, then RL](../../recipes/axolotl-sft/README.md#sft-then-rl)).

The example sets two engine options, `gpu-memory-utilization` and `max-model-len`. The example in
the [RL Rollouts guide](https://docs.coreweave.com/products/inference/rl-rollouts) sets none. Add
only engine options that `cwtrain deployment parameters` lists.

`deployment.run.json` is the request body of `POST /v1alpha1/inference/deployments`. So you can
also create the same deployment with the `curl` request in the
[RL Rollouts guide](https://docs.coreweave.com/products/inference/rl-rollouts) and
`-d @deployment.run.json`, or in the **Hot loading** section of the Cloud Console's create drawer.

### Watching and re-running

In a new shell, run the setup block in [Running it on Kubernetes](#running-it-on-kubernetes) again
first: these commands and step 8 use its `K`, its `R` and the values in `run.env`. The block keeps
your filled-in `run.env`.

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
credentials. Only the admin may delete from the bucket. If another deployment uses one of this
deployment's snapshots as its base model, wait until that deployment is gone before running this
block.

```bash
[ -z "${ZSH_VERSION-}" ] || setopt interactivecomments
( set -eu   # run from anywhere in the repository; stops at the first unset value or failed step
  cd "$(git rev-parse --show-toplevel)/training/examples/axolotl-sft-qwen3-4b/k8s"
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
  # List exact identities while the deployment exists. A failed list must stop cleanup,
  # even if it wrote JSON before failing. Validate every identity before deleting anything.
  hot_loads=$(cwtrain hotload list --deployment "$DEPLOYMENT_ID" --limit 100000)
  identities=$(printf '%s\n' "$hot_loads" | python3 -c 'import json, sys
hot_loads = json.load(sys.stdin)
if not isinstance(hot_loads, list):
    sys.exit("expected a hot-load history list; refusing cleanup")
identities = [h["spec"]["identity"] for h in hot_loads]
for identity in identities:
    if (not isinstance(identity, str) or not 1 <= len(identity) <= 128
            or identity in (".", "..", "initial", "_manifests") or "/" in identity
            or any(ord(c) < 32 or ord(c) == 127 for c in identity)):
        sys.exit(f"unsafe snapshot identity {identity!r}; remove its objects by hand")
print("\n".join(sorted(set(identities))))')
  cwtrain deployment delete "$DEPLOYMENT_ID" --ignore-not-found
  kubectl -n "$NAMESPACE" delete job "cwtrain-$JOB_NAME" "cwtrain-stage-$JOB_NAME" cwtrain-make-bucket --ignore-not-found
  kubectl -n "$NAMESPACE" delete secret "$STORAGE_SECRET" "$ADMIN_STORAGE_SECRET" "$API_TOKEN_SECRET" \
    --ignore-not-found
  kubectl -n "$NAMESPACE" delete configmap "$KIT_NAME" --ignore-not-found
  [ -z "$TRAINER_SERVICE_ACCOUNT" ] || kubectl -n "$NAMESPACE" delete serviceaccount "$TRAINER_SERVICE_ACCOUNT" --ignore-not-found
  [ -n "$identities" ] || echo "found no hot-loads of $DEPLOYMENT_ID, so nothing under $HOTLOAD_PREFIX/ is removed" >&2
  # A trailing slash bounds each snapshot prefix. No wildcard filters: names such as
  # experiment and experiment-v2, or names containing *, must never select each other.
  printf '%s\n' "$identities" | while IFS= read -r identity; do
    [ -n "$identity" ] || continue
    aws s3 rm --recursive "s3://$BUCKET/$HOTLOAD_PREFIX/$identity/" \
      --endpoint-url https://cwobject.com --region "$BUCKET_ZONE"
    aws s3 rm "s3://$BUCKET/$HOTLOAD_PREFIX/_manifests/$identity.json" \
      --endpoint-url https://cwobject.com --region "$BUCKET_ZONE"
  done
)
```

The block removes only the exact snapshot identities returned by the deployment's hot-load
history: `<prefix>/<identity>/` and the corresponding `_manifests/<identity>.json`. It does not
expand a run name into a wildcard. The prefix comes from `deployment.run.json`, or from
`HOTLOAD_PREFIX` in `run.env` if you created the deployment another way. Uploads that never reached
the hot-load API are not in that history and need separate cleanup.

The staged base model and the bucket Job's write check (`_probe`) remain in the bucket. Other
deployments can use the same model path even when they run in different Kubernetes namespaces.
Remove those shared objects separately only after every deployment using them is gone.

The block also deletes the Jobs, the Secrets, the kit ConfigMap and the trainer's service account that
`run.env` names, and the bucket Job. It does not delete the namespace or the trainer's org access
policy. You can run it again after a partial cleanup. But a deleted deployment lists no hot-loads,
so if the deployment is already gone, identify and remove its exact snapshot paths by hand
(`aws s3 ls` the hot-load prefix).

The snapshot goes with the deployment's other runs. So if an RL run starts from it, clean up after
that run. Examples that share a namespace also share the Secrets and the service account, unless
their `run.env` names differ; the defaults are the same in every example, so clean up after the
last one. If the example created the namespace and nothing else runs there, delete it too:
`kubectl delete namespace "$NAMESPACE"`.
