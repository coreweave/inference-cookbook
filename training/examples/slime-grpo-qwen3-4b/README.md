# Example: GRPO on Qwen3-4B with slime

This example runs the [slime recipe](../../recipes/slime/README.md) on Kubernetes: GRPO on
Qwen3-4B with the DAPO-Math-17k prompts. The rollouts come from a Dedicated Inference deployment
with two replicas, and every weight update is hot-loaded into it. The recipe's README explains the
loop, its patches, its settings and its failure handling.

| | |
| --- | --- |
| Recipe | [slime](../../recipes/slime/README.md): `train.py` (synchronous), at the slime commit that `SLIME_COMMIT` pins |
| Model | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), at the commit that `HF_REVISION` pins |
| Data | [`zhuzilin/dapo-math-17k`](https://huggingface.co/datasets/zhuzilin/dapo-math-17k), which the recipe's launcher downloads |
| Reward | slime's `math` reward (`RM_TYPE`), which grades the boxed answer |
| Trainer | 2 GPUs of a `gb300-4x` node, TP=2; 30 rollout steps (`NUM_ROLLOUT`) of 8 prompts × 4 samples |
| Deployment | [`deployment.example.json`](deployment.example.json): `dynamo-vllm` on `gb300-4x`, 1 GPU per replica, 2 replicas, `autoscaling.concurrency` 256 |

## What you need

Follow the Dedicated Inference
[getting-started guide](https://docs.coreweave.com/products/inference/getting-started) for the
account prerequisites (Inference access, GPU quota, an API token with the Inference Admin role),
the gateway, and the bucket policy. This example also needs:

- **A hot-load-enabled deployment.** You enable hot-load when you create the deployment: engine
  `dynamo-vllm` and a `hotLoad` section that names the snapshot bucket and path prefix. CoreWeave
  enables hot-load per organization. Until it is enabled for yours, the create fails with
  `hot-load is not enabled for this organization`; ask CoreWeave support and give your
  organization ID. The [RL Rollouts docs](https://docs.coreweave.com/products/inference/rl-rollouts)
  describe the feature. [`deployment.example.json`](deployment.example.json) is a complete request
  body for `POST /v1alpha1/inference/deployments` (see
  [Creating the deployment](#creating-the-deployment)). The
  [library README](../../../libs/cwtrain/README.md#prerequisites) describes the prerequisites and
  the API. The deployment's `autoscaling.concurrency` caps how many sequences each replica decodes
  at once; see [Concurrency](#concurrency).
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
- **A trainer image with slime's dependencies.** The image that `run.env.example` pins,
  `radixark/miles:v0.1.1`, carries slime's torch, SGLang and Ray stack for amd64 and arm64 (GB200,
  GB300). With this image, the recipe's launcher:
  - clones slime at `SLIME_COMMIT`.
  - checks out the Megatron-LM commit that slime is developed against (`MEGATRON_COMMIT`, from
    `docker/Dockerfile` in the slime repo), applies slime's Megatron patches, and puts it first on
    `PYTHONPATH`.
  - skips slime's numpy 1 assertion, because the image ships numpy 2 (`SLIME_RELAX_NUMPY_CHECK=1`,
    the recipe's `patches/0002-relax-numpy-check.patch`).

  The official `slimerl/slime` images are amd64 only and carry slime's own Megatron. With one of
  them, set `SLIME_RELAX_NUMPY_CHECK=0` and leave `MEGATRON_COMMIT` empty. The launcher moves the
  image's `/root/slime` clone to `SLIME_COMMIT` before it patches it. At start, the Job installs
  `cwtrain` from the wheel that the recipe's `k8s/make-kit.sh` bundles into a ConfigMap.
- **The base model, staged once.** Stage it into the bucket with
  [`stage-model-job.yaml`](../../../k8s/stage-model-job.yaml). The
  deployment loads it from there. When `BASE_S3_URI` is set, the trainer fetches the same objects,
  so the run starts from exactly the weights the deployment serves. Otherwise the trainer downloads
  `HF_MODEL` from Hugging Face. In both cases the hook publishes the trainer's copy as the run's
  chain root (`<run>-v000000`) before the first delta. So the two copies must share the
  architecture and tensor layout.

## Running it on Kubernetes

This example's `k8s/run.env.example`, with the recipe's `k8s/` directory and the files that the
training recipes share in [`k8s/`](../../../k8s/README.md) at the repository root, runs this
topology on a Kubernetes cluster with a GPU node of `INSTANCE_TYPE`, such as CoreWeave Kubernetes
Service (CKS). You need `kubectl`, `envsubst` (GNU gettext; `brew install gettext` on macOS), `uv`,
the `aws` CLI, and the `cwtrain` CLI. To install the `cwtrain` CLI, run
`uv tool install ./libs/cwtrain` from the repository root. Work in this example's `k8s/`
directory, with a filled-in copy of `run.env.example` loaded into your shell. In zsh, the
default shell on macOS, the first line turns on `interactivecomments`, so that the `#`
comments in the blocks you paste are comments, as in bash:

```bash
[ -z "${ZSH_VERSION-}" ] || setopt interactivecomments
cd training/examples/slime-grpo-qwen3-4b/k8s
K=../../../../k8s                # the shared files: k8s/ at the repository root
R=../../../recipes/slime/k8s     # the recipe: its Job, scripts and kit
[ -e run.env ] || cp run.env.example run.env   # once; fill in every <placeholder>, render.sh refuses one
set -a; . ./run.env; set +a      # the commands below use its values
CTX=$(kubectl config current-context)
```

| Step | Command |
| --- | --- |
| 1. Namespace | `"$K/render.sh" run.env "$K/namespace.yaml" \| kubectl apply -f -` |
| 2. Trainer service account and object storage Secrets (AWS-style `config` + `credentials` files) | `kubectl -n "$NAMESPACE" create serviceaccount "$TRAINER_SERVICE_ACCOUNT"`, then `WORKLOAD_IDENTITY=1 AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$STORAGE_SECRET"` (with a trainer key instead, export it as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` and leave out `WORKLOAD_IDENTITY=1`); then, with the admin's key exported, `AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$ADMIN_STORAGE_SECRET"` |
| 3. Wheel + scripts + patch ConfigMap | `(cd ../../../../libs/cwtrain && uv build --wheel) && "$R/make-kit.sh" "$CTX" "$NAMESPACE" "$KIT_NAME"` |
| 4. Bucket + policy (skip if done with `aws s3api`) | `"$K/render.sh" run.env "$K/make-bucket-job.yaml" \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m job/cwtrain-make-bucket`; it is done when `kubectl -n "$NAMESPACE" get job cwtrain-make-bucket` shows COMPLETIONS `1/1` |
| 5. Stage the base model into the bucket | `"$K/render.sh" run.env "$K/stage-model-job.yaml" \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m "job/cwtrain-stage-$JOB_NAME"`; it is done when `kubectl -n "$NAMESPACE" get job "cwtrain-stage-$JOB_NAME"` shows COMPLETIONS `1/1` (it retries once, so follow the logs again while it shows `0/1`) |
| 6. Create the deployment | [below](#creating-the-deployment); put its `id` and `rollout_base_url` in `run.env` as `DEPLOYMENT_ID` and `ROLLOUT_BASE_URL` |
| 7. Inference API key Secret | `kubectl -n "$NAMESPACE" create secret generic "$INFERENCE_KEY_SECRET" --from-file=api_key=<file>` |
| 8. CoreWeave API token Secret | `kubectl -n "$NAMESPACE" create secret generic "$API_TOKEN_SECRET" --from-file=token=<file>` |
| 9. Trainer | `"$K/render.sh" run.env "$R/trainer-job.yaml" \| kubectl apply -f -` |

Step 7's key is the bearer token that the deployment's gateway accepts for inference requests.
When the gateway uses CoreWeave authentication, it is a CoreWeave API token. Step 8's token needs
the Inference Admin role, because it creates hot-loads. One token can serve both. The trainer pod
does not start until both Secrets exist. Reading the tokens from files keeps them out of your shell
history.

The recipe's `trainer-job.yaml` runs its `run-trainer.sh`, which installs the wheel and starts the
recipe's [`launch.py`](../../recipes/slime/launch.py). The launcher:

1. Checks out and patches slime.
2. Fetches the base model (from the bucket when `BASE_S3_URI` is set, otherwise from Hugging Face)
   and the dapo-math-17k prompts.
3. Converts the checkpoint to Megatron `torch_dist` with slime's converter and model script
   (`SLIME_MODEL_SCRIPT`, default `qwen3-4B`).
4. Starts Ray.
5. Starts `train.py` with the [recipe's flags](../../recipes/slime/README.md#slime-flags). This
   example's `run.env.example` trains Qwen3-4B with GRPO and TP=2 on two GPUs.

### Creating the deployment

Copy [`deployment.example.json`](deployment.example.json) to `deployment.run.json` in this example's
`k8s/` (git-ignored). Fill in the gateway id and the bucket. The example pins `dynamo-vllm` to
`0.30.0+crwv.5`, which has hot-load support; its ModelExpress client is built into the engine. Put your API token in `COREWEAVE_API_TOKEN` (or
`CW_API_TOKEN`, as the CoreWeave docs set it). Then `cwtrain deployment parameters` lists the
runtime versions, instance types and engine options your organization can use. Add
`--gateway <gateway id>` to show only that gateway's zone:

```bash
cwtrain deployment parameters
[ -e deployment.run.json ] || cp ../deployment.example.json deployment.run.json   # then fill it in
```

Create the deployment from the filled-in file and wait until it serves:

```bash
cwtrain deployment create --file deployment.run.json --wait
```

`create` refuses a file that still holds a `<placeholder>`. If the name is taken, it names the
deployment that has it. With `--wait`, it returns when the deployment is READY. This takes several
minutes, and longer for large models. Without `--wait`, it returns at once, and
`cwtrain deployment get "$DEPLOYMENT_ID"` shows the status later. In both cases it prints the
deployment as JSON, also when the wait fails. A deployment whose wait failed still exists and
holds its GPUs until you delete it (see [Cleaning up](#cleaning-up)).

Then:

1. Put the deployment's `id` in `run.env` as `DEPLOYMENT_ID`.
2. Put its `rollout_base_url` in `run.env` as `ROLLOUT_BASE_URL`. If it is `null`, the gateway
   could not be read yet; `cwtrain deployment get "$DEPLOYMENT_ID"` shows it later.
3. Check that `ROLLOUT_MODEL` matches its `model_name`.
4. Load `run.env` again: `set -a; . ./run.env; set +a`.

The trainer also waits for READY before its first hot-load, within `HOTLOAD_TIMEOUT`.

The example sets two engine options, `gpu-memory-utilization` and `max-model-len`. The example in
the [RL Rollouts guide](https://docs.coreweave.com/products/inference/rl-rollouts) sets none. Add
other options only from the list that `cwtrain deployment parameters` prints.

`deployment.run.json` is the request body of `POST /v1alpha1/inference/deployments`. So the
`curl` request in the [RL Rollouts guide](https://docs.coreweave.com/products/inference/rl-rollouts)
creates the same deployment with `-d @deployment.run.json`. The Cloud Console can also create it,
in the **Hot loading** section of its create drawer.

### Watching and re-running

In a new shell, run the setup block in [Running it on Kubernetes](#running-it-on-kubernetes) again
first: these commands and step 9 use its `K` and `R` and the values in `run.env`. The block keeps
your filled-in `run.env`.

```bash
kubectl -n "$NAMESPACE" logs -f "job/cwtrain-$JOB_NAME"   # the trainer, including cwtrain's lines
cwtrain hotload list --deployment "$DEPLOYMENT_ID"         # one hot-load per published version
```

You cannot change a Job while it exists. Before the next run, delete the finished Job with
`kubectl -n "$NAMESPACE" delete job "cwtrain-$JOB_NAME"`, then render and apply step 9 again.
The `RUN_ID` in `run.env.example` takes the time of each render plus a random number. So the new
run publishes under new identities, as it must (see the recipe's
[Resuming a run](../../recipes/slime/README.md#resuming-a-run)). Your shell's `$RUN_ID` comes from
when you loaded `run.env`, not from the render. `cwtrain hotload list` shows the identities that
the Job uses.

### Cleaning up

The deployment holds its GPUs until you delete it. Run this block with the kubectl context you used
for the steps above (`kubectl config current-context`), and with the admin's object-storage
credentials. Only the admin may delete from the bucket:

```bash
[ -z "${ZSH_VERSION-}" ] || setopt interactivecomments
( set -eu   # run from anywhere in the repository; stops at the first unset value or failed step
  cd "$(git rev-parse --show-toplevel)/training/examples/slime-grpo-qwen3-4b/k8s"
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

Examples in the same namespace share these objects unless their `run.env` names differ, and the
defaults are the same in every example. So clean up after the last of them. If the example
created the namespace and nothing else runs there, delete it too:
`kubectl delete namespace "$NAMESPACE"`.

## Concurrency

The recipe's [Concurrency](../../recipes/slime/README.md#concurrency) section explains the two
settings that decide how many sequences are sampled at once. In this example, the deployment runs
two replicas at `autoscaling.concurrency` 256, and `run.env.example` keeps `ROLLOUT_CONCURRENCY=16`
group requests of 4 samples in flight. So a whole step, 8 prompts × 4 samples, runs at once.

## Pitfalls

- **Turn off thinking for Qwen3 rollouts.** With the default chat template, every response fills
  the token budget with `<think>` reasoning. All rewards are 0, the gradient is 0, and the first
  delta is empty. `--apply-chat-template-kwargs '{"enable_thinking":false}'` fixes it;
  `run.env.example` sets it in `EXTRA_TRAIN_ARGS`.
- **Pick a verifier that can see the answer.** slime's `deepscaler` reward grades only the text
  after a `</think>` marker. So it returns 0 for every response when thinking is off, or when
  thinking is on but truncated. `--rm-type math` grades the boxed answer wherever it is.
