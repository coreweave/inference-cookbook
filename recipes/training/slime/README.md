# Recipe: slime RL with rollouts on a CoreWeave hot-load deployment

Train with [slime](https://github.com/THUDM/slime) and serve rollouts from a CoreWeave Dedicated
Inference deployment. The deployment receives every weight update by checkpoint hot-load. The
trainer runs no inference engines:

1. After each sync, slime writes a versioned weight directory.
2. The `cwtrain` post-write hook uploads it and creates a hot-load.
3. The deployment swaps the weights in place on all replicas.
4. The next rollout batch samples the new weights through the deployment's OpenAI-compatible
   endpoint.

![One training step: the trainer writes weight_vNNNNNN, the cwtrain hook uploads it to Object Storage, creates a hot-load and polls until every replica is READY; the next rollout batch samples fresh weights from the OpenAI-compatible endpoint.](../../../docs/hotload-flow.drawio.svg)

The same loop with miles, which needs no patch, is in [`../miles/`](../miles/README.md).

## What slime needs that miles does not

slime expects to own SGLang engines. When none run, its `update_weights` returns early with "No
updatable SGLang engines are running; skip weight update". That also skips the disk-delta writer,
so the hook never runs.
[`patches/0001-standalone-delta-publish.patch`](patches/0001-standalone-delta-publish.patch) adds
`--update-weight-standalone-publish`. The flag:

- lets `update_weights` run the disk-delta writer with an empty engine list. The hook receives
  `rollout_engines=[]`, and every engine call is a no-op.
- skips slime's SGLang router when `--rollout-num-gpus 0`.
- removes the `--update-weight-local-checkpoint-dir` requirement, which applies only to slime's own
  engines.
- refuses NCCL transport, full mode, rollout GPUs and `--rollout-external-engine-addrs`, so no
  other sync path runs without its consumers.

Without the flag, slime's behaviour does not change.

### About the patches

The patches add what slime does not ship yet. They are not a second integration layer: the
`cwtrain` library carries no patches, and the miles recipe needs none because miles has the
standalone disk-delta path. The patches are small and stay in this recipe, so each one can go when
slime ships an equivalent.

| Patch | Size | What it changes | Needed when |
| --- | --- | --- | --- |
| `0001-standalone-delta-publish` | about 40 lines | adds `--update-weight-standalone-publish` as described above | always, until slime can publish deltas without its own engines |
| `0002-relax-numpy-check` | about 7 lines | lets slime's numpy 1 assertion be skipped with `SLIME_RELAX_NUMPY_CHECK=1` | only on images that ship numpy 2, such as the arm64 miles image |
| `0003-zero-kl-without-rollout-logprobs` | about 8 lines | shapes slime's zero KL from response lengths when a sample has no `rollout_log_probs` | only as a fallback when the endpoint's log-probs cannot be aligned with a batch |

The launcher (`launch.py`) applies the patches in order and checks each one before it applies it.
If a patch does not apply, the launcher reverts the patches it applied and stops the job. The
message names the tested commit, so a newer slime never starts half-patched. Patches that are
already applied are skipped.

### Tested versions

| Component | Version |
| --- | --- |
| slime | [`8ee9c1e1c`](https://github.com/THUDM/slime/commit/8ee9c1e1c8871ccd6dc8ec812edfaefa3dd1156b) (the default `SLIME_COMMIT`, checked out regardless of the slime the image ships) |
| Megatron-LM, when not taken from the image | [`1dcf0dafa`](https://github.com/NVIDIA/Megatron-LM/commit/1dcf0dafa884ad52ffb243625717a3471643e087) (the default `MEGATRON_COMMIT`), with slime's own Megatron patches |
| Trainer image | `radixark/miles` with `SLIME_RELAX_NUMPY_CHECK=1`; `run.env.example` pins `radixark/miles:v0.1.1`, the release closest to the slime commit above |

A newer slime usually works if every patch still applies. The launcher tells you when one does
not. On slime's main branch, the code that patch 0001 touches is different. So the launcher moves
the official slime images' clone of main to `SLIME_COMMIT`.

`--rollout-external-engine-addrs` is not an alternative. It keeps the sync running, but slime then
calls each address's `/pull_weights` and `update_weights_from_disk` routes. A CoreWeave deployment
does not expose them.

## What you need

Follow the Dedicated Inference
[getting-started guide](https://docs.coreweave.com/products/inference/getting-started) for the
account prerequisites (Inference access, GPU quota, an API token with the Inference Admin role),
the gateway, and the bucket policy. This recipe also needs:

- **A hot-load-enabled deployment.** You enable hot-load when you create the deployment: engine
  `dynamo-vllm` and a `hotLoad` section that names the snapshot bucket and path prefix. CoreWeave
  enables hot-load per organization. Until it is enabled for yours, the create fails with
  `hot-load is not enabled for this organization`; ask CoreWeave support and give your
  organization ID. The [RL Rollouts docs](https://docs.coreweave.com/products/inference/rl-rollouts)
  describe the feature. [`deployment.example.json`](deployment.example.json) is a complete request
  body for `POST /v1alpha1/inference/deployments` (see
  [Creating the deployment](#creating-the-deployment)). The
  [library README](../../../libs/cwtrain/README.md#prerequisites) describes the prerequisites and
  the API. The example's `autoscaling.concurrency` caps how many sequences each replica decodes at
  once; see [Concurrency](#concurrency).
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
- **A trainer image with slime's dependencies.** The default, `radixark/miles:v0.1.1`, carries
  slime's torch, SGLang and Ray stack for amd64 and arm64 (GB200, GB300). With this image, the
  launcher:
  - clones slime at `SLIME_COMMIT`.
  - checks out the Megatron-LM commit that slime is developed against (`MEGATRON_COMMIT`, from
    `docker/Dockerfile` in the slime repo), applies slime's Megatron patches, and puts it first on
    `PYTHONPATH`.
  - skips slime's numpy 1 assertion, because the image ships numpy 2 (`SLIME_RELAX_NUMPY_CHECK=1`,
    `patches/0002-relax-numpy-check.patch`).

  The official `slimerl/slime` images are amd64 only and carry slime's own Megatron. With one of
  them, set `SLIME_RELAX_NUMPY_CHECK=0` and leave `MEGATRON_COMMIT` empty. The launcher moves the
  image's `/root/slime` clone to `SLIME_COMMIT` before it patches it. At start, the Job installs
  `cwtrain` from the wheel that `k8s/make-kit.sh` bundles into a ConfigMap.
- **The base model, staged once.** Stage it into the bucket with
  [`stage-model-job.yaml`](../../../k8s/stage-model-job.yaml). The
  deployment loads it from there. When `BASE_S3_URI` is set, the trainer fetches the same objects,
  so the run starts from exactly the weights the deployment serves. Otherwise the trainer downloads
  `HF_MODEL` from Hugging Face. In both cases the hook publishes the trainer's copy as the run's
  chain root (`<run>-v000000`) before the first delta. So the two copies must share the
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

- `CWTRAIN_HOTLOAD_WAIT=0` stops blocking until the hot-load completes.
- `CWTRAIN_HOTLOAD_TIMEOUT` sets the hot-load timeout in seconds (default 1800).
- `CWTRAIN_HOTLOAD_VERIFY=1` re-checks every replica after each hot-load.

The hook's module docstring (`cwtrain/integrations/slime.py`) lists all its settings. In the
Kubernetes run, `run.env` sets these and the prompt-cache policy: `HOTLOAD_WAIT`,
`HOTLOAD_TIMEOUT`, `HOTLOAD_VERIFY`, `HOTLOAD_PROMPT_CACHE_POLICY`.

slime's `train.py` is synchronous: each rollout round samples the weights that the hook just
hot-loaded. So, unless you set it, the launcher sets
`CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY=reset_all`. This asks the platform to drop KV cache computed
under the previous weights, instead of reusing cached prompt prefixes across the swap. Set
`preserve` to keep the cache.

Snapshot identities are `<RUN_ID>-vNNNNNN`. The platform keys weight versions by identity, so never
reuse a run ID for different weights.

## slime flags

```bash
python train.py ... \
  --rollout-num-gpus 0 \
  --rollout-function-path coreweave_rollout.generate_rollout \
  --update-weight-mode delta \
  --update-weight-transport disk \
  --update-weight-standalone-publish \
  --update-weight-disk-dir /work/deltas \
  --update-weight-delta-encoding xor \
  --update-weight-delta-checksum adler32 \
  --custom-update-weight-post-write-path cwtrain.integrations.slime.hook
```

Delta mode publishes zstd XOR deltas with Adler-32 checksums. This is the only delta format the
hot-load API accepts, and the hook rejects any other encoding or checksum before it uploads. Delta
mode does not work with `--colocate`.

[`coreweave_rollout.py`](coreweave_rollout.py) is the rollout function that slime imports.
`generate_rollout(args, rollout_id, data_source, evaluation)`:

- makes one `/v1/completions` request per GRPO group (`n` choices) to the deployment, through its
  gateway.
- sends the prompt as the token ids that slime trains on.
- takes the exact sampled token ids. It re-tokenizes the text only when the endpoint does not
  report them.
- scores the responses with slime's own `--rm-type`.
- attaches the endpoint's sampled-token log-probs as `rollout_log_probs` when they align with the
  response tokens, including the token that ended the response. Otherwise it leaves them unset.
  The trainer recomputes log-probs in both cases.

It is a thin slime adapter over `cwtrain.integrations.rollout`. Configure it with
`CWTRAIN_ROLLOUT_BASE_URL`, `CWTRAIN_ROLLOUT_MODEL` and, optionally,
`CWTRAIN_ROLLOUT_API_KEY_FILE`. The hook blocks until every replica serves the new version, so the
next rollout is on-policy without an engine handshake.

## Which entrypoint

slime ships `train.py` only. Its asynchronous training is the fully-async rollout worker
(`--fully-async`), which needs slime's own engines. This recipe does not cover it. The miles recipe
shows the one-step-ahead `train_async.py` variant.

## Running it on Kubernetes

This recipe's `k8s/` directory, with the files that the training recipes share in
[`k8s/`](../../../k8s/README.md) at the repository root, runs this topology on a Kubernetes cluster
with a GPU node of `INSTANCE_TYPE`, such as CoreWeave Kubernetes Service (CKS). You need `kubectl`,
`envsubst` (GNU gettext; `brew install gettext` on macOS), `uv`, the `aws` CLI, and the `cwtrain`
CLI. To install the `cwtrain` CLI, run `uv tool install ./libs/cwtrain` from the repository root.
Work in this recipe's `k8s/` directory, with a filled-in copy of `run.env.example` loaded into your
shell:

```bash
cd recipes/training/slime/k8s
K=../../../../k8s                # the shared files: k8s/ at the repository root
[ -e run.env ] || cp run.env.example run.env   # once; fill in every <placeholder>, render.sh refuses one
set -a; . ./run.env; set +a      # the commands below use its values
CTX=$(kubectl config current-context)
```

| Step | Command |
| --- | --- |
| 1. Namespace | `"$K/render.sh" run.env "$K/namespace.yaml" \| kubectl apply -f -` |
| 2. Trainer service account and object storage Secrets (AWS-style `config` + `credentials` files) | `kubectl -n "$NAMESPACE" create serviceaccount "$TRAINER_SERVICE_ACCOUNT"`, then `WORKLOAD_IDENTITY=1 AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$STORAGE_SECRET"` (with a trainer key instead, export it as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` and leave out `WORKLOAD_IDENTITY=1`); then, with the admin's key exported, `AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$ADMIN_STORAGE_SECRET"` |
| 3. Wheel + scripts + patch ConfigMap | `(cd ../../../../libs/cwtrain && uv build --wheel) && ./make-kit.sh "$CTX" "$NAMESPACE" "$KIT_NAME"` |
| 4. Bucket + policy (skip if done with `aws s3api`) | `"$K/render.sh" run.env "$K/make-bucket-job.yaml" \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m job/cwtrain-make-bucket`; it is done when `kubectl -n "$NAMESPACE" get job cwtrain-make-bucket` shows COMPLETIONS `1/1` |
| 5. Stage the base model into the bucket | `"$K/render.sh" run.env "$K/stage-model-job.yaml" \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m "job/cwtrain-stage-$JOB_NAME"`; it is done when `kubectl -n "$NAMESPACE" get job "cwtrain-stage-$JOB_NAME"` shows COMPLETIONS `1/1` (it retries once, so follow the logs again while it shows `0/1`) |
| 6. Create the deployment | [below](#creating-the-deployment); put its `id` and `rollout_base_url` in `run.env` as `DEPLOYMENT_ID` and `ROLLOUT_BASE_URL` |
| 7. Inference API key Secret | `kubectl -n "$NAMESPACE" create secret generic "$INFERENCE_KEY_SECRET" --from-file=api_key=<file>` |
| 8. CoreWeave API token Secret | `kubectl -n "$NAMESPACE" create secret generic "$API_TOKEN_SECRET" --from-file=token=<file>` |
| 9. Trainer | `"$K/render.sh" run.env trainer-job.yaml \| kubectl apply -f -` |

Step 7's key is the bearer token that the deployment's gateway accepts for inference requests.
When the gateway uses CoreWeave authentication, it is a CoreWeave API token. Step 8's token needs
the Inference Admin role, because it creates hot-loads. One token can serve both. The trainer pod
does not start until both Secrets exist. Reading the tokens from files keeps them out of your shell
history.

`trainer-job.yaml` runs `run-trainer.sh`, which installs the wheel and starts
[`launch.py`](launch.py). The launcher:

1. Checks out and patches slime.
2. Fetches the base model (from the bucket when `BASE_S3_URI` is set, otherwise from Hugging Face)
   and the dapo-math-17k prompts.
3. Converts the checkpoint to Megatron `torch_dist` with slime's converter and model script
   (`SLIME_MODEL_SCRIPT`, default `qwen3-4B`).
4. Starts Ray.
5. Starts `train.py` with the flags above (GRPO, TP=2 on two GPUs by default).

### Creating the deployment

Copy [`deployment.example.json`](deployment.example.json) to `deployment.run.json` in this recipe's
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
first: these commands and step 9 use its `K` and the values in `run.env`. The block keeps your
filled-in `run.env`.

```bash
kubectl -n "$NAMESPACE" logs -f "job/cwtrain-$JOB_NAME"   # the trainer, including cwtrain's lines
cwtrain hotload list --deployment "$DEPLOYMENT_ID"         # one hot-load per published version
```

You cannot change a Job while it exists. Before the next run, delete the finished Job with
`kubectl -n "$NAMESPACE" delete job "cwtrain-$JOB_NAME"`, then render and apply step 9 again.
The `RUN_ID` in `run.env.example` takes the time of each render plus a random number. So the new
run publishes under new identities, as it must (see [Resuming a run](#resuming-a-run)). Your
shell's `$RUN_ID` comes from when you loaded `run.env`, not from the render. `cwtrain hotload list`
shows the identities that the Job uses.

### Cleaning up

The deployment holds its GPUs until you delete it. Run this block with the kubectl context you used
for the steps above (`kubectl config current-context`), and with the admin's object-storage
credentials. Only the admin may delete from the bucket:

```bash
( set -eu   # run from anywhere in the repository; stops at the first unset value or failed step
  cd "$(git rev-parse --show-toplevel)/recipes/training/slime/k8s"
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

Recipes in the same namespace share these objects unless their `run.env` names differ, and the
defaults are the same in every recipe. So clean up after the last of them. If the recipe created
the namespace and nothing else runs there, delete it too: `kubectl delete namespace "$NAMESPACE"`.

## Concurrency

Same as the miles recipe: see [Concurrency](../miles/README.md#concurrency). The deployment's
`autoscaling.concurrency` caps the sequences each replica decodes at once (vLLM's `--max-num-seqs`;
256 in the example). `ROLLOUT_CONCURRENCY` in `run.env` (default 16) sets how many group requests
the trainer keeps in flight. Each request asks for `--n-samples-per-prompt` completions.

## Sequence of one training step (delta mode)

| Step | Who | What |
| --- | --- | --- |
| baseline | slime rank 0 | Captures `--hf-checkpoint` bytes as the delta baseline and calls the hook with the delta root |
| baseline | hook | Publishes `--hf-checkpoint` as full snapshot `<run>-v000000`, unless a completed hot-load with that identity already exists; creates the weight mirror |
| sync N | slime ranks | Diff against the previous sync, write `weight_v00000N/` (shards + index with `base_version`) |
| sync N | hook (rank 0) | Apply the delta to the mirror and check every changed tensor's Adler-32; upload to `<prefix>/<run>-v00000N/`, create an incremental hot-load with predecessor `<run>-v00000(N-1)`, and wait until the hot-load completes (every replica READY on `<run>-v00000N`) |
| sync N, N % FULL_EVERY == 0 | hook (rank 0) | Same, but publish a **full** snapshot rebuilt from the mirror (with per-tensor checksums) so the chain restarts |
| sync N | slime | Barrier, then continues with the next rollout batch |

The hook blocks rank 0 before slime's barrier, so training pauses until the deployment serves the
new version. Set `CWTRAIN_HOTLOAD_WAIT=0` to give up that guarantee for more throughput.

## Full snapshots every N versions, and what is verified

Same as the miles recipe: see
[Full snapshots every N versions](../miles/README.md#full-snapshots-every-n-versions-and-what-is-verified).
The hook keeps a byte-exact mirror of the served weights, verifies every delta against the
trainer's Adler-32 checksums, publishes a checksummed full snapshot every `FULL_EVERY` versions,
and writes per-version checksum manifests. A COMPLETED hot-load means every replica serves the
new version. `CWTRAIN_HOTLOAD_VERIFY=1` re-reads the replicas anyway.

## Resuming a run

On restart, slime captures its baseline again, so the hook is called with the delta root again.
The hook looks for a completed `<run>-v000000` and skips the upload if it finds one. Version
numbers also restart at 1: slime clears the delta directory, and the deltas are again relative to
`--hf-checkpoint`. So a resumed run must publish under a *new* run ID. With the same run ID, the
hook would try to reuse identities that already exist with different bytes. Always start a
resumed run with a fresh `CWTRAIN_HOTLOAD_RUN_ID`.

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
  delta is empty. `--apply-chat-template-kwargs '{"enable_thinking":false}'` fixes it;
  `run.env.example` sets it in `EXTRA_TRAIN_ARGS`.
- **Pick a verifier that can see the answer.** slime's `deepscaler` reward grades only the text
  after a `</think>` marker. So it returns 0 for every response when thinking is off, or when
  thinking is on but truncated. `--rm-type math` grades the boxed answer wherever it is.
- **The uploaded index names identities, not counters.** slime writes `000001` / `000000` as the
  index's `metadata.version` and `metadata.base_version`. The platform takes the delta's identity
  and its predecessor from the hot-load request, not from these fields. `upload_snapshot` still
  rewrites them to the snapshot identities during upload, so the index in the bucket agrees with
  the request. It uploads the shards and their per-tensor checksums unchanged.
- **A sync that changes nothing publishes nothing.** slime still writes `weight_vNNNNNN/` with an
  index and no shards. The hook logs it and reports `hotload/skipped_empty`. The skipped version is
  byte-identical, so the hook chains the next delta onto the last version the deployment received.
- **A replica that a failed replay left on `unknown` recovers on the next hot-load.** After a
  rejected delta, a replica can report weight version `unknown`. It stays paused until a later
  hot-load succeeds. Fix the cause, then create the hot-load again (see
  [Failure handling](#failure-handling)). `cwtrain hotload status <id>` shows each replica's state
  and error. If hot-loads still stall with `chain root unresolved`, the platform cannot tell which
  version a replica serves. The deployment's pods run on CoreWeave's side, so send CoreWeave
  support the deployment id and the hot-load's id.
- **A missing bucket policy is the most common reason weights fail to load.** The model download at
  deployment start and every hot-load read the bucket as the inference service account.

## Failure handling

Same as the miles recipe: see [Failure handling](../miles/README.md#failure-handling).

One symptom is specific to slime. If the log says `No updatable SGLang engines are running; skip
weight update`, the standalone-publish patch is not applied or `--update-weight-standalone-publish`
is missing. No hot-load is ever created.

## Verifying by hand

```bash
cwtrain hotload list --deployment <uuid>                 # newest first, with state
cwtrain hotload status <hot-load uuid>                   # replicas and their current identity
cwtrain hotload inspect /work/deltas/weight_v000003      # what the hook would publish
```

With the mirror on, `<delta dir>-hotload/manifests/<identity>.json` lists every tensor's Adler-32
for that version, which is what the engine's copy of that version hashes to. The bucket has the
same manifests under `<prefix>/_manifests/`.
