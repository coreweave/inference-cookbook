# Example: GRPO on Nemotron 3.5 Lightning with NeMo-RL

This example runs the [NeMo-RL recipe](../../recipes/nemo-rl/README.md) on Kubernetes: GRPO on
Nemotron 3.5 Lightning (30B-A3B, BF16) with the DAPO-Math-17k prompts. The rollouts come from a
Dedicated Inference deployment with one replica, and every refit is hot-loaded into it. It uses the
recipe's config, [`grpo_coreweave.yaml`](../../recipes/nemo-rl/grpo_coreweave.yaml), and its
staging Job, [`stage-nemotron-job.yaml`](../../recipes/nemo-rl/k8s/stage-nemotron-job.yaml), as
they are. The recipe's README explains the loop, its settings and its failure handling.

| | |
| --- | --- |
| Recipe | [NeMo-RL](../../recipes/nemo-rl/README.md): synchronous GRPO with the DTensor (Automodel) policy backend |
| Model | [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16), at the commit that `HF_REVISION` pins, without its MTP heads ([`prepare_model.py`](../../recipes/nemo-rl/prepare_model.py), which the staging Job runs) |
| Data | [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k), which NeMo-RL's `DAPOMath17K` dataset loads; the recipe's config names it |
| Reward | NeMo-RL's `math` environment, with `hf_math_verify` |
| Trainer | 8 GPUs of a `gd-8xh100ib-i128` node, eight-way expert parallelism; 30 GRPO steps (`NUM_STEPS`) of 8 prompts × 4 generations |
| Deployment | [`deployment.example.json`](deployment.example.json): `dynamo-vllm` on 2 H100 GPUs, 1 replica, `autoscaling.concurrency` 8 |

## What you need

Follow the Dedicated Inference
[getting-started guide](https://docs.coreweave.com/products/inference/getting-started) for the
account prerequisites, the gateway and the bucket policy. Then get these:

- **A hot-load-enabled deployment** (engine `dynamo-vllm`, with a `hotLoad` section). Use
  [`deployment.example.json`](deployment.example.json) and see
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
- **The NeMo-RL container**, `nvcr.io/nvidia/nemo-rl:v0.7.0` (amd64 and arm64 on NGC). The
  recipe's launch script patches `/opt/nemo-rl` in place and installs the `cwtrain` wheel into the
  project venv.
- **The base model, staged separately for each deployment.** Stage it into the bucket with the
  recipe's [`stage-nemotron-job.yaml`](../../recipes/nemo-rl/k8s/stage-nemotron-job.yaml). The
  deployment loads it from there. The trainer fetches the same objects from `BASE_S3_URI`, which
  `run.env.example` sets to that location (`s3://<bucket>/<model prefix>/<model slug>`). The
  recipe's launcher does not download from Hugging Face, so the trainer stops at start if the
  model is not there. The synchronizer publishes the first refit as a full snapshot, the root of
  the run's chain. So the deployment's copy does not have to stay byte-identical to the policy.
  But it sets the layout: the synchronizer refuses an export whose tensor names, dtypes or shapes
  differ from that checkpoint.

## Running it on Kubernetes

This example's `k8s/run.env.example`, with the recipe's `k8s/` directory and the files that the
training recipes share in [`k8s/`](../../../k8s/README.md) at the repository root, runs this
topology on a Kubernetes cluster with a GPU node of `INSTANCE_TYPE`, such as CoreWeave Kubernetes
Service (CKS). You need these tools:

- `kubectl`
- `envsubst` (GNU gettext; `brew install gettext` on macOS)
- `uv`
- the `aws` CLI
- the `cwtrain` CLI. To install it, run `uv tool install ./libs/cwtrain` from the repository root.

Work in this example's `k8s/` directory, with a filled-in copy of `run.env.example` loaded into your
shell. In zsh, the default shell on macOS, the first line turns on `interactivecomments`, so
that the `#` comments in the blocks you paste are comments, as in bash:

```bash
[ -z "${ZSH_VERSION-}" ] || setopt interactivecomments
cd training/examples/nemo-rl-grpo-nemotron-3.5-lightning-30b-a3b/k8s
K=../../../../k8s                # the shared files: k8s/ at the repository root
R=../../../recipes/nemo-rl/k8s   # the recipe: its Jobs, scripts and kit
[ -e run.env ] || cp run.env.example run.env   # once; fill in every <placeholder>, render.sh refuses one
set -a; . ./run.env; set +a      # the commands below use its values
CTX=$(kubectl config current-context)
```

| Step | Command |
| --- | --- |
| 1. Namespace | `"$K/render.sh" run.env "$K/namespace.yaml" \| kubectl apply -f -` |
| 2. Trainer service account and object storage Secrets (AWS-style `config` + `credentials` files) | `kubectl -n "$NAMESPACE" create serviceaccount "$TRAINER_SERVICE_ACCOUNT"`, then `WORKLOAD_IDENTITY=1 AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$STORAGE_SECRET"` (with a trainer key instead, export it as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` and leave out `WORKLOAD_IDENTITY=1`); then, with the admin's key exported, `AWS_REGION=$BUCKET_ZONE "$K/make-storage-secret.sh" "$CTX" "$NAMESPACE" "$ADMIN_STORAGE_SECRET"` |
| 3. Wheel + config + patch ConfigMap | `(cd ../../../../libs/cwtrain && uv build --wheel) && "$R/make-kit.sh" "$CTX" "$NAMESPACE" "$KIT_NAME"` |
| 4. Bucket + policy (skip if done with `aws s3api`) | `"$K/render.sh" run.env "$K/make-bucket-job.yaml" \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m job/cwtrain-make-bucket`; it is done when `kubectl -n "$NAMESPACE" get job cwtrain-make-bucket` shows COMPLETIONS `1/1` |
| 5. Stage the base model into the bucket (required: the trainer reads it from there) | `"$K/render.sh" run.env "$R/stage-nemotron-job.yaml" \| kubectl apply -f -`, then `kubectl -n "$NAMESPACE" logs -f --pod-running-timeout=10m "job/cwtrain-stage-$JOB_NAME"`; it is done when `kubectl -n "$NAMESPACE" get job "cwtrain-stage-$JOB_NAME"` shows COMPLETIONS `1/1` (it retries once, so follow the logs again while it shows `0/1`) |
| 6. Create the deployment | [below](#creating-the-deployment); put its `id` and `rollout_base_url` in `run.env` as `DEPLOYMENT_ID` and `ROLLOUT_BASE_URL` |
| 7. Inference API key Secret | `kubectl -n "$NAMESPACE" create secret generic "$INFERENCE_KEY_SECRET" --from-file=api_key=<file>` |
| 8. CoreWeave API token Secret | `kubectl -n "$NAMESPACE" create secret generic "$API_TOKEN_SECRET" --from-file=token=<file>` |
| 9. Trainer | `"$K/render.sh" run.env "$R/trainer-job.yaml" \| kubectl apply -f -` |

Notes on the steps:

- **Step 7** uses the bearer token that the deployment's gateway accepts for inference requests.
  When the gateway uses CoreWeave authentication, this is a CoreWeave API token.
- **Step 8** uses a token with the Inference Admin role, because it creates hot-loads. One token
  can serve both steps. The trainer pod does not start until both Secrets exist. Reading the
  tokens from files keeps them out of your shell history.
- **Step 2**'s storage key must stay valid through staging, node provisioning and training. Use an
  object storage key that lasts for the whole run. If you copy temporary `credential_process`
  output into a static Secret, it does not refresh.

The recipe's `run-trainer.sh` installs the wheel, with its dependencies, into `/opt/nemo_rl_venv`.
Then it starts the recipe's [`launch.py`](../../recipes/nemo-rl/launch.py), which:

1. applies the patch,
2. installs the wheel into the prebuilt worker venvs (without dependencies),
3. fetches the base model,
4. runs `python examples/run_grpo.py --config grpo_coreweave.yaml` with the venv's interpreter.
   The config is the recipe's [`grpo_coreweave.yaml`](../../recipes/nemo-rl/grpo_coreweave.yaml).

Neither script uses `uv run`, because it re-syncs the venv and removes the wheel.

`EXTRA_OVERRIDES` is split on whitespace, one override per word. Quotes reach Hydra unchanged. So
write list values without spaces: `policy.generation.stop_strings=["</answer>","<|end|>"]`. The
unit tests for the launcher's argument building are in the recipe's `tests/`.

### Creating the deployment

1. Copy [`deployment.example.json`](deployment.example.json) to `deployment.run.json` in this
   example's `k8s/` (git-ignored).
2. Fill in the gateway id, instance type, bucket and unique deployment prefix. Set `model.path` to
   `$MODEL_PREFIX/$MODEL_SLUG` from the same `run.env` that you used for staging.
3. The example pins `dynamo-vllm` to `0.30.0+crwv.5`. Make sure that the gateway zone has this
   version with hot-load support. Put your API token in `COREWEAVE_API_TOKEN` (or `CW_API_TOKEN`,
   as the CoreWeave docs set it). Then `cwtrain deployment parameters` lists the runtime versions,
   instance types and engine options for your organization. Add `--gateway <gateway id>` to show
   only that gateway's zone.

```bash
cwtrain deployment parameters
[ -e deployment.run.json ] || cp ../deployment.example.json deployment.run.json   # then fill it in
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
first: these commands and step 9 use its `K` and `R` and the values in `run.env`. The block keeps
your filled-in `run.env`.

```bash
kubectl -n "$NAMESPACE" logs -f "job/cwtrain-$JOB_NAME"   # the trainer, including cwtrain's lines
cwtrain hotload list --deployment "$DEPLOYMENT_ID"         # one hot-load per published version
```

You cannot change a Job while it exists. Before the next run, delete the finished Job with
`kubectl -n "$NAMESPACE" delete job "cwtrain-$JOB_NAME"`. Then render and apply step 9 again.

The `RUN_ID` in `run.env.example` uses the time of each render plus a random number. So each new
run publishes under new identities, as it must. For what a restart under the same id does, see
the recipe's
[Sequence of one training step](../../recipes/nemo-rl/README.md#sequence-of-one-training-step).
Your shell's `$RUN_ID` comes from when you loaded `run.env`, not from the render.
`cwtrain hotload list` shows the identities that the Job uses.

### Cleaning up

The deployment holds its GPUs until you delete it. Run the shared [`k8s/cleanup-
run.sh`](../../../k8s/cleanup-run.sh) with the kubectl context that you used for the steps above
(`kubectl config current-context`), and with the admin's object-storage credentials. Only the admin
may delete from the bucket. If another deployment uses one of this deployment's snapshots as its
base model, wait until that deployment is gone before running it. From anywhere in the repository:

```bash
R="$(git rev-parse --show-toplevel)" &&
  "$R/k8s/cleanup-run.sh" \
  "$R/training/examples/nemo-rl-grpo-nemotron-3.5-lightning-30b-a3b/k8s/run.env" INFERENCE_KEY_SECRET
```

The script deletes the deployment, the Jobs (the trainer's, the staging and bucket Jobs), the kit
ConfigMap, the trainer's service account, and the storage, admin and API-token Secrets; each extra
argument names the `run.env` variable of one more Secret — here the inference key. In the bucket it
removes only the exact snapshot identities returned by the deployment's hot-load history:
`<prefix>/<identity>/` and the corresponding `_manifests/<identity>.json`; it does not expand a run
name into a wildcard. The prefix comes from `deployment.run.json`, or from `HOTLOAD_PREFIX` in
`run.env` if you created the deployment another way.

It does not delete the namespace or the trainer's org access policy. You can run it again after a
partial cleanup. But a deleted deployment lists no hot-loads: if the deployment is already gone,
identify and remove its exact snapshot paths by hand (`aws s3 ls` the hot-load prefix), as for
uploads that never reached the hot-load API, which are not in the history.

The staged base model and the bucket Job's write check (`_probe`) remain in the bucket. Other
deployments can use the same model path even when they run in different Kubernetes namespaces.
Remove those shared objects separately only after every deployment using them is gone.

Examples that share a namespace also share these objects, unless their `run.env` names differ.
The defaults are the same in every example, so clean up after the last one. If this example
created the namespace and nothing else runs there, delete it too:
`kubectl delete namespace "$NAMESPACE"`.

## Concurrency

The recipe's [Concurrency](../../recipes/nemo-rl/README.md#concurrency) section explains the two
settings that decide how many sequences are sampled at once. In this example,
[`deployment.example.json`](deployment.example.json) sets `autoscaling.concurrency` to 8, and
`run.env.example` keeps `ROLLOUT_CONCURRENCY` at the config's 32. With these settings (8 prompts ×
4 generations per step, one replica at 8), the 32 sequences of a step run in groups of at most
eight.

## Pitfalls

- **Turn off Nemotron thinking** (`tokenizer.chat_template_kwargs: {enable_thinking: false}`).
  Otherwise the responses fill the token budget before the boxed answer. The recipe's config
  turns it off.
