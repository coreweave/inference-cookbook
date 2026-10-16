# Shared Kubernetes and storage files

The training recipes ([miles](../training/recipes/miles/README.md),
[slime](../training/recipes/slime/README.md), [NeMo-RL](../training/recipes/nemo-rl/README.md) and
[Axolotl SFT](../training/recipes/axolotl-sft/README.md)) run on Kubernetes with the files in this
directory. Each recipe's own `k8s/` directory holds only what is specific to it: `run.env.example`,
`trainer-job.yaml`, `run-trainer.sh` and `make-kit.sh`, and for NeMo-RL a staging Job and script of
its own.

## Using the files

Work in your recipe's `k8s/` directory, with your `run.env` there, as the recipe README's "Running
it on Kubernetes" describes. Its commands call these files by their path from that directory, which
the setup block sets as `K`:

```bash
cd training/recipes/miles/k8s
K=../../../../k8s
"$K/render.sh" run.env "$K/namespace.yaml" | kubectl apply -f -
```

`run.env`, your filled-in copy of the recipe's `run.env.example`, sets every value that the
manifests below use. The recipe's `make-kit.sh` bundles `stage-model.sh` and
`bucket-policy.example.json` from here into the kit ConfigMap (`KIT_NAME`), with the cwtrain wheel
and the recipe's own scripts. The Jobs read them from there. A recipe that stages its model another
way bundles its own script under the key `stage-model.sh` and renders its own Job: NeMo-RL uses
`stage-nemotron.sh` and `stage-nemotron-job.yaml`.

## Files

| File | What it does | What it needs |
| --- | --- | --- |
| [`render.sh`](render.sh) | Prints a manifest with the values from an env file filled in: `render.sh run.env manifest.yaml`. It refuses a `${VAR}` that the env file does not set, and a value that still holds a `<placeholder>`. | The env file and the manifest, and `envsubst` |
| [`namespace.yaml`](namespace.yaml) | The namespace for the recipe's Jobs, ConfigMap and Secrets | `NAMESPACE` |
| [`make-storage-secret.sh`](make-storage-secret.sh) | Creates an object storage Secret with AWS-style `config` and `credentials` files, from the key in your environment. With `WORKLOAD_IDENTITY=1`, the Secret holds no key: the trainer gets temporary credentials through Workload Identity Federation. | Kube context, namespace and Secret name, and optionally the endpoint and region, as arguments. The region (the bucket's Availability Zone) can also come from `AWS_REGION` or `AWS_DEFAULT_REGION`. The key comes from `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. |
| [`make-bucket-job.yaml`](make-bucket-job.yaml) | A Job that creates the bucket with the admin's key, and attaches `bucket-policy.example.json` from the kit ConfigMap, filled in. Running it again keeps the policy's other statements and the principals that an earlier run added. | `NAMESPACE`, `BUCKET`, `BUCKET_ZONE`, `PLATFORM_PRINCIPAL_ARN`, `ORG_ID`, `TRAINER_PRINCIPAL_ARN`, `ADMIN_PRINCIPAL_ARN`, `HOTLOAD_PREFIX`, `MODEL_PREFIX`, `CW_S3_ENDPOINT`, `ADMIN_STORAGE_SECRET`, `KIT_NAME` |
| [`stage-model-job.yaml`](stage-model-job.yaml) | A Job that runs `stage-model.sh` from the kit ConfigMap with the admin's key | `NAMESPACE`, `JOB_NAME`, `HF_MODEL`, `HF_REVISION`, `BUCKET`, `MODEL_PREFIX`, `MODEL_SLUG`, `CW_S3_ENDPOINT`, `ADMIN_STORAGE_SECRET`, `KIT_NAME` |
| [`stage-model.sh`](stage-model.sh) | Runs in that Job. It downloads `HF_MODEL` at `HF_REVISION` (a full commit SHA) from Hugging Face, and uploads it to `s3://BUCKET/MODEL_PREFIX/MODEL_SLUG/` with cwtrain's snapshot uploader. | The Job's environment |
| [`bucket-policy.example.json`](bucket-policy.example.json) | The bucket policy: the inference service account and your org read, the trainer writes only under the hot-load prefix, and only the admin deletes | `[BUCKET-NAME]`, `[ORG-ID]`, `[HOT-LOAD-PREFIX]`, `[OIDC-ISSUER-URL]`, `[NAMESPACE]`, `[SERVICE-ACCOUNT]` and `[ADMIN-USER-ID]`. The bucket Job fills them in from `run.env`. For `aws s3api`, copy it to `bucket-policy.run.json` in your recipe's `k8s/` (git-ignored) and fill them in there. |
| [`trainer-access-policy.example.json`](trainer-access-policy.example.json) | The org access policy for the trainer's service account: the token exchange, reading the bucket, and writing under the hot-load prefix | `[OIDC-ISSUER-URL]`, `[NAMESPACE]`, `[SERVICE-ACCOUNT]`, `[BUCKET-NAME]` and `[HOT-LOAD-PREFIX]`. Copy it to `trainer-access-policy.run.json` in your recipe's `k8s/` (git-ignored) and fill them in there. |
| [`find-wheel.sh`](find-wheel.sh) | Sourced by each recipe's `make-kit.sh`: finds the cwtrain wheel to bundle | `CWTRAIN_WHEEL`, else the newest wheel in `libs/cwtrain/dist` |

The "What you need" section of each recipe README explains the bucket policy, the trainer's
principal and the object storage credentials.

## Changing a file

Change these files here; no recipe keeps a copy.

`pytest k8s/tests` runs every training recipe's steps against these files, with recording stand-ins
for `kubectl`, `cwtrain` and `aws`. It also fails when another directory holds a copy of one of
them.
