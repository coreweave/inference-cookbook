# cwtrain

`cwtrain` is the Python library and CLI for training against
[CoreWeave Dedicated Inference](https://docs.coreweave.com/products/inference/dedicated). In this
setup, the training loop gets its rollouts from a Dedicated Inference deployment, not from its own
inference engines. The library covers each part of the loop:

- **The deployment** (`cwtrain.infra`, `cwtrain deployment ...`): create the deployment that a run
  samples from, wait until it serves, find the rollout URL, and delete it at the end.
- **Weight updates** (`cwtrain.hotload`, `cwtrain hotload ...`): after each training step, publish
  the trainer's weights to the running deployment as a full checkpoint or a compressed delta. Then
  wait until every replica serves them. This uses checkpoint hot-load from
  [RL Rollouts](https://docs.coreweave.com/products/inference/rl-rollouts), so no redeploy is
  needed.
- **Rollouts and frameworks** (`cwtrain.integrations`): a rollout sampler for the deployment's
  OpenAI-compatible endpoint, the post-write hook for slime and miles, and the generation backend
  and weight synchronizer for NeMo-RL.

The [NeMo Gym evaluation recipe](https://github.com/coreweave/inference-cookbook/blob/main/evaluation/recipes/nemo-gym/README.md)
uses the library without hot-load. `InferenceClient` and `cwtrain.infra.lifecycle` create and
clean up a temporary deployment. `RolloutConfig` holds the endpoint credentials and concurrency.

## Install

Install `cwtrain` from [PyPI](https://pypi.org/project/cwtrain/):

```bash
pip install cwtrain
```

This also installs the `cwtrain` CLI. The framework integrations need no extras: install
`cwtrain` into the framework's environment, which provides what they import (torch, and for
NeMo-RL also safetensors). To install the unreleased code on `main`, install from the repository
instead:

```bash
pip install "cwtrain @ git+https://github.com/coreweave/inference-cookbook#subdirectory=libs/cwtrain"
```

Then point it at your account and bucket:

```bash
export COREWEAVE_API_TOKEN=...        # CoreWeave API token with the Inference Admin role
export AWS_ACCESS_KEY_ID=...          # CoreWeave AI Object Storage credentials
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=<bucket zone>      # the bucket's CoreWeave Availability Zone, e.g. US-EAST-04A
```

When `COREWEAVE_API_TOKEN` is not set, `cwtrain` reads the token from `CW_API_TOKEN`, the variable
that the CoreWeave documentation exports.

## Prerequisites

RL Rollouts is in preview and is enabled per organization. To turn it on, contact CoreWeave
support and give your organization ID. Until it is on, creating a deployment with a `hotLoad`
section fails with `hot-load is not enabled for this organization`.

Create a new deployment with hot-load enabled. You can set hot-load only when you create a
deployment; you cannot add it to an existing one. The deployment needs:

- engine `dynamo-vllm`;
- a `hotLoad` section naming the snapshot `bucket`, an optional `pathPrefix`, and
  `transitionMode: TRANSITION_MODE_ASYNC`;
- all of its gateways in one zone.

Your organization must own the bucket, and the bucket must have the bucket policy from the
Dedicated Inference docs
([Grant inference access to your bucket](https://docs.coreweave.com/products/inference/models#grant-inference-access-to-your-bucket)).
The policy gives the canonical inference service account `s3:ListBucket` and `s3:GetObject`. Add
a statement that lets your trainer's user write under the deployment's `pathPrefix`. The trainer
reads, writes with `s3:PutObject` (multipart for large files) and aborts its failed uploads. It
never deletes objects or changes the policy, so grant neither. The bucket can be empty when you
create the deployment, because the service validates snapshots at each hot-load request.

With the trainer's storage credentials, `cwtrain hotload check --deployment <id>` checks those
permissions before you train. It lists the prefix, reads a key and starts a multipart upload that
it aborts, so it leaves nothing behind. Trainer Jobs can run `python -m
cwtrain.integrations.preflight`, which does the same with the hooks' environment.

## Quick start

### Create the deployment

Put the `CreateDeploymentRequest` body in a file, such as the miles recipe's
[`deployment.example.json`](https://github.com/coreweave/inference-cookbook/blob/main/training/recipes/miles/deployment.example.json)
with its placeholders filled in, and create the deployment:

```bash
cwtrain deployment parameters --gateway <gateway id>   # versions, instance types, engine options
cwtrain deployment create --file deployment.json --wait
```

`create` prints the deployment as JSON, with its `id` and its `rollout_base_url` for rollouts.
With `--wait`, it returns when the deployment is READY. This takes minutes, and longer for large
models.

### Publish weights

```python
from cwtrain import HotLoader, InferenceClient

loader = HotLoader(InferenceClient(), deployment_id="<deployment-uuid>")

# After saving a HuggingFace checkpoint for the current step:
hot_load = loader.publish("/ckpt/step_000100", identity="step_000100")
print(hot_load.state.name)  # COMPLETED
```

`publish` validates the directory and uploads it to the deployment's bucket and prefix. Then it
creates the hot-load and blocks until every replica serves the new weights. To get the pending
hot-load back at once, pass `wait=False` and call `loader.wait(hot_load.id)` later. To receive a
`ProgressEvent` on every poll, pass `on_progress=`.

### Manage the deployment from Python

The CLI's deployment commands use `cwtrain.infra`. A launcher can do the same:

```python
import json

from cwtrain.infra import InferenceClient, lifecycle

with open("deployment.json") as f:
    spec = json.load(f)
with InferenceClient() as client:
    deployment = client.create_deployment(spec)
    try:
        deployment = lifecycle.wait_for_deployment(client, deployment.id)
        base_url = lifecycle.rollout_base_url(lifecycle.endpoints(client, deployment)[0])
        ...  # train, with rollouts sent to base_url
    finally:
        lifecycle.delete_deployment(client, deployment.id)  # it holds GPUs until deleted
```

### CLI

The same tasks from the shell:

```bash
cwtrain deployment create --file deployment.json --wait   # prints its id and rollout_base_url
cwtrain hotload publish --deployment <uuid> --checkpoint /ckpt/step_000100 --identity step_000100
# a delta already in the bucket under <pathPrefix>/step_000110/
cwtrain hotload trigger --deployment <uuid> --identity step_000110 --previous-identity step_000100
cwtrain hotload status <hot-load uuid>
cwtrain hotload list --deployment <uuid>
cwtrain hotload check --deployment <uuid>   # the trainer's access to the snapshot prefix
cwtrain deployment delete <uuid> --wait
```

The commands come in two groups, `cwtrain deployment` and `cwtrain hotload`:

| Command | What it does | Options (defaults) |
| --- | --- | --- |
| `deployment create` | Create a deployment from a `CreateDeploymentRequest` JSON body, such as a recipe's `deployment.run.json`, and print it with its gateways' `endpoints` and the `rollout_base_url` rollouts go to. A file that still holds a `<placeholder>` outside the free-form `runtime.engineConfig` and `runtime.engineEnv` is refused, and the request carries a client-chosen ID, so a retry cannot create a second deployment | `--file` (required; `-` reads stdin), `--wait` (until READY), `--timeout` (3600 s) |
| `deployment get <deployment id>` | Show a deployment, its hot-load settings, its endpoints and its `rollout_base_url` | |
| `deployment delete <deployment id>` | Delete a deployment | `--wait` (until it is gone), `--timeout` (600 s), `--ignore-not-found` |
| `deployment parameters` | List the runtime versions, instance types and engine options your organization can use | `--gateway` (repeatable: only what those gateways' zones offer) |
| `hotload inspect <dir>` | Validate a local checkpoint directory and print its type, files and size, and for a delta its predecessor; works offline, without credentials | `--previous-identity` |
| `hotload publish` | Upload a checkpoint, create the hot-load and wait for it | `--deployment`, `--checkpoint`, `--identity` (required); `--previous-identity`, `--reset-prompt-cache`, `--no-wait`, `--timeout` (1800 s), `--workers` (8 parallel uploads) |
| `hotload trigger` | Create the hot-load for a snapshot already in the bucket, and wait for it | `--deployment`, `--identity` (required); `--previous-identity`, `--reset-prompt-cache`, `--no-wait`, `--timeout` (1800 s) |
| `hotload wait <hot-load id>` | Block until a hot-load is completed, failed or canceled | `--timeout` (1800 s) |
| `hotload status <hot-load id>` | Show a hot-load and every replica's state | |
| `hotload list` | List hot-loads, newest first | `--deployment`, `--limit` (20) |
| `hotload check` | Check that the storage credentials can list, read and write the deployment's snapshot prefix, as a trainer does. It leaves nothing behind and names the S3 permission a refused call needs | `--deployment` (required) |

`--previous-identity` names the identity that a delta applies to. `trigger` needs it for a delta.
Without it, `publish` and `inspect` derive it from the delta's `base_version`, as described in
[Deltas](#deltas). `--reset-prompt-cache` requests the `PROMPT_CACHE_POLICY_RESET_ALL` policy, which
discards the KV cache computed with the earlier weights. Without it, the hot-load uses the API's
default, `PROMPT_CACHE_POLICY_PRESERVE`. `--no-wait` returns when the service accepts the
hot-load. Use `wait` to follow it later.

Put global options before the group, for example `cwtrain --verbose hotload list`:

- `--base-url`: the API origin. The default is `https://api.coreweave.com`.
- `--token`: the API token. The default is `$COREWEAVE_API_TOKEN`, else `$CW_API_TOKEN`.
- `--verbose`: debug logs, and tracebacks for operational failures.

CLI results are JSON on stdout. Logs and errors go to stderr. Exit codes are `0` for success,
`1` for operational failures (including failed hot-loads), and `2` for invalid arguments.
`--timeout` must be finite and greater than zero; `--workers` and `--limit` must be positive
integers.

## Hot-load

### What to expect

A hot-load completes when every replica serves the new weights. On large models this takes
minutes. By default, `publish` and the framework integrations wait for it. The first delta after a
full snapshot takes longer to load than the deltas that follow it. Every full snapshot uploads and
installs the whole model.

### Deltas

`cwtrain` detects a delta directory automatically. The identity that the delta applies to must be
the identity of a *completed* hot-load on the same deployment. Pass it as `previous_identity=`
(the CLI's `--previous-identity`). Without it, `publish` and `inspect_snapshot` take the delta's
numeric `base_version` and name it `weight_v{N:06d}`, which is how slime and miles name their
versions. For all other identities, such as `step_000100` above, always pass it.

Chains have limits. The API accepts at most 128 deltas after a full snapshot. Past that, it
returns `resolved snapshot chain is too long`. But a replica that starts with no peer to copy the
weights from replays the chain from object storage, and it handles at most 63 deltas. Publish a
full checkpoint at least every 64 versions. By default, the slime/miles hook and the NeMo-RL
synchronizer do this every 20 versions.

### Identities are immutable

Objects under an identity are never replaced. `upload_snapshot` (and so `publish` and
`HotLoader.upload`) writes the SHA-256 of every object into its metadata. When an identity already
has objects, each one must match the local file before anything is written. A matching partial
upload is completed, so a retry of an interrupted upload is safe and cheap. These cases raise
`SnapshotConflictError` and write nothing: different content, an object that cannot be verified,
or an object that is not part of the snapshot. To publish different weights, use a new identity.

### Rollback

The API has no rollback operation. `loader.rollback("step_000090")` gives one on the client side.
It creates a new hot-load for an identity from an earlier completed hot-load. It reuses that
snapshot type, so the chain resolves the same way. The objects must still be in the bucket.

### Concurrency

Only one hot-load can be active per deployment. By default, `HotLoader.trigger` and `publish` wait
for an active hot-load to finish. `InferenceClient.create_hot_load` raises
`HotLoadInProgressError` instead. Every create request has a client-generated `hot_load_id`. A
retry after a network failure returns the existing hot-load and does not start a second one.

## Snapshot layout

Every snapshot lives under `s3://<bucket>/<pathPrefix>/<identity>/`. An identity names one
version, such as `step_000100`: 1 to 128 characters forming a single path segment (no `/`, and
not `.` or `..`). The name `initial` is reserved for the deployment's base weights.

| Type | Contents |
| --- | --- |
| Full | A HuggingFace safetensors checkpoint: `config.json`, `model.safetensors.index.json`, `*.safetensors` shards. A checkpoint saved as one `model.safetensors` without an index, as `save_pretrained` saves a model below its shard size, gets an index built from that file's header, which the upload adds. |
| Delta (API type `INCREMENTAL`) | A zstd-compressed XOR delta with Adler-32 checksums against the previous snapshot: `model.safetensors.index.json` plus `*.safetensors` shards. slime writes this layout with `--update-weight-mode delta --update-weight-transport disk --update-weight-delta-encoding xor --update-weight-delta-checksum adler32`, and miles with `--update-weight-transfer-mode disk-delta --update-weight-delta-encoding xor --update-weight-delta-checksum adler32`. These flags select the format; each recipe lists the full set its framework needs, including slime's `--update-weight-standalone-publish` from the recipe's patch. |

`inspect_snapshot` checks the layout before the upload. It tells the two types apart by the
`delta_encoding` key in the index file's `metadata`. It checks that:

- the required files are there;
- the index is well formed;
- every shard in the index is a `*.safetensors` file directly inside the directory (subdirectories
  are not uploaded);
- a delta uses the one encoding, compression and checksum combination that the API accepts;
- the directory is not a LoRA adapter, which hot-load cannot serve: merge it into its base model
  first.

It does not read tensor contents, decode deltas, or verify checksums. So a snapshot that passes
can still be refused by the engine, whose checks are described below. A delta that changes no
tensors has no shards and cannot be uploaded. The framework integrations skip such versions.

### What lands in the bucket

A run that publishes a full snapshot and then one delta leaves this under the deployment's prefix.
`_manifests/` holds optional per-version checksum records (see `HotLoader.upload_manifest`).

```text
s3://my-bucket/rl/qwen3-4b/
├── step_000100/                          full snapshot
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors.index.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   └── tokenizer.json, tokenizer_config.json, ...
├── step_000101/                          delta against step_000100
│   ├── model.safetensors.index.json
│   └── model-00000-of-00001.safetensors
└── _manifests/
    └── step_000101.json
```

**Full snapshot.** An ordinary HuggingFace checkpoint. It can have per-tensor checksums, as the
snapshots from `WeightMirror.write_full_snapshot` do. Then the index says so, and every shard header
lists the Adler-32 of each tensor's bytes. The engine verifies each tensor on install:

```json
{
  "metadata": {"total_size": 8044936192, "checksum_format": "adler32"},
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors"
  }
}
```

```json
{
  "__metadata__": {
    "format": "pt",
    "model.embed_tokens.weight": "45971a16",
    "model.layers.0.mlp.down_proj.weight": "df031f77"
  },
  "model.embed_tokens.weight": {"dtype": "BF16", "shape": [151936, 2560], "data_offsets": [0, 777912320]},
  "model.layers.0.mlp.down_proj.weight": {"dtype": "BF16", "shape": [2560, 9728], "data_offsets": [777912320, 827719680]}
}
```

**Delta.** Only the tensors that changed. Each one is a standalone zstd frame of the XOR of its new
and old bytes, so the header records opaque `U8` blobs of the compressed length. The header's
`__metadata__` holds the Adler-32 of each tensor's *new* bytes. The engine checks these after it
applies the delta, so a wrong base or a corrupt blob fails the hot-load. slime and miles number
versions with their own counters (`000001`). The platform takes the delta's identity and its
predecessor from the hot-load request, not from the index. `upload_snapshot` still changes
`version` and `base_version` in the uploaded index to the snapshot identities, so the index names
the same versions as the request. It does not change the shards:

```json
{
  "metadata": {
    "version": "step_000101",
    "base_version": "step_000100",
    "delta_encoding": "xor",
    "compression_format": "zstd",
    "checksum_format": "adler32"
  },
  "weight_map": {
    "model.layers.0.mlp.down_proj.weight": "model-00000-of-00001.safetensors"
  }
}
```

```json
{
  "__metadata__": {"model.layers.0.mlp.down_proj.weight": "db401f66"},
  "model.layers.0.mlp.down_proj.weight": {"dtype": "U8", "shape": [1704248], "data_offsets": [0, 1704248]}
}
```

The checksums, sizes and offsets above are examples. A tensor that is not in a delta's
`weight_map` is unchanged. The delta format is slime's disk-delta convention on top of plain
safetensors. Its reference is slime's
[delta weight sync documentation](https://github.com/THUDM/slime/blob/main/docs/en/advanced/delta-weight-sync.md).

## Layers

| Layer | Use it when |
| --- | --- |
| `cwtrain.infra.InferenceClient` | You want the raw API, one call per method. Gateways: `create_gateway`, `get_gateway`, `list_gateways`, `delete_gateway`, `get_gateway_parameters`. Deployments: `create_deployment`, `get_deployment`, `list_deployments`, `delete_deployment`, `get_deployment_parameters`. Hot-loads: `create_hot_load`, `get_hot_load`, `list_hot_loads`, `list_hot_load_replicas`, `cancel_hot_load`. They return typed `Gateway`, `Deployment` and `HotLoad` objects, each with the wire object in `raw`. Every create carries a client-chosen ID, so a retry returns what an earlier attempt created rather than creating a second resource. |
| `cwtrain.infra.lifecycle` | You manage the deployment from code: `wait_for_gateway` and `wait_for_deployment` block until READY, `delete_gateway` and `delete_deployment` until it is gone, and `endpoints` and `rollout_base_url` give the URL rollouts go to. |
| `inspect_snapshot` / `upload_snapshot` | You manage uploads yourself, for example from a post-write hook that already has the checkpoint path. |
| `HotLoader` | You want one call per training step. |

API errors map to the API's status codes and are in `cwtrain.errors` (`NotFoundError`,
`AlreadyExistsError`, `FailedPreconditionError`, `HotLoadInProgressError`, ...). Every error that
the library raises derives from `cwtrain.errors.CWTrainError`.

## Integrations

| Framework | Entry point | Tested with | Recipe |
| --- | --- | --- | --- |
| miles | `--custom-update-weight-post-write-path cwtrain.integrations.slime.hook` plus the recipe's rollout function | `radixark/miles:v0.1.1` | [`training/recipes/miles/`](https://github.com/coreweave/inference-cookbook/blob/main/training/recipes/miles/README.md) |
| slime | same hook, `--update-weight-standalone-publish` (patch in the recipe) plus the recipe's rollout function | slime `8ee9c1e` | [`training/recipes/slime/`](https://github.com/coreweave/inference-cookbook/blob/main/training/recipes/slime/README.md) |
| NeMo-RL | `policy.generation.backend: coreweave` (`cwtrain.integrations.nemo_rl.CoreWeaveGeneration`), `HotLoadWeightSynchronizer` attached to it, `CoreWeaveDTensorPolicyWorker` as `worker_extension_cls_fqn`; the recipe's patches register the backend and the optional NeMo Gym rollouts | NeMo-RL `v0.7.0` | [`training/recipes/nemo-rl/`](https://github.com/coreweave/inference-cookbook/blob/main/training/recipes/nemo-rl/README.md) |

Install `cwtrain` into the framework's environment. The integrations add no dependencies of their
own: the NeMo-RL recipe, for example, installs the plain wheel into NeMo-RL's virtual environment,
and into its policy-worker environments without dependencies. Imported outside NeMo-RL's
environment, `cwtrain.integrations.nemo_rl` fails with an error that names the missing module and
the NeMo-RL version it is tested with.

`cwtrain.integrations.slime.hook` publishes each `weight_vNNNNNN` directory that the trainer
writes. In delta mode, it does the following:

1. It publishes `--hf-checkpoint` as the full chain root.
2. It publishes every delta as an incremental snapshot chained to the previous version.
3. Every `CWTRAIN_HOTLOAD_FULL_EVERY`-th version (default 20), it publishes a full snapshot that it
   rebuilds from a local mirror (`cwtrain.hotload.WeightMirror`).

You configure the hook with `CWTRAIN_HOTLOAD_*` environment variables. It blocks rank 0 until the
hot-load completes. The platform reports completion only when every replica serves the new
weights.

`cwtrain.integrations.rollout` is the rollout sampler that every recipe uses, for all frameworks.
It sends one `/v1/completions` request per GRPO group to the deployment's gateway or in-cluster
frontend. It retries failed requests. Where the endpoint supports them, it uses token-id prompts
and exact response token IDs. `RolloutConfig` reads its settings (endpoint, model, API key and
concurrency) from `CWTRAIN_ROLLOUT_*` environment variables.

`cwtrain.hotload.delta` has two functions that the NeMo-RL synchronizer uses to publish.
`write_delta` turns two full HF checkpoints into the zstd XOR delta format that slime and miles
write. `write_full_snapshot` copies a checkpoint into a checksummed full snapshot, optionally with
only the named tensors. Neither needs torch or the safetensors package.

## Configuration

| Setting | Source | Default |
| --- | --- | --- |
| API token | `InferenceClient(token=...)`, the CLI's `--token`, `COREWEAVE_API_TOKEN`, or `CW_API_TOKEN` (the name the CoreWeave documentation uses) | required |
| API origin | `InferenceClient(base_url=...)`, the CLI's `--base-url`, or `COREWEAVE_API_URL` for the slime/miles hook and the NeMo-RL synchronizer | `https://api.coreweave.com` |
| Object storage endpoint | `s3_client(endpoint_url=...)`, then `CW_S3_ENDPOINT`, then the AWS configuration (`AWS_ENDPOINT_URL_S3`, `AWS_ENDPOINT_URL`, or the profile's `endpoint_url` or `services` section), even an AWS endpoint; `ignore_configured_endpoint_urls` turns the AWS configuration's endpoints off | `https://cwobject.com`; inside a CoreWeave cluster `http://cwlota.com` caches reads but does not speed up uploads |
| Object storage credentials | standard boto3 chain | |
| Object storage region: the bucket's CoreWeave Availability Zone | `s3_client(region_name=...)`, `AWS_REGION`, `AWS_DEFAULT_REGION`, or the profile's `region` | required |
| Hot-load integrations | `CWTRAIN_HOTLOAD_*`: deployment, run ID, waiting, timeout, full-snapshot interval, mirror and prompt-cache policy; the full list is in the docstring of `cwtrain.integrations.slime` | see the docstring |
| Rollout sampler | `CWTRAIN_ROLLOUT_*`: endpoint, model, API key or key file, concurrency, timeouts and retries, log-probs and token ids; the full list is in the docstring of `cwtrain.integrations.rollout` | see the docstring |

## Test servers

`InferenceClient` accepts extra request headers (`headers=`) and a TLS verification override
(`verify=`). The slime/miles hook and the NeMo-RL synchronizer read them from
`CWTRAIN_API_HEADERS` (a JSON object) and `CWTRAIN_API_INSECURE` (`1`, `true`, `yes` or `on`).
These settings are for tests against servers other than the CoreWeave API. The CoreWeave API needs
only the Bearer token, so leave both unset when you use it.

## Contributing

[CONTRIBUTING.md](https://github.com/coreweave/inference-cookbook/blob/main/CONTRIBUTING.md)
describes how to run the library's checks, how to publish releases, and the live hot-load test.
