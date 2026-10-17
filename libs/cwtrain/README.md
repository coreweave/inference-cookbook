# cwtrain

[![PyPI](https://img.shields.io/pypi/v/cwtrain)](https://pypi.org/project/cwtrain/) [![Python](https://img.shields.io/pypi/pyversions/cwtrain)](https://pypi.org/project/cwtrain/)

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
  OpenAI-compatible endpoint, the post-write hook for slime and miles, and the generation backend,
  weight synchronizer and policy-worker extension for NeMo-RL.

The [CoreWeave Inference Cookbook](https://github.com/coreweave/inference-cookbook) has the recipes
that use `cwtrain` with miles, slime, NeMo-RL and Axolotl, and examples that run them end to end.

## Install

```bash
pip install cwtrain
```

This also installs the `cwtrain` CLI. The framework integrations need no extras: install
`cwtrain` into the framework's environment, which provides what they import (torch, and for
NeMo-RL also safetensors).

Then point it at your account and bucket. `COREWEAVE_API_TOKEN` is a CoreWeave API token with the
Inference Admin role. The AWS keys are your CoreWeave AI Object Storage credentials, and
`AWS_REGION` is the bucket's CoreWeave Availability Zone, such as `US-EAST-04A`:

```bash
export COREWEAVE_API_TOKEN=...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=...
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

Your organization must own the bucket. Its bucket policy needs three statements:

- the one from the Dedicated Inference docs
  ([Grant inference access to your bucket](https://docs.coreweave.com/products/inference/models#grant-inference-access-to-your-bucket)),
  which lets the canonical inference service account list and read the bucket;
- your organization's statement, which keeps list and read access for your users and the trainer.
  A bucket policy makes the bucket explicit-allow only, so without it all other access stops;
- one that lets the trainer write under the deployment's `pathPrefix`, with `s3:PutObject` and
  `s3:AbortMultipartUpload`. The trainer never deletes objects or changes the policy.

The cookbook's
[example bucket policy](https://github.com/coreweave/inference-cookbook/blob/main/k8s/bucket-policy.example.json)
has these statements. The bucket can be empty when you create the deployment, because the service
checks snapshots at each hot-load request.

With the trainer's storage credentials, `cwtrain hotload check --deployment <id>` checks those
permissions before you train. It lists the prefix, reads a key and starts a multipart upload that
it aborts, so it writes no object. When a call is refused, it names the S3 permission that is
missing.

## Quick start

### Create the deployment

Put a `CreateDeploymentRequest` body in a file, with its placeholders filled in. Each example in
the [cookbook](https://github.com/coreweave/inference-cookbook) has one, `deployment.example.json`.
`deployment parameters` lists the runtime versions, instance types and engine options that your
gateway's zone offers. Then create the deployment:

```bash
cwtrain deployment parameters --gateway <gateway id>
cwtrain deployment create --file deployment.json --wait
```

`create` prints the deployment as JSON, with its `id` and its `rollout_base_url` for rollouts.
With `--wait`, it returns when the deployment is READY. This takes minutes, and longer for large
models. If the deployment fails, or is not READY within `--timeout` (an hour by default), `create`
still prints it and exits with status 1.

### Publish weights

```python
from cwtrain import HotLoader, InferenceClient

loader = HotLoader(InferenceClient(), deployment_id="<deployment-uuid>")

# After saving a HuggingFace checkpoint for the current step:
hot_load = loader.publish("/ckpt/step_000100", identity="step_000100")
print(hot_load.state.name)  # COMPLETED
```

`publish` validates the directory and uploads it to the deployment's bucket and prefix. Then it
creates the hot-load and blocks until every replica serves the new weights. With `wait=False`, it
returns once the hot-load is created; call `loader.wait(hot_load.id)` to wait for it later.

### From the shell

```bash
cwtrain hotload publish --deployment <uuid> --checkpoint /ckpt/step_000100 --identity step_000100
cwtrain hotload list --deployment <uuid>
cwtrain hotload status <hot-load uuid>
cwtrain deployment delete <uuid> --wait
```

The deployment holds its GPUs until you delete it. `cwtrain --help` lists the commands, and
`cwtrain <group> <command> --help` shows a command's options.
