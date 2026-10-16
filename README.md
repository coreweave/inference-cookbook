# CoreWeave Inference Cookbook

Runnable reinforcement-learning (RL) recipes for training against a live
[CoreWeave Dedicated Inference](https://docs.coreweave.com/products/inference/dedicated)
deployment, and `cwtrain`, the Python library and CLI that the recipes share.

The recipes use [RL Rollouts](https://docs.coreweave.com/products/inference/rl-rollouts), the
Dedicated Inference capability for serving rollouts with **checkpoint hot-load**. After every
training step the trainer publishes its new weights. The deployment swaps them in place on every
replica, and the next rollout batch comes from its OpenAI-compatible endpoint. The trainer runs
no inference engines of its own.

`cwtrain` does the CoreWeave side of a run. It creates the deployment that the run samples from,
and deletes it at the end. It publishes the trainer's weights to the deployment by checkpoint
hot-load and samples rollouts from its endpoint. It integrates with miles, slime and NeMo-RL. The
NeMo Gym evaluation recipe uses it to evaluate the NeMo-RL recipe's hot-load deployment.

![Flow of one training step: cwtrain uploads the new weights, requests a hot-load and waits until every replica serves them; the next rollouts come from the deployment's endpoint.](docs/hotload-flow.drawio.svg)

## Why train against a hot-load deployment

- **No inference engines in the trainer.** Rollouts come from a deployment that runs and scales
  on its own GPUs, behind the same OpenAI-compatible API as any other Dedicated Inference
  deployment.
- **On-policy rollouts.** A hot-load completes only when every replica serves the new weights, and
  the integrations wait for that before the next rollout batch.
- **Small, verified updates.** After one full checkpoint, each step uploads only the tensors that
  changed. It sends them as a compressed delta with a checksum per tensor. The engine verifies each
  changed tensor before it serves the new weights. A full checkpoint every N steps keeps the chain
  of deltas short.
- **Your framework, or your own loop.** The library integrates with miles, slime and NeMo-RL, and
  its CLI and Python API work from any other training loop.

## Recipes

A recipe wires one trainer or evaluation harness to Dedicated Inference: its launcher, any patches,
and the settings and modes it supports. It runs the framework's own training or evaluation loop. An
example runs a recipe with a concrete model, dataset and deployment, and a case study shows a real
problem solved with Dedicated Inference, with its results. The [examples](#examples) are in
`training/examples/`; case studies will go in `training/case-studies/`.
[CONTRIBUTING](CONTRIBUTING.md#what-goes-where) says what goes where.

| Recipe | Framework | Algorithm | Upstream changes | Tested with |
| --- | --- | --- | --- | --- |
| [miles](training/recipes/miles/README.md) | [miles](https://github.com/radixark/miles) | GRPO, synchronous or one step ahead | None: miles calls the `cwtrain` hook after each weight sync | `radixark/miles:v0.1.1` |
| [slime](training/recipes/slime/README.md) | [slime](https://github.com/THUDM/slime) | GRPO | Three small patches, applied by the launcher, so slime publishes weights without inference engines of its own | slime `8ee9c1e` |
| [NeMo-RL](training/recipes/nemo-rl/README.md) | [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) | GRPO, with NeMo-RL's math environment or a Gym environment that an example brings | Patches, applied by the launcher, that register the `coreweave` backend and optional Gym rollouts | NeMo-RL `v0.7.0` |
| [Axolotl SFT](training/recipes/axolotl-sft/README.md) | [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) | SFT, full or LoRA (merged) | None: stock `axolotl train`; cwtrain publishes the result as a full snapshot | Axolotl `v0.20.0` |
| [NeMo Gym](evaluation/recipes/nemo-gym/README.md) | [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) | Evaluation with scored rollouts: BrowseComp with an LLM judge | None: Gym runs on CPU against the NeMo-RL recipe's hot-load deployment, runs the agent and scores responses | NeMo Gym `v0.6.0` |

## Examples

Each example runs one training recipe on Kubernetes, end to end: what you need, the steps, and the
cleanup. It has its own settings (`run.env.example`) and deployment request. It uses its recipe's
manifests, and the setup that all examples share in [`k8s/`](k8s/README.md), with the example
bucket policy.

| Example | Recipe | Model | Data | Deployment |
| --- | --- | --- | --- | --- |
| [miles-grpo-qwen3-4b](training/examples/miles-grpo-qwen3-4b/README.md) | miles, GRPO | Qwen3-4B | DAPO-Math-17k | 2 replicas, 1 GPU each |
| [slime-grpo-qwen3-4b](training/examples/slime-grpo-qwen3-4b/README.md) | slime, GRPO | Qwen3-4B | DAPO-Math-17k | 2 replicas, 1 GPU each |
| [nemo-rl-grpo-nemotron-3.5-lightning-30b-a3b](training/examples/nemo-rl-grpo-nemotron-3.5-lightning-30b-a3b/README.md) | NeMo-RL, GRPO | Nemotron 3.5 Lightning 30B-A3B BF16 | SimpleQA, as a Gym search agent | 4 replicas, 2 H100 GPUs each |
| [axolotl-sft-qwen3-4b](training/examples/axolotl-sft-qwen3-4b/README.md) | Axolotl SFT, full or LoRA | Qwen3-4B | 2,000 rows of NuminaMath-CoT | 2 replicas, 1 GPU each |

## Before you start

- A CoreWeave account with Dedicated Inference access, and **RL Rollouts enabled for your
  organization**. Until it is enabled, creating a deployment with a `hotLoad` section fails with
  `hot-load is not enabled for this organization`.
- Inference GPU quota for the instance type you deploy on.
- A CoreWeave API token with the **Inference Admin** role, to create deployments and start
  hot-loads. Polling hot-loads needs only the Inference Viewer role.
- An [AI Object Storage](https://docs.coreweave.com/products/storage/object-storage) bucket that
  holds the base model. Its bucket policy must let the Dedicated Inference service account read the
  bucket ([Grant inference access to your bucket](https://docs.coreweave.com/products/inference/models#grant-inference-access-to-your-bucket))
  and let your trainer write snapshots under the hot-load prefix. The trainer needs no delete or
  policy rights, and a principal of its own: the recipes run it as a Kubernetes service account
  with Workload Identity Federation. **A bucket policy makes the bucket explicit-allow only.** Keep
  your organization's statement in the policy, or all other access to the bucket stops. The recipes'
  [`bucket-policy.example.json`](k8s/bucket-policy.example.json) has this structure.
- A gateway, and a deployment created with engine `dynamo-vllm` and a `hotLoad` section naming the
  bucket and path prefix for snapshots ([example request](training/examples/miles-grpo-qwen3-4b/deployment.example.json)).
  The recipes use `cwtrain deployment create` to create it from such a request. You can also use
  the API, or the **Hot loading** section of the Cloud Console's create drawer. You can enable
  hot-load only when you create a deployment; you cannot add it later. All of the deployment's
  gateways must be in one zone.
- Python 3.12 or newer.
- For the recipes: `kubectl` access to a CoreWeave Kubernetes cluster with GPU nodes for the
  trainer.

## Quick start: create a deployment and hot-load a checkpoint

Install the library and its CLI from [PyPI](https://pypi.org/project/cwtrain/):

```bash
pip install cwtrain
```

Point it at your account and bucket:

```bash
export COREWEAVE_API_TOKEN=...                           # API token with the Inference Admin role
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...   # AI Object Storage key that can write the bucket
export AWS_REGION=<bucket zone>                          # the bucket's Availability Zone, e.g. US-EAST-04A
```

`cwtrain` also reads `CW_API_TOKEN`, the name that the CoreWeave documentation uses, when
`COREWEAVE_API_TOKEN` is not set.

Create the deployment from a request body like an example's
[`deployment.example.json`](training/examples/miles-grpo-qwen3-4b/deployment.example.json), with
every `<placeholder>` filled in:

```bash
cwtrain deployment create --file deployment.json --wait
```

The command waits until the deployment is READY, which takes several minutes. Then it prints the
deployment as JSON. Its `id` is the `<deployment-uuid>` for the commands below. Its
`rollout_base_url` is where rollout requests go.

Publish a HuggingFace safetensors checkpoint directory (with `config.json` and
`model.safetensors.index.json`):

```bash
cwtrain hotload publish \
  --deployment <deployment-uuid> \
  --checkpoint /ckpt/step_000100 \
  --identity step_000100
```

`publish` validates the directory and uploads it to `s3://<bucket>/<pathPrefix>/step_000100/`.
The bucket and prefix come from the deployment's `hotLoad` section. Then `publish` requests the
hot-load and waits until every replica serves the new weights. The request gives the identity and
tells whether the snapshot is a full checkpoint or a delta. A delta also gives the identity that it
applies to.

When the snapshot is already in the bucket, `trigger` requests the hot-load without an upload, and
waits for it. Pass `--previous-identity` when the snapshot is a delta. Pass `--no-wait` to return
when the service accepts the hot-load:

```bash
cwtrain hotload trigger --deployment <deployment-uuid> \
  --identity step_000110 --previous-identity step_000100
```

The same from Python:

```python
from cwtrain import HotLoader, InferenceClient

loader = HotLoader(InferenceClient(), deployment_id="<deployment-uuid>")
hot_load = loader.publish("/ckpt/step_000100", identity="step_000100")
print(hot_load.state.name)  # COMPLETED
```

The deployment holds its GPUs until you delete it:

```bash
cwtrain deployment delete <deployment-uuid> --wait
```

## What to expect

- A hot-load completes when every replica serves the new weights. On large models this takes
  minutes. By default, the integrations pause training until it completes.
- The first delta after a full checkpoint takes longer to load than the deltas that follow it.
- Full checkpoints (the first one, and each periodic one that starts a new chain) upload and
  install the whole model.
- The integrations keep weights on the trainer's local disk. The slime and miles hook keeps a
  byte-exact copy of the served weights. The NeMo-RL synchronizer keeps the last published export
  next to the new one. Plan disk space for one or two extra copies of the checkpoint, plus one more
  while a full checkpoint is written.

## How it works

- Every published checkpoint lives at `s3://<bucket>/<pathPrefix>/<identity>/`. The API calls it a
  **snapshot**, and its **identity** (for example `step_000100`) names that version. An identity is
  never reused. Uploads verify the objects that are already there, so a retry is safe and different
  bytes are refused.
- A **full** snapshot is a HuggingFace safetensors checkpoint. A **delta** holds only the tensors
  that changed. Each is a zstd-compressed XOR against the previous version, with an Adler-32
  checksum of its new bytes. The engine verifies the checksum before it serves the weights.
- A full snapshot followed by deltas forms a **checkpoint chain**. When the integrations publish
  deltas, they start a new chain with a full snapshot every 20 versions by default. This limit
  matters because a replica that starts with no peer to copy from replays the whole chain from
  object storage.

The [library README](libs/cwtrain/README.md) covers the snapshot layout, the API layers, the chain
limits, and the configuration.

## Repository layout

```text
libs/cwtrain/                 Python library and CLI the recipes share: deployments, checkpoint
                              hot-load (uploads, deltas, the weight mirror), the rollout sampler,
                              and the framework integrations
k8s/                          Kubernetes and object storage setup that the training recipes share
training/recipes/miles/       GRPO with miles, with Kubernetes manifests
training/recipes/slime/       GRPO with slime, with Kubernetes manifests and the patches slime needs
training/recipes/nemo-rl/     GRPO with NeMo-RL, with Kubernetes manifests and the patches it needs
training/recipes/axolotl-sft/ SFT with Axolotl, published to the deployment as a full snapshot
training/examples/            One run of a training recipe each: model, data, deployment and steps
evaluation/recipes/nemo-gym/  Scored evaluation with NeMo Gym against the NeMo-RL hot-load deployment
docs/                         Diagrams
```

## Documentation and support

- Product documentation: [RL Rollouts](https://docs.coreweave.com/products/inference/rl-rollouts)
  and the [Dedicated Inference getting-started guide](https://docs.coreweave.com/products/inference/getting-started).
- This repository: the [library README](libs/cwtrain/README.md), and the README of each recipe
  and example.
- Bugs and questions about this repository: [GitHub issues](https://github.com/coreweave/inference-cookbook/issues).
- Security issues: see [SECURITY.md](SECURITY.md). Do not open a public issue.
- Account access, RL Rollouts enablement and GPU quota: CoreWeave support.

## Roadmap

Next: more evaluation recipes, and agent skills.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions are
accepted under the [CoreWeave CLA](CLA.md). You agree to it when you sign off every commit
(`git commit -s`).

## License

Licensed under the [Apache-2.0 license](LICENSE.md); the copyright notice is in [NOTICE](NOTICE).
