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

![The hot-load flow of this recipe, once per run, after training: the trainer Job runs axolotl train, and axolotl merge-lora after a LoRA run; launch.py checks the result and writes one full snapshot; cwtrain uploads it to AI Object Storage, creates a hot-load and polls until every replica is READY; the deployment's OpenAI-compatible endpoint then serves the fine-tuned weights.](../../../docs/hotload-flow-sft.drawio.svg)

The snapshot is also a plain Hugging Face checkpoint in your bucket, so an RL recipe can start
from it (see [SFT, then RL](#sft-then-rl)).

## Running it

Each example runs this recipe end to end on Kubernetes, with what you need, the steps and the
cleanup:

- [`axolotl-sft-qwen3-4b`](../../examples/axolotl-sft-qwen3-4b/README.md): a full fine-tune of
  Qwen3-4B on 2,000 rows of NuminaMath-CoT, with this recipe's `sft_coreweave.yaml`.

An example's `run.env` holds the settings that this README names. This recipe's `k8s/` directory
holds the trainer Job (`trainer-job.yaml`), the script that the Job runs (`run-trainer.sh`), and
`make-kit.sh`, which bundles the cwtrain wheel, the launcher and the Axolotl config into the kit
ConfigMap.

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
   example's cleanup finds this one.
5. **No prompt cache from the base weights.** Unless you set `HOTLOAD_PROMPT_CACHE_POLICY`, the
   launcher asks the platform to drop the KV cache computed under the previous weights
   (`reset_all`), so that a request right after the swap does not reuse cached prompt prefixes.
6. **RL Rollouts for your organization**, and a deployment created with engine `dynamo-vllm` and a
   `hotLoad` section (see the example's
   [What you need](../../examples/axolotl-sft-qwen3-4b/README.md#what-you-need)).

The snapshot is a full checkpoint, never a delta: the first snapshot that a deployment receives
must be full, because no delta can chain onto its base weights.

## Versions

| Component | Version |
| --- | --- |
| Axolotl | [`v0.20.0`](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.20.0): the trainer image `axolotlai/axolotl:0.20.0-py3.12-cu130-2.13.0@sha256:80ab8e2eb7a6f1987656a5b8c09e4c317e7481bcfb543674fd7414ef61a73c4b` (amd64 and arm64), `TRAINER_IMAGE` in the example's `run.env.example` |
| Inference engine | `dynamo-vllm` `0.30.0+crwv.5` |

An end-to-end run trained on two B200 GPUs, an amd64 node, against a deployment of two one-GPU
H100 replicas. A full fine-tune and a LoRA run each published a full snapshot of the base model's
398 tensors, which hot-loaded on both replicas, and the same greedy request answered differently
after each swap. The example's `INSTANCE_TYPE`, `gb300-4x` (arm64), is not yet validated.

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

[`sft_coreweave.yaml`](sft_coreweave.yaml) is the Axolotl config that the launcher runs: a full
fine-tune of Qwen3-4B on 2,000 rows of
[NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), one shard at a pinned
revision, with FSDP2. The
[`axolotl-sft-qwen3-4b`](../../examples/axolotl-sft-qwen3-4b/README.md) example runs it on two GPUs
for 30 steps. Its solutions box their final answers, as the RL recipes' `--rm-type math` reward
expects. When you change the config, keep these settings:

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

Axolotl sends telemetry unless `AXOLOTL_DO_NOT_TRACK` is set; the example's `run.env.example`
sets it to `1`.

## LoRA

A hot-load serves full weights only. So after a LoRA run, the launcher merges the adapter into the
base model with `axolotl merge-lora`, and publishes the merged model. Turn LoRA on in
`EXTRA_TRAIN_ARGS`:

```bash
EXTRA_TRAIN_ARGS='--adapter lora --lora-r 32 --lora-alpha 64 --lora-target-linear --no-fsdp-config.activation-checkpointing'
```

The merge gets the same options as the training run and writes `/work/out/merged`. With FSDP2,
Axolotl saves the adapter only, so the merge is the step that produces full weights.

Keep `--no-fsdp-config.activation-checkpointing`, which turns off the config's FSDP activation
checkpointing for the LoRA run. With it on, Axolotl v0.20.0 saves the adapter under the
checkpoint wrappers' module names (`..._checkpoint_wrapped_module...`), so `merge-lora` matches
none of its weights to the base model: it logs `Applied LoRA to 0/398 tensors`, and the merged
model is the base model.

## SFT, then RL

The snapshot is a plain Hugging Face checkpoint, so an RL recipe can start from it, on the
deployment that already serves it. In the `run.env` of an RL example, such as
[`miles-grpo-qwen3-4b`](../../examples/miles-grpo-qwen3-4b/README.md), set the snapshot as
`BASE_S3_URI`, and point the run at the SFT deployment:

```bash
BASE_S3_URI=s3://<bucket>/hotload/<RUN_ID>-v000030
DEPLOYMENT_ID=<the SFT deployment's id>
ROLLOUT_BASE_URL=<the SFT deployment's rollout_base_url>
ROLLOUT_MODEL=qwen3-4b-sft-hotload
```

`ROLLOUT_MODEL` is the SFT deployment's `model_name`; the RL example's default names a different
deployment. Skip the RL example's bucket, model-staging and deployment-creation steps: the SFT
run's bucket and deployment carry over.

The RL trainer then starts from exactly the fine-tuned weights. The RL hook publishes the weights
again as its run's chain root, `<rl-run>-v000000`, before the first delta. The RL example's
cleanup script then removes both runs' snapshots.

To evaluate the fine-tuned model, send requests to the deployment's endpoint for its
`model_name`. The [web-search-agent case study's evaluation](../../case-studies/web-search-agent/README.md#evaluation)
shows how to evaluate an existing deployment with NeMo Gym.

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
  bucket. Fix the cause, delete the Job, and render and apply it again: step 8 of the example's
  [Running it on Kubernetes](../../examples/axolotl-sft-qwen3-4b/README.md#running-it-on-kubernetes).
  If the fix is in a file of the kit, such as `sft_coreweave.yaml`, run step 3 first: the Job reads
  those files from the kit ConfigMap.
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
