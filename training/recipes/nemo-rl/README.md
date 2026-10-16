# Recipe: NeMo-RL GRPO with rollouts on a CoreWeave hot-load deployment

Train with [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) and sample rollouts from a CoreWeave
Dedicated Inference deployment. The deployment receives every refit by checkpoint hot-load.
NeMo-RL starts no inference engines:

- A generation backend samples from the deployment's OpenAI-compatible endpoint.
- A weight synchronizer exports the policy and hot-loads it before each rollout round, including
  the update from the preceding optimizer step.

Training pauses until every replica serves the new version, so the next rollout is on-policy.

![One training step: the trainer exports weights, the cwtrain synchronizer uploads a full or delta snapshot to Object Storage, creates a hot-load and polls until every replica is READY; the next rollout batch samples fresh weights from the OpenAI-compatible endpoint.](../../../docs/hotload-flow.drawio.svg)

The [slime](../slime/README.md) and [miles](../miles/README.md) recipes run the same loop. Those
frameworks expose a post-write hook. NeMo-RL has two seams instead, and this recipe uses both.

## Running it

Each example runs this recipe end to end on Kubernetes, with what you need, the steps and the
cleanup:

- [`nemo-rl-grpo-nemotron-3.5-lightning-30b-a3b`](../../examples/nemo-rl-grpo-nemotron-3.5-lightning-30b-a3b/README.md):
  GRPO on Nemotron 3.5 Lightning with the DAPO-Math-17k prompts.

An example's `run.env` holds the settings that this README names. This recipe's `k8s/` directory
holds the trainer Job (`trainer-job.yaml`), the script that the Job runs (`run-trainer.sh`), the Job
that stages the base model (`stage-nemotron-job.yaml`) with its script (`stage-nemotron.sh`), and
`make-kit.sh`, which bundles the cwtrain wheel and the recipe's scripts, config and patches into the
kit ConfigMap.

## How it plugs into NeMo-RL

| NeMo-RL seam | What the recipe provides |
| --- | --- |
| `policy.generation.backend` | `coreweave`: `cwtrain.integrations.nemo_rl.CoreWeaveGeneration`, a `GenerationInterface`. It turns NeMo-RL's token-id batches into `/v1/completions` requests (prompts as token ids, `n` per distinct prompt). It returns `output_ids`, `logprobs`, lengths and `truncated` in the same shape as the vLLM backend. It needs no GPUs and no Ray workers. |
| `policy_generation.weight_synchronizer` | `HotLoadWeightSynchronizer`. On every refit it exports the weights in HF layout and publishes a full snapshot (first refit and every `full_every`) or a zstd XOR delta against the previous export. Then it hot-loads the snapshot and waits until every replica is READY on the new identity. Per-version checksum manifests go next to the snapshots. |
| `policy.worker_extension_cls_fqn` | `CoreWeaveDTensorPolicyWorker` (and a Megatron counterpart): the stock policy worker plus an `export_hf_checkpoint` RPC. It reuses the HF-layout weight generator of NeMo-RL's NCCL refit, so TP/PP shards are gathered the same way. The refit casts every tensor to the policy dtype. The DTensor worker instead keeps each tensor in the dtype that the deployment's checkpoint stores. The Megatron worker writes the dtypes that Megatron-Bridge exports. Rank 0 writes the safetensors shards, the index and the checkpoint's non-weight files. |

NeMo-RL's registries are hardcoded. So
[`patches/0001-coreweave-generation-backend.patch`](patches/0001-coreweave-generation-backend.patch)
adds these changes to the `v0.7.0` release (the newest release with an arm64 container):

- A `coreweave` branch in `grpo.setup()`. It builds only the policy's Ray cluster (no inference
  cluster, no NCCL collective). It creates the policy with the cwtrain worker extension for the
  configured training backend, or with the `worker_extension_cls_fqn` you set. It registers that
  class's Ray Python environment (reusing the stock worker's prebuilt venv) and attaches the
  synchronizer.
- A short-circuit at the top of `refit_policy_generation`. When the generation object has a
  `weight_synchronizer`, it calls `sync_weights()` and returns. Newer NeMo-RL releases have this
  seam.

Nothing else in NeMo-RL changes. The loop, data, loss and checkpointing are stock.

### About the patch

The patch (about 120 changed lines) is necessary until NeMo-RL can load a generation backend and a
weight synchronizer from configuration. All of the work happens in the `cwtrain` library, which
has no patches. The launcher ([`launch.py`](launch.py)) checks the patch against the checkout in
the container before it applies it. If a patch fails, the launcher reverts what it applied and
stops the job with a message that names the tested version. So a different NeMo-RL never starts
half-patched.

### Tested versions

| Component | Version |
| --- | --- |
| NeMo-RL | [`v0.7.0`](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.7.0); the patch applies to the tag and to the checkout inside the container |
| Trainer image | `nvcr.io/nvidia/nemo-rl:v0.7.0` (arm64 and amd64) |
| Bundled training Gym | `0.4.0` |
| Inference engine | `dynamo-vllm` `0.30.0+crwv.5` |

## Configure

[`grpo_coreweave.yaml`](grpo_coreweave.yaml) inherits NeMo-RL's `grpo_math_1B.yaml` and overrides
what this topology needs. The model is `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`. The
trainer uses eight GPUs and eight-way expert parallelism, with sequence packing off. The example's
hot-load deployment requests two H100s, for the BF16 model and memory headroom during refits. The
trainer and the deployment must load the same prepared BF16 checkpoint.

The pinned NeMo-RL AutoModel ignores the optional `mtp.*` speculative-decoding heads.
[`prepare_model.py`](prepare_model.py) makes a separate checkpoint without those heads:

- It turns off MTP in the config.
- It keeps every main-model shard byte-identical.
- It rejects checkpoints that mix MTP and main-model tensors in one shard.

The staging Job runs this step before it uploads. Use the prepared copy for both the trainer and
the deployment, and keep speculative decoding off. Otherwise cwtrain's tensor-layout check rejects
the first export. Use `k8s/stage-nemotron-job.yaml`, which gives 80 GiB for the raw checkpoint.
The shared [`stage-model-job.yaml`](../../../k8s/stage-model-job.yaml) keeps its 60 GiB default.

Set a unique `DEPLOYMENT_RUN` in `run.env` for every new inference deployment, and stage its base
model under that prefix. Two deployments in one organization that use the same model path can
collide in the weight UID registry of the model exchange server. A new served-model name does not
prevent this. Keep the value the same while you stage, create and train one deployment.

The backend configuration is:

```yaml
policy:
  worker_extension_cls_fqn: cwtrain.integrations.nemo_rl.worker.CoreWeaveDTensorPolicyWorker  # optional; the patch defaults to it
  dtensor_cfg:
    env_vars: {PYTHONPATH: /work/py}   # the worker venv sees the extension without cwtrain's deps
  generation:
    backend: coreweave
    colocated: {enabled: false, resources: {gpus_per_node: null, num_nodes: null}}
    coreweave_cfg:
      base_url: ${oc.env:CWTRAIN_ROLLOUT_BASE_URL}     # the deployment's gateway endpoint, /v1
      model: ${oc.env:CWTRAIN_ROLLOUT_MODEL}
      api_key_file: ${oc.env:CWTRAIN_ROLLOUT_API_KEY_FILE,null}
      max_model_len: ${policy.max_total_sequence_length}
      deployment_id: ${oc.env:CWTRAIN_HOTLOAD_DEPLOYMENT_ID}
      run_id: ${oc.env:CWTRAIN_HOTLOAD_RUN_ID}          # unique per run; identities are <run>-vNNNNNN
      full_every: ${oc.env:CWTRAIN_HOTLOAD_FULL_EVERY,null}  # null: cwtrain's default, 20
      state_dir: /work/hotload                          # exports, deltas, manifests
```

Every `coreweave_cfg` key except `max_model_len` and `require_logprobs` can also come from the
`CWTRAIN_ROLLOUT_*` and `CWTRAIN_HOTLOAD_*` environment variables that the other recipes use. The
API token (`COREWEAVE_API_TOKEN`, from a Secret in the Job) and the object storage credentials come
only from the environment.

`coreweave_cfg.timeout` sets two timeouts: the rollout requests and the hot-load wait. To set them
separately, leave it unset and use `CWTRAIN_ROLLOUT_TIMEOUT` and `CWTRAIN_HOTLOAD_TIMEOUT`. In the
Kubernetes run, use `HOTLOAD_TIMEOUT` in `run.env`, next to `HOTLOAD_WAIT`, `HOTLOAD_VERIFY` and
`HOTLOAD_PROMPT_CACHE_POLICY`.

GRPO here is synchronous. So, unless you set it, the launcher sets
`CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY=reset_all`: each rollout round then starts without KV cache
from the previous weights.

The recipe runs the DTensor (Automodel) policy backend. The Megatron worker extension exists but is
not tested on a live deployment.

## Sequence of one training step

| Step | Who | What |
| --- | --- | --- |
| refit | synchronizer | `export_hf_checkpoint` on every policy worker (rank 0 writes `<state_dir>/exports/vN`, each tensor in the base checkpoint's dtype), verify the tensor names, dtypes and shapes equal the base checkpoint's, write the checksum manifest |
| refit | synchronizer | v0 and every `full_every`: checksummed **full** snapshot of the export; otherwise a zstd XOR **delta** against the previous export, chained to the last published identity |
| refit | synchronizer | upload, create the hot-load, and wait until it completes (every replica READY on the new identity) |
| rollout | generation backend | one request per distinct prompt with `n` = its repeats; token ids in, token ids out (or re-tokenized text plus the stop token), sampled-token log-probs when they align |
| train | NeMo-RL | stock GRPO step; the next refit publishes the result |

If a refit's export is byte-identical to the previous one, the synchronizer publishes nothing.

A restarted run that keeps its run id finds versions that it already published. Before the
synchronizer records or uploads anything under an identity, full or delta, it checks the export
against that identity's stored checksum manifest:

- Matching weights are reused without an upload. If the deployment has moved on to a later
  version, the synchronizer hot-loads them again first.
- Different weights stop the run with a request for a new run id. So a restart never overwrites a
  published version.

## Concurrency

Two settings control how many sequences are sampled at once:

- **The deployment's `autoscaling.concurrency`**, a per-replica limit. The platform passes it to
  vLLM as `--max-num-seqs`. So a replica decodes at most that many sequences at once, and later
  requests wait in a queue. vLLM never schedules more than its KV cache holds, so a higher value
  only moves the wait into the engine.
- **`ROLLOUT_CONCURRENCY`** in `run.env`, the number of requests that the generation backend keeps
  in flight. Each request is one prompt that asks for `grpo.num_generations_per_prompt`
  completions. The Job passes it as `CWTRAIN_ROLLOUT_CONCURRENCY`, which
  `coreweave_cfg.concurrency` reads. An empty value keeps the config's 32.

When requests × completions is more than replicas × `autoscaling.concurrency`, the extra sequences
wait for a free slot. Gym sends one completion per request.

## Pitfalls

- **`snapshot access check failed` at the start of the trainer Job means that the trainer may not
  use the hot-load prefix.** It needs `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` and
  `s3:AbortMultipartUpload` there, from `AllowTrainer` and from its own org access policy. Check
  that `AllowTrainer` names the trainer's principal (`TRAINER_PRINCIPAL_ARN`), and that
  `HOTLOAD_PREFIX` is the deployment's `hotLoad.pathPrefix`. With Workload Identity Federation,
  also check that the cluster's issuer is registered and that the Job runs as
  `TRAINER_SERVICE_ACCOUNT`. `cwtrain hotload check --deployment <id>` runs the same check with the
  credentials of your shell.
- **Token ids in, token ids out.** NeMo-RL trains on exact ids and uses generation log-probs for
  its importance-sampling correction. The backend sends prompts as token ids and asks for
  `return_tokens_as_token_ids`. Fallbacks:
  - If the endpoint rejects a token-id prompt, the backend retries it as text. After a text
    request succeeds, the rest of the run sends text.
  - If the endpoint drops the flag, the backend re-tokenizes the response. It restores the
    end-of-sequence or custom stop token that vLLM leaves out of the text. It keeps log-probs only
    when the endpoint's token strings match the re-tokenized tokens one for one.

  When `coreweave_cfg.require_logprobs` is on, a response without aligned log-probs stops the step,
  instead of feeding zeros to the correction. The config ties this key to
  `loss_fn.use_importance_sampling_correction`.
- **Stop criteria come from the config and the batch.** Every request sends `stop_strings` and
  `stop_token_ids` from `policy.generation`, merged with the batch's per-row stop strings. A
  matched stop string stays in the output, as with NeMo-RL's own backends. The gateway accepts at
  most four stop strings and four stop token ids per request. More configured stops fail at setup.
- **The export must match the deployment's tensor set.** The platform rejects a full snapshot with
  a tied `lm_head.weight` or an adapter tensor that the base checkpoint does not have. So the
  export is filtered to the base checkpoint's names, and the synchronizer stops if a name is
  missing. Dtypes must also match:
  - NeMo-RL's refit generator casts every tensor to the policy dtype. This would turn FP32 tensors
    (for example, MoE router biases) into bf16 and round their values. The export keeps each tensor
    in the base checkpoint's dtype instead. The synchronizer stops before it publishes if a dtype
    or shape differs.
  - The export casts only between F16, BF16, F32 and F64. A quantized base checkpoint (FP8 or
    integer weights with their scales) stops the export, because a plain cast ignores the scales
    and serves wrong weights.
  - The Megatron worker does not apply dtypes. It writes the tensors as Megatron-Bridge exports
    them. If the bridge rounds FP32 tensors, the synchronizer's check stops the refit, so no cast
    hides the rounding.
- **The worker extension must be importable in the worker venv.** Ray imports the worker class in
  NeMo-RL's per-class venv before any configured env vars apply. So `PYTHONPATH` does not reach
  that import. The launch script installs the wheel without dependencies into the container's
  prebuilt worker venvs (`$NEMO_RL_VENV_DIR/*`). A venv that NeMo-RL builds fresh needs the same
  step.
- **Rank 0 and the driver share `state_dir`.** On one node, an emptyDir is enough. Across nodes, use
  a shared filesystem, or pin the driver to rank 0's node.
- **Do not `uv run` after you install extra packages.** `uv run` syncs the environment back to the
  lockfile. Call the venv's interpreter directly, as the launch script does.
- **A traceback at exit is not a failed run.** NeMo-RL v0.7.0 shuts down Ray before the policy
  object's destructor runs. So the last lines of a successful run can be `Exception ignored in
  Policy.__del__ ... PythonFinalizationError`. The process still exits 0. Judge the run by the last
  `[cwtrain] hot-load` line and the Job status.

## Failure handling

See [Failure handling](../miles/README.md#failure-handling) in the miles recipe, with two
differences. NeMo-RL keeps no weight mirror, so there is no `MirrorError`. Its synchronizer chains
each delta onto the last version it published.

One NeMo-RL-specific check: if the log never shows `Using CoreWeave hot-load deployment for
generation`, the patch was not applied to the container's checkout. NeMo-RL then fell through to a
GPU backend.

## NeMo Gym rollouts

To use Gym in the Nemotron 3.5 Lightning DAPO run of the
[example](../../examples/nemo-rl-grpo-nemotron-3.5-lightning-30b-a3b/README.md):

1. Add this Hydra override to `EXTRA_OVERRIDES`:

   ```sh
   EXTRA_OVERRIDES="env.should_use_nemo_gym=true"
   ```

2. Rebuild the kit, so that it includes both patches.

The launcher then selects NeMo-RL's Gym runner. `cwtrain` converts the DAPO questions and answers
into a `NemoGymDataset` under `/work/hotload/nemo-gym`. It starts the
`math_with_judge_simple_agent` with math-verify scoring (the LLM judge is off). Gym uses the
configured CoreWeave rollout URL, served model and mounted API key file. The default, `false`,
keeps NeMo-RL's math environment.

Training uses NeMo-RL's bundled Gym revision and actor environments. Gym owns sampling and rewards.
NeMo-RL owns optimization. cwtrain keeps the blocking checkpoint hot-load between batches.
Asynchronous GRPO and nonblocking hot-load are rejected. The Gym actor resolves the credentials
itself, so they are not in its serialized config or Ray runtime environment.

The cwtrain Gym adapter formats prompts with the staged trainer tokenizer. It sends integer token
IDs to CoreWeave's `/v1/completions` endpoint. It requires exact generated token IDs and
sampled-token log-probs, including the sampled stop token, and it checks the returned prompt count.
It does not need a `/tokenize` endpoint or text re-tokenization.

Limits of the adapter:

- It supports one text-only completion per request. It rejects tool calls, structured outputs and
  multiple choices per request.
- The flag-only preset is single-turn DAPO math. For custom text environments, provide an explicit
  `NemoGymDataset`, a Gym data processor and `env.nemo_gym.config_paths`.
- Multi-turn trajectories also require exact preservation of the token prefix.

### Validation

An end-to-end run with Nemotron 3.5 Lightning BF16 completed two GRPO steps of 32 Gym trajectories
each. It used eight trainer H100s and a two-H100 inference replica. The first full checkpoint and
the trained delta between steps both hot-loaded. The serving replica's identity was verified before
sampling. The Job exited successfully, and generation still worked after the trained delta was
applied. This is a smoke test of training and weight propagation, not a model-quality benchmark.

For a short validation run, add `++data.train.max_samples=32` and `grpo.max_num_steps=2` to
`EXTRA_OVERRIDES`. The sample limit is optional. Without it, the launcher converts the complete
published DAPO train split. On clusters with spot nodes, the Job also needs a toleration for
`qos.coreweave.cloud/interruptable` with effect `NoExecute`.

For a Gym validation run, use your hot-load bucket and a unique prefix under `hotload/`, such as
`hotload/nemo-gym-validation/<run-id>`. The bucket policy lets the trainer write only under
`hotload/`:

1. Set the temporary deployment's `hotLoad.bucket` and `hotLoad.pathPrefix` to match. The base
   model can stay in its existing source bucket.
2. Before you allocate compute, run `cwtrain hotload check --deployment <id>` with the trainer's
   credentials. It lists, reads and starts a multipart upload under the prefix, then aborts it, so
   it leaves no object behind. `AllowTrainer` must cover the prefix.
3. Make sure that the inference service can read the prefix. The example policy lets it read the
   whole bucket.
4. After the run, use the admin's credentials to remove only that run's objects and any incomplete
   multipart uploads. Then remove its temporary deployment and gateway.

Do not create or delete a bucket for validation.
