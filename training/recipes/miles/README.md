# Recipe: miles RL with rollouts on a CoreWeave hot-load deployment

Train with [miles](https://github.com/radixark/miles) and serve rollouts from a CoreWeave
Dedicated Inference deployment. The deployment receives every weight update by checkpoint
hot-load. The trainer owns no inference engines:

1. miles writes a versioned weight directory after each sync.
2. The `cwtrain` post-write hook uploads it and triggers a hot-load.
3. The deployment swaps the weights in place on all replicas.
4. The next rollout batch samples the new weights through the deployment's OpenAI-compatible
   endpoint.

![One training step: the trainer writes weight_vNNNNNN, the cwtrain hook uploads it to Object Storage, creates a hot-load and polls until every replica is READY; the next rollout batch samples fresh weights from the OpenAI-compatible endpoint.](../../../docs/hotload-flow.drawio.svg)

The same loop with slime is in [`../slime/`](../slime/README.md).

## Running it

Each example runs this recipe end to end on Kubernetes, with what you need, the steps and the
cleanup:

- [`miles-grpo-qwen3-4b`](../../examples/miles-grpo-qwen3-4b/README.md): GRPO on Qwen3-4B with the
  DAPO-Math-17k prompts.

An example's `run.env` holds the settings that this README names. This recipe's `k8s/` directory
holds the trainer Job (`trainer-job.yaml`), the script that the Job runs (`run-trainer.sh`), and
`make-kit.sh`, which bundles the cwtrain wheel and the recipe's scripts into the kit ConfigMap.

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

- `CWTRAIN_HOTLOAD_WAIT=0`: do not block until the hot-load completes.
- `CWTRAIN_HOTLOAD_TIMEOUT`: seconds one hot-load may take (default 1800).
- `CWTRAIN_HOTLOAD_VERIFY=1`: re-check every replica after each hot-load.

The hook's module docstring lists all its settings (`cwtrain/integrations/slime.py`; miles uses
slime's hook contract). In the Kubernetes run, `run.env` sets these and the prompt-cache policy
(`HOTLOAD_WAIT`, `HOTLOAD_TIMEOUT`, `HOTLOAD_VERIFY`, `HOTLOAD_PROMPT_CACHE_POLICY`).

With `train.py` (the default), the loop is synchronous: each rollout round samples the weights
that the hook just hot-loaded. So, unless you set it, the launcher sets
`CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY=reset_all`. This asks the platform to drop the KV cache
computed under the previous weights, instead of reusing cached prompt prefixes after the swap.
`train_async.py` generates one step ahead and keeps the platform default, `preserve`.

Snapshot identities are `<RUN_ID>-vNNNNNN`. The platform keys weight versions by identity, so
never reuse a run ID for different weights.

## miles flags

miles uses slime's post-write hook contract, with different names for the disk-delta flags. miles
also runs the weight sync when it owns no engines. So this topology has **no SGLang engines at
all**:

- The deployment is the rollout engine.
- The disk-delta publisher runs on the trainer.
- The hook hot-loads each version.

miles never pauses, reloads or resumes an engine. The hot-load API runs the swap on all replicas.

```bash
python train.py ... \
  --rollout-num-gpus 0 \
  --rollout-function-path coreweave_rollout.CoreWeaveRolloutFn \
  --update-weight-transfer-mode disk-delta \
  --update-weight-disk-dir /work/deltas \
  --update-weight-local-checkpoint-dir /work/local-ckpt \
  --update-weight-delta-encoding xor \
  --update-weight-delta-checksum adler32 \
  --custom-update-weight-post-write-path cwtrain.integrations.slime.hook
```

Delta mode publishes zstd XOR deltas with Adler-32 checksums. This is the only delta format the
hot-load API accepts. The hook rejects any other encoding or checksum before it uploads.
`disk-delta` does not work with `--colocate`.

[`coreweave_rollout.py`](coreweave_rollout.py) is the rollout function. It is a thin miles adapter
over `cwtrain.integrations.rollout`:

- It sends one `/v1/completions` request per GRPO group (`n` choices) to the deployment, through
  the deployment's gateway.
- The prompt goes out as the token ids that the trainer scores.
- Responses come back as the exact sampled token ids. Only when the endpoint does not report them
  does it re-tokenize the text.
- Rewards come from miles' own `--rm-type`.
- When the endpoint's sampled-token log-probs align with the response tokens, including the token
  that ended the response, it attaches them as `rollout_log_probs`. With `--use-tis`, a group where
  they do not align is sampled again, up to twice, before the rollout fails. Without it, they stay
  unset, and the trainer recomputes log-probs either way.

Configure it with `CWTRAIN_ROLLOUT_BASE_URL`, `CWTRAIN_ROLLOUT_MODEL` and, optionally,
`CWTRAIN_ROLLOUT_API_KEY_FILE`. The hook blocks until every replica serves the new version, so the
next rollout is on-policy without an engine handshake.

## Truncated importance sampling

The deployment and the trainer compute slightly different log-probs for the same tokens, so even
fresh rollouts are a little off-policy. The recipe passes `--use-tis` in both loops: miles weights
each token's loss by the trainer's probability over the deployment's, capped at 2 (`--tis-clip`).
On a Mixture-of-Experts model, combine it with routing replay (next section), which removes most of
the gap first. The [routing replay case study](../../case-studies/rollout-routing-replay/README.md)
measures the gap with and without replay.

- `USE_TIS=0` in `run.env` (the launcher's `--tis 0`) turns it off.
- TIS reads `rollout_log_probs` from every sample. A group whose log-probs do not line up with
  its tokens is sampled again, up to twice, before the rollout fails. With the sampled token ids
  from the deployment, they always line up.
- Sample at temperature 1, the recipe's `--rollout-temperature`. The deployment reports log-probs
  before temperature and miles scores after it, so the rollout function refuses TIS at any other
  temperature.
- miles logs the weights as `tis`, the clipped share as `tis_clipfrac`, and the mean gap between
  the two log-probs as `train_rollout_logprob_abs_diff`.

## Routing replay for MoE models

On a Mixture-of-Experts model, the trainer's router can pick other experts for a token than the
deployment did. With `--use-rollout-routing-replay`, miles runs its MoE layers on the experts that
the deployment used, so it trains the forward pass that sampled the rollout. The rollout function
supplies them:

- Start the deployment with the `enable-return-routed-experts` engine option.
- Pass `--use-rollout-routing-replay` to miles. The rollout function then asks for each sequence's
  routes and stores them as `rollout_routed_experts`. The flag is the only switch: without it the
  rollout asks for no routes, and a `CWTRAIN_ROLLOUT_ROUTED_EXPERTS` that disagrees is ignored with
  a warning.
- A response carries one sequence's routes, so each group of `n` becomes `n` requests of one
  choice.
- A response without routes, or whose routes do not line up with its tokens or have another layer
  count or top-k than `--num-layers` and `--moe-router-topk`, fails the rollout. miles refuses to
  train a batch without routes in any case.

## Hooking into miles' per-model scripts

miles ships one launch script per model under `scripts/` (`run_qwen3_4b.py`,
`run_qwen3_30b_a3b.py`, ...). Each is a small typer CLI. It composes argument groups (checkpoint,
rollout, optimizer, GRPO, performance, parallelism) and passes them to
`miles.utils.external_utils.command_utils`. That module's `convert_checkpoint` and `execute_train`
do the Megatron conversion, start Ray and submit `train.py`. The architecture flags come from
`scripts/models/<type>.py`.

The scripts take `--extra-args`. But they own SGLang engines, so they also hardcode `--colocate`
and the `--sglang-*` flags, and `--extra-args` cannot remove a `store_true` flag. So the recipe
does not wrap a stock script. [`run_hotload.py`](run_hotload.py) is a script in the same style. It
uses the same helpers and model definitions, and replaces only the engine-side group:

| Stock `scripts/run_<model>.py` | `run_hotload.py` |
| --- | --- |
| `--colocate`, `--sglang-*`, `--rollout-num-gpus-per-engine` | `--rollout-num-gpus 0 --rollout-function-path coreweave_rollout.CoreWeaveRolloutFn` |
| NCCL weight sync into its own engines | `--update-weight-transfer-mode disk-delta ... --custom-update-weight-post-write-path cwtrain.integrations.slime.hook` |
| model, data, conversion via `command_utils` | same calls |
| `U.execute_train(train_args, ...)` | same call, plus the `CWTRAIN_*` / `COREWEAVE_*` / `AWS_*` / `CW_S3_*` environment forwarded to the Ray workers |

Any model with a `scripts/models/<type>.py` works by name (`--model-name Qwen3-8B --tp 2`).
`--extra-args` appends any other flags. To edit a stock script instead, change three things:

1. Remove `--colocate` and the `sglang_args` group.
2. Add the CoreWeave group above.
3. Forward the environment through `execute_train`'s `extra_env_vars`.

## Versions

| Component | Version |
| --- | --- |
| miles | [`v0.1.1`](https://github.com/radixark/miles/releases/tag/v0.1.1): the trainer image `radixark/miles:v0.1.1` (amd64 and arm64), pinned by digest in the example's `run.env.example`. `run-trainer.sh` runs the miles checkout inside that image. |

Releases after v0.1.1 keep the launcher helpers that `run_hotload.py` calls (`convert_checkpoint`,
`execute_train`, `hf_download_dataset`, `exec_command_cpu`) in `command_utils.legacy`, not in
`command_utils`. The script takes them from either module. When neither has them, it stops before
any download and names the pinned image. A newer image works if it accepts the flags above.

## Synchronous or one-step-ahead: `train.py` vs `train_async.py`

`--train-script train_async.py` (the recipe's `TRAIN_SCRIPT`) selects miles' asynchronous loop:

1. It generates rollout `i+1` while step `i` trains.
2. Before the weight update, it drains the in-flight generation, so no request spans two weight
   versions.
3. It calls the same `update_weights` path, so the hook hot-loads as in the synchronous loop.

The data is different: the policy that sampled each rollout is one update behind the policy being
trained. `--async-correction` picks the denominator of the PPO ratio:

| Value | Flag | Notes |
| --- | --- | --- |
| `keep_old_actor` (default) | `--keep-old-actor` | Recomputes it with the weights that generated the rollout. Costs one extra copy of the actor weights on the trainer. TIS then corrects only for the gap between the deployment's and the trainer's numerics. |
| `none` | | Recomputes it with the current weights. TIS then corrects for the stale step too. With `USE_TIS=0`, this is plain GRPO on one-step-stale samples. |

`ASYNC_CORRECTION=tis` from earlier versions of this recipe is now `USE_TIS=1` (the default) with
`ASYNC_CORRECTION=none`. The launcher refuses `tis` with that hint.

On a hot-load deployment, the gain is that generation overlaps the training step. The hot-load
itself still runs between steps. slime's upstream tree does not ship a `train_async.py`. Its
asynchronous mode is the fully-async rollout worker, which this cookbook does not cover.

## Concurrency

Two settings decide how many sequences are sampled at once:

- **The deployment's `autoscaling.concurrency`**, a per-replica limit. The platform passes it to
  vLLM as `--max-num-seqs`. So a replica decodes at most that many sequences at once, and later
  requests wait in a queue. vLLM never schedules more than its KV cache holds, so a higher value
  only moves the wait into the engine.
- **`ROLLOUT_CONCURRENCY`** in `run.env`: the number of GRPO group requests that the trainer keeps
  in flight. The Job passes it as `CWTRAIN_ROLLOUT_CONCURRENCY`. Each request asks for
  `--n-samples-per-prompt` completions. So the trainer has up to `ROLLOUT_CONCURRENCY` times that
  many sequences in flight.

When the trainer sends more sequences than replicas × `autoscaling.concurrency`, the extra
sequences wait for a free slot.

## Sequence of one training step (delta mode)

| Step | Who | What |
| --- | --- | --- |
| baseline | miles rank 0 | Captures `--hf-checkpoint` bytes as the delta baseline and calls the hook with the delta root |
| baseline | hook | Publishes `--hf-checkpoint` as full snapshot `<run>-v000000`, unless a completed hot-load with that identity already exists; creates the weight mirror |
| sync N | miles ranks | Diff against the previous sync, write `weight_v00000N/` (shards + index with `base_version`) |
| sync N | hook (rank 0) | Apply the delta to the mirror and check every changed tensor's Adler-32; upload to `<prefix>/<run>-v00000N/`, create an incremental hot-load with predecessor `<run>-v00000(N-1)`, and wait until the hot-load completes (every replica READY on `<run>-v00000N`) |
| sync N, N % FULL_EVERY == 0 | hook (rank 0) | Same, but publish a **full** snapshot rebuilt from the mirror (with per-tensor checksums) so the chain restarts |
| sync N | miles | Barrier, then continues with the next rollout batch |

The hook blocks rank 0 before the barrier, so training pauses until the deployment serves the new
version. Set `CWTRAIN_HOTLOAD_WAIT=0` to give up that guarantee for more throughput.

## Full snapshots every N versions, and what is verified

A delta chain must not grow without limit, for three reasons:

- A replica that starts with no peer to copy the weights from replays the whole chain from object
  storage. It can replay at most 63 deltas.
- The API rejects a chain longer than 128 deltas.
- One bad delta corrupts every version after it.

So, by default, the hook publishes a full checkpoint every 20th version
(`CWTRAIN_HOTLOAD_FULL_EVERY`, the recipe's `FULL_EVERY`, at most 64):

- At the baseline call, the hook copies `--hf-checkpoint` to `<delta dir>-hotload/mirror/` and
  computes the checksum of every tensor. This **mirror** is a byte-exact copy of what the
  deployment serves. The trainer itself has no such copy, because miles writes only XOR deltas.
- Every sync applies the new `weight_vNNNNNN` delta to the mirror in place. For each tensor, the
  hook decompresses the zstd data, XORs it onto the tensor's bytes, and compares the Adler-32 of
  the result with the checksum in the shard header. It writes a tensor only when the checksums
  match. A mismatch raises an error before anything is uploaded, and stops the run.
- Every `FULL_EVERY`-th version, the hook writes the mirror out as a full HF checkpoint and
  publishes it as a **full** snapshot under the same identity scheme. The shards carry each
  tensor's checksum in `__metadata__`, and the index says `checksum_format: adler32`. The next
  delta chains onto this snapshot.
- Every version also gets a manifest of all tensor checksums, at
  `<delta dir>-hotload/manifests/<identity>.json` and
  `s3://<bucket>/<prefix>/_manifests/<identity>.json`.

What checks the weights that end up in the engine:

| Where | Check |
| --- | --- |
| Trainer, mirror | Adler-32 of every changed tensor after applying the delta equals the trainer's checksum. Proves the uploaded delta reconstructs the trainer's exact bytes. |
| Engine, install from object storage | The engine applies the delta the same way and verifies the same per-tensor Adler-32; for a full snapshot with `checksum_format` it verifies every tensor. Mismatch = failed hot-load, never a silent load. |
| Engine, install from a peer | Replicas that receive the version from another replica over NIXL check each tensor's name, size and dtype against the source. They do not recompute checksums. |
| API, the hot-load itself | The platform marks a replica READY only after reading the engine's own weight version back, and completes the hot-load only when every replica is READY. `CWTRAIN_HOTLOAD_VERIFY=1` re-reads the replicas of the completed hot-load anyway (all must be READY on the new identity, or `HotLoadVerificationError` stops the run); it is off by default. |

Cost:

- **Disk.** The mirror is a second copy of the checkpoint on the trainer's local disk. Each chain
  cut stages a third copy before the upload. Plan for about twice the model's size in free disk.
- **Launch time.** The hook builds the mirror (copy and checksums) at every launch.
- **Sync time.** Every sync reads and rewrites the changed tensors, in parallel across tensors.
  Every N-th version adds a full upload and a full hot-load.

On large models, the copies and a full hot-load each take minutes. With `CWTRAIN_HOTLOAD_WAIT=1`,
training waits for every hot-load. `CWTRAIN_HOTLOAD_MIRROR=0` turns off the mirror. It requires
`FULL_EVERY=0`, which never cuts the chain.

## Resuming a run

On restart, miles captures its baseline again, so it calls the hook with the delta root again. The
hook looks for a completed `<run>-v000000` and skips the upload when it finds one. Version numbers
also restart at 1: miles clears the delta directory, and the deltas are again relative to
`--hf-checkpoint`. So a resumed run must publish under a *new* run ID. With the same run ID, the
hook would try to reuse identities that already exist with different bytes. Always start a
resumed run with a new `CWTRAIN_HOTLOAD_RUN_ID`.

## Pitfalls

- **`snapshot access check failed` at the start of the trainer Job means that the trainer may not
  use the hot-load prefix.** It needs `s3:ListBucket`, `s3:GetObject`, `s3:PutObject` and
  `s3:AbortMultipartUpload` there, from `AllowTrainer` and from its own org access policy. Check
  that `AllowTrainer` names the trainer's principal (`TRAINER_PRINCIPAL_ARN`), and that
  `HOTLOAD_PREFIX` is the deployment's `hotLoad.pathPrefix`. With Workload Identity Federation,
  also check that the cluster's issuer is registered and that the Job runs as
  `TRAINER_SERVICE_ACCOUNT`. `cwtrain hotload check --deployment <id>` runs the same check with the
  credentials of your shell.
- **The uploaded index names identities, not counters.** miles writes `000001` / `000000` as the
  index's `metadata.version` and `metadata.base_version`. The platform takes the delta's identity
  and its predecessor from the hot-load request, not from these fields. `upload_snapshot` still
  rewrites them to the snapshot identities during the upload, so the index in the bucket agrees
  with the request. It uploads the shards and their per-tensor checksums unchanged.
- **A sync that changes nothing publishes nothing.** miles still writes `weight_vNNNNNN/`, with an
  index and no shards. The hook logs it and reports `hotload/skipped_empty`. The skipped version is
  byte-identical to the last one, so the hook chains the next delta onto the last version that the
  deployment received.
- **A replica that a failed replay left on `unknown` recovers on the next hot-load.** After a
  rejected delta, a replica can report weight version `unknown`. It stays paused until a later
  hot-load succeeds. Fix the cause, then create the hot-load again (see
  [Failure handling](#failure-handling)). `cwtrain hotload status <id>` shows each replica's state
  and error. If hot-loads still stall with `chain root unresolved`, the platform cannot tell which
  version a replica serves. The deployment's pods run on CoreWeave's side, so send CoreWeave
  support the deployment id and the hot-load id.
- **A missing bucket policy is the most common reason that weights fail to load.** The model
  download at deployment start and every hot-load read the bucket as the inference service account.

## Failure handling

- A snapshot that fails validation (missing files, unsupported checksum) raises an error before
  the upload.
- A hot-load that ends `FAILED` raises `HotLoadFailedError` with per-replica reasons. Nothing
  retries automatically: the run stops so that it does not continue off-policy. The snapshot is
  already in the bucket. When you have fixed the cause, create its hot-load again. For a delta (the
  default):
  `cwtrain hotload trigger --deployment <id> --identity <run>-vNNNNNN --previous-identity <run>-v(NNNNNN-1)`.
  For a full snapshot, leave out `--previous-identity`.
- Delta chains have limits. The API rejects a chain longer than 128 deltas (`resolved snapshot
  chain is too long`). A replica that starts with no peer to copy from can replay only 63. The
  default `FULL_EVERY=20` stays below both limits. If you set it to 0, set it back so that the hook
  restarts the chain with a full snapshot on schedule.
- With `CWTRAIN_HOTLOAD_VERIFY=1`, a `HotLoadVerificationError` means that the API marked the
  hot-load COMPLETED, but a replica in its list does not report the new identity. Check
  `cwtrain hotload status <id>`. A replica stuck on `unknown` recovers on the next successful
  hot-load (see [Pitfalls](#pitfalls)).
- A `MirrorError` means that a delta the trainer wrote does not reconstruct to the trainer's
  checksums. Causes include disk corruption, or a delta written against a different base. Nothing
  was uploaded, and the deployment still serves the previous version.

## Verifying by hand

```bash
cwtrain hotload list --deployment <uuid>                 # newest first, with state
cwtrain hotload status <hot-load uuid>                   # replicas and their current identity
cwtrain hotload inspect /work/deltas/weight_v000003      # what the hook would publish
```

With the mirror on, `<delta dir>-hotload/manifests/<identity>.json` lists the Adler-32 of every
tensor for that version. The bucket has a copy under `<prefix>/_manifests/`. These are the
checksums of the engine's copy of that version. The engine checks them when it installs a version
from object storage. On every hot-load, the platform checks that each replica reports the new
version.
