# Recipe: NeMo Gym evaluation on a CoreWeave deployment

Run five multiple-choice tasks through [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) with
**Nemotron 3.5 Lightning 30B-A3B BF16** on CoreWeave Dedicated Inference. Gym runs the agent and
the answer verifier locally. Its `vllm_model` adapter converts Gym's Responses format into remote
`/v1/chat/completions` requests. You do not need a local inference engine or GPU.

The recipe uses the [cwtrain library](../../../libs/cwtrain/README.md):

- [`resources.py`](resources.py) uses `InferenceClient` and the `cwtrain.infra.lifecycle` helpers
  to create and delete an authenticated gateway and a one-GPU deployment.
- [`launch.py`](launch.py) uses `RolloutConfig` for the endpoint, model, key file and collection
  concurrency.

Gym sends the chat requests, runs the agent and scores the answers.

This recipe is separate from the [NeMo-RL training recipe](../../training/nemo-rl/README.md). It
collects scored trajectories and does not update weights. The deployment does not need hot-load.
Keep the deployment's weights fixed during collection, so that the results describe one policy
version.

## Versions

| Component | Version |
| --- | --- |
| NeMo Gym | [`v0.6.0`](https://github.com/NVIDIA-NeMo/Gym/releases/tag/v0.6.0) |
| Inference runtime | `vllm`, `0.30.0+crwv.5` |
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` |

The commands use the `gym` CLI from v0.6.0. This Nemotron configuration with Gym v0.6.0 and vLLM
0.30.0 is not yet validated against a live deployment.

## Prerequisites

- Python 3.13.14 or newer, Git and `uv` on a machine that can reach the deployment.
- A CoreWeave API token with the Inference Admin role in your organization.
- The Nemotron 3.5 Lightning BF16 checkpoint and tokenizer files in an existing CoreWeave AI Object
  Storage bucket that the inference service can read. See the
  [model storage requirements](https://docs.coreweave.com/products/inference/models#grant-inference-access-to-your-bucket).
  The recipe does not change the bucket or its policy.
- One H100 80GB in the selected zone. Or, to skip resource creation, an existing deployment of
  this model.

The config turns off Nemotron thinking with `enable_thinking: false` and limits each answer to 512
generated tokens. For another model, remove or change `chat_template_kwargs` to match its
template. The five tasks test connectivity and scoring. They are not an accuracy benchmark.

## Install and configure

Start in the cookbook root. Use a fresh Gym checkout, so that an existing `env.yaml` does not
override the model config of this recipe.

```bash
export GYM_RECIPE="$PWD/recipes/evaluation/nemo-gym"
export COOKBOOK="$PWD"
export GYM_DIR="$(mktemp -d)/Gym"
export GYM_RUN="$(mktemp -d)"
git clone --branch v0.6.0 --depth 1 https://github.com/NVIDIA-NeMo/Gym.git "$GYM_DIR"
cd "$GYM_DIR"
uv sync --frozen --python 3.13.14
source .venv/bin/activate
uv pip install --python "$GYM_DIR/.venv/bin/python" -e "$COOKBOOK/libs/cwtrain"

# One org-scoped token authenticates management and CoreWeave IAM inference requests.
export CWTRAIN_ROLLOUT_API_KEY_FILE=/absolute/path/to/coreweave-api-token
export COREWEAVE_API_TOKEN="$(cat "$CWTRAIN_ROLLOUT_API_KEY_FILE")"
export CWTRAIN_ROLLOUT_CONCURRENCY=2
```

After you install cwtrain, run the venv's `python` directly. `uv run` can sync cwtrain out of the
venv. Gym creates more local server environments when it first starts. Keep the Gym checkout,
logs, results, filled-in deployment spec and run journal outside Git.

## Create a temporary gateway and deployment

1. List the available runtime versions and instance types. To scope the list to an existing
   gateway in the target zone, add `--gateway-id`. Omit it when no gateway exists yet.

   ```bash
   python "$GYM_RECIPE/resources.py" parameters --gateway-id '<existing-gateway-id>'
   cp "$GYM_RECIPE/deployment.example.json" "$GYM_RUN/deployment.json"
   ```

2. Fill in the placeholders in `$GYM_RUN/deployment.json`:

   - Keep the pinned `vllm` `0.30.0+crwv.5` build if the target zone has it. The `+crwv.5` suffix
     identifies CoreWeave's build of vLLM 0.30.0.
   - Choose an H100 80GB instance type.
   - Set the bucket and path of the existing model.
   - Keep `model.name` as `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.

   The example requests one H100 and one replica, with hot-load off. It reserves 90% of GPU memory
   for the BF16 model, limits context to 4,096 tokens and running sequences to eight, and uses the
   `nemotron_v3` reasoning parser. For the serving requirements, see the
   [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16#1x-h100-vllm).
   Use only the engine options that the Dedicated Inference parameters endpoint lists. It does not
   expose every upstream vLLM flag.

3. Create the resources and check their status:

   ```bash
   python "$GYM_RECIPE/resources.py" provision \
     --organization '<your-org-id>' --zone US-WEST-04A \
     --deployment-spec "$GYM_RUN/deployment.json" --state "$GYM_RUN/state.json"
   python "$GYM_RECIPE/resources.py" status --state "$GYM_RUN/state.json"
   ```

   `provision` returns immediately. Run `status` again until `ready` is `true`. The first model
   startup can take several minutes. `status` shows the gateway endpoint and the model name.

4. Set the endpoint, with `/v1` appended:

   ```bash
   export CWTRAIN_ROLLOUT_BASE_URL='https://<gateway-endpoint>/v1'
   export CWTRAIN_ROLLOUT_MODEL='nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16'
   ```

The journal (`state.json`) records new UUIDs before it creates anything. It refuses to overwrite
an existing run. **If provisioning or evaluation fails, still run the [cleanup](#cleanup)
command**: a partial create can leave billable resources. Keep the journal until cleanup confirms
that the resources are gone.

To use an existing deployment, skip provisioning. Set its endpoint, served model name and key
file. Do not run this recipe's cleanup against existing resources.

## Start Gym and collect rollouts

1. Start the servers through the cwtrain launcher:

   ```bash
   python "$GYM_RECIPE/launch.py" start --gym-dir "$GYM_DIR"
   ```

   Wait until `mcqa`, `mcqa_simple_agent` and `policy_model` start. Keep this terminal open. The
   resource config already includes the agent wiring. Do not add the generic `simple_agent.yaml`,
   which requires a resources-server name.

2. In a second terminal, set the same `GYM_RECIPE`, `GYM_DIR`, `GYM_RUN` and `CWTRAIN_ROLLOUT_*`
   variables. Then run:

   ```bash
   source "$GYM_DIR/.venv/bin/activate"
   python "$GYM_RECIPE/launch.py" collect --gym-dir "$GYM_DIR" \
     --output "$GYM_RUN/coreweave_mcqa.jsonl"
   ```

The launcher runs `gym eval run --no-serve` against the running servers. It uses five tasks, one
repeat and the concurrency set in cwtrain. It writes these files in `$GYM_RUN`:

| File | Contents |
| --- | --- |
| `coreweave_mcqa.jsonl` | Trajectories and rewards |
| `coreweave_mcqa_materialized_inputs.jsonl` | The inputs |
| `coreweave_mcqa_aggregate_metrics.json` | Aggregate scores |

### Verify

A completed run has five trajectories with rewards. A zero reward can mean a wrong answer, or an
answer that does not use the boxed-letter format. It does not mean a connection error. Read the
generated answers before you interpret the aggregate score.

## Cleanup

1. Stop the servers with Ctrl-C in the first terminal.
2. Delete the resources that this run created:

   ```bash
   python "$GYM_RECIPE/resources.py" cleanup --state "$GYM_RUN/state.json"
   ```

Cleanup deletes the recorded deployment first, then the gateway, and confirms that both are gone.
It checks the names before it deletes. You can run it again after an interruption or a timeout,
with the same journal. It does not touch the shared model bucket.

## Why this is separate from NeMo-RL training

Gym supplies environments, agent execution and rewards. NeMo-RL supplies the optimizer and the
training loop. The [NeMo-RL training recipe](../../training/nemo-rl/README.md#nemo-gym-rollouts)
connects them with `env.should_use_nemo_gym=true`.

This evaluation does not request training token metadata. So successful chat and reward
collection does not verify the training path, which requires exact token IDs and log-probs. See
the training recipe for the endpoint requirements and validation status.

Source references: [Gym v0.6.0 quickstart](https://github.com/NVIDIA-NeMo/Gym/blob/v0.6.0/README.md),
[HTTP model adapter](https://github.com/NVIDIA-NeMo/Gym/blob/v0.6.0/responses_api_models/vllm_model/app.py),
[NeMo-RL v0.7.0 GRPO setup](https://github.com/NVIDIA-NeMo/RL/blob/v0.7.0/nemo_rl/algorithms/grpo.py),
and [Gym integration design](https://github.com/NVIDIA-NeMo/RL/blob/v0.7.0/docs/design-docs/nemo-gym-integration.md).
