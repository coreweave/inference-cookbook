# Changelog

The major changes in each `cwtrain` release. A version's section is also its release notes on
GitHub. Versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

- **Hot-load deployments:** `cwtrain deployment create` and `InferenceClient.create_deployment`
  require `dynamo-vllm` `0.32.0+crwv.5` or later for a deployment with a `hotLoad` section, and use
  it when `runtime.version` is unset. They turn on the six ModelExpress refit options
  (`MX_REFIT_*`) in `runtime.engineEnv` unless the request sets them. An older version is refused
  before any request is sent.
- **W&B metrics:** when the trainer logs to W&B, cwtrain writes into the trainer's run. It
  records each hot-load's timings, from the platform and per replica, and the rollouts' request
  latency, retries and throughput. It fills the prefix-cache counts that slime's and miles' hit
  rate reads. It adds the deployment and hot-load settings to the run config, without the endpoint
  or any key. Each phase of a training step is a `timeline/<phase>/start` and `/end` span.
  `cwtrain.integrations.slime.run_train_script` runs slime's or miles' train script so that these
  metrics plot against their `rollout/step`. Without a W&B run, cwtrain logs nothing.

## 1.0.2 - 2026-10-23

- **Routing replay:** rollout completions return each token's routed experts, which slime's and
  miles' `--use-rollout-routing-replay` take.
- **Truncated importance sampling:** when the trainer reads the engine's log-probs, for example
  with `--use-tis`, they are required. Groups whose log-probs don't align are resampled, and a
  temperature other than 1 is refused.
- **Caching and sampling:** each rollout's requests carry a prefix-cache salt for their policy
  version, so no version reuses another's cached prompts. One-choice requests get their own seed.
- **Refused trainer flags:** the rollout functions replace the trainers' rollout loop, so they
  refuse the loop's options they would otherwise skip, such as custom generate functions and
  dynamic sampling.

## 1.0.1 - 2026-10-17

- **PyPI page:** the README, which is the PyPI page, keeps only the overview, install,
  prerequisites and quick start, and shows the latest version and the supported Python versions.
  It no longer installs from the repository's `main` branch. The code is the same as 1.0.0.

## 1.0.0 - 2026-10-16

- **Stable:** the Development Status classifier changes from Alpha to Production/Stable.
- **Single-file checkpoints:** cwtrain publishes a checkpoint saved as one `model.safetensors`
  without an index, and writes the index the platform requires.
- **No `nemo-rl` extra:** it only added torch and safetensors, which NeMo-RL's environment already
  has. Existing `cwtrain[nemo-rl]` installs keep working; pip and uv warn that the extra no longer
  exists.

## 0.1.1 - 2026-10-14

- **PyPI page:** the README installs cwtrain from PyPI. The code is the same as 0.1.0.

## 0.1.0 - 2026-10-13

- First release: deployments, checkpoint hot-load, rollouts, and the miles, slime and NeMo-RL
  integrations.
