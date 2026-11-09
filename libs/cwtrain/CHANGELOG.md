# Changelog

The major changes in each `cwtrain` release. A version's section is also its release notes on
GitHub. Versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

- **NeMo-RL v0.8.0:** the recipe pins the release container with Gym v0.7.0 and Ray 2.58.0,
  ports its backend and Gym patches, and uses the renamed AutoModel config and worker.
  Checkpoint export preserves stored tensor dtypes and phase timing supports the new Timer API.
  Release-source and CPU checks pass; the GPU training results remain from v0.7.0.

- **Readable dashboard timestamps:** timeline tooltips show local dates and times with
  milliseconds and UTC offsets. Span durations use days, hours, minutes, and seconds.
  System charts format their timestamp axes as dates and times.

- **Top-k sampling:** Miles and Slime now forward `rollout_top_k` through group sampling.
  Explicit disable values (`-1` and `0`) reach the completions API instead of inheriting the
  served model's generation defaults, which could silently enable top-k filtering.

- **Post-training dashboards:** `cwtrain dashboard setup` creates a native W&B saved view from
  recorded metrics, with framework-specific training, rollout, hot-load, evaluation, and system
  charts. Miles and Slime include a run comparison Overview and an interactive reward step
  selector above elapsed-time phase spans. The standard package includes dashboard dependencies
  with lower bounds, supporting W&B 0.28.2 and later without requiring a 0.31 upgrade;
  unchanged setup reruns reuse the view without rewriting personal changes. A separate
  `cwtrain dashboard timeline` command exports unsampled run history to local HTML.

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
