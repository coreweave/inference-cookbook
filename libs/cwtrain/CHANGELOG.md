# Changelog

The major changes in each `cwtrain` release. A version's section is also its release notes on
GitHub. Versions follow [Semantic Versioning](https://semver.org/).

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
