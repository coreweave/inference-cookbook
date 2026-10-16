#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Runs in the NeMo-RL trainer Job (nvcr.io/nvidia/nemo-rl image). Installs cwtrain into the NeMo-RL
# venv, then starts launch.py. launch.py patches NeMo-RL, makes cwtrain importable in the policy
# workers, stages the model and starts GRPO. For the settings, see run.env.example.
set -euo pipefail
VENV_PY=${NEMO_RL_PYTHON:-/opt/nemo_rl_venv/bin/python}
# uv pip install, not uv run: uv run re-syncs the venv to NeMo-RL's lockfile and drops cwtrain
uv pip install --quiet --python "$VENV_PY" /opt/cwtrain/cwtrain-*.whl
# A setting missing from run.env reaches the Job as an empty value. Unset it, so that the default
# in grpo_coreweave.yaml applies.
concurrency=${CWTRAIN_ROLLOUT_CONCURRENCY:-}
[[ -n "${concurrency//[[:space:]]/}" ]] || unset CWTRAIN_ROLLOUT_CONCURRENCY
# cwtrain does not accept an empty HOTLOAD_TIMEOUT as a number. Unset it, so that the hook uses its
# default.
hotload_timeout=${CWTRAIN_HOTLOAD_TIMEOUT:-}
[[ -n "${hotload_timeout//[[:space:]]/}" ]] || unset CWTRAIN_HOTLOAD_TIMEOUT
# Stop now, not at the first refit, if the trainer's credentials cannot list, read and write under
# the deployment's hot-load prefix. The check leaves no object behind.
"$VENV_PY" -m cwtrain.integrations.preflight
exec "$VENV_PY" /opt/cwtrain/launch.py
