#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Runs in the Axolotl trainer Job (axolotlai/axolotl image). Installs cwtrain into a venv of its
# own, then starts launch.py. launch.py stages the base model, runs axolotl train (and merge-lora
# after a LoRA run), and publishes the result into the deployment as a full hot-load snapshot.
# For the settings, see an example's k8s/run.env.example.
set -euo pipefail
CW_PY=${WORK:-/work}/cwtrain/bin/python
# cwtrain gets a venv of its own; Axolotl's environment stays as the image built it
uv venv --quiet --python 3.12 "${WORK:-/work}/cwtrain"
uv pip install --quiet --python "$CW_PY" /opt/cwtrain/cwtrain-*.whl
# cwtrain does not accept an empty HOTLOAD_TIMEOUT as a number. Unset it, so that the publish step
# uses its default.
hotload_timeout=${CWTRAIN_HOTLOAD_TIMEOUT:-}
[[ -n "${hotload_timeout//[[:space:]]/}" ]] || unset CWTRAIN_HOTLOAD_TIMEOUT
# An empty W&B setting from run.env reaches the Job as an empty value. Unset it, so that W&B's own
# default applies (WANDB_PROJECT empty: no W&B at all).
for var in WANDB_PROJECT WANDB_ENTITY; do
  value=${!var:-}
  [[ -n "${value//[[:space:]]/}" ]] || unset "$var"
done
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  # The publish step records itself in Axolotl's W&B run, with the image's wandb version
  wandb_version=$(python3 -c 'import wandb; print(wandb.__version__)' 2>/dev/null || true)
  uv pip install --quiet --python "$CW_PY" "wandb${wandb_version:+==$wandb_version}"
fi
# Stop now, not after training, if the trainer's credentials cannot list, read and write under the
# deployment's hot-load prefix. The check leaves no object behind.
"$CW_PY" -m cwtrain.integrations.preflight
exec "$CW_PY" /opt/cwtrain/launch.py
