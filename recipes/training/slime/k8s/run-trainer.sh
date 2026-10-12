#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Runs inside the slime trainer Job. Installs cwtrain, then hands over to launch.py, which prepares
# slime (pinned commit, the recipe's patches), stages the model, converts it, starts Ray and trains
# with every weight version hot-loaded into the deployment. Settings: see run.env.example.
set -euo pipefail
# An empty HOTLOAD_TIMEOUT in run.env reaches the Job as an empty value, which cwtrain rejects as a
# number. Unset it so the hook uses its default.
hotload_timeout=${CWTRAIN_HOTLOAD_TIMEOUT:-}
[[ -n "${hotload_timeout//[[:space:]]/}" ]] || unset CWTRAIN_HOTLOAD_TIMEOUT
pip install --quiet --no-cache-dir /opt/cwtrain/cwtrain-*.whl "huggingface_hub[cli]"
# Stop now, not at the first snapshot, if the trainer's credentials cannot list, read and write
# under the deployment's hot-load prefix. The check leaves no object behind.
python3 -m cwtrain.integrations.preflight
exec python3 /opt/cwtrain/launch.py
