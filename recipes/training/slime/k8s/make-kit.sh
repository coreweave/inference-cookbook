#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Bundles the cwtrain wheel, the recipe scripts, the rollout module and the patches into a ConfigMap
# (default "cwtrain-kit"), which the Jobs mount at /opt/cwtrain.
# Build the wheel first: (cd libs/cwtrain && uv build --wheel).
#
# Usage: make-kit.sh <kube-context> <namespace> [configmap name, default cwtrain-kit]
set -euo pipefail
CTX=${1:?kube context}; NS=${2:?namespace}; KIT=${3:-cwtrain-kit}
HERE=$(cd "$(dirname "$0")" && pwd)
# CWTRAIN_WHEEL, or else the newest wheel in libs/cwtrain/dist (by file time)
WHEEL=${CWTRAIN_WHEEL:-}
if [ -z "$WHEEL" ]; then
  for candidate in "$HERE"/../../../../libs/cwtrain/dist/cwtrain-*.whl; do
    if [ -f "$candidate" ] && { [ -z "$WHEEL" ] || [ "$candidate" -nt "$WHEEL" ]; }; then WHEEL=$candidate; fi
  done
fi
if [ -n "${CWTRAIN_WHEEL:-}" ] && [ ! -f "$WHEEL" ]; then echo "CWTRAIN_WHEEL=$WHEEL is not a file" >&2; exit 1; fi
[ -f "$WHEEL" ] || { echo "no wheel under libs/cwtrain/dist; run: (cd libs/cwtrain && uv build --wheel)" >&2; exit 1; }
patches=()
for f in "$HERE"/../patches/*.patch; do
  [ -e "$f" ] && patches+=(--from-file="$(basename "$f")=$f")
done
kubectl --context "$CTX" -n "$NS" create configmap "$KIT" \
  --from-file="$(basename "$WHEEL")=$WHEEL" \
  --from-file=stage-model.sh="$HERE/stage-model.sh" \
  --from-file=run-trainer.sh="$HERE/run-trainer.sh" \
  --from-file=bucket-policy.example.json="$HERE/../bucket-policy.example.json" \
  --from-file=coreweave_rollout.py="$HERE/../coreweave_rollout.py" \
  --from-file=launch.py="$HERE/../launch.py" \
  ${patches[@]+"${patches[@]}"} \
  --dry-run=client -o yaml | kubectl --context "$CTX" apply -f -
echo "$KIT updated in $CTX/$NS with $(basename "$WHEEL")"
