#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Bundle the cwtrain wheel, the recipe's launch script, NeMo-RL config and patches into a ConfigMap
# (default "cwtrain-kit") mounted at /opt/cwtrain by the Jobs. Rebuild the wheel first:
# (cd libs/cwtrain && uv build --wheel).
#
# Usage: make-kit.sh <kube-context> <namespace> [configmap name, default cwtrain-kit]
set -euo pipefail
CTX=${1:?kube context}; NS=${2:?namespace}; KIT=${3:-cwtrain-kit}
HERE=$(cd "$(dirname "$0")" && pwd)
# Use CWTRAIN_WHEEL, or else the newest wheel in libs/cwtrain/dist (by file time).
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
  --from-file=stage-model.sh="$HERE/stage-nemotron.sh" \
  --from-file=prepare_model.py="$HERE/../prepare_model.py" \
  --from-file=run-trainer.sh="$HERE/run-trainer.sh" \
  --from-file=bucket-policy.example.json="$HERE/../bucket-policy.example.json" \
  --from-file=grpo_coreweave.yaml="$HERE/../grpo_coreweave.yaml" \
  --from-file=launch.py="$HERE/../launch.py" \
  ${patches[@]+"${patches[@]}"} \
  --dry-run=client -o yaml | kubectl --context "$CTX" apply -f -
echo "$KIT updated in $CTX/$NS with $(basename "$WHEEL")"
