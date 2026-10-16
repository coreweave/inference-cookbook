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
SHARED=$HERE/../../../../k8s
# CWTRAIN_WHEEL, or else the newest wheel in libs/cwtrain/dist (by file time)
# shellcheck source=SCRIPTDIR/../../../../k8s/find-wheel.sh
. "$SHARED/find-wheel.sh"
patches=()
for f in "$HERE"/../patches/*.patch; do
  [ -e "$f" ] && patches+=(--from-file="$(basename "$f")=$f")
done
kubectl --context "$CTX" -n "$NS" create configmap "$KIT" \
  --from-file="$(basename "$WHEEL")=$WHEEL" \
  --from-file=stage-model.sh="$SHARED/stage-model.sh" \
  --from-file=run-trainer.sh="$HERE/run-trainer.sh" \
  --from-file=bucket-policy.example.json="$SHARED/bucket-policy.example.json" \
  --from-file=coreweave_rollout.py="$HERE/../coreweave_rollout.py" \
  --from-file=launch.py="$HERE/../launch.py" \
  ${patches[@]+"${patches[@]}"} \
  --dry-run=client -o yaml | kubectl --context "$CTX" apply -f -
echo "$KIT updated in $CTX/$NS with $(basename "$WHEEL")"
