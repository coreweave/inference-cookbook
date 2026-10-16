#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Bundle the cwtrain wheel, the recipe's launcher and Axolotl config, and the shared scripts into a
# ConfigMap (default "cwtrain-kit"), which the Jobs mount at /opt/cwtrain. Build the wheel first:
# (cd libs/cwtrain && uv build --wheel).
#
# Usage: make-kit.sh <kube-context> <namespace> [configmap name, default cwtrain-kit]
set -euo pipefail
CTX=${1:?kube context}; NS=${2:?namespace}; KIT=${3:-cwtrain-kit}
HERE=$(cd "$(dirname "$0")" && pwd)
SHARED=$HERE/../../../../k8s
# Use CWTRAIN_WHEEL, or else the newest wheel in libs/cwtrain/dist (by file time).
# shellcheck source=SCRIPTDIR/../../../../k8s/find-wheel.sh
. "$SHARED/find-wheel.sh"
kubectl --context "$CTX" -n "$NS" create configmap "$KIT" \
  --from-file="$(basename "$WHEEL")=$WHEEL" \
  --from-file=stage-model.sh="$SHARED/stage-model.sh" \
  --from-file=run-trainer.sh="$HERE/run-trainer.sh" \
  --from-file=bucket-policy.example.json="$SHARED/bucket-policy.example.json" \
  --from-file=launch.py="$HERE/../launch.py" \
  --from-file=sft_coreweave.yaml="$HERE/../sft_coreweave.yaml" \
  --dry-run=client -o yaml | kubectl --context "$CTX" apply -f -
echo "$KIT updated in $CTX/$NS with $(basename "$WHEEL")"
