#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Bundle the cwtrain wheel, the recipe scripts and the rollout module into a ConfigMap (default
# "cwtrain-kit"), which the Jobs mount at /opt/cwtrain. Build the wheel first:
# (cd libs/cwtrain && uv build --wheel).
#
# Usage: make-kit.sh <kube-context> <namespace> [configmap name, default cwtrain-kit]
set -euo pipefail
CTX=${1:?kube context}; NS=${2:?namespace}; KIT=${3:-cwtrain-kit}
HERE=$(cd "$(dirname "$0")" && pwd)
SHARED=$HERE/../../../../k8s
# Use CWTRAIN_WHEEL if set, else the newest wheel in libs/cwtrain/dist (by file time, not version).
# shellcheck source=SCRIPTDIR/../../../../k8s/find-wheel.sh
. "$SHARED/find-wheel.sh"
patch_args=()
for patch in "$HERE"/../patches/*.patch; do
  [ -e "$patch" ] && patch_args+=(--from-file="$(basename "$patch")=$patch")
done
kubectl --context "$CTX" -n "$NS" create configmap "$KIT" \
  --from-file="$(basename "$WHEEL")=$WHEEL" \
  --from-file=stage-model.sh="$SHARED/stage-model.sh" \
  --from-file=run-trainer.sh="$HERE/run-trainer.sh" \
  --from-file=bucket-policy.example.json="$SHARED/bucket-policy.example.json" \
  --from-file=coreweave_rollout.py="$HERE/../coreweave_rollout.py" \
  --from-file=coreweave_resume.py="$HERE/../coreweave_resume.py" \
  --from-file=coreweave_train.py="$HERE/../coreweave_train.py" \
  --from-file=apply_patches.py="$HERE/../apply_patches.py" \
  ${patch_args[@]+"${patch_args[@]}"} \
  --from-file=run_hotload.py="$HERE/../run_hotload.py" \
  --dry-run=client -o yaml | kubectl --context "$CTX" apply -f -
echo "$KIT updated in $CTX/$NS with $(basename "$WHEEL")"
