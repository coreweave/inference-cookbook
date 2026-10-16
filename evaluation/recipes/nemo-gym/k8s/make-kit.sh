#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Bundle the cwtrain wheel, the Gym launcher, its configs and run-eval.sh into a ConfigMap
# (default "cwtrain-gym-kit") mounted at /opt/cwtrain-gym by eval-job.yaml. Rebuild the wheel
# first: (cd libs/cwtrain && uv build --wheel).
#
# Usage: make-kit.sh <kube-context> <namespace> [configmap name, default cwtrain-gym-kit]
set -euo pipefail
CTX=${1:?kube context}; NS=${2:?namespace}; KIT=${3:-cwtrain-gym-kit}
HERE=$(cd "$(dirname "$0")" && pwd)
RECIPE=$(cd "$HERE/.." && pwd)
SHARED=$HERE/../../../../k8s
# Use CWTRAIN_WHEEL, or else the newest wheel in libs/cwtrain/dist (by file time).
# shellcheck source=SCRIPTDIR/../../../../k8s/find-wheel.sh
. "$SHARED/find-wheel.sh"
kubectl --context "$CTX" -n "$NS" create configmap "$KIT" \
  --from-file="$(basename "$WHEEL")=$WHEEL" \
  --from-file=run-eval.sh="$HERE/run-eval.sh" \
  --from-file=launch.py="$RECIPE/launch.py" \
  --from-file=you_search.py="$RECIPE/you_search.py" \
  --from-file=coreweave.yaml="$RECIPE/coreweave.yaml" \
  --from-file=browsecomp.yaml="$RECIPE/browsecomp.yaml" \
  --from-file=exclude_domains.json="$RECIPE/exclude_domains.json" \
  --dry-run=client -o yaml | kubectl --context "$CTX" apply -f -
echo "$KIT updated in $CTX/$NS with $(basename "$WHEEL")"
