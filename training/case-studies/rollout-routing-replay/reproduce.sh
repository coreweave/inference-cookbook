#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Runs the whole study against a GLM-5.3-Flash deployment that has the enable-return-routed-experts
# engine option and a hot-load prefix: probe it, record on v0, hot-load v1 (full snapshot,
# RESET_ALL), record, hot-load v2 (delta, PRESERVE), record, analyze. cwtrain uploads the
# hot-loads, so set the deployment's bucket credentials as for the recipes.
#
# Needs CWTRAIN_ROLLOUT_BASE_URL, CWTRAIN_ROLLOUT_API_KEY, COREWEAVE_API_TOKEN and DEPLOYMENT_ID,
# and about 1 TB in WORK (default ./rollout-routing-replay) for the weights and two noisy copies of
# their router shards. FORWARD=1 adds the trainer-side passes, which need about 640 GB of GPU
# memory and flash-linear-attention; without it the analysis runs its CPU part only.
set -euo pipefail
: "${CWTRAIN_ROLLOUT_BASE_URL:?}" "${CWTRAIN_ROLLOUT_API_KEY:?}" "${COREWEAVE_API_TOKEN:?}"
: "${DEPLOYMENT_ID:?}"
export CWTRAIN_ROLLOUT_MODEL=${CWTRAIN_ROLLOUT_MODEL:-glm-5.3-flash}
MODEL=zai-org/GLM-5.3-Flash-BF16
REVISION=f9fed3bbc82291073373d67452e9f25317ef7577
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-$PWD/rollout-routing-replay}
mkdir -p "$WORK" && cd "$WORK"

record() { python "$HERE/record.py" --model-path v0 --out recording "$@"; }

echo "== the served weights: the noise versions start from them, and the analysis reads them"
[[ -d v0 ]] || hf download "$MODEL" --revision "$REVISION" --local-dir v0

echo "== probe the deployment"
python "$HERE/probe.py" --model-path v0 --out probe.json

echo "== record on v0, then send P again: it is in the prefix cache now"
record --version v0 --sets P,single,two_turn
record --version v0-resend --sets P

echo "== hot-load v1: noise on the routers, a full snapshot, RESET_ALL"
python "$HERE/router_noise.py" --base v0 --out v1 --seed 1 --scale 0.005
cwtrain hotload publish --deployment "$DEPLOYMENT_ID" --checkpoint v1 --identity v1 --reset-prompt-cache
record --version v1 --sets P,Q
record --version v1-resend --sets Q

echo "== hot-load v2: more noise, a delta, PRESERVE"
python "$HERE/router_noise.py" --base v1 --out v2 --seed 2 --scale 0.005 --delta v2-delta
cwtrain hotload publish --deployment "$DEPLOYMENT_ID" --checkpoint v2-delta --identity v2
record --version v2 --sets Q,R

echo "== analyze"
cp v1.noise.json recording/v1/router_noise.json
cp v2.noise.json recording/v2/router_noise.json
args=(--recording recording --model-path v0 --out report.json)
if [[ ${FORWARD:-0} == 1 ]]; then
  args+=(--forward)
fi
python "$HERE/analyze.py" "${args[@]}"
echo "report: $WORK/report.json"
