#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Runs inside the miles trainer Job. Installs cwtrain, then starts training with run_hotload.py.
# That launcher uses miles' launcher helpers and scripts/models/<model> definitions, and stages the
# base model. No SGLang engines start: the CoreWeave deployment is the rollout engine, and the
# cwtrain hook hot-loads every weight version.
set -euo pipefail
# An empty HOTLOAD_TIMEOUT in run.env reaches the Job as an empty string, which cwtrain rejects as
# a number. Unset it, so that the hook uses its default.
hotload_timeout=${CWTRAIN_HOTLOAD_TIMEOUT:-}
[[ -n "${hotload_timeout//[[:space:]]/}" ]] || unset CWTRAIN_HOTLOAD_TIMEOUT

: "${HF_MODEL:?}" "${NUM_ROLLOUT:=6}" "${ACTOR_GPUS:=2}" "${TP_SIZE:=2}"
WORK=${WORK:-/work}

echo "== install cwtrain + launcher"
pip install --quiet --no-cache-dir /opt/cwtrain/cwtrain-*.whl "huggingface_hub[cli]"
mkdir -p $WORK/py && cp /opt/cwtrain/coreweave_rollout.py /opt/cwtrain/run_hotload.py $WORK/py/

echo "== check snapshot access"
# Stop now, not at the first snapshot, if the trainer's credentials cannot list, read and write
# under the deployment's hot-load prefix. The check leaves no object behind.
python3 -m cwtrain.integrations.preflight

echo "== train (${TRAIN_SCRIPT:-train.py})"
cd /root/miles
# run_hotload.py takes the base model from BASE_S3_URI if set (the copy the deployment loaded),
# else from Hugging Face. It fetches dapo-math-17k, converts to Megatron torch_dist, starts Ray and
# submits the job with miles' execute_train.
exec python3 $WORK/py/run_hotload.py \
  --model-name "$HF_MODEL" ${MEGATRON_MODEL_TYPE:+--megatron-model-type "$MEGATRON_MODEL_TYPE"} \
  --model-dir "$WORK/models" --data-dir "$WORK/datasets" --work-dir "$WORK" \
  --num-gpus "$ACTOR_GPUS" --tp "$TP_SIZE" \
  --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-4}" --global-batch-size "${GLOBAL_BATCH_SIZE:-32}" \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-2048}" --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}" \
  --rm-type "${RM_TYPE:-math}" \
  --tis "${USE_TIS:-1}" \
  --train-script "${TRAIN_SCRIPT:-train.py}" --async-correction "${ASYNC_CORRECTION:-keep_old_actor}" \
  --rollout-module-dir "$WORK/py" \
  --base-s3-uri "${BASE_S3_URI:-}" \
  --extra-args="${EXTRA_TRAIN_ARGS:-}"
