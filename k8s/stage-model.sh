#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Runs inside the stage-model Job: Hugging Face -> local disk -> s3://$BUCKET/$MODEL_PREFIX/$MODEL_SLUG/
set -euo pipefail
[[ ${HF_REVISION:-} =~ ^[0-9a-f]{40}$ ]] || { echo "HF_REVISION must be a full 40-character Hugging Face commit SHA" >&2; exit 2; }
pip install --quiet --no-cache-dir /opt/cwtrain/cwtrain-*.whl "huggingface_hub[cli]"
echo "downloading ${HF_MODEL}"
hf download "${HF_MODEL}" --revision "${HF_REVISION}" --local-dir "/work/${MODEL_SLUG}" --exclude "original/*" --exclude "*.pth" --exclude "*.gguf"
du -sh "/work/${MODEL_SLUG}"
python - <<'PY'
import os, logging
from cwtrain.hotload import inspect_snapshot, upload_snapshot
logging.basicConfig(level=logging.INFO)
info = inspect_snapshot(f"/work/{os.environ['MODEL_SLUG']}")
print(f"{info.type.name} snapshot, {len(info.files)} files, {info.total_bytes/2**30:.2f} GiB")
uri = upload_snapshot(info, bucket=os.environ["BUCKET"], path_prefix=os.environ["MODEL_PREFIX"],
                      identity=os.environ["MODEL_SLUG"], workers=16)
print("uploaded to", uri)
PY
