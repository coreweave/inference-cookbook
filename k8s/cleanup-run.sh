#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Delete everything one example's run created:
#
#   cleanup-run.sh <run.env> [extra-secret-variable...]
#
# Removes the deployment, the example's Jobs, its Secrets and kit ConfigMap, the trainer service
# account, and the exact snapshot identities in the deployment's hot-load history:
# <prefix>/<identity>/ and <prefix>/_manifests/<identity>.json, nothing else. Each extra argument
# names a run.env variable that holds one more Secret to delete (INFERENCE_KEY_SECRET, GYM_SECRET);
# the example README's "Cleaning up" section shows its call. Run it with the kubectl context used
# for the setup steps and the admin's object-storage credentials exported: only the admin may
# delete from the bucket. The staged base model and the bucket Job's write check (_probe) stay,
# because other deployments may use them.
set -eu
ENV_FILE=${1:?usage: cleanup-run.sh <run.env> [extra-secret-variable...]}; shift
cd "$(dirname "$ENV_FILE")"   # deployment.run.json, when present, sits next to run.env
set -a; # shellcheck disable=SC1090
. "./${ENV_FILE##*/}"; set +a
extra_secrets=()
for var in "$@"; do extra_secrets+=("${!var:?set $var in run.env}"); done
# deployment.run.json, when present, decides; a file without hotLoad.pathPrefix stops the script.
# Like the API, ignore slashes at either end of the prefix ("/hotload/" is "hotload").
HOTLOAD_PREFIX=$(HOTLOAD_PREFIX=${HOTLOAD_PREFIX:-} python3 -c 'import json, os, sys
if os.path.exists("deployment.run.json"):
    p = json.load(open("deployment.run.json")).get("hotLoad", {}).get("pathPrefix") or ""
    p.strip("/") or sys.exit("deployment.run.json has no hotLoad.pathPrefix, so the snapshots sit at the top of the bucket; remove them by hand")
else:
    p = os.environ["HOTLOAD_PREFIX"]
print(p.strip("/"))')
: "${DEPLOYMENT_ID:?set DEPLOYMENT_ID in run.env}" \
  "${HOTLOAD_PREFIX:?set HOTLOAD_PREFIX in run.env to the hotLoad.pathPrefix of the deployment}" "${NAMESPACE:?}" \
  "${BUCKET:?}" "${BUCKET_ZONE:?}" "${MODEL_PREFIX:?}" "${MODEL_SLUG:?}" "${JOB_NAME:?}" "${KIT_NAME:?}" \
  "${STORAGE_SECRET:?}" "${ADMIN_STORAGE_SECRET:?}" "${API_TOKEN_SECRET:?}" \
  "${TRAINER_SERVICE_ACCOUNT?set TRAINER_SERVICE_ACCOUNT in run.env, empty for default}"
case "$DEPLOYMENT_ID $NAMESPACE $BUCKET $BUCKET_ZONE $MODEL_PREFIX $MODEL_SLUG $HOTLOAD_PREFIX $JOB_NAME
      $KIT_NAME $STORAGE_SECRET $ADMIN_STORAGE_SECRET $API_TOKEN_SECRET
      $TRAINER_SERVICE_ACCOUNT ${extra_secrets[*]+${extra_secrets[*]}}" in
  *'<'[A-Za-z]*'>'*) echo "fill in run.env before cleaning up" >&2; exit 1;;
esac
# List exact identities while the deployment exists. A failed list must stop cleanup,
# even if it wrote JSON before failing. Validate every identity before deleting anything.
hot_loads=$(cwtrain hotload list --deployment "$DEPLOYMENT_ID" --limit 100000)
identities=$(printf '%s\n' "$hot_loads" | python3 -c 'import json, sys
hot_loads = json.load(sys.stdin)
if not isinstance(hot_loads, list):
    sys.exit("expected a hot-load history list; refusing cleanup")
identities = [h["spec"]["identity"] for h in hot_loads]
for identity in identities:
    if (not isinstance(identity, str) or not 1 <= len(identity) <= 128
            or identity in (".", "..", "initial", "_manifests") or "/" in identity
            or any(ord(c) < 32 or ord(c) == 127 for c in identity)):
        sys.exit(f"unsafe snapshot identity {identity!r}; remove its objects by hand")
print("\n".join(sorted(set(identities))))')
cwtrain deployment delete "$DEPLOYMENT_ID" --ignore-not-found
kubectl -n "$NAMESPACE" delete job "cwtrain-$JOB_NAME" "cwtrain-stage-$JOB_NAME" cwtrain-make-bucket --ignore-not-found
kubectl -n "$NAMESPACE" delete secret "$STORAGE_SECRET" "$ADMIN_STORAGE_SECRET" "$API_TOKEN_SECRET" \
  ${extra_secrets[@]+"${extra_secrets[@]}"} --ignore-not-found
kubectl -n "$NAMESPACE" delete configmap "$KIT_NAME" --ignore-not-found
[ -z "$TRAINER_SERVICE_ACCOUNT" ] || kubectl -n "$NAMESPACE" delete serviceaccount "$TRAINER_SERVICE_ACCOUNT" --ignore-not-found
[ -n "$identities" ] || echo "found no hot-loads of $DEPLOYMENT_ID, so nothing under $HOTLOAD_PREFIX/ is removed" >&2
# A trailing slash bounds each snapshot prefix. No wildcard filters: names such as
# experiment and experiment-v2, or names containing *, must never select each other.
printf '%s\n' "$identities" | while IFS= read -r identity; do
  [ -n "$identity" ] || continue
  aws s3 rm --recursive "s3://$BUCKET/$HOTLOAD_PREFIX/$identity/" \
    --endpoint-url https://cwobject.com --region "$BUCKET_ZONE"
  aws s3 rm "s3://$BUCKET/$HOTLOAD_PREFIX/_manifests/$identity.json" \
    --endpoint-url https://cwobject.com --region "$BUCKET_ZONE"
done
