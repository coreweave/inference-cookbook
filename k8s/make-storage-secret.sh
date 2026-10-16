#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Create the object-storage Secret that the Jobs mount at /aws, from an existing access key.
# The script reads the key from the environment. It never prints the key or passes it as a command
# argument. Export the key; do not type it on the command line, because shell history keeps it:
#
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
#   AWS_REGION=<zone> ./make-storage-secret.sh <kube-context> <namespace> <secret-name> [endpoint-url] [region]
#
# The region is required and has no default. Set it to the bucket's CoreWeave Availability Zone
# (S3 tools call it the region) with [region], AWS_REGION or AWS_DEFAULT_REGION.
#
# To create a CoreWeave AI Object Storage key, run `cwic cwobject token create --name <name>`. Its
# --duration takes seconds or Permanent (default: one hour). Set it to the length of the run, for
# example 604800 for a week. Do not use a permanent key.
#
# With WORKLOAD_IDENTITY=1 the Secret holds no key, only the region and endpoint. Use this for a
# trainer that runs as its own Kubernetes service account: the trainer Job exchanges the service
# account's token for temporary credentials (your example's README, "What you need").
set -euo pipefail
CTX=${1:?kube context}; NS=${2:?namespace}; NAME=${3:?secret name}
ENDPOINT=${4:-https://cwobject.com}; REGION=${5:-${AWS_REGION:-${AWS_DEFAULT_REGION:-}}}
: "${REGION:?set AWS_REGION, AWS_DEFAULT_REGION or the region argument to the CoreWeave Availability Zone of the bucket, for example US-EAST-04A}"
WORKLOAD_IDENTITY=${WORKLOAD_IDENTITY:-0}
if [ "$WORKLOAD_IDENTITY" != 1 ]; then
  : "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID, or WORKLOAD_IDENTITY=1 for a Secret without a key}"
  : "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY}"
fi
case "$REGION $ENDPOINT" in
  *'<'*'>'*) echo "make-storage-secret.sh: fill in the region and endpoint: '$REGION' '$ENDPOINT' still holds a <placeholder>" >&2; exit 1;;
esac
case "${AWS_ACCESS_KEY_ID:-} ${AWS_SECRET_ACCESS_KEY:-}" in
  *'<'*'>'*) echo "make-storage-secret.sh: AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY still holds a <placeholder>" >&2; exit 1;;
esac
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
printf '[default]\nregion = %s\nendpoint_url = %s\n' "$REGION" "$ENDPOINT" > "$TMP/config"
if [ "$WORKLOAD_IDENTITY" = 1 ]; then
  : > "$TMP/credentials"
else
  printf '[default]\naws_access_key_id = %s\naws_secret_access_key = %s\n' "$AWS_ACCESS_KEY_ID" "$AWS_SECRET_ACCESS_KEY" > "$TMP/credentials"
fi
kubectl --context "$CTX" -n "$NS" create secret generic "$NAME" \
  --from-file=config="$TMP/config" --from-file=credentials="$TMP/credentials" \
  --dry-run=client -o yaml | kubectl --context "$CTX" apply -f - >/dev/null
key=$([ "$WORKLOAD_IDENTITY" = 1 ] && echo "no key: workload identity" || echo "with a key")
echo "secret $NAME created in $CTX/$NS (endpoint $ENDPOINT, region $REGION, $key)"
