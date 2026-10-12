#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Render a manifest with the values in an env file: ./render.sh run.env trainer-job.yaml | kubectl apply -f -
#
# The env file must set every ${VAR} that the manifest uses (an empty value is fine), and no value
# may still hold a <placeholder> from run.env.example. envsubst renders both cases with no error.
set -euo pipefail
ENV_FILE=${1:?env file}; MANIFEST=${2:?manifest}
set -a; # shellcheck disable=SC1090
source "$ENV_FILE"; set +a
missing=(); unfilled=(); placeholder='<[A-Za-z][A-Za-z0-9 _-]*>'
for VAR in $(grep -o '\${[A-Za-z_][A-Za-z0-9_]*}' "$MANIFEST" | tr -d '${}' | sort -u); do
  if [ -z "${!VAR+set}" ]; then
    missing+=("$VAR")
  # EXTRA_* and *_HEADERS hold free-form text, where "</answer>" or "<trace-id>" is not a placeholder
  elif [[ $VAR != EXTRA_* && $VAR != *_HEADERS && ${!VAR} =~ $placeholder ]]; then
    unfilled+=("$VAR")
  fi
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "render.sh: $MANIFEST uses ${missing[*]}, which $ENV_FILE does not set" >&2
  exit 1
fi
if [ ${#unfilled[@]} -gt 0 ]; then
  echo "render.sh: fill in ${unfilled[*]} in $ENV_FILE; it still holds a <placeholder>" >&2
  exit 1
fi
# envsubst does not re-indent multi-line values, which breaks the YAML block scalars that carry
# EXTRA_* arguments. So flatten those values to one line first.
for VAR in $(compgen -v EXTRA_); do
  printf -v "$VAR" '%s' "${!VAR//$'\n'/ }"; export "${VAR?}"
done
envsubst < "$MANIFEST"
