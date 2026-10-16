#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook
#
# Sourced by each recipe's make-kit.sh: sets WHEEL to the cwtrain wheel that the kit bundles, or
# exits with what to do. That is CWTRAIN_WHEEL if set, else the newest wheel in libs/cwtrain/dist
# (by file time, not version), found relative to this file.
WHEEL=${CWTRAIN_WHEEL:-}
if [ -z "$WHEEL" ]; then
  for candidate in "$(dirname "${BASH_SOURCE[0]}")"/../libs/cwtrain/dist/cwtrain-*.whl; do
    if [ -f "$candidate" ] && { [ -z "$WHEEL" ] || [ "$candidate" -nt "$WHEEL" ]; }; then WHEEL=$candidate; fi
  done
fi
if [ -n "${CWTRAIN_WHEEL:-}" ] && [ ! -f "$WHEEL" ]; then echo "CWTRAIN_WHEEL=$WHEEL is not a file" >&2; exit 1; fi
[ -f "$WHEEL" ] || { echo "no wheel under libs/cwtrain/dist; run: (cd libs/cwtrain && uv build --wheel)" >&2; exit 1; }
