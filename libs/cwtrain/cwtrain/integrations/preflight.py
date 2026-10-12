# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Pre-flight check for a trainer Job: can its credentials publish snapshots?

Run it before training starts, with the environment the hot-load hooks get::

    python -m cwtrain.integrations.preflight

It reads the deployment's hot-load bucket and prefix, then lists the prefix, reads a key that does
not exist and starts a multipart upload that it aborts. That needs the trainer's permissions and
leaves nothing behind. So a bucket policy or organization access policy that does not cover the
trainer fails here, in seconds, instead of at the first snapshot, after a training step.

==========================================  =====================================================
Variable                                    Meaning
==========================================  =====================================================
``CWTRAIN_HOTLOAD_DEPLOYMENT_ID``           The deployment that receives the snapshots.
``COREWEAVE_API_TOKEN``                     API token, else ``CW_API_TOKEN``.
``COREWEAVE_API_URL``                       API origin override.
``CWTRAIN_API_HEADERS``                     JSON object of extra headers (dev servers only).
``CWTRAIN_API_INSECURE``                    ``1`` disables TLS verification (dev servers only).
==========================================  =====================================================

Object storage credentials and the endpoint come from the same places as for the hooks; see
:func:`cwtrain.infra.storage.s3_client`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

from botocore.exceptions import BotoCoreError

from cwtrain.errors import CWTrainError
from cwtrain.hotload import HotLoader
from cwtrain.infra.client import DEFAULT_BASE_URL, InferenceClient, token_from_env
from cwtrain.integrations._env import flag, json_headers

DEPLOYMENT_ENV = "CWTRAIN_HOTLOAD_DEPLOYMENT_ID"


def check(environ: Mapping[str, str] | None = None, *, s3: Any = None, **client_kwargs: Any) -> str:
    """Check the trainer's snapshot access for the deployment the environment names.

    ``client_kwargs`` go to :class:`InferenceClient`; tests pass a transport.

    Returns:
        The ``s3://`` URI of the snapshot prefix.

    Raises:
        CWTrainError: a setting is missing or invalid, the deployment has no hot-load section, or
            the object store refused a call (:class:`~cwtrain.hotload.SnapshotAccessError`).
    """
    env = os.environ if environ is None else environ
    deployment_id = env.get(DEPLOYMENT_ENV, "").strip()
    if not deployment_id:
        raise CWTrainError(f"{DEPLOYMENT_ENV} is required")
    try:
        client = InferenceClient(
            token_from_env(env),
            base_url=env.get("COREWEAVE_API_URL", "").strip() or DEFAULT_BASE_URL,
            headers=json_headers(env, "CWTRAIN_API_HEADERS") or None,
            verify=not flag(env, "CWTRAIN_API_INSECURE", False),
            **client_kwargs,
        )
    except ValueError as exc:  # no token and no headers
        raise CWTrainError(str(exc)) from exc
    with client:
        return HotLoader(client, deployment_id, s3=s3).check_snapshot_access()


def main(environ: Mapping[str, str] | None = None) -> int:
    try:
        uri = check(environ)
    except (CWTrainError, BotoCoreError) as exc:
        print(f"snapshot access check failed: {exc}", file=sys.stderr)
        return 1
    print(f"snapshot access OK: {uri} (list, read, multipart write and abort)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
