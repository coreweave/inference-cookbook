# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The object-storage client. Uses CoreWeave AI Object Storage unless configured otherwise.

Snapshot uploads (:mod:`cwtrain.hotload`) and the recipes' model staging use :func:`s3_client`, so
they all resolve the endpoint and region the same way.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

from cwtrain.errors import CWTrainError

log = logging.getLogger(__name__)

DEFAULT_S3_ENDPOINT = "https://cwobject.com"
# A region that no AWS configuration names and no endpoint serves; see _endpoint_is_configured.
_PROBE_REGION = "cwtrain-endpoint-probe"


def s3_client(
    endpoint_url: str | None = None,
    *,
    region_name: str | None = None,
    **kwargs: Any,
) -> Any:
    """A boto3 S3 client pointed at CoreWeave AI Object Storage.

    Credentials come from the standard boto3 chain (``AWS_ACCESS_KEY_ID`` /
    ``AWS_SECRET_ACCESS_KEY``, a profile, or ``~/.aws/credentials``).

    The endpoint is the first one found:

    1. ``endpoint_url``.
    2. ``$CW_S3_ENDPOINT``.
    3. The endpoint in the standard AWS configuration, even an AWS one: ``AWS_ENDPOINT_URL_S3``,
       ``AWS_ENDPOINT_URL``, or the active profile's ``endpoint_url`` or ``services`` section.
       Skipped when ``ignore_configured_endpoint_urls`` is set.
    4. The primary CoreWeave endpoint (``DEFAULT_S3_ENDPOINT``), which works from anywhere.

    Inside a CoreWeave cluster, the LOTA endpoint (``http://cwlota.com``) caches reads, such as
    model staging. It does not make uploads faster.

    The region is required and has no default. It is the first of ``region_name``,
    ``$AWS_REGION``, ``$AWS_DEFAULT_REGION`` and the active profile's ``region``. On CoreWeave the
    region is the bucket's Availability Zone, such as ``US-EAST-04A``.

    Raises:
        CWTrainError: no region is set.
    """
    import boto3
    from botocore.config import Config

    endpoint_url = endpoint_url or os.environ.get("CW_S3_ENDPOINT", "").strip() or None
    config = kwargs.pop("config", None)
    if config is None:
        config = Config(
            retries={"max_attempts": 5, "mode": "adaptive"},
            s3={"addressing_style": "virtual"},
            # boto3 runs up to 10 multipart streams per file; size the pool for several files.
            max_pool_connections=max(10, _env_int("CW_S3_MAX_POOL", 128)),
        )
    else:
        config = copy.deepcopy(config)  # botocore rewrites a config's retries in place
    region = region_name or _configured_region()
    if endpoint_url is None and not _endpoint_is_configured(config):
        # Without this, boto3 falls back to the AWS endpoint.
        endpoint_url = DEFAULT_S3_ENDPOINT
    return boto3.client(
        "s3", endpoint_url=endpoint_url, region_name=region, config=config, **kwargs
    )


def _endpoint_is_configured(config: Any) -> bool:
    """Whether the AWS configuration names an S3 endpoint, as boto3 resolves it.

    boto3 does the lookup, so every source it reads counts and its
    ``ignore_configured_endpoint_urls`` switches apply. A configured endpoint is the same in every
    region, but the AWS default contains the region name. In a real region the two can be the
    same URL. In a made-up region they differ only when an endpoint is configured. So this
    compares, in that region, a client that honors the configuration with one that ignores it.
    The probes use placeholder keys: the lookup needs no credentials, and resolving real ones can
    stall on an instance metadata timeout.
    """
    import boto3
    from botocore.config import Config

    probe = {
        "region_name": _PROBE_REGION,
        "aws_access_key_id": "probe",
        "aws_secret_access_key": "probe",
    }
    honoring = Config(ignore_configured_endpoint_urls=config.ignore_configured_endpoint_urls)
    honored = boto3.client("s3", config=honoring, **probe).meta.endpoint_url
    ignoring = Config(ignore_configured_endpoint_urls=True)
    default = boto3.client("s3", config=ignoring, **probe).meta.endpoint_url
    return honored != default


def _configured_region() -> str:
    """``$AWS_REGION``, else the region boto3 resolves (``$AWS_DEFAULT_REGION``, the profile)."""
    import boto3

    # boto3 reads AWS_DEFAULT_REGION and the profile's region itself, but not AWS_REGION
    region = os.environ.get("AWS_REGION", "").strip()
    region = region or (boto3.session.Session().region_name or "").strip()
    if not region:
        raise CWTrainError(
            "no object storage region: set AWS_REGION, AWS_DEFAULT_REGION or the AWS profile's "
            "region to the bucket's CoreWeave Availability Zone, for example US-EAST-04A"
        )
    return region


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("ignoring %s=%r: not an integer; using %d", name, raw, default)
        return default
