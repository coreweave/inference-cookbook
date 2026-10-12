# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The CoreWeave services a training run uses: Dedicated Inference and AI Object Storage.

* :class:`InferenceClient` - a thin typed client for the Dedicated Inference API: gateways,
  deployments and hot-loads, one call per method.
* :mod:`cwtrain.infra.lifecycle` - waits built on the client: until a gateway or deployment is
  READY, or until a deleted one is gone. Also finds a deployment's rollout URL.
* :func:`s3_client` - the object-storage client for snapshot uploads and model staging.

The hot-load workflow built on these is in :mod:`cwtrain.hotload`.
"""

from cwtrain.infra.client import (
    DEFAULT_BASE_URL,
    TOKEN_ENV_VAR,
    TOKEN_ENV_VARS,
    InferenceClient,
    token_from_env,
)
from cwtrain.infra.models import (
    ChecksumFormat,
    CompressionFormat,
    Condition,
    ConditionStatus,
    Deployment,
    DeploymentHotLoadConfig,
    Gateway,
    HotLoad,
    HotLoadReplica,
    HotLoadSpec,
    HotLoadState,
    HotLoadStatus,
    IncrementalSnapshotMetadata,
    PromptCachePolicy,
    ReplicaState,
    ResourceStatus,
    SnapshotRef,
    SnapshotType,
    TransitionMode,
)
from cwtrain.infra.storage import DEFAULT_S3_ENDPOINT, s3_client

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_S3_ENDPOINT",
    "TOKEN_ENV_VAR",
    "TOKEN_ENV_VARS",
    "ChecksumFormat",
    "CompressionFormat",
    "Condition",
    "ConditionStatus",
    "Deployment",
    "DeploymentHotLoadConfig",
    "Gateway",
    "HotLoad",
    "HotLoadReplica",
    "HotLoadSpec",
    "HotLoadState",
    "HotLoadStatus",
    "IncrementalSnapshotMetadata",
    "InferenceClient",
    "PromptCachePolicy",
    "ReplicaState",
    "ResourceStatus",
    "SnapshotRef",
    "SnapshotType",
    "TransitionMode",
    "s3_client",
    "token_from_env",
]
