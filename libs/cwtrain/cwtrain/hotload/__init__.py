# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Checkpoint hot-load: publish a trainer's weights into a running Dedicated Inference deployment.

The package has three layers. Use the one that fits:

* :func:`inspect_snapshot` / :func:`upload_snapshot` - validate a local checkpoint directory
  and upload it to the deployment's checkpoint bucket.
* :class:`HotLoader` - the end-to-end operation: inspect, upload, create the hot-load, and
  wait for every replica to serve the new weights.
* :class:`WeightMirror` - for delta-publishing trainers (slime, miles): a byte-exact local copy
  of the served weights, used to cut delta chains with full snapshots and to verify checksums.

The API client, :class:`~cwtrain.infra.InferenceClient`, is in :mod:`cwtrain.infra`. The API
errors are in :mod:`cwtrain.errors`. The API's hot-load types are re-exported here.
"""

from cwtrain.hotload.errors import (
    HotLoadFailedError,
    HotLoadTimeoutError,
    HotLoadVerificationError,
    SnapshotAccessError,
    SnapshotConflictError,
    SnapshotError,
)
from cwtrain.hotload.mirror import DeltaReport, MirrorError, WeightMirror
from cwtrain.hotload.runner import HotLoader, ProgressEvent
from cwtrain.hotload.snapshot import (
    SnapshotInfo,
    check_snapshot_access,
    delta_index_for_upload,
    inspect_snapshot,
    snapshot_prefix,
    upload_snapshot,
)
from cwtrain.infra.models import (
    ChecksumFormat,
    CompressionFormat,
    HotLoad,
    HotLoadReplica,
    HotLoadSpec,
    HotLoadState,
    HotLoadStatus,
    IncrementalSnapshotMetadata,
    PromptCachePolicy,
    ReplicaState,
    SnapshotRef,
    SnapshotType,
    TransitionMode,
)

__all__ = [
    "ChecksumFormat",
    "CompressionFormat",
    "DeltaReport",
    "HotLoad",
    "HotLoadFailedError",
    "HotLoadReplica",
    "HotLoadSpec",
    "HotLoadState",
    "HotLoadStatus",
    "HotLoadTimeoutError",
    "HotLoadVerificationError",
    "HotLoader",
    "IncrementalSnapshotMetadata",
    "MirrorError",
    "ProgressEvent",
    "PromptCachePolicy",
    "ReplicaState",
    "SnapshotAccessError",
    "SnapshotConflictError",
    "SnapshotError",
    "SnapshotInfo",
    "SnapshotRef",
    "SnapshotType",
    "TransitionMode",
    "WeightMirror",
    "check_snapshot_access",
    "delta_index_for_upload",
    "inspect_snapshot",
    "snapshot_prefix",
    "upload_snapshot",
]
