# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Exceptions raised by the hot-load workflow.

The errors the CoreWeave API returns, and :class:`~cwtrain.errors.CWTrainError`, the base of every
cwtrain error, are in :mod:`cwtrain.errors`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cwtrain.errors import CWTrainError

if TYPE_CHECKING:
    from cwtrain.infra.models import HotLoad, HotLoadReplica


class SnapshotError(CWTrainError):
    """A local checkpoint directory is not a usable hot-load snapshot."""


class SnapshotConflictError(SnapshotError):
    """The bucket already holds different content under a snapshot identity.

    Identities are immutable: once objects exist under ``<prefix>/<identity>/``, an upload may only
    complete them with identical content. Publish under a new identity instead.
    """

    def __init__(self, identity: str, conflicts: list[str]) -> None:
        self.identity = identity
        self.conflicts = conflicts
        shown = ", ".join(conflicts[:3]) + (
            f" and {len(conflicts) - 3} more" if len(conflicts) > 3 else ""
        )
        super().__init__(
            f"snapshot identity {identity!r} already holds different content ({shown}); "
            "identities are immutable, so publish under a new identity"
        )


class SnapshotAccessError(CWTrainError):
    """The storage credentials cannot do what a trainer does under the snapshot prefix.

    ``action`` is the S3 permission the failing call needs, such as ``s3:PutObject``, and
    ``code`` is the error code the object store returned.
    """

    def __init__(self, uri: str, action: str, code: str) -> None:
        self.uri = uri
        self.action = action
        self.code = code
        super().__init__(
            f"the storage credentials cannot use {uri}: {action} failed with {code}. The "
            "trainer's principal needs s3:ListBucket, s3:GetObject, s3:PutObject and "
            "s3:AbortMultipartUpload there, from the bucket policy (AllowTrainer) and, for a "
            "principal of its own, its organization access policy"
        )


class HotLoadFailedError(CWTrainError):
    """A hot-load reached a terminal state other than COMPLETED."""

    def __init__(self, hot_load: HotLoad, replicas: list[HotLoadReplica]) -> None:
        self.hot_load = hot_load
        self.replicas = replicas
        failed = [r for r in replicas if r.error_reason]
        reasons = "; ".join(f"{r.name}: {r.error_reason}" for r in failed[:5])
        more = f" (+{len(failed) - 5} more)" if len(failed) > 5 else ""
        super().__init__(
            f"hot-load {hot_load.id} for {hot_load.identity!r} ended {hot_load.state.name}"
            + (f": {reasons}{more}" if reasons else "")
        )


class HotLoadVerificationError(CWTrainError):
    """A hot-load reported COMPLETED but a replica does not serve the requested snapshot."""

    def __init__(
        self, hot_load: HotLoad, replicas: list[HotLoadReplica], problems: list[str]
    ) -> None:
        self.hot_load = hot_load
        self.replicas = replicas
        self.problems = problems
        super().__init__(
            f"hot-load {hot_load.id} for {hot_load.identity!r} is {hot_load.state.name} but "
            f"{len(problems)} replica(s) do not serve it: " + "; ".join(problems[:5])
        )


class HotLoadTimeoutError(CWTrainError):
    """The hot-load did not reach a terminal state within the wait budget."""

    def __init__(self, hot_load: HotLoad, replicas: list[HotLoadReplica], timeout: float) -> None:
        self.hot_load = hot_load
        self.replicas = replicas
        self.timeout = timeout
        ready = sum(1 for r in replicas if r.state.name == "READY")
        super().__init__(
            f"hot-load {hot_load.id} for {hot_load.identity!r} still {hot_load.state.name} after "
            f"{timeout:.0f}s ({ready}/{len(replicas)} replicas ready)"
        )
