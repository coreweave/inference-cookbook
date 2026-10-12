# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""End-to-end hot-load: inspect a checkpoint, upload it, trigger the swap, wait for completion."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cwtrain.errors import CWTrainError, HotLoadInProgressError
from cwtrain.hotload.errors import (
    HotLoadFailedError,
    HotLoadTimeoutError,
    HotLoadVerificationError,
    SnapshotError,
)
from cwtrain.hotload.snapshot import (
    SnapshotInfo,
    check_snapshot_access,
    inspect_snapshot,
    snapshot_prefix,
    upload_snapshot,
    validate_identity,
)
from cwtrain.infra.client import InferenceClient
from cwtrain.infra.models import (
    Deployment,
    HotLoad,
    HotLoadReplica,
    HotLoadState,
    IncrementalSnapshotMetadata,
    PromptCachePolicy,
    ReplicaState,
    SnapshotType,
)
from cwtrain.infra.storage import s3_client

log = logging.getLogger(__name__)

#: Sub-prefix next to the snapshots where :meth:`HotLoader.upload_manifest` stores per-version
#: tensor checksum manifests. The API does not read it.
MANIFEST_PREFIX = "_manifests"
#: While waiting, a hot-load is logged when its state or a replica's state changes, and also
#: every this many seconds, so a long install still shows progress.
_WAIT_LOG_HEARTBEAT = 60.0


@dataclass(frozen=True)
class ProgressEvent:
    """One observation of a hot-load while waiting for it."""

    hot_load: HotLoad
    replicas: list[HotLoadReplica]
    elapsed: float

    @property
    def ready(self) -> int:
        return sum(1 for r in self.replicas if r.state is ReplicaState.READY)

    @property
    def total(self) -> int:
        return len(self.replicas)

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for r in self.replicas:
            counts[r.state.name] = counts.get(r.state.name, 0) + 1
        parts = ", ".join(f"{k.lower()}={v}" for k, v in sorted(counts.items()))
        return f"{self.hot_load.state.name.lower()} after {self.elapsed:.0f}s" + (
            f" ({parts})" if parts else ""
        )


class HotLoader:
    """Drive checkpoint hot-loads for one deployment.

    Typical use inside a training loop::

        loader = HotLoader(InferenceClient(), deployment_id)
        loader.publish("/ckpt/step_000100", identity="step_000100")

    The deployment's hot-load configuration (bucket, prefix) is read once and cached.

    Args:
        client: An :class:`InferenceClient`.
        deployment_id: The UUID of a deployment created with a ``hot_load`` section.
        s3: A boto3 S3 client. Built from the default credential chain when omitted.
        poll_interval: Seconds between status polls while waiting.
        timeout: Default wait budget in seconds for a hot-load to finish.
    """

    def __init__(
        self,
        client: InferenceClient,
        deployment_id: str,
        *,
        s3: Any = None,
        poll_interval: float = 5.0,
        timeout: float = 1800.0,
    ) -> None:
        self.client = client
        self.deployment_id = deployment_id
        self._s3 = s3
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._deployment: Deployment | None = None

    @property
    def s3(self) -> Any:
        """The boto3 client used for uploads, built from the default chain on first use."""
        if self._s3 is None:
            self._s3 = s3_client()
        return self._s3

    # ------------------------------------------------------------------ deployment

    @property
    def deployment(self) -> Deployment:
        if self._deployment is None:
            self.refresh_deployment()
        assert self._deployment is not None
        return self._deployment

    def refresh_deployment(self) -> Deployment:
        """Re-read the deployment and cache it.

        Raises:
            CWTrainError: the deployment has no ``hot_load`` section.
        """
        dep = self.client.get_deployment(self.deployment_id)
        if dep.hot_load is None:
            raise CWTrainError(
                f"deployment {self.deployment_id} was created without a hot_load section; "
                "hot-loading cannot be enabled after creation"
            )
        self._deployment = dep
        return dep

    def wait_for_deployment_ready(
        self, timeout: float | None = None, poll_interval: float | None = None
    ) -> Deployment:
        """Block until the deployment reports READY, which a hot-load requires.

        Raises:
            CWTrainError: the deployment is not READY within ``timeout`` seconds.
        """
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        interval = poll_interval if poll_interval is not None else self.poll_interval
        while True:
            dep = self.refresh_deployment()
            if dep.ready:
                return dep
            if time.monotonic() >= deadline:
                raise CWTrainError(
                    f"deployment {self.deployment_id} is {dep.status.name}, not READY"
                )
            time.sleep(interval)

    # ------------------------------------------------------------------- hot-loads

    def active_hot_load(self) -> HotLoad | None:
        """The hot-load currently PENDING or IN_PROGRESS on this deployment, if any."""
        for hl in self.client.list_hot_loads(self.deployment_id):
            if not hl.terminal:
                return hl
        return None

    def latest_completed(self) -> HotLoad | None:
        """The most recent COMPLETED hot-load. An incremental delta must chain to it."""
        for hl in self.client.list_hot_loads(self.deployment_id):
            if hl.completed:
                return hl
        return None

    def current_identity(self) -> str:
        """Identity the deployment serves: the latest completed hot-load or ``"initial"``."""
        latest = self.latest_completed()
        return latest.identity if latest else "initial"

    def check_snapshot_access(self) -> str:
        """Check that the storage credentials can list, read and write the snapshot prefix.

        See :func:`~cwtrain.hotload.snapshot.check_snapshot_access`; nothing is left behind.
        Returns the prefix's ``s3://`` URI.
        """
        cfg = self.deployment.hot_load
        assert cfg is not None
        return check_snapshot_access(self.s3, cfg.bucket, cfg.path_prefix)

    def upload(
        self,
        checkpoint: str | os.PathLike[str] | SnapshotInfo,
        identity: str,
        *,
        previous_identity: str | None = None,
        workers: int = 8,
    ) -> SnapshotInfo:
        """Inspect and upload a checkpoint without triggering the hot-load."""
        validate_identity(identity)
        info = (
            checkpoint
            if isinstance(checkpoint, SnapshotInfo)
            else inspect_snapshot(checkpoint, previous_identity=previous_identity)
        )
        cfg = self.deployment.hot_load
        assert cfg is not None
        upload_snapshot(
            info,
            bucket=cfg.bucket,
            path_prefix=cfg.path_prefix,
            identity=identity,
            s3=self.s3,
            workers=workers,
        )
        return info

    def trigger(
        self,
        identity: str,
        info: SnapshotInfo | None = None,
        *,
        previous_identity: str | None = None,
        snapshot_type: SnapshotType | None = None,
        prompt_cache_policy: PromptCachePolicy = PromptCachePolicy.UNSPECIFIED,
        hot_load_id: str | None = None,
        wait_for_active: bool = True,
        timeout: float | None = None,
    ) -> HotLoad:
        """Create the hot-load for an already uploaded snapshot.

        The type and delta metadata come from ``info`` when given. Without ``info``, a
        ``previous_identity`` makes the hot-load INCREMENTAL on that predecessor. Otherwise the
        hot-load is FULL.

        If ``wait_for_active`` is true and another hot-load is in progress, this waits for it to
        finish instead of raising ``HotLoadInProgressError``. The total wait is at most
        ``timeout`` seconds (default: the loader's timeout).

        Raises:
            SnapshotError: ``previous_identity`` together with ``info`` or a FULL
                ``snapshot_type``, an INCREMENTAL ``snapshot_type`` without it, or an invalid
                ``previous_identity``.
        """
        if info is not None:
            if previous_identity is not None:
                raise SnapshotError(
                    "pass info or previous_identity, not both; info carries the base"
                )
            snapshot_type = info.type
            incremental = info.incremental
        elif previous_identity is not None:
            if snapshot_type is SnapshotType.FULL:
                raise SnapshotError("a full snapshot has no previous_identity")
            snapshot_type = SnapshotType.INCREMENTAL
            incremental = IncrementalSnapshotMetadata(
                previous_snapshot_identity=validate_identity(previous_identity)
            )
        elif snapshot_type is SnapshotType.INCREMENTAL:
            raise SnapshotError("an incremental snapshot needs previous_identity")
        else:
            snapshot_type = SnapshotType.FULL if snapshot_type is None else snapshot_type
            incremental = None
        return self._create(
            identity,
            snapshot_type=snapshot_type,
            incremental=incremental,
            prompt_cache_policy=prompt_cache_policy,
            hot_load_id=hot_load_id,
            wait_for_active=wait_for_active,
            timeout=timeout,
        )

    def _create(
        self,
        identity: str,
        *,
        snapshot_type: SnapshotType,
        incremental: IncrementalSnapshotMetadata | None,
        prompt_cache_policy: PromptCachePolicy,
        hot_load_id: str | None,
        wait_for_active: bool,
        timeout: float | None,
    ) -> HotLoad:
        hot_load_id = hot_load_id or str(uuid.uuid4())
        budget = timeout if timeout is not None else self.timeout
        deadline = time.monotonic() + budget
        while True:
            try:
                return self.client.create_hot_load(
                    self.deployment_id,
                    identity,
                    snapshot_type=snapshot_type,
                    incremental=incremental,
                    prompt_cache_policy=prompt_cache_policy,
                    hot_load_id=hot_load_id,
                )
            except HotLoadInProgressError:
                remaining = deadline - time.monotonic()
                if not wait_for_active or remaining <= 0:
                    raise
                active = self.active_hot_load()
                if active is None:
                    # It finished between the create and the list; wait briefly and retry.
                    time.sleep(min(self.poll_interval, remaining))
                    continue
                log.info("waiting for active hot-load %s (%s) first", active.id, active.identity)
                self.wait(active.id, timeout=remaining, raise_on_failure=False)

    def wait(
        self,
        hot_load_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float | None = None,
        on_progress: Callable[[ProgressEvent], None] | None = None,
        raise_on_failure: bool = True,
    ) -> HotLoad:
        """Poll until the hot-load is terminal.

        Raises:
            HotLoadFailedError: the hot-load ended FAILED or CANCELED and ``raise_on_failure``.
            HotLoadTimeoutError: the budget elapsed first.
        """
        budget = timeout if timeout is not None else self.timeout
        interval = poll_interval if poll_interval is not None else self.poll_interval
        start = time.monotonic()
        last_seen: tuple[Any, ...] | None = None
        last_logged = 0.0
        while True:
            hl = self.client.get_hot_load(hot_load_id)
            replicas = self.client.list_hot_load_replicas(hot_load_id)
            event = ProgressEvent(hl, replicas, time.monotonic() - start)
            # The summary includes elapsed time, so compare the states it reports instead.
            seen = (
                hl.state,
                tuple(
                    sorted((r.zone, r.name, r.state, r.current_snapshot_identity) for r in replicas)
                ),
            )
            if seen != last_seen or event.elapsed - last_logged >= _WAIT_LOG_HEARTBEAT:
                log.info("hot-load %s (%s): %s", hl.id, hl.identity, event.summary())
                last_seen, last_logged = seen, event.elapsed
            if on_progress:
                on_progress(event)
            if hl.terminal:
                if hl.completed or not raise_on_failure:
                    return hl
                raise HotLoadFailedError(hl, replicas)
            if event.elapsed >= budget:
                raise HotLoadTimeoutError(hl, replicas, budget)
            time.sleep(min(interval, max(0.0, budget - event.elapsed)))

    def publish(
        self,
        checkpoint: str | os.PathLike[str] | SnapshotInfo,
        identity: str,
        *,
        previous_identity: str | None = None,
        prompt_cache_policy: PromptCachePolicy = PromptCachePolicy.UNSPECIFIED,
        wait: bool = True,
        timeout: float | None = None,
        on_progress: Callable[[ProgressEvent], None] | None = None,
        workers: int = 8,
    ) -> HotLoad:
        """Upload ``checkpoint`` as ``identity`` and hot-load it onto the deployment.

        Most training loops need only this call. It validates the directory, uploads it to the
        deployment's checkpoint bucket, creates the hot-load, and by default blocks until every
        replica serves the new weights.

        Args:
            checkpoint: A directory holding a HuggingFace safetensors checkpoint or a slime/miles
                delta, or a :class:`SnapshotInfo` from :func:`inspect_snapshot`.
            identity: The snapshot identity, a single path segment such as ``step_000100``.
            previous_identity: For deltas, the identity of the base snapshot. Derived from the
                delta index's ``base_version`` when omitted.
            prompt_cache_policy: ``PRESERVE`` (server default) or ``RESET_ALL``.
            wait: Block until terminal. When false, return the PENDING hot-load.
            timeout: Wait budget in seconds. Default: the loader's timeout.
            on_progress: Called with a :class:`ProgressEvent` at each poll.
            workers: Number of concurrent uploads.
        """
        info = self.upload(
            checkpoint, identity, previous_identity=previous_identity, workers=workers
        )
        hl = self.trigger(identity, info, prompt_cache_policy=prompt_cache_policy)
        if not wait:
            return hl
        return self.wait(hl.id, timeout=timeout, on_progress=on_progress)

    def verify_served(self, hot_load_id: str, identity: str) -> list[HotLoadReplica]:
        """Check that a finished hot-load put ``identity`` on every replica.

        The hot-load must be COMPLETED, and every replica must be READY with
        ``current_snapshot_identity == identity``. Returns the replicas.

        Raises:
            HotLoadVerificationError: any replica is not READY on ``identity``.
            HotLoadFailedError: the hot-load is not COMPLETED.
        """
        hl = self.client.get_hot_load(hot_load_id)
        replicas = self.client.list_hot_load_replicas(hot_load_id)
        if not hl.completed:
            raise HotLoadFailedError(hl, replicas)
        problems = [
            f"{r.name}: {r.state.name} on {r.current_snapshot_identity or 'unknown'!r}"
            + (f" ({r.error_reason})" if r.error_reason else "")
            for r in replicas
            if r.state is not ReplicaState.READY or r.current_snapshot_identity != identity
        ]
        if not replicas:
            problems.append("no replicas reported")
        if problems:
            raise HotLoadVerificationError(hl, replicas, problems)
        log.info("verified %d replica(s) serve %s", len(replicas), identity)
        return replicas

    def manifest_key(self, identity: str) -> str:
        """Object key of the checksum manifest for ``identity`` in the deployment's bucket."""
        cfg = self.deployment.hot_load
        assert cfg is not None
        return (
            snapshot_prefix(cfg.path_prefix, f"{MANIFEST_PREFIX}/{validate_identity(identity)}")
            + ".json"
        )

    def upload_manifest(self, identity: str, manifest: dict[str, Any]) -> str:
        """Store a per-tensor checksum manifest next to the snapshots; returns its ``s3://`` URI.

        Manifests go to ``<path_prefix>/_manifests/<identity>.json``. Anyone with bucket access
        can use them to check a version's weights.
        """
        cfg = self.deployment.hot_load
        assert cfg is not None
        key = self.manifest_key(identity)
        self.s3.put_object(
            Bucket=cfg.bucket,
            Key=key,
            Body=json.dumps(manifest, indent=0, sort_keys=True).encode(),
            ContentType="application/json",
        )
        return f"s3://{cfg.bucket}/{key}"

    def read_manifest(self, identity: str) -> dict[str, Any] | None:
        """The checksum manifest stored for ``identity``, or ``None`` when there is none."""
        cfg = self.deployment.hot_load
        assert cfg is not None
        try:
            obj = self.s3.get_object(Bucket=cfg.bucket, Key=self.manifest_key(identity))
        except Exception as exc:  # botocore's ClientError carries the code in ``response``
            code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "NotFound", "404"}:
                return None
            raise
        return json.loads(obj["Body"].read())

    def completed_hot_load(self, identity: str) -> HotLoad | None:
        """The most recent COMPLETED hot-load of ``identity`` on this deployment, if any."""
        for hl in self.client.list_hot_loads(self.deployment_id):
            if hl.identity == identity and hl.completed:
                return hl
        return None

    def rollback(
        self,
        identity: str,
        *,
        prompt_cache_policy: PromptCachePolicy = PromptCachePolicy.UNSPECIFIED,
        hot_load_id: str | None = None,
        wait: bool = True,
        timeout: float | None = None,
    ) -> HotLoad:
        """Re-serve a snapshot from a completed hot-load.

        The API has no rollback operation. A rollback is a new hot-load of an older identity
        whose objects are still in the bucket. The snapshot type and delta metadata are copied
        from the earlier hot-load, so the chain resolves the same way. An active hot-load is
        waited for first, as in :meth:`trigger`. With ``wait``, this also waits for the rollback.

        Raises:
            CWTrainError: no completed hot-load of ``identity`` exists.
        """
        previous = self.completed_hot_load(identity)
        if previous is None:
            raise CWTrainError(f"no completed hot-load for identity {identity!r} to roll back to")
        hl = self._create(
            identity,
            snapshot_type=previous.spec.type,
            incremental=previous.spec.incremental_snapshot_metadata,
            prompt_cache_policy=prompt_cache_policy,
            hot_load_id=hot_load_id,
            wait_for_active=True,
            timeout=timeout,
        )
        return self.wait(hl.id, timeout=timeout) if wait else hl


__all__ = ["MANIFEST_PREFIX", "HotLoader", "ProgressEvent", "HotLoadState"]
