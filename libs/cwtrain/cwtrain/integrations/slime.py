# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""slime / miles post-write hook: hot-load every published weight version.

Both frameworks call a user function after each disk weight sync has been written::

    --custom-update-weight-post-write-path cwtrain.integrations.slime.hook

The function receives ``(args, version_dir, rollout_engines)``. The frameworks call it on every
trainer rank in full mode and on rank 0 in delta mode. This hook does its work on rank 0 only.

miles runs the disk sync with zero rollout engines by default. Upstream slime skips it unless it
starts with ``--update-weight-standalone-publish``, from the patch in
``training/recipes/slime/patches``.

How the calls map onto hot-load snapshots
-----------------------------------------

*Delta mode* (``--update-weight-mode delta --update-weight-transport disk``):

1. At baseline capture, the trainer calls the hook with the delta root directory. The trainer's
   baseline is the raw bytes of ``--hf-checkpoint``, and each later delta is an XOR against the
   previous version. So the hook publishes ``--hf-checkpoint`` as a **full** snapshot named
   ``<run>-v000000``. This is the chain root that the API requires. The deployment's own base
   weights cannot be the root.
2. Each sync then calls the hook with ``weight_vNNNNNN``. The hook publishes it as an
   **incremental** snapshot ``<run>-vNNNNNN`` whose predecessor is ``<run>-v(NNNNNN-1)``.
3. Every ``CWTRAIN_HOTLOAD_FULL_EVERY``-th version (default 20) is published as a **full**
   snapshot instead. The hook builds it from a byte-exact local mirror of the served weights
   (:class:`~cwtrain.hotload.WeightMirror`). This cuts the delta chain, so a chain never holds
   more than ``FULL_EVERY - 1`` deltas. The limit matters: a replica that starts with no peer to
   copy the weights from replays the chain from object storage. ModelExpress replays at most 64
   entries including the full root (63 deltas). The API rejects chains longer than 128.

*Full mode* (``--update-weight-mode full --update-weight-transport disk``): each
``weight_vNNNNNN`` is a complete HF checkpoint and is published as a full snapshot. No baseline is
needed.

What is verified
----------------

* With the mirror on, each delta is applied locally. Each changed tensor is checked against the
  Adler-32 the trainer wrote. A mismatch stops the run before anything is uploaded.
* Each version gets a manifest of every tensor's checksum, in ``<state dir>/manifests/`` and
  ``s3://<bucket>/<prefix>/_manifests/<identity>.json``. The engine's materialized checkpoint
  must hash to these checksums. Full snapshots carry them in their shards, and the engine
  verifies them on install.
* Some baselines are not published by this process: one found already hot-loaded, or one taken
  as given with ``PUBLISH_BASELINE=0``. Such a baseline is checked against its stored manifest
  before any delta chains onto it (:meth:`SlimePublisher._check_recorded_baseline`).
* A COMPLETED hot-load means every replica serves the new identity. The platform marks a
  replica READY only after it reads the engine's weight version back. It completes the hot-load
  only when every replica is READY. As a diagnostic, ``CWTRAIN_HOTLOAD_VERIFY=1`` re-reads the
  replicas and fails unless all are READY on the new identity.

Configuration is by environment variable because the frameworks pass nothing else:

==========================================  =====================================================
``CWTRAIN_HOTLOAD_DEPLOYMENT_ID``           Deployment UUID (required).
``CWTRAIN_HOTLOAD_RUN_ID``                  Identity prefix. Must be unique per training run: the
                                            platform keys weight versions by identity. Defaults to
                                            ``<disk dir name>-<UTC timestamp>``.
``CWTRAIN_HOTLOAD_WAIT``                    ``1`` (default) blocks until every replica serves the
                                            version, so the next rollout samples the new weights.
                                            ``0`` returns once the hot-load is accepted.
``CWTRAIN_HOTLOAD_TIMEOUT``                 Seconds to wait for a hot-load (default 1800).
``CWTRAIN_HOTLOAD_PUBLISH_BASELINE``        ``1`` (default) publishes the full baseline snapshot in
                                            delta mode; ``0`` asserts that ``<run>-v000000``
                                            exists and holds ``--hf-checkpoint``.
``CWTRAIN_HOTLOAD_FULL_EVERY``              Publish a full snapshot every N versions (default 20,
                                            at most 64; 0 never cuts the chain). Turns the
                                            mirror on.
``CWTRAIN_HOTLOAD_MIRROR``                  Keep the local weight mirror (needs numpy and
                                            zstandard). Default: on iff ``FULL_EVERY`` > 0.
``CWTRAIN_HOTLOAD_VERIFY``                  ``1`` re-checks every replica after each hot-load
                                            (default ``0``; COMPLETED already covers it).
``CWTRAIN_HOTLOAD_STATE_DIR``               Mirror, staged snapshots and manifests. Default
                                            ``<disk dir>-hotload`` next to the delta directory.
``CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY``     ``preserve`` (server default) or ``reset_all``, which
                                            drops KV cache computed under the previous weights;
                                            the recipes' synchronous loops set ``reset_all``.
``CWTRAIN_HOTLOAD_UPLOAD_WORKERS``          Concurrent uploads (default 8).
``COREWEAVE_API_TOKEN``                     API token, else ``CW_API_TOKEN``; see
                                            :mod:`cwtrain.infra.client`.
``COREWEAVE_API_URL``                       API origin override.
``CWTRAIN_API_HEADERS``                     JSON object of extra headers (dev servers only).
``CWTRAIN_API_INSECURE``                    ``1`` disables TLS verification (dev servers only).
``AWS_ACCESS_KEY_ID`` / ``..._SECRET_...``  Object storage credentials, via the boto3 chain.
``CW_S3_ENDPOINT``                          Object storage endpoint override; otherwise the AWS
                                            configuration's, else ``https://cwobject.com``.
==========================================  =====================================================
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cwtrain.errors import CWTrainError
from cwtrain.hotload import (
    HotLoad,
    HotLoader,
    ProgressEvent,
    PromptCachePolicy,
    SnapshotInfo,
    SnapshotType,
    WeightMirror,
    inspect_snapshot,
)
from cwtrain.hotload.delta import checkpoint_checksums
from cwtrain.hotload.snapshot import validate_identity
from cwtrain.infra.client import DEFAULT_BASE_URL, InferenceClient
from cwtrain.integrations._env import flag as _flag
from cwtrain.integrations._env import json_headers as _json_headers
from cwtrain.integrations._env import optional_flag as _optional_flag

log = logging.getLogger(__name__)

ENV_PREFIX = "CWTRAIN_HOTLOAD_"
DEFAULT_FULL_EVERY = 20
# FULL_EVERY=N leaves at most N - 1 deltas between full snapshots. ModelExpress replays at most 64
# chain entries, full root included, when a replica cold-starts from object storage.
MAX_FULL_EVERY = 64
#: Deltas such a replica can replay, and deltas the API accepts after a chain's full root.
MAX_COLD_START_DELTAS = MAX_FULL_EVERY - 1
MAX_API_DELTAS = 128
# slime writes weight_v{N:06d}: six digits, more when N reaches a million.
_VERSION_DIR = re.compile(r"^weight_v(\d{6,})$")


@dataclass(frozen=True)
class HookConfig:
    """Hook settings, normally read from the environment with :meth:`from_env`."""

    deployment_id: str
    run_id: str
    wait: bool = True
    timeout: float = 1800.0
    publish_baseline: bool = True
    full_every: int = DEFAULT_FULL_EVERY
    mirror: bool | None = None
    verify: bool = False
    state_dir: str = ""
    prompt_cache_policy: PromptCachePolicy = PromptCachePolicy.UNSPECIFIED
    upload_workers: int = 8
    api_base_url: str = DEFAULT_BASE_URL
    api_headers: dict[str, str] = field(default_factory=dict)
    api_insecure: bool = False

    def __post_init__(self) -> None:
        if not self.deployment_id:
            raise CWTrainError(f"{ENV_PREFIX}DEPLOYMENT_ID is required")
        if not self.run_id:
            raise CWTrainError(f"{ENV_PREFIX}RUN_ID must not be empty")
        if not 0 <= self.full_every <= MAX_FULL_EVERY:
            raise CWTrainError(
                f"{ENV_PREFIX}FULL_EVERY must be 0-{MAX_FULL_EVERY}, got {self.full_every}: "
                f"a replica that cold-starts from object storage replays at most "
                f"{MAX_COLD_START_DELTAS} deltas"
            )
        if self.full_every and self.mirror is False:
            raise CWTrainError(
                f"{ENV_PREFIX}FULL_EVERY needs the mirror; unset {ENV_PREFIX}MIRROR, or set "
                f"{ENV_PREFIX}FULL_EVERY=0 to never cut the delta chain"
            )
        # Fail on a bad prefix before any training step has been paid for.
        validate_identity(self.identity(999999))

    def warn_if_the_chain_is_never_cut(self) -> None:
        """Say what ``FULL_EVERY=0`` risks; called once per publisher or synchronizer."""
        if not self.full_every:
            log.warning(
                "%sFULL_EVERY=0 never cuts the delta chain: past %d deltas a replica that "
                "cold-starts without a peer cannot load its weights, and past %d the API "
                "rejects new hot-loads",
                ENV_PREFIX,
                MAX_COLD_START_DELTAS,
                MAX_API_DELTAS,
            )

    @property
    def mirror_enabled(self) -> bool:
        return self.mirror if self.mirror is not None else self.full_every > 0

    def identity(self, version: int) -> str:
        """The snapshot identity for a slime weight version number."""
        return f"{self.run_id}-v{int(version):06d}"

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, default_run_id: str | None = None
    ) -> HookConfig:
        env = os.environ if environ is None else environ
        policy_raw = env.get(f"{ENV_PREFIX}PROMPT_CACHE_POLICY", "").strip()
        policy = (
            PromptCachePolicy.parse(policy_raw) if policy_raw else PromptCachePolicy.UNSPECIFIED
        )
        if policy_raw and policy is PromptCachePolicy.UNSPECIFIED:
            raise CWTrainError(
                f"{ENV_PREFIX}PROMPT_CACHE_POLICY must be 'preserve' or 'reset_all', "
                f"got {policy_raw!r}"
            )
        headers = _json_headers(env, "CWTRAIN_API_HEADERS")
        try:
            raw = env.get(f"{ENV_PREFIX}FULL_EVERY", "").strip()
            full_every = int(raw) if raw else DEFAULT_FULL_EVERY
        except ValueError as exc:
            raise CWTrainError(f"{ENV_PREFIX}FULL_EVERY must be an integer, got {raw!r}") from exc
        return cls(
            deployment_id=env.get(f"{ENV_PREFIX}DEPLOYMENT_ID", "").strip(),
            run_id=env.get(f"{ENV_PREFIX}RUN_ID", "").strip() or (default_run_id or ""),
            wait=_flag(env, f"{ENV_PREFIX}WAIT", True),
            timeout=float(env.get(f"{ENV_PREFIX}TIMEOUT", "1800")),
            publish_baseline=_flag(env, f"{ENV_PREFIX}PUBLISH_BASELINE", True),
            full_every=full_every,
            mirror=_optional_flag(env, f"{ENV_PREFIX}MIRROR"),
            verify=_flag(env, f"{ENV_PREFIX}VERIFY", False),
            state_dir=env.get(f"{ENV_PREFIX}STATE_DIR", "").strip(),
            prompt_cache_policy=policy,
            upload_workers=int(env.get(f"{ENV_PREFIX}UPLOAD_WORKERS", "8")),
            api_base_url=env.get("COREWEAVE_API_URL", "").strip() or DEFAULT_BASE_URL,
            api_headers=headers,
            api_insecure=_flag(env, "CWTRAIN_API_INSECURE", False),
        )


def default_run_id(args: Any, now: datetime | None = None) -> str:
    """``<disk dir name>-<UTC timestamp>``: unique per process, readable in the hot-load list."""
    disk_dir = getattr(args, "update_weight_disk_dir", None) or "slime"
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    base = re.sub(r"[^A-Za-z0-9._-]", "-", Path(str(disk_dir)).name) or "slime"
    return f"{base}-{stamp}"[:100]


def is_rank_zero(args: Any = None, environ: Mapping[str, str] | None = None) -> bool:
    """True on the trainer rank that should publish.

    Order of evidence: ``args.rank`` when the framework sets it, an initialised
    ``torch.distributed`` group, then the ``RANK`` environment variable.
    """
    rank = getattr(args, "rank", None)
    if isinstance(rank, int):
        return rank == 0
    try:
        import torch.distributed as dist  # type: ignore[import-not-found]

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except ImportError:
        pass
    env = os.environ if environ is None else environ
    return env.get("RANK", "0") == "0"


class SlimePublisher:
    """Turns slime/miles post-write callbacks into hot-loads on one deployment."""

    def __init__(self, config: HookConfig, loader: HotLoader | None = None) -> None:
        self.config = config
        # The hook re-reads its settings on every call, so warn here, once per publisher.
        config.warn_if_the_chain_is_never_cut()
        # Versions actually hot-loaded. A delta whose base was a skipped no-op sync chains onto
        # the last snapshot the deployment received.
        self._published_versions: list[int] = []
        self._mirror: WeightMirror | None = None
        self._state_dir: Path | None = None
        if loader is None:
            client = InferenceClient(
                base_url=config.api_base_url,
                headers=config.api_headers or None,
                verify=not config.api_insecure,
            )
            loader = HotLoader(client, config.deployment_id, timeout=config.timeout)
        self.loader = loader

    def close(self) -> None:
        """Release the API client's connections (a no-op for injected loaders without one)."""
        client = getattr(self.loader, "client", None)
        if client is not None and hasattr(client, "close"):
            client.close()

    @property
    def has_state(self) -> bool:
        """Whether this publisher holds run state a replacement would lose."""
        return bool(self._published_versions) or self._mirror is not None

    # ------------------------------------------------------------------ dispatch

    def handle(self, args: Any, version_dir: str | os.PathLike[str]) -> dict[str, float] | None:
        path = Path(version_dir)
        match = _VERSION_DIR.match(path.name)
        if match is None:
            return self.publish_baseline(args, disk_dir=path)
        return self.publish_version(int(match.group(1)), path)

    def state_dir(self, disk_dir: Path | None = None) -> Path:
        """Where the mirror, staged full snapshots and manifests live."""
        if self._state_dir is None:
            if self.config.state_dir:
                self._state_dir = Path(self.config.state_dir)
            elif disk_dir is not None:
                self._state_dir = disk_dir.parent / f"{disk_dir.name}-hotload"
            else:
                raise CWTrainError("the hook needs the delta directory to place its state dir")
        return self._state_dir

    # ------------------------------------------------------------------ baseline

    def publish_baseline(self, args: Any, disk_dir: Path | None = None) -> dict[str, float] | None:
        """Publish ``--hf-checkpoint`` as the full snapshot ``<run>-v000000``.

        Skipped when ``PUBLISH_BASELINE=0``, and when a completed hot-load of that identity
        already exists. (A resumed run captures its baseline again but must not upload it
        again.) With the mirror enabled, this also creates the mirror from ``--hf-checkpoint``.
        A skipped baseline is checked against its manifest first, so a wrong checkpoint is not
        copied.

        Returns:
            ``hotload/*`` metrics, or ``None`` when there are none.

        Raises:
            CWTrainError: ``--hf-checkpoint`` is missing or is not a full HF checkpoint, or it
                does not match the recorded baseline.
        """
        hf_checkpoint = getattr(args, "hf_checkpoint", None)
        if disk_dir is None:
            disk_dir = Path(getattr(args, "update_weight_disk_dir", None) or "deltas")
        identity = self.config.identity(0)
        metrics: dict[str, float] = {}

        if self.config.mirror_enabled and not hf_checkpoint:
            raise CWTrainError("the weight mirror needs args.hf_checkpoint (--hf-checkpoint)")
        if self.config.publish_baseline and not hf_checkpoint:
            raise CWTrainError(
                "the trainer args carry no hf_checkpoint; the delta baseline is the bytes of "
                "--hf-checkpoint, so it must be set to publish the chain root"
            )
        info = None
        if self.config.publish_baseline or self.config.mirror_enabled:
            info = inspect_snapshot(hf_checkpoint)
            if info.type is not SnapshotType.FULL:
                raise CWTrainError(f"--hf-checkpoint {hf_checkpoint} is not a full HF checkpoint")
        asserted = not self.config.publish_baseline
        existing = None if asserted else self.loader.completed_hot_load(identity)

        if self.config.mirror_enabled:
            started = time.monotonic()
            checksums = None
            if asserted or existing is not None:  # a baseline this process does not publish
                checksums = self._check_recorded_baseline(identity, hf_checkpoint, asserted)
            self._mirror = WeightMirror.create(
                hf_checkpoint,
                self.state_dir(disk_dir) / "mirror",
                version="000000",
                checksums=checksums,
            )
            metrics["hotload/mirror_seconds"] = round(time.monotonic() - started, 3)

        if asserted:
            # The chain root exists. Later deltas must still resolve their base to it.
            log.info("hot-load baseline publish disabled; assuming %s exists", identity)
            self._published_versions.append(0)
            if self._mirror is not None:
                self._write_manifest(identity)
            return metrics or None
        if existing is not None:
            log.info("baseline %s already hot-loaded (%s); skipping", identity, existing.id)
            self._published_versions.append(0)
            if self._mirror is not None:
                self._write_manifest(identity)
            metrics["hotload/baseline_skipped"] = 1.0
            return metrics

        assert info is not None
        log.info(
            "publishing baseline %s from %s (%.1f GiB)",
            identity,
            hf_checkpoint,
            info.total_bytes / 2**30,
        )
        metrics.update(self._publish(info, identity))
        return metrics

    # ------------------------------------------------------------------- version

    def publish_version(self, version: int, version_dir: Path) -> dict[str, float]:
        identity = self.config.identity(version)
        info = inspect_snapshot(
            version_dir, identity_for_version=self._base_identity, allow_empty_delta=True
        )
        metrics: dict[str, float] = {}

        if self._mirror is not None:
            if not info.empty_delta:
                started = time.monotonic()
                report = self._mirror.apply_delta(version_dir)
                metrics.update(
                    {
                        "hotload/mirror_seconds": round(time.monotonic() - started, 3),
                        "hotload/tensors_changed": float(report.tensors_changed),
                        "hotload/tensors_verified": float(report.tensors_changed),
                        "hotload/bytes_changed": float(report.bytes_changed),
                    }
                )
            else:
                self._mirror.advance(f"{version:06d}")

        cut = bool(
            self._mirror is not None
            and self.config.full_every
            and version % self.config.full_every == 0
        )
        if cut:
            metrics.update(self._publish_full_from_mirror(identity))
            return metrics

        if info.empty_delta:
            # A sync that changed no bytes (zero gradient, all-equal rewards) writes only an index.
            # The deployment already serves these weights. Publish nothing, and let the next delta
            # chain onto the last published version.
            log.info("version %d changed no tensors; skipping hot-load", version)
            metrics.update({"hotload/skipped_empty": 1.0, "hotload/incremental": 1.0})
            return metrics
        if info.is_incremental and info.incremental is not None:
            log.info(
                "publishing delta %s (base %s) from %s",
                identity,
                info.incremental.previous_snapshot_identity,
                version_dir,
            )
        else:
            log.info("publishing full snapshot %s from %s", identity, version_dir)
        metrics.update(self._publish(info, identity))
        return metrics

    def _publish_full_from_mirror(self, identity: str) -> dict[str, float]:
        """Cut the delta chain: publish the mirror's current weights as a full snapshot."""
        assert self._mirror is not None
        staged = self.state_dir() / "staging" / identity
        started = time.monotonic()
        self._mirror.write_full_snapshot(staged)
        metrics = {
            "hotload/chain_cut": 1.0,
            "hotload/stage_seconds": round(time.monotonic() - started, 3),
        }
        try:
            info = inspect_snapshot(staged)
            log.info(
                "publishing full snapshot %s from the mirror (chain cut every %d versions, "
                "%.1f GiB)",
                identity,
                self.config.full_every,
                info.total_bytes / 2**30,
            )
            metrics.update(self._publish(info, identity))
        finally:
            shutil.rmtree(staged, ignore_errors=True)
        return metrics

    def _base_identity(self, base_version: str) -> str:
        """Identity of the snapshot a delta really applies to.

        miles numbers every sync, including skipped syncs that changed nothing. A skipped version
        is byte-identical to its predecessor. So the delta chains onto the newest version at or
        below ``base_version`` that was actually hot-loaded.
        """
        base = int(base_version)
        published = [v for v in self._published_versions if v <= base]
        return self.config.identity(max(published) if published else base)

    # ------------------------------------------------------------------ manifests

    def _check_recorded_baseline(
        self, identity: str, hf_checkpoint: str, asserted: bool
    ) -> dict[str, str] | None:
        """Check a baseline this process does not publish against its stored checksum manifest.

        Every delta is an XOR against the local ``--hf-checkpoint``. So the deployment's
        ``identity`` must hold exactly those bytes. Otherwise each delta chains onto weights the
        deployment does not have.

        A baseline found already hot-loaded (a resumed run) is reused only when its manifest
        proves the match. With ``asserted`` (``PUBLISH_BASELINE=0``), the match is taken as
        given: a missing manifest only warns, and the local checksums become the manifest.
        (``cwtrain hotload publish`` records no manifest.) In both cases, a manifest that differs
        or cannot be read stops the run.

        This hashes ``--hf-checkpoint`` in place, before the mirror copies it. A wrong checkpoint
        then costs one checksum pass, not a copy.

        Returns:
            The checksums for the mirror to reuse, or ``None`` when there was no manifest to
            compare them with.
        """
        fresh_run = f"start a fresh {ENV_PREFIX}RUN_ID" + (
            f" and unset {ENV_PREFIX}PUBLISH_BASELINE so the new run publishes its baseline"
            if asserted
            else ""
        )
        try:
            recorded = self.loader.read_manifest(identity)
        except Exception as exc:  # noqa: BLE001 - any failure leaves the baseline unverified
            raise CWTrainError(
                f"the checksum manifest of {identity} cannot be read ({exc}), so --hf-checkpoint "
                f"cannot be checked against the published baseline; fix read access to "
                f"_manifests/{identity}.json under the hot-load prefix, or {fresh_run}"
            ) from exc
        if recorded is None:
            if asserted:
                log.warning(
                    "%s has no checksum manifest to check --hf-checkpoint against; "
                    "%sPUBLISH_BASELINE=0 asserts that they match, so its manifest is recorded "
                    "from --hf-checkpoint",
                    identity,
                    ENV_PREFIX,
                )
                return None
            raise CWTrainError(
                f"{identity} is already hot-loaded but has no checksum manifest, so "
                f"--hf-checkpoint cannot be checked against it; if it holds exactly "
                f"--hf-checkpoint, set {ENV_PREFIX}PUBLISH_BASELINE=0 to assert that, or "
                f"{fresh_run}"
            )
        local = checkpoint_checksums(hf_checkpoint)
        stored = recorded.get("tensors") or {}
        differences = {
            "with other bytes": sorted(
                n for n in local.keys() & stored.keys() if local[n] != stored[n]
            ),
            "only in --hf-checkpoint": sorted(local.keys() - stored.keys()),
            "only in the published baseline": sorted(stored.keys() - local.keys()),
        }
        found = "; ".join(
            f"{len(names)} tensor(s) {kind}, e.g. {names[:3]}"
            for kind, names in differences.items()
            if names
        )
        if found:
            raise CWTrainError(
                f"--hf-checkpoint differs from the published {identity}: {found}; point "
                f"--hf-checkpoint at the weights published as {identity}, or {fresh_run}"
            )
        return local

    def _write_manifest(self, identity: str) -> None:
        """Record every tensor's checksum for ``identity`` locally and, best effort, in S3."""
        assert self._mirror is not None
        manifest = self._mirror.manifest(
            identity, run_id=self.config.run_id, deployment_id=self.config.deployment_id
        )
        local = self.state_dir() / "manifests" / f"{identity}.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(manifest, indent=0, sort_keys=True))
        try:
            uri = self.loader.upload_manifest(identity, manifest)
        except Exception as exc:  # noqa: BLE001 - the manifest is a record, not the payload
            log.warning("could not upload checksum manifest for %s: %s", identity, exc)
        else:
            log.info(
                "checksum manifest for %s: %s (%d tensors)", identity, uri, len(manifest["tensors"])
            )

    # -------------------------------------------------------------------- shared

    def _publish(self, info: SnapshotInfo, identity: str) -> dict[str, float]:
        metrics: dict[str, float] = {
            "hotload/snapshot_bytes": float(info.total_bytes),
            "hotload/incremental": 1.0 if info.is_incremental else 0.0,
        }
        started = time.monotonic()
        self.loader.upload(info, identity, workers=self.config.upload_workers)
        metrics["hotload/upload_seconds"] = round(time.monotonic() - started, 3)

        started = time.monotonic()
        # CWTRAIN_HOTLOAD_TIMEOUT bounds the whole activation. Waiting for the deployment, for
        # an active hot-load, and for this hot-load share one budget.
        deadline = started + self.config.timeout

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        # The API admits a hot-load only on a READY deployment. A new deployment may still be
        # downloading its base weights when the trainer publishes the first snapshot.
        self.loader.wait_for_deployment_ready(timeout=remaining())
        hot_load = self.loader.trigger(
            identity, info, prompt_cache_policy=self.config.prompt_cache_policy, timeout=remaining()
        )
        if self.config.wait:
            last: list[ProgressEvent] = []
            hot_load = self.loader.wait(hot_load.id, timeout=remaining(), on_progress=last.append)
            if last:
                metrics["hotload/replicas_ready"] = float(last[-1].ready)
                metrics["hotload/replicas_total"] = float(last[-1].total)
            if self.config.verify:
                replicas = self.loader.verify_served(hot_load.id, identity)
                metrics["hotload/replicas_verified"] = float(len(replicas))
        metrics["hotload/hot_load_seconds"] = round(time.monotonic() - started, 3)
        metrics["hotload/completed"] = 1.0 if hot_load.completed else 0.0
        log.info("hot-load %s for %s: %s", hot_load.id, identity, hot_load.state.name)
        version = _version_of(identity, self.config.run_id)
        if version is not None:
            self._published_versions.append(version)
        if self._mirror is not None:
            # Record the manifest only after the deployment accepts the version.
            self._write_manifest(identity)
        return metrics


def _version_of(identity: str, run_id: str) -> int | None:
    prefix = f"{run_id}-v"
    if identity.startswith(prefix) and identity[len(prefix) :].isdigit():
        return int(identity[len(prefix) :])
    return None


# One publisher per process so the deployment config and HTTP client are reused across syncs.
_publishers: dict[str, SlimePublisher] = {}


def _publisher(args: Any) -> SlimePublisher:
    default_run_id = (
        None if os.environ.get(f"{ENV_PREFIX}RUN_ID", "").strip() else _process_run_id(args)
    )
    config = HookConfig.from_env(default_run_id=default_run_id)
    publisher = _publishers.get(config.deployment_id)
    if publisher is not None and publisher.config != config:
        changed = sorted(
            f.name
            for f in dataclasses.fields(config)
            if getattr(config, f.name) != getattr(publisher.config, f.name)
        )
        if publisher.has_state:
            # A new publisher forgets which versions were hot-loaded and drops the mirror. The
            # next delta could then chain onto an identity the deployment never received.
            raise CWTrainError(
                f"hot-load settings changed mid-run ({', '.join(changed)}); "
                f"{ENV_PREFIX}* must stay fixed for the lifetime of a run"
            )
        log.warning("hot-load settings changed before anything was published: %s", changed)
        publisher.close()
        publisher = None
    if publisher is None:
        publisher = SlimePublisher(config)
        _publishers[config.deployment_id] = publisher
    return publisher


_default_run_id: str | None = None


def _process_run_id(args: Any) -> str:
    """A default run ID chosen once per process, so every version shares one prefix."""
    global _default_run_id
    if _default_run_id is None:
        _default_run_id = default_run_id(args)
        log.warning(
            "%sRUN_ID is unset; using %r. Set it explicitly so identities are stable across "
            "restarts",
            ENV_PREFIX,
            _default_run_id,
        )
    return _default_run_id


def hook(
    args: Any, version_dir: str, rollout_engines: Sequence[Any] = ()
) -> dict[str, float] | None:
    """slime / miles ``--custom-update-weight-post-write-path`` entry point.

    Returns:
        A metrics dict (``hotload/*`` keys) on rank 0, and ``None`` on other ranks.

    Exceptions propagate, so a failed publish stops the run instead of drifting off-policy without
    notice.
    """
    del rollout_engines  # The deployment is the rollout engine; slime's engines are not used.
    if not is_rank_zero(args):
        return None
    return _publisher(args).handle(args, version_dir)


def reset() -> None:
    """Drop cached publishers and run IDs (tests, or a new run in the same process)."""
    global _default_run_id
    for publisher in _publishers.values():
        publisher.close()
    _publishers.clear()
    _default_run_id = None


__all__ = [
    "HookConfig",
    "HotLoad",
    "SlimePublisher",
    "default_run_id",
    "hook",
    "is_rank_zero",
    "reset",
]
