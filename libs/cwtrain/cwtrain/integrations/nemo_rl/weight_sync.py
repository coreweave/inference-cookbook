# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""NeMo-RL ``WeightSynchronizer`` that refits by checkpoint hot-load.

When the generation object has a ``weight_synchronizer``, ``refit_policy_generation`` hands the
whole weight transfer to it. On each refit, this implementation does these steps:

1. Ask every policy worker to export the current weights in HF layout
   (``export_hf_checkpoint``, from the worker extensions in
   :mod:`cwtrain.integrations.nemo_rl.worker`). Each tensor uses the dtype of the deployment's
   checkpoint. Check that the export has exactly that checkpoint's tensors, dtypes and shapes.
2. Publish the export as a full snapshot (on the first sync, and every ``full_every`` versions),
   or as a zstd XOR delta against the previous export (:func:`cwtrain.hotload.delta.write_delta`).
3. Hot-load it, and wait until the hot-load completes. The platform reports completion only when
   every replica serves it. As a diagnostic, ``verify`` re-checks each replica after that.
4. Write a per-version checksum manifest next to the snapshots.

Settings come from ``policy.generation.coreweave_cfg`` (``deployment_id``, ``run_id``,
``full_every``, ``verify``, ``wait``, ``timeout``, ``state_dir``, ``prompt_cache_policy``). The
``CWTRAIN_HOTLOAD_*`` environment variables of the slime/miles hook are the fallback.

The policy's rank-0 worker writes the export. So ``state_dir`` must be visible to both that worker
and the training driver (one node, or a shared filesystem).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cwtrain.errors import CWTrainError
from cwtrain.hotload import HotLoader, ProgressEvent, inspect_snapshot
from cwtrain.hotload.delta import checkpoint_checksums, write_delta, write_full_snapshot
from cwtrain.hotload.mirror import tensor_locations
from cwtrain.infra.client import InferenceClient
from cwtrain.integrations.nemo_rl import _missing_dependency
from cwtrain.integrations.slime import ENV_PREFIX, HookConfig

try:
    from nemo_rl.weight_sync.interfaces import WeightSynchronizer
except ModuleNotFoundError as exc:
    raise _missing_dependency(exc) from exc

log = logging.getLogger(__name__)

_CFG_KEYS = (
    "deployment_id",
    "run_id",
    "full_every",
    "verify",
    "wait",
    "timeout",
    "state_dir",
    "prompt_cache_policy",
    "upload_workers",
)


def hook_config_from(cfg: dict[str, Any], environ: dict[str, str] | None = None) -> HookConfig:
    """Build the hook config from ``coreweave_cfg`` keys, with ``CWTRAIN_HOTLOAD_*`` as fallback.

    The run ID defaults to ``nemo-rl-<UTC timestamp>``.
    """
    env = dict(os.environ if environ is None else environ)
    for key in _CFG_KEYS:
        if cfg.get(key) is not None:
            env[f"{ENV_PREFIX}{key.upper()}"] = str(cfg[key])
    default_run_id = "nemo-rl-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return HookConfig.from_env(env, default_run_id=default_run_id)


def _replicas_summary(metrics: dict[str, float]) -> str:
    """The replica part of the per-refit line: what the wait and any re-check saw."""
    if "hotload/replicas_total" not in metrics:
        return "replicas not waited for"
    ready, total = metrics["hotload/replicas_ready"], metrics["hotload/replicas_total"]
    summary = f"replicas ready {int(ready)}/{int(total)}"
    if "hotload/replicas_verified" in metrics:
        summary += f", verified {int(metrics['hotload/replicas_verified'])}"
    return summary


class HotLoadWeightSynchronizer(WeightSynchronizer):
    """Export -> (full | delta) snapshot -> hot-load -> verify, once per refit."""

    def __init__(
        self,
        policy: Any,
        generation: Any,
        generation_config: dict[str, Any],
        *,
        hf_checkpoint: str,
        loader: HotLoader | None = None,
    ) -> None:
        cw = dict(generation_config.get("coreweave_cfg") or {})
        self.config = hook_config_from(cw)
        self.config.warn_if_the_chain_is_never_cut()
        self._policy = policy
        self._generation = generation
        self.hf_checkpoint = Path(hf_checkpoint)
        self.state_dir = Path(self.config.state_dir or cw.get("state_dir") or "/work/hotload")
        # The deployment serves this layout. Every export must match each tensor's dtype and shape.
        self.base_layout = {
            name: (loc.dtype, loc.shape)
            for name, loc in tensor_locations(self.hf_checkpoint).items()
        }
        self.base_names = set(self.base_layout)
        self.base_dtypes = {name: dtype for name, (dtype, _shape) in self.base_layout.items()}
        if not self.base_names:
            raise CWTrainError(f"{self.hf_checkpoint} holds no safetensors weights")
        if loader is None:
            client = InferenceClient(
                base_url=self.config.api_base_url,
                headers=self.config.api_headers or None,
                verify=not self.config.api_insecure,
            )
            loader = HotLoader(client, self.config.deployment_id, timeout=self.config.timeout)
        self.loader = loader
        self._version = 0
        self._stale = True
        self._last_published: tuple[int, Path] | None = None
        log.info(
            "hot-load weight synchronizer: deployment=%s run=%s full_every=%d state_dir=%s",
            self.config.deployment_id,
            self.config.run_id,
            self.config.full_every,
            self.state_dir,
        )

    # ----------------------------------------------------------------- interface

    def init_communicator(self) -> None:  # no collective to set up
        return None

    @property
    def is_stale(self) -> bool:
        return self._stale

    @property
    def served_identity(self) -> str | None:
        """The identity the deployment serves after the last sync; ``None`` before the first."""
        if self._last_published is None:
            return None
        return self.config.identity(self._last_published[0])

    def mark_stale(self) -> None:
        self._stale = True

    def shutdown(self) -> None:
        return None

    def sync_weights(
        self, *, timer: Any = None, kv_scales: dict[str, float] | None = None
    ) -> dict[str, float]:
        if kv_scales:
            raise NotImplementedError("FP8 kv scales are not part of a hot-load snapshot")
        version = self._version
        identity = self.config.identity(version)
        metrics: dict[str, float] = {"hotload/version": float(version)}

        started = time.monotonic()
        export_dir = self.state_dir / "exports" / f"v{version:06d}"
        self._export(export_dir)
        metrics["hotload/export_seconds"] = round(time.monotonic() - started, 3)

        checksums = checkpoint_checksums(export_dir)

        cut = version == 0 or self._last_published is None
        if self.config.full_every and version % self.config.full_every == 0:
            cut = True
        # Check each identity, full or delta, before anything is recorded or uploaded under it.
        # A restarted run must never overwrite a published version with different weights:
        # object storage keeps old shards of equal size while the manifest describes new ones.
        existing = self.loader.completed_hot_load(identity)
        self._check_reusable(identity, checksums, existing)

        staged: Path | None = None
        if existing is not None:
            # A restarted run publishes again the versions it already hot-loaded. The snapshot is
            # in the bucket, but the deployment may serve a later version now. Make sure the next
            # rollout samples from this version.
            log.info("%s already hot-loaded (%s); reusing it", identity, existing.id)
            print(f"[cwtrain] {identity} already hot-loaded; reusing it", flush=True)
            served = self.loader.current_identity()
            if served != identity:
                print(
                    f"[cwtrain] deployment serves {served}; hot-loading {identity} again",
                    flush=True,
                )
                metrics.update(
                    self._activate(
                        identity,
                        # _activate waits (and re-checks, with verify on), so the rollback
                        # only creates the hot-load.
                        lambda remaining: self.loader.rollback(
                            identity,
                            prompt_cache_policy=self.config.prompt_cache_policy,
                            wait=False,
                            timeout=remaining,
                        ),
                    )
                )
                metrics["hotload/restored"] = 1.0
            self._write_manifest(identity, checksums)
            if self._last_published is not None:
                shutil.rmtree(self._last_published[1], ignore_errors=True)
            self._last_published = (version, export_dir)
            self._version += 1
            self._stale = False
            metrics["hotload/reused_existing"] = 1.0
            return metrics
        if cut:
            staged = write_full_snapshot(
                export_dir, self.state_dir / "staging" / identity, checksums
            )
            info = inspect_snapshot(staged)
            metrics["hotload/incremental"] = 0.0
        else:
            prev_version, prev_export = self._last_published  # type: ignore[misc]
            staged = self.state_dir / "deltas" / f"weight_v{version:06d}"
            stats = write_delta(
                prev_export,
                export_dir,
                staged,
                version=f"{version:06d}",
                base_version=f"{prev_version:06d}",
            )
            metrics.update(
                {
                    "hotload/tensors_changed": float(stats.tensors_changed),
                    "hotload/bytes_changed": float(stats.bytes_changed),
                    "hotload/wire_bytes": float(stats.wire_bytes),
                    "hotload/incremental": 1.0,
                }
            )
            if stats.is_empty:
                log.info("version %d changed no tensors; the deployment already serves it", version)
                print(
                    f"[cwtrain] version {version} changed no tensors; nothing to hot-load",
                    flush=True,
                )
                shutil.rmtree(staged, ignore_errors=True)
                shutil.rmtree(export_dir, ignore_errors=True)
                self._version += 1
                self._stale = False
                metrics["hotload/skipped_empty"] = 1.0
                return metrics
            info = inspect_snapshot(staged, previous_identity=self.config.identity(prev_version))

        try:
            metrics.update(self._publish(info, identity))
        finally:
            shutil.rmtree(staged, ignore_errors=True)
        # Record the manifest only after the deployment accepts the version.
        self._write_manifest(identity, checksums)
        if self._last_published is not None:
            shutil.rmtree(self._last_published[1], ignore_errors=True)
        self._last_published = (version, export_dir)
        self._version += 1
        self._stale = False
        # NeMo-RL's driver does not surface library logging; print one line per refit.
        print(
            f"[cwtrain] hot-load {identity} ({'full' if cut else 'delta'}): "
            f"export {metrics['hotload/export_seconds']:.1f}s, upload "
            f"{metrics.get('hotload/upload_seconds', 0):.1f}s, hot-load "
            f"{metrics.get('hotload/hot_load_seconds', 0):.1f}s, {_replicas_summary(metrics)}",
            flush=True,
        )
        return metrics

    # ------------------------------------------------------------------- steps

    def _export(self, export_dir: Path) -> None:
        shutil.rmtree(export_dir, ignore_errors=True)
        results = self._policy.run_all_workers_single_data(
            "export_hf_checkpoint",
            path=str(export_dir),
            allowed_names=sorted(self.base_names),
            config_dir=str(self.hf_checkpoint),
            dtypes=self.base_dtypes,
        )
        if not (export_dir / "model.safetensors.index.json").is_file():
            raise CWTrainError(
                f"policy workers reported {results!r} but no export appeared at {export_dir}; "
                "the training driver must share state_dir with the policy's rank-0 worker"
            )
        exported = tensor_locations(export_dir)
        if set(exported) != self.base_names:
            missing = sorted(self.base_names - set(exported))[:5]
            extra = sorted(set(exported) - self.base_names)[:5]
            raise CWTrainError(
                f"export tensor set differs from the deployment's base checkpoint: "
                f"missing {missing}, extra {extra}"
            )
        changed = [
            f"{name} is {loc.dtype} {list(loc.shape)}, the base checkpoint's is "
            f"{self.base_layout[name][0]} {list(self.base_layout[name][1])}"
            for name, loc in sorted(exported.items())
            if (loc.dtype, loc.shape) != self.base_layout[name]
        ]
        if changed:
            raise CWTrainError(
                f"export layout differs from the deployment's base checkpoint for "
                f"{len(changed)} tensor(s), e.g. {changed[:3]}"
            )

    def _publish(self, info: Any, identity: str) -> dict[str, float]:
        metrics: dict[str, float] = {"hotload/snapshot_bytes": float(info.total_bytes)}
        started = time.monotonic()
        self.loader.upload(info, identity, workers=self.config.upload_workers)
        metrics["hotload/upload_seconds"] = round(time.monotonic() - started, 3)
        metrics.update(
            self._activate(
                identity,
                lambda remaining: self.loader.trigger(
                    identity,
                    info,
                    prompt_cache_policy=self.config.prompt_cache_policy,
                    timeout=remaining,
                ),
            )
        )
        return metrics

    def _activate(self, identity: str, start: Callable[[float], Any]) -> dict[str, float]:
        """Create the hot-load with ``start``, wait for it, and re-check every replica if asked.

        With ``wait`` (the default), this returns when the hot-load is COMPLETED. The platform
        reports COMPLETED only when every replica serves the new identity. As a diagnostic,
        ``verify`` re-reads the replicas after that.

        The configured timeout bounds the whole activation: waiting for the deployment, for an
        active hot-load (``start`` receives the remaining budget), and for this hot-load.
        """
        metrics: dict[str, float] = {}
        started = time.monotonic()
        deadline = started + self.config.timeout

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        self.loader.wait_for_deployment_ready(timeout=remaining())
        hot_load = start(remaining())
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
        return metrics

    def _check_reusable(self, identity: str, checksums: dict[str, str], existing: Any) -> None:
        """Refuse to publish under an identity whose stored manifest names different weights.

        The export also becomes the base of the next delta. So an already hot-loaded version is
        reused only when its stored manifest proves the deployment serves exactly this export. A
        missing or unreadable manifest refuses the reuse.

        Raises:
            CWTrainError: The stored manifest differs from ``checksums``, or ``existing`` is set
                and its manifest is missing or cannot be read.
        """
        try:
            stored = self.loader.read_manifest(identity)
        except Exception as exc:  # noqa: BLE001 - any failure means the reuse cannot be verified
            if existing is None:
                log.warning("could not read the stored manifest for %s: %s", identity, exc)
                return
            raise CWTrainError(
                f"{identity} is already hot-loaded but its checksum manifest cannot be read "
                f"({exc}); the export cannot be verified against it, so start a new run id "
                "(CWTRAIN_HOTLOAD_RUN_ID)"
            ) from exc
        if stored is None:
            if existing is not None:
                raise CWTrainError(
                    f"{identity} is already hot-loaded but has no checksum manifest, so this "
                    "export cannot be verified against it; start a new run id "
                    "(CWTRAIN_HOTLOAD_RUN_ID)"
                )
            return
        if stored.get("tensors") != checksums:
            state = "is hot-loaded" if existing is not None else "was published"
            raise CWTrainError(
                f"{identity} {state} with different weights than this export; this run id has "
                "been used before, so start a new one (CWTRAIN_HOTLOAD_RUN_ID) instead of "
                "reusing it"
            )

    def _write_manifest(self, identity: str, checksums: dict[str, str]) -> None:
        manifest = {
            "identity": identity,
            "checksum_format": "adler32",
            "tensors": dict(sorted(checksums.items())),
            "run_id": self.config.run_id,
            "deployment_id": self.config.deployment_id,
        }
        local = self.state_dir / "manifests" / f"{identity}.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(manifest, indent=0, sort_keys=True))
        try:
            uri = self.loader.upload_manifest(identity, manifest)
        except Exception as exc:  # noqa: BLE001 - the manifest is a record, not the payload
            log.warning("could not upload checksum manifest for %s: %s", identity, exc)
        else:
            log.info("checksum manifest for %s: %s", identity, uri)


__all__ = ["HotLoadWeightSynchronizer", "hook_config_from"]
