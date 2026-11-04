# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Report cwtrain's metrics and settings into the trainer's W&B run, when it has one.

cwtrain never starts a run and never imports ``wandb`` itself. A trainer that logs to W&B has
imported it and opened ``wandb.run`` in its own processes; these functions write into that run
from the same process, and do nothing without one. A failure to log is warned about once and never
stops training.

Where the metrics land:

* slime and miles: the hot-load hook logs ``hotload/*`` on the ``rollout/step`` axis, where step
  ``v`` is weight version ``v``, the version that rollout ``v`` of the synchronous loop samples.
  The rollout functions bind cwtrain's ``coreweave/*`` rollout metrics to the same axis.
* NeMo-RL logs every row with an explicit step, so the synchronizer does not log; the recipe's
  patch merges its metrics into the step's ``train/`` metrics (``train/hotload/*``).

Both also record cwtrain's settings in the run config, under ``cwtrain``.

The timeline: each phase of a step is a span, logged as ``timeline/<lane>/start`` and
``timeline/<lane>/end`` (Unix seconds) with ``timeline/step`` (see :data:`LANES`). A row carries the
spans its process saw, and every timeline key (NaN for the others, :func:`timeline_row`), so that a
query for all of them returns every row and can put the phases on one wall-clock axis.
"""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

log = logging.getLogger(__name__)

#: Config section that holds cwtrain's settings.
CONFIG_SECTION = "cwtrain"

#: The phases of a training step, in the order the timeline chart shows them:
#:
#: * ``rollout``: sampling on the deployment; ``reward``: scoring the samples.
#: * ``logprobs``: the trainer's log-prob pass; ``update``: its optimizer steps (one span per step
#:   with slime and miles); ``train`` and ``merge``: Axolotl's whole run and its LoRA merge.
#: * ``export``, ``mirror``, ``upload``: getting the new weights to object storage.
#: * ``hotload``: from requesting the hot-load until every replica serves it.
#:   ``hotload_first_replica`` and ``hotload_peers``: the platform's own timestamps for the
#:   replica that installs the snapshot first and for the others, which usually copy from it.
LANES = (
    "rollout",
    "reward",
    "logprobs",
    "update",
    "train",
    "merge",
    "export",
    "mirror",
    "upload",
    "hotload",
    "hotload_first_replica",
    "hotload_peers",
)
#: The training step a row's spans belong to. Step ``i`` is the hot-load of weight version
#: ``i``, rollout ``i`` that samples it, and the training on that rollout.
TIMELINE_STEP = "timeline/step"

_bound: set[tuple[str, str, str]] = set()
_warned = False


def active_run() -> Any | None:
    """The process's open W&B run, or ``None``. Never imports ``wandb``."""
    wandb = sys.modules.get("wandb")
    return getattr(wandb, "run", None) if wandb is not None else None


def _warn_once(what: str, exc: BaseException) -> None:
    global _warned
    if not _warned:
        _warned = True
        log.warning("could not %s in the W&B run (%s); training continues", what, exc)


def bind(prefixes: Sequence[str], step_metric: str) -> bool:
    """Plot every metric under ``prefixes`` (``"coreweave/"``) against ``step_metric``.

    Returns whether a run was open.
    """
    run = active_run()
    if run is None:
        return False
    try:
        for prefix in prefixes:
            key = (str(getattr(run, "id", "")), prefix, step_metric)
            if key not in _bound:
                run.define_metric(f"{prefix}*", step_metric=step_metric)
                _bound.add(key)
    except Exception as exc:  # noqa: BLE001 - tracking must never stop training
        _warn_once("define metrics", exc)
    return True


def log_metrics(
    metrics: Mapping[str, float],
    *,
    step_metric: str,
    step: float,
    prefixes: Sequence[str],
) -> bool:
    """Log ``metrics`` with ``step_metric`` set to ``step``, as one row of the open run.

    The row advances W&B's internal step, as every row of slime, miles and the Hugging Face
    Trainer does; they plot against step metrics instead. Never use this in a trainer that logs
    with explicit ``step=`` values (NeMo-RL): W&B would then drop the trainer's next rows as out of
    order.

    Returns whether the row was logged.
    """
    run = active_run()
    if run is None or not metrics:
        return False
    bind(prefixes, step_metric)
    try:
        run.log({**timeline_row(metrics), step_metric: step})
    except Exception as exc:  # noqa: BLE001 - tracking must never stop training
        _warn_once("log metrics", exc)
        return False
    return True


def update_config(values: Mapping[str, Any], section: str = CONFIG_SECTION) -> bool:
    """Merge ``values`` into the run config under ``section``. Returns whether it was written."""
    run = active_run()
    if run is None:
        return False
    try:
        current = dict(run.config.get(section) or {})
        current.update(values)
        run.config.update({section: current}, allow_val_change=True)
    except Exception as exc:  # noqa: BLE001 - tracking must never stop training
        _warn_once("update the config", exc)
        return False
    return True


def span(lane: str, start: float, end: float) -> dict[str, float]:
    """One phase as metrics: ``timeline/<lane>/start`` and ``/end``, in Unix seconds."""
    if lane not in LANES:
        raise ValueError(f"unknown timeline lane {lane!r}; one of {', '.join(LANES)}")
    return {f"timeline/{lane}/start": round(start, 3), f"timeline/{lane}/end": round(end, 3)}


def timeline_keys() -> list[str]:
    """Every timeline key, as a chart queries them."""
    return [f"timeline/{lane}/{edge}" for lane in LANES for edge in ("start", "end")] + [
        TIMELINE_STEP
    ]


def timeline_row(metrics: Mapping[str, float]) -> dict[str, float]:
    """``metrics`` with every timeline key, NaN for the lanes it has no span for.

    A chart reads the timeline through W&B's keyed history, which returns only the rows that
    carry every key it asks for. Each process logs only its own phases, so a row without the
    other lanes' keys would never reach the chart. The charts skip NaN. Metrics without any
    timeline key come back as they are.
    """
    out = dict(metrics)
    if any(k.startswith("timeline/") for k in out):
        for key in timeline_keys():
            out.setdefault(key, math.nan)
    return out


def deployment_config(deployment: Any) -> dict[str, Any]:
    """The parts of a deployment that explain a run's rollout and hot-load numbers.

    ``deployment`` is a :class:`cwtrain.infra.Deployment`. Leaves out the gateway, the buckets and
    any other where-it-runs detail.
    """
    raw = getattr(deployment, "raw", None) or {}
    spec = raw.get("spec") or raw
    runtime = spec.get("runtime") or {}
    resources = spec.get("resources") or {}
    autoscaling = spec.get("autoscaling") or {}
    out: dict[str, Any] = {
        "id": getattr(deployment, "id", ""),
        "name": getattr(deployment, "name", ""),
        "model": getattr(deployment, "model_name", ""),
        "engine": runtime.get("engine", ""),
        "engine_version": runtime.get("version") or runtime.get("engineVersion", ""),
        "engine_config": dict(runtime.get("engineConfig") or {}),
        "instance_type": resources.get("instanceType", ""),
        "gpus_per_replica": resources.get("gpuCount"),
        "replicas_min": autoscaling.get("min"),
        "replicas_max": autoscaling.get("max"),
        "autoscaling_concurrency": autoscaling.get("concurrency"),
    }
    return {k: v for k, v in out.items() if v not in (None, "", {})}


def version() -> str:
    """cwtrain's installed version."""
    import cwtrain

    return str(cwtrain.__version__)


__all__ = [
    "CONFIG_SECTION",
    "LANES",
    "TIMELINE_STEP",
    "active_run",
    "bind",
    "deployment_config",
    "log_metrics",
    "span",
    "timeline_keys",
    "timeline_row",
    "update_config",
    "version",
]
