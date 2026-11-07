# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""A NeMo-RL ``Timer`` that also records when each phase ran, for the W&B timeline.

NeMo-RL's timer keeps only durations. :class:`SpanTimer` also keeps each label's first start and
last end in wall-clock time, so the GRPO step's phases can go on the timeline
(:mod:`cwtrain.integrations.tracking`). The recipe's patch constructs it in ``grpo_train`` for the
``coreweave`` backend, and passes it to :meth:`HotLoadWeightSynchronizer.pop_metrics`.
"""

from __future__ import annotations

import time

from cwtrain.integrations import tracking
from cwtrain.integrations.nemo_rl import _missing_dependency

try:
    from nemo_rl.utils.timer import Timer
except ModuleNotFoundError as exc:
    raise _missing_dependency(exc) from exc

#: NeMo-RL's GRPO timer labels, and the timeline lane each one is.
LANE_OF_LABEL = {
    "generation": "rollout",
    "reward_calculation": "reward",
    "policy_and_reference_logprobs": "logprobs",
    "policy_training": "update",
}


class SpanTimer(Timer):
    """NeMo-RL's ``Timer``, plus each label's wall-clock span since the last :meth:`pop_spans`.

    A label timed more than once (dynamic sampling generates several batches) spans from its first
    start to its last end.
    """

    def __init__(self, context: dict[str, object] | None = None) -> None:
        super().__init__(context=context)
        self._wall_starts: dict[str, float] = {}
        self._spans: dict[str, tuple[float, float]] = {}

    def start(self, label: str, should_log: bool = True) -> None:
        super().start(label, should_log=should_log)
        self._wall_starts[label] = time.time()

    def stop(self, label: str, should_log: bool = True) -> float:
        elapsed = super().stop(label, should_log=should_log)
        started = self._wall_starts.pop(label, None)
        if started is not None:
            first = self._spans.get(label, (started, started))[0]
            self._spans[label] = (min(first, started), time.time())
        return elapsed

    def pop_spans(self) -> dict[str, float]:
        """The GRPO phases since the last call, as ``timeline/*`` metrics, then forget them."""
        spans, self._spans = self._spans, {}
        out: dict[str, float] = {}
        for label, (start, end) in spans.items():
            lane = LANE_OF_LABEL.get(label)
            if lane is not None:
                out.update(tracking.span(lane, start, end))
        return out


__all__ = ["LANE_OF_LABEL", "SpanTimer"]
