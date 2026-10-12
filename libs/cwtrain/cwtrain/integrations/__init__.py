# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Framework integrations.

* :mod:`cwtrain.integrations.slime`: the slime/miles post-write hook that hot-loads every weight
  version.
* :mod:`cwtrain.integrations.rollout`: the shared rollout sampler for OpenAI-compatible endpoints.
* :mod:`cwtrain.integrations.nemo_rl`: the NeMo-RL generation backend, weight synchronizer and
  policy-worker extension.

Names are resolved lazily so that importing one integration does not import the others' deps.
"""

from __future__ import annotations

from typing import Any

__all__ = ["HookConfig", "hook"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from cwtrain.integrations import slime

        return getattr(slime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
