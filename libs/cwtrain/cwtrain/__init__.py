# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""cwtrain: train against CoreWeave Dedicated Inference.

A training loop whose rollouts come from a Dedicated Inference deployment needs three things, one
subpackage each:

* :mod:`cwtrain.infra` - the deployment and its gateway (create them, wait until they serve,
  delete them, find the endpoint) through the Dedicated Inference API, and the object-storage
  client.
* :mod:`cwtrain.hotload` - publishing the trainer's new weights into the running deployment after
  each step (checkpoint hot-load), as full checkpoints or compressed deltas.
* :mod:`cwtrain.integrations` - the rollout sampler for the deployment's OpenAI-compatible
  endpoint, and the adapters for miles, slime and NeMo-RL.

Top-level names are resolved lazily, so importing a subpackage that only needs torch (for example
the NeMo-RL worker extension) does not pull in the HTTP and object-storage clients.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from typing import Any

# name -> the subpackage that defines it
_EXPORTS = {
    "HotLoader": "cwtrain.hotload",
    "InferenceClient": "cwtrain.infra",
    "inspect_snapshot": "cwtrain.hotload",
    "upload_snapshot": "cwtrain.hotload",
}
__all__ = list(_EXPORTS)

# pyproject.toml is the only place the version is written; read it back from the installed
# distribution so the two cannot disagree.
try:
    __version__ = _distribution_version("cwtrain")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
