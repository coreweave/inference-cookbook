# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""NeMo-RL integration: a generation backend and a weight synchronizer for hot-load deployments.

* :class:`CoreWeaveGeneration` (``policy.generation.backend: coreweave``) samples rollouts from a
  CoreWeave Dedicated Inference deployment over its OpenAI-compatible endpoint. It owns no GPUs.
* :class:`HotLoadWeightSynchronizer` replaces NeMo-RL's IPC/NCCL refit: it exports the policy's
  weights in HF layout, publishes them as a full snapshot or an XOR delta, and hot-loads them.
* :class:`CoreWeaveDTensorPolicyWorker` / :class:`CoreWeaveMegatronPolicyWorker` are policy-worker
  extensions (``policy.worker_extension_cls_fqn``) that add the ``export_hf_checkpoint`` RPC the
  synchronizer calls. They import only torch and safetensors so they load in the worker venv.

NeMo-RL needs a small patch to know the backend; see ``training/recipes/nemo-rl/patches``.

The integration adds no dependencies: install ``cwtrain`` into NeMo-RL's environment, which
provides NeMo-RL, torch and safetensors.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CoreWeaveDTensorPolicyWorker",
    "CoreWeaveGeneration",
    "CoreWeaveMegatronPolicyWorker",
    "HotLoadWeightSynchronizer",
]

#: The NeMo-RL release that the integration and the recipe's patches are tested with.
_TESTED_NEMO_RL = "v0.7.0"


def _missing_dependency(exc: ModuleNotFoundError) -> ModuleNotFoundError:
    """Explain a module that NeMo-RL's environment would have provided."""
    return ModuleNotFoundError(
        f"No module named {exc.name!r}. cwtrain's NeMo-RL integration runs in NeMo-RL's "
        f"environment, which provides it: install cwtrain there (tested with NeMo-RL "
        f"{_TESTED_NEMO_RL}).",
        name=exc.name,
    )


def __getattr__(name: str) -> Any:
    if name == "CoreWeaveGeneration":
        from cwtrain.integrations.nemo_rl.generation import CoreWeaveGeneration

        return CoreWeaveGeneration
    if name == "HotLoadWeightSynchronizer":
        from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

        return HotLoadWeightSynchronizer
    if name in ("CoreWeaveDTensorPolicyWorker", "CoreWeaveMegatronPolicyWorker"):
        from cwtrain.integrations.nemo_rl import worker

        return getattr(worker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
