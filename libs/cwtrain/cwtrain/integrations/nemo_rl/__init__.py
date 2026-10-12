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

NeMo-RL needs a small patch to know the backend; see ``recipes/training/nemo-rl/patches``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CoreWeaveDTensorPolicyWorker",
    "CoreWeaveGeneration",
    "CoreWeaveMegatronPolicyWorker",
    "HotLoadWeightSynchronizer",
]


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
