# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""slime rollout function that samples from a CoreWeave Dedicated Inference deployment.

Select it with ``--rollout-function-path coreweave_rollout.generate_rollout`` and run slime with
``--rollout-num-gpus 0 --update-weight-standalone-publish``: the deployment is the rollout
engine, and the ``cwtrain`` post-write hook keeps it on-policy by hot-loading every published
weight version. slime starts no SGLang engines and no router; the hot-load API orchestrates the
weight swap across the deployment's replicas.

Configuration is read from ``CWTRAIN_ROLLOUT_*`` environment variables; see
:mod:`cwtrain.integrations.rollout`, which holds everything that is not slime-specific.
"""

from __future__ import annotations

import functools
import logging
from argparse import Namespace
from typing import Any

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.rm_hub import batched_async_rm
from slime.rollout.sample_hooks import apply_rollout_sample_hooks
from slime.utils.async_utils import run
from slime.utils.processing_utils import load_tokenizer

from cwtrain.errors import CWTrainError
from cwtrain.integrations.rollout import (
    NO_EVALUATION,
    RolloutConfig,
    check_routes_fit,
    rollout_cache_salt,
    sample_groups,
    shared_sampler,
)

log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def _tokenizer(hf_checkpoint: str) -> Any:
    """One tokenizer per checkpoint path, bounded for processes reused across runs."""
    return load_tokenizer(hf_checkpoint, trust_remote_code=False)


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput:
    """slime's ``--rollout-function-path`` entry point. It does not evaluate."""
    if evaluation:
        raise CWTrainError(NO_EVALUATION)
    return run(_generate(args, rollout_id, data_source))


async def _generate(args: Namespace, rollout_id: int, data_source: Any) -> RolloutFnTrainOutput:
    tokenizer = _tokenizer(args.hf_checkpoint)
    groups = data_source.get_samples(args.rollout_batch_size)
    config = RolloutConfig.for_trainer(args)
    metrics = await sample_groups(
        shared_sampler(config),
        groups,
        tokenizer=tokenizer,
        max_tokens=int(args.rollout_max_response_len),
        temperature=float(args.rollout_temperature),
        top_p=float(getattr(args, "rollout_top_p", 1.0) or 1.0),
        stop=getattr(args, "rollout_stop", None),
        # one rollout samples one version; keep its prompts out of later versions' cache
        cache_salt=rollout_cache_salt(rollout_id),
    )
    if config.routed_experts:
        check_routes_fit(groups, int(args.num_layers), int(args.moe_router_topk))
    # --rollout-sample-hook-path, such as slime's spill_routed_experts, as slime's own rollout runs
    groups = await apply_rollout_sample_hooks(args, groups)
    flat = [s for g in groups for s in g]
    rewards = await batched_async_rm(args, flat)
    for sample, reward in zip(flat, rewards, strict=True):
        sample.reward = float(reward)
    metrics["coreweave/rollout_id"] = float(rollout_id)
    log.info("rollout %d: %s", rollout_id, metrics)
    return RolloutFnTrainOutput(samples=groups, metrics=metrics)
