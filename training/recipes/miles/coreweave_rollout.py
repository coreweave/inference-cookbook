# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""miles rollout function that samples from a CoreWeave Dedicated Inference deployment.

Select it with ``--rollout-function-path coreweave_rollout.CoreWeaveRolloutFn`` and run miles
with ``--rollout-num-gpus 0``: the deployment is the rollout engine, and the ``cwtrain``
post-write hook keeps it on-policy by hot-loading every published weight version. miles never
pauses, reloads or resumes anything itself; the hot-load API orchestrates the swap across replicas.

Configuration is read from ``CWTRAIN_ROLLOUT_*`` environment variables; see
:mod:`cwtrain.integrations.rollout`, which holds everything that is not miles-specific.
"""

from __future__ import annotations

import logging

from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.rm_hub import batched_async_rm
from miles.utils.processing_utils import load_tokenizer

from cwtrain.integrations.rollout import (
    RolloutConfig,
    check_routes_fit,
    rollout_cache_salt,
    sample_groups,
    shared_sampler,
)

log = logging.getLogger(__name__)


class CoreWeaveRolloutFn(BaseRolloutFn):
    """One ``/v1/completions`` request per GRPO group, ``n`` choices per request."""

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        self.args = input.args
        self.data_source = input.data_source
        self.config = RolloutConfig.for_trainer(self.args)
        self.tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=False)

    async def __call__(self, input: RolloutFnInput):
        if input.evaluation:
            return RolloutFnEvalOutput(data={}, metrics={})
        assert isinstance(input, RolloutFnTrainInput)
        args = self.args
        groups = self.data_source.get_samples(args.rollout_batch_size)
        metrics = await sample_groups(
            shared_sampler(self.config),
            groups,
            tokenizer=self.tokenizer,
            max_tokens=int(args.rollout_max_response_len),
            temperature=float(args.rollout_temperature),
            top_p=float(getattr(args, "rollout_top_p", 1.0) or 1.0),
            stop=getattr(args, "rollout_stop", None),
            # one rollout samples one version; keep its prompts out of later versions' cache
            cache_salt=rollout_cache_salt(input.rollout_id),
        )
        if self.config.routed_experts:
            check_routes_fit(groups, int(args.num_layers), int(args.moe_router_topk))
        flat = [s for g in groups for s in g]
        rewards = await batched_async_rm(args, flat)
        for sample, reward in zip(flat, rewards, strict=True):
            sample.reward = float(reward)
        metrics["coreweave/weight_version"] = (
            float(input.weight_version) if input.weight_version is not None else -1.0
        )
        log.info("rollout %d: %s", input.rollout_id, metrics)
        return RolloutFnTrainOutput(samples=groups, metrics=metrics)
