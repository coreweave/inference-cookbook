# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The slime rollout function's wiring, against stand-ins for the slime modules it imports."""

import asyncio
import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from cwtrain.errors import CWTrainError
from cwtrain.integrations.rollout import RolloutError

RECIPE = Path(__file__).resolve().parents[1]


class Output:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class Sample:
    prompt = "p"
    rollout_routed_experts = None
    reward = None


@pytest.fixture
def rollout(monkeypatch):
    """The recipe's module, with the hooks it ran and the configs it sampled with."""
    seen = {"hooks": [], "configs": [], "layers": 4}

    async def batched_async_rm(args, samples):
        return [1.0] * len(samples)

    async def apply_rollout_sample_hooks(args, value, **kwargs):
        seen["hooks"].append(value)
        return value

    modules = {
        "slime": {},
        "slime.rollout": {},
        "slime.utils": {},
        "slime.rollout.base_types": {
            "RolloutFnEvalOutput": Output,
            "RolloutFnTrainOutput": Output,
        },
        "slime.rollout.rm_hub": {"batched_async_rm": batched_async_rm},
        "slime.rollout.sample_hooks": {"apply_rollout_sample_hooks": apply_rollout_sample_hooks},
        "slime.utils.async_utils": {"run": asyncio.run},
        "slime.utils.processing_utils": {"load_tokenizer": lambda path, trust_remote_code: None},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("CWTRAIN_ROLLOUT_BASE_URL", "https://gw.invalid/v1")
    monkeypatch.setenv("CWTRAIN_ROLLOUT_MODEL", "m")
    spec = importlib.util.spec_from_file_location("slime_rollout", RECIPE / "coreweave_rollout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def sample_groups(sampler, groups, **kwargs):
        seen["configs"].append(sampler)
        seen.setdefault("salts", []).append(kwargs.get("cache_salt"))
        for group in groups:
            for sample in group:
                if sampler.routed_experts:
                    sample.rollout_routed_experts = np.zeros((3, seen["layers"], 2), np.int32)
        return {}

    monkeypatch.setattr(module, "shared_sampler", lambda config: config)
    monkeypatch.setattr(module, "sample_groups", sample_groups)
    return module, seen


def args(replay: bool, **flags) -> Namespace:
    defaults = {
        "hf_checkpoint": "ckpt",
        "rollout_batch_size": 1,
        "rollout_max_response_len": 8,
        "rollout_temperature": 1.0,
        "num_layers": 4,
        "moe_router_topk": 2,
        "use_rollout_routing_replay": replay,
        "use_tis": False,
    }
    return Namespace(**{**defaults, **flags})


class Data:
    def get_samples(self, n):
        return [[Sample()]]


def test_routes_follow_the_replay_flag_and_the_sample_hooks_run(rollout) -> None:
    module, seen = rollout
    out = module.generate_rollout(args(True), 3, Data())
    assert seen["configs"][-1].routed_experts is True
    assert out.samples[0][0].rollout_routed_experts.shape == (3, 4, 2)
    assert seen["hooks"] == [out.samples]  # e.g. spill_routed_experts, as slime's own rollout
    module.generate_rollout(args(False), 4, Data())
    assert seen["configs"][-1].routed_experts is False


def test_routes_with_another_layer_count_than_the_trainer_fail_the_rollout(rollout) -> None:
    module, seen = rollout
    seen["layers"] = 5
    with pytest.raises(RolloutError, match="5 layers and top-2, but the trainer has"):
        module.generate_rollout(args(True), 3, Data())


def test_tis_requires_logprobs_and_temperature_one(rollout) -> None:
    module, seen = rollout
    module.generate_rollout(args(False, use_tis=True), 3, Data())
    assert seen["configs"][-1].require_logprobs is True
    module.generate_rollout(args(False), 4, Data())
    assert seen["configs"][-1].require_logprobs is False
    with pytest.raises(CWTrainError, match="--use-tis compares log-probs"):
        module.generate_rollout(args(False, use_tis=True, rollout_temperature=0.7), 5, Data())


class Untouched:
    def get_samples(self, n):
        raise AssertionError("the rollout took prompts before it refused the trainer's flags")


def test_a_generate_function_is_refused_before_the_rollout_takes_prompts(rollout) -> None:
    module, _ = rollout
    retool = "examples.retool.generate_with_retool.generate"
    with pytest.raises(CWTrainError, match="--custom-generate-function-path is not supported"):
        module.generate_rollout(args(False, custom_generate_function_path=retool), 3, Untouched())


def test_evaluation_is_refused_rather_than_left_empty(rollout) -> None:
    module, _ = rollout
    with pytest.raises(CWTrainError, match="evaluation is not supported"):
        module.generate_rollout(args(False), 3, Data(), evaluation=True)


def test_each_rollout_is_salted_with_the_run_and_the_rollout(rollout, monkeypatch) -> None:
    module, seen = rollout
    monkeypatch.setenv("CWTRAIN_HOTLOAD_RUN_ID", "grpo-7")
    module.generate_rollout(args(False), 3, Data())
    from cwtrain.integrations.nvext import _PROCESS_TOKEN

    assert seen["salts"] == [f"grpo-7-{_PROCESS_TOKEN}-r3"]
