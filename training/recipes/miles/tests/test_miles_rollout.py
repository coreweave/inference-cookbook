# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The miles rollout function's wiring, against stand-ins for the miles modules it imports."""

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


class BaseRolloutFn:
    def __init__(self, input):
        pass


class TrainInput:
    def __init__(self, rollout_id: int):
        self.evaluation, self.rollout_id, self.weight_version = False, rollout_id, None


class Sample:
    prompt = "p"
    rollout_routed_experts = None
    reward = None


@pytest.fixture
def rollout(monkeypatch):
    """The recipe's module, with the configs it sampled with."""
    seen = {"configs": [], "layers": 4}

    async def batched_async_rm(args, samples):
        return [1.0] * len(samples)

    modules = {
        "miles": {},
        "miles.rollout": {},
        "miles.utils": {},
        "miles.rollout.base_types": {
            "BaseRolloutFn": BaseRolloutFn,
            "RolloutFnConstructorInput": Output,
            "RolloutFnEvalOutput": Output,
            "RolloutFnInput": TrainInput,
            "RolloutFnTrainInput": TrainInput,
            "RolloutFnTrainOutput": Output,
        },
        "miles.rollout.rm_hub": {"batched_async_rm": batched_async_rm},
        "miles.utils.processing_utils": {"load_tokenizer": lambda path, trust_remote_code: None},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("CWTRAIN_ROLLOUT_BASE_URL", "https://gw.invalid/v1")
    monkeypatch.setenv("CWTRAIN_ROLLOUT_MODEL", "m")
    spec = importlib.util.spec_from_file_location("miles_rollout", RECIPE / "coreweave_rollout.py")
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


class Data:
    def get_samples(self, n):
        return [[Sample()]]


def run(module, replay: bool, **flags):
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
    args = Namespace(**{**defaults, **flags})
    fn = module.CoreWeaveRolloutFn(Output(args=args, data_source=Data()))
    return asyncio.run(fn(TrainInput(rollout_id=3)))


def test_routes_follow_the_replay_flag(rollout) -> None:
    module, seen = rollout
    out = run(module, replay=True)
    assert seen["configs"][-1].routed_experts is True
    assert out.samples[0][0].rollout_routed_experts.shape == (3, 4, 2)
    run(module, replay=False)
    assert seen["configs"][-1].routed_experts is False


def test_routes_with_another_layer_count_than_the_trainer_fail_the_rollout(rollout) -> None:
    module, seen = rollout
    seen["layers"] = 5
    with pytest.raises(RolloutError, match="5 layers and top-2, but the trainer has"):
        run(module, replay=True)


def test_tis_requires_logprobs_and_temperature_one(rollout) -> None:
    module, seen = rollout
    run(module, replay=False, use_tis=True)
    assert seen["configs"][-1].require_logprobs is True
    run(module, replay=False)
    assert seen["configs"][-1].require_logprobs is False
    with pytest.raises(CWTrainError, match="--use-tis compares log-probs"):
        run(module, replay=False, use_tis=True, rollout_temperature=0.7)


def test_each_rollout_is_salted_with_the_run_and_the_rollout(rollout, monkeypatch) -> None:
    module, seen = rollout
    monkeypatch.setenv("CWTRAIN_HOTLOAD_RUN_ID", "grpo-7")
    run(module, replay=False)
    from cwtrain.integrations.nvext import _PROCESS_TOKEN

    assert seen["salts"] == [f"grpo-7-{_PROCESS_TOKEN}-r3"]
