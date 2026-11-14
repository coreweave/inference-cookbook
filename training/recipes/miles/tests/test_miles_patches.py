# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Exercise patched upstream behavior without importing the GPU runtime."""

import asyncio
import runpy
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
from cwtrain.integrations.launch import apply_patches

PATCHES = Path(__file__).resolve().parents[1] / "patches"


def patched(tmp_path, name, source, patch):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    apply_patches(tmp_path, [PATCHES / patch], tested="miles v0.2.0")
    return runpy.run_path(str(path))


@pytest.mark.parametrize(
    "rollout,eval_gpus,debug,colocate,expected",
    [(0, 0, False, False, False), (None, 0, False, False, False),
     (2, 0, False, False, True), (0, 2, False, False, True),
     (2, 0, True, False, False), (2, 2, True, False, True),
     (0, 0, False, True, True)],
)
def test_external_rollout_skips_local_engines(tmp_path, rollout, eval_gpus, debug, colocate, expected):
    namespace = patched(tmp_path, "miles/utils/arguments.py", '''
def validate(args):
    if args.colocate:
        args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes

    if args.debug_train_only:
        args.rollout_num_gpus = 0
    args.starts_inference_engines = not args.debug_train_only or args.eval_num_gpus > 0

    if args.use_critic and not args.debug_rollout_only:
        if args.offload_train is None:
            args.offload_train = True
''', "0001-zero-local-inference-engines.patch")
    args = SimpleNamespace(rollout_num_gpus=rollout, eval_num_gpus=eval_gpus,
                           debug_train_only=debug, colocate=colocate, use_critic=False,
                           actor_num_gpus_per_node=2, actor_num_nodes=1)
    namespace["validate"](args)
    assert args.starts_inference_engines is expected


@pytest.mark.parametrize("failure,expected", [(False, 0), (True, 1)])
def test_wandb_finish_preserves_exception_status(tmp_path, monkeypatch, failure, expected):
    namespace = patched(tmp_path, "miles/utils/tracking_utils/base.py", '''
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

class TrackingBackend:
    pass

class WandbBackend(TrackingBackend):
    def finish(self) -> None:
        import wandb

        wandb.finish()


class TensorboardBackend(TrackingBackend):
    pass
''', "0002-report-tracking-failure.patch")
    codes = []
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(finish=lambda **kw: codes.append(kw["exit_code"])))

    async def close():
        async with AsyncExitStack() as stack:
            stack.callback(namespace["WandbBackend"]().finish)
            if failure:
                raise RuntimeError("injected training failure")

    if failure:
        with pytest.raises(RuntimeError, match="injected training failure"):
            asyncio.run(close())
    else:
        asyncio.run(close())
    assert codes == [expected]
