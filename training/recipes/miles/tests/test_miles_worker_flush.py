# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from cwtrain.integrations.launch import apply_patches

SOURCES = {
    "miles/ray/train_actor.py": 'class TrainRayActor:\n    @rpc(concurrency_group="fault_injector")\n    def control_fault_hook(self, command: FaultHookCommand) -> FaultHookRecord:\n        return fault_hook_controller.apply(command)\n\n    @rpc(concurrency_group="kill_self")\n    def kill_self(self) -> None:\n        os._exit(1)\n',
    "miles/ray/train/group.py": "class TrainerController:\n    async def dispose(self) -> None:\n        if (disposer := self._watcher_disposer) is not None:\n            await disposer()\n            self._watcher_disposer = None\n\n    # ------------------------ API :: train ------------------------\n\n    async def train(\n        self,\n        rollout_id: int,\n    ):\n        pass\n",
    "miles/ray/rollout/rollout_executor.py": 'from miles.utils.output_store.plan import StorePlanError\nfrom miles.utils.simple_checkpointer import atomic_save_folder\nfrom miles.utils.timer import timer\nfrom miles.utils.tracking_utils.tracking import init_tracking\nfrom miles.utils.weight_version import assert_samples_weight_version_sane, assert_weight_version_is_published\nfrom miles.utils.workers.rpc.common.metadata import DedicatedRpcLoopWorker\nfrom miles.utils.workers.worker_provider.base import BaseWorkerProvider\n\nclass RolloutExecutor:\n    async def dispose(self) -> None:\n        # the driver may dispose an executor whose init never started; a partial init still tears down what it built\n        if self._disposed or self._init_once.state is InitState.NOT_INITED:\n            return\n        self._disposed = True\n\n        if self.generate_rollout is not None and not self.use_legacy_rollout_v1:\n            await maybe_await(self.generate_rollout.dispose())\n        if (\n            close := getattr(self.data_source, "close", None)\n        ) is not None:  # config-access-exempt: custom data sources may omit this lifecycle method\n            close()\n        event_analyzer.run_sample_ownership_analysis(args=self.args)\n        event_analyzer.run_analysis_from_args(self.args)\n        if self._metric_checker is not None:\n            self._metric_checker.dispose()\n        if isinstance(self.eval_generate_rollout, CheckpointEvalFn):\n            await maybe_await(self.eval_generate_rollout.dispose())\n        # Last, so a failed CI check raises after the rest of this dispose; only once init reached the store.\n        if self._delete_store_refs is not None:\n            await asyncio.to_thread(self._sweep_output_store)\n\n    def _sweep_output_store(self) -> None:\n        """Remove what the run left in the output store, which nothing reads once it ends."""\n        pass\n',
}


@pytest.fixture
def methods(tmp_path):
    for name, source in SOURCES.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    patch = (
        Path(__file__).resolve().parents[1] / "patches/0004-flush-worker-tracking.patch"
    )
    apply_patches(tmp_path, [patch], tested="miles v0.2.0")

    def load(name, **namespace):
        tree = ast.parse((tmp_path / name).read_text())
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "dispose"
        )
        exec(  # noqa: S102 - execute only the patched fixture method
            compile(ast.Module(body=[method], type_ignores=[]), name, "exec"), namespace
        )
        return namespace["dispose"]

    return load


def test_trainer_flushes_healthy_cells_with_deadline(methods):
    live, dead = (
        SimpleNamespace(is_alive=True, execute=AsyncMock()),
        SimpleNamespace(is_alive=False, execute=AsyncMock()),
    )
    stop = AsyncMock()
    owner = SimpleNamespace(_watcher_disposer=stop, _cells=[live, dead])
    asyncio.run(methods("miles/ray/train/group.py", asyncio=asyncio)(owner))
    stop.assert_awaited_once()
    live.execute.assert_awaited_once_with(
        "finish_tracking", kill_on_failure=False, timeout=60
    )
    dead.execute.assert_not_called()
    assert owner._watcher_disposer is None


@pytest.mark.parametrize("failure", [False, True])
def test_rollout_flushes_even_when_cleanup_fails(methods, failure):
    finish = Mock()
    cleanup = Mock(side_effect=RuntimeError("cleanup failed") if failure else None)
    owner = SimpleNamespace(
        _disposed=False,
        _init_once=SimpleNamespace(state="ready"),
        generate_rollout=None,
        data_source=SimpleNamespace(close=cleanup),
        args=None,
        _metric_checker=None,
        eval_generate_rollout=None,
        _delete_store_refs=None,
    )
    dispose = methods(
        "miles/ray/rollout/rollout_executor.py",
        finish_tracking=finish,
        InitState=SimpleNamespace(NOT_INITED="new"),
        CheckpointEvalFn=type("CheckpointEvalFn", (), {}),
        event_analyzer=SimpleNamespace(
            run_sample_ownership_analysis=Mock(), run_analysis_from_args=Mock()
        ),
    )
    if failure:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            asyncio.run(dispose(owner))
    else:
        asyncio.run(dispose(owner))
    finish.assert_called_once_with()
    asyncio.run(dispose(owner))
    finish.assert_called_once_with()
