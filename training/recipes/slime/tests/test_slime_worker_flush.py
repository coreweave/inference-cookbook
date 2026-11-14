# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from cwtrain.integrations.launch import apply_patches

SOURCES = {
    "slime/ray/train_actor.py": 'class Owner:\n        except Exception as e:\n            logger.info(f"Warning: Failed to set NUMA affinity: {e}")\n\n    def clear_memory(self):\n        if self.args.debug_rollout_only:\n            return\n        print_memory("before TrainRayActor.clear_memory")\n',
    "slime/ray/serving.py": 'class Owner:\n    def register_trainers(self, role, actors):\n        # Keep cleanup handles outside the manager, which may itself crash.\n        self.training_actors[role] = actors\n\n    def release_trainers(self):\n        with Cleanup() as cleanup:\n            for actors in self.training_actors.values():\n                for actor in actors[:]:\n                    # Retain failed handles so a repeated detach can retry them.\n                    def release(actor=actor, actors=actors):\n                        ray.kill(actor, no_restart=True)\n                        actors.remove(actor)\n\n                    cleanup.run("terminate training actor", release)\n\n    def detach_training(self, job_id, timeout=None):\n        pass\n',
    "slime/ray/actor_group.py": "class Owner:\n    def offload(self):\n        return ray.get([actor.sleep.remote() for actor in self._actor_handlers])\n\n    def release(self):\n        actors, self._actor_handlers = self._actor_handlers, []\n        for actor in actors:\n            ray.kill(actor, no_restart=True)\n        if actors:\n            time.sleep(5)\n\n    def create(self, rollout_manager=None):\n        pass\n",
    "slime/utils/cleanup.py": '"""Cleanup policy shared by resource owners and the training driver."""\n\nimport logging\nimport time\n\nlogger = logging.getLogger(__name__)\n\n\nclass Cleanup:\n    """Try every cleanup step, keeping the original failure and one deadline.\n\n    Owners must pass ``remaining`` to blocking operations. Arbitrary custom\n    hooks cannot be interrupted safely in a thread; the driver\'s bounded RPC\n    and actor termination provide the outer limit for those hooks.\n    """\n\n    def __init__(self, timeout=60):\n        self.deadline = time.monotonic() + timeout\n        self.errors = []\n\n    @property\n    def remaining(self):\n        return max(0, self.deadline - time.monotonic())\n\n    def run(self, description, function, *args, **kwargs):\n        try:\n            return function(*args, **kwargs)\n        except Exception as error:\n            self.errors.append(error)\n            logger.exception("Failed to %s", description)\n\n    def __enter__(self):\n        return self\n\n    def __exit__(self, exc_type, exc, traceback):\n        # During error unwinding, report cleanup failures without replacing\n        # the training/startup error. Otherwise cleanup failure is observable.\n        if exc is None and self.errors:\n            raise self.errors[0]\n',
}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    for name, source in SOURCES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    patch = (
        Path(__file__).resolve().parents[1] / "patches/0005-flush-worker-tracking.patch"
    )
    apply_patches(tmp_path, [patch], tested="slime v0.4.0")
    ray = SimpleNamespace(
        get=Mock(),
        kill=Mock(),
        exceptions=SimpleNamespace(
            RayActorError=type("RayActorError", (Exception,), {})
        ),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    cleanup = ModuleType("slime.utils.cleanup")
    cleanup.__dict__.update(runpy.run_path(str(tmp_path / "slime/utils/cleanup.py")))
    monkeypatch.setitem(sys.modules, "slime.utils.cleanup", cleanup)
    tracking = ModuleType("slime.ray.tracking")
    tracking.__dict__.update(runpy.run_path(str(tmp_path / "slime/ray/tracking.py")))
    monkeypatch.setitem(sys.modules, "slime.ray.tracking", tracking)
    return tmp_path, ray, cleanup.Cleanup, tracking.flush_training_actors


def actors():
    return [
        SimpleNamespace(finish_tracking=SimpleNamespace(remote=Mock(return_value=i)))
        for i in range(2)
    ]


def test_flush_starts_all_writers_and_waits_with_one_deadline(runtime, caplog):
    caplog.set_level("INFO")
    _, ray, _, flush = runtime
    workers = actors()

    def wait(ref, timeout):
        assert all(worker.finish_tracking.remote.called for worker in workers)
        assert 0 < timeout <= 10

    ray.get.side_effect = wait
    flush(workers, timeout=10)
    assert ray.get.call_count == 2
    assert "Flushed training metrics from 2/2 workers" in caplog.text


def test_dead_worker_does_not_skip_peer_flush(runtime):
    _, ray, _, flush = runtime
    ray.get.side_effect = [ray.exceptions.RayActorError("dead"), None]
    flush(actors())
    assert ray.get.call_count == 2


@pytest.mark.parametrize(
    "path,method",
    [
        ("slime/ray/serving.py", "release_trainers"),
        ("slime/ray/actor_group.py", "release"),
    ],
)
def test_flush_failure_still_releases_all_workers(runtime, path, method):
    root, ray, cleanup, _ = runtime
    ray.get.side_effect = TimeoutError("flush timeout")
    node = next(
        n
        for n in ast.walk(ast.parse((root / path).read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == method
    )
    namespace = {"ray": ray, "Cleanup": cleanup, "time": SimpleNamespace(sleep=Mock())}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)  # noqa: S102 - patched fixture only
    workers = actors()
    owner = SimpleNamespace(
        training_actors={"actor": workers[:]}, _actor_handlers=workers[:]
    )
    with pytest.raises(TimeoutError, match="flush timeout"):
        namespace[method](owner)
    assert ray.get.call_count == 2
    assert ray.kill.call_count == 2
