# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""CPU source checks: NEMO_RL_SOURCE_DIR=<v0.8.0 checkout> pytest <this file>."""

import ast
import importlib.util
import os
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = os.environ.get("NEMO_RL_SOURCE_DIR")
pytestmark = pytest.mark.skipif(not SOURCE, reason="needs the upstream v0.8.0 source checkout")
RECIPE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def patched(tmp_path_factory):
    source = Path(SOURCE)
    root = tmp_path_factory.mktemp("nemo-rl-v080")
    patches = sorted((RECIPE / "patches").glob("*.patch"))
    paths = {
        line[6:]
        for patch in patches
        for line in patch.read_text().splitlines()
        if line.startswith("--- a/")
    }
    # Always use the pristine tag, even when the source checkout has local patch experiments.
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(["git", "show", f"v0.8.0:{name}"], cwd=source))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for patch in patches:
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
        subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=root, check=True)
    for name in paths:
        compile((root / name).read_text(), name, "exec")
    return root


def execute(nodes, scope):
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<upstream v0.8.0>", "exec"), scope)


def function(root, path, name):
    return next(
        node
        for node in ast.parse((root / path).read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def test_policy_allocation_has_no_local_inference_cluster(patched):
    setup = function(patched, "nemo_rl/algorithms/grpo.py", "setup")
    branch = next(
        node
        for node in ast.walk(setup)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "generation_config['backend'] == 'coreweave'"
    )
    cluster = Mock(return_value="policy-cluster")
    scope = dict(
        cluster_config={"gpus_per_node": 8},
        rm_gpus_per_node=0,
        policy_nodes=1,
        segment_size=None,
        prepare_segment_topology=lambda *_: (None, [], {}),
        RayVirtualCluster=cluster,
    )
    execute(branch.body, scope)
    assert scope["train_cluster"] == "policy-cluster"
    assert scope["inference_cluster"] is None
    assert scope["inference_gpus_per_node"] == scope["inference_nodes"] == 0
    assert cluster.call_args.kwargs["bundle_ct_per_node_list"] == [8]


def test_backend_reuses_automodel_worker_and_existing_endpoint(patched, monkeypatch):
    setup = function(patched, "nemo_rl/algorithms/grpo.py", "setup")
    branch = next(
        node
        for node in ast.walk(setup)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "backend == 'coreweave'"
    )
    cw = types.ModuleType("cwtrain.integrations.nemo_rl")
    cw.CoreWeaveGeneration = Mock(
        return_value=SimpleNamespace(
            rollout_config=SimpleNamespace(base_url="https://policy.invalid/v1", model="policy")
        )
    )
    cw.HotLoadWeightSynchronizer = Mock(return_value="hotload")
    registry = types.ModuleType("nemo_rl.distributed.ray_actor_environment_registry")
    stock = "nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker"
    registry.ACTOR_ENVIRONMENT_REGISTRY = {stock: "automodel-python"}
    monkeypatch.setitem(sys.modules, cw.__name__, cw)
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    policy = Mock(return_value="policy-workers")
    gym = Mock(return_value=("gym-actor", 1.0))
    scope = dict(
        policy_config={"model_name": "base"},
        generation_config={},
        train_cluster="policy-cluster",
        tokenizer=None,
        processor=None,
        weights_path=None,
        optimizer_path=None,
        init_reference_model=False,
        _make_policy=policy,
        _spinup_nemo_gym=gym,
        enable_nemo_gym=True,
        setup_timing_metrics=SimpleNamespace(extras={}),
        Path=Path,
        os=os,
        time=SimpleNamespace(perf_counter=lambda: 1.0),
    )
    execute(branch.body, scope)
    worker = policy.call_args.kwargs["worker_extension_cls_fqn"]
    assert registry.ACTOR_ENVIRONMENT_REGISTRY[worker] == "automodel-python"
    assert scope["policy_generation"].weight_synchronizer == "hotload"
    gym.assert_called_once_with(["https://policy.invalid/v1"], "policy")


def test_upstream_refit_syncs_params_then_delegates_to_hotload(patched):
    fn = function(patched, "nemo_rl/algorithms/grpo.py", "refit_policy_generation")
    policy = Mock()
    sync = Mock()
    sync.sync_weights.return_value = {"uploaded": 1}
    scope = dict(SGLangGeneration=type("SGLangGeneration", (), {}), nullcontext=nullcontext)
    execute([fn], scope)
    result = scope[fn.name](policy, SimpleNamespace(weight_synchronizer=sync), False)
    policy.sync_params_before_refit.assert_called_once_with()
    sync.sync_weights.assert_called_once_with(timer=None, kv_scales=None)
    assert result == {"uploaded": 1}


def test_real_automodel_export_preserves_checkpoint_dtypes():
    torch = pytest.importorskip("torch")
    from cwtrain.integrations.nemo_rl.worker import _check_keeps_tensors
    from torch.distributed.tensor import DTensor

    path = Path(SOURCE) / "nemo_rl/models/policy/workers/automodel_policy_worker.py"
    names = {
        "_refit_tensor_dtype",
        "automodel_params_generator",
        "_maybe_merge_lora_weight",
        "_maybe_adapt_tensor_to_hf",
    }
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    scope = dict(torch=torch, DTensor=DTensor, LinearLoRA=type("LinearLoRA", (), {}))
    execute(nodes, scope)
    generator = scope["automodel_params_generator"]
    _check_keeps_tensors(generator)
    model = torch.nn.Linear(2, 2)
    assert all(t.dtype == torch.float32 for _, t in generator(model, None))
    assert all(t.dtype == torch.bfloat16 for _, t in generator(model, torch.bfloat16))


def test_recipe_passes_v080_config_migration_checks():
    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location(
        "v080_config_checks", Path(SOURCE) / "nemo_rl/utils/outdated_config_checks.py"
    )
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    config = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(Path(SOURCE) / "examples/configs/grpo_math_1B.yaml"),
            OmegaConf.load(RECIPE / "grpo_coreweave.yaml"),
        ),
        resolve=False,
    )
    checks.check_outdated_config(config)
    assert config["policy"]["automodel_cfg"]["enabled"]


@pytest.mark.parametrize("coreweave", [True, False])
def test_gym_actor_runtime_environment_keeps_secrets_out_of_ray(patched, coreweave):
    env = {
        "WANDB_API_KEY": "secret",
        "COREWEAVE_API_TOKEN": "secret",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "PYTHONPATH": "/work/py",
    }
    scope = dict(
        NEMO_GYM_ACTOR_FQN="Gym",
        make_actor_runtime_env=lambda _: {"env_vars": dict(env), "py_executable": "python"},
    )
    fn = function(patched, "nemo_rl/environments/nemo_gym.py", "_gym_actor_runtime_env")
    execute([fn], scope)
    result = scope[fn.name]({"coreweave_training": coreweave})
    assert result["env_vars"] == ({"PYTHONPATH": "/work/py"} if coreweave else env)


def test_span_timer_accepts_actual_v080_timer_options(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "nemo_rl.utils.timer", Path(SOURCE) / "nemo_rl/utils/timer.py"
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    monkeypatch.setitem(sys.modules, "nemo_rl.utils.timer", native)
    spec = importlib.util.spec_from_file_location(
        "v080_span_timer",
        RECIPE.parents[2] / "libs/cwtrain/cwtrain/integrations/nemo_rl/timeline.py",
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    timer = adapter.SpanTimer(context={"worker": "driver"})
    timer.start("generation", should_log=False)
    timer.stop("generation", should_log=False)
    assert timer.pop_spans().keys() == {"timeline/rollout/start", "timeline/rollout/end"}
