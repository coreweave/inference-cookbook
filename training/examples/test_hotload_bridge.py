# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Exercise the actual notebook worker and laptop driver over an in-memory S3 mailbox."""

import ast
import asyncio
import copy
import importlib.util
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("hotload_laptop", ROOT / "hotload_laptop.py")
laptop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(laptop)


def notebook_tracking():
    tree = ast.parse((ROOT / "hotload_notebook.py").read_text())
    cell = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and any(isinstance(c, ast.ClassDef) and c.name == "NotebookTracker" for c in n.body)
    )
    cell = copy.deepcopy(cell)
    cell.decorator_list = []
    cell.body[-1] = ast.Return(
        value=ast.Tuple(
            elts=[
                ast.Name(id="NotebookTracker", ctx=ast.Load()),
                ast.Name(id="start_notebook_tracking", ctx=ast.Load()),
            ],
            ctx=ast.Load(),
        )
    )
    scope = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[])),
            "hotload_notebook.py",
            "exec",
        ),
        scope,
    )
    return scope["_"](os=os)


NotebookTracker, start_notebook_tracking = notebook_tracking()


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.events = []

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise MissingObject()
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body
        self.events.append(Key)

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise MissingObject()
        return {"ContentLength": len(self.objects[Key])}


def mailbox():
    return laptop.Mailbox(
        MemoryS3(), "test-bucket", "test-run", session="session-a", timeout=2, poll_interval=0.001
    )


def worker(box, *, empty=False, fail_upload_once=False, fail_train=False):
    """Compile the real marimo cell with fake training and S3 upload dependencies."""
    tree = ast.parse((ROOT / "hotload_notebook.py").read_text())
    cell = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and any(isinstance(c, ast.AsyncFunctionDef) and c.name == "process_request" for c in n.body)
    )
    cell = copy.deepcopy(cell)
    cell.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[]))
    scope = {}
    exec(compile(module, "hotload_notebook.py", "exec"), scope)
    state = SimpleNamespace(
        session=box.session,
        name=box.run_id,
        step=0,
        pending=None,
        failed=False,
        last_identity=f"{box.run_id}-v000000",
    )
    calls = SimpleNamespace(train=0, export=0, upload=0, sample=0)

    def train_step(batch):
        calls.train += 1
        if fail_train:
            raise RuntimeError("interrupted after optimizer mutation")
        return {"reward": 0.5}

    def export_delta(step):
        calls.export += 1
        return Path("delta"), SimpleNamespace(is_empty=empty), {"verified": 2}

    def upload_snapshot(info, *, bucket, path_prefix, identity, s3):
        calls.upload += 1
        if fail_upload_once and calls.upload == 1:
            raise OSError("interrupted multipart upload")
        index = {
            "metadata": {
                "version": identity,
                "base_version": state.pending["previous_identity"],
                "delta_encoding": "xor",
                "compression_format": "zstd",
                "checksum_format": "adler32",
            },
            "weight_map": {"weight": "model.safetensors"},
        }
        s3.put_object(
            Bucket=bucket, Key=f"{path_prefix}/{identity}/model.safetensors", Body=b"weights"
        )
        s3.put_object(
            Bucket=bucket,
            Key=f"{path_prefix}/{identity}/model.safetensors.index.json",
            Body=json.dumps(index).encode(),
        )

    def put(relative, session, **fields):
        assert session == box.session
        box.put(relative, **fields)

    async def sample_remote(tasks, *, base_url, model, token):
        assert base_url == "https://gateway.example/v1"
        assert model == box.run_id
        assert token == "wandb-test-key"
        calls.sample += 1
        return [[{"ids": [2], "logprobs": [-0.5], "text": "42"}] * 4] * 8

    args = dict(
        PREFIX=box.prefix,
        cfg={"bucket": box.bucket},
        check_message=lambda m, s, step=None: box.check(m, step),
        export_delta=export_delta,
        inspect_snapshot=lambda *a, **k: object(),
        make_prompts=lambda step: [{"prompt_ids": [1], "answer": 42}] * 8,
        os=SimpleNamespace(environ={"WANDB_API_KEY": "wandb-test-key"}),
        sample_remote=sample_remote,
        put_message=put,
        read_message=box.read,
        run=state,
        s3=box.s3,
        time=time,
        train_step=train_step,
        upload_snapshot=upload_snapshot,
    )
    (process_request,) = scope["_"](**args)
    return lambda: asyncio.run(process_request()), state, calls


def request(box, step=1, previous=None):
    box.put(
        f"requests/{step:06d}.json",
        step=step,
        previous_identity=previous or f"{box.run_id}-v000000",
        rollout_base_url="https://gateway.example/v1",
        model=box.run_id,
    )


class Loader:
    def __init__(self, box, fail_verify=False):
        self.box, self.fail_verify = box, fail_verify
        self.events = []

    def trigger(self, identity, **kwargs):
        self.events.append("trigger")
        assert self.box.read("results/000001.json") is not None
        return SimpleNamespace(id="hot-load-id")

    def wait(self, hot_load_id):
        self.events.append("wait")
        assert self.box.read("acks/000001.json") is None

    def verify_served(self, hot_load_id, identity):
        self.events.append("verify")
        assert self.box.read("acks/000001.json") is None
        if self.fail_verify:
            raise RuntimeError("replica still serves old weights")


def test_training_to_verified_hotload_barrier():
    box = mailbox()
    process, state, calls = worker(box)
    request(box)
    loader = Loader(box)
    with ThreadPoolExecutor() as executor:
        future = executor.submit(process)
        identity = laptop.finish_step(box, loader, 1, state.last_identity)
        assert future.result(timeout=3)["verified"] == 2
    assert loader.events == ["trigger", "wait", "verify"]
    assert state.last_identity == identity == "test-run-v000001"
    assert (calls.train, calls.export, calls.upload) == (1, 1, 1)
    keys = box.s3.events
    assert keys.index(f"{box.prefix}/hotload/{identity}/model.safetensors.index.json") < keys.index(
        f"{box.prefix}/results/000001.json"
    )


def test_retry_failed_upload_does_not_train_twice():
    box = mailbox()
    process, state, calls = worker(box, fail_upload_once=True)
    request(box)
    with pytest.raises(OSError):
        process()
    assert box.read("results/000001.json") is None
    with ThreadPoolExecutor() as executor:
        future = executor.submit(process)
        laptop.finish_step(box, Loader(box), 1, "test-run-v000000")
        future.result(timeout=3)
    assert (calls.sample, calls.train, calls.export, calls.upload) == (1, 1, 1, 2)
    assert state.pending is None


def test_empty_delta_retains_served_identity_without_hotload():
    box = mailbox()
    process, state, calls = worker(box, empty=True)
    request(box)
    loader = Loader(box)
    with ThreadPoolExecutor() as executor:
        future = executor.submit(process)
        assert laptop.finish_step(box, loader, 1, state.last_identity) == "test-run-v000000"
        future.result(timeout=3)
    assert not loader.events
    assert not calls.upload
    assert state.step == 1


def test_no_ack_if_replica_verification_fails():
    box = mailbox()
    process, state, calls = worker(box)
    request(box)
    # An acknowledgment is deliberately absent; stop the worker using the driver-error marker.
    with ThreadPoolExecutor() as executor:
        future = executor.submit(process)
        with pytest.raises(RuntimeError, match="replica still"):
            laptop.finish_step(box, Loader(box, fail_verify=True), 1, state.last_identity)
        assert box.read("acks/000001.json") is None
        box.put("control/error.json", step=0)
        with pytest.raises(RuntimeError, match="Laptop driver failed"):
            future.result(timeout=3)


def test_stale_session_and_wrong_predecessor_rejected_before_training():
    box = mailbox()
    process, state, calls = worker(box)
    request(box, previous="wrong-version")
    with pytest.raises(ValueError, match="different served"):
        process()
    assert calls.train == 0
    request(box)
    message = box.read("requests/000001.json")
    message["session"] = "old-session"
    box.s3.objects[f"{box.prefix}/requests/000001.json"] = json.dumps(message).encode()
    with pytest.raises(ValueError, match="session changed"):
        process()
    assert calls.train == 0


def test_training_failure_prevents_reexecution():
    box = mailbox()
    process, state, calls = worker(box, fail_train=True)
    request(box)
    with pytest.raises(RuntimeError, match="optimizer mutation"):
        process()
    assert box.read("control/trainer-error.json")["error"] == "RuntimeError"
    with pytest.raises(RuntimeError, match="fresh run ID"):
        process()
    assert calls.train == 1


def test_missing_and_wrong_delta_index_rejected():
    box = mailbox()
    result = dict(
        schema=1,
        run_id=box.run_id,
        session=box.session,
        step=1,
        previous_identity="test-run-v000000",
        identity="test-run-v000001",
        empty=False,
    )
    with pytest.raises(ValueError, match="before its snapshot index"):
        laptop.accept_result(box, result, 1, result["previous_identity"])
    key = f"{box.prefix}/hotload/test-run-v000001/model.safetensors.index.json"
    box.s3.objects[key] = json.dumps({"metadata": {"version": "000001"}}).encode()
    with pytest.raises(ValueError, match="unexpected version"):
        laptop.accept_result(box, result, 1, result["previous_identity"])


def test_molab_has_no_platform_or_inference_clients():
    tree = ast.parse((ROOT / "hotload_notebook.py").read_text())
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert not {
        "InferenceClient",
        "HotLoader",
        "subprocess",
        "httpx",
    } & set(imports)
    assert "COREWEAVE_API_TOKEN" not in (ROOT / "hotload_notebook.py").read_text()


@pytest.mark.parametrize("owned_gateway", [False, True])
def test_cleanup_preserves_reused_gateway_and_waits_for_deployment(
    tmp_path, monkeypatch, owned_gateway
):
    from cwtrain.infra import lifecycle

    events = []
    monkeypatch.setattr(
        lifecycle, "delete_deployment", lambda c, ident: events.append(("deployment", ident))
    )
    monkeypatch.setattr(
        lifecycle, "delete_gateway", lambda c, ident: events.append(("gateway", ident))
    )
    state = dict(
        deployment_id="owned-deployment", gateway_id="gateway", created_gateway=owned_gateway
    )
    path = tmp_path / "state.json"
    laptop.cleanup(object(), state, path)
    assert events == [("deployment", "owned-deployment")] + (
        [("gateway", "gateway")] if owned_gateway else []
    )
    saved = json.loads(path.read_text())
    assert saved["deployment_id"] is None
    assert saved["gateway_id"] == (None if owned_gateway else "gateway")


def test_gateway_create_id_is_recoverable_after_request_timeout(tmp_path):
    path = tmp_path / "state.json"
    state = {}

    class Client:
        def list_gateways(self):
            return []

        def create_gateway(self, spec):
            saved = json.loads(path.read_text())
            assert saved["gateway_id"] == spec["id"]
            assert saved["created_gateway"] is True
            assert spec["weightsAndBiasesAuth"] == {}
            assert "coreWeaveAuth" not in spec
            raise TimeoutError("create response was lost")

    args = SimpleNamespace(region="US-WEST-04A", run_id="test-run")
    with pytest.raises(TimeoutError):
        laptop.create_deployment(Client(), mailbox(), args, state, path)
    assert state["gateway_id"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://gateway.example/v1",
        "https://user:pass@gateway.example/v1",
        "https://gateway.example/v1?key=x",
    ],
)
def test_invalid_rollout_endpoint_is_rejected_before_sampling(endpoint):
    box = mailbox()
    process, state, calls = worker(box)
    request(box)
    message = box.read("requests/000001.json")
    message["rollout_base_url"] = endpoint
    box.s3.objects[f"{box.prefix}/requests/000001.json"] = json.dumps(message).encode()
    with pytest.raises(ValueError, match="Invalid rollout"):
        process()
    assert calls.sample == calls.train == 0


class TrackingRun:
    def __init__(self, *, fail_log=False):
        self.fail_log = fail_log
        self.summary = {}
        self.logged = []
        self.definitions = []
        self.exit_codes = []

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def log(self, payload, **kwargs):
        if self.fail_log:
            raise OSError("tracking service unavailable")
        self.logged.append((payload, kwargs))

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)


def tracking_sdk():
    return SimpleNamespace(
        plot=SimpleNamespace(line_series=lambda **kwargs: kwargs),
        Table=lambda **kwargs: kwargs,
    )


def test_wandb_charts_use_verified_steps_and_allowlisted_metrics():
    run = TrackingRun()
    tracker = NotebookTracker(tracking_sdk(), run)
    metrics = dict(
        reward=0.5,
        loss=0.1,
        changed_mb=100,
        wire_mb=10,
        sample_s=2,
        train_s=1,
        hotload_s=3,
        logprob_gap=None,
        grad_norm=float("nan"),
        COREWEAVE_API_TOKEN="do-not-log",
        completion="private-rollout",
    )
    tracker.log_step(1, "test-run-v000001", metrics, empty=False)
    metrics["reward"] = 0.75
    tracker.log_step(2, "test-run-v000002", metrics, empty=False)
    payload, kwargs = run.logged[-1]
    assert kwargs == {"step": 2}
    assert payload["delta/compression_ratio"] == 10
    assert payload["charts/delta_size"]["xs"] == [1, 2]
    assert payload["charts/delta_size"]["ys"] == [[100, 100], [10, 10]]
    assert payload["charts/stage_seconds"]["keys"] == ["sample", "train", "hotload"]
    assert "train/logprob_gap" not in payload and "train/grad_norm" not in payload
    assert "do-not-log" not in repr(payload) and "private-rollout" not in repr(payload)
    assert run.summary == {"completed_steps": 2, "last_served_identity": "test-run-v000002"}
    tracker.finish(0)
    assert len(run.logged[-1][0]["tables/verified_steps"]["data"]) == 2
    assert run.exit_codes == [0]


def test_wandb_failure_cannot_block_served_acknowledgment():
    box = mailbox()
    process, state, calls = worker(box)
    request(box)
    tracker = NotebookTracker(tracking_sdk(), TrackingRun(fail_log=True))
    with ThreadPoolExecutor() as executor:
        future = executor.submit(process)
        identity = laptop.finish_step(box, Loader(box), 1, state.last_identity)
        metrics = future.result(timeout=3)
        assert identity == "test-run-v000001"
        tracker.log_step(1, identity, metrics, empty=False)
    tracker.finish(1)
    assert tracker.run.exit_codes == [1]
    assert box.read("acks/000001.json")["identity"] == "test-run-v000001"


def test_wandb_real_sdk_offline_recording(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("WANDB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
    options = dict(api_key="", project="cwtrain-offline-test", entity="", mode="offline")
    tracker = start_notebook_tracking(options, tmp_path, "offline-test")
    run_dir = Path(tracker.run.dir).parent
    try:
        tracker.log_step(
            1,
            "offline-test-v000001",
            dict(
                reward=0.5, loss=0.1, changed_mb=100, wire_mb=10, sample_s=2, train_s=1, hotload_s=3
            ),
            empty=False,
        )
        assert tracker.run.summary["completed_steps"] == 1
    finally:
        tracker.finish(0)
    recordings = list(run_dir.glob("run-*.wandb"))
    assert recordings and recordings[0].stat().st_size > 0


def test_tutorial_target_counts_acknowledged_steps_across_cell_reruns():
    """Run the editable cell against a pending upload and the next laptop request."""
    source = (ROOT / "hotload_notebook.py").read_text()
    tree = ast.parse(source)
    cell = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and any(arg.arg == "process_request" for arg in node.args.args)
    )
    cell.decorator_list = []
    # Simulate the learner changing target_step in the first line from 0 to 2.
    assert isinstance(cell.body[0], ast.Assign)
    cell.body[0].value = ast.Constant(2)
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module([cell], [])), "tutorial", "exec"), scope)

    class Paused(Exception):
        pass

    def stop(condition, *_):
        if condition:
            raise Paused

    mo = SimpleNamespace(
        stop=stop,
        callout=lambda *args, **kwargs: None,
        md=lambda message: None,
        status=SimpleNamespace(progress_bar=lambda values, **kwargs: values),
    )
    run = SimpleNamespace(bootstrapped=True, step=1, pending={"step": 1}, last_identity="root")
    completed = []
    history = {}

    async def process_request():
        if run.pending:
            run.pending = None
        else:
            run.step += 1
        run.last_identity = f"v{run.step}"
        completed.append(run.step)
        return {"tensors_changed": 1, "reward": 0.5}

    def set_history(update):
        history.update(update(history))

    args = dict(
        mo=mo,
        os=SimpleNamespace(environ={"WANDB_API_KEY": "test-key"}),
        process_request=process_request,
        run=run,
        set_history=set_history,
        wandb_tracker=None,
    )
    asyncio.run(scope["_"](**args))
    assert completed == [1, 2]
    assert list(history) == [1, 2]
    asyncio.run(scope["_"](**args))
    assert completed == [1, 2]
