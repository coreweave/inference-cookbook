# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Tests for the NeMo-RL integration against stand-ins for NeMo-RL's classes.

NeMo-RL itself is not a test dependency: the modules the integration imports are replaced by small
fakes (an ABC with the same abstract methods, a dict-based BatchedDataDict) before import.
"""

from __future__ import annotations

import abc
import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from cwtrain.errors import CWTrainError
from cwtrain.hotload import HotLoadState
from cwtrain.hotload.delta import checkpoint_checksums
from cwtrain.hotload.mirror import tensor_locations
from tests.conftest import DEPLOYMENT_ID, make_weights, perturb, write_real_checkpoint

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module", autouse=True)
def fake_nemo_rl():
    """Install just enough of nemo_rl for cwtrain.integrations.nemo_rl to import."""

    class GenerationInterface(abc.ABC):
        @abc.abstractmethod
        def init_collective(self, *a, **k): ...

        @abc.abstractmethod
        def generate(self, data, greedy): ...

        @abc.abstractmethod
        def prepare_for_generation(self, *a, **k): ...

        @abc.abstractmethod
        def finish_generation(self, *a, **k): ...

        @abc.abstractmethod
        def shutdown(self): ...

    class BatchedDataDict(dict):
        pass

    class WeightSynchronizer(abc.ABC):
        @abc.abstractmethod
        def sync_weights(self, *, timer=None, kv_scales=None): ...

        @property
        @abc.abstractmethod
        def is_stale(self): ...

        @abc.abstractmethod
        def mark_stale(self): ...

        @abc.abstractmethod
        def init_communicator(self): ...

        @abc.abstractmethod
        def shutdown(self): ...

    names = [
        "nemo_rl",
        "nemo_rl.distributed",
        "nemo_rl.distributed.batched_data_dict",
        "nemo_rl.models",
        "nemo_rl.models.generation",
        "nemo_rl.models.generation.interfaces",
        "nemo_rl.weight_sync",
        "nemo_rl.weight_sync.interfaces",
    ]
    modules = {n: types.ModuleType(n) for n in names}
    modules["nemo_rl.distributed.batched_data_dict"].BatchedDataDict = BatchedDataDict
    modules["nemo_rl.models.generation.interfaces"].GenerationInterface = GenerationInterface
    modules["nemo_rl.weight_sync.interfaces"].WeightSynchronizer = WeightSynchronizer
    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    for name in list(sys.modules):
        if name.startswith("cwtrain.integrations.nemo_rl"):
            del sys.modules[name]
    yield modules
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def convert_ids_to_tokens(self, token_id):
        return {2: "<eos>", 999: "<stop>"}.get(token_id, chr(token_id))


def use_transport(monkeypatch, respond):
    """Route every CompletionSampler at a mock server; returns the recorded request bodies."""
    from cwtrain.integrations.rollout import CompletionSampler

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return respond(body)

    mock = httpx.MockTransport(handler)
    original = CompletionSampler.__init__
    monkeypatch.setattr(
        CompletionSampler,
        "__init__",
        lambda self, config, *, transport=None: original(self, config, transport=mock),
    )
    return seen


def gen_config(**cw):
    return {
        "backend": "coreweave",
        "max_new_tokens": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": [2],
        "stop_strings": None,
        "_pad_token_id": 0,
        "coreweave_cfg": {"base_url": "https://gw/v1", "model": "m", **cw},
    }


def test_generate_groups_prompts_and_returns_vllm_shaped_tensors(monkeypatch) -> None:
    from cwtrain.integrations.nemo_rl.generation import CoreWeaveGeneration

    def respond(body):
        n = body["n"]
        if body["prompt"] == [5, 6, 7]:  # exact ids incl. EOS, all log-probs aligned
            choices = [
                {
                    "index": i,
                    "text": "ab",
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": ["token_id:97", "token_id:98", "token_id:2"],
                        "token_logprobs": [-0.1, -0.2, -0.3],
                    },
                }
                for i in range(n)
            ]
        else:  # plain token strings -> re-tokenized text, truncated on length
            choices = [
                {
                    "index": i,
                    "text": "xyz",
                    "finish_reason": "length",
                    "logprobs": {"tokens": ["x", "y", "z"], "token_logprobs": [-1.0, -1.0, -1.0]},
                }
                for i in range(n)
            ]
        return httpx.Response(200, json={"choices": choices})

    seen = use_transport(monkeypatch, respond)
    gen = CoreWeaveGeneration(gen_config(max_model_len=16), Tokenizer())
    data = {
        "input_ids": torch.tensor([[5, 6, 7, 0], [5, 6, 7, 0], [9, 0, 0, 0]]),
        "input_lengths": torch.tensor([3, 3, 1]),
    }
    out = gen.generate(data, greedy=False)

    assert len(seen) == 2  # two distinct prompts; the repeated one asked n=2 as token ids
    by_prompt = {tuple(b["prompt"]): b for b in seen}
    assert by_prompt[(5, 6, 7)]["n"] == 2 and by_prompt[(9,)]["n"] == 1
    assert by_prompt[(5, 6, 7)]["return_tokens_as_token_ids"] is True
    assert by_prompt[(5, 6, 7)]["max_tokens"] == 8 and by_prompt[(9,)]["max_tokens"] == 8
    assert out["output_ids"].shape == (3, 7)  # padded prompt (4) + longest response (3)
    assert out["output_ids"][0].tolist() == [5, 6, 7, 97, 98, 2, 0]
    assert out["output_ids"][2].tolist() == [9, 120, 121, 122, 0, 0, 0]
    assert out["generation_lengths"].tolist() == [3, 3, 3]
    assert out["unpadded_sequence_lengths"].tolist() == [6, 6, 4]
    assert out["truncated"].tolist() == [False, False, True]
    assert out["logprobs"][0].tolist()[:6] == pytest.approx([0, 0, 0, -0.1, -0.2, -0.3])
    assert out["logprobs"][2].tolist()[1:4] == pytest.approx([-1.0, -1.0, -1.0])
    assert gen.misaligned_logprobs == 0


def test_too_many_configured_stops_fail_at_setup() -> None:
    from cwtrain.integrations.nemo_rl.generation import CoreWeaveGeneration
    from cwtrain.integrations.rollout import RolloutError

    cfg = {**gen_config(), "stop_token_ids": [1, 2, 3, 4, 5]}
    with pytest.raises(RolloutError, match="at most 4 stop token ids"):
        CoreWeaveGeneration(cfg, Tokenizer())


def test_a_4xx_on_text_too_keeps_token_id_prompts(monkeypatch) -> None:
    from cwtrain.integrations.nemo_rl.generation import CoreWeaveGeneration
    from cwtrain.integrations.rollout import RolloutError

    seen = use_transport(monkeypatch, lambda body: httpx.Response(400, text="bad max_tokens"))
    gen = CoreWeaveGeneration(gen_config(), Tokenizer())
    data = {"input_ids": torch.tensor([[65, 66]]), "input_lengths": torch.tensor([2])}
    with pytest.raises(RolloutError, match="bad max_tokens"):
        gen.generate(data)
    assert [type(b["prompt"]) for b in seen] == [list, str] and gen.token_id_prompts is True


def test_generate_greedy_and_text_fallback(monkeypatch) -> None:
    from cwtrain.integrations.nemo_rl.generation import CoreWeaveGeneration

    def respond(body):
        if isinstance(body["prompt"], list):
            return httpx.Response(400, text="prompt must be a string")
        choice = {
            "index": 0,
            "text": "ok",
            "finish_reason": "stop",
            "logprobs": {"tokens": ["o", "k", "<eos>"], "token_logprobs": [-0.5, -0.5, -0.5]},
        }
        return httpx.Response(200, json={"choices": [choice]})

    seen = use_transport(monkeypatch, respond)
    gen = CoreWeaveGeneration(gen_config(), Tokenizer())
    data = {"input_ids": torch.tensor([[65, 66]]), "input_lengths": torch.tensor([2])}
    out = gen.generate(data, greedy=True)
    assert seen[0]["prompt"] == [65, 66] and seen[1]["prompt"] == "AB"  # decoded fallback
    assert seen[1]["temperature"] == 0.0
    assert gen.token_id_prompts is False
    # re-tokenized "ok" plus the EOS the engine emitted; three log-probs align with three ids
    assert out["output_ids"][0].tolist() == [65, 66, 111, 107, 2]
    assert out["logprobs"][0].tolist()[2:] == pytest.approx([-0.5, -0.5, -0.5])


class FakePolicy:
    """Stands in for nemo_rl's Policy: 'exports' by copying a prepared checkpoint into place."""

    def __init__(self, exports: list[Path]) -> None:
        self.exports = exports
        self.calls: list[dict] = []

    def run_all_workers_single_data(self, method_name, **kwargs):
        assert method_name == "export_hf_checkpoint"
        self.calls.append(kwargs)
        shutil.copytree(self.exports.pop(0), kwargs["path"])
        return [{"rank": 0, "tensors": 3}, {"rank": 1, "tensors": 3}]


class StubLoader:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str | None]] = []
        self.manifests: list[str] = []
        self.stored: dict[str, dict] = {}
        self.verified: list[str] = []
        self.rollbacks: list[str] = []
        self.current = "initial"

    def upload(self, info, identity, *, workers=8):
        prev = info.incremental.previous_snapshot_identity if info.incremental else None
        self.uploads.append((info.type.name, identity, prev))
        return info

    def wait_for_deployment_ready(self, timeout=None):
        return None

    def trigger(self, identity, info=None, *, prompt_cache_policy=None, **_):
        return SimpleNamespace(id=f"hl-{identity}", identity=identity)

    def wait(self, hot_load_id, *, timeout=None, on_progress=None):
        if on_progress:
            on_progress(SimpleNamespace(ready=2, total=2))
        return SimpleNamespace(id=hot_load_id, completed=True, state=HotLoadState.COMPLETED)

    def verify_served(self, hot_load_id, identity):
        self.verified.append(identity)
        return [1, 2]

    def upload_manifest(self, identity, manifest):
        self.manifests.append(identity)
        self.stored[identity] = manifest
        return f"s3://b/_manifests/{identity}.json"

    def read_manifest(self, identity):
        return self.stored.get(identity)

    completed: set[str] = set()

    def completed_hot_load(self, identity):
        return SimpleNamespace(id=f"hl-{identity}") if identity in self.completed else None

    def current_identity(self):
        return self.current

    def rollback(self, identity, **_):
        self.rollbacks.append(identity)
        self.current = identity
        return SimpleNamespace(id=f"hl-{identity}", identity=identity)


@pytest.mark.parametrize("value,expected", [(None, 20), ("", 20), (5, 5)])
def test_full_every_from_the_config_defaults_to_20(value, expected) -> None:
    from cwtrain.integrations.nemo_rl.weight_sync import hook_config_from

    # an unset ${oc.env:CWTRAIN_HOTLOAD_FULL_EVERY} can resolve to "", which means the default
    cfg = hook_config_from({"deployment_id": DEPLOYMENT_ID, "full_every": value}, environ={})
    assert cfg.full_every == expected and cfg.mirror_enabled and not cfg.verify


def test_weight_synchronizer_publishes_full_then_deltas_then_cuts(
    tmp_path: Path, monkeypatch
) -> None:
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    w0 = make_weights()
    base = write_real_checkpoint(tmp_path / "base", w0)
    w1 = perturb(w0, ("model.layers.0.weight",), seed=1)
    w2 = perturb(w1, ("lm_head.bias",), seed=2)
    exports = [write_real_checkpoint(tmp_path / f"e{i}", w) for i, w in enumerate((w0, w1, w1, w2))]
    policy = FakePolicy(list(exports))
    loader = StubLoader()
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {
            "verify": True,
            "run_id": "nemo-test",
            "full_every": 3,
            "state_dir": str(tmp_path / "state"),
        },
    }
    sync = HotLoadWeightSynchronizer(
        policy, SimpleNamespace(), cfg, hf_checkpoint=str(base), loader=loader
    )
    assert sync.is_stale and sync.config.identity(0) == "nemo-test-v000000"
    assert sync.served_identity is None
    served = tmp_path / "state" / "served_identity"  # where the Gym model server reads it
    assert not served.exists()

    m0 = sync.sync_weights()  # v0: full snapshot of the export
    assert not sync.is_stale and m0["hotload/incremental"] == 0.0
    assert m0["hotload/replicas_verified"] == 2.0
    m1 = sync.sync_weights()  # v1: delta against v0
    assert m1["hotload/incremental"] == 1.0 and m1["hotload/tensors_changed"] == 1
    m2 = sync.sync_weights()  # v2: identical export, nothing published
    assert m2["hotload/skipped_empty"] == 1.0
    assert sync.served_identity == "nemo-test-v000001"  # the deployment still serves v1
    assert served.read_text().strip() == "nemo-test-v000001"
    m3 = sync.sync_weights()  # v3: full_every=3, chain cut with a full snapshot
    assert m3["hotload/incremental"] == 0.0

    assert loader.uploads == [
        ("FULL", "nemo-test-v000000", None),
        ("INCREMENTAL", "nemo-test-v000001", "nemo-test-v000000"),
        ("FULL", "nemo-test-v000003", None),
    ]
    assert loader.verified == ["nemo-test-v000000", "nemo-test-v000001", "nemo-test-v000003"]
    # manifests only for versions the deployment accepted; the empty v2 was never hot-loaded
    assert loader.manifests == ["nemo-test-v000000", "nemo-test-v000001", "nemo-test-v000003"]
    assert policy.calls[0]["allowed_names"] == sorted(w0)  # exactly the deployment's tensors
    assert policy.calls[0]["config_dir"] == str(base)
    remaining = sorted(p.name for p in (tmp_path / "state" / "exports").iterdir())
    assert remaining == ["v000003"]  # only the latest published export stays as delta base
    manifest = json.loads((tmp_path / "state" / "manifests" / "nemo-test-v000003.json").read_text())
    assert manifest["tensors"] == checkpoint_checksums(exports[3])
    sync.mark_stale()
    assert sync.is_stale


def test_weight_synchronizer_warns_once_when_the_chain_is_never_cut(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    import logging

    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    w0 = make_weights()
    base = write_real_checkpoint(tmp_path / "base", w0)
    w1 = perturb(w0, ("lm_head.bias",), seed=1)
    exports = [write_real_checkpoint(tmp_path / f"e{i}", w) for i, w in enumerate((w0, w1))]
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {"run_id": "nemo-test", "full_every": 0, "state_dir": str(tmp_path / "s")},
    }
    with caplog.at_level(logging.WARNING, logger="cwtrain.integrations.slime"):
        sync = HotLoadWeightSynchronizer(
            FakePolicy(exports),
            SimpleNamespace(),
            cfg,
            hf_checkpoint=str(base),
            loader=StubLoader(),
        )
        assert caplog.text.count("never cuts the delta chain") == 1  # when the run starts
        sync.sync_weights()  # v0, full
        assert sync.sync_weights()["hotload/incremental"] == 1.0  # v1, a delta
    assert caplog.text.count("never cuts the delta chain") == 1  # and not again on a refit


def test_weight_synchronizer_rejects_a_different_tensor_set(tmp_path: Path, monkeypatch) -> None:
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    w0 = make_weights()
    base = write_real_checkpoint(tmp_path / "base", w0)
    smaller = dict(w0)
    del smaller["lm_head.bias"]
    policy = FakePolicy([write_real_checkpoint(tmp_path / "e0", smaller)])
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {"run_id": "nemo-test", "state_dir": str(tmp_path / "state")},
    }
    sync = HotLoadWeightSynchronizer(
        policy, SimpleNamespace(), cfg, hf_checkpoint=str(base), loader=StubLoader()
    )
    with pytest.raises(CWTrainError, match="differs from the deployment's base checkpoint"):
        sync.sync_weights()


@pytest.mark.parametrize(
    ("settings", "replicas"),
    [
        ({}, "replicas ready 2/2"),  # waiting is the check: COMPLETED means every replica serves it
        ({"verify": True}, "replicas ready 2/2, verified 2"),
        ({"wait": False}, "replicas not waited for"),
    ],
)
def test_weight_synchronizer_prints_what_it_checked(
    tmp_path: Path, monkeypatch, capsys, settings, replicas
) -> None:
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    for name in ("CWTRAIN_HOTLOAD_VERIFY", "CWTRAIN_HOTLOAD_WAIT"):
        monkeypatch.delenv(name, raising=False)
    w0 = make_weights()
    base = write_real_checkpoint(tmp_path / "base", w0)
    loader = StubLoader()
    pending = SimpleNamespace(completed=False, state=HotLoadState.PENDING)
    loader.trigger = lambda identity, info=None, **_: SimpleNamespace(
        id=f"hl-{identity}", identity=identity, **vars(pending)
    )
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {"run_id": "nemo-test", "state_dir": str(tmp_path / "state"), **settings},
    }
    sync = HotLoadWeightSynchronizer(
        FakePolicy([write_real_checkpoint(tmp_path / "e0", w0)]),
        SimpleNamespace(),
        cfg,
        hf_checkpoint=str(base),
        loader=loader,
    )
    sync.sync_weights()
    (line,) = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[cwtrain] hot-load nemo-test-v000000")
    ]
    assert line.endswith(f"s, {replicas}")


def test_worker_export_writes_a_loadable_checkpoint(tmp_path: Path) -> None:
    from cwtrain.integrations.nemo_rl.worker import _export

    params = [
        ("model.layers.0.weight", torch.arange(8, dtype=torch.float32).reshape(2, 4)),
        ("model.layers.1.weight", torch.ones(3, dtype=torch.bfloat16)),
        ("lm_head.weight", torch.zeros(2)),  # tied weight the deployment's checkpoint lacks
    ]
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "config.json").write_text("{}")
    (tmp_path / "cfg" / "model.safetensors").write_bytes(b"not copied")
    result = _export(
        iter(params),
        str(tmp_path / "export"),
        ["model.layers.0.weight", "model.layers.1.weight"],
        str(tmp_path / "cfg"),
        shard_bytes=16,
    )
    assert result["tensors"] == 2 and result["shards"] == 2
    locs = tensor_locations(tmp_path / "export")
    assert set(locs) == {"model.layers.0.weight", "model.layers.1.weight"}
    assert locs["model.layers.0.weight"].dtype == "F32"
    assert locs["model.layers.1.weight"].dtype == "BF16"
    assert (tmp_path / "export" / "config.json").is_file()
    assert not (tmp_path / "export" / "model.safetensors").exists()
    index = json.loads((tmp_path / "export" / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 8 * 4 + 3 * 2
    with pytest.raises(RuntimeError, match="missing"):
        _export(iter(params[:1]), str(tmp_path / "e2"), ["model.layers.0.weight", "gone"], None, 16)


def test_worker_export_writes_each_tensor_in_the_checkpoint_dtype(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    from cwtrain.integrations.nemo_rl.worker import _export

    fine = 1.0 + 2**-20  # exact in FP32, 1.0 once rounded to bf16
    bias = "model.layers.0.mlp.gate.e_score_correction_bias"
    gate = "model.layers.0.mlp.gate.weight"
    norm = "model.layers.0.input_layernorm.weight"
    params = [
        (bias, torch.full((4,), fine)),  # FP32 in the policy and in the checkpoint
        (gate, torch.full((2, 4), fine)),  # FP32 in the policy, BF16 in the checkpoint
        (norm, torch.ones(4, dtype=torch.bfloat16)),  # no dtype given: written as yielded
    ]
    _export(
        iter(params),
        str(tmp_path / "export"),
        None,
        None,
        shard_bytes=2**20,
        dtypes={bias: "F32", gate: "BF16"},
    )
    locs = tensor_locations(tmp_path / "export")
    assert {n: loc.dtype for n, loc in locs.items()} == {bias: "F32", gate: "BF16", norm: "BF16"}
    written: dict[str, torch.Tensor] = {}
    for shard in (tmp_path / "export").glob("*.safetensors"):
        written.update(load_file(str(shard)))
    assert torch.equal(written[bias], torch.full((4,), fine))  # not rounded through bf16
    assert torch.equal(written[gate], torch.full((2, 4), fine).to(torch.bfloat16))  # rounded once


def test_worker_export_rejects_a_dtype_it_cannot_write(tmp_path: Path) -> None:
    from cwtrain.integrations.nemo_rl.worker import _export

    unknown = "F8_E8M0 in the deployment's checkpoint, a safetensors dtype the export does not know"
    with pytest.raises(RuntimeError, match=unknown):
        _export(
            iter([("w", torch.ones(2))]),
            str(tmp_path / "export"),
            None,
            None,
            2**20,
            {"w": "F8_E8M0"},
        )
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize(
    ("held", "stored"),
    [
        (torch.float32, "BF16"),
        (torch.bfloat16, "F32"),
        (torch.float32, "F16"),
        (torch.int8, "I8"),  # already the checkpoint's dtype
        (torch.float8_e4m3fn, "F8_E4M3"),  # already the checkpoint's dtype
    ],
)
def test_worker_export_casts_between_floats_or_keeps_the_dtype(
    tmp_path: Path, held, stored
) -> None:
    from cwtrain.integrations.nemo_rl.worker import _export

    params = iter([("w", torch.ones(4).to(held))])
    _export(params, str(tmp_path / "export"), None, None, 2**20, {"w": stored})
    assert tensor_locations(tmp_path / "export")["w"].dtype == stored


@pytest.mark.parametrize(
    ("held", "stored"),
    [
        (torch.bfloat16, "F8_E4M3"),  # FP8 block-quantized weights: a cast ignores their scales
        (torch.float32, "F8_E5M2"),
        (torch.float32, "I8"),  # int8-quantized weights
        (torch.int64, "F32"),
    ],
)
def test_worker_export_refuses_casts_a_quantized_checkpoint_needs_scales_for(
    tmp_path: Path, held, stored
) -> None:
    import re

    from cwtrain.integrations.nemo_rl.worker import _export

    params = iter([("w", torch.ones(4).to(held))])
    expected = f"w is {stored} in the deployment's checkpoint and {held} in the policy"
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        _export(params, str(tmp_path / "export"), None, None, 2**20, {"w": stored})
    assert not (tmp_path / "export").exists()


def nemo_rl_v070_generator(model, target_dtype):
    """NeMo-RL v0.7.0's dtensor_params_generator for a plain module: no DTensor, LoRA, adapter."""
    for name, tensor in model.state_dict().items():
        yield name, tensor.to(target_dtype, non_blocking=True).contiguous()


def test_dtensor_export_skips_nemo_rls_cast_to_one_dtype(monkeypatch) -> None:
    import cwtrain.integrations.nemo_rl.worker as worker

    monkeypatch.setattr(worker, "dtensor_params_generator", nemo_rl_v070_generator)
    model = torch.nn.Linear(2, 2)  # FP32, like master weights
    kept = dict(worker._dtensor_hf_params(model, torch.bfloat16, {"weight": "F32", "bias": "F32"}))
    assert {t.dtype for t in kept.values()} == {torch.float32}  # uncast: _export applies dtypes
    cast = dict(worker._dtensor_hf_params(model, torch.bfloat16, None))
    assert {t.dtype for t in cast.values()} == {torch.bfloat16}  # no dtypes: the refit's own cast


def test_dtensor_export_stops_if_nemo_rl_casts_despite_none(monkeypatch) -> None:
    import cwtrain.integrations.nemo_rl.worker as worker

    def changed(model, target_dtype):  # a NeMo-RL that falls back to a dtype of its own for None
        for name, tensor in model.state_dict().items():
            yield name, tensor.to(target_dtype or torch.bfloat16)

    monkeypatch.setattr(worker, "dtensor_params_generator", changed)
    with pytest.raises(RuntimeError, match="no longer keeps each tensor's dtype"):
        worker._dtensor_hf_params(torch.nn.Linear(2, 2), torch.bfloat16, {"weight": "F32"})


def recasting_generator(source: torch.dtype, via: torch.dtype, back: bool):
    """A NeMo-RL whose None still casts ``source`` tensors to ``via`` (and back, if ``back``)."""

    def generator(model, target_dtype):
        for name, tensor in model.state_dict().items():
            if target_dtype is None and tensor.dtype == source:
                tensor = tensor.to(via).to(source) if back else tensor.to(via)
            yield name, tensor.to(target_dtype, non_blocking=True).contiguous()

    return generator


@pytest.mark.parametrize(
    ("source", "via", "back", "found"),
    [
        (torch.float32, torch.bfloat16, False, "F32 came back torch.bfloat16"),
        (torch.float32, torch.bfloat16, True, "F32 changed value"),
        (torch.float64, torch.float32, True, "F64 changed value"),
        (torch.float16, torch.bfloat16, False, "F16 came back torch.bfloat16"),
        (torch.bfloat16, torch.float16, True, "BF16 changed value"),  # past F16's range
    ],
)
def test_dtensor_export_stops_if_nemo_rl_casts_some_dtypes_despite_none(
    monkeypatch, source, via, back, found
) -> None:
    import cwtrain.integrations.nemo_rl.worker as worker

    monkeypatch.setattr(worker, "dtensor_params_generator", recasting_generator(source, via, back))
    with pytest.raises(RuntimeError, match=found):
        worker._dtensor_hf_params(torch.nn.Linear(2, 2), torch.bfloat16, {"weight": "F32"})


def test_dtensor_export_stops_if_nemo_rl_rounds_the_tensors_in_place(monkeypatch) -> None:
    import cwtrain.integrations.nemo_rl.worker as worker

    def in_place(model, target_dtype):  # rounds the module's own FP32 tensors, yields them
        for name, tensor in model.state_dict().items():
            if tensor.dtype == torch.float32:
                tensor.copy_(tensor.to(torch.bfloat16))
            yield name, tensor.to(target_dtype, non_blocking=True).contiguous()

    monkeypatch.setattr(worker, "dtensor_params_generator", in_place)
    with pytest.raises(RuntimeError, match="F32 changed value"):
        worker._dtensor_hf_params(torch.nn.Linear(2, 2), torch.bfloat16, {"weight": "F32"})


def test_dtensor_export_stops_if_nemo_rl_yields_other_tensors_for_a_plain_module(
    monkeypatch,
) -> None:
    import cwtrain.integrations.nemo_rl.worker as worker

    def sharded_only(model, target_dtype):  # a NeMo-RL that yields only DTensor shards, say
        return iter(())

    monkeypatch.setattr(worker, "dtensor_params_generator", sharded_only)
    with pytest.raises(RuntimeError, match="cannot check that it keeps each tensor's dtype"):
        worker._dtensor_hf_params(torch.nn.Linear(2, 2), torch.bfloat16, {"weight": "F32"})


def test_dtensor_export_checks_the_generator_once_per_process(monkeypatch) -> None:
    import cwtrain.integrations.nemo_rl.worker as worker

    probes: list = []

    def counting(model, target_dtype):
        if not isinstance(model, torch.nn.Linear):
            probes.append(model)
        yield from nemo_rl_v070_generator(model, target_dtype)

    monkeypatch.setattr(worker, "dtensor_params_generator", counting)
    model = torch.nn.Linear(2, 2)
    for _refit in range(3):
        assert len(dict(worker._dtensor_hf_params(model, torch.bfloat16, {"weight": "F32"}))) == 2
    assert len(probes) == 1


class ExportingPolicy:
    """Stands in for nemo_rl's Policy: runs the real exporter over the policy's own tensors.

    ``megatron`` runs the Megatron worker's export instead of the DTensor one.
    """

    def __init__(self, versions: list[dict], *, megatron: bool = False) -> None:
        self.versions = versions
        self.megatron = megatron
        self.calls: list[dict] = []

    def run_all_workers_single_data(self, method_name, **kwargs):
        from cwtrain.integrations.nemo_rl.worker import _export, _megatron_export

        assert method_name == "export_hf_checkpoint"
        self.calls.append(kwargs)
        params = iter(self.versions.pop(0).items())
        args = (kwargs["path"], kwargs["allowed_names"], kwargs["config_dir"], 2**20)
        export = _megatron_export if self.megatron else _export
        return [export(params, *args, kwargs.get("dtypes"))]


def test_weight_synchronizer_exports_in_the_base_checkpoint_dtypes(
    tmp_path: Path, monkeypatch
) -> None:
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    base = write_real_checkpoint(tmp_path / "base")  # F32, BF16 and I8 tensors

    def policy_weights(value: float) -> dict:
        # FP32 master weights: the policy holds every float tensor in FP32
        return {
            "model.layers.0.weight": torch.full((4, 8), value),
            "model.layers.1.weight": torch.full((3, 5), value),
            "lm_head.bias": torch.arange(16, dtype=torch.int8),
        }

    policy = ExportingPolicy([policy_weights(0.5), policy_weights(0.25)])
    loader = StubLoader()
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {"run_id": "nemo-test", "state_dir": str(tmp_path / "state")},
    }
    sync = HotLoadWeightSynchronizer(
        policy, SimpleNamespace(), cfg, hf_checkpoint=str(base), loader=loader
    )
    sync.sync_weights()  # v0: full snapshot
    sync.sync_weights()  # v1: delta
    assert policy.calls[0]["dtypes"] == {
        "model.layers.0.weight": "F32",
        "model.layers.1.weight": "BF16",
        "lm_head.bias": "I8",
    }
    assert [kind for kind, _identity, _base in loader.uploads] == ["FULL", "INCREMENTAL"]
    layout = {n: (loc.dtype, loc.shape) for n, loc in tensor_locations(base).items()}
    latest = tensor_locations(tmp_path / "state" / "exports" / "v000001")
    assert {n: (loc.dtype, loc.shape) for n, loc in latest.items()} == layout


def test_weight_synchronizer_rejects_an_export_in_other_dtypes(tmp_path: Path, monkeypatch) -> None:
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    w0 = make_weights()
    base = write_real_checkpoint(tmp_path / "base", w0)
    as_bf16 = dict(w0)
    _dtype, shape, data = w0["model.layers.0.weight"]
    as_bf16["model.layers.0.weight"] = ("BF16", shape, data[: len(data) // 2])  # same shape
    policy = FakePolicy([write_real_checkpoint(tmp_path / "e0", as_bf16)])
    loader = StubLoader()
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {"run_id": "nemo-test", "state_dir": str(tmp_path / "state")},
    }
    sync = HotLoadWeightSynchronizer(
        policy, SimpleNamespace(), cfg, hf_checkpoint=str(base), loader=loader
    )
    with pytest.raises(CWTrainError, match="layout differs") as excinfo:
        sync.sync_weights()
    assert "model.layers.0.weight is BF16 [4, 8], the base checkpoint's is F32 [4, 8]" in str(
        excinfo.value
    )
    assert loader.uploads == []  # nothing published


def test_weight_synchronizer_stops_before_publishing_a_quantized_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    fp8 = torch.ones(4, 8).to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()
    bf16 = torch.ones(2, 4, dtype=torch.bfloat16).view(torch.int16).numpy().tobytes()
    scale = torch.ones(1, 1).numpy().tobytes()
    base = write_real_checkpoint(  # FP8 block-quantized, as such checkpoints ship
        tmp_path / "base",
        {
            "model.layers.0.weight": ("F8_E4M3", (4, 8), fp8),
            "model.layers.0.weight_scale_inv": ("F32", (1, 1), scale),
            "lm_head.weight": ("BF16", (2, 4), bf16),
        },
    )
    policy = ExportingPolicy(
        [
            {  # the policy trains dequantized weights
                "model.layers.0.weight": torch.full((4, 8), 0.5, dtype=torch.bfloat16),
                "model.layers.0.weight_scale_inv": torch.ones(1, 1),
                "lm_head.weight": torch.ones(2, 4, dtype=torch.bfloat16),
            }
        ]
    )
    loader = StubLoader()
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {"run_id": "nemo-test", "state_dir": str(tmp_path / "state")},
    }
    sync = HotLoadWeightSynchronizer(
        policy, SimpleNamespace(), cfg, hf_checkpoint=str(base), loader=loader
    )
    with pytest.raises(RuntimeError, match="F8_E4M3 in the deployment's checkpoint"):
        sync.sync_weights()
    assert loader.uploads == []  # nothing reached the deployment


def test_weight_synchronizer_stops_a_megatron_export_that_rounded_fp32(
    tmp_path: Path, monkeypatch
) -> None:
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    base = write_real_checkpoint(tmp_path / "base")  # F32, BF16 and I8 tensors
    rounded = {  # a bridge that exported every float tensor as bf16
        "model.layers.0.weight": torch.full((4, 8), 0.5, dtype=torch.bfloat16),
        "model.layers.1.weight": torch.full((3, 5), 0.5, dtype=torch.bfloat16),
        "lm_head.bias": torch.arange(16, dtype=torch.int8),
    }
    loader = StubLoader()
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {"run_id": "nemo-test", "state_dir": str(tmp_path / "state")},
    }
    sync = HotLoadWeightSynchronizer(
        ExportingPolicy([rounded], megatron=True),
        SimpleNamespace(),
        cfg,
        hf_checkpoint=str(base),
        loader=loader,
    )
    # casting the bf16 values back to F32 would match the layout and serve rounded weights
    with pytest.raises(CWTrainError, match="model.layers.0.weight is BF16") as excinfo:
        sync.sync_weights()
    assert "the base checkpoint's is F32" in str(excinfo.value)
    assert loader.uploads == []


def test_weight_synchronizer_reuses_an_existing_first_version(tmp_path: Path, monkeypatch) -> None:
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    w0 = make_weights()
    base = write_real_checkpoint(tmp_path / "base", w0)
    w1 = perturb(w0, ("lm_head.bias",), seed=4)
    policy = FakePolicy(
        [write_real_checkpoint(tmp_path / "e0", w0), write_real_checkpoint(tmp_path / "e1", w1)]
    )
    loader = StubLoader()
    loader.completed = {"nemo-test-v000000"}  # a previous attempt published it
    loader.current = "nemo-test-v000000"  # and the deployment still serves it
    loader.stored["nemo-test-v000000"] = {"tensors": checkpoint_checksums(tmp_path / "e0")}
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {"run_id": "nemo-test", "state_dir": str(tmp_path / "state")},
    }
    sync = HotLoadWeightSynchronizer(
        policy, SimpleNamespace(), cfg, hf_checkpoint=str(base), loader=loader
    )
    m0 = sync.sync_weights()
    assert m0["hotload/reused_existing"] == 1.0 and loader.uploads == []
    assert loader.rollbacks == [] and "hotload/restored" not in m0
    sync.sync_weights()  # the delta still chains onto the reused version
    assert loader.uploads == [("INCREMENTAL", "nemo-test-v000001", "nemo-test-v000000")]


def test_weight_synchronizer_refuses_to_reuse_different_weights(
    tmp_path: Path, monkeypatch
) -> None:
    from cwtrain.errors import CWTrainError
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    w0 = make_weights()
    base = write_real_checkpoint(tmp_path / "base", w0)
    policy = FakePolicy([write_real_checkpoint(tmp_path / "e0", w0)])
    loader = StubLoader()
    loader.completed = {"nemo-test-v000000"}
    loader.stored["nemo-test-v000000"] = {"tensors": {"model.embed_tokens.weight": "00000000"}}
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {"run_id": "nemo-test", "state_dir": str(tmp_path / "state")},
    }
    sync = HotLoadWeightSynchronizer(
        policy, SimpleNamespace(), cfg, hf_checkpoint=str(base), loader=loader
    )
    with pytest.raises(CWTrainError, match="different weights"):
        sync.sync_weights()
    assert loader.uploads == [] and loader.manifests == []


def test_weight_synchronizer_reserves_a_reused_version_the_deployment_moved_past(
    tmp_path: Path, monkeypatch
) -> None:
    from cwtrain.hotload.delta import checkpoint_checksums
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    w0 = make_weights()
    base = write_real_checkpoint(tmp_path / "base", w0)
    policy = FakePolicy([write_real_checkpoint(tmp_path / "e0", w0)])
    loader = StubLoader()
    # an earlier run of this run id reached v5, which the deployment still serves
    loader.completed = {"nemo-test-v000000", "nemo-test-v000005"}
    loader.current = "nemo-test-v000005"
    loader.stored["nemo-test-v000000"] = {"tensors": checkpoint_checksums(tmp_path / "e0")}
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {
            "run_id": "nemo-test",
            "state_dir": str(tmp_path / "state"),
            "verify": True,
        },
    }
    sync = HotLoadWeightSynchronizer(
        policy, SimpleNamespace(), cfg, hf_checkpoint=str(base), loader=loader
    )
    m0 = sync.sync_weights()
    # no upload, but v0 is hot-loaded again and verified before the trainer is called in sync
    assert loader.uploads == []
    assert loader.rollbacks == ["nemo-test-v000000"] and loader.verified == ["nemo-test-v000000"]
    assert m0["hotload/reused_existing"] == 1.0 and m0["hotload/restored"] == 1.0
    assert m0["hotload/replicas_verified"] == 2.0 and sync.is_stale is False
    assert loader.current == "nemo-test-v000000"


def test_generate_forwards_configured_stop_criteria(monkeypatch) -> None:
    from cwtrain.integrations.nemo_rl.generation import CoreWeaveGeneration

    def respond(body):
        choice = {
            "index": 0,
            "text": "ab</answer>",
            "finish_reason": "stop",
            "stop_reason": "</answer>",
            "logprobs": {
                "tokens": ["token_id:97", "token_id:98", "token_id:60"],
                "token_logprobs": [-0.1, -0.2, -0.3],
            },
        }
        return httpx.Response(200, json={"choices": [choice] * body["n"]})

    seen = use_transport(monkeypatch, respond)
    cfg = gen_config()
    cfg["stop_strings"] = ["</answer>"]
    cfg["stop_token_ids"] = [999, 2]
    gen = CoreWeaveGeneration(cfg, Tokenizer())
    data = {
        "input_ids": torch.tensor([[5, 6], [7, 8]]),
        "input_lengths": torch.tensor([2, 2]),
        "stop_strings": [["row"], None],
    }
    gen.generate(data, greedy=False)
    by_prompt = {tuple(b["prompt"]): b for b in seen}
    # configured stops first, then the row's own; token stops and stop-string retention forwarded
    assert by_prompt[(5, 6)]["stop"] == ["</answer>", "row"]
    assert by_prompt[(7, 8)]["stop"] == ["</answer>"]
    assert all(b["stop_token_ids"] == [999, 2] for b in seen)
    assert all(b["include_stop_str_in_output"] is True for b in seen)


@pytest.mark.parametrize(
    ("sampling", "compared"),
    [({"temperature": 0.7}, False), ({"top_p": 0.9}, False), ({"top_k": 20}, False),
     ({"top_k": -1}, True)],
)  # fmt: skip
def test_required_logprobs_need_sampling_without_temperature_top_p_or_top_k(
    sampling, compared
) -> None:
    from cwtrain.integrations.nemo_rl.generation import CoreWeaveGeneration
    from cwtrain.integrations.rollout import RolloutError

    config = {**gen_config(importance_sampling=True), **sampling}
    if compared:  # top_k -1 turns top-k off, as in vLLM
        CoreWeaveGeneration(config, Tokenizer())
    else:
        with pytest.raises(RolloutError, match="which it reports before them"):
            CoreWeaveGeneration(config, Tokenizer())
    # without the correction nothing compares them, even with log-probs required (as Gym does)
    CoreWeaveGeneration({**gen_config(require_logprobs=True), **sampling}, Tokenizer())


def test_requests_are_salted_with_the_identity_the_deployment_serves(monkeypatch) -> None:
    from cwtrain.integrations.nemo_rl.generation import CoreWeaveGeneration

    def respond(body):
        choice = {
            "index": 0,
            "text": "ab",
            "finish_reason": "length",
            "logprobs": {"token_logprobs": [-0.1, -0.2], "tokens": ["token_id:97", "token_id:98"]},
        }
        return httpx.Response(200, json={"choices": [choice]})

    seen = use_transport(monkeypatch, respond)
    data = {"input_ids": torch.tensor([[5, 6]]), "input_lengths": torch.tensor([2])}
    gen = CoreWeaveGeneration(gen_config(), Tokenizer())
    gen.generate(data, greedy=False)  # no synchronizer yet: no salt
    gen.weight_synchronizer = SimpleNamespace(served_identity="nemo-test-v000004")
    gen.generate(data, greedy=False)
    assert "nvext" not in seen[0]
    assert seen[1]["nvext"] == {"cache_salt": "nemo-test-v000004"}
    assert seen[0]["seed"] != seen[1]["seed"]
    CoreWeaveGeneration(gen_config(seeds=False), Tokenizer()).generate(data, greedy=False)
    assert "seed" not in seen[2]  # coreweave_cfg.seeds: false


def test_generate_rejects_missing_logprobs_when_required(monkeypatch) -> None:
    from cwtrain.integrations.nemo_rl.generation import CoreWeaveGeneration
    from cwtrain.integrations.rollout import RolloutError

    def respond(body):
        choice = {"index": 0, "text": "ab", "finish_reason": "length"}  # no logprobs at all
        return httpx.Response(200, json={"choices": [choice]})

    use_transport(monkeypatch, respond)
    data = {"input_ids": torch.tensor([[5, 6]]), "input_lengths": torch.tensor([2])}

    gen = CoreWeaveGeneration(gen_config(), Tokenizer())  # require_logprobs defaults to on
    with pytest.raises(RolloutError, match="without aligned sampled-token log-probs"):
        gen.generate(data, greedy=False)

    gen = CoreWeaveGeneration(gen_config(require_logprobs=False), Tokenizer())
    out = gen.generate(data, greedy=False)
    assert out["output_ids"][0].tolist() == [5, 6, 97, 98]
    assert out["logprobs"][0].tolist() == [0.0, 0.0, 0.0, 0.0]

    with pytest.raises(RolloutError, match="require_logprobs"):
        CoreWeaveGeneration(gen_config(logprobs=False), Tokenizer())


class WordTokenizer:
    """Re-tokenizes "equation" as equ+ation while the endpoint sampled eq+uation."""

    eos_token_id = 2
    pad_token_id = 0
    vocab = {1: "equ", 2: "<eos>", 3: "ation", 65: "A"}

    def encode(self, text, add_special_tokens=False):
        assert text == "equation"
        return [1, 3]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.vocab.get(i, "?") for i in ids)

    def convert_ids_to_tokens(self, token_id):
        return self.vocab.get(token_id, "?")


def test_generate_text_fallback_validates_token_identities(monkeypatch) -> None:
    from cwtrain.integrations.nemo_rl.generation import CoreWeaveGeneration
    from cwtrain.integrations.rollout import RolloutError

    def respond(body):
        choice = {
            "index": 0,
            "text": "equation",
            "finish_reason": "length",
            "logprobs": {"tokens": ["eq", "uation"], "token_logprobs": [-0.3, -0.7]},
        }
        return httpx.Response(200, json={"choices": [choice]})

    use_transport(monkeypatch, respond)
    data = {"input_ids": torch.tensor([[65]]), "input_lengths": torch.tensor([1])}

    # same count, different tokens: never accepted as the response's log-probs
    gen = CoreWeaveGeneration(gen_config(token_id_prompts=False), WordTokenizer())
    with pytest.raises(RolloutError, match="without aligned sampled-token log-probs"):
        gen.generate(data, greedy=False)

    gen = CoreWeaveGeneration(
        gen_config(token_id_prompts=False, require_logprobs=False), WordTokenizer()
    )
    out = gen.generate(data, greedy=False)
    assert out["output_ids"][0].tolist() == [65, 1, 3]
    assert out["logprobs"][0].tolist() == [0.0, 0.0, 0.0] and gen.misaligned_logprobs == 1


def test_generate_text_fallback_restores_the_omitted_stop_token(monkeypatch) -> None:
    from cwtrain.integrations.nemo_rl.generation import CoreWeaveGeneration

    def respond(body):
        if body["prompt"] == "A":  # custom stop token: dropped from the text, kept in the tokens
            choice = {
                "index": 0,
                "text": "ab",
                "finish_reason": "stop",
                "stop_reason": 999,
                "logprobs": {
                    "tokens": ["a", "b", "<stop>"],
                    "token_logprobs": [-0.1, -0.2, -0.3],
                },
            }
        else:  # end-of-sequence retained in the text: must not be appended twice
            choice = {
                "index": 0,
                "text": "ok\x02",
                "finish_reason": "stop",
                "stop_reason": None,
                "logprobs": {
                    "tokens": ["o", "k", "<eos>"],
                    "token_logprobs": [-0.5, -0.6, -0.7],
                },
            }
        return httpx.Response(200, json={"choices": [choice]})

    use_transport(monkeypatch, respond)
    cfg = gen_config(token_id_prompts=False)
    cfg["stop_token_ids"] = [999]
    gen = CoreWeaveGeneration(cfg, Tokenizer())
    data = {"input_ids": torch.tensor([[65], [66]]), "input_lengths": torch.tensor([1, 1])}
    out = gen.generate(data, greedy=False)
    assert out["output_ids"][0].tolist() == [65, 97, 98, 999]
    assert out["output_ids"][1].tolist() == [66, 111, 107, 2]
    assert out["logprobs"][0].tolist()[1:] == pytest.approx([-0.1, -0.2, -0.3])
    assert out["logprobs"][1].tolist()[1:] == pytest.approx([-0.5, -0.6, -0.7])
    assert out["truncated"].tolist() == [False, False] and gen.misaligned_logprobs == 0


def _restart_fixture(tmp_path: Path, monkeypatch, versions: int):
    """Base checkpoint plus per-version exports w0, w1, w2 and a loader ready for a restart."""
    from cwtrain.hotload.delta import checkpoint_checksums

    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    w0 = make_weights()
    w1 = perturb(w0, ("lm_head.bias",), seed=4)
    w2 = perturb(w1, ("model.layers.0.weight",), seed=5)
    base = write_real_checkpoint(tmp_path / "base", w0)
    exports = [write_real_checkpoint(tmp_path / f"e{i}", w) for i, w in enumerate((w0, w1, w2))]
    loader = StubLoader()
    loader.completed = {f"nemo-test-v{i:06d}" for i in range(versions)}
    loader.current = f"nemo-test-v{versions - 1:06d}"
    for i in range(versions):
        loader.stored[f"nemo-test-v{i:06d}"] = {"tensors": checkpoint_checksums(exports[i])}
    cfg = {
        "backend": "coreweave",
        "coreweave_cfg": {"run_id": "nemo-test", "state_dir": str(tmp_path / "state")},
    }
    return base, exports, loader, cfg


def test_weight_synchronizer_refuses_to_overwrite_a_published_delta(
    tmp_path: Path, monkeypatch
) -> None:
    from cwtrain.errors import CWTrainError
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    base, exports, loader, cfg = _restart_fixture(tmp_path, monkeypatch, versions=2)
    # the earlier attempt's v1 had other weights than this restart produces
    loader.stored["nemo-test-v000001"]["tensors"]["lm_head.bias"] = "00000000"
    sync = HotLoadWeightSynchronizer(
        FakePolicy(exports), SimpleNamespace(), cfg, hf_checkpoint=str(base), loader=loader
    )
    assert sync.sync_weights()["hotload/reused_existing"] == 1.0  # v0 matches: reused
    with pytest.raises(CWTrainError, match="different weights"):
        sync.sync_weights()  # v1 differs: refused before any record or object changes
    assert loader.uploads == [] and loader.manifests == ["nemo-test-v000000"]
    assert loader.stored["nemo-test-v000001"]["tensors"]["lm_head.bias"] == "00000000"


def test_weight_synchronizer_reuses_matching_deltas_after_a_restart(
    tmp_path: Path, monkeypatch
) -> None:
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    base, exports, loader, cfg = _restart_fixture(tmp_path, monkeypatch, versions=2)
    cfg["coreweave_cfg"]["verify"] = True  # off by default; this test checks it runs
    sync = HotLoadWeightSynchronizer(
        FakePolicy(exports), SimpleNamespace(), cfg, hf_checkpoint=str(base), loader=loader
    )
    m0, m1, m2 = sync.sync_weights(), sync.sync_weights(), sync.sync_weights()
    # v0 and v1 match what was published: no uploads, each hot-loaded again to be the served one
    assert m0["hotload/reused_existing"] == 1.0 and m1["hotload/reused_existing"] == 1.0
    assert loader.rollbacks == ["nemo-test-v000000", "nemo-test-v000001"]
    assert loader.verified == ["nemo-test-v000000", "nemo-test-v000001", "nemo-test-v000002"]
    # v2 is new: a delta against the reused v1 export, chained onto v1
    assert loader.uploads == [("INCREMENTAL", "nemo-test-v000002", "nemo-test-v000001")]
    assert "hotload/reused_existing" not in m2 and m2["hotload/incremental"] == 1.0


def test_worker_export_failure_keeps_iterating_and_raises(tmp_path: Path, monkeypatch) -> None:
    import cwtrain.integrations.nemo_rl.worker as worker

    consumed: list[str] = []

    def params():
        for i in range(4):
            consumed.append(f"t{i}")  # stands in for the per-tensor collective on every rank
            yield f"t{i}", torch.ones(4)

    real_add = worker._ShardWriter.add

    def add(self, name, tensor):
        if name == "t1":
            raise OSError("No space left on device")
        real_add(self, name, tensor)

    monkeypatch.setattr(worker._ShardWriter, "add", add)
    with pytest.raises(RuntimeError, match="export failed on rank 0: No space left"):
        worker._export(params(), str(tmp_path / "export"), None, None, shard_bytes=4)
    assert consumed == ["t0", "t1", "t2", "t3"]  # peers' collectives were all matched
    assert not (tmp_path / "export").exists() and not (tmp_path / "export.tmp").exists()


def test_ranks_agree_on_export_failure(monkeypatch) -> None:
    import cwtrain.integrations.nemo_rl.worker as worker

    assert worker._any_rank_failed(True) is True and worker._any_rank_failed(False) is False
    reduced: list[int] = []

    def all_reduce(tensor, op=None):
        reduced.append(int(tensor.item()))
        tensor.fill_(1)  # another rank reported a failure

    monkeypatch.setattr(worker.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(worker.torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(worker.torch.cuda, "is_available", lambda: False)
    assert worker._any_rank_failed(False) is True and reduced == [0]


@pytest.mark.parametrize("stored", ["missing", "unreadable"])
def test_reuse_requires_a_readable_manifest(tmp_path: Path, monkeypatch, stored: str) -> None:
    from cwtrain.errors import CWTrainError
    from cwtrain.integrations.nemo_rl.weight_sync import HotLoadWeightSynchronizer

    base, exports, loader, cfg = _restart_fixture(tmp_path, monkeypatch, versions=1)
    if stored == "missing":
        loader.stored.clear()
    else:

        def unreadable(identity):
            raise OSError("AccessDenied")

        loader.read_manifest = unreadable
    sync = HotLoadWeightSynchronizer(
        FakePolicy(exports), SimpleNamespace(), cfg, hf_checkpoint=str(base), loader=loader
    )
    with pytest.raises(CWTrainError, match="cannot be (verified|read)"):
        sync.sync_weights()
    assert loader.uploads == [] and loader.manifests == [] and loader.rollbacks == []


def test_rollout_concurrency_from_the_config_or_the_environment() -> None:
    from cwtrain.integrations.nemo_rl.generation import rollout_config_from

    env = {"CWTRAIN_ROLLOUT_BASE_URL": "u", "CWTRAIN_ROLLOUT_MODEL": "m"}
    # the recipe's ${oc.env:CWTRAIN_ROLLOUT_CONCURRENCY,32} resolves to a string
    assert rollout_config_from({"concurrency": "32"}, env).concurrency == 32
    assert rollout_config_from({}, {**env, "CWTRAIN_ROLLOUT_CONCURRENCY": "24"}).concurrency == 24
    # an empty value means the default rather than a crash
    assert rollout_config_from({"concurrency": ""}, env).concurrency == 16
    # the config wins over the environment
    both = {**env, "CWTRAIN_ROLLOUT_CONCURRENCY": "99"}
    assert rollout_config_from({"concurrency": 48}, both).concurrency == 48
    # an error names the setting the value came from
    with pytest.raises(CWTrainError, match="coreweave_cfg.concurrency must be at least 1"):
        rollout_config_from({"concurrency": 0}, env)
    # NeMo-RL has no routing replay, so it does not pay for routes it would not read
    routed = {**env, "CWTRAIN_ROLLOUT_ROUTED_EXPERTS": "1"}
    assert rollout_config_from({}, routed).routed_experts is False


@pytest.mark.parametrize(
    ("cfg", "env", "message"),
    [
        ({"timeout": 0}, {}, "coreweave_cfg.timeout must be a finite number above 0, got '0'"),
        ({"max_attempts": "x"}, {}, "coreweave_cfg.max_attempts must be a whole number, got 'x'"),
        ({"logprobs": "maybe"}, {}, "coreweave_cfg.logprobs must be a boolean, got 'maybe'"),
        ({"base_url": " "}, {}, "coreweave_cfg.base_url is required"),
        # a value from the environment keeps the variable's name
        ({}, {"CWTRAIN_ROLLOUT_TIMEOUT": "inf"}, "CWTRAIN_ROLLOUT_TIMEOUT must be a finite number"),
    ],
)
def test_rollout_config_errors_name_where_the_value_was_set(cfg, env, message) -> None:
    from cwtrain.integrations.nemo_rl.generation import rollout_config_from

    base = {"CWTRAIN_ROLLOUT_BASE_URL": "u", "CWTRAIN_ROLLOUT_MODEL": "m"}
    with pytest.raises(CWTrainError) as excinfo:
        rollout_config_from(cfg, {**base, **env})
    assert str(excinfo.value).startswith(message)
