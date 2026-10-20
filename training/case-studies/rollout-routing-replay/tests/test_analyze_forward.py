# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The trainer side's hooks on a tiny MoE model in torch, on CPU. Skipped where torch is missing."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze  # noqa: E402
import probe  # noqa: E402
import router_noise  # noqa: E402

LAYOUT = probe.moe_layout(
    {"num_hidden_layers": 3, "num_experts_per_tok": 2, "n_routed_experts": 8,
     "first_k_dense_replace": 1}
)  # fmt: skip
HIDDEN, VOCAB, SCALE = 16, 40, 2.5


class TopkRouter(torch.nn.Module):
    """A sigmoid router that returns (logits, weights, experts), as GLM's does in transformers."""

    def __init__(self):
        super().__init__()
        weight = torch.randn(LAYOUT.experts, HIDDEN) * 0.5
        self.weight = torch.nn.Parameter(weight.to(torch.bfloat16))
        self.norm_topk_prob, self.routed_scaling_factor = True, SCALE

    def forward(self, x):
        logits = x.float() @ self.weight.float().T
        weights, experts = logits.sigmoid().topk(LAYOUT.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True) * SCALE
        return logits, weights.to(torch.bfloat16), experts


class Moe(torch.nn.Module):
    """Experts weighted by the router's output, or, if ``ignore_hooks``, by its own top-k."""

    def __init__(self, ignore_hooks: bool):
        super().__init__()
        self.gate, self.ignore_hooks = TopkRouter(), ignore_hooks
        self.experts = torch.nn.Parameter(torch.randn(LAYOUT.experts, HIDDEN, HIDDEN))

    def forward(self, x):
        logits, weights, experts = self.gate(x)
        if self.ignore_hooks:  # like a fused kernel that routes by itself
            weights, experts = logits.sigmoid().topk(LAYOUT.top_k, dim=-1)
        out = torch.zeros_like(x)
        for slot in range(LAYOUT.top_k):
            mixed = torch.einsum("th,thk->tk", x, self.experts[experts[:, slot]])
            out = out + weights[:, slot, None].float() * mixed
        return x + out


class Layer(torch.nn.Module):
    def __init__(self, moe: bool, ignore_hooks: bool):
        super().__init__()
        self.mlp = Moe(ignore_hooks) if moe else torch.nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        return self.mlp(x)


class Model(torch.nn.Module):
    def __init__(self, ignore_hooks: bool = False):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.model = torch.nn.Module()
        layers = [Layer(i in LAYOUT.moe, ignore_hooks) for i in range(LAYOUT.layers)]
        self.model.layers = torch.nn.ModuleList(layers)
        self.head = torch.nn.Linear(HIDDEN, VOCAB)

    def forward(self, input_ids):
        x = self.embed(input_ids[0])
        for layer in self.model.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.head(x)[None])


def trainer(ignore_hooks: bool = False) -> analyze.Trainer:
    """analyze.Trainer around the tiny model, without loading a checkpoint."""
    t = object.__new__(analyze.Trainer)
    t.model = Model(ignore_hooks).eval()
    t.routing = analyze.Routing(t.model)
    analyze.check_routers(t.routing.routers, LAYOUT)
    t.base = {i: m.weight.detach().clone() for i, m in t.routing.routers.items()}
    t.layout, t.device = LAYOUT, torch.device("cpu")
    return t


IDS = [3, 14, 15, 9, 26, 5, 35, 8, 9, 7, 9, 3]


def test_the_hooks_record_choices_and_replay_given_experts_with_the_routers_gates() -> None:
    t = trainer()
    t.logprobs(IDS)
    chosen = t.routing.choices(LAYOUT.layers)
    assert chosen.shape == (len(IDS) - 1, LAYOUT.layers, LAYOUT.top_k)
    assert not chosen[:, 0].any()  # the dense layer
    replay = np.zeros_like(chosen)
    replay[:, LAYOUT.moe] = [6, 7]
    x = torch.randn(len(IDS) - 1, HIDDEN)
    t.routing.replay = replay
    logits, weights, experts = t.routing.routers[1](x)
    t.routing.replay = None
    assert (experts == 6).sum() + (experts == 7).sum() == experts.numel()
    gates = logits.float().sigmoid().gather(-1, experts)
    gates = gates / gates.sum(-1, keepdim=True) * SCALE
    assert torch.allclose(weights.float(), gates, rtol=1e-2)


def test_the_self_check_passes_a_working_hook_and_stops_one_the_model_ignores() -> None:
    checks = trainer().self_check(IDS)
    assert checks["self_check_max_abs_dlogp"] <= analyze.SELF_CHECK_TOLERANCE
    analyze.check_replay(checks)
    ignored = trainer(ignore_hooks=True).self_check(IDS)
    assert ignored["self_check_max_abs_dlogp"] == 0.0  # the first check alone would pass it
    with pytest.raises(SystemExit, match="does not change the forward pass"):
        analyze.check_replay(ignored)


def noise_record(t: analyze.Trainer, seed: int) -> dict:
    tensors = {}
    for layer, module in t.routing.routers.items():
        bits = module.weight.detach().view(torch.int16).numpy().view(np.uint16)
        sigma = 0.01 * router_noise.rms(bits)
        new = router_noise.noised(bits, seed=seed, layer=layer, sigma=sigma)
        tensors[f"model.layers.{layer}.mlp.gate.weight"] = {
            "layer": layer,
            "sigma": sigma,
            "sha256_before": router_noise.sha256(bits),
            "sha256_after": router_noise.sha256(new),
        }
    return {"seed": seed, "tensors": tensors}


def router_bits(t: analyze.Trainer, layer: int) -> np.ndarray:
    return t.routing.routers[layer].weight.detach().view(torch.int16).numpy().view(np.uint16)


def test_router_versions_are_rebuilt_from_the_noise_records_and_their_digests() -> None:
    t = trainer()
    record = noise_record(t, seed=1)
    analyze.set_routers(t.routing, t.base, [record])
    for name, entry in record["tensors"].items():
        assert router_noise.sha256(router_bits(t, entry["layer"])) == entry["sha256_after"], name
    analyze.set_routers(t.routing, t.base, [])  # back to the base weights
    for entry in record["tensors"].values():
        assert router_noise.sha256(router_bits(t, entry["layer"])) == entry["sha256_before"]
    tampered = {**record, "seed": 2}  # other noise than the one published
    with pytest.raises(SystemExit, match="the rebuilt weights differ from the published ones"):
        analyze.set_routers(t.routing, t.base, [tampered])
    with pytest.raises(SystemExit, match="the weights before the noise differ"):
        analyze.set_routers(t.routing, t.base, [record, record])  # the second sees v1, not v0
