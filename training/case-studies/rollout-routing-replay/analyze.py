# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Measure how far a trainer's forward pass is from the rollouts in a recording.

The CPU part needs only the tokenizer:

- ``text_round_trip``: would a pipeline that keeps only text train on the tokens the engine
  sampled? Single-turn: tokenize the response text. Two-turn: render the whole conversation with
  the chat template and compare it with the second turn's exact prompt.
- ``two_turn_rows``: do the second turn's rows over the first turn's positions equal the first
  turn's rows?
- ``hotload_rows``: a hot-load set sent again after a hot-load. How many prompt tokens came from
  the prefix cache, and do their rows equal the previous version's?

The GPU part (``--forward``) loads the model with Hugging Face transformers and runs each sequence
twice: once with the trainer's own routing, once with the recorded routes replayed. The replayed
experts get their gate weights from the trainer's own router scores, renormalized, as slime's and
miles' routing replay computes them. Against the engine's log-probs it reports the mean absolute
log-prob gap, the k3 KL estimate and the share of tokens whose probability ratio lies outside
[1/2, 2] (the tokens IcePop masks and TIS truncates), with 95% bootstrap intervals over
sequences. It stops first unless replaying the trainer's own choices reproduces its pass and
replaying them one row off changes it, and it runs no pass on a recording with problems. Only
sigmoid routers are supported. For the hot-load sets it rebuilds each version's routers from the
noise records and, where two versions choose differently in the first MoE layer, asks which of
them the returned rows follow.

    python analyze.py --recording recording --model-path zai-org/GLM-5.3-Flash-BF16 --out cpu.json
    python analyze.py --recording recording --model-path /models/glm --forward --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from probe import MoeLayout, agreement, load_config, moe_layout, structure_problems

HOTLOAD_SETS = ("P", "Q", "R")
#: Replaying the trainer's own choices must give back its own log-probs to this precision.
SELF_CHECK_TOLERANCE = 1e-6
#: Replaying its choices one row off must move its log-probs at least this much on average.
SHIFTED_MIN_CHANGE = 0.05


# --- the recording ---


def load_set(root: Path, version: str, name: str) -> list[dict[str, Any]]:
    """One set of one version, each sample with its ``routes``."""
    directory = root / version
    samples = [json.loads(line) for line in (directory / f"{name}.jsonl").read_text().splitlines()]
    with np.load(directory / f"{name}.npz") as routes:
        for sample in samples:
            sample["routes"] = routes[sample["id"]]
    return samples


def available(root: Path) -> dict[str, list[str]]:
    """The sets of each version in the recording: ``{"v0": ["P", "single", ...], ...}``."""
    return {
        d.name: sorted(f.stem for f in d.glob("*.jsonl"))
        for d in sorted(root.iterdir())
        if d.is_dir() and any(d.glob("*.jsonl"))
    }


def validate(samples: list[dict[str, Any]], layout: MoeLayout) -> list[str]:
    """Problems with the samples' routes: row counts and structure."""
    problems = []
    for s in samples:
        rows = len(s["prompt_ids"]) + len(s["response_ids"]) - 1
        if s["routes"].shape[0] != rows:
            problems.append(f"{s['id']}: {s['routes'].shape[0]} rows, not {rows}")
        problems += [f"{s['id']}: {p}" for p in structure_problems(s["routes"], layout)]
        if len(s["logprobs"]) != len(s["response_ids"]):
            problems.append(f"{s['id']}: {len(s['logprobs'])} log-probs")
    return problems


def kept_first_turn_tokens(one: dict[str, Any], two: dict[str, Any]) -> int:
    """How many of turn 1's response tokens turn 2's prompt repeats.

    All of them, or all but a closing stop token that turn 2's prompt replaces with the
    follow-up.
    """
    p1, response = len(one["prompt_ids"]), one["response_ids"]
    c1 = len(response)
    if two["prompt_ids"][:p1] == one["prompt_ids"]:
        tail = two["prompt_ids"][p1 : p1 + c1]
        if tail == response:
            return c1
        if c1 and tail[: c1 - 1] == response[: c1 - 1]:
            return c1 - 1
    raise ValueError(f"{two['id']} does not start with the tokens of {one['id']}")


def conversations(samples: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """The two-turn set's (first turn, second turn) pairs."""
    turns: dict[int, dict[int, dict]] = {}
    for s in samples:
        turns.setdefault(s["row"], {})[s["turn"]] = s
    return [(t[1], t[2]) for _, t in sorted(turns.items()) if 1 in t and 2 in t]


# --- the CPU checks ---


def retokenize(tokenizer: Any, sample: dict[str, Any]) -> list[int]:
    """The response as a text-only pipeline gets it back: its text, tokenized, and the stop token.

    The stop token is not in the text, so a pipeline appends the end-of-turn token itself.
    """
    ids = [int(i) for i in tokenizer.encode(sample["text"], add_special_tokens=False)]
    exact = sample["response_ids"]
    if sample["finish_reason"] == "stop" and exact and (not ids or ids[-1] != exact[-1]):
        ids.append(exact[-1])
    return ids


def first_difference(a: Sequence[int], b: Sequence[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def text_round_trip(
    tokenizer: Any, single: list[dict[str, Any]], pairs: list[tuple[dict, dict]]
) -> dict[str, Any]:
    changed = [s for s in single if retokenize(tokenizer, s) != s["response_ids"]]
    same_length = [s for s in changed if len(retokenize(tokenizer, s)) == len(s["response_ids"])]
    result: dict[str, Any] = {
        "single_turn": {
            "sequences": len(single),
            "changed": len(changed),
            "changed_same_length": len(same_length),
        },
    }
    by_finish: dict[str, dict[str, int]] = {}
    for one, two in pairs:
        conversation = [
            *one["messages"],
            {"role": "assistant", "content": one["text"]},
            {"role": "user", "content": two["follow_up"]},
        ]
        rendered = [
            int(i)
            for i in tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True, return_dict=False
            )
        ]
        bucket = by_finish.setdefault(one["finish_reason"], {"conversations": 0, "changed": 0})
        bucket["conversations"] += 1
        if rendered != two["prompt_ids"]:
            bucket["changed"] += 1
            where = first_difference(rendered, two["prompt_ids"])
            in_response = where >= len(one["prompt_ids"])
            bucket["first_change_in_response"] = bucket.get("first_change_in_response", 0)
            bucket["first_change_in_response"] += int(in_response)
    result["two_turn"] = {"by_first_turn_finish": by_finish}
    return result


def two_turn_rows(pairs: list[tuple[dict, dict]], layout: MoeLayout) -> dict[str, Any]:
    """The second turn's rows over the first turn's prompt and response, against the first's."""
    prompt, response = [], []
    for one, two in pairs:
        p, rows = len(one["prompt_ids"]), shared_rows(one, two)
        prompt.append(agreement(one["routes"][:p], two["routes"][:p], layout))
        response.append(agreement(one["routes"][p:rows], two["routes"][p:rows], layout))
    cached = [int(two["cached_tokens"] or 0) for _, two in pairs]
    return {
        "conversations": len(pairs),
        "second_turn_cached_tokens": {"mean": float(np.mean(cached)) if cached else 0.0},
        "first_turn_prompt_rows_agree": float(np.mean(prompt)) if prompt else None,
        "first_turn_response_rows_agree": float(np.mean(response)) if response else None,
    }


def shared_rows(one: dict[str, Any], two: dict[str, Any]) -> int:
    """Turn 1's rows whose positions hold the same tokens in turn 2."""
    return min(one["routes"].shape[0], len(one["prompt_ids"]) + kept_first_turn_tokens(one, two))


def hotload_rows(before: list[dict], after: list[dict], layout: MoeLayout) -> dict[str, Any]:
    """The same prompts on two versions: cached tokens on the second send, and how rows compare."""
    earlier = {s["id"]: s for s in before}
    cached_part, rest, cached = [], [], []
    identical = 0
    for s in after:
        b = earlier[s["id"]]
        p = len(s["prompt_ids"])
        hit = int(s["cached_tokens"] or 0)
        cached.append(hit)
        if hit:
            cached_part.append(agreement(b["routes"][:hit], s["routes"][:hit], layout))
            identical += int(np.array_equal(b["routes"][:hit], s["routes"][:hit]))
        rest.append(agreement(b["routes"][hit:p], s["routes"][hit:p], layout))
    return {
        "prompts": len(after),
        "prompts_with_cached_tokens": sum(h > 0 for h in cached),
        "cached_tokens_mean": float(np.mean(cached)) if cached else 0.0,
        "cached_rows_agree": float(np.mean(cached_part)) if cached_part else None,
        "cached_rows_byte_identical": identical,
        "recomputed_prompt_rows_agree": float(np.mean(rest)) if rest else None,
    }


# --- statistics ---


def bootstrap(per_seq: list[tuple[float, int]], n: int = 2000, seed: int = 0) -> dict[str, Any]:
    """A token-weighted mean over sequences, with a 95% percentile interval."""
    rng = np.random.default_rng(seed)
    sums = np.array([s for s, _ in per_seq], dtype=np.float64)
    counts = np.array([c for _, c in per_seq], dtype=np.float64)
    point = sums.sum() / max(counts.sum(), 1)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(per_seq), len(per_seq))
        draws.append(sums[pick].sum() / max(counts[pick].sum(), 1))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"value": float(point), "ci95": [float(lo), float(hi)]}


def ratio(
    a: list[tuple[float, int]], b: list[tuple[float, int]], n: int = 2000, seed: int = 1
) -> dict[str, Any]:
    """The ratio of two token-weighted means over the same sequences, resampled together."""
    rng = np.random.default_rng(seed)
    sa = np.array([s for s, _ in a], dtype=np.float64)
    sb = np.array([s for s, _ in b], dtype=np.float64)

    def value(pick: np.ndarray) -> float:
        return float(sa[pick].sum() / max(sb[pick].sum(), 1e-30))

    draws = [value(rng.integers(0, len(a), len(a))) for _ in range(n)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"value": value(np.arange(len(a))), "ci95": [float(lo), float(hi)]}


def gaps(trainer: np.ndarray, engine: Sequence[float]) -> dict[str, tuple[float, int]]:
    """Per-sequence sums for the log-prob metrics, as (sum, tokens)."""
    if len(trainer) != len(engine):
        raise ValueError(f"{len(trainer)} trainer log-probs for {len(engine)} engine log-probs")
    d = trainer.astype(np.float64) - np.asarray(engine, dtype=np.float64)
    n = len(d)
    return {
        "abs": (float(np.abs(d).sum()), n),
        "k3": (float((np.exp(d) - 1 - d).sum()), n),
        "outside": (float((np.abs(d) > math.log(2)).sum()), n),
    }


def summarize(per: dict[str, list[dict[str, tuple[float, int]]]]) -> dict[str, Any]:
    """Each metric for each routing, and own over replay with its interval.

    Tokens outside [1/2, 2] are also counted. When replay leaves none, the ratio has no finite
    value, and the counts say what there is to say.
    """
    out: dict[str, Any] = {}
    for metric in ("abs", "k3", "outside"):
        for name, rows in per.items():
            out[f"{metric}_{name}"] = bootstrap([r[metric] for r in rows])
        if "own" in per and "replay" in per:
            own = [r[metric] for r in per["own"]]
            replay = [r[metric] for r in per["replay"]]
            if sum(s for s, _ in replay) > 0:
                out[f"{metric}_own_over_replay"] = ratio(own, replay)
    out["outside_tokens"] = {
        name: int(sum(r["outside"][0] for r in rows)) for name, rows in per.items()
    } | {"of": int(sum(r["outside"][1] for r in next(iter(per.values()))))}
    return out


# --- the GPU part ---


def check_scoring(config: dict[str, Any]) -> None:
    """Stop unless the model's routers score with a sigmoid, the only gates the replay computes."""
    text = config.get("text_config") or config
    scoring = text.get("scoring_func") or "softmax"
    if scoring != "sigmoid":
        raise SystemExit(
            f"the model's routers use {scoring} scores; the replay here computes gates for "
            "sigmoid routers only (GLM, DeepSeek V3)"
        )


def check_routers(routers: dict[int, Any], layout: MoeLayout) -> None:
    """Stop unless the model has a router hook on exactly the config's MoE layers."""
    if sorted(routers) != layout.moe:
        found = sorted(routers) or "none"
        raise SystemExit(f"found routers on layers {found}, but the MoE layers are {layout.moe}")


class Routing:
    """Forward hooks on every router: record its choices, or replay given ones."""

    def __init__(self, model: Any):
        self.routers: dict[int, Any] = {}
        for name, module in model.named_modules():
            if "topkrouter" in type(module).__name__.lower():
                self.routers[int(name.split(".layers.")[1].split(".")[0])] = module
        self.routers = dict(sorted(self.routers.items()))
        self.replay: np.ndarray | None = None  # [tokens, layers, top_k]
        self.chosen: dict[int, np.ndarray] = {}
        for layer, module in self.routers.items():
            module.register_forward_hook(self._hook(layer))

    def _hook(self, layer: int) -> Callable:
        import torch

        def hook(module: Any, inputs: Any, output: Any) -> Any:
            logits, weights, idx = output
            self.chosen[layer] = idx.detach().cpu().numpy()
            if self.replay is None:
                return None
            experts = torch.as_tensor(self.replay[:, layer, :].astype(np.int64), device=idx.device)
            experts = experts.reshape(idx.shape)
            w = logits.float().sigmoid().gather(-1, experts)
            if getattr(module, "norm_topk_prob", True):
                w = w / (w.sum(-1, keepdim=True) + 1e-20)
            w = w * float(getattr(module, "routed_scaling_factor", 1.0))
            return logits, w.to(weights.dtype), experts

        return hook

    def choices(self, layers: int) -> np.ndarray:
        """The last pass's own choices as a route array with every decoder layer."""
        rows = next(iter(self.chosen.values())).shape[0]
        out = np.zeros((rows, layers, next(iter(self.chosen.values())).shape[-1]), np.int64)
        for layer, idx in self.chosen.items():
            out[:, layer] = idx
        return out


def set_routers(routing: Routing, base: dict[int, Any], records: list[dict[str, Any]]) -> None:
    """Load the base router weights, then apply each noise record in order, checking digests."""
    import torch
    from router_noise import noised, sha256

    for layer, module in routing.routers.items():
        module.weight.data.copy_(base[layer])
    for record in records:
        for name, entry in record["tensors"].items():
            module = routing.routers[entry["layer"]]
            bits = module.weight.detach().cpu().view(torch.int16).numpy().view(np.uint16)
            if sha256(bits) != entry["sha256_before"]:
                raise SystemExit(f"{name}: the weights before the noise differ from the recording")
            new = noised(bits, seed=record["seed"], layer=entry["layer"], sigma=entry["sigma"])
            if sha256(new) != entry["sha256_after"]:
                raise SystemExit(f"{name}: the rebuilt weights differ from the published ones")
            tensor = torch.from_numpy(new.view(np.int16).copy()).view(torch.bfloat16)
            module.weight.data.copy_(tensor.to(module.weight.device))


class Trainer:
    """The trainer side: log-probs and routing choices of the Hugging Face model."""

    def __init__(self, model_path: str, layout: MoeLayout):
        import torch
        import transformers

        config = load_config(model_path)
        check_scoring(config)
        cls = getattr(transformers, config["architectures"][0])
        began = time.monotonic()
        self.model = cls.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
        ).eval()
        self.load_seconds = round(time.monotonic() - began, 1)
        self.routing = Routing(self.model)
        check_routers(self.routing.routers, layout)
        self.base = {i: m.weight.detach().clone() for i, m in self.routing.routers.items()}
        self.layout = layout
        self.device = next(self.model.parameters()).device

    def logprobs(self, ids: Sequence[int], replay: np.ndarray | None = None) -> np.ndarray:
        """The log-prob of each token after the first, given the tokens before it."""
        import torch

        self.routing.replay = replay
        try:
            with torch.no_grad():
                inputs = torch.tensor([list(ids[:-1])], device=self.device)
                logits = self.model(input_ids=inputs).logits[0].float()
                target = torch.tensor(list(ids[1:]), device=logits.device)
                lp = torch.log_softmax(logits, -1).gather(-1, target[:, None])[:, 0]
        finally:
            self.routing.replay = None
        return lp.cpu().numpy()

    def self_check(self, ids: Sequence[int]) -> dict[str, float]:
        """Replay the trainer's own choices, as they are and one row off, on one sequence.

        The first must reproduce its pass and the second must change it: a hook whose return
        value the model ignores passes the first only. :func:`check_replay` enforces both.
        """
        own = self.logprobs(ids)
        chosen = self.routing.choices(self.layout.layers)
        again = self.logprobs(ids, chosen)
        moved = self.logprobs(ids, np.concatenate([chosen[1:], chosen[-1:]]))
        return {
            "self_check_max_abs_dlogp": float(np.abs(again - own).max()),
            "self_check_shifted_mean_abs_dlogp": float(np.abs(moved - own).mean()),
        }


def check_replay(result: dict[str, float]) -> None:
    """Stop before any comparison unless the replay hook passed both self-checks."""
    same = result["self_check_max_abs_dlogp"]
    if not same <= SELF_CHECK_TOLERANCE:
        raise SystemExit(
            f"replaying the trainer's own choices moved its log-probs by up to {same:.2e} (more "
            f"than {SELF_CHECK_TOLERANCE:.0e}): the replay does not reproduce the router"
        )
    moved = result["self_check_shifted_mean_abs_dlogp"]
    if not moved >= SHIFTED_MIN_CHANGE:
        raise SystemExit(
            f"replaying the trainer's choices one row off moved its log-probs by {moved:.2e} on "
            f"average (less than {SHIFTED_MIN_CHANGE}): the replay hook does not change the "
            "forward pass"
        )


def forward_single(trainer: Trainer, samples: list[dict], limit: int = 0) -> dict[str, Any]:
    layout = trainer.layout
    per: dict[str, list] = {"own": [], "replay": []}
    disagree, any_layer = [], []
    for k, s in enumerate(samples[: limit or None]):
        ids, p = s["prompt_ids"] + s["response_ids"], len(s["prompt_ids"])
        own = trainer.logprobs(ids)[p - 1 :]
        chosen = trainer.routing.choices(layout.layers)
        replay = trainer.logprobs(ids, s["routes"])[p - 1 :]
        per["own"].append(gaps(own, s["logprobs"]))
        per["replay"].append(gaps(replay, s["logprobs"]))
        moe = layout.moe
        same = (np.sort(chosen[:, moe], -1) == np.sort(s["routes"][:, moe], -1)).all(-1)
        disagree.append((float((~same).sum()), int(same.size)))
        any_layer.append((float((~same).any(-1).sum()), int(same.shape[0])))
        if k % 16 == 0:
            (o, n), (r, _) = per["own"][-1]["abs"], per["replay"][-1]["abs"]
            print(f"single {k}: |dlogp| own {o / n:.4f}, replay {r / n:.4f}", flush=True)
    return {
        "sequences": len(per["own"]),
        "tokens": sum(n for _, n in (r["abs"] for r in per["own"])),
        "routers_disagree": bootstrap(disagree),
        "tokens_with_a_disagreeing_layer": bootstrap(any_layer),
        **summarize(per),
    }


def forward_two_turn(trainer: Trainer, pairs: list[tuple[dict, dict]]) -> dict[str, Any]:
    """One pass over each whole conversation, as a trainer computes every turn's log-probs.

    Routes: the trainer's own; the second turn's rows for every position; or the first turn's rows
    over the positions it shares with the second, then the second turn's. Turn 1's tokens are
    those that turn 2's prompt repeats, which leaves out a closing stop token that it replaced.
    """
    per = {f"{turn}_{name}": [] for turn in ("turn1", "turn2") for name in ("own", "last", "each")}
    for one, two in pairs:
        ids = two["prompt_ids"] + two["response_ids"]
        p1, p2 = len(one["prompt_ids"]), len(two["prompt_ids"])
        kept, rows = kept_first_turn_tokens(one, two), shared_rows(one, two)
        each = two["routes"].copy()
        each[:rows] = one["routes"][:rows]
        for name, replay in (("own", None), ("last", two["routes"]), ("each", each)):
            lp = trainer.logprobs(ids, replay)
            per[f"turn1_{name}"].append(gaps(lp[p1 - 1 : p1 - 1 + kept], one["logprobs"][:kept]))
            per[f"turn2_{name}"].append(gaps(lp[p2 - 1 :], two["logprobs"]))
    out: dict[str, Any] = {"conversations": len(pairs)}
    for key, rows in per.items():
        out[f"abs_{key}"] = bootstrap([r["abs"] for r in rows])
        out[f"outside_{key}"] = bootstrap([r["outside"] for r in rows])
    return out


def forward_hotload(
    trainer: Trainer, recording: Path, noise: dict[str, list[dict]], sets: dict[str, list[str]]
) -> dict[str, Any]:
    """Which version's weights computed the prompt rows that a hot-load set got back?

    The first MoE layer's router sees the same input under every version, since no router comes
    before it. Where two versions' routers choose differently there, the row the engine returned
    shows which of the two it ran: it mostly matches that version's choice.
    """
    first = trainer.layout.moe[0]
    recorded = {
        (version, name): load_set(recording, version, name)
        for version, names in sets.items()
        for name in names
        if name in HOTLOAD_SETS
    }
    prompts = {s["id"]: s["prompt_ids"] for samples in recorded.values() for s in samples}
    choices: dict[str, dict[str, np.ndarray]] = {}
    for version in sorted(noise):
        set_routers(trainer.routing, trainer.base, noise[version])
        for sid, ids in prompts.items():
            trainer.logprobs(ids + [ids[-1]])  # a row for every prompt token
            chosen = trainer.routing.choices(trainer.layout.layers)[:, first]
            choices.setdefault(version, {})[sid] = np.sort(chosen, axis=-1)
    set_routers(trainer.routing, trainer.base, [])
    versions = sorted(noise)
    out: dict[str, Any] = {}
    for (version, name), samples in sorted(recorded.items()):
        cell: dict[str, Any] = {}
        for a, b in zip(versions, versions[1:], strict=False):
            for part in ("cached", "recomputed"):
                tokens, follows_a, follows_b = 0, 0, 0
                for s in samples:
                    hit = int(s["cached_tokens"] or 0)
                    rows = slice(0, hit) if part == "cached" else slice(hit, len(s["prompt_ids"]))
                    engine = np.sort(s["routes"][rows, first].astype(np.int64), axis=-1)
                    ca, cb = choices[a][s["id"]][rows], choices[b][s["id"]][rows]
                    differ = ~(ca == cb).all(-1)
                    tokens += int(differ.sum())
                    follows_a += int((engine[differ] == ca[differ]).all(-1).sum())
                    follows_b += int((engine[differ] == cb[differ]).all(-1).sum())
                if tokens:
                    cell[f"{part}_{a}_or_{b}"] = {
                        "tokens": tokens,
                        f"follows_{a}": follows_a / tokens,
                        f"follows_{b}": follows_b / tokens,
                    }
        out[f"{name}@{version}"] = cell
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure a recording against a trainer.")
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--model-path", required=True, help="Hugging Face id or local directory")
    parser.add_argument("--revision", help="the model's Hugging Face revision, for its tokenizer")
    parser.add_argument("--out", type=Path, required=True, help="the report, as JSON")
    parser.add_argument("--forward", action="store_true", help="run the trainer-side passes")
    parser.add_argument(
        "--parts", default="single,two_turn,hotload", help="the trainer-side passes to run"
    )
    parser.add_argument("--limit", type=int, default=0, help="at most this many single samples")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.revision)
    layout = moe_layout(load_config(args.model_path))
    sets = available(args.recording)
    report: dict[str, Any] = {"sets": sets}
    single = load_set(args.recording, "v0", "single")
    pairs = conversations(load_set(args.recording, "v0", "two_turn"))
    problems = []
    for version, names in sets.items():
        for name in names:
            problems += validate(load_set(args.recording, version, name), layout)
    report["problems"] = problems
    report["text_round_trip"] = text_round_trip(tokenizer, single, pairs)
    report["two_turn_rows"] = two_turn_rows(pairs, layout)
    order = sorted(sets)
    for earlier, later in zip(order, order[1:], strict=False):
        for name in HOTLOAD_SETS:
            if name in sets[earlier] and name in sets[later]:
                report[f"hotload_{name}_{earlier}_to_{later}"] = hotload_rows(
                    load_set(args.recording, earlier, name),
                    load_set(args.recording, later, name),
                    layout,
                )
    print(json.dumps({k: v for k, v in report.items() if k != "sets"}, indent=2), flush=True)

    if args.forward and problems:
        print(f"no trainer-side passes: {len(problems)} problems in the recording", flush=True)
    elif args.forward:
        parts = set(args.parts.split(","))
        trainer = Trainer(args.model_path, layout)
        report["load_seconds"] = trainer.load_seconds
        checks = trainer.self_check(single[0]["prompt_ids"] + single[0]["response_ids"])
        print(f"self-check: {json.dumps(checks)}", flush=True)
        check_replay(checks)
        report |= checks
        if "single" in parts:
            report["single"] = forward_single(trainer, single, args.limit)
            print(json.dumps({"single": report["single"]}, indent=2), flush=True)
        if "two_turn" in parts:
            report["two_turn"] = forward_two_turn(trainer, pairs)
            print(json.dumps({"two_turn": report["two_turn"]}, indent=2), flush=True)
        noise = {"v0": []}
        for version in order[1:]:
            path = args.recording / version / "router_noise.json"
            if path.exists():
                noise[version] = noise[sorted(noise)[-1]] + [json.loads(path.read_text())]
        if "hotload" in parts and len(noise) > 1:
            report["hotload"] = forward_hotload(trainer, args.recording, noise, sets)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
