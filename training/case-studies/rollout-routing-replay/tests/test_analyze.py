# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze  # noqa: E402
import probe  # noqa: E402
import record  # noqa: E402

LAYOUT = probe.moe_layout(
    {
        "num_hidden_layers": 3,
        "num_experts_per_tok": 2,
        "n_routed_experts": 288,
        "first_k_dense_replace": 1,
    }
)
USER, ASSISTANT, THINK, END_THINK, STOP = 1, 2, 3, 4, 5


def chars(text: str) -> list[int]:
    return [10 + ord(c) % 1000 for c in text]


class Tokenizer:
    """Characters as tokens, except "ab" and "</think>"; a GLM-like chat template."""

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        ids, i = [], 0
        while i < len(text):
            if text.startswith("</think>", i):
                ids.append(END_THINK)
                i += len("</think>")
            elif text.startswith("ab", i):
                ids.append(999)
                i += 2
            else:
                ids.append(10 + ord(text[i]) % 1000)
                i += 1
        return ids

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        ids = []
        for m in messages:
            if m["role"] == "user":
                ids += [USER, *self.encode(m["content"], False)]
            else:
                thinking, _, answer = m["content"].rpartition("</think>")
                if not _:  # an unfinished thought renders as an answer after an empty one
                    thinking, answer = "", m["content"]
                ids += [ASSISTANT, THINK, *self.encode(thinking, False), END_THINK]
                ids += self.encode(answer.strip(), False)
        return ids + ([ASSISTANT, THINK] if add_generation_prompt else [])


def routes(rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((rows, LAYOUT.layers, LAYOUT.top_k), np.uint16)
    for r in range(rows):
        for layer in LAYOUT.moe:
            out[r, layer] = rng.choice(288, LAYOUT.top_k, replace=False)
    return out


def sample(sid: str, prompt: list[int], response: list[int], text: str, **extra) -> dict:
    given = extra.pop("routes", None)
    return {
        "id": sid,
        "prompt_ids": prompt,
        "response_ids": response,
        "logprobs": [-0.1] * len(response),
        "text": text,
        "finish_reason": extra.pop("finish_reason", "stop"),
        "cached_tokens": extra.pop("cached_tokens", 0),
        "routes": routes(len(prompt) + len(response) - 1, 0) if given is None else given,
        **extra,
    }


def test_a_text_round_trip_finds_non_canonical_tokens_and_unfinished_thoughts() -> None:
    tok = Tokenizer()
    prompt = [USER, *chars("q"), ASSISTANT, THINK]
    canonical = sample("single-0000", prompt, [*chars("x"), END_THINK, 999, STOP], "x</think>ab")
    # the engine sampled "a" then "b", which the tokenizer merges: same text, other tokens
    split = sample(
        "single-0001", prompt, [*chars("x"), END_THINK, *chars("ab"), STOP], "x</think>ab"
    )
    result = analyze.text_round_trip(tok, [canonical, split], [])
    assert result["single_turn"] == {"sequences": 2, "changed": 1, "changed_same_length": 0}


def test_two_turn_re_rendering_changes_conversations_whose_first_turn_was_cut_off() -> None:
    tok = Tokenizer()
    messages = [{"role": "user", "content": "q"}]
    prompt = record.render(tok, messages)
    pairs = []
    for i, (text, finish) in enumerate([("t</think>a", "stop"), ("thinking", "length")]):
        response = tok.encode(text.replace("</think>", ""), False)
        if finish == "stop":
            response = [*tok.encode("t", False), END_THINK, *tok.encode("a", False), USER]
        one = sample(
            f"two_turn-{i:04d}-1",
            prompt,
            response,
            text,
            finish_reason=finish,
            row=i,
            turn=1,
            messages=messages,
        )
        suffix = record.follow_up_suffix(tok, messages, "again")
        two = sample(
            f"two_turn-{i:04d}-2",
            record.second_turn_prompt(prompt, response, suffix),
            [*chars("z"), STOP],
            "z",
            row=i,
            turn=2,
            messages=messages,
            follow_up="again",
        )
        pairs.append((one, two))
    result = analyze.text_round_trip(tok, [], pairs)["two_turn"]["by_first_turn_finish"]
    assert result["stop"] == {"conversations": 1, "changed": 0}
    assert result["length"] == {"conversations": 1, "changed": 1, "first_change_in_response": 1}


def test_recordings_load_validate_and_pair_their_turns(tmp_path: Path) -> None:
    samples = [
        sample("two_turn-0003-1", [1, 2], [3, 4], "x", row=3, turn=1),
        sample("two_turn-0003-2", [1, 2, 3, 4, 5], [6], "y", row=3, turn=2),
        sample("two_turn-0004-1", [1, 2], [3], "z", row=4, turn=1),  # no second turn
    ]
    record.write_set(tmp_path / "v0", "two_turn", [dict(s) for s in samples])
    loaded = analyze.load_set(tmp_path, "v0", "two_turn")
    assert analyze.available(tmp_path) == {"v0": ["two_turn"]}
    assert analyze.validate(loaded, LAYOUT) == []
    pairs = analyze.conversations(loaded)
    assert [(a["id"], b["id"]) for a, b in pairs] == [("two_turn-0003-1", "two_turn-0003-2")]
    broken = dict(loaded[0], routes=loaded[0]["routes"][:-1])
    assert "rows, not 3" in analyze.validate([broken], LAYOUT)[0]


def test_second_turn_rows_are_compared_over_the_first_turns_positions() -> None:
    one = sample("t-1", [1, 2, 3], [4, 5], "x", routes=routes(4, 1))
    second = routes(6, 2)
    second[:3] = one["routes"][:3]  # the prompt rows match, the response rows do not
    two = sample("t-2", [1, 2, 3, 4, 5, 6], [7], "y", routes=second, cached_tokens=0)
    result = analyze.two_turn_rows([(one, two)], LAYOUT)
    assert result["first_turn_prompt_rows_agree"] == 1.0
    assert result["first_turn_response_rows_agree"] < 0.1


def test_hotload_rows_tell_cached_rows_from_recomputed_ones() -> None:
    before = sample("Q-0000", list(range(8)), [9, 9], "x", routes=routes(9, 3))
    kept = before["routes"].copy()
    kept[4:] = routes(9, 4)[4:]  # 4 cached prompt rows kept, the rest recomputed
    after = sample("Q-0000", list(range(8)), [9, 9], "x", routes=kept, cached_tokens=4)
    result = analyze.hotload_rows([before], [after], LAYOUT)
    assert result["prompts_with_cached_tokens"] == 1 and result["cached_rows_byte_identical"] == 1
    assert result["cached_rows_agree"] == 1.0 and result["recomputed_prompt_rows_agree"] < 0.2
    fresh = dict(after, cached_tokens=0, routes=routes(9, 5))
    result = analyze.hotload_rows([before], [fresh], LAYOUT)
    assert result["prompts_with_cached_tokens"] == 0 and result["cached_rows_agree"] is None


def test_log_prob_gaps_and_their_summary() -> None:
    trainer = np.log(np.array([0.5, 0.5, 0.1]))
    engine = np.log([0.5, 0.2, 0.1])
    g = analyze.gaps(trainer, engine)
    assert g["abs"] == (pytest.approx(math.log(2.5)), 3) and g["outside"] == (1.0, 3)
    zero = {"abs": (0.0, 3), "k3": (0.0, 3), "outside": (0.0, 3)}
    summary = analyze.summarize({"own": [g, g], "replay": [zero, zero]})
    assert "outside_own_over_replay" not in summary  # no finite ratio over zero tokens
    assert summary["outside_tokens"] == {"own": 2, "replay": 0, "of": 6}
    assert summary["abs_own"]["value"] == pytest.approx(math.log(2.5) / 3)


def test_bootstrap_intervals_contain_the_point_and_ratios_resample_pairs() -> None:
    per = [(float(i), 10) for i in range(20)]
    result = analyze.bootstrap(per)
    assert result["ci95"][0] <= result["value"] <= result["ci95"][1]
    doubled = analyze.ratio([(2 * s, n) for s, n in per], per)
    assert doubled["value"] == pytest.approx(2.0) and doubled["ci95"] == pytest.approx([2.0, 2.0])
