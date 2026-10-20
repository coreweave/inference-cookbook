# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import json
import math
import sys
import types
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
            record.second_turn_prompt(prompt, response, suffix, (USER, STOP)),
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
    record.write_set(tmp_path / "v0", "two_turn", record.SetResult([dict(s) for s in samples]))
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


def test_gaps_refuse_log_probs_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="2 trainer log-probs for 1 engine log-probs"):
        analyze.gaps(np.zeros(2), [0.0])


# --- the forward passes, with a trainer that needs no GPU ---


class FakeRouting:
    """Own choices: experts 0 and 1 in every MoE layer."""

    def __init__(self):
        self.rows = 0

    def choices(self, layers: int) -> np.ndarray:
        out = np.zeros((self.rows, layers, LAYOUT.top_k), np.int64)
        out[:, LAYOUT.moe] = [0, 1]
        return out


class FakeTrainer:
    """Log-probs that name their token, -id / 100, and 0.5 lower under any replay."""

    def __init__(self):
        self.layout, self.routing, self.calls = LAYOUT, FakeRouting(), []

    def logprobs(self, ids, replay=None):
        self.calls.append((list(ids), replay))
        self.routing.rows = len(ids) - 1
        scores = -np.asarray(ids[1:], dtype=np.float64) / 100
        return scores if replay is None else scores - 0.5


def engine_logprobs(ids: list[int]) -> list[float]:
    return [-t / 100 for t in ids]


def test_single_turn_passes_score_the_response_tokens_with_and_without_the_routes() -> None:
    rows = routes(4, 6)
    rows[:2, LAYOUT.moe] = [0, 1]  # the first two rows route as the trainer does
    s = sample("single-0000", [1, 2, 3], [4, 5], "x", routes=rows)
    s["logprobs"] = engine_logprobs([4, 5])
    trainer = FakeTrainer()
    result = analyze.forward_single(trainer, [s])
    assert result["abs_own"]["value"] == pytest.approx(0.0)  # positions 3 and 4, exactly
    assert result["abs_replay"]["value"] == pytest.approx(0.5)
    assert trainer.calls[1][1] is s["routes"]
    assert result["routers_disagree"]["value"] == pytest.approx(0.5)
    assert result["tokens"] == 2


def conversation(turn1_end: int) -> tuple[dict, dict]:
    """Turn 1 is [7, 8] then [9, 10, turn1_end]; turn 2's prompt continues with [1, 20, 21]."""
    one = sample("two_turn-0000-1", [7, 8], [9, 10, turn1_end], "x", routes=routes(4, 7))
    one["logprobs"] = engine_logprobs([9, 10, turn1_end])
    two = sample("two_turn-0000-2", [7, 8, 9, 10, USER, 20, 21], [22, 23], "y", routes=routes(8, 8))
    two["logprobs"] = engine_logprobs([22, 23])
    return one, two


def test_turn_two_repeats_turn_one_and_may_replace_its_closing_stop_token() -> None:
    kept, replaced = conversation(USER), conversation(STOP)
    assert analyze.kept_first_turn_tokens(*kept) == 3  # turn 1 stopped on the follow-up's token
    assert analyze.kept_first_turn_tokens(*replaced) == 2  # its stop token gave way to [1, ...]
    assert analyze.shared_rows(*kept) == analyze.shared_rows(*replaced) == 4
    one, two = conversation(USER)
    two["prompt_ids"] = [7, 8, 99, 10, USER, 20, 21]
    with pytest.raises(ValueError, match="does not start with the tokens of two_turn-0000-1"):
        analyze.kept_first_turn_tokens(one, two)


@pytest.mark.parametrize("turn1_end", [USER, STOP])
def test_two_turn_passes_score_each_turns_tokens_at_their_positions(turn1_end) -> None:
    one, two = conversation(turn1_end)
    trainer = FakeTrainer()
    result = analyze.forward_two_turn(trainer, [(one, two)])
    # the slices land on the right tokens: a token off would show as a gap
    assert result["abs_turn1_own"]["value"] == pytest.approx(0.0)
    assert result["abs_turn2_own"]["value"] == pytest.approx(0.0)
    assert result["abs_turn1_last"]["value"] == pytest.approx(0.5)
    own, last, each = (replay for _, replay in trainer.calls)
    assert own is None and last is two["routes"]
    assert np.array_equal(each[:4], one["routes"]) and np.array_equal(each[4:], two["routes"][4:])


def test_the_replay_supports_sigmoid_routers_on_the_configs_moe_layers_only() -> None:
    analyze.check_scoring({"text_config": {"scoring_func": "sigmoid"}})
    with pytest.raises(SystemExit, match="softmax scores"):
        analyze.check_scoring({"num_experts": 128})  # Qwen3-MoE
    analyze.check_routers({1: None, 2: None}, LAYOUT)
    with pytest.raises(SystemExit, match="found routers on layers none, but the MoE layers"):
        analyze.check_routers({}, LAYOUT)
    with pytest.raises(SystemExit, match=r"found routers on layers \[1\]"):
        analyze.check_routers({1: None}, LAYOUT)


def test_a_replay_that_misses_its_tolerance_or_changes_nothing_stops_the_analysis() -> None:
    analyze.check_replay(
        {"self_check_max_abs_dlogp": 0.0, "self_check_shifted_mean_abs_dlogp": 1.0}
    )
    with pytest.raises(SystemExit, match="moved its log-probs by up to 2.50e-01"):
        analyze.check_replay(
            {"self_check_max_abs_dlogp": 0.25, "self_check_shifted_mean_abs_dlogp": 1.0}
        )
    with pytest.raises(SystemExit, match="the replay hook does not change the forward pass"):
        analyze.check_replay(
            {"self_check_max_abs_dlogp": 0.0, "self_check_shifted_mean_abs_dlogp": 0.0}
        )


@pytest.fixture
def recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-sample recording, a model directory with a config, and a stand-in tokenizer."""
    one, two = conversation(USER)
    two_turn = [dict(one, row=0, turn=1, messages=[], follow_up="again")]
    two_turn.append(dict(two, row=0, turn=2, messages=[], follow_up="again"))
    record.write_set(tmp_path / "rec" / "v0", "two_turn", record.SetResult(two_turn))
    single = sample("single-0000", [1, 2, 3], [4, 5], "x")
    record.write_set(tmp_path / "rec" / "v0", "single", record.SetResult([single]))
    model = tmp_path / "model"
    model.mkdir()
    config = {"num_hidden_layers": 3, "num_experts_per_tok": 2, "n_routed_experts": 288}
    config |= {"first_k_dense_replace": 1, "scoring_func": "sigmoid"}
    (model / "config.json").write_text(json.dumps(config))
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Tokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return tmp_path


def run_main(root: Path, *extra: str) -> int:
    args = ["--recording", str(root / "rec"), "--model-path", str(root / "model")]
    return analyze.main([*args, "--out", str(root / "report.json"), "--forward", *extra])


def test_no_forward_pass_runs_on_a_recording_with_problems(recording, monkeypatch) -> None:
    single = recording / "rec" / "v0" / "single.jsonl"
    broken = json.loads(single.read_text())
    broken["logprobs"] = broken["logprobs"][:1]  # one log-prob for two response tokens
    single.write_text(json.dumps(broken) + "\n")
    monkeypatch.setattr(analyze, "Trainer", lambda *a: pytest.fail("loaded the model"))
    assert run_main(recording) == 1
    assert "1 log-probs" in json.loads((recording / "report.json").read_text())["problems"][0]


def test_a_failed_self_check_stops_before_any_report(recording, monkeypatch) -> None:
    class Trainer:
        load_seconds = 0.0

        def __init__(self, model_path, layout):
            pass

        def self_check(self, ids):  # the identity replay moved the log-probs
            return {"self_check_max_abs_dlogp": 0.25, "self_check_shifted_mean_abs_dlogp": 1.0}

    monkeypatch.setattr(analyze, "Trainer", Trainer)
    with pytest.raises(SystemExit, match="does not reproduce the router"):
        run_main(recording)
    assert not (recording / "report.json").exists()


def test_bootstrap_intervals_contain_the_point_and_ratios_resample_pairs() -> None:
    per = [(float(i), 10) for i in range(20)]
    result = analyze.bootstrap(per)
    assert result["ci95"][0] <= result["value"] <= result["ci95"][1]
    doubled = analyze.ratio([(2 * s, n) for s, n in per], per)
    assert doubled["value"] == pytest.approx(2.0) and doubled["ci95"] == pytest.approx([2.0, 2.0])
