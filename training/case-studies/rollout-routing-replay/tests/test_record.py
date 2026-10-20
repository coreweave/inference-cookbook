# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import asyncio
import base64
import json
import struct
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
from cwtrain.hotload import SnapshotError, inspect_snapshot
from cwtrain.integrations.rollout import RolloutConfig, RolloutError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import record  # noqa: E402
import router_noise  # noqa: E402

USER, ASSISTANT, THINK, END_THINK, END_OF_TEXT = 1, 2, 3, 4, 5
STOPS = {USER, END_OF_TEXT}  # GLM stops on <|user|> or <|endoftext|>


class Tokenizer:
    """A GLM-like chat template over character ids: user turns, then assistant turns in <think>."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        assert tokenize and not return_dict
        ids = []
        for m in messages:
            text = [10 + ord(c) % 1000 for c in m["content"]]
            ids += [USER, *text] if m["role"] == "user" else [ASSISTANT, THINK, END_THINK, *text]
        return ids + ([ASSISTANT, THINK] if add_generation_prompt else [])


def test_the_second_turn_is_the_first_turns_ids_then_the_rendered_follow_up() -> None:
    tok = Tokenizer()
    messages = [{"role": "user", "content": "hi"}]
    suffix = record.follow_up_suffix(tok, messages, "ok?")
    assert suffix == [USER, *[10 + ord(c) for c in "ok?"], ASSISTANT, THINK]
    prompt = record.render(tok, messages)
    # A stop token that ended turn 1 is left out, whichever it is: the suffix starts the next turn.
    assert record.second_turn_prompt(prompt, [50, USER], suffix, STOPS) == prompt + [50, *suffix]
    turn1 = [50, END_OF_TEXT]
    assert record.second_turn_prompt(prompt, turn1, suffix, STOPS) == prompt + [50, *suffix]
    cut_off = [50, 99]
    assert record.second_turn_prompt(prompt, cut_off, suffix, STOPS) == prompt + [*cut_off, *suffix]


def test_long_prompts_differ_from_the_first_token_on_and_reach_their_length() -> None:
    tok = Tokenizer()
    a, b = (record.render(tok, record.long_prompt(i, 600, tok)) for i in (0, 1))
    assert len(a) >= 600 and len(b) >= 600
    assert a[:2] == b[:2] and a[: a.index(10 + ord("0")) + 5] != b[: b.index(10 + ord("0")) + 5]


def encode(routes: np.ndarray) -> dict:
    data = base64.b64encode(routes.tobytes()).decode("ascii")
    return {"data": data, "shape": list(routes.shape), "start": 0, "dtype": str(routes.dtype)}


class Engine:
    """Completions with routes.

    The first ``drop`` responses come without routes and the first ``unavailable`` with HTTP 503.
    ``fill`` answers with ``max_tokens`` tokens; ``fail`` answers that prompt with HTTP 400.
    """

    def __init__(self, drop=0, cached=0, unavailable=0, fill=False, fail=None):
        self.drop, self.cached, self.requests = drop, cached, []
        self.unavailable, self.fill, self.fail = unavailable, fill, fail

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if len(self.requests) <= self.unavailable:
            return httpx.Response(503, text="busy")
        if body["prompt"] == self.fail:
            return httpx.Response(400, text="bad prompt")
        out = [7] * body["max_tokens"] if self.fill else [7, 8, 9]
        rows = len(body["prompt"]) + len(out) - 1
        routes = (np.arange(rows * 2 * 2, dtype=np.uint16) % 288).reshape(rows, 2, 2)
        choice = {
            "index": 0,
            "text": "abc",
            "finish_reason": "length",
            "logprobs": {
                "tokens": [f"token_id:{t}" for t in out],
                "token_logprobs": [-0.5] * len(out),
            },
        }
        usage = {"prompt_tokens_details": {"cached_tokens": self.cached}}
        result = {"choices": [choice], "usage": usage, "nvext": {"routed_experts": encode(routes)}}
        if len(self.requests) <= self.drop:
            del result["nvext"]
        return httpx.Response(200, json=result)


def recorder(engine: Engine) -> record.Recorder:
    config = RolloutConfig(base_url="https://gw.invalid/v1", model="m")
    return record.Recorder(config, transport=httpx.MockTransport(engine))


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_wait(seconds: float) -> None:
        pass

    monkeypatch.setattr(record.asyncio, "sleep", no_wait)


def test_a_recorded_sample_has_exact_tokens_log_probs_routes_and_cached_tokens() -> None:
    engine = Engine(drop=1, cached=128)
    rec = recorder(engine)
    done = asyncio.run(rec.complete([1, 2, 3, 4], 3))
    assert done["response_ids"] == [7, 8, 9] and done["logprobs"] == [-0.5] * 3
    assert done["routes"].shape == (6, 2, 2) and done["cached_tokens"] == 128
    assert rec.retries == ["no routed experts"]  # that response was asked again
    # the request is cwtrain's sampler's, as a trainer sends it
    sent = engine.requests[-1]
    assert sent["prompt"] == [1, 2, 3, 4] and sent["return_tokens_as_token_ids"] is True
    assert sent["n"] == 1 and sent["stream"] is False and sent["temperature"] == 1.0
    assert sent["nvext"] == {"extra_fields": ["routed_experts"]} and sent["top_p"] == 1.0


def test_the_sampler_retries_busy_endpoints_itself() -> None:
    engine = Engine(unavailable=2)
    rec = recorder(engine)
    assert asyncio.run(rec.complete([1, 2], 3))["response_ids"] == [7, 8, 9]
    assert len(engine.requests) == 3 and rec.retries == []


def test_responses_that_never_carry_routes_fail() -> None:
    engine = Engine(drop=99)
    with pytest.raises(RolloutError, match="no routed experts"):
        asyncio.run(recorder(engine).complete([1, 2], 3))
    assert len(engine.requests) == record.ATTEMPTS


def test_sets_are_written_as_jsonl_and_routes_keyed_by_sample_id(tmp_path: Path) -> None:
    tok = Tokenizer()
    rows = [{"prompt": f"q{i}", "label": str(i)} for i in range(4)]

    async def both():
        rec = recorder(Engine())
        single = await record.record_single(rec, tok, rows, 0, 2)
        turns = await record.record_two_turn(rec, tok, rows, 2, 1, "again?", STOPS)
        return single, turns

    single, turns = asyncio.run(both())
    assert [s["id"] for s in turns.samples] == ["two_turn-0002-1", "two_turn-0002-2"]
    # turn 2 starts with turn 1's prompt and sampled ids
    first = turns.samples[0]["prompt_ids"] + turns.samples[0]["response_ids"]
    assert turns.samples[1]["prompt_ids"][: len(first)] == first
    stats = record.write_set(tmp_path, "single", single)
    assert stats["samples"] == 2 and stats["truncated"] == 2
    assert stats["skipped"] == 0 and stats["failed"] == 0
    lines = (tmp_path / "single.jsonl").read_text().splitlines()
    assert [json.loads(line)["label"] for line in lines] == ["0", "1"]
    assert "seconds" not in json.loads(lines[0])
    routes = np.load(tmp_path / "single.npz")
    assert sorted(routes.files) == ["single-0000", "single-0001"]


def test_a_two_turn_conversation_stays_within_the_length_limit_or_is_skipped() -> None:
    tok = Tokenizer()
    rows = [{"prompt": "q0"}]
    engine = Engine(fill=True)  # every answer uses its whole budget

    async def run(follow_up: str) -> record.SetResult:
        return await record.record_two_turn(recorder(engine), tok, rows, 0, 1, follow_up, STOPS)

    # A 1,200-token follow-up: turn 1 leaves room for it and for a turn-2 answer.
    result = asyncio.run(run("x" * 1200))
    assert len(result.samples) == 2 and result.skipped == 0
    for request in engine.requests:
        assert len(request["prompt"]) + request["max_tokens"] <= record.MAX_LEN
    assert engine.requests[-1]["max_tokens"] >= record.TURN2_MIN
    # A follow-up that leaves no room for turn 1 skips the conversation without a request.
    sent = len(engine.requests)
    result = asyncio.run(run("x" * 2000))
    assert result.samples == [] and result.skipped == 1 and len(engine.requests) == sent


def test_a_request_that_fails_for_good_does_not_discard_its_set(tmp_path: Path) -> None:
    tok = Tokenizer()
    rows = [{"prompt": f"q{i}"} for i in range(3)]
    bad = record.render(tok, record.messages_of(rows[1]))
    rec = recorder(Engine(fail=bad))
    result = asyncio.run(record.record_single(rec, tok, rows, 0, 3))
    assert [s["id"] for s in result.samples] == ["single-0000", "single-0002"]
    assert len(result.failures) == 1 and result.failures[0].startswith("single-0001: RolloutError")
    assert record.write_set(tmp_path, "single", result)["failed"] == 1


# --- router noise ---


def bf16(values) -> np.ndarray:
    return router_noise.f32_to_bf16(np.asarray(values, dtype=np.float32))


def test_bfloat16_rounding_is_to_nearest_with_ties_to_even() -> None:
    one = 0x3F80
    # 1 + 2**-8 lies halfway between 1 and the next bfloat16: ties go to the even one, 1.
    # 1 + 3 * 2**-8 lies halfway between the 1st and 2nd steps above 1: it goes to the 2nd.
    halfway = [1.0, 1 + 2**-8, 1 + 3 * 2**-8, 1 + 2**-8 + 2**-20, -2.5]
    assert bf16(halfway).tolist() == [one, one, one + 2, one + 1, 0xC020]
    exact = np.array([0.5, -3.0, 1.5, 0.0078125], dtype=np.float32)
    assert (router_noise.bf16_to_f32(bf16(exact)) == exact).all()


def test_noise_is_seeded_per_layer_and_has_the_requested_spread() -> None:
    bits = bf16(np.full((64, 256), 0.5))
    a = router_noise.noised(bits, seed=1, layer=3, sigma=0.01)
    assert (a == router_noise.noised(bits, seed=1, layer=3, sigma=0.01)).all()
    assert (a != router_noise.noised(bits, seed=1, layer=4, sigma=0.01)).any()
    assert (a != router_noise.noised(bits, seed=2, layer=3, sigma=0.01)).any()
    spread = (router_noise.bf16_to_f32(a) - 0.5).std()
    assert spread == pytest.approx(0.01, rel=0.1)


def write_shard(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header, offset, blobs = {}, 0, []
    for name, bits in tensors.items():
        blob = bits.astype(np.uint16).tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(bits.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    base = tmp_path / "v0"
    base.mkdir()
    rng = np.random.default_rng(0)
    weights = lambda *shape: bf16(rng.standard_normal(shape) * 0.02)  # noqa: E731
    shards = {
        "a.safetensors": {
            "model.layers.0.mlp.gate.weight": weights(8, 16),
            "model.layers.0.self_attn.q_proj.weight": weights(16, 16),
        },
        "b.safetensors": {"model.layers.1.mlp.up_proj.weight": weights(16, 16)},
        "c.safetensors": {"model.layers.2.mlp.gate.weight": weights(8, 16)},  # an MTP layer
    }
    weight_map = {}
    for shard, tensors in shards.items():
        write_shard(base / shard, tensors)
        weight_map |= dict.fromkeys(tensors, shard)
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (base / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 2}}))
    return base


def test_a_noisy_version_changes_only_the_decoder_routers(checkpoint: Path) -> None:
    from cwtrain.hotload.mirror import tensor_locations

    out = checkpoint.with_name("v1")
    rec = router_noise.make_version(checkpoint, out, seed=1, scale=0.05)
    assert list(rec["tensors"]) == ["model.layers.0.mlp.gate.weight"]
    entry = rec["tensors"]["model.layers.0.mlp.gate.weight"]
    assert entry["layer"] == 0 and entry["changed_elements"] > 0
    before, after = tensor_locations(checkpoint), tensor_locations(out)
    for name, loc in after.items():
        old = before[name].path.read_bytes()[before[name].offset :][: loc.nbytes]
        new = loc.path.read_bytes()[loc.offset :][: loc.nbytes]
        assert (old != new) == (name == "model.layers.0.mlp.gate.weight"), name
        if name == "model.layers.0.mlp.gate.weight":
            assert router_noise.sha256(np.frombuffer(new, np.uint16)) == entry["sha256_after"]
            assert router_noise.sha256(np.frombuffer(old, np.uint16)) == entry["sha256_before"]
    # shards without a decoder router are links to the base's, not copies
    assert (out / "b.safetensors").stat().st_ino == (checkpoint / "b.safetensors").stat().st_ino
    assert (out / "a.safetensors").stat().st_ino != (checkpoint / "a.safetensors").stat().st_ino


def test_main_writes_the_delta_and_the_record_next_to_the_version(checkpoint: Path) -> None:
    v1 = checkpoint.with_name("v1")
    args = ["--base", str(checkpoint), "--out", str(v1), "--seed", "1", "--scale", "0.05"]
    assert router_noise.main([*args, "--delta", str(checkpoint.with_name("v1-delta"))]) == 0
    rec = json.loads(v1.with_name("v1.noise.json").read_text())
    assert rec["base"] == "v0" and rec["version"] == "v1" and rec["delta"]["tensors_changed"] == 1
    index = json.loads(
        (checkpoint.with_name("v1-delta") / "model.safetensors.index.json").read_text()
    )
    assert index["metadata"]["base_version"] == "v0" and index["metadata"]["version"] == "v1"
    assert not (v1 / "v1.noise.json").exists()


def test_a_noise_delta_publishes_only_with_its_previous_identity(checkpoint: Path) -> None:
    v1, v2, delta = (checkpoint.with_name(n) for n in ("v1", "v2", "v2-delta"))
    noise = ["--seed", "1", "--scale", "0.05"]
    assert router_noise.main(["--base", str(checkpoint), "--out", str(v1), *noise]) == 0
    noise = ["--seed", "2", "--scale", "0.05", "--delta", str(delta)]
    assert router_noise.main(["--base", str(v1), "--out", str(v2), *noise]) == 0
    # cwtrain hotload publish --previous-identity v1, as reproduce.sh runs it
    info = inspect_snapshot(delta, previous_identity="v1")
    assert info.is_incremental and info.incremental.previous_snapshot_identity == "v1"
    # Without it, cwtrain reads the base from the index as a slime version number, which "v1" isn't.
    with pytest.raises(SnapshotError, match="base_version 'v1'"):
        inspect_snapshot(delta)
