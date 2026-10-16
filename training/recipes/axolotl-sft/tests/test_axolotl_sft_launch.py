# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Unit tests for the Axolotl SFT launcher (no Axolotl, GPUs or API needed)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cwtrain.errors import CWTrainError
from cwtrain.hotload import HotLoadState, PromptCachePolicy, SnapshotType, inspect_snapshot
from cwtrain.hotload.mirror import read_safetensors_header, tensor_locations
from cwtrain.integrations.launch import LaunchError

spec = importlib.util.spec_from_file_location(
    "axolotl_sft_launch", Path(__file__).resolve().parents[1] / "launch.py"
)
launch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launch
spec.loader.exec_module(launch)

BASE = {
    "HF_MODEL": "Qwen/Qwen3-4B",
    "MODEL_SLUG": "Qwen3-4B",
    "HF_REVISION": "1cfa9a7208912126459214e8b04321603b3df60c",
    "CWTRAIN_HOTLOAD_DEPLOYMENT_ID": "dep-1",
    "CWTRAIN_HOTLOAD_RUN_ID": "sft-run",
}
EMBED, NORM, HEAD = "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
EMBED_BYTES = bytes(range(16))  # BF16 [4, 2]


def save_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, data = {}, b""
    for name, (dtype, shape, raw) in tensors.items():
        offsets = [len(data), len(data) + len(raw)]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": offsets}
        data += raw
    body = json.dumps(header).encode()
    body += b" " * (-len(body) % 8)
    path.write_bytes(struct.pack("<Q", len(body)) + body + data)


def tokenizer(*added: str) -> str:
    tokens = ["<|endoftext|>", "<|im_start|>", *added]
    return json.dumps({"added_tokens": [{"id": i, "content": t} for i, t in enumerate(tokens)]})


def make_base(root: Path) -> Path:
    """A sharded base checkpoint with a tied head, as Qwen3-4B is saved."""
    root.mkdir(parents=True)
    save_safetensors(root / SHARDS[0], {EMBED: ("BF16", [4, 2], EMBED_BYTES)})
    save_safetensors(root / SHARDS[1], {NORM: ("BF16", [2], b"\x01\x02\x03\x04")})
    index = {"weight_map": {EMBED: SHARDS[0], NORM: SHARDS[1]}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    (root / "config.json").write_text('{"rope_theta": 1000000}')
    (root / "generation_config.json").write_text("{}")
    (root / "tokenizer.json").write_text(tokenizer())
    (root / "tokenizer_config.json").write_text("{}")
    return root


def make_result(root: Path, *, head: bytes = EMBED_BYTES, norm: str = "BF16", added=()) -> Path:
    """What the FSDP2 final save writes: one file, the tied head saved again, run files."""
    root.mkdir(parents=True)
    tensors = {EMBED: ("BF16", [4, 2], EMBED_BYTES), NORM: (norm, [2], b"\x05\x06\x07\x08")}
    if head:
        tensors[HEAD] = ("BF16", [4, 2], head)
    save_safetensors(root / "model.safetensors", tensors)
    (root / "config.json").write_text('{"rope_parameters": {"rope_theta": 1000000}}')
    (root / "tokenizer.json").write_text(tokenizer(*added))
    for name in ("training_args.bin", "debug.log", "README.md"):
        (root / name).write_text("run file")
    return root


def test_settings_come_from_the_environment() -> None:
    extra = """--chat-template-kwargs '{"enable_thinking": false}'\n--learning-rate 2e-5"""
    job = launch.SftJob.from_env({**BASE, "WORK": "/data", "EXTRA_TRAIN_ARGS": extra})
    assert job.identity == "sft-run-v000030"  # <RUN_ID>-v<MAX_STEPS>, as cleanup expects
    assert job.model_dir == Path("/data/models/Qwen3-4B") and job.output_dir == Path("/data/out")
    assert job.config == Path("/opt/cwtrain/sft_coreweave.yaml")
    assert job.extra_train_args == (
        "--chat-template-kwargs", '{"enable_thinking": false}', "--learning-rate", "2e-5",
    )  # fmt: skip
    assert job.policy is PromptCachePolicy.RESET_ALL and job.timeout == 1800
    assert job.wait and not job.verify
    env = {**BASE, "CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY": "preserve", "MAX_STEPS": "100"}
    job = launch.SftJob.from_env(env)
    assert job.policy is PromptCachePolicy.PRESERVE and job.identity == "sft-run-v000100"


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"EXTRA_TRAIN_ARGS": "--max-steps 10"}, "set MAX_STEPS"),
        ({"EXTRA_TRAIN_ARGS": "--max_steps=10"}, "set MAX_STEPS"),
        ({"EXTRA_TRAIN_ARGS": "--x 'unterminated"}, "EXTRA_TRAIN_ARGS"),
        ({"CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY": "keep"}, "preserve or reset_all"),
        ({"CWTRAIN_HOTLOAD_TIMEOUT": "soon"}, "CWTRAIN_HOTLOAD_TIMEOUT"),
        ({"CWTRAIN_HOTLOAD_WAIT": "maybe"}, "CWTRAIN_HOTLOAD_WAIT"),
        ({"CWTRAIN_HOTLOAD_RUN_ID": "a/b"}, "single path segment"),
        ({"CWTRAIN_HOTLOAD_DEPLOYMENT_ID": " "}, "CWTRAIN_HOTLOAD_DEPLOYMENT_ID is required"),
    ],
)
def test_bad_settings_stop_the_launcher_before_training(settings, error) -> None:
    with pytest.raises(CWTrainError, match=error):
        launch.SftJob.from_env({**BASE, **settings})


def test_the_train_command_starts_one_process_per_gpu() -> None:
    env = {**BASE, "WORK": "/w", "TRAINER_GPUS": "2", "EXTRA_TRAIN_ARGS": "--adapter lora"}
    job = launch.SftJob.from_env(env)
    assert launch.train_command(job) == [
        "axolotl", "train", "/opt/cwtrain/sft_coreweave.yaml",
        "--base-model", "/w/models/Qwen3-4B", "--output-dir", "/w/out", "--max-steps", "30",
        "--adapter", "lora",
        "--launcher", "torchrun", "--", "--nproc_per_node", "2",
    ]  # fmt: skip
    merge = launch.merge_command(job)
    assert merge[:3] == ["axolotl", "merge-lora", "/opt/cwtrain/sft_coreweave.yaml"]
    assert merge[-4:] == ["--adapter", "lora", "--lora-model-dir", "/w/out"]


def test_the_base_model_comes_from_the_bucket_or_a_pinned_revision(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(launch, "run", lambda args, **kw: calls.append([str(a) for a in args]))
    monkeypatch.setattr(launch, "stage_from_s3", lambda uri, dest: calls.append([uri, str(dest)]))
    job = launch.SftJob.from_env({**BASE, "WORK": str(tmp_path)})
    launch.stage_base(job)
    assert calls[0][:5] == ["hf", "download", "Qwen/Qwen3-4B", "--revision", BASE["HF_REVISION"]]
    calls.clear()
    launch.stage_base(dataclasses.replace(job, base_s3_uri="s3://b/hotload/r-v000030"))
    assert calls == [["s3://b/hotload/r-v000030", str(job.model_dir)]]
    with pytest.raises(LaunchError, match="HF_REVISION"):
        launch.stage_base(dataclasses.replace(job, hf_revision="main"))


def test_stage_takes_the_weights_from_the_result_and_the_rest_from_the_base(tmp_path) -> None:
    base, result = make_base(tmp_path / "base"), make_result(tmp_path / "out")
    staged = launch.stage(result, base, tmp_path / "staged")
    assert sorted(path.name for path in staged.iterdir()) == [
        "config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json",
    ]  # fmt: skip
    assert (staged / "config.json").read_text() == (base / "config.json").read_text()
    assert (staged / "model.safetensors").samefile(result / "model.safetensors")  # a hard link


def test_served_names_leave_out_a_tied_head_equal_to_the_embedding(tmp_path, capsys) -> None:
    base, result = make_base(tmp_path / "base"), make_result(tmp_path / "out")
    assert launch.served_names(result, base, tokenizer_dir=result) == {EMBED, NORM}
    assert f"left out {HEAD}: a copy of {EMBED}" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("result", "error"),
    [
        ({"head": bytes(16)}, "extra lm_head.weight"),
        ({"norm": "F16"}, r"model.norm.weight is F16 \[2\], the base's BF16 \[2\]"),
        ({"added": ("<|new|>",)}, r"added the tokens \['<\|new\|>'\]"),
    ],
    ids=["a different head", "another dtype", "an added token"],
)
def test_served_names_refuse_another_model(tmp_path, result, error) -> None:
    base, out = make_base(tmp_path / "base"), make_result(tmp_path / "out", **result)
    with pytest.raises(LaunchError, match=error):
        launch.served_names(out, base, tokenizer_dir=out)


def test_served_names_refuse_a_missing_tensor_and_missing_weights(tmp_path) -> None:
    base = make_base(tmp_path / "base")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(LaunchError, match="no safetensors weights.*FULL_STATE_DICT"):
        launch.served_names(out, base, tokenizer_dir=out)
    save_safetensors(out / "model.safetensors", {EMBED: ("BF16", [4, 2], EMBED_BYTES)})
    with pytest.raises(LaunchError, match="missing model.norm.weight"):
        launch.served_names(out, base, tokenizer_dir=out)


class FakeLoader:
    """The HotLoader calls the launcher makes, recorded; time passes as each call says."""

    def __init__(self, clock: list[float], *, hot_loaded=(), fail=None, takes=None) -> None:
        self.clock, self.hot_loaded, self.fail = clock, set(hot_loaded), fail
        self.takes = takes or {}
        self.calls: list[tuple] = []

    def _call(self, *call) -> None:
        self.calls.append(call)
        self.clock[0] += self.takes.get(call[0], 0)
        if call[0] == self.fail:
            raise CWTrainError("deployment dep-1 is UPDATING, not READY")

    def completed_hot_load(self, identity):
        self._call("completed", identity)
        return SimpleNamespace(id="old") if identity in self.hot_loaded else None

    def upload(self, snapshot, identity):
        self._call("upload", Path(snapshot), identity)
        return "info"

    def wait_for_deployment_ready(self, *, timeout):
        self._call("ready", timeout)

    def trigger(self, identity, info, *, prompt_cache_policy, timeout):
        self._call("trigger", identity, info, prompt_cache_policy, timeout)
        return SimpleNamespace(id="hl-1", state=HotLoadState.PENDING)

    def wait(self, hot_load_id, *, timeout):
        self._call("wait", hot_load_id, timeout)
        return SimpleNamespace(id=hot_load_id, state=HotLoadState.COMPLETED)

    def verify_served(self, hot_load_id, identity):
        self._call("verify", hot_load_id, identity)


@pytest.fixture
def clock(monkeypatch) -> list[float]:
    now = [0.0]
    monkeypatch.setattr(launch.time, "monotonic", lambda: now[0])
    return now


def test_publish_starts_the_budget_after_the_upload(tmp_path, clock) -> None:
    job = launch.SftJob.from_env({**BASE, "CWTRAIN_HOTLOAD_VERIFY": "1"})
    loader = FakeLoader(clock, takes={"upload": 5000, "ready": 100, "trigger": 100})
    assert launch.publish(loader, tmp_path, job).state is HotLoadState.COMPLETED
    assert loader.calls == [
        ("upload", tmp_path, "sft-run-v000030"),
        ("ready", 1800),  # the upload took longer than the budget, which starts after it
        ("trigger", "sft-run-v000030", "info", PromptCachePolicy.RESET_ALL, 1700),
        ("wait", "hl-1", 1600),
        ("verify", "hl-1", "sft-run-v000030"),
    ]


def test_publish_says_how_to_hot_load_the_uploaded_snapshot(tmp_path, clock, capsys) -> None:
    job = launch.SftJob.from_env(BASE)
    loader = FakeLoader(clock, fail="ready")
    with pytest.raises(CWTrainError, match="not READY"):
        launch.publish(loader, tmp_path, job)
    out = capsys.readouterr().out
    assert "Once `cwtrain deployment get dep-1` shows READY" in out
    command = "cwtrain hotload trigger --deployment dep-1 --identity sft-run-v000030"
    assert f"{command} --reset-prompt-cache" in out
    preserve = dataclasses.replace(job, policy=PromptCachePolicy.PRESERVE)
    assert launch.recovery_command(preserve) == command


STUB = """import json, os, shutil, sys
args = sys.argv[1:]
with open(os.environ["STUB_CALLS"], "a") as f:
    f.write(json.dumps({"args": args, "env": sorted(os.environ), "cwd": os.getcwd()}) + "\\n")
out = args[args.index("--output-dir") + 1]
if args[0] == "train":
    shutil.copytree(os.environ["STUB_TRAIN_OUTPUT"], out, dirs_exist_ok=True)
else:
    shutil.copytree(os.environ["STUB_MERGED"], os.path.join(out, "merged"))
"""


@pytest.fixture
def stub_axolotl(tmp_path, monkeypatch) -> Path:
    """An ``axolotl`` on PATH that records its calls and writes the given outputs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    (bin_dir / "axolotl").write_text(f'#!/bin/sh\nexec "{sys.executable}" "{stub}" "$@"\n')
    (bin_dir / "axolotl").chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", f"{bin_dir}:{launch.os.environ['PATH']}")
    monkeypatch.setenv("STUB_CALLS", str(calls))
    for name in ("COREWEAVE_API_TOKEN", "CW_API_TOKEN", "CWTRAIN_API_HEADERS"):
        monkeypatch.setenv(name, "secret")
    return calls


def run_main(tmp_path, monkeypatch, clock, **loader_args) -> FakeLoader:
    base = make_base(tmp_path / "base")
    loader = FakeLoader(clock, **loader_args)
    monkeypatch.setattr(launch, "api_client", lambda env: "client")
    monkeypatch.setattr(launch, "HotLoader", lambda client, deployment_id: loader)
    monkeypatch.setattr(launch, "stage_from_s3", lambda uri, dest: shutil.copytree(base, dest))
    launch.main({**BASE, "WORK": str(tmp_path / "work"), "BASE_S3_URI": "s3://b/models/Qwen3-4B"})
    return loader


def test_main_trains_without_the_api_credentials_and_publishes_the_base_tensors(
    tmp_path, monkeypatch, clock, stub_axolotl
) -> None:
    monkeypatch.setenv("STUB_TRAIN_OUTPUT", str(make_result(tmp_path / "result")))
    loader = run_main(tmp_path, monkeypatch, clock)

    (train,) = [json.loads(line) for line in stub_axolotl.read_text().splitlines()]
    assert train["args"][:2] == ["train", "/opt/cwtrain/sft_coreweave.yaml"]
    assert not {"COREWEAVE_API_TOKEN", "CW_API_TOKEN", "CWTRAIN_API_HEADERS"} & set(train["env"])
    assert "STUB_CALLS" in train["env"] and train["cwd"] == str(tmp_path / "work")
    assert [call[0] for call in loader.calls] == ["completed", "upload", "ready", "trigger", "wait"]

    snapshot = loader.calls[1][1]
    assert snapshot == tmp_path / "work" / "sft-run-v000030"
    info = inspect_snapshot(snapshot)
    assert info.type is SnapshotType.FULL
    assert sorted(path.name for path in info.files) == [
        "config.json", "generation_config.json", "model.safetensors",
        "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    ]  # fmt: skip
    assert json.loads(info.index_bytes)["metadata"]["checksum_format"] == "adler32"
    assert set(tensor_locations(snapshot)) == {EMBED, NORM}  # 2 of the 3 tensors
    header, _ = read_safetensors_header(snapshot / "model.safetensors")
    assert {EMBED, NORM} <= set(header["__metadata__"])  # a checksum per tensor
    assert (snapshot / "config.json").read_text() == '{"rope_theta": 1000000}'  # the base's


def lora_run(tmp_path, monkeypatch, *, changed: bool) -> None:
    """A LoRA run's output, and a merge in the base's layout that changed the norm or nothing."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    save_safetensors(adapter / "adapter_model.safetensors", {"lora_A": ("BF16", [1], b"\0\0")})
    (adapter / "tokenizer.json").write_text(tokenizer())
    merged = make_base(tmp_path / "merged")  # merge-lora writes the base's layout
    if changed:
        save_safetensors(merged / SHARDS[1], {NORM: ("BF16", [2], b"\x05\x06\x07\x08")})
    monkeypatch.setenv("STUB_TRAIN_OUTPUT", str(adapter))
    monkeypatch.setenv("STUB_MERGED", str(merged))


def test_main_merges_a_lora_adapter_and_publishes_the_merged_model(
    tmp_path, monkeypatch, clock, stub_axolotl
) -> None:
    lora_run(tmp_path, monkeypatch, changed=True)
    loader = run_main(tmp_path, monkeypatch, clock)

    calls = [json.loads(line)["args"] for line in stub_axolotl.read_text().splitlines()]
    assert [call[0] for call in calls] == ["train", "merge-lora"]
    assert calls[1][-2:] == ["--lora-model-dir", str(tmp_path / "work" / "out")]
    snapshot = loader.calls[1][1]
    assert set(tensor_locations(snapshot)) == {EMBED, NORM}
    assert (snapshot / SHARDS[1]).is_file()


def test_main_stops_when_the_merge_left_the_base_model_unchanged(
    tmp_path, monkeypatch, clock, stub_axolotl
) -> None:
    lora_run(tmp_path, monkeypatch, changed=False)
    with pytest.raises(LaunchError, match="applied no LoRA weight"):
        run_main(tmp_path, monkeypatch, clock)
    assert not (tmp_path / "work" / "sft-run-v000030").exists()  # nothing to upload


def test_main_stops_before_training_when_the_identity_is_already_hot_loaded(
    tmp_path, monkeypatch, clock, stub_axolotl
) -> None:
    with pytest.raises(LaunchError, match="sft-run-v000030 is already hot-loaded"):
        run_main(tmp_path, monkeypatch, clock, hot_loaded=["sft-run-v000030"])
    assert not stub_axolotl.exists() and not (tmp_path / "work").exists()
