# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from cwtrain.errors import CWTrainError
from cwtrain.hotload import HotLoadState, PromptCachePolicy, SnapshotInfo, SnapshotType
from cwtrain.hotload.delta import checkpoint_checksums
from cwtrain.hotload.errors import HotLoadVerificationError, SnapshotError
from cwtrain.hotload.mirror import MirrorError, tensor_locations
from cwtrain.integrations import slime
from cwtrain.integrations.slime import HookConfig, SlimePublisher, default_run_id, is_rank_zero
from tests.conftest import (
    DEPLOYMENT_ID,
    make_weights,
    perturb,
    write_delta_checkpoint,
    write_full_checkpoint,
    write_real_checkpoint,
    write_real_delta,
)

RUN = "qwen3-grpo-20260922"


class StubHotLoad:
    def __init__(self, identity: str, state: HotLoadState = HotLoadState.COMPLETED) -> None:
        self.id = f"hl-{identity}"
        self.identity = identity
        self.state = state

    @property
    def completed(self) -> bool:
        return self.state is HotLoadState.COMPLETED


class StubLoader:
    """Records what the hook asks a HotLoader to do."""

    def __init__(self, completed: set[str] | None = None) -> None:
        self.completed_identities = set(completed or ())
        self.uploads: list[tuple[SnapshotInfo, str, int]] = []
        self.triggers: list[tuple[str, SnapshotInfo, PromptCachePolicy]] = []
        self.waits: list[str] = []
        self.verified: list[tuple[str, str]] = []
        self.manifests: list[tuple[str, dict]] = []
        self.recorded: dict[str, dict] = {}  # manifests stored before this run
        self.verify_error: Exception | None = None

    def wait_for_deployment_ready(self, timeout=None, poll_interval=None):
        pass

    def completed_hot_load(self, identity: str):
        return StubHotLoad(identity) if identity in self.completed_identities else None

    def upload(self, info, identity, *, previous_identity=None, workers=8):
        self.uploads.append((info, identity, workers))
        return info

    def trigger(
        self, identity, info=None, *, prompt_cache_policy=PromptCachePolicy.UNSPECIFIED, **_
    ):
        self.triggers.append((identity, info, prompt_cache_policy))
        return StubHotLoad(identity, HotLoadState.PENDING)

    def wait(self, hot_load_id, *, timeout=None, on_progress=None, **_):
        self.waits.append(hot_load_id)
        identity = hot_load_id.removeprefix("hl-")
        if on_progress:
            on_progress(SimpleNamespace(ready=2, total=2))
        self.completed_identities.add(identity)
        return StubHotLoad(identity)

    def verify_served(self, hot_load_id, identity):
        self.verified.append((hot_load_id, identity))
        if self.verify_error is not None:
            raise self.verify_error
        return [SimpleNamespace(name=f"r{i}", current_snapshot_identity=identity) for i in range(2)]

    def read_manifest(self, identity):
        return self.recorded.get(identity)

    def upload_manifest(self, identity, manifest):
        self.manifests.append((identity, manifest))
        return f"s3://ckpts/runs/_manifests/{identity}.json"


@pytest.fixture(autouse=True)
def _reset_module_state():
    slime.reset()
    yield
    slime.reset()


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """The settings every hook test needs, with ``FULL_EVERY=0`` instead of the default (20).

    ``0`` turns the weight mirror off. Tests of the mirror, chain cuts or defaults set or unset
    ``CWTRAIN_HOTLOAD_FULL_EVERY`` themselves.
    """
    values = {
        "CWTRAIN_HOTLOAD_DEPLOYMENT_ID": DEPLOYMENT_ID,
        "CWTRAIN_HOTLOAD_RUN_ID": RUN,
        # most tests use synthetic checkpoints, which the weight mirror rejects
        "CWTRAIN_HOTLOAD_FULL_EVERY": "0",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("CWTRAIN_HOTLOAD_VERIFY", raising=False)  # tests rely on its default
    return values


@pytest.fixture
def args(tmp_path: Path) -> SimpleNamespace:
    hf = write_full_checkpoint(tmp_path / "hf" / "Qwen3-8B")
    delta_dir = tmp_path / "deltas"
    delta_dir.mkdir()
    return SimpleNamespace(
        hf_checkpoint=str(hf),
        update_weight_disk_dir=str(delta_dir),
        update_weight_mode="delta",
        rank=0,
    )


def install_stub(monkeypatch: pytest.MonkeyPatch, stub: StubLoader) -> None:
    monkeypatch.setattr(
        slime,
        "SlimePublisher",
        lambda config, loader=None: SlimePublisher(config, loader=stub),
    )


# ------------------------------------------------------------------------------- config


def test_config_from_env_defaults(env, monkeypatch) -> None:
    monkeypatch.delenv("CWTRAIN_HOTLOAD_FULL_EVERY")
    cfg = HookConfig.from_env()
    assert cfg.full_every == 20 and cfg.mirror_enabled
    assert cfg.deployment_id == DEPLOYMENT_ID and cfg.run_id == RUN
    assert cfg.wait and cfg.publish_baseline and cfg.timeout == 1800.0
    assert cfg.prompt_cache_policy is PromptCachePolicy.UNSPECIFIED
    assert cfg.identity(7) == f"{RUN}-v000007"


def test_config_from_env_overrides(env, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_WAIT", "0")
    monkeypatch.setenv("CWTRAIN_HOTLOAD_TIMEOUT", "60")
    monkeypatch.setenv("CWTRAIN_HOTLOAD_PUBLISH_BASELINE", "false")
    monkeypatch.setenv("CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY", "reset_all")
    monkeypatch.setenv("COREWEAVE_API_URL", "https://api.example.test")
    monkeypatch.setenv("CWTRAIN_API_HEADERS", json.dumps({"X-Test-Header": "value"}))
    monkeypatch.setenv("CWTRAIN_API_INSECURE", "1")
    cfg = HookConfig.from_env()
    assert not cfg.wait and cfg.timeout == 60 and not cfg.publish_baseline
    assert cfg.prompt_cache_policy is PromptCachePolicy.RESET_ALL
    assert cfg.api_base_url == "https://api.example.test"
    assert cfg.api_headers == {"X-Test-Header": "value"} and cfg.api_insecure


def test_config_rejects_bad_values(env, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY", "sometimes")
    with pytest.raises(CWTrainError, match="PROMPT_CACHE_POLICY"):
        HookConfig.from_env()
    monkeypatch.delenv("CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY")
    monkeypatch.setenv("CWTRAIN_HOTLOAD_WAIT", "maybe")
    with pytest.raises(CWTrainError, match="boolean"):
        HookConfig.from_env()
    monkeypatch.delenv("CWTRAIN_HOTLOAD_WAIT")
    monkeypatch.setenv("CWTRAIN_API_HEADERS", "[1,2]")
    with pytest.raises(CWTrainError, match="JSON object"):
        HookConfig.from_env()


def test_config_requires_deployment_and_valid_run_id(monkeypatch) -> None:
    monkeypatch.delenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", raising=False)
    with pytest.raises(CWTrainError, match="DEPLOYMENT_ID"):
        HookConfig.from_env(default_run_id="x")
    with pytest.raises(SnapshotError, match="single path segment"):
        HookConfig(deployment_id=DEPLOYMENT_ID, run_id="bad/run")
    with pytest.raises(SnapshotError):
        HookConfig(deployment_id=DEPLOYMENT_ID, run_id="r" * 125)  # + "-v000000" > 128


def test_default_run_id_is_dir_name_plus_timestamp() -> None:
    args = SimpleNamespace(update_weight_disk_dir="/shared/fs/my run:1")
    rid = default_run_id(args, now=datetime(2026, 9, 22, 12, 0, tzinfo=UTC))
    assert rid == "my-run-1-20260922T120000Z"


def test_is_rank_zero_prefers_args_then_env(monkeypatch) -> None:
    assert is_rank_zero(SimpleNamespace(rank=0))
    assert not is_rank_zero(SimpleNamespace(rank=3))
    monkeypatch.setenv("RANK", "2")
    assert not is_rank_zero(SimpleNamespace())
    monkeypatch.setenv("RANK", "0")
    assert is_rank_zero(SimpleNamespace())


# --------------------------------------------------------------------------------- hook


def test_baseline_call_publishes_hf_checkpoint_as_full_v0(env, args, monkeypatch) -> None:
    stub = StubLoader()
    install_stub(monkeypatch, stub)

    metrics = slime.hook(args, args.update_weight_disk_dir, [])

    assert [u[1] for u in stub.uploads] == [f"{RUN}-v000000"]
    info = stub.uploads[0][0]
    assert info.type is SnapshotType.FULL and info.path == Path(args.hf_checkpoint)
    assert stub.triggers[0][0] == f"{RUN}-v000000"
    assert stub.waits == [f"hl-{RUN}-v000000"]
    assert metrics is not None
    assert metrics["hotload/completed"] == 1.0 and metrics["hotload/incremental"] == 0.0
    assert metrics["hotload/replicas_ready"] == 2.0


def test_baseline_skipped_when_already_completed(env, args, monkeypatch) -> None:
    stub = StubLoader(completed={f"{RUN}-v000000"})
    install_stub(monkeypatch, stub)
    metrics = slime.hook(args, args.update_weight_disk_dir, [])
    assert stub.uploads == [] and stub.triggers == []
    assert metrics == {"hotload/baseline_skipped": 1.0}


def test_baseline_disabled(env, args, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_PUBLISH_BASELINE", "0")
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    assert slime.hook(args, args.update_weight_disk_dir, []) is None
    assert stub.uploads == []

    # the pre-published baseline is still the chain root: an empty v1 is skipped and the first
    # real delta, v2, must chain onto v000000 rather than the never-uploaded v000001
    deltas = Path(args.update_weight_disk_dir)
    empty = deltas / "weight_v000001"
    empty.mkdir()
    (empty / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": "000001",
                    "base_version": "000000",
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                },
                "weight_map": {},
            }
        )
    )
    assert slime.hook(args, str(empty), [])["hotload/skipped_empty"] == 1.0
    v2 = write_delta_checkpoint(deltas / "weight_v000002", version=2)
    slime.hook(args, str(v2), [])
    info, identity, _ = stub.uploads[-1]
    assert identity == f"{RUN}-v000002"
    assert info.incremental is not None
    assert info.incremental.previous_snapshot_identity == f"{RUN}-v000000"


def test_baseline_requires_hf_checkpoint(env, args, monkeypatch) -> None:
    install_stub(monkeypatch, StubLoader())
    args.hf_checkpoint = None
    with pytest.raises(CWTrainError, match="hf_checkpoint"):
        slime.hook(args, args.update_weight_disk_dir, [])


def test_delta_version_publishes_incremental_with_run_prefixed_base(env, args, monkeypatch) -> None:
    stub = StubLoader(completed={f"{RUN}-v000000"})
    install_stub(monkeypatch, stub)
    v1 = write_delta_checkpoint(Path(args.update_weight_disk_dir) / "weight_v000001", version=1)

    metrics = slime.hook(args, str(v1), [])

    ((info, identity, _workers),) = stub.uploads
    assert identity == f"{RUN}-v000001"
    assert info.type is SnapshotType.INCREMENTAL
    assert info.incremental is not None
    assert info.incremental.previous_snapshot_identity == f"{RUN}-v000000"
    assert metrics["hotload/incremental"] == 1.0 and metrics["hotload/completed"] == 1.0


def test_full_mode_version_publishes_full_snapshot(env, args, monkeypatch) -> None:
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    v3 = write_full_checkpoint(Path(args.update_weight_disk_dir) / "weight_v000003")
    slime.hook(args, str(v3), [])
    ((info, identity, _),) = stub.uploads
    assert identity == f"{RUN}-v000003" and info.type is SnapshotType.FULL


def test_no_wait_returns_after_trigger(env, args, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_WAIT", "0")
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    v1 = write_delta_checkpoint(Path(args.update_weight_disk_dir) / "weight_v000001", version=1)
    metrics = slime.hook(args, str(v1), [])
    assert stub.waits == []
    assert metrics["hotload/completed"] == 0.0 and "hotload/replicas_ready" not in metrics


def test_prompt_cache_policy_forwarded(env, args, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY", "reset_all")
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    v1 = write_full_checkpoint(Path(args.update_weight_disk_dir) / "weight_v000001")
    slime.hook(args, str(v1))
    assert stub.triggers[0][2] is PromptCachePolicy.RESET_ALL


def test_non_zero_rank_is_a_noop(env, args, monkeypatch) -> None:
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    args.rank = 1
    assert slime.hook(args, args.update_weight_disk_dir, []) is None
    assert stub.uploads == []


def test_publisher_is_reused_across_calls(env, args, monkeypatch) -> None:
    created: list[SlimePublisher] = []
    stub = StubLoader()

    def factory(config, loader=None):
        publisher = SlimePublisher(config, loader=stub)
        created.append(publisher)
        return publisher

    monkeypatch.setattr(slime, "SlimePublisher", factory)
    slime.hook(args, args.update_weight_disk_dir, [])
    v1 = write_delta_checkpoint(Path(args.update_weight_disk_dir) / "weight_v000001", version=1)
    slime.hook(args, str(v1), [])
    assert len(created) == 1
    assert [u[1] for u in stub.uploads] == [f"{RUN}-v000000", f"{RUN}-v000001"]


def test_default_run_id_used_once_per_process(monkeypatch, args) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_DEPLOYMENT_ID", DEPLOYMENT_ID)
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "0")  # synthetic checkpoints, no mirror
    monkeypatch.delenv("CWTRAIN_HOTLOAD_RUN_ID", raising=False)
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    slime.hook(args, args.update_weight_disk_dir, [])
    v1 = write_delta_checkpoint(Path(args.update_weight_disk_dir) / "weight_v000001", version=1)
    slime.hook(args, str(v1), [])
    prefixes = {identity.rsplit("-v", 1)[0] for _, identity, _ in stub.uploads}
    assert len(prefixes) == 1
    assert next(iter(prefixes)).startswith("deltas-")


def test_unsupported_delta_checksum_fails_loudly(env, args, monkeypatch) -> None:
    install_stub(monkeypatch, StubLoader())
    bad = write_delta_checkpoint(
        Path(args.update_weight_disk_dir) / "weight_v000001", version=1, checksum="xxh3-128"
    )
    with pytest.raises(SnapshotError, match="adler32"):
        slime.hook(args, str(bad), [])


def test_empty_delta_is_skipped_and_next_delta_chains_to_last_published(
    env, args, monkeypatch
) -> None:
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    slime.hook(args, args.update_weight_disk_dir, [])  # baseline -> v000000 published
    deltas = Path(args.update_weight_disk_dir)
    # v1: index only, no shards (a sync that changed nothing)
    empty = deltas / "weight_v000001"
    empty.mkdir()
    (empty / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": "000001",
                    "base_version": "000000",
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                },
                "weight_map": {},
            }
        )
    )
    metrics = slime.hook(args, str(empty), [])
    assert metrics == {"hotload/skipped_empty": 1.0, "hotload/incremental": 1.0}
    assert [u[1] for u in stub.uploads] == [f"{RUN}-v000000"]
    # v2 declares base 000001, which was never hot-loaded; it must chain onto v000000.
    v2 = write_delta_checkpoint(deltas / "weight_v000002", version=2)
    slime.hook(args, str(v2), [])
    info, identity, _ = stub.uploads[-1]
    assert identity == f"{RUN}-v000002"
    assert info.incremental is not None
    assert info.incremental.previous_snapshot_identity == f"{RUN}-v000000"


def test_explicit_run_id_does_not_warn_about_default(env, args, monkeypatch, caplog) -> None:
    install_stub(monkeypatch, StubLoader())
    with caplog.at_level("WARNING"):
        slime.hook(args, args.update_weight_disk_dir, [])
    assert "RUN_ID is unset" not in caplog.text


# ------------------------------------------------------------- verification & chain cuts


def test_verify_rechecks_every_replica_when_on(env, args, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_VERIFY", "1")
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    metrics = slime.hook(args, args.update_weight_disk_dir, [])
    assert stub.verified == [(f"hl-{RUN}-v000000", f"{RUN}-v000000")]
    assert metrics["hotload/replicas_verified"] == 2.0


def test_verification_failure_stops_the_run(env, args, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_VERIFY", "1")
    stub = StubLoader()
    stub.verify_error = HotLoadVerificationError(
        StubHotLoad("x"), [], ["r1: READY on 'stale-v000000'"]
    )
    install_stub(monkeypatch, stub)
    with pytest.raises(HotLoadVerificationError, match="stale"):
        slime.hook(args, args.update_weight_disk_dir, [])


@pytest.mark.parametrize("value", [None, "0"])
def test_replicas_are_not_rechecked_by_default(env, args, monkeypatch, value) -> None:
    # a COMPLETED hot-load already means every replica serves the new identity
    if value is not None:
        monkeypatch.setenv("CWTRAIN_HOTLOAD_VERIFY", value)
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    metrics = slime.hook(args, args.update_weight_disk_dir, [])
    assert stub.verified == [] and "hotload/replicas_verified" not in metrics


@pytest.mark.parametrize("raw,expected", [("", 20), ("  ", 20), ("1", 1), ("64", 64)])
def test_full_every_defaults_to_20_and_allows_up_to_64(env, monkeypatch, raw, expected) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", raw)
    cfg = HookConfig.from_env()
    assert cfg.full_every == expected and cfg.mirror_enabled


def test_full_every_above_64_is_rejected(env, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "65")
    with pytest.raises(CWTrainError, match="at most 63 deltas"):
        HookConfig.from_env()


def test_full_every_zero_warns_once_per_run(env, args, monkeypatch, caplog) -> None:
    install_stub(monkeypatch, StubLoader())
    with caplog.at_level(logging.WARNING, logger="cwtrain.integrations.slime"):
        cfg = HookConfig.from_env()
        assert "never cuts" not in caplog.text  # reading the settings alone does not warn
        slime._publisher(args)
        slime._publisher(args)  # every hook call re-reads the settings
    assert cfg.full_every == 0 and not cfg.mirror_enabled
    assert caplog.text.count("never cuts the delta chain") == 1
    assert "past 63 deltas" in caplog.text and "past 128 the API" in caplog.text


def test_full_every_parse_error_names_the_value(env, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "20.0")
    with pytest.raises(CWTrainError, match=r"must be an integer, got '20\.0'"):
        HookConfig.from_env()


def test_turning_the_mirror_off_needs_full_every_zero(env, monkeypatch) -> None:
    monkeypatch.delenv("CWTRAIN_HOTLOAD_FULL_EVERY")
    monkeypatch.setenv("CWTRAIN_HOTLOAD_MIRROR", "0")
    with pytest.raises(CWTrainError, match="FULL_EVERY=0"):
        HookConfig.from_env()


def test_full_every_rejects_bad_values(env, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "-1")
    with pytest.raises(CWTrainError, match="FULL_EVERY"):
        HookConfig.from_env()
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "10")
    monkeypatch.setenv("CWTRAIN_HOTLOAD_MIRROR", "0")
    with pytest.raises(CWTrainError, match="needs the mirror"):
        HookConfig.from_env()
    monkeypatch.delenv("CWTRAIN_HOTLOAD_MIRROR")
    cfg = HookConfig.from_env()
    assert cfg.full_every == 10 and cfg.mirror_enabled and not cfg.verify


@pytest.fixture
def real_args(tmp_path: Path) -> SimpleNamespace:
    weights = make_weights()
    hf = write_real_checkpoint(tmp_path / "hf" / "Qwen3-4B", weights)
    delta_dir = tmp_path / "deltas"
    delta_dir.mkdir()
    ns = SimpleNamespace(
        hf_checkpoint=str(hf), update_weight_disk_dir=str(delta_dir), rank=0, weights=weights
    )
    return ns


def read_all(root: Path) -> dict[str, bytes]:
    out = {}
    for name, loc in tensor_locations(root).items():
        with loc.path.open("rb") as f:
            f.seek(loc.offset)
            out[name] = f.read(loc.nbytes)
    return out


def test_full_every_cuts_the_chain_with_a_verified_full_snapshot(
    env, real_args, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "2")
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    args = real_args
    deltas = Path(args.update_weight_disk_dir)
    state_dir = tmp_path / "deltas-hotload"

    # baseline: mirror created from --hf-checkpoint, manifest for v0, full snapshot published
    m0 = slime.hook(args, str(deltas), [])
    assert (state_dir / "mirror" / "mirror-state.json").is_file()
    assert m0["hotload/mirror_seconds"] >= 0 and m0["hotload/incremental"] == 0.0
    assert [i for i, _m in stub.manifests] == [f"{RUN}-v000000"]
    assert set(stub.manifests[0][1]["tensors"]) == set(args.weights)

    # v1: delta -> applied to the mirror, verified, published as incremental
    w0 = args.weights
    w1 = perturb(w0, ("model.layers.0.weight",), seed=1)
    slime.hook(args, str(write_real_delta(deltas / "weight_v000001", w0, w1, version=1)), [])
    assert stub.uploads[-1][0].type is SnapshotType.INCREMENTAL
    assert read_all(state_dir / "mirror") == {n: d for n, (_t, _s, d) in w1.items()}

    # v2: delta applied, but published as a FULL snapshot rebuilt from the mirror
    w2 = perturb(w1, ("lm_head.bias",), seed=2)
    m2 = slime.hook(args, str(write_real_delta(deltas / "weight_v000002", w1, w2, version=2)), [])
    info, identity, _ = stub.uploads[-1]
    assert identity == f"{RUN}-v000002" and info.type is SnapshotType.FULL
    assert m2["hotload/chain_cut"] == 1.0 and m2["hotload/incremental"] == 0.0
    assert m2["hotload/tensors_verified"] == 1.0
    assert info.index_metadata["checksum_format"] == "adler32"
    assert not (state_dir / "staging" / identity).exists()  # staged copy removed after upload
    assert [i for i, _m in stub.manifests] == [f"{RUN}-v00000{i}" for i in range(3)]
    manifest = stub.manifests[-1][1]
    assert (
        manifest["tensors"]
        == json.loads((state_dir / "manifests" / f"{RUN}-v000002.json").read_text())["tensors"]
    )

    # v3: chains onto the full snapshot v2, not onto v1
    w3 = perturb(w2, ("model.layers.1.weight",), seed=3)
    slime.hook(args, str(write_real_delta(deltas / "weight_v000003", w2, w3, version=3)), [])
    info, identity, _ = stub.uploads[-1]
    assert info.type is SnapshotType.INCREMENTAL
    assert info.incremental.previous_snapshot_identity == f"{RUN}-v000002"
    assert read_all(state_dir / "mirror") == {n: d for n, (_t, _s, d) in w3.items()}


def test_mirror_stops_the_run_on_a_corrupt_delta(env, real_args, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_MIRROR", "1")
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    args = real_args
    deltas = Path(args.update_weight_disk_dir)
    slime.hook(args, str(deltas), [])
    w1 = perturb(args.weights, ("lm_head.bias",))
    bad = write_real_delta(
        deltas / "weight_v000001", args.weights, w1, version=1, corrupt="lm_head.bias"
    )
    with pytest.raises(MirrorError, match="checksum mismatch"):
        slime.hook(args, str(bad), [])
    assert [u[1] for u in stub.uploads] == [f"{RUN}-v000000"]  # nothing published for v1


def test_mirror_tracks_empty_syncs(env, real_args, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "3")
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    args = real_args
    deltas = Path(args.update_weight_disk_dir)
    slime.hook(args, str(deltas), [])
    # v1 changed nothing: index only
    empty = write_real_delta(deltas / "weight_v000001", args.weights, args.weights, version=1)
    assert not list(empty.glob("*.safetensors"))
    m1 = slime.hook(args, str(empty), [])
    assert m1["hotload/skipped_empty"] == 1.0
    # a skipped sync is not on the deployment, so it gets no checksum manifest
    assert f"{RUN}-v000001" not in [identity for identity, _m in stub.manifests]
    assert not (tmp_path / "deltas-hotload" / "manifests" / f"{RUN}-v000001.json").exists()
    # v2 applies against base 000001, which the mirror recorded as a no-op
    w2 = perturb(args.weights, ("lm_head.bias",))
    slime.hook(
        args, str(write_real_delta(deltas / "weight_v000002", args.weights, w2, version=2)), []
    )
    assert stub.uploads[-1][0].incremental.previous_snapshot_identity == f"{RUN}-v000000"
    # v3 is a chain cut even though it changed nothing
    empty3 = write_real_delta(deltas / "weight_v000003", w2, w2, version=3)
    m3 = slime.hook(args, str(empty3), [])
    assert m3["hotload/chain_cut"] == 1.0 and stub.uploads[-1][0].type is SnapshotType.FULL
    assert [identity for identity, _m in stub.manifests] == [
        f"{RUN}-v000000",
        f"{RUN}-v000002",
        f"{RUN}-v000003",
    ]
    assert read_all(tmp_path / "deltas-hotload" / "mirror") == {
        n: d for n, (_t, _s, d) in w2.items()
    }


def mirror_built(args: SimpleNamespace) -> bool:
    state = Path(args.update_weight_disk_dir).parent / "deltas-hotload"
    return (state / "mirror").exists() or (state / "mirror.tmp").exists()


@pytest.mark.parametrize("skipped", ["publish disabled", "already hot-loaded"])
def test_a_baseline_published_before_must_match_its_manifest(
    env, real_args, monkeypatch, skipped
) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "3")  # turns the mirror on
    if skipped == "publish disabled":
        monkeypatch.setenv("CWTRAIN_HOTLOAD_PUBLISH_BASELINE", "0")
    stub = StubLoader(completed={f"{RUN}-v000000"} if skipped == "already hot-loaded" else None)
    published = checkpoint_checksums(Path(real_args.hf_checkpoint))
    published["model.layers.0.weight"] = "00000000"  # the local copy has changed since
    stub.recorded[f"{RUN}-v000000"] = {"tensors": published}
    install_stub(monkeypatch, stub)
    with pytest.raises(
        CWTrainError, match=r"differs from the published .*v000000: 1 tensor\(s\) with other bytes"
    ) as e:
        slime.hook(real_args, real_args.update_weight_disk_dir, [])
    # a fresh run id only helps a run that publishes its own baseline
    assert ("unset CWTRAIN_HOTLOAD_PUBLISH_BASELINE" in str(e.value)) == (
        skipped == "publish disabled"
    )
    assert stub.manifests == []  # the published record is not overwritten
    assert not mirror_built(real_args)  # checked before the checkpoint is copied


def test_a_baseline_mismatch_names_each_kind_of_difference(env, real_args, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "3")
    monkeypatch.setenv("CWTRAIN_HOTLOAD_PUBLISH_BASELINE", "0")
    stub = StubLoader()
    published = checkpoint_checksums(Path(real_args.hf_checkpoint))
    published["model.layers.0.weight"] = "00000000"
    del published["lm_head.bias"]
    published["model.layers.9.weight"] = "00000001"
    stub.recorded[f"{RUN}-v000000"] = {"tensors": published}
    install_stub(monkeypatch, stub)
    with pytest.raises(CWTrainError) as e:
        slime.hook(real_args, real_args.update_weight_disk_dir, [])
    message = str(e.value)
    assert "1 tensor(s) with other bytes, e.g. ['model.layers.0.weight']" in message
    assert "1 tensor(s) only in --hf-checkpoint, e.g. ['lm_head.bias']" in message
    assert "1 tensor(s) only in the published baseline, e.g. ['model.layers.9.weight']" in message


@pytest.mark.parametrize("skipped", ["publish disabled", "already hot-loaded"])
def test_a_baseline_matching_its_manifest_is_reused(env, real_args, monkeypatch, skipped) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "3")
    if skipped == "publish disabled":
        monkeypatch.setenv("CWTRAIN_HOTLOAD_PUBLISH_BASELINE", "0")
    stub = StubLoader(completed={f"{RUN}-v000000"} if skipped == "already hot-loaded" else None)
    checksums = checkpoint_checksums(Path(real_args.hf_checkpoint))
    stub.recorded[f"{RUN}-v000000"] = {"tensors": checksums}
    install_stub(monkeypatch, stub)
    slime.hook(real_args, real_args.update_weight_disk_dir, [])
    assert stub.uploads == []
    assert [(identity, m["tensors"]) for identity, m in stub.manifests] == [
        (f"{RUN}-v000000", checksums)
    ]


def test_an_already_hot_loaded_baseline_needs_its_manifest(env, real_args, monkeypatch) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "3")
    stub = StubLoader(completed={f"{RUN}-v000000"})  # hot-loaded before, but nothing recorded
    install_stub(monkeypatch, stub)
    with pytest.raises(CWTrainError, match="already hot-loaded but has no checksum manifest") as e:
        slime.hook(real_args, real_args.update_weight_disk_dir, [])
    assert "set CWTRAIN_HOTLOAD_PUBLISH_BASELINE=0 to assert that" in str(e.value)
    assert stub.manifests == [] and stub.uploads == []
    assert not mirror_built(real_args)


def test_an_asserted_baseline_without_a_manifest_is_recorded_with_a_warning(
    env, real_args, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "3")
    monkeypatch.setenv("CWTRAIN_HOTLOAD_PUBLISH_BASELINE", "0")  # published with the CLI, say
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    with caplog.at_level(logging.WARNING, logger="cwtrain.integrations.slime"):
        slime.hook(real_args, real_args.update_weight_disk_dir, [])
    assert "no checksum manifest to check --hf-checkpoint against" in caplog.text
    assert [(identity, m["tensors"]) for identity, m in stub.manifests] == [
        (f"{RUN}-v000000", checkpoint_checksums(Path(real_args.hf_checkpoint)))
    ]


@pytest.mark.parametrize("skipped", ["publish disabled", "already hot-loaded"])
def test_an_unreadable_manifest_stops_the_run(env, real_args, monkeypatch, skipped) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "3")
    if skipped == "publish disabled":
        monkeypatch.setenv("CWTRAIN_HOTLOAD_PUBLISH_BASELINE", "0")
    stub = StubLoader(completed={f"{RUN}-v000000"} if skipped == "already hot-loaded" else None)

    def unreadable(identity):
        raise OSError("AccessDenied")

    stub.read_manifest = unreadable
    install_stub(monkeypatch, stub)
    with pytest.raises(CWTrainError, match="cannot be read .*fix read access") as e:
        slime.hook(real_args, real_args.update_weight_disk_dir, [])
    assert ("unset CWTRAIN_HOTLOAD_PUBLISH_BASELINE" in str(e.value)) == (
        skipped == "publish disabled"
    )
    assert stub.manifests == []
    assert not mirror_built(real_args)


def test_manifest_upload_failure_is_not_fatal(env, real_args, monkeypatch, caplog) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_MIRROR", "1")
    stub = StubLoader()

    def boom(identity, manifest):
        raise RuntimeError("s3 down")

    stub.upload_manifest = boom
    install_stub(monkeypatch, stub)
    with caplog.at_level("WARNING"):
        metrics = slime.hook(real_args, real_args.update_weight_disk_dir, [])
    assert metrics["hotload/completed"] == 1.0
    assert "could not upload checksum manifest" in caplog.text


def test_changed_settings_after_publishing_stop_the_run(env, args, monkeypatch) -> None:
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    slime.hook(args, args.update_weight_disk_dir, [])  # baseline published: the run has state
    monkeypatch.setenv("CWTRAIN_HOTLOAD_TIMEOUT", "60")
    v1 = write_delta_checkpoint(Path(args.update_weight_disk_dir) / "weight_v000001", version=1)
    with pytest.raises(CWTrainError, match=r"settings changed mid-run \(timeout\)"):
        slime.hook(args, str(v1), [])


def test_changed_settings_before_publishing_replace_the_publisher(
    env, args, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_PUBLISH_BASELINE", "0")
    created: list[SlimePublisher] = []
    stub = StubLoader()

    def factory(config, loader=None):
        created.append(SlimePublisher(config, loader=stub))
        return created[-1]

    monkeypatch.setattr(slime, "SlimePublisher", factory)
    first = slime._publisher(args)
    assert not first.has_state
    monkeypatch.setenv("CWTRAIN_HOTLOAD_VERIFY", "1")  # a real change: the default is off
    second = slime._publisher(args)
    assert second is not first and len(created) == 2
    assert "changed before anything was published" in caplog.text


def test_flag_parsing_keeps_unset_distinct_from_false() -> None:
    from cwtrain.integrations.slime import _flag, _optional_flag

    assert _flag({}, "X", True) is True and _flag({"X": "off"}, "X", True) is False
    assert _optional_flag({}, "X") is None and _optional_flag({"X": "yes"}, "X") is True
    with pytest.raises(CWTrainError, match="must be a boolean"):
        _flag({"X": "maybe"}, "X", True)


def test_manifest_is_recorded_only_after_the_hot_load_succeeds(
    env, real_args, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", "3")  # turns the mirror on
    monkeypatch.setenv("CWTRAIN_HOTLOAD_VERIFY", "1")  # the failure below comes from the check
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    args = real_args
    deltas = Path(args.update_weight_disk_dir)
    slime.hook(args, str(deltas), [])  # baseline v0: published, then recorded
    assert [identity for identity, _m in stub.manifests] == [f"{RUN}-v000000"]
    stub.verify_error = HotLoadVerificationError(StubHotLoad("x"), [], ["r1: READY on 'old'"])
    w1 = perturb(args.weights, ("lm_head.bias",))
    with pytest.raises(HotLoadVerificationError):
        slime.hook(
            args, str(write_real_delta(deltas / "weight_v000001", args.weights, w1, version=1)), []
        )
    # v1 never became the served version, so no manifest claims it
    assert [identity for identity, _m in stub.manifests] == [f"{RUN}-v000000"]
    assert not (tmp_path / "deltas-hotload" / "manifests" / f"{RUN}-v000001.json").exists()


def test_empty_values_count_as_unset() -> None:
    from cwtrain.integrations.slime import HookConfig

    cfg = HookConfig.from_env(
        {
            "CWTRAIN_HOTLOAD_DEPLOYMENT_ID": DEPLOYMENT_ID,
            "CWTRAIN_HOTLOAD_RUN_ID": RUN,
            "CWTRAIN_HOTLOAD_WAIT": "",  # templated from an unset variable
            "CWTRAIN_HOTLOAD_MIRROR": " ",
        }
    )
    assert cfg.wait is True and cfg.mirror is None


def test_seven_digit_versions_are_versions(env, args, monkeypatch) -> None:
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    slime.hook(args, args.update_weight_disk_dir, [])
    v = write_delta_checkpoint(
        Path(args.update_weight_disk_dir) / "weight_v1000000", version=1000000
    )
    slime.hook(args, str(v), [])
    assert stub.uploads[-1][1] == f"{RUN}-v1000000"


@pytest.mark.parametrize("full_every", [0, 1, 20])
@pytest.mark.parametrize("bad_index", ["array", "missing", "nested", "no_shards"])
def test_invalid_delta_rejected_before_mirror_or_publish(
    env, real_args, monkeypatch, full_every, bad_index
):
    monkeypatch.setenv("CWTRAIN_HOTLOAD_FULL_EVERY", str(full_every))
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    deltas = Path(real_args.update_weight_disk_dir)
    slime.hook(real_args, str(deltas), [])
    mirror = deltas.parent / "deltas-hotload" / "mirror"
    before = (
        {p.name: p.read_bytes() for p in mirror.iterdir() if p.is_file()} if mirror.exists() else {}
    )
    w1 = perturb(real_args.weights, ("lm_head.bias",))
    directory = write_real_delta(deltas / "weight_v000001", real_args.weights, w1, version=1)
    path = directory / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    if bad_index == "array":
        index = []
    elif bad_index == "missing":
        index["weight_map"]["lm_head.bias"] = "missing.safetensors"
    elif bad_index == "nested":
        (directory / "nested").mkdir()
        for shard in directory.glob("*.safetensors"):
            shard.rename(directory / "nested" / shard.name)
        index["weight_map"] = {
            name: "nested/" + shard for name, shard in index["weight_map"].items()
        }
    else:
        for shard in directory.glob("*.safetensors"):
            shard.unlink()
    path.write_text(json.dumps(index))
    with pytest.raises(SnapshotError):
        slime.hook(real_args, str(directory), [])
    assert len(stub.uploads) == len(stub.triggers) == 1  # baseline only
    if mirror.exists():
        assert {p.name: p.read_bytes() for p in mirror.iterdir() if p.is_file()} == before
        assert not (mirror / ".pending").exists()


@pytest.mark.parametrize(
    "bad_index", [[], {"metadata": None, "weight_map": {"w": "missing.safetensors"}}]
)
def test_invalid_baseline_rejected_before_mirror_copy(env, real_args, monkeypatch, bad_index):
    monkeypatch.delenv("CWTRAIN_HOTLOAD_FULL_EVERY")  # production default mirror enabled
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    path = Path(real_args.hf_checkpoint) / "model.safetensors.index.json"
    path.write_text(json.dumps(bad_index))
    with pytest.raises(SnapshotError):
        slime.hook(real_args, real_args.update_weight_disk_dir, [])
    assert not mirror_built(real_args)
    assert not stub.uploads and not stub.triggers


def test_null_metadata_baseline_can_be_mirrored_and_published(env, real_args, monkeypatch):
    monkeypatch.delenv("CWTRAIN_HOTLOAD_FULL_EVERY")
    stub = StubLoader()
    install_stub(monkeypatch, stub)
    path = Path(real_args.hf_checkpoint) / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["metadata"] = None
    path.write_text(json.dumps(index))
    slime.hook(real_args, real_args.update_weight_disk_dir, [])
    assert mirror_built(real_args)
    assert len(stub.uploads) == 1
