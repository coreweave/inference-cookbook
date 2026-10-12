# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from cwtrain.integrations.launch import (
    LaunchError,
    PatchError,
    apply_patches,
    bash_array,
    checkout,
    default_prompt_cache_policy,
    env_int,
    forwarded_env,
    is_secret_env_var,
    main,
    require,
    run,
    split_args,
    stage_from_s3,
)


@pytest.mark.parametrize(
    "given,expected",
    [(None, "reset_all"), ("", "reset_all"), ("  ", "reset_all"), ("preserve", "preserve")],
)
def test_default_prompt_cache_policy_keeps_an_explicit_value(given, expected) -> None:
    env = {} if given is None else {"CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY": given}
    default_prompt_cache_policy(env)
    assert env["CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY"] == expected


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "framework"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    (root / "loss.py").write_text("def kl(xs):\n    return [0 for x in xs]\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "base")
    return root


def make_patch(repo: Path, tmp_path: Path, name: str, old: str, new: str) -> Path:
    """A patch file for one edit of loss.py, leaving the checkout clean."""
    path = repo / "loss.py"
    original = path.read_text()
    path.write_text(original.replace(old, new))
    patch = tmp_path / name
    patch.write_text(git(repo, "diff"))
    git(repo, "checkout", "--", "loss.py")
    return patch


def test_apply_patches_applies_in_order_and_is_idempotent(repo: Path, tmp_path: Path) -> None:
    good = make_patch(repo, tmp_path, "0001-good.patch", "for x in xs", "for x in (xs or [])")
    assert apply_patches(repo, [good], tested="base") == ["0001-good.patch: applied"]
    assert apply_patches(repo, [good], tested="base") == ["0001-good.patch: already applied"]


def test_overlapping_patches_are_validated_together_and_rolled_back(
    repo: Path, tmp_path: Path
) -> None:
    base = (repo / "loss.py").read_text()
    # each applies to the base on its own, but they edit the same line differently
    first = make_patch(repo, tmp_path, "0001-first.patch", "for x in xs", "for x in (xs or [])")
    second = make_patch(repo, tmp_path, "0002-second.patch", "for x in xs", "for x in list(xs)")
    with pytest.raises(PatchError, match="0002-second.patch does not apply.*after 0001-first"):
        apply_patches(repo, [first, second], tested="v1")
    assert (repo / "loss.py").read_text() == base  # the first patch was reverted
    assert git(repo, "status", "--porcelain") == ""


def test_a_patch_for_another_version_leaves_the_checkout_untouched(
    repo: Path, tmp_path: Path
) -> None:
    base = (repo / "loss.py").read_text()
    good = make_patch(repo, tmp_path, "0001-good.patch", "for x in xs", "for x in (xs or [])")
    foreign = tmp_path / "0002-foreign.patch"
    foreign.write_text(
        "diff --git a/loss.py b/loss.py\n--- a/loss.py\n+++ b/loss.py\n"
        "@@ -1,2 +1,2 @@\n def kl(xs):\n-    return 'something else'\n+    return []\n"
    )
    with pytest.raises(PatchError, match="validated against v1"):
        apply_patches(repo, [good, foreign], tested="v1")
    assert (repo / "loss.py").read_text() == base


def test_checkout_clones_once_and_moves_to_the_ref(repo: Path, tmp_path: Path) -> None:
    first = git(repo, "rev-parse", "HEAD").strip()
    (repo / "loss.py").write_text("changed\n")
    git(repo, "commit", "-qam", "second")
    dest = checkout(str(repo), first, tmp_path / "clone")
    assert git(dest, "rev-parse", "HEAD").strip() == first
    checkout(str(repo), "HEAD", dest)  # an existing checkout is reused, not re-cloned
    assert (dest / ".git").is_dir()


def test_bash_array_reads_a_sourced_array(tmp_path: Path) -> None:
    script = tmp_path / "model.sh"
    script.write_text('MODEL_ARGS=(--num-layers 36 --rotary-base "1000000" --name "two words")\n')
    assert bash_array(script, "MODEL_ARGS") == [
        "--num-layers",
        "36",
        "--rotary-base",
        "1000000",
        "--name",
        "two words",
    ]
    (tmp_path / "broken.sh").write_text("exit 3\n")
    with pytest.raises(LaunchError, match="sourcing"):
        bash_array(tmp_path / "broken.sh", "MODEL_ARGS")


def test_split_args_keeps_quoted_values_and_ignores_newlines() -> None:
    text = '--foo 1\n--json \'{"a": [1, 2]}\'  --name "x y"'
    assert split_args(text) == ["--foo", "1", "--json", '{"a": [1, 2]}', "--name", "x y"]
    assert split_args("") == [] and split_args(None) == []
    assert split_args("--path $HOME") == ["--path", "$HOME"]  # nothing is expanded
    with pytest.raises(LaunchError, match="cannot parse EXTRA"):
        split_args('--bad "unterminated', name="EXTRA")


def test_forwarded_env_and_settings() -> None:
    env = {"CWTRAIN_HOTLOAD_RUN_ID": "r", "AWS_REGION": "x", "HOME": "/root", "NCCL_DEBUG": "INFO"}
    assert forwarded_env(env) == {
        "CWTRAIN_HOTLOAD_RUN_ID": "r",
        "AWS_REGION": "x",
        "NCCL_DEBUG": "INFO",
    }
    assert env_int({"N": " 4 "}, "N", 1) == 4 and env_int({}, "N", 1) == 1
    with pytest.raises(LaunchError, match="N must be an integer"):
        env_int({"N": "four"}, "N", 1)
    with pytest.raises(LaunchError, match="MODEL_SLUG is required"):
        require({"MODEL_SLUG": " "}, "MODEL_SLUG")


# Credential-shaped values for the tests; none of them may reach the Ray runtime env.
SECRETS = {
    "COREWEAVE_API_TOKEN": "cw-api-token-value",
    "AWS_ACCESS_KEY_ID": "aws-access-key-id-value",
    "AWS_SECRET_ACCESS_KEY": "aws-secret-access-key-value",
    "AWS_SESSION_TOKEN": "aws-session-token-value",
    "AWS_SECURITY_TOKEN": "aws-security-token-value",
    "HF_TOKEN": "hf-token-value",
    "CWTRAIN_ROLLOUT_API_KEY": "rollout-api-key-value",
    "CWTRAIN_API_HEADERS": '{"Authorization": "Bearer header-token-value"}',
}
# Settings the workers need, including paths to credentials and names that only mention tokens.
SETTINGS = {
    "CWTRAIN_HOTLOAD_RUN_ID": "r",
    "CWTRAIN_ROLLOUT_TOKEN_IDS": "1",
    "CWTRAIN_ROLLOUT_TOKEN_ID_PROMPTS": "1",
    "CWTRAIN_ROLLOUT_API_KEY_FILE": "/secrets/inference/api_key",
    "AWS_CONFIG_FILE": "/aws/config",
    "AWS_SHARED_CREDENTIALS_FILE": "/aws/credentials",
    "AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/token",
    "AWS_REGION": "US-EAST-04A",
    "AWS_PROFILE": "default",
    "HF_HOME": "/work/hf",
    "HF_TOKEN_PATH": "/secrets/hf/token",
    "CW_S3_ENDPOINT": "https://cwobject.com",
    "NCCL_DEBUG": "INFO",
}


def test_forwarded_env_leaves_out_credentials() -> None:
    env = {**SECRETS, **SETTINGS, "HOME": "/root"}
    assert forwarded_env(env) == SETTINGS
    # what the launchers hand to `ray job submit --runtime-env-json`, and what miles logs
    argv = shlex.join(
        ["ray", "job", "submit", "--runtime-env-json", json.dumps({"env_vars": forwarded_env(env)})]
    )
    for value in SECRETS.values():
        assert value not in argv


def test_extra_prefixes_do_not_bring_credentials_back() -> None:
    env = {"SLIME_RELAX_NUMPY_CHECK": "1", "SLIME_API_TOKEN": "t", "MEGATRON_PASSWORD": "p"}
    assert forwarded_env(env, ("SLIME_", "MEGATRON_")) == {"SLIME_RELAX_NUMPY_CHECK": "1"}


@pytest.mark.parametrize("name", [*SECRETS, "TOKEN", "hf_token", "DB_PASSWD", "X_PRIVATE_KEY"])
def test_credentials_are_recognised_by_name(name: str) -> None:
    assert is_secret_env_var(name)


@pytest.mark.parametrize("name", [*SETTINGS, "CWTRAIN_HOTLOAD_STATE_DIR", "MAX_TOKENS_PER_GPU"])
def test_settings_and_paths_are_not_credentials(name: str) -> None:
    assert not is_secret_env_var(name)


def test_run_raises_with_the_command_line(tmp_path: Path) -> None:
    assert run(["true"]).returncode == 0
    with pytest.raises(LaunchError, match="exited with 3: boom"):
        run(["bash", "-c", "echo boom >&2; exit 3"], capture=True)


class PagingS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.downloads: list[str] = []

    def get_paginator(self, _name: str):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix, Delimiter):  # noqa: N803
                keys = [k for k in objects if k.startswith(Prefix) and "/" not in k[len(Prefix) :]]
                yield {"Contents": [{"Key": k} for k in keys[:1]]}
                yield {"Contents": [{"Key": k} for k in keys[1:]]}

        return Paginator()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloads.append(key)
        Path(filename).write_bytes(self.objects[key])


def test_stage_from_s3_fetches_the_prefix_once(tmp_path: Path) -> None:
    s3 = PagingS3(
        {
            "models/Qwen3-4B/config.json": b"{}",
            "models/Qwen3-4B/model.safetensors": b"weights",
            "models/Qwen3-4B/sub/ignored.bin": b"nested",
            "models/Other/config.json": b"other",
            "models/NoConfig/model.safetensors": b"w",
        }
    )
    dest = tmp_path / "Qwen3-4B"
    assert stage_from_s3("s3://b/models/Qwen3-4B", dest, s3=s3) == 2
    assert sorted(p.name for p in dest.iterdir()) == ["config.json", "model.safetensors"]
    assert stage_from_s3("s3://b/models/Qwen3-4B/", dest, s3=s3) == 0  # already staged
    assert len(s3.downloads) == 2
    with pytest.raises(LaunchError, match="no objects"):
        stage_from_s3("s3://b/models/Missing", tmp_path / "missing", s3=s3)
    with pytest.raises(LaunchError, match="no config.json"):
        stage_from_s3("s3://b/models/NoConfig", tmp_path / "noconfig", s3=s3)
    with pytest.raises(LaunchError, match="s3:// URI"):
        stage_from_s3("/local/path", tmp_path / "x", s3=s3)


def test_a_failed_download_never_looks_staged(tmp_path: Path) -> None:
    s3 = PagingS3(
        {
            "m/Qwen3-4B/config.json": b"{}",
            "m/Qwen3-4B/model-00001.safetensors": b"a",
            "m/Qwen3-4B/model-00002.safetensors": b"b",
        }
    )
    real = s3.download_file

    def flaky(bucket, key, filename):
        if key.endswith("00002.safetensors"):
            raise OSError("connection reset")
        real(bucket, key, filename)

    s3.download_file = flaky
    dest = tmp_path / "Qwen3-4B"
    with pytest.raises(OSError, match="connection reset"):
        stage_from_s3("s3://b/m/Qwen3-4B", dest, s3=s3, workers=1)
    assert not dest.exists() and not (tmp_path / "Qwen3-4B.partial").exists()

    s3.download_file = real  # the retry downloads everything again and then succeeds
    assert stage_from_s3("s3://b/m/Qwen3-4B", dest, s3=s3) == 3
    assert (dest / "config.json").is_file() and (dest / "model-00002.safetensors").is_file()


def test_stage_model_command_line(tmp_path: Path, monkeypatch, capsys) -> None:
    import cwtrain.integrations.launch as launch

    calls: list[tuple] = []
    monkeypatch.setattr(launch, "stage_from_s3", lambda *a, **kw: calls.append((a, kw)) or 1)
    assert main(["stage-model", "s3://b/m", str(tmp_path), "--workers", "4"]) == 0
    assert calls == [(("s3://b/m", str(tmp_path)), {"workers": 4})]

    def fail(*_a, **_kw):
        raise LaunchError("nothing there")

    monkeypatch.setattr(launch, "stage_from_s3", fail)
    assert main(["stage-model", "s3://b/m", str(tmp_path)]) == 1
    assert "error: nothing there" in capsys.readouterr().err


def test_a_failed_revert_is_reported_not_hidden(repo: Path, tmp_path: Path, monkeypatch) -> None:
    import cwtrain.integrations.launch as launch

    first = make_patch(repo, tmp_path, "0001-first.patch", "for x in xs", "for x in (xs or [])")
    second = make_patch(repo, tmp_path, "0002-second.patch", "for x in xs", "for x in list(xs)")
    real_run = launch.run

    def run(cmd, **kw):
        if "--reverse" in [str(c) for c in cmd]:
            return subprocess.CompletedProcess(cmd, 1, "", "patch does not apply")
        return real_run(cmd, **kw)

    monkeypatch.setattr(launch, "run", run)
    with pytest.raises(PatchError, match="left partly patched: reverting 0001-first.patch failed"):
        apply_patches(repo, [first, second], tested="v1")


def test_require_patches(tmp_path: Path) -> None:
    from cwtrain.integrations.launch import require_patches

    (tmp_path / "b.patch").write_text("")
    (tmp_path / "a.patch").write_text("")
    assert [p.name for p in require_patches(tmp_path)] == ["a.patch", "b.patch"]
    with pytest.raises(LaunchError, match="no megatron\\*.patch files"):
        require_patches(tmp_path, "megatron*.patch")
