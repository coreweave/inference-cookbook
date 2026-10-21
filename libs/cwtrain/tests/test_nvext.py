# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cwtrain.integrations import nvext
from cwtrain.integrations.nvext import (
    as_cache_salt,
    require_served_salt,
    rollout_cache_salt,
    served_salt,
    with_nvext,
    write_served_identity,
)


def test_with_nvext_adds_fields_and_keeps_the_ones_already_there() -> None:
    payload = {"model": "m", "nvext": {"cache_salt": "s"}}
    merged = with_nvext(payload, extra_fields=["routed_experts"])
    assert merged["nvext"] == {"cache_salt": "s", "extra_fields": ["routed_experts"]}
    assert payload["nvext"] == {"cache_salt": "s"}  # the input is not changed
    assert with_nvext({"model": "m"}, cache_salt="t")["nvext"] == {"cache_salt": "t"}


def test_salts_follow_vllms_rules() -> None:
    assert as_cache_salt("run-7-r3") == "run-7-r3"
    for name in ("a/b", "a@b", "a\\b", "x" * 129, ""):
        salt = as_cache_salt(name)
        assert len(salt) == 64 and salt == as_cache_salt(name)  # SHA-256, stable


@pytest.fixture
def fresh_warning(monkeypatch):
    monkeypatch.setattr(nvext, "_warned_without_wait", False)


def test_a_rollouts_salt_warns_once_when_hot_loads_are_not_awaited(caplog, fresh_warning) -> None:
    env = {"CWTRAIN_HOTLOAD_RUN_ID": "grpo-7", "CWTRAIN_HOTLOAD_WAIT": "0"}
    with caplog.at_level(logging.WARNING, logger="cwtrain.integrations.nvext"):
        assert rollout_cache_salt(3, env) == f"grpo-7-{nvext._PROCESS_TOKEN}-r3"
        rollout_cache_salt(4, env)
    assert caplog.text.count("CWTRAIN_HOTLOAD_WAIT is off") == 1
    caplog.clear()
    nvext._warned_without_wait = False
    with caplog.at_level(logging.WARNING, logger="cwtrain.integrations.nvext"):
        rollout_cache_salt(3, {"CWTRAIN_HOTLOAD_RUN_ID": "grpo-7"})  # waiting is the default
    assert caplog.text == ""


def test_the_served_identity_is_recorded_and_read_as_a_salt(tmp_path: Path) -> None:
    path = tmp_path / "state" / "served_identity"
    assert served_salt(path) is None  # nothing recorded yet
    write_served_identity(path, "run-7-v000003")
    assert served_salt(path) == "run-7-v000003"
    write_served_identity(path, "run-7-v000004")
    assert served_salt(path) == "run-7-v000004" and not list(path.parent.glob("*.tmp"))
    write_served_identity(path, None)
    assert served_salt(path) is None and not path.exists()


def test_a_request_that_must_be_salted_fails_without_a_served_identity(tmp_path: Path) -> None:
    path = tmp_path / "served_identity"
    with pytest.raises(RuntimeError, match="must run where they can read the trainer's"):
        require_served_salt(path)
    write_served_identity(path, "run-7-v000003")
    assert require_served_salt(path) == "run-7-v000003"
