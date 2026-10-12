# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import tomllib
from pathlib import Path

import httpx
import pytest

import cwtrain
from cwtrain.infra import InferenceClient

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_version_is_the_one_pyproject_declares() -> None:
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    assert cwtrain.__version__ == metadata.version("cwtrain") == declared


def test_the_version_falls_back_when_the_package_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def not_installed(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", not_installed)
    try:
        assert importlib.reload(cwtrain).__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(cwtrain)
    assert cwtrain.__version__ == metadata.version("cwtrain")


def test_requests_identify_the_installed_version() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"hotLoads": []})

    with InferenceClient("token", transport=httpx.MockTransport(handler)) as client:
        list(client.list_hot_loads())
    assert seen[0].headers["User-Agent"] == f"cwtrain/{cwtrain.__version__}"


def test_the_distribution_carries_the_apache_license() -> None:
    meta = metadata.metadata("cwtrain")
    assert meta["License-Expression"] == "Apache-2.0"
    assert "LICENSE" in (meta.get_all("License-File") or [])
    assert (PYPROJECT.parent / "LICENSE").read_text().lstrip().startswith("Apache License")
