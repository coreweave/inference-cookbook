# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Importing the NeMo-RL integration outside NeMo-RL's environment says where to install it.

These tests need neither torch nor NeMo-RL, so they run in every environment.
"""

from __future__ import annotations

import importlib
import sys

import pytest


class NotInstalled:
    """A meta path finder that fails imports of one package as if it were not installed."""

    def __init__(self, package: str) -> None:
        self.package = package

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.package or fullname.startswith(f"{self.package}."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@pytest.mark.parametrize(
    ("module", "missing"),
    [
        ("cwtrain.integrations.nemo_rl.generation", "nemo_rl"),
        ("cwtrain.integrations.nemo_rl.weight_sync", "nemo_rl"),
        ("cwtrain.integrations.nemo_rl.worker", "torch"),
    ],
)
def test_import_without_nemo_rl_environment_names_it(monkeypatch, module, missing) -> None:
    # Drop cached modules, including the nemo_rl fakes that the NeMo-RL tests install, so the
    # imports run again and reach the finder.
    for name in list(sys.modules):
        if name == missing or name.startswith(("nemo_rl.", "cwtrain.integrations.nemo_rl.")):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [NotInstalled(missing), *sys.meta_path])

    with pytest.raises(ModuleNotFoundError, match="NeMo-RL's environment") as info:
        importlib.import_module(module)
    assert info.value.name == missing
    assert "NeMo-RL v0.7.0" in str(info.value)
