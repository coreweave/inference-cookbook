# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Ensure a failed Slime attempt is not finalized as a successful W&B run."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cwtrain.integrations.launch import apply_patches


@pytest.mark.parametrize("failure,expected", [(False, 0), (True, 1)])
def test_finish_tracking_reports_failure(tmp_path, monkeypatch, failure, expected):
    source = tmp_path / "slime/observability/logging_utils.py"
    source.parent.mkdir(parents=True)
    source.write_text('''import logging

import wandb


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")

''')
    patch = Path(__file__).resolve().parents[1] / "patches/0004-report-tracking-failure.patch"
    apply_patches(tmp_path, [patch], tested="slime v0.4.0")
    codes = []
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        run=object(), finish=lambda **kw: codes.append(kw["exit_code"])))
    finish = runpy.run_path(str(source))["finish_tracking"]

    def attempt():
        try:
            if failure:
                raise RuntimeError("injected training failure")
        finally:
            finish(SimpleNamespace(use_wandb=True))

    if failure:
        with pytest.raises(RuntimeError, match="injected training failure"):
            attempt()
    else:
        attempt()
    assert codes == [expected]
