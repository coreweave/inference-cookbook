# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The importance-sampling settings in every config the cookbook ships, as NeMo-RL v0.7.0 reads them.

``ClippedPGLossFn`` refuses a truncation type without the correction. It also applies a truncation
ratio whenever one is set, correction or not, and raises at the first training step if the ratio
has no type. So wherever the type is null, the ratio must be null too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
RECIPE = ROOT / "training" / "recipes" / "nemo-rl"


def load(path: Path):
    """A config without its ``defaults``: NeMo-RL's own base config exists only in its container."""
    config = OmegaConf.load(path)
    config.pop("defaults", None)
    return config


RECIPE_CONFIG = load(RECIPE / "grpo_coreweave.yaml")


def importance_sampling(config) -> tuple:
    loss = config.loss_fn
    return (
        loss.get("use_importance_sampling_correction"),
        loss.get("truncated_importance_sampling_type"),
        loss.get("truncated_importance_sampling_ratio"),
    )


def check_nemo_rl_accepts(config) -> None:
    correction, kind, ratio = importance_sampling(config)
    if kind is not None:
        assert correction, "NeMo-RL refuses a truncation type without the correction"
    else:
        assert ratio is None, "NeMo-RL raises at the first training step on a ratio without a type"


def configs_on_the_recipe() -> list[Path]:
    """Every config that names the recipe's grpo_coreweave.yaml as its defaults."""
    return [
        path
        for path in sorted((ROOT / "training").rglob("*.yaml"))
        if re.search(r'^defaults:\s*"?grpo_coreweave\.yaml"?\s*$', path.read_text(), re.MULTILINE)
    ]


def test_the_recipe_turns_on_tis() -> None:
    assert importance_sampling(RECIPE_CONFIG) == (True, "tis", 2.0)
    check_nemo_rl_accepts(RECIPE_CONFIG)


@pytest.mark.parametrize("path", configs_on_the_recipe(), ids=lambda path: path.parent.name)
def test_a_config_on_the_recipe_merges_to_settings_nemo_rl_accepts(path: Path) -> None:
    check_nemo_rl_accepts(OmegaConf.merge(RECIPE_CONFIG, load(path)))


def test_the_readme_opt_out_turns_the_correction_off_entirely() -> None:
    text = " ".join((RECIPE / "README.md").read_text().split())
    bullet = text[text.index("To turn it off") :]
    overrides = re.findall(r"`(loss_fn\.\w+=[^`]+)`", bullet[: bullet.index("EXTRA_OVERRIDES")])
    merged = OmegaConf.merge(RECIPE_CONFIG, OmegaConf.from_dotlist(overrides))
    assert importance_sampling(merged) == (False, None, None)
    check_nemo_rl_accepts(merged)
