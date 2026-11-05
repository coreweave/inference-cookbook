# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from dataclasses import FrozenInstanceError

import pytest

from cwtrain.dashboard.catalog import SPEC_VERSION, SUPPORTED_FRAMEWORKS, catalog, system_panels


def metric_panels(framework):
    return {key: panel for panel in catalog(framework) for key in panel.metrics}


@pytest.mark.parametrize("framework", ["miles", "slime"])
def test_optimizer_and_rollout_axes_remain_distinct(framework):
    panels = metric_panels(framework)
    assert panels["train/loss"].x == "train/step"
    assert panels["train/train_rollout_logprob_abs_diff"].x == "train/step"
    assert panels["train/train_rollout_logprob_abs_diff"].section == "Rollout"
    assert panels["rollout/raw_reward"].x == "rollout/step"
    assert panels["hotload/snapshot_bytes"].x == "rollout/step"


def test_other_frameworks_keep_their_native_step_and_schema():
    nemo = metric_panels("nemo-rl")
    assert {panel.x for panel in nemo.values()} == {"_step"}
    assert "train/hotload/snapshot_bytes" in nemo
    assert "hotload/snapshot_bytes" not in nemo
    assert nemo["validation/accuracy"].section == "Evaluation"
    assert nemo["train/gen_kl_error"].section == "Rollout"
    assert "not update-policy KL" in nemo["train/gen_kl_error"].description
    axolotl = metric_panels("axolotl")
    assert {panel.x for panel in axolotl.values()} == {"train/global_step"}
    assert not any(panel.section in ("Rollout", "Evaluation") for panel in axolotl.values())
    assert "hotload/tensors_changed" not in axolotl


def test_entropy_and_reward_mix_do_not_claim_unavailable_measurements():
    miles, slime, nemo = (metric_panels(f) for f in ("miles", "slime", "nemo-rl"))
    assert "train/entropy_loss" not in miles
    assert "train/entropy_loss" in slime
    assert "train/approx_entropy" in nemo
    assert miles["rollout/zero_std/all_zero_percentage"].unit == "fraction"
    assert slime["rollout/zero_std/count_0.0"].unit == "groups"
    assert "denominator" in slime["rollout/zero_std/count_0.0"].description
    assert nemo["train/baseline_reward/pct_0"].unit == "percent"
    assert nemo["train/is_oob_ratio"].unit == "fraction"
    assert miles["train/pg_clipfrac"].unit == "fraction"


def test_miles_prioritizes_distinct_update_and_reference_kl_estimates():
    training = [panel for panel in catalog("miles") if panel.section == "Training"]
    update, reference = training[:2]
    assert update.title == "Approximate update KL"
    assert update.metrics == ("train/ppo_kl",)
    assert reference.title == "KL to reference policy"
    assert reference.metrics == ("train/kl_loss",)
    assert update.x == reference.x == "train/step"
    assert update.unit == reference.unit == "estimate"
    assert "Signed" in update.description and "old-policy" in update.description
    assert "before multiplication by kl_loss_coef" in reference.description
    assert "only with use_kl_loss" in reference.description
    mismatch = metric_panels("miles")["train/train_rollout_kl"]
    assert mismatch.section == "Rollout"
    assert "not reference-policy KL" in mismatch.description


def test_catalog_is_immutable_versioned_and_uses_unscaled_native_values():
    assert SPEC_VERSION == 1
    assert SUPPORTED_FRAMEWORKS == ("miles", "slime", "nemo-rl", "axolotl")
    for framework in SUPPORTED_FRAMEWORKS:
        panels = catalog(framework)
        assert isinstance(panels, tuple)
        assert all(p.metrics and isinstance(p.metrics, tuple) and p.scale == 1 for p in panels)
        sizes = [p for p in panels if any(k.endswith("/snapshot_bytes") for k in p.metrics)]
        assert len(sizes) == 1 and sizes[0].unit == "bytes"
        with pytest.raises(FrozenInstanceError):
            panels[0].title = "Changed"
    with pytest.raises(ValueError, match="unsupported framework"):
        catalog("unknown")


def test_system_metrics_keep_writer_scope_and_numeric_device_order():
    keys = [
        "system.gpu.10.gpu/l:worker-a",
        "system.gpu.2.gpu/l:worker-a",
        "system.gpu.2.gpu/l:worker-a",
        "system.gpu.0.gpu/l:worker-b",
        "system.gpu.0.gpu",
        "system.gpu.process.0.gpu",
        "system.gpu.2.memoryAllocatedBytes/l:worker-a",
        "system.gpu.2.memory/l:worker-a",  # Memory activity is not capacity usage.
        "system.gpu.2.powerWatts/l:worker-a",
        "system.gpu.2.temp/l:worker-a",
        "system.cpu/l:worker-a",
        "system.memory_percent/l:worker-a",
    ]
    panels = system_panels(keys)
    assert panels == system_panels(list(reversed(keys)))
    utilization = [p for p in panels if p.title.startswith("GPU utilization")]
    assert len(utilization) == 3
    assert utilization[0].metrics == ("system.gpu.0.gpu",)
    assert utilization[1].metrics == (
        "system.gpu.2.gpu/l:worker-a",
        "system.gpu.10.gpu/l:worker-a",
    )
    assert utilization[2].metrics == ("system.gpu.0.gpu/l:worker-b",)
    emitted = {key for panel in panels for key in panel.metrics}
    assert "system.gpu.process.0.gpu" not in emitted
    assert "system.gpu.2.memory/l:worker-a" not in emitted
    assert all(p.x == "_timestamp" and p.section == "System" and p.scale == 1 for p in panels)
    assert [p.unit for p in panels[-5:]] == ["bytes", "watts", "degrees C", "percent", "percent"]
    assert panels[-2].metrics == ("system.cpu/l:worker-a",)
    assert panels[-1].metrics == ("system.memory_percent/l:worker-a",)
    assert "lifetime" in panels[-2].description
    assert "cgroup" in panels[-1].description


def test_system_discovery_does_not_invent_missing_devices_or_host_metrics():
    assert system_panels([]) == ()
    assert system_panels(["_timestamp", "system.gpu.process.0.gpu", "system.gpu.0.gpu/extra"]) == ()
    panels = system_panels(["system.gpu.7.temp", "system.cpu"])
    assert [p.metrics for p in panels] == [("system.gpu.7.temp",), ("system.cpu",)]
