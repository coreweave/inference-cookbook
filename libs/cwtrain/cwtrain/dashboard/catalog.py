# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Versioned native-metric panels for the cookbook's W&B workspaces.

``catalog`` describes recipe metrics; callers can omit panels whose keys were not
logged. Append ``system_panels`` using discovered event keys, since distributed
writers add source suffixes. Every panel retains its native units and axis: no
metric conversion, cross-framework normalization, or source aggregation happens here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Framework = Literal["miles", "slime", "nemo-rl", "axolotl"]
SUPPORTED_FRAMEWORKS: tuple[Framework, ...] = ("miles", "slime", "nemo-rl", "axolotl")
SPEC_VERSION = 1


@dataclass(frozen=True, slots=True)
class Panel:
    """A line panel; ``scale`` is metadata and remains 1 for native values."""

    title: str
    metrics: tuple[str, ...]
    x: str
    unit: str
    description: str = ""
    scale: float = 1
    section: str = ""


def _panel(
    section: str,
    title: str,
    x: str,
    unit: str,
    *metrics: str,
    description: str = "",
) -> Panel:
    return Panel(title, metrics, x, unit, description=description, section=section)


def _training(framework: Framework) -> tuple[Panel, ...]:
    if framework in ("miles", "slime"):
        update_kl = _panel(
            "Training",
            "Approximate update KL",
            "train/step",
            "estimate",
            "train/ppo_kl",
            description=(
                "Signed mean of old-policy minus current-policy log-probabilities; "
                "distinct from reference-policy KL."
            ),
        )
        panels = [
            _panel("Training", "Policy reward", "rollout/step", "reward", "rollout/raw_reward"),
            _panel("Training", "Training loss", "train/step", "loss", "train/loss"),
            _panel("Training", "Gradient norm", "train/step", "norm", "train/grad_norm"),
            _panel("Training", "Learning rate", "train/step", "rate", "train/lr-pg_0"),
            _panel(
                "Training",
                "Clipped fractions",
                "train/step",
                "fraction",
                "train/pg_clipfrac",
                "train/tis_clipfrac",
                description=(
                    "Policy-gradient and importance-weight clipping use different mechanisms."
                ),
            ),
            _panel(
                "Training",
                "Trainer / rollout importance ratio",
                "train/step",
                "ratio",
                "train/tis",
                description=(
                    "Sampling correction before clamping; not a current / old policy ratio."
                ),
            ),
        ]
        if framework == "miles":
            panels[0:0] = [
                update_kl,
                _panel(
                    "Training",
                    "KL to reference policy",
                    "train/step",
                    "estimate",
                    "train/kl_loss",
                    description=(
                        "Sampled current-policy / reference-policy KL estimate before "
                        "multiplication by kl_loss_coef; computed only with use_kl_loss."
                    ),
                ),
            ]
            panels.extend(
                (
                    _panel(
                        "Training",
                        "Policy-weight ESS ratio",
                        "train/step",
                        "ratio",
                        "train/ess_ratio",
                        description="Normalized ESS from token-level current / old policy weights.",
                    ),
                    _panel(
                        "Training",
                        "Uniform-reward groups",
                        "rollout/step",
                        "fraction",
                        "rollout/zero_std/all_zero_percentage",
                        "rollout/zero_std/all_one_percentage",
                        description="These keys contain fractions despite their percentage names.",
                    ),
                )
            )
            # Miles can log entropy_loss=0 when entropy is not computed. A zero curve
            # would incorrectly suggest a measured policy-entropy collapse.
        else:
            panels.insert(4, update_kl)
            panels.extend(
                (
                    _panel(
                        "Training",
                        "Policy entropy",
                        "train/step",
                        "entropy",
                        "train/entropy_loss",
                        description="Full-distribution entropy reported by the slime trainer.",
                    ),
                    _panel(
                        "Training",
                        "Uniform-reward groups",
                        "rollout/step",
                        "groups",
                        "rollout/zero_std/count_0.0",
                        "rollout/zero_std/count_1.0",
                        description="Logged counts; no total-group denominator is assumed.",
                    ),
                )
            )
        return tuple(panels)

    if framework == "nemo-rl":
        return (
            _panel("Training", "Policy reward", "_step", "reward", "train/reward"),
            _panel("Training", "Training loss", "_step", "loss", "train/loss"),
            _panel("Training", "Gradient norm", "_step", "norm", "train/grad_norm"),
            _panel("Training", "Learning rate", "_step", "rate", "train/lr"),
            _panel(
                "Training",
                "Policy probability ratio",
                "_step",
                "ratio",
                "train/probs_ratio_min",
                "train/probs_ratio",
                "train/probs_ratio_max",
                description="Current / old policy probability before clipping; includes the tails.",
            ),
            _panel(
                "Training",
                "Clipped policy probability ratio",
                "_step",
                "ratio",
                "train/probs_ratio_clamped_min",
                "train/probs_ratio_clamped",
                "train/probs_ratio_clamped_max",
                description="Framework-reported current / old policy ratio after clipping.",
            ),
            _panel(
                "Training",
                "Sampling importance ratio",
                "_step",
                "ratio",
                "train/sampling_importance_ratio",
                description="Framework-reported sampling correction ratio.",
            ),
            _panel(
                "Training",
                "Importance ratios out of bounds",
                "_step",
                "fraction",
                "train/is_oob_ratio",
            ),
            _panel(
                "Training",
                "Sampled entropy estimate",
                "_step",
                "entropy",
                "train/approx_entropy",
                description="Importance-weighted sampled-token estimate.",
            ),
            _panel(
                "Training",
                "Baseline reward mix",
                "_step",
                "percent",
                "train/baseline_reward/pct_0",
                "train/baseline_reward/pct_1",
                "train/baseline_reward/pct_mixed",
                description="Percentages of baseline entries, not uniform-reward group shares.",
            ),
        )

    x = "train/global_step"
    return (
        _panel("Training", "Training loss", x, "loss", "train/loss"),
        _panel("Training", "Perplexity", x, "perplexity", "train/ppl"),
        _panel("Training", "Gradient norm", x, "norm", "train/grad_norm"),
        _panel("Training", "Learning rate", x, "rate", "train/learning_rate"),
        _panel(
            "Training",
            "Cumulative tokens",
            x,
            "tokens",
            "train/tokens/total",
            "train/tokens/trainable",
        ),
        _panel(
            "Training",
            "Training throughput per GPU",
            x,
            "tokens/s/GPU",
            "train/tokens/train_per_sec_per_gpu",
            description="Absent initial throughput observations remain missing.",
        ),
    )


def _rollout(framework: Framework) -> tuple[Panel, ...]:
    if framework == "axolotl":
        return ()
    if framework == "nemo-rl":
        return (
            _panel(
                "Rollout",
                "Generated length summary",
                "_step",
                "tokens",
                "train/mean_gen_tokens_per_sample",
                "train/max_gen_tokens_per_sample",
                description="Median and minimum are not recorded by this integration.",
            ),
            _panel("Rollout", "Truncated responses", "_step", "fraction", "train/truncation_rate"),
            _panel("Rollout", "Generation time", "_step", "seconds", "timing/train/generation"),
            _panel(
                "Rollout",
                "Valid samples",
                "_step",
                "count",
                "train/num_valid_samples",
                description="Recorded valid count; attempted completions are not available.",
            ),
            _panel(
                "Rollout",
                "Trainer–generation KL estimates",
                "_step",
                "estimate",
                "train/gen_kl_error",
                "train/policy_kl_error",
                description=(
                    "Both directions of trainer / generation mismatch; not update-policy KL."
                ),
            ),
            _panel(
                "Rollout",
                "Token multiplicative probability error",
                "_step",
                "ratio",
                "train/token_mult_prob_error",
                description="Train / generation probability disagreement; 1 indicates agreement.",
            ),
            _panel(
                "Rollout",
                "Sequences masked by log-prob error",
                "_step",
                "count",
                "train/num_masked_seqs_by_logprob_error",
            ),
        )

    x = "rollout/step"
    volume = ("coreweave/requests", "coreweave/samples")
    if framework == "miles":
        volume += ("rollout/num_training_samples",)
    panels = (
        _panel(
            "Rollout",
            "Response length summary",
            x,
            "tokens",
            "rollout/response_len/min",
            "rollout/response_len/median",
            "rollout/response_len/mean",
            "rollout/response_len/max",
            description="Per-batch summaries, not individual response lengths.",
        ),
        _panel("Rollout", "Truncated responses", x, "fraction", "rollout/truncated_ratio"),
        _panel(
            "Rollout",
            "Generation throughput",
            x,
            "tokens/s",
            "coreweave/response_tokens_per_second",
            description="Batch response tokens / wall generation time; not per GPU.",
        ),
        _panel(
            "Rollout",
            "Request latency",
            x,
            "seconds",
            "coreweave/request_seconds_p50",
            "coreweave/request_seconds_p95",
            "coreweave/request_seconds_max",
            description="Successful attempts including queue time; not time to first token.",
        ),
        _panel("Rollout", "Generation time", x, "seconds", "coreweave/generation_seconds"),
        _panel(
            "Rollout",
            "Batch volume",
            x,
            "count",
            *volume,
            description="A request can return multiple samples; these are not success rates.",
        ),
        _panel(
            "Rollout",
            "Retries and resampling",
            x,
            "count",
            "coreweave/request_retries",
            "coreweave/resampled_groups",
            description="Different event counts; neither measures final failures.",
        ),
        _panel(
            "Rollout",
            "Aligned log-prob samples",
            x,
            "fraction",
            "coreweave/logprobs_aligned_fraction",
            description="Samples whose rollout log-probs align with their response tokens.",
        ),
        _panel(
            "Rollout",
            "Trainer–rollout log-prob gap",
            "train/step",
            "absolute difference",
            "train/train_rollout_logprob_abs_diff",
            description="Training-step diagnostic; separate from policy update KL.",
        ),
    )
    if framework == "miles":
        panels += (
            _panel(
                "Rollout",
                "Trainer–rollout KL",
                "train/step",
                "estimate",
                "train/train_rollout_kl",
                description="Trainer / rollout mismatch; not reference-policy KL.",
            ),
        )
    return panels


def catalog(framework: Framework) -> tuple[Panel, ...]:
    """Return recipe panels; append discovered ``system_panels`` separately.

    Evaluation here is NeMo's in-training validation. Separate ``job_type=eval``
    runs have a different schema and are not combined with training histories.
    """
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(f"unsupported framework {framework!r}; choose {SUPPORTED_FRAMEWORKS}")
    p = "train/hotload/" if framework == "nemo-rl" else "hotload/"
    x = (
        "_step"
        if framework == "nemo-rl"
        else "train/global_step"
        if framework == "axolotl"
        else "rollout/step"
    )
    hotload = (
        _panel(
            "Hot-load",
            "Hot-load completion time",
            x,
            "seconds",
            p + "hot_load_seconds",
            description="Includes admission/completion waiting and optional replica verification.",
        ),
        _panel("Hot-load", "Snapshot upload time", x, "seconds", p + "upload_seconds"),
        _panel(
            "Hot-load",
            "Published snapshot size",
            x,
            "bytes",
            p + "snapshot_bytes",
            description=(
                "Local full-checkpoint or compressed-delta artifact size, not network traffic."
            ),
        ),
    )
    if framework != "axolotl":
        hotload += (
            _panel(
                "Hot-load",
                "Tensors changed",
                x,
                "tensors",
                p + "tensors_changed",
                description="When emitted for incremental updates; missing counts are not zero.",
            ),
        )
    evaluation = ()
    if framework == "nemo-rl":
        evaluation = (
            _panel("Evaluation", "Validation accuracy", "_step", "fraction", "validation/accuracy"),
            _panel(
                "Evaluation",
                "Validation response length",
                "_step",
                "tokens",
                "validation/avg_length",
            ),
        )
    return _training(framework) + _rollout(framework) + hotload + evaluation


_GPU_KEY = re.compile(
    r"^system\.gpu\.(\d+)\.(gpu|memoryAllocatedBytes|powerWatts|temp)(?:/l:(.+))?$"
)
_HOST_KEY = re.compile(r"^system\.(cpu|memory_percent)(?:/l:(.+))?$")
_GPU_FIELDS = (
    ("gpu", "GPU utilization", "percent", "Device activity during the sampling interval."),
    ("memoryAllocatedBytes", "GPU memory used", "bytes", "Device memory allocation in bytes."),
    ("powerWatts", "GPU power", "watts", "Device power draw at the recorded sample."),
    ("temp", "GPU temperature", "degrees C", "Device temperature at the recorded sample."),
)


def system_panels(keys: Sequence[str]) -> tuple[Panel, ...]:
    """Group discovered device keys by metric and writer, never across writers.

    ``system.gpu.process.*`` is a separate scope and is deliberately excluded.
    Device IDs are source-local indices, not a unique physical-device inventory.
    All GPU panels precede CPU/RAM. Use event timestamps, not writer-local runtimes.
    """
    gpu: dict[tuple[str, str], list[tuple[int, str]]] = {}
    host: dict[tuple[str, str], str] = {}
    for key in set(keys):
        if match := _GPU_KEY.fullmatch(key):
            device, field, source = match.groups()
            gpu.setdefault((field, source or ""), []).append((int(device), key))
        elif match := _HOST_KEY.fullmatch(key):
            field, source = match.groups()
            host[(field, source or "")] = key

    panels = []
    for field, title, unit, description in _GPU_FIELDS:
        for source in sorted(s for f, s in gpu if f == field):
            metrics = tuple(key for _, key in sorted(gpu[(field, source)]))
            label = f"{title} · {source}" if source else title
            panels.append(
                _panel(
                    "System",
                    label,
                    "_timestamp",
                    unit,
                    *metrics,
                    description=description
                    + " Writers may observe the same devices; kept separate.",
                )
            )
    for field, title, description in (
        (
            "cpu",
            "Reported process CPU",
            "Monitored-process CPU, not host-wide CPU. Sampling semantics depend on W&B SDK: "
            "0.28/0.30 report a process-lifetime average, not instantaneous utilization.",
        ),
        (
            "memory_percent",
            "RAM used",
            "Container/cgroup memory utilization when available; otherwise host RAM utilization.",
        ),
    ):
        for source in sorted(s for f, s in host if f == field):
            label = f"{title} · {source}" if source else title
            panels.append(
                _panel(
                    "System",
                    label,
                    "_timestamp",
                    "percent",
                    host[(field, source)],
                    description=description,
                )
            )
    return tuple(panels)
