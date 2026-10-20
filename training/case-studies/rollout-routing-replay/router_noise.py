# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Seeded noise on a checkpoint's router weights, as a reproducible stand-in for a training step.

The case study hot-loads two versions made this way, to see which routes a deployment returns after
each hot-load. Only the routers of the decoder layers change (``layers.N.mlp.gate.weight``), so a
few percent of routing choices flip and nothing else does. The analysis rebuilds the same weights
from the seed and checks them against the digests that this script records.

Make v1 from the served checkpoint, then v2 from v1 together with the delta that turns v1 into v2::

    python router_noise.py --base GLM-5.3-Flash-BF16 --out v1 --seed 1 --scale 0.02
    python router_noise.py --base v1 --out v2 --seed 2 --scale 0.02 --delta v2-delta

Every file except the shards that hold routers is hard-linked from ``--base``. The digests go to
``<out>.noise.json``, next to the new checkpoint rather than in it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROUTER = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate\.weight$")


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """bfloat16 values, stored as ``uint16`` bit patterns, as float32."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """float32 values rounded to the nearest bfloat16, ties to even, as ``uint16`` bit patterns.

    This is how torch converts finite values, so both sides of the case study get the same bits.
    """
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


def rms(bits: np.ndarray) -> float:
    """The root mean square of a bfloat16 tensor, accumulated in float64."""
    values = bf16_to_f32(bits).astype(np.float64)
    return math.sqrt(float(np.mean(values * values)))


def noised(bits: np.ndarray, *, seed: int, layer: int, sigma: float) -> np.ndarray:
    """``bits`` plus uniform noise with standard deviation ``sigma``, rounded back to bfloat16.

    The noise uses only PCG64 and IEEE arithmetic, so it is the same on every platform.
    """
    uniform = np.random.Generator(np.random.PCG64([seed, layer])).random(bits.shape)
    noise = (2.0 * uniform - 1.0) * (sigma * math.sqrt(3.0))
    return f32_to_bf16((bf16_to_f32(bits).astype(np.float64) + noise).astype(np.float32))


def sha256(bits: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()


def router_tensors(names: Sequence[str], num_layers: int) -> dict[str, int]:
    """The router weights of the decoder layers, by name, with their layer index.

    Layers at ``num_layers`` and beyond (multi-token prediction) serve no rollouts, so they keep
    their weights.
    """
    found = {}
    for name in names:
        match = ROUTER.search(name)
        if match and int(match.group(1)) < num_layers:
            found[name] = int(match.group(1))
    return dict(sorted(found.items(), key=lambda item: item[1]))


def num_hidden_layers(checkpoint: Path) -> int:
    config = json.loads((checkpoint / "config.json").read_text())
    return int((config.get("text_config") or config)["num_hidden_layers"])


def make_version(base: Path, out: Path, *, seed: int, scale: float) -> dict[str, Any]:
    """Write ``base`` with noisy routers to ``out`` and return the record of what changed.

    Each router gets noise with a standard deviation of ``scale`` times its own RMS.
    """
    from cwtrain.hotload.mirror import tensor_locations

    locations = tensor_locations(base)
    routers = router_tensors(list(locations), num_hidden_layers(base))
    if not routers:
        raise SystemExit(f"no router weights named like layers.N.mlp.gate.weight in {base}")
    index = json.loads((base / "model.safetensors.index.json").read_text())
    if "checksum_format" in index:
        raise SystemExit(f"{base} carries per-tensor checksums; start from a plain checkpoint")
    shards = {locations[name].path.name for name in routers}
    out.mkdir(parents=True, exist_ok=False)
    for source in sorted(base.iterdir()):
        if not source.is_file():
            continue
        if source.name in shards:
            shutil.copyfile(source, out / source.name)
        else:
            try:
                os.link(source, out / source.name)
            except OSError:  # another file system
                shutil.copyfile(source, out / source.name)
    record: dict[str, Any] = {"seed": seed, "scale": scale, "tensors": {}}
    for name, layer in routers.items():
        loc = locations[name]
        if loc.dtype != "BF16":
            raise SystemExit(f"{name} is {loc.dtype}; this script handles BF16 routers only")
        with (out / loc.path.name).open("r+b") as f:
            f.seek(loc.offset)
            bits = np.frombuffer(f.read(loc.nbytes), dtype=np.uint16).reshape(loc.shape)
            sigma = scale * rms(bits)
            new = noised(bits, seed=seed, layer=layer, sigma=sigma)
            f.seek(loc.offset)
            f.write(new.tobytes())
        record["tensors"][name] = {
            "layer": layer,
            "sigma": sigma,
            "changed_elements": int((new != bits).sum()),
            "sha256_before": sha256(bits),
            "sha256_after": sha256(new),
        }
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add seeded noise to a checkpoint's routers.")
    parser.add_argument(
        "--base", type=Path, required=True, help="checkpoint directory to start from"
    )
    parser.add_argument("--out", type=Path, required=True, help="new checkpoint directory")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True, help="noise std over router RMS")
    parser.add_argument("--delta", type=Path, help="also write the delta from --base to --out here")
    args = parser.parse_args(argv)

    record = make_version(args.base, args.out, seed=args.seed, scale=args.scale)
    record |= {"base": args.base.name, "version": args.out.name}
    if args.delta:
        from cwtrain.hotload.delta import write_delta

        stats = write_delta(
            args.base, args.out, args.delta, version=args.out.name, base_version=args.base.name
        )
        record["delta"] = {"tensors_changed": stats.tensors_changed, "wire_bytes": stats.wire_bytes}
    noise_file = args.out.with_name(args.out.name + ".noise.json")
    noise_file.write_text(json.dumps(record, indent=2) + "\n")
    changed = sum(t["changed_elements"] for t in record["tensors"].values())
    print(f"{len(record['tensors'])} routers, {changed} elements changed; record in {noise_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
