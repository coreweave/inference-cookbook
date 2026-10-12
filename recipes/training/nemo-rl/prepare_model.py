# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Prepare a separate Nemotron checkpoint without unused speculative-decoding heads."""

import argparse
import json
import os
import shutil
from pathlib import Path

from cwtrain.hotload.mirror import tensor_locations
from cwtrain.hotload.snapshot import inspect_snapshot


def prepare(source: Path, destination: Path) -> None:
    """Keep main-model shards byte-identical; reject mixed MTP/main-model shards."""
    inspect_snapshot(source)
    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["NemotronHForCausalLM"]:
        raise ValueError("expected a NemotronHForCausalLM checkpoint")
    tensors = tensor_locations(source)
    keep = {name: loc for name, loc in tensors.items() if not name.startswith("mtp.")}
    kept_shards = {loc.path.name for loc in keep.values()}
    mtp_shards = {
        loc.path.name for name, loc in tensors.items() if name.startswith("mtp.")
    }
    if kept_shards & mtp_shards:
        raise ValueError(
            "MTP and main-model tensors share a shard; refusing to drop model weights"
        )
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for path in source.iterdir():
            if not path.is_file() or path.name in mtp_shards:
                continue
            if path.name in {"config.json", "model.safetensors.index.json"}:
                continue
            if path.name in kept_shards:
                # Source and destination are on one filesystem. A hard link avoids copying 60 GiB.
                os.link(path, destination / path.name)
            else:
                shutil.copy2(path, destination / path.name)
        config["num_nextn_predict_layers"] = 0
        config.pop("mtp_layers_block_type", None)
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        index = {
            "metadata": {"total_size": sum(loc.nbytes for loc in keep.values())},
            "weight_map": {name: loc.path.name for name, loc in keep.items()},
        }
        (destination / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
        inspect_snapshot(destination)
    except BaseException:
        shutil.rmtree(destination)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
