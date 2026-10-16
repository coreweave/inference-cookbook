# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

import importlib.util
import json
import struct
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_nemotron", Path(__file__).resolve().parents[1] / "prepare_model.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def checkpoint(root, mixed=False):
    root.mkdir()
    config = {"architectures": ["NemotronHForCausalLM"], "num_nextn_predict_layers": 1}
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer_config.json").write_text("{}")
    groups = (
        [["backbone.weight", "mtp.weight"]]
        if mixed
        else [["backbone.weight"], ["mtp.weight"]]
    )
    weights = {}
    for i, names in enumerate(groups):
        shard = f"model-{i}.safetensors"
        header = {
            name: {"dtype": "F32", "shape": [1], "data_offsets": [j * 4, (j + 1) * 4]}
            for j, name in enumerate(names)
        }
        body = json.dumps(header).encode()
        (root / shard).write_bytes(
            struct.pack("<Q", len(body)) + body + b"\0" * (4 * len(names))
        )
        weights.update({name: shard for name in names})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )


def test_preparation_keeps_main_weights_and_source_unchanged(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "prepared"
    checkpoint(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    module.prepare(source, destination)
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert index == {
        "metadata": {"total_size": 4},
        "weight_map": {"backbone.weight": "model-0.safetensors"},
    }
    assert (destination / "model-0.safetensors").read_bytes() == before[
        "model-0.safetensors"
    ]
    assert not (destination / "model-1.safetensors").exists()
    assert (
        json.loads((destination / "config.json").read_text())[
            "num_nextn_predict_layers"
        ]
        == 0
    )
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    with pytest.raises(FileExistsError):
        module.prepare(source, destination)


def test_mixed_shard_is_rejected_before_writing(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "prepared"
    checkpoint(source, mixed=True)
    with pytest.raises(ValueError, match="share a shard"):
        module.prepare(source, destination)
    assert not destination.exists()
