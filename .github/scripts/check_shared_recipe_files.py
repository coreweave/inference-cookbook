# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Fail when a shared setup file is missing from a training recipe or differs between recipes.

Each recipe under recipes/training/ is self-contained, so each one has its own copy of the files in
SHARED. The copies must be byte-identical. When you change one, copy it to every training recipe.
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

SHARED = (
    "bucket-policy.example.json",
    "deployment.example.json",
    "docs/hotload-flow.drawio.svg",
    "k8s/make-bucket-job.yaml",
    "k8s/make-storage-secret.sh",
    "k8s/namespace.yaml",
    "k8s/render.sh",
    "k8s/stage-model-job.yaml",
    "k8s/stage-model.sh",
    "trainer-access-policy.example.json",
)


def main(root: Path) -> int:
    recipes = sorted(p for p in (root / "recipes" / "training").iterdir() if p.is_dir())
    failed = False
    for rel in SHARED:
        by_digest: dict[str, list[str]] = defaultdict(list)
        missing = []
        for recipe in recipes:
            path = recipe / rel
            if path.is_file():
                by_digest[hashlib.sha256(path.read_bytes()).hexdigest()].append(recipe.name)
            else:
                missing.append(recipe.name)
        if missing:
            failed = True
            print(f"{rel} is missing from {', '.join(missing)}; every recipe carries a copy")
        if len(by_digest) > 1:
            failed = True
            groups = "; ".join(", ".join(names) for names in by_digest.values())
            print(f"{rel} differs between recipes ({groups}); copy one version to all of them")
    if not failed:
        print(f"{len(SHARED)} shared files are identical across {len(recipes)} training recipes")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(__file__).resolve().parents[2]))
