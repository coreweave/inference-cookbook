# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Apply the recipe fixes to the pinned Miles checkout before starting workers."""

import argparse
from pathlib import Path

from cwtrain.integrations.launch import apply_patches, require_patches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miles-dir", type=Path, default=Path("/root/miles"))
    parser.add_argument("--patch-dir", type=Path, default=Path(__file__).parent / "patches")
    args = parser.parse_args()
    apply_patches(args.miles_dir, require_patches(args.patch_dir), tested="miles v0.2.0")


if __name__ == "__main__":
    main()
