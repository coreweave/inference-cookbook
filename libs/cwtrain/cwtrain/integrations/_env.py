# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Settings from environment variables, parsed the same way by every integration.

An error names the variable, or ``labels[variable]`` when the caller copied the value in from
somewhere else, such as a config file, so that it names the setting the user actually wrote.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

from cwtrain.errors import CWTrainError

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _shown(name: str, labels: Mapping[str, str] | None) -> str:
    return (labels or {}).get(name, name)


def optional_flag(
    environ: Mapping[str, str], name: str, *, labels: Mapping[str, str] | None = None
) -> bool | None:
    """A boolean setting that distinguishes "unset" (``None``) from an explicit value.

    An empty value counts as unset: templated manifests render an unset variable as ``""``.
    """
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise CWTrainError(f"{_shown(name, labels)} must be a boolean, got {raw!r}")


def flag(
    environ: Mapping[str, str],
    name: str,
    default: bool,
    *,
    labels: Mapping[str, str] | None = None,
) -> bool:
    """A boolean setting; unset (or empty) means ``default``."""
    value = optional_flag(environ, name, labels=labels)
    return default if value is None else value


def positive_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    labels: Mapping[str, str] | None = None,
) -> int:
    """A count of at least 1; unset (or empty) means ``default``."""
    raw = (environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise CWTrainError(f"{_shown(name, labels)} must be a whole number, got {raw!r}") from None
    if value < 1:
        raise CWTrainError(f"{_shown(name, labels)} must be at least 1, got {raw!r}")
    return value


def positive_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    labels: Mapping[str, str] | None = None,
) -> float:
    """A finite number above 0, such as seconds; unset (or empty) means ``default``."""
    raw = (environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise CWTrainError(f"{_shown(name, labels)} must be a number, got {raw!r}") from None
    if not (math.isfinite(value) and value > 0):  # NaN and infinities fail this too
        raise CWTrainError(f"{_shown(name, labels)} must be a finite number above 0, got {raw!r}")
    return value


def json_headers(
    environ: Mapping[str, str], name: str, *, labels: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Extra HTTP headers given as a JSON object; unset (or empty) means none."""
    raw = (environ.get(name) or "").strip()
    if not raw:
        return {}
    try:
        return {str(k): str(v) for k, v in json.loads(raw).items()}
    except (ValueError, AttributeError) as exc:
        raise CWTrainError(f"{_shown(name, labels)} must be a JSON object") from exc
