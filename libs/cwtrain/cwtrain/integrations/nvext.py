# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""What rollout requests put in Dynamo's ``nvext``: the prefix-cache salt, and the merging.

A cache salt goes into the deployment's prefix-cache key, so requests reuse cached prompt blocks
only when their salts are equal (see "Prefix cache" in :mod:`cwtrain.integrations.rollout`).
slime and miles salt each rollout with :func:`rollout_cache_salt`. NeMo-RL salts with the identity
that the deployment serves: its weight synchronizer writes it to :data:`SERVED_IDENTITY_FILE` in
the state directory, and both NeMo-RL rollout paths read it, its generation backend and cwtrain's
Gym model server.

Standard library only, so that Gym's model server can import it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cwtrain.integrations._env import flag

log = logging.getLogger(__name__)

#: vLLM's rules for a cache salt: at most this many characters, and none of ``@ / \`` or NUL.
MAX_CACHE_SALT = 128
_SALT_FORBIDDEN = frozenset("@/\\\0")

#: Names the run when ``CWTRAIN_HOTLOAD_RUN_ID`` is unset; drawn once per process.
_PROCESS_RUN = "run-" + secrets.token_hex(6)

#: The file in a NeMo-RL run's state directory that holds the identity the deployment serves.
SERVED_IDENTITY_FILE = "served_identity"

_warned_without_wait = False


def with_nvext(payload: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    """``payload`` with ``fields`` added to its ``nvext`` object; fields already there stay."""
    return {**payload, "nvext": {**payload.get("nvext", {}), **fields}}


def as_cache_salt(name: str) -> str:
    """``name`` as a cache salt: itself if vLLM accepts it, else its SHA-256 in hex."""
    if 0 < len(name) <= MAX_CACHE_SALT and not _SALT_FORBIDDEN.intersection(name):
        return name
    return hashlib.sha256(name.encode()).hexdigest()


def warn_without_wait(wait: bool) -> None:
    """Warn once per process when hot-loads are not awaited, which makes the salt unsafe.

    Replicas switch to a new version one at a time. A request with the new salt that reaches a
    replica still on the old version caches old KV under the new salt.
    """
    global _warned_without_wait
    if wait or _warned_without_wait:
        return
    _warned_without_wait = True
    log.warning(
        "CWTRAIN_HOTLOAD_WAIT is off: a rollout can start before every replica serves the new "
        "version, and a replica that has not switched yet caches KV of the old weights under the "
        "new cache salt"
    )


def rollout_cache_salt(rollout: int | str, environ: Mapping[str, str] | None = None) -> str:
    """The cache salt of one rollout of this run, ``<run>-r<rollout>``.

    The run is ``CWTRAIN_HOTLOAD_RUN_ID``, the prefix of every identity the run publishes, or a
    token drawn once per process. Neither two runs on one deployment nor two rollouts of one run
    share cached prompts. Within a rollout, the choices of a group and the turns of a
    conversation still do. Warns once when ``CWTRAIN_HOTLOAD_WAIT`` is off.
    """
    env = os.environ if environ is None else environ
    warn_without_wait(flag(env, "CWTRAIN_HOTLOAD_WAIT", True))
    run = env.get("CWTRAIN_HOTLOAD_RUN_ID", "").strip() or _PROCESS_RUN
    return as_cache_salt(f"{run}-r{rollout}")


def write_served_identity(path: str | os.PathLike[str], identity: str | None) -> None:
    """Record the identity the deployment serves, atomically; ``None`` removes the record."""
    path = Path(path)
    if identity is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(identity + "\n")
    temporary.replace(path)


def served_salt(path: str | os.PathLike[str]) -> str | None:
    """The cache salt for the identity recorded in ``path``; ``None`` before the first record."""
    try:
        identity = Path(path).read_text().strip()
    except OSError:
        return None
    return as_cache_salt(identity) if identity else None


__all__ = [
    "MAX_CACHE_SALT",
    "SERVED_IDENTITY_FILE",
    "as_cache_salt",
    "rollout_cache_salt",
    "served_salt",
    "warn_without_wait",
    "with_nvext",
    "write_served_identity",
]
