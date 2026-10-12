# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Building blocks for the recipes' trainer launchers.

Before it starts the framework, every recipe's trainer Job does the same steps: check out a pinned
framework commit, apply the recipe's patches, stage the base checkpoint from the deployment's
bucket, and forward the hot-load and rollout settings to the Ray workers. These helpers implement
each step, so a recipe's launcher only describes what is specific to its framework.

This module uses only the standard library and :mod:`cwtrain.hotload`, and imports no training
framework. The staging step is also a command line::

    python -m cwtrain.integrations.launch stage-model s3://bucket/models/Qwen3-4B /work/Qwen3-4B
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cwtrain.errors import CWTrainError

#: Environment the hot-load hook, the rollout function and the object-storage client read inside
#: the Ray workers. Credentials among them are never forwarded; see :func:`forwarded_env`.
FORWARDED_PREFIXES = ("CWTRAIN_", "COREWEAVE_", "AWS_", "CW_S3_", "HF_", "NCCL_", "GLOO_")
PROMPT_CACHE_POLICY_ENV = "CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY"

# A name holding one of these words, or ending in one of these suffixes, names a credential.
_SECRET_WORDS = frozenset({"SECRET", "SECRETS", "PASSWORD", "PASSWD", "CREDENTIAL", "CREDENTIALS"})
_SECRET_SUFFIXES = (
    "TOKEN",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "ACCESS_KEY_ID",
    "PRIVATE_KEY",
    "HEADERS",
)
# ...unless the variable holds the path to one, as the recipes' Secrets mounted as files do.
_PATH_SUFFIXES = ("_FILE", "_PATH", "_DIR")


class LaunchError(CWTrainError):
    """A launcher step could not be completed."""


class PatchError(LaunchError):
    """A recipe patch does not apply to the framework checkout it was given."""


def say(message: str) -> None:
    """A progress line on stdout, where Job logs collect it."""
    print(message, flush=True)


def run(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str] | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` (a list, never a shell string), echoing it first; raise on failure."""
    args = [str(a) for a in cmd]
    say(f"$ {shlex.join(args)}")
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=capture)
    if check and result.returncode != 0:
        detail = f": {result.stderr.strip()[-500:]}" if capture and result.stderr else ""
        raise LaunchError(f"{shlex.join(args)} exited with {result.returncode}{detail}")
    return result


def git_describe(repo: str | os.PathLike[str]) -> str:
    """``<short sha> <date>`` of a checkout, or ``"unknown"`` when it is not a git checkout."""
    result = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--format=%h %cs"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def checkout(url: str, ref: str, dest: str | os.PathLike[str]) -> Path:
    """Clone ``url`` into ``dest`` unless it is already a checkout, then check out ``ref``."""
    dest = Path(dest)
    if not (dest / ".git").exists():
        run(["git", "clone", "--quiet", url, dest])
    run(["git", "-C", dest, "checkout", "--quiet", ref])
    return dest


def fetch_commit(url: str, commit: str, dest: str | os.PathLike[str]) -> Path:
    """A shallow checkout of one commit (with submodules) into an empty ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    run(["git", "-C", dest, "init", "-q"])
    run(["git", "-C", dest, "remote", "add", "origin", url], check=False)
    run(["git", "-C", dest, "fetch", "-q", "--depth", "1", "origin", commit])
    run(["git", "-C", dest, "checkout", "-q", "FETCH_HEAD"])
    run(
        ["git", "-C", dest, "submodule", "update", "--init", "--recursive", "--depth", "1"],
        check=False,
    )
    return dest


def _applies(repo: Path, patch: Path, *, reverse: bool) -> bool:
    cmd = [
        "git",
        "-C",
        str(repo),
        "apply",
        "--check",
        *(["--reverse"] if reverse else []),
        str(patch),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def apply_patches(
    repo: str | os.PathLike[str], patches: Iterable[str | os.PathLike[str]], *, tested: str
) -> list[str]:
    """Apply ``patches`` to ``repo`` in order, all-or-nothing, skipping ones already applied.

    Each patch is checked against the checkout after the earlier patches, so patches that touch
    the same code are validated in application order.

    Args:
        repo: The framework checkout.
        patches: The patch files. They are applied in sorted order.
        tested: The framework version the patches are validated against. The error names it.

    Returns:
        One status line per patch.

    Raises:
        PatchError: A patch does not apply. The patches this call applied are reverted first, so
            the checkout is left as it was found. If a revert fails, the error says so.
    """
    repo = Path(repo)
    applied: list[Path] = []
    status: list[str] = []
    try:
        for patch in sorted(Path(p) for p in patches):
            if _applies(repo, patch, reverse=True):
                status.append(f"{patch.name}: already applied")
            elif _applies(repo, patch, reverse=False):
                run(["git", "-C", repo, "apply", patch])
                applied.append(patch)
                status.append(f"{patch.name}: applied")
            else:
                after = f" after {', '.join(p.name for p in applied)}" if applied else ""
                raise PatchError(
                    f"{patch.name} does not apply to {repo} ({git_describe(repo)}){after}. The "
                    f"recipe is validated against {tested}; use that version, or check whether "
                    "upstream now ships this change and the patch can be dropped (see the recipe "
                    "README)."
                )
            say(status[-1])
    except BaseException as exc:
        stuck = []
        for patch in reversed(applied):
            if run(["git", "-C", repo, "apply", "--reverse", patch], check=False).returncode:
                stuck.append(patch.name)
                say(f"{patch.name}: could not be reverted")
            else:
                say(f"{patch.name}: reverted")
        if stuck:
            raise PatchError(
                f"{repo} is left partly patched: reverting {', '.join(stuck)} failed after "
                f"{exc}. Restore the checkout (git -C {repo} checkout -- .) before retrying."
            ) from exc
        raise
    return status


def require_patches(directory: str | os.PathLike[str], pattern: str = "*.patch") -> list[Path]:
    """The patches a recipe ships in ``directory``; a kit without any is a build error."""
    patches = sorted(Path(directory).glob(pattern))
    if not patches:
        raise LaunchError(
            f"no {pattern} files in {directory}; the recipe needs its patches, so rebuild the kit "
            "with k8s/make-kit.sh"
        )
    return patches


def stage_from_s3(
    uri: str, dest: str | os.PathLike[str], *, workers: int = 16, s3: Any = None
) -> int:
    """Download the objects directly under ``uri`` into ``dest``; return how many were fetched.

    Use the deployment's own model path as ``uri``. The trainer then has a delta baseline that is
    byte-identical to what the deployment serves.

    Nothing is fetched when ``dest`` already holds a checkpoint (a ``config.json``). Files download
    into a sibling ``<dest>.partial`` directory. It is renamed to ``dest`` only after every object
    has downloaded, so an interrupted download never looks like a staged checkpoint.

    Raises:
        LaunchError: ``uri`` is not an ``s3://`` URI, or it holds no ``config.json``.
    """
    dest = Path(dest)
    if (dest / "config.json").is_file():
        say(f"base checkpoint already at {dest}")
        return 0
    if not uri.startswith("s3://"):
        raise LaunchError(f"expected an s3:// URI, got {uri!r}")
    bucket, _, prefix = uri[len("s3://") :].partition("/")
    prefix = prefix.rstrip("/") + "/"
    if s3 is None:
        from cwtrain.infra.storage import s3_client

        s3 = s3_client()
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix, Delimiter="/"
    )
    keys = [obj["Key"] for page in pages for obj in page.get("Contents") or []]
    if not keys:
        raise LaunchError(f"no objects under {uri}")
    if not any(key.rsplit("/", 1)[-1] == "config.json" for key in keys):
        raise LaunchError(f"{uri} holds no config.json, so it is not a HuggingFace checkpoint")
    partial = dest.with_name(dest.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)

    def get(key: str) -> str:
        s3.download_file(bucket, key, str(partial / key.rsplit("/", 1)[-1]))
        return key

    try:
        with ThreadPoolExecutor(max(1, workers)) as pool:
            for key in pool.map(get, keys):
                say(f"fetched {key}")
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    if dest.exists():  # an incomplete earlier attempt: no config.json, as checked above
        shutil.rmtree(dest)
    partial.rename(dest)
    say(f"base checkpoint: {len(keys)} files from {uri}")
    return len(keys)


def bash_array(script: str | os.PathLike[str], name: str, *, cwd: str | None = None) -> list[str]:
    """The elements of bash array ``name`` after sourcing ``script`` (e.g. slime's MODEL_ARGS)."""
    program = f'set -e; source "$1"; printf "%s\\0" "${{{name}[@]}}"'
    result = subprocess.run(
        ["bash", "-c", program, "bash", str(script)], cwd=cwd, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise LaunchError(f"sourcing {script} failed: {result.stderr.strip()[-500:]}")
    return [item for item in result.stdout.split("\0") if item]


def split_args(text: str | None, *, name: str = "extra arguments") -> list[str]:
    """Shell-style words of ``text``; newlines are plain whitespace, nothing is expanded."""
    if not text or not text.strip():
        return []
    try:
        return shlex.split(text)
    except ValueError as exc:
        raise LaunchError(f"cannot parse {name} {text!r}: {exc}") from exc


def is_secret_env_var(name: str) -> bool:
    """Whether environment variable ``name`` holds a credential, judged by its name alone.

    Credentials include ``COREWEAVE_API_TOKEN``, ``AWS_ACCESS_KEY_ID``,
    ``AWS_SECRET_ACCESS_KEY``, ``AWS_SESSION_TOKEN``, ``HF_TOKEN``, ``CWTRAIN_ROLLOUT_API_KEY`` and
    ``CWTRAIN_API_HEADERS`` (which can carry an authorization header). The access key ID counts
    because it is half of the key pair. A path to a credential does not count
    (``CWTRAIN_ROLLOUT_API_KEY_FILE``, ``AWS_SHARED_CREDENTIALS_FILE``). A setting that only
    mentions tokens, such as ``CWTRAIN_ROLLOUT_TOKEN_IDS``, does not count either.
    """
    upper = name.upper()
    if upper.endswith(_PATH_SUFFIXES):
        return False
    if _SECRET_WORDS.intersection(upper.split("_")):
        return True
    return any(upper == s or upper.endswith("_" + s) for s in _SECRET_SUFFIXES)


def forwarded_env(
    environ: Mapping[str, str] | None = None, prefixes: Sequence[str] = FORWARDED_PREFIXES
) -> dict[str, str]:
    """The variables whose names start with one of ``prefixes``, except credentials.

    The result becomes the Ray job's runtime env, which must not hold secrets. It travels on the
    ``ray job submit`` command line, where every process on the node can read it. Frameworks echo
    it into the job log (miles logs the whole command), and Ray stores it with the job. So
    credentials (:func:`is_secret_env_var`) are left out.

    The workers still get the credentials: each Ray worker inherits the environment of the
    ``ray start`` that started its node. In the recipes' single-pod Jobs, that is the pod's
    environment, Secrets included. A Ray cluster that spans several pods needs the same Secrets
    in each pod.
    """
    env = os.environ if environ is None else environ
    return {
        k: v for k, v in env.items() if k.startswith(tuple(prefixes)) and not is_secret_env_var(k)
    }


def default_prompt_cache_policy(env: MutableMapping[str, str], policy: str = "reset_all") -> None:
    """Default the hot-load prompt-cache policy for a synchronous RL loop.

    A synchronous loop starts each rollout round from the weights it just hot-loaded. So it asks
    the platform to drop the KV cache computed with the previous weights. With the platform
    default (preserve), a cached prompt prefix carries the old policy into the new round.

    An explicit value, ``preserve`` included, is kept. A blank value counts as unset, because an
    unset run.env value renders as an empty string.
    """
    if not env.get(PROMPT_CACHE_POLICY_ENV, "").strip():
        env[PROMPT_CACHE_POLICY_ENV] = policy


def env_text(environ: Mapping[str, str], name: str, default: str) -> str:
    """A text setting; unset or blank means ``default``."""
    return environ.get(name, "").strip() or default


def env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LaunchError(f"{name} must be an integer, got {raw!r}") from exc


def require(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise LaunchError(f"{name} is required")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cwtrain.integrations.launch")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stage-model", help="download a checkpoint from s3://bucket/prefix")
    p.add_argument("uri")
    p.add_argument("dest")
    p.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "stage-model":
            stage_from_s3(args.uri, args.dest, workers=args.workers)
    except LaunchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "FORWARDED_PREFIXES",
    "PROMPT_CACHE_POLICY_ENV",
    "LaunchError",
    "PatchError",
    "apply_patches",
    "bash_array",
    "checkout",
    "default_prompt_cache_policy",
    "env_int",
    "env_text",
    "fetch_commit",
    "forwarded_env",
    "git_describe",
    "is_secret_env_var",
    "require",
    "require_patches",
    "run",
    "say",
    "split_args",
    "stage_from_s3",
]
