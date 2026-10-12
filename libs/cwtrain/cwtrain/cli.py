# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""``cwtrain`` command line: the deployment rollouts come from, and the hot-loads into it.

Examples::

    cwtrain deployment create --file deployment.json --wait
    cwtrain hotload publish --deployment <uuid> --checkpoint ./step_000100 --identity step_000100
    cwtrain hotload trigger --deployment <uuid> --identity <delta> --previous-identity <base>
    cwtrain hotload status <hot-load-uuid>
    cwtrain hotload list --deployment <uuid>
    cwtrain hotload check --deployment <uuid>
    cwtrain deployment delete <uuid> --wait
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import traceback
import uuid
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from boto3.exceptions import RetriesExceededError, S3UploadFailedError
from botocore.exceptions import (
    ClientError,
    CredentialRetrievalError,
    HTTPClientError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
)
from botocore.exceptions import (
    ConnectionError as StorageConnectionError,
)

from cwtrain.errors import AlreadyExistsError, APIError, CWTrainError, NotFoundError
from cwtrain.hotload import HotLoader, PromptCachePolicy, inspect_snapshot
from cwtrain.infra import lifecycle
from cwtrain.infra.client import (
    DEFAULT_BASE_URL,
    TOKEN_ENV_VAR,
    TOKEN_ENV_VARS,
    InferenceClient,
    token_from_env,
)
from cwtrain.infra.models import Deployment

log = logging.getLogger(__name__)

#: Libraries that log every HTTP request and credential lookup at INFO, which buries the hot-load
#: progress lines. They log only warnings, unless --verbose is set.
_CHATTY_LOGGERS = ("httpx", "httpcore", "botocore", "boto3", "s3transfer")
#: A value left as in the recipes' templates, such as ``<gateway uuid in one zone>`` or
#: ``<path to Nemotron 3.5 Lightning BF16 checkpoint and tokenizer files>``.
_PLACEHOLDER = re.compile(r"<[A-Za-z][^<>]*>")
#: Free-form request fields, which are not checked for placeholders: the engine's own options
#: and environment can legitimately hold values such as ``<think>`` or ``<s>``.
_FREE_FORM = frozenset({"runtime.engineConfig", "runtime.engineEnv"})


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items() if k not in {"raw", "index_bytes"}}
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__fspath__"):
        return str(obj)
    return obj


def _print(obj: Any) -> None:
    json.dump(_jsonable(obj), sys.stdout, indent=2)
    sys.stdout.write("\n")


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero") from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer greater than zero") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return number


def _prompt_cache_policy(args: argparse.Namespace) -> PromptCachePolicy:
    """The policy for ``publish`` and ``trigger``, the commands with ``--reset-prompt-cache``."""
    return PromptCachePolicy.RESET_ALL if args.reset_prompt_cache else PromptCachePolicy.UNSPECIFIED


def _read_request(path: str) -> dict[str, Any]:
    """The JSON request body in ``path`` (``-`` for stdin); refused while it holds placeholders."""
    try:
        text = sys.stdin.read() if path == "-" else Path(path).read_text()
    except OSError as exc:
        raise CWTrainError(f"cannot read {path}: {exc.strerror or exc}") from exc
    try:
        body = json.loads(text)
    except ValueError as exc:
        raise CWTrainError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise CWTrainError(f"{path} must hold a JSON object, the request body")
    unfilled = list(_placeholders(body))
    if unfilled:
        shown = ", ".join(f"{where} = {value!r}" for where, value in unfilled[:5])
        raise CWTrainError(f"fill in {path} first; it still has placeholders: {shown}")
    return body


def _placeholders(value: Any, where: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{where}.{key}" if where else str(key)
            if path not in _FREE_FORM:
                yield from _placeholders(item, path)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _placeholders(item, f"{where}[{i}]")
    elif isinstance(value, str) and _PLACEHOLDER.search(value):
        yield where, value


def _deployment_view(client: InferenceClient, deployment: Deployment) -> dict[str, Any]:
    """The deployment as JSON, with its gateways' endpoints and the URL rollouts are sent to.

    If the gateways cannot be read, the deployment is shown without endpoints. After a create, the
    deployment exists either way, and its ID must not get lost.
    """
    try:
        found = lifecycle.endpoints(client, deployment)
    except APIError as exc:
        log.warning(
            "cannot read the gateways of deployment %s, so its endpoints are not shown: %s",
            deployment.id,
            exc,
        )
        found = ()
    view = _jsonable(deployment)
    view["endpoints"] = list(found)
    view["rollout_base_url"] = lifecycle.rollout_base_url(found[0]) if found else None
    return view


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cwtrain")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API origin")
    parser.add_argument(
        "--token",
        default=None,
        help=f"API token (default: ${TOKEN_ENV_VAR}, else ${TOKEN_ENV_VARS[1]})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="group", required=True)

    dep = sub.add_parser(
        "deployment", help="create, show and delete the deployment rollouts come from"
    ).add_subparsers(dest="cmd", required=True)

    p = dep.add_parser("create", help="create a deployment from a request body in a JSON file")
    p.add_argument(
        "-f", "--file", required=True, help="CreateDeploymentRequest JSON; '-' reads stdin"
    )
    p.add_argument("--wait", action="store_true", help="block until the deployment is READY")
    p.add_argument("--timeout", type=_positive_float, default=3600.0)

    p = dep.add_parser("get", help="show a deployment, its hot-load settings and its endpoints")
    p.add_argument("deployment")

    p = dep.add_parser("delete", help="delete a deployment")
    p.add_argument("deployment")
    p.add_argument("--wait", action="store_true", help="block until the deployment is gone")
    p.add_argument("--timeout", type=_positive_float, default=600.0)
    p.add_argument(
        "--ignore-not-found", action="store_true", help="succeed when there is no such deployment"
    )

    p = dep.add_parser(
        "parameters", help="list the runtime versions, instance types and engine options"
    )
    p.add_argument(
        "--gateway",
        action="append",
        dest="gateways",
        help="only what the gateway's zone offers; repeat for several gateways",
    )

    hot = sub.add_parser("hotload", help="checkpoint hot-load operations").add_subparsers(
        dest="cmd", required=True
    )

    p = hot.add_parser("inspect", help="validate a local checkpoint directory")
    p.add_argument("checkpoint")
    p.add_argument("--previous-identity")

    p = hot.add_parser("publish", help="upload a checkpoint and hot-load it")
    p.add_argument("--deployment", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--identity", required=True)
    p.add_argument("--previous-identity")
    p.add_argument("--reset-prompt-cache", action="store_true")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--timeout", type=_positive_float, default=1800.0)
    p.add_argument("--workers", type=_positive_int, default=8)

    p = hot.add_parser("trigger", help="hot-load an already uploaded snapshot")
    p.add_argument("--deployment", required=True)
    p.add_argument("--identity", required=True)
    p.add_argument("--previous-identity")
    p.add_argument("--reset-prompt-cache", action="store_true")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--timeout", type=_positive_float, default=1800.0)

    p = hot.add_parser("status", help="show a hot-load and its replicas")
    p.add_argument("hot_load_id")

    p = hot.add_parser("wait", help="block until a hot-load is terminal")
    p.add_argument("hot_load_id")
    p.add_argument("--timeout", type=_positive_float, default=1800.0)

    p = hot.add_parser("list", help="list hot-loads")
    p.add_argument("--deployment")
    p.add_argument("--limit", type=_positive_int, default=20)

    p = hot.add_parser(
        "check",
        help="check that the storage credentials can list, read and write the snapshot prefix",
    )
    p.add_argument("--deployment", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for name in _CHATTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET if args.verbose else logging.WARNING)
    try:
        return _run(args)
    except CWTrainError as exc:
        return _error(str(exc), verbose=args.verbose)
    except (NoCredentialsError, PartialCredentialsError, CredentialRetrievalError, ProfileNotFound):
        return _error(
            "object-storage credentials are missing or unavailable; configure "
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY or a valid AWS profile",
            verbose=args.verbose,
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        return _error(
            f"object storage rejected {exc.operation_name} ({code}); check storage credentials, "
            "the deployment's bucket and prefix, and bucket read/write permissions",
            verbose=args.verbose,
        )
    except (StorageConnectionError, HTTPClientError, S3UploadFailedError, RetriesExceededError):
        return _error(
            "object-storage transfer failed; check the endpoint (CW_S3_ENDPOINT or the AWS "
            "endpoint settings), network connectivity, "
            "storage credentials and bucket permissions, then retry",
            verbose=args.verbose,
        )


def _error(message: str, *, verbose: bool) -> int:
    print(f"error: {message}", file=sys.stderr)
    if verbose:
        traceback.print_exc(file=sys.stderr)
    return 1


def _run(args: argparse.Namespace) -> int:
    if args.group == "hotload" and args.cmd == "inspect":
        _print(inspect_snapshot(args.checkpoint, previous_identity=args.previous_identity))
        return 0

    token = args.token or token_from_env()
    if not token or not token.strip():
        raise CWTrainError(
            f"a CoreWeave API token is required: set {TOKEN_ENV_VAR} (or {TOKEN_ENV_VARS[1]}) "
            f"or pass --token before {args.group}"
        )
    with InferenceClient(token.strip(), base_url=args.base_url) as client:
        if args.group == "deployment":
            return _run_deployment(args, client)
        return _run_hotload(args, client)


def _run_deployment(args: argparse.Namespace, client: InferenceClient) -> int:
    if args.cmd == "create":
        return _create_deployment(args, client)
    if args.cmd == "get":
        _print(_deployment_view(client, client.get_deployment(args.deployment)))
        return 0
    if args.cmd == "delete":
        if args.wait:
            existed = lifecycle.delete_deployment(client, args.deployment, timeout=args.timeout)
        else:
            try:
                client.delete_deployment(args.deployment)
                existed = True
            except NotFoundError:
                existed = False
        if not existed and not args.ignore_not_found:
            raise CWTrainError(f"deployment {args.deployment} not found")
        result = ("deleted" if args.wait else "deleting") if existed else "not_found"
        _print({"id": args.deployment, "result": result})
        return 0
    if args.cmd == "parameters":
        _print(client.get_deployment_parameters(args.gateways))
        return 0
    raise SystemExit(2)  # pragma: no cover - argparse guards this


def _create_deployment(args: argparse.Namespace, client: InferenceClient) -> int:
    spec = _read_request(args.file)
    if not spec.get("hotLoad"):
        log.warning(
            "%s has no hotLoad section, and hot-loading cannot be enabled after creation",
            args.file,
        )
    # Choose the ID here so that it is in the log even if the response never arrives.
    spec["id"] = spec.get("id") or str(uuid.uuid4())
    log.info("creating deployment %r as %s", spec.get("name"), spec["id"])
    try:
        deployment = client.create_deployment(spec)
    except AlreadyExistsError as exc:
        same_name = [d for d in client.list_deployments() if d.name == spec.get("name")]
        if not same_name:
            raise
        raise CWTrainError(
            f"deployment {spec.get('name')!r} already exists as {same_name[0].id}; "
            "`cwtrain deployment get` shows it, or delete it or choose another name"
        ) from exc
    if args.wait:
        try:
            deployment = lifecycle.wait_for_deployment(client, deployment.id, timeout=args.timeout)
        except CWTrainError:
            # The deployment exists and holds GPUs either way. Print it, with its ID, before the
            # error. Use the latest state from the API if it can still be read.
            try:
                deployment = client.get_deployment(deployment.id)
            except CWTrainError:
                pass
            _print(_deployment_view(client, deployment))
            raise
    _print(_deployment_view(client, deployment))
    return 0


def _run_hotload(args: argparse.Namespace, client: InferenceClient) -> int:
    if args.cmd == "status":
        _print(
            {
                "hot_load": client.get_hot_load(args.hot_load_id),
                "replicas": client.list_hot_load_replicas(args.hot_load_id),
            }
        )
        return 0
    if args.cmd == "list":
        items = []
        for hl in client.list_hot_loads(args.deployment):
            items.append(hl)
            if len(items) >= args.limit:
                break
        _print(items)
        return 0

    if args.cmd == "check":
        loader = HotLoader(client, args.deployment)
        uri = loader.check_snapshot_access()
        cfg = loader.deployment.hot_load
        assert cfg is not None
        _print(
            {
                "deployment": args.deployment,
                "bucket": cfg.bucket,
                "path_prefix": cfg.path_prefix,
                "uri": uri,
                "result": "ok",
            }
        )
        return 0

    loader = HotLoader(client, getattr(args, "deployment", None) or "", timeout=args.timeout)
    if args.cmd == "publish":
        hl = loader.publish(
            args.checkpoint,
            args.identity,
            previous_identity=args.previous_identity,
            prompt_cache_policy=_prompt_cache_policy(args),
            wait=not args.no_wait,
            workers=args.workers,
        )
    elif args.cmd == "trigger":
        hl = loader.trigger(
            args.identity,
            previous_identity=args.previous_identity,
            prompt_cache_policy=_prompt_cache_policy(args),
        )
        if not args.no_wait:
            hl = loader.wait(hl.id)
    elif args.cmd == "wait":
        hl = loader.wait(args.hot_load_id)
    else:  # pragma: no cover - argparse guards this
        raise SystemExit(2)
    _print(hl)
    return 0 if hl.completed or not hl.terminal else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
