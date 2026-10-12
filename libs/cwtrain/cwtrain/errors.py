# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The base exception of cwtrain, and the errors the CoreWeave API returns.

Each API error class maps to one gRPC / Connect status code. Catch the specific class (for example
a hot-load already in progress) instead of parsing messages. Errors specific to a subsystem live
next to it, such as the hot-load errors in :mod:`cwtrain.hotload.errors`.
"""

from __future__ import annotations

from typing import Any


class CWTrainError(Exception):
    """Base class for every error raised by cwtrain."""


class APIError(CWTrainError):
    """The CoreWeave API rejected a request.

    ``code`` is the canonical lower-snake-case status name (``failed_precondition``), whether the
    server sent it as a Connect string or a numeric gRPC code.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        http_status: int | None = None,
        details: list[Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details or []


class InvalidArgumentError(APIError):
    """The request was malformed or violates a validation rule."""


class NotFoundError(APIError):
    """The gateway, deployment or hot-load does not exist in the caller's organization."""


class AlreadyExistsError(APIError):
    """The resource already exists: same ID, or for a gateway or deployment, same name."""


class FailedPreconditionError(APIError):
    """The resource is not in a state that admits the request."""


class HotLoadInProgressError(FailedPreconditionError):
    """Another hot-load is still active on the deployment."""


class PermissionDeniedError(APIError):
    """The caller is not allowed to act on this resource."""


class UnauthenticatedError(APIError):
    """The API token is missing or invalid."""


class UnavailableError(APIError):
    """The service could not complete the request; retry later."""


class UnimplementedError(APIError):
    """The RPC exists but is not implemented."""


#: HTTP statuses that a retry can fix. 501 (UNIMPLEMENTED) is a 5xx that never gets better.
TRANSIENT_HTTP_STATUSES = frozenset({500, 502, 503, 504})


def is_transient(err: BaseException) -> bool:
    """Return whether a retry may succeed.

    True for ``UNAVAILABLE`` (including a lost connection), and for an HTTP 500, 502, 503 or 504
    whose code is not ``UNIMPLEMENTED``, such as a load balancer's 502 without a code. The client
    retries these errors itself. Pollers such as :mod:`cwtrain.infra.lifecycle` also continue
    through them.
    """
    if isinstance(err, UnavailableError):
        return True
    return (
        isinstance(err, APIError)
        and err.http_status in TRANSIENT_HTTP_STATUSES
        and err.code != "unimplemented"
    )


_CODE_TO_ERROR: dict[str, type[APIError]] = {
    "invalid_argument": InvalidArgumentError,
    "not_found": NotFoundError,
    "already_exists": AlreadyExistsError,
    "failed_precondition": FailedPreconditionError,
    "permission_denied": PermissionDeniedError,
    "unauthenticated": UnauthenticatedError,
    "unavailable": UnavailableError,
    "unimplemented": UnimplementedError,
}

# gRPC numeric codes, for servers that transcode errors as google.rpc.Status.
_GRPC_CODES: dict[int, str] = {
    1: "canceled",
    2: "unknown",
    3: "invalid_argument",
    4: "deadline_exceeded",
    5: "not_found",
    6: "already_exists",
    7: "permission_denied",
    8: "resource_exhausted",
    9: "failed_precondition",
    10: "aborted",
    11: "out_of_range",
    12: "unimplemented",
    13: "internal",
    14: "unavailable",
    15: "data_loss",
    16: "unauthenticated",
}

_HTTP_TO_CODE: dict[int, str] = {
    400: "invalid_argument",
    401: "unauthenticated",
    403: "permission_denied",
    404: "not_found",
    409: "already_exists",
    412: "failed_precondition",
    429: "resource_exhausted",
    501: "unimplemented",
    503: "unavailable",
}


def error_from_response(http_status: int, body: Any) -> APIError:
    """Build the most specific :class:`APIError` for an error response body."""
    code: str | None = None
    message = ""
    details: list[Any] = []
    if isinstance(body, dict):
        raw_code = body.get("code")
        if isinstance(raw_code, int):
            code = _GRPC_CODES.get(raw_code)
        elif isinstance(raw_code, str):
            code = raw_code.lower()
        message = str(body.get("message") or "")
        details = list(body.get("details") or [])
    elif isinstance(body, str):
        message = body
    if not code:
        code = _HTTP_TO_CODE.get(http_status, "internal" if http_status >= 500 else "unknown")
    if not message:
        message = f"HTTP {http_status}"

    cls = _CODE_TO_ERROR.get(code, APIError)
    if cls is FailedPreconditionError and "already in progress" in message:
        cls = HotLoadInProgressError
    return cls(code, message, http_status=http_status, details=details)
