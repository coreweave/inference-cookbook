# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""REST client for the CoreWeave Dedicated Inference API: gateways, deployments and hot-loads.

Authentication uses a CoreWeave API token, sent as ``Authorization: Bearer``. It is the same
token the ``cwic`` CLI and the Terraform provider use. Pass it explicitly or set
``COREWEAVE_API_TOKEN``. If that is not set, ``CW_API_TOKEN`` (the name in the CoreWeave
documentation) is read.
"""

from __future__ import annotations

import logging
import os
import random
import time
import uuid
from collections.abc import Iterator, Mapping
from typing import Any

import httpx

from cwtrain import __version__
from cwtrain.errors import (
    AlreadyExistsError,
    APIError,
    NotFoundError,
    UnavailableError,
    error_from_response,
    is_transient,
)
from cwtrain.infra.models import (
    Deployment,
    Gateway,
    HotLoad,
    HotLoadReplica,
    IncrementalSnapshotMetadata,
    PromptCachePolicy,
    SnapshotType,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.coreweave.com"
DEFAULT_USER_AGENT = f"cwtrain/{__version__}"
TOKEN_ENV_VAR = "COREWEAVE_API_TOKEN"
#: Variables :func:`token_from_env` reads, in order: cwtrain's own, then the one in the CoreWeave
#: documentation.
TOKEN_ENV_VARS = (TOKEN_ENV_VAR, "CW_API_TOKEN")
API_PREFIX = "/v1alpha1/inference"

# The API caps page sizes at 1000. Ask for the cap so a listing is usually one round trip.
_MAX_PAGE_SIZE = 1000


def token_from_env(environ: Mapping[str, str] | None = None) -> str | None:
    """The API token in the first of :data:`TOKEN_ENV_VARS` that is set and not blank."""
    env = os.environ if environ is None else environ
    for name in TOKEN_ENV_VARS:
        value = env.get(name, "").strip()
        if value:
            return value
    return None


class InferenceClient:
    """Synchronous client for the gateway, deployment and hot-load endpoints.

    Reads and creates are retried with exponential backoff on transient failures
    (``UNAVAILABLE``, HTTP 5xx, connection errors). Every create sends a client-chosen ID, so a
    retry finds the resource an earlier attempt created instead of creating a second one.

    Args:
        token: CoreWeave API token. Defaults to ``$COREWEAVE_API_TOKEN``, then ``$CW_API_TOKEN``.
        base_url: API origin. Defaults to the production API.
        timeout: Per-request timeout in seconds.
        max_retries: Attempts for retryable failures, including the first.
        transport: Optional ``httpx`` transport, used by tests to stub the API.
        user_agent: ``User-Agent`` header value.
        headers: Extra headers sent on every request. When given, ``token`` becomes optional.
        verify: TLS verification, passed to ``httpx``; ``False`` for self-signed dev servers.
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 4,
        transport: httpx.BaseTransport | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        headers: dict[str, str] | None = None,
        verify: bool | str = True,
    ) -> None:
        token = token or token_from_env()
        if not token and not headers:
            raise ValueError(
                f"a CoreWeave API token is required: pass token= or set {TOKEN_ENV_VAR} "
                f"(or {TOKEN_ENV_VARS[1]})"
            )
        self.base_url = base_url.rstrip("/")
        self.max_retries = max(1, max_retries)
        base_headers = {"Accept": "application/json", "User-Agent": user_agent}
        if token:
            base_headers["Authorization"] = f"Bearer {token}"
        # Extra headers support API servers that authenticate without the Bearer token, so the
        # token is optional when headers are given. The production API accepts only the token.
        base_headers.update(headers or {})
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers=base_headers,
            verify=verify,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> InferenceClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------------------------- gateways

    def create_gateway(self, spec: dict[str, Any]) -> Gateway:
        """Create a gateway from a ``CreateGatewayRequest`` JSON body.

        Safe to retry in the same way as :meth:`create_deployment`.
        """
        return Gateway.from_json(self._create("gateway", spec))

    def get_gateway(self, gateway_id: str) -> Gateway:
        data = self._request("GET", f"{API_PREFIX}/gateways/{gateway_id}")
        return Gateway.from_json(data.get("gateway") or {})

    def list_gateways(self) -> list[Gateway]:
        data = self._request("GET", f"{API_PREFIX}/gateways")
        return [Gateway.from_json(item) for item in data.get("items") or []]

    def delete_gateway(self, gateway_id: str) -> None:
        """Start deleting a gateway; :func:`cwtrain.infra.lifecycle.delete_gateway` waits."""
        self._request("DELETE", f"{API_PREFIX}/gateways/{gateway_id}")

    def get_gateway_parameters(self) -> dict[str, Any]:
        """Return the availability zones accepted by gateway creation."""
        return self._request("GET", f"{API_PREFIX}/gateways/parameters")

    # ----------------------------------------------------------------- deployments

    def get_deployment_parameters(self, gateway_ids: list[str] | None = None) -> dict[str, Any]:
        """Return runtime versions, allowed config keys and instance types.

        Scope to the intended gateways so the returned options apply to their zones.
        """
        params = {"gatewayIds": gateway_ids} if gateway_ids else None
        return self._request("GET", f"{API_PREFIX}/deployments/parameters", params=params)

    def get_deployment(self, deployment_id: str) -> Deployment:
        data = self._request("GET", f"{API_PREFIX}/deployments/{deployment_id}")
        return Deployment.from_json(data.get("deployment") or {})

    def list_deployments(self, gateway_id: str | None = None) -> list[Deployment]:
        params = {"parentGatewayId": gateway_id} if gateway_id else None
        data = self._request("GET", f"{API_PREFIX}/deployments", params=params)
        return [Deployment.from_json(item) for item in data.get("items") or []]

    def create_deployment(self, spec: dict[str, Any]) -> Deployment:
        """Create a deployment from a ``CreateDeploymentRequest`` JSON body.

        ``spec`` uses the API's camelCase field names (``gatewayIds``, ``runtime``, ``model``,
        ``hotLoad``, and so on). Hot-loading can only be enabled at creation, not by a later
        update. A deployment that will be hot-loaded needs the ``hotLoad`` section in this call.

        The request sends an ``id``: the one in ``spec``, or a new UUID. This makes the call safe
        to retry when the outcome is unknown. If an earlier attempt created the deployment, the
        API answers ``ALREADY_EXISTS``, and the deployment is fetched and returned.

        Raises:
            AlreadyExistsError: another deployment in the organization has the same name.
        """
        return Deployment.from_json(self._create("deployment", spec))

    def delete_deployment(self, deployment_id: str) -> None:
        """Start deleting a deployment; :func:`cwtrain.infra.lifecycle.delete_deployment` waits."""
        self._request("DELETE", f"{API_PREFIX}/deployments/{deployment_id}")

    # ------------------------------------------------------------------- hot-loads

    def create_hot_load(
        self,
        deployment_id: str,
        identity: str,
        *,
        snapshot_type: SnapshotType = SnapshotType.FULL,
        incremental: IncrementalSnapshotMetadata | None = None,
        prompt_cache_policy: PromptCachePolicy = PromptCachePolicy.UNSPECIFIED,
        hot_load_id: str | None = None,
    ) -> HotLoad:
        """Trigger a hot-load of ``identity`` on ``deployment_id``.

        A ``hot_load_id`` is always sent: the one supplied, or a new UUID. This makes the call
        safe to retry. If the server answers ``ALREADY_EXISTS`` for that ID, the existing hot-load
        is fetched and returned.

        Raises:
            InvalidArgumentError: the request violates a validation rule.
            NotFoundError: the deployment does not exist.
            HotLoadInProgressError: another hot-load is active on the deployment.
            FailedPreconditionError: the deployment is not READY, hot-load is not enabled,
                the chain is too long, the predecessor is unknown, or a snapshot is missing files.
        """
        if snapshot_type is SnapshotType.INCREMENTAL and incremental is None:
            raise ValueError("incremental snapshots require IncrementalSnapshotMetadata")
        if snapshot_type is SnapshotType.FULL and incremental is not None:
            raise ValueError("full snapshots must not carry IncrementalSnapshotMetadata")

        hot_load_id = hot_load_id or str(uuid.uuid4())
        body: dict[str, Any] = {
            "deploymentId": deployment_id,
            "hotLoadId": hot_load_id,
            "identity": identity,
            "type": snapshot_type.wire,
        }
        if incremental is not None:
            body["incrementalSnapshotMetadata"] = incremental.to_json()
        if prompt_cache_policy is not PromptCachePolicy.UNSPECIFIED:
            body["promptCachePolicy"] = prompt_cache_policy.wire

        try:
            data = self._request("POST", f"{API_PREFIX}/hotLoads", json=body, retry=True)
        except AlreadyExistsError:
            # An earlier attempt of this call succeeded; the API contract says to GET it by ID.
            log.info("hot-load %s already exists; fetching it", hot_load_id)
            return self.get_hot_load(hot_load_id)
        return HotLoad.from_json(data.get("hotLoad") or {})

    def get_hot_load(self, hot_load_id: str) -> HotLoad:
        data = self._request("GET", f"{API_PREFIX}/hotLoads/{hot_load_id}")
        return HotLoad.from_json(data.get("hotLoad") or {})

    def list_hot_loads(
        self, deployment_id: str | None = None, *, page_size: int = _MAX_PAGE_SIZE
    ) -> Iterator[HotLoad]:
        """Yield hot-loads, newest first, across all pages."""
        params: dict[str, Any] = {"maxPageSize": page_size}
        if deployment_id:
            params["parentDeploymentId"] = deployment_id
        for page in self._paginate(f"{API_PREFIX}/hotLoads", params):
            for item in page.get("hotLoads") or []:
                yield HotLoad.from_json(item)

    def list_hot_load_replicas(
        self, hot_load_id: str, *, page_size: int = _MAX_PAGE_SIZE
    ) -> list[HotLoadReplica]:
        """Return every replica status of a hot-load, ordered by (zone, name)."""
        replicas: list[HotLoadReplica] = []
        for page in self._paginate(
            f"{API_PREFIX}/hotLoads/{hot_load_id}/replicas", {"maxPageSize": page_size}
        ):
            replicas.extend(HotLoadReplica.from_json(r) for r in page.get("replicas") or [])
        return replicas

    def cancel_hot_load(self, hot_load_id: str) -> HotLoad:
        """Cancel a hot-load."""
        data = self._request("POST", f"{API_PREFIX}/hotLoads/{hot_load_id}:cancel", json={})
        return HotLoad.from_json(data.get("hotLoad") or {})

    # -------------------------------------------------------------------- plumbing

    def _create(self, kind: str, spec: dict[str, Any]) -> dict[str, Any]:
        """POST a gateway or deployment under a client-chosen ID and return the wire object.

        ``ALREADY_EXISTS`` means the ID or the name is taken. If the resource with our ID has the
        requested name, an earlier attempt of this call created it, so it is returned. Anything
        else is a conflict with another resource.
        """
        body = dict(spec)
        body["id"] = resource_id = body.get("id") or str(uuid.uuid4())
        name = body.get("name")
        path = f"{API_PREFIX}/{kind}s"
        try:
            data = self._request("POST", path, json=body, retry=True)
        except AlreadyExistsError as exc:
            try:
                existing = self._request("GET", f"{path}/{resource_id}").get(kind) or {}
            except NotFoundError:
                message = f"a {kind} named {name!r} already exists in this organization"
            else:
                existing_name = (existing.get("spec") or {}).get("name")
                if existing_name == name:
                    log.info("%s %s already exists; fetched it", kind, resource_id)
                    return existing
                message = f"{kind} {resource_id} already exists, named {existing_name!r}"
            raise AlreadyExistsError(
                exc.code,
                f"{message} ({exc.message})",
                http_status=exc.http_status,
                details=exc.details,
            ) from exc
        return data.get(kind) or {}

    def _paginate(self, path: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            page_params = dict(params)
            if token:
                page_params["pageToken"] = token
            page = self._request("GET", path, params=page_params)
            yield page
            token = page.get("nextPageToken") or None
            if not token:
                return
            if token in seen_tokens:
                raise APIError(
                    "internal", f"GET {path} returned page token {token!r} twice; stopping"
                )
            seen_tokens.add(token)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        retry: bool | None = None,
    ) -> dict[str, Any]:
        retry = (method == "GET") if retry is None else retry
        attempts = self.max_retries if retry else 1
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._http.request(method, path, params=params, json=json)
            except httpx.TransportError as exc:
                last_exc = UnavailableError("unavailable", str(exc))
            else:
                if response.is_success:
                    if not response.content:
                        return {}
                    try:
                        return response.json()
                    except ValueError:
                        # A proxy or portal sent 2xx with a non-JSON body: retry like a 503.
                        last_exc = UnavailableError(
                            "unavailable",
                            f"HTTP {response.status_code} with a non-JSON body: "
                            f"{response.text[:200]!r}",
                        )
                else:
                    err = self._error_from(response)
                    if not is_transient(err):
                        raise err
                    last_exc = err
            if attempt < attempts:
                sleep = delay * (1 + random.random() * 0.25)
                log.warning(
                    "%s %s failed (%s); retrying in %.1fs (%d/%d)",
                    method,
                    path,
                    last_exc,
                    sleep,
                    attempt,
                    attempts,
                )
                time.sleep(sleep)
                delay = min(delay * 2, 8.0)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _error_from(response: httpx.Response) -> APIError:
        body: Any
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return error_from_response(response.status_code, body)
