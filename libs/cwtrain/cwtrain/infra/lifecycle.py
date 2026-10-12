# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Wait for gateways and deployments to become READY or to be deleted, and find rollout URLs.

:class:`~cwtrain.infra.client.InferenceClient` makes one API call per method. Its creates and
deletes return when the API accepts them. A training run needs the outcome: a serving deployment
before the first hot-load, and at the end, confirmation that it is gone and its GPUs are released.
These functions poll the API until that outcome.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from cwtrain.errors import (
    APIError,
    CWTrainError,
    FailedPreconditionError,
    NotFoundError,
    is_transient,
)
from cwtrain.infra.client import InferenceClient
from cwtrain.infra.models import Condition, ConditionStatus, Deployment, Gateway, ResourceStatus

log = logging.getLogger(__name__)


class LifecycleError(CWTrainError):
    """A gateway or deployment did not reach the state waited for, or did not go away."""


def wait_for_gateway(
    client: InferenceClient, gateway_id: str, *, timeout: float = 600.0, poll_interval: float = 5.0
) -> Gateway:
    """Block until the gateway is READY and return it.

    Raises:
        LifecycleError: the gateway FAILED, is being deleted, or is not READY within ``timeout``.
    """
    return _wait_ready("gateway", client.get_gateway, gateway_id, timeout, poll_interval)


def wait_for_deployment(
    client: InferenceClient,
    deployment_id: str,
    *,
    timeout: float = 3600.0,
    poll_interval: float = 10.0,
) -> Deployment:
    """Block until the deployment is READY and return it.

    A new deployment loads its model before it serves. This takes minutes, more for large models.

    Raises:
        LifecycleError: the deployment FAILED, is being deleted, or is not READY within
            ``timeout``.
    """
    return _wait_ready("deployment", client.get_deployment, deployment_id, timeout, poll_interval)


def delete_gateway(
    client: InferenceClient, gateway_id: str, *, timeout: float = 600.0, poll_interval: float = 5.0
) -> bool:
    """Delete the gateway and wait until it is gone; ``False`` when there was nothing to delete.

    The API refuses to delete a gateway while deployments use it, including deployments that are
    being deleted. This function retries the delete until ``timeout``.

    Raises:
        LifecycleError: the gateway still exists after ``timeout``.
    """
    return _delete(
        "gateway",
        client.delete_gateway,
        client.get_gateway,
        gateway_id,
        timeout,
        poll_interval,
        wait_while_in_use=True,
    )


def delete_deployment(
    client: InferenceClient,
    deployment_id: str,
    *,
    timeout: float = 600.0,
    poll_interval: float = 5.0,
) -> bool:
    """Delete the deployment and wait until it is gone; ``False`` when there was nothing to delete.

    Raises:
        LifecycleError: the deployment still exists after ``timeout``.
    """
    return _delete(
        "deployment",
        client.delete_deployment,
        client.get_deployment,
        deployment_id,
        timeout,
        poll_interval,
    )


def endpoints(client: InferenceClient, deployment: Deployment) -> tuple[str, ...]:
    """The public endpoints of the deployment's gateways, each gateway's primary endpoint first."""
    found: list[str] = []
    for gateway_id in deployment.gateway_ids:
        for endpoint in client.get_gateway(gateway_id).endpoints:
            if endpoint not in found:
                found.append(endpoint)
    return tuple(found)


def rollout_base_url(endpoint: str) -> str:
    """The OpenAI-compatible base URL of a gateway endpoint, in ``CWTRAIN_ROLLOUT_BASE_URL`` form.

    The API reports endpoints as ``https://<host>``. The OpenAI-compatible routes are under ``/v1``.
    """
    url = endpoint.strip().rstrip("/")
    if "://" not in url:
        url = "https://" + url
    return url if url.endswith("/v1") else url + "/v1"


def _wait_ready[R: (Gateway, Deployment)](
    kind: str,
    get: Callable[[str], R],
    resource_id: str,
    timeout: float,
    poll_interval: float,
) -> R:
    deadline = time.monotonic() + timeout
    last: ResourceStatus | None = None
    seen = "not read yet"
    while True:
        try:
            resource = get(resource_id)
        except APIError as exc:
            if not is_transient(exc):
                raise
            # The client already retried; a longer outage only delays the wait.
            log.warning("cannot read %s %s (%s); still waiting", kind, resource_id, exc)
        else:
            if resource.ready:
                return resource
            seen = resource.status.name + _problems(resource.conditions)
            if resource.status in (ResourceStatus.FAILED, ResourceStatus.DELETING):
                raise LifecycleError(f"{kind} {resource_id} is {seen}")
            if resource.status is not last:
                log.info("%s %s is %s; waiting for READY", kind, resource_id, seen)
                last = resource.status
        if time.monotonic() >= deadline:
            raise LifecycleError(f"{kind} {resource_id} is not READY after {timeout:.0f}s: {seen}")
        time.sleep(poll_interval)


def _delete(
    kind: str,
    delete: Callable[[str], None],
    get: Callable[[str], object],
    resource_id: str,
    timeout: float,
    poll_interval: float,
    *,
    wait_while_in_use: bool = False,
) -> bool:
    deadline = time.monotonic() + timeout
    requested = False
    # True when a delete got no answer, so the API may have done it.
    maybe_requested = False
    while True:
        try:
            if not requested:
                delete(resource_id)
                requested = True
                log.info("deleting %s %s", kind, resource_id)
            get(resource_id)
        except NotFoundError:
            # Gone. After an unanswered delete, that delete most likely removed it. A resource
            # that never existed answers the first delete with NOT_FOUND.
            return requested or maybe_requested
        except FailedPreconditionError as exc:
            if requested or not wait_while_in_use:
                raise
            # The gateway is still in use by deployments that are being deleted, or will be.
            log.info("%s %s cannot be deleted yet: %s", kind, resource_id, exc.message)
        except APIError as exc:
            if not is_transient(exc):
                raise
            # A repeated delete is safe: the API answers it like the first one.
            maybe_requested = maybe_requested or not requested
            log.warning("%s %s: %s; retrying", kind, resource_id, exc)
        if time.monotonic() >= deadline:
            raise LifecycleError(f"{kind} {resource_id} still exists after {timeout:.0f}s")
        time.sleep(poll_interval)


def _problems(conditions: tuple[Condition, ...]) -> str:
    """The conditions that are not met, as ``: Type (Reason): message; ...``, or nothing."""
    unmet = [c for c in conditions if c.status is ConditionStatus.FALSE and (c.reason or c.message)]
    details = "; ".join(
        c.type + (f" ({c.reason})" if c.reason else "") + (f": {c.message}" if c.message else "")
        for c in unmet
    )
    return f": {details}" if details else ""
