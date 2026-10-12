# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Waiting for gateways and deployments to come up or go away, and finding the rollout URL."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from cwtrain.errors import (
    APIError,
    FailedPreconditionError,
    NotFoundError,
    PermissionDeniedError,
    UnavailableError,
)
from cwtrain.infra import Deployment, Gateway, lifecycle
from cwtrain.infra.lifecycle import LifecycleError

GONE = NotFoundError("not_found", "gone")
UNAVAILABLE = UnavailableError("unavailable", "upstream reset")
# a load balancer's answer without a gRPC code, after the client's own retries
BAD_GATEWAY = APIError("internal", "HTTP 502", http_status=502)


def gateway(status: str, *endpoints: str, gateway_id: str = "gw") -> Gateway:
    return Gateway.from_json(
        {
            "spec": {"id": gateway_id, "name": gateway_id},
            "status": {"status": status, "endpoints": list(endpoints)},
        }
    )


def deployment(status: str, *gateway_ids: str, conditions: tuple[dict, ...] = ()) -> Deployment:
    return Deployment.from_json(
        {
            "spec": {"id": "dep", "name": "dep", "gatewayIds": list(gateway_ids)},
            "status": {"status": status, "conditions": list(conditions)},
        }
    )


def test_waiting_polls_until_the_gateway_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep = Mock()
    monkeypatch.setattr(lifecycle.time, "sleep", sleep)
    client = Mock()
    client.get_gateway.side_effect = [gateway("STATUS_CREATING"), gateway("STATUS_READY")]
    assert lifecycle.wait_for_gateway(client, "gw", poll_interval=7).ready
    sleep.assert_called_once_with(7)


@pytest.mark.parametrize("status", ["STATUS_FAILED", "STATUS_DELETING"])
def test_a_deployment_that_cannot_become_ready_stops_the_wait(status: str) -> None:
    failed = {
        "type": "Ready",
        "status": "STATUS_FALSE",
        "reason": "Failed",
        "message": "image pull failed",
    }
    client = Mock()
    client.get_deployment.return_value = deployment(status, conditions=(failed,))
    with pytest.raises(LifecycleError, match=r"is \w+: Ready \(Failed\): image pull failed"):
        lifecycle.wait_for_deployment(client, "dep", timeout=3600)
    client.get_deployment.assert_called_once()


def test_the_wait_times_out_with_the_last_status() -> None:
    client = Mock()
    client.get_deployment.return_value = deployment("STATUS_CREATING")
    with pytest.raises(LifecycleError, match="not READY after 0s: CREATING"):
        lifecycle.wait_for_deployment(client, "dep", timeout=0)


def test_the_wait_rides_out_an_unavailable_api() -> None:
    client = Mock()
    client.get_deployment.side_effect = [UNAVAILABLE, deployment("STATUS_READY")]
    assert lifecycle.wait_for_deployment(client, "dep").ready


def test_the_wait_and_the_delete_ride_out_a_bad_gateway() -> None:
    client = Mock()
    client.get_deployment.side_effect = [BAD_GATEWAY, deployment("STATUS_READY")]
    assert lifecycle.wait_for_deployment(client, "dep").ready
    client = Mock()
    client.delete_deployment.side_effect = [BAD_GATEWAY, None]
    client.get_deployment.side_effect = GONE
    assert lifecycle.delete_deployment(client, "dep") is True
    assert client.delete_deployment.call_count == 2


def test_an_error_a_retry_cannot_fix_ends_the_wait_and_the_delete() -> None:
    denied = PermissionDeniedError("permission_denied", "not allowed", http_status=403)
    client = Mock()
    client.get_deployment.side_effect = denied
    with pytest.raises(PermissionDeniedError):
        lifecycle.wait_for_deployment(client, "dep")
    client.get_deployment.assert_called_once()
    client.delete_deployment.side_effect = denied
    with pytest.raises(PermissionDeniedError):
        lifecycle.delete_deployment(client, "dep")
    client.delete_deployment.assert_called_once()


def test_deleting_waits_until_the_deployment_is_gone() -> None:
    client = Mock()
    client.get_deployment.side_effect = [deployment("STATUS_DELETING"), GONE]
    assert lifecycle.delete_deployment(client, "dep") is True
    client.delete_deployment.assert_called_once_with("dep")


def test_deleting_what_does_not_exist_reports_it() -> None:
    client = Mock()
    client.delete_deployment.side_effect = GONE
    assert lifecycle.delete_deployment(client, "dep") is False
    client.get_deployment.assert_not_called()


def test_a_delete_whose_answer_was_lost_counts_once_the_deployment_is_gone() -> None:
    # the first delete reached the API but its answer did not; the retry finds nothing left
    client = Mock()
    client.delete_deployment.side_effect = [UNAVAILABLE, GONE]
    assert lifecycle.delete_deployment(client, "dep") is True
    assert client.delete_deployment.call_count == 2


def test_deleting_times_out_while_the_deployment_remains() -> None:
    client = Mock()
    client.get_deployment.return_value = deployment("STATUS_DELETING")
    with pytest.raises(LifecycleError, match="deployment dep still exists after 0s"):
        lifecycle.delete_deployment(client, "dep", timeout=0)


def test_a_gateway_is_deleted_once_its_deployments_are_gone() -> None:
    in_use = FailedPreconditionError(
        "failed_precondition", "gateway has associated deployments, cannot delete"
    )
    client = Mock()
    client.delete_gateway.side_effect = [in_use, UNAVAILABLE, None]
    client.get_gateway.side_effect = GONE
    assert lifecycle.delete_gateway(client, "gw") is True
    assert client.delete_gateway.call_count == 3


def test_a_refused_deployment_delete_fails_at_once() -> None:
    # only a gateway waits out FAILED_PRECONDITION (deployments still using it)
    refused = FailedPreconditionError("failed_precondition", "deployment cannot be deleted now")
    client = Mock()
    client.delete_deployment.side_effect = refused
    with pytest.raises(FailedPreconditionError, match="cannot be deleted now"):
        lifecycle.delete_deployment(client, "dep", timeout=0.2)  # waiting would end in a timeout
    client.delete_deployment.assert_called_once()


def test_endpoints_and_the_rollout_url() -> None:
    client = Mock()
    client.get_gateway.side_effect = lambda gateway_id: {
        "a": gateway("STATUS_READY", "https://a.example", "https://b.example", gateway_id="a"),
        "b": gateway("STATUS_READY", "https://b.example", gateway_id="b"),
    }[gateway_id]
    found = lifecycle.endpoints(client, deployment("STATUS_READY", "a", "b"))
    assert found == ("https://a.example", "https://b.example")
    assert lifecycle.endpoints(client, deployment("STATUS_READY")) == ()


@pytest.mark.parametrize(
    "endpoint", ["https://gw.example", "https://gw.example/", "gw.example", "https://gw.example/v1"]
)
def test_the_rollout_url_is_the_endpoints_v1_route(endpoint: str) -> None:
    assert lifecycle.rollout_base_url(endpoint) == "https://gw.example/v1"
