# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import json

import httpx
import pytest

from cwtrain.errors import (
    AlreadyExistsError,
    APIError,
    FailedPreconditionError,
    HotLoadInProgressError,
    InvalidArgumentError,
    NotFoundError,
    UnauthenticatedError,
    UnavailableError,
    UnimplementedError,
    is_transient,
)
from cwtrain.infra import (
    HotLoadState,
    IncrementalSnapshotMetadata,
    InferenceClient,
    PromptCachePolicy,
    SnapshotType,
)
from tests.conftest import DEPLOYMENT_ID, FakeAPI, error_json, hot_load_json, replica_json


def test_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COREWEAVE_API_TOKEN", raising=False)
    with pytest.raises(ValueError):
        InferenceClient()
    monkeypatch.setenv("COREWEAVE_API_TOKEN", "from-env")
    with InferenceClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    ) as c:
        assert c._http.headers["Authorization"] == "Bearer from-env"


def test_get_deployment(client: InferenceClient, api: FakeAPI) -> None:
    dep = client.get_deployment(DEPLOYMENT_ID)
    assert dep.id == DEPLOYMENT_ID and dep.hot_load is not None
    assert api.requests[-1].headers["Authorization"] == "Bearer test-token"


def test_create_hot_load_sends_wire_enums_and_client_id(
    client: InferenceClient, api: FakeAPI
) -> None:
    hl = client.create_hot_load(
        DEPLOYMENT_ID,
        "step_000100",
        prompt_cache_policy=PromptCachePolicy.RESET_ALL,
        hot_load_id="22222222-2222-4222-8222-222222222222",
    )
    body = json.loads(api.requests[-1].content)
    assert body == {
        "deploymentId": DEPLOYMENT_ID,
        "hotLoadId": "22222222-2222-4222-8222-222222222222",
        "identity": "step_000100",
        "type": "SNAPSHOT_TYPE_FULL",
        "promptCachePolicy": "PROMPT_CACHE_POLICY_RESET_ALL",
    }
    assert hl.state is HotLoadState.PENDING and hl.identity == "step_000100"


def test_create_incremental_hot_load(client: InferenceClient, api: FakeAPI) -> None:
    client.create_hot_load(
        DEPLOYMENT_ID,
        "step_000200",
        snapshot_type=SnapshotType.INCREMENTAL,
        incremental=IncrementalSnapshotMetadata("step_000100"),
    )
    body = json.loads(api.requests[-1].content)
    assert body["type"] == "SNAPSHOT_TYPE_INCREMENTAL"
    assert body["incrementalSnapshotMetadata"] == {
        "previousSnapshotIdentity": "step_000100",
        "compressionFormat": "COMPRESSION_FORMAT_ZSTD",
        "checksumFormat": "CHECKSUM_FORMAT_ADLER32",
    }
    assert "promptCachePolicy" not in body  # server default applies


def test_create_validates_incremental_metadata_pairing(client: InferenceClient) -> None:
    with pytest.raises(ValueError):
        client.create_hot_load(DEPLOYMENT_ID, "x", snapshot_type=SnapshotType.INCREMENTAL)
    with pytest.raises(ValueError):
        client.create_hot_load(DEPLOYMENT_ID, "x", incremental=IncrementalSnapshotMetadata("y"))


def test_create_already_exists_refetches_by_id(client: InferenceClient, api: FakeAPI) -> None:
    hl_id = "33333333-3333-4333-8333-333333333333"
    api.hot_loads[hl_id] = hot_load_json(hl_id, "step_1", "HOT_LOAD_STATE_COMPLETED")
    hl = client.create_hot_load(DEPLOYMENT_ID, "step_1", hot_load_id=hl_id)
    assert hl.id == hl_id and hl.completed
    assert [r.method for r in api.requests] == ["POST", "GET"]


def test_create_in_progress_maps_to_specific_error(client: InferenceClient, api: FakeAPI) -> None:
    api.hot_loads["active"] = hot_load_json("active", "step_1", "HOT_LOAD_STATE_IN_PROGRESS")
    with pytest.raises(HotLoadInProgressError) as exc:
        client.create_hot_load(DEPLOYMENT_ID, "step_2")
    assert isinstance(exc.value, FailedPreconditionError)
    assert exc.value.code == "failed_precondition"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (400, error_json("invalid_argument", "bad"), InvalidArgumentError),
        (400, error_json(3, "bad"), InvalidArgumentError),
        (404, error_json(5, "gone"), NotFoundError),
        (401, error_json("unauthenticated", "no"), UnauthenticatedError),
        (409, error_json("already_exists", "dup"), AlreadyExistsError),
        (412, error_json("failed_precondition", "not ready"), FailedPreconditionError),
        (501, error_json("unimplemented", "later"), UnimplementedError),
        (404, "plain text", NotFoundError),
    ],
)
def test_error_mapping(api: FakeAPI, client: InferenceClient, status, body, expected) -> None:
    api.queue(
        "GET",
        f"/v1alpha1/inference/hotLoads/{'x' * 4}",
        httpx.Response(status, json=body)
        if not isinstance(body, str)
        else httpx.Response(status, text=body),
    )
    with pytest.raises(expected):
        client.get_hot_load("xxxx")


@pytest.mark.parametrize(
    "err,transient",
    [
        (UnavailableError("unavailable", "connection reset"), True),
        (APIError("internal", "HTTP 502", http_status=502), True),
        (APIError("internal", "HTTP 500", http_status=500), True),
        (UnimplementedError("unimplemented", "not yet", http_status=501), False),
        (APIError("unimplemented", "not yet", http_status=503), False),
        (NotFoundError("not_found", "gone", http_status=404), False),
        (ValueError("not an API error"), False),
    ],
)
def test_is_transient(err: BaseException, transient: bool) -> None:
    assert is_transient(err) is transient


def test_get_retries_on_unavailable_then_succeeds(api: FakeAPI, client: InferenceClient) -> None:
    path = "/v1alpha1/inference/hotLoads/hl1"
    api.hot_loads["hl1"] = hot_load_json("hl1", "s", "HOT_LOAD_STATE_COMPLETED")
    api.queue(
        "GET",
        path,
        httpx.Response(503, json=error_json("unavailable", "try later")),
        httpx.Response(500, text="boom"),
    )
    assert client.get_hot_load("hl1").completed
    assert sum(1 for r in api.requests if r.url.path == path) == 3


def test_get_gives_up_after_max_retries(api: FakeAPI, client: InferenceClient) -> None:
    path = "/v1alpha1/inference/hotLoads/hl1"
    api.queue("GET", path, *[httpx.Response(503, json=error_json("unavailable", "x"))] * 3)
    with pytest.raises(UnavailableError):
        client.get_hot_load("hl1")


def test_connection_errors_are_retried(api: FakeAPI, client: InferenceClient) -> None:
    path = "/v1alpha1/inference/hotLoads/hl1"
    api.hot_loads["hl1"] = hot_load_json("hl1", "s", "HOT_LOAD_STATE_COMPLETED")

    def boom(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    api.queue("GET", path, boom)
    assert client.get_hot_load("hl1").completed


def test_non_retryable_errors_are_not_retried(api: FakeAPI, client: InferenceClient) -> None:
    path = "/v1alpha1/inference/hotLoads/hl1"
    api.queue("GET", path, httpx.Response(404, json=error_json("not_found", "no")))
    with pytest.raises(NotFoundError):
        client.get_hot_load("hl1")
    assert sum(1 for r in api.requests if r.url.path == path) == 1


def test_list_hot_loads_follows_pagination(api: FakeAPI, client: InferenceClient) -> None:
    path = "/v1alpha1/inference/hotLoads"
    api.queue(
        "GET",
        path,
        httpx.Response(
            200,
            json={
                "hotLoads": [hot_load_json("b", "s2", "HOT_LOAD_STATE_COMPLETED")],
                "nextPageToken": "tok",
            },
        ),
        lambda req: (
            httpx.Response(
                200,
                json={
                    "hotLoads": [hot_load_json("a", "s1", "HOT_LOAD_STATE_COMPLETED")],
                    "nextPageToken": "",
                },
            )
            if req.url.params["pageToken"] == "tok"
            else httpx.Response(500)
        ),
    )
    ids = [hl.id for hl in client.list_hot_loads(DEPLOYMENT_ID)]
    assert ids == ["b", "a"]
    first = next(r for r in api.requests if r.url.path == path)
    assert first.url.params["parentDeploymentId"] == DEPLOYMENT_ID
    assert first.url.params["maxPageSize"] == "1000"


def test_list_replicas(api: FakeAPI, client: InferenceClient) -> None:
    api.replicas["hl1"] = [
        replica_json("replica-a", "READY", "s1"),
        replica_json("replica-b", "ERROR", "initial", "quarantined"),
    ]
    reps = client.list_hot_load_replicas("hl1")
    assert [r.name for r in reps] == ["replica-a", "replica-b"]
    assert reps[1].error_reason == "quarantined"
    assert reps[0].zone == "US-EAST-04A"


def test_cancel_is_unimplemented(client: InferenceClient) -> None:
    with pytest.raises(UnimplementedError):
        client.cancel_hot_load("hl1")


GATEWAY_ID = "22222222-2222-4222-8222-222222222222"
GATEWAYS = "/v1alpha1/inference/gateways"
DEPLOYMENTS = "/v1alpha1/inference/deployments"


def gateway_json(gateway_id: str = GATEWAY_ID, name: str = "rl-gateway") -> dict:
    return {
        "spec": {
            "id": gateway_id,
            "name": name,
            "organizationId": "org1",
            "zones": ["US-WEST-04A"],
        },
        "status": {"status": "STATUS_READY", "endpoints": ["https://example.invalid"]},
    }


def posted(api: FakeAPI, path: str) -> list[dict]:
    return [
        json.loads(r.content) for r in api.requests if r.method == "POST" and r.url.path == path
    ]


def test_gateway_calls_return_typed_gateways(api: FakeAPI, client: InferenceClient) -> None:
    spec = {
        "name": "rl-gateway",
        "zones": ["US-WEST-04A"],
        "coreWeaveAuth": {},
        "bodyBasedRouting": {"apiType": "API_TYPE_OPENAI"},
    }
    api.queue("POST", GATEWAYS, httpx.Response(200, json={"gateway": gateway_json()}))
    created = client.create_gateway(spec)
    assert created.id == GATEWAY_ID and created.ready and created.organization_id == "org1"
    assert created.zones == ("US-WEST-04A",) and created.endpoints == ("https://example.invalid",)
    (body,) = posted(api, GATEWAYS)
    assert body == {**spec, "id": body["id"]}  # the request as given, plus a client-chosen ID

    api.gateways[GATEWAY_ID] = {"gateway": gateway_json()}
    assert client.get_gateway(GATEWAY_ID) == created
    api.queue("GET", GATEWAYS, httpx.Response(200, json={"items": [gateway_json()]}))
    listed = client.list_gateways()
    assert listed == [created] and listed[0].raw == gateway_json()

    api.queue("GET", f"{GATEWAYS}/parameters", httpx.Response(200, json={"zones": ["US-WEST-04A"]}))
    assert client.get_gateway_parameters()["zones"] == ["US-WEST-04A"]
    api.queue("GET", f"{DEPLOYMENTS}/parameters", httpx.Response(200, json={}))
    client.get_deployment_parameters(["gw1", "gw2"])
    assert api.requests[-1].url.params.get_list("gatewayIds") == ["gw1", "gw2"]
    api.queue("DELETE", f"{GATEWAYS}/{GATEWAY_ID}", httpx.Response(200, json={}))
    client.delete_gateway(GATEWAY_ID)
    assert api.requests[-1].method == "DELETE"


def test_create_list_delete_deployment(api: FakeAPI, client: InferenceClient) -> None:
    from tests.conftest import deployment_json

    created = deployment_json()
    api.queue("POST", DEPLOYMENTS, httpx.Response(200, json=created))
    api.queue("GET", DEPLOYMENTS, httpx.Response(200, json={"items": [created["deployment"]]}))
    api.queue("DELETE", f"{DEPLOYMENTS}/{DEPLOYMENT_ID}", httpx.Response(200, json={}))

    spec = {"id": DEPLOYMENT_ID, "name": "qwen3-8b-rl", "gatewayIds": ["g"], "hotLoad": {}}
    dep = client.create_deployment(spec)
    assert dep.id == DEPLOYMENT_ID and dep.hot_load is not None
    assert posted(api, DEPLOYMENTS) == [spec]  # a given ID is kept

    listed = client.list_deployments("g")
    assert [d.id for d in listed] == [DEPLOYMENT_ID]
    assert api.requests[-1].url.params["parentGatewayId"] == "g"

    client.delete_deployment(DEPLOYMENT_ID)
    assert api.requests[-1].method == "DELETE"


ALREADY_EXISTS = httpx.Response(409, json=error_json("already_exists", "resource already exists"))


@pytest.mark.parametrize("kind", ["gateway", "deployment"])
def test_a_retried_create_returns_what_the_lost_attempt_created(
    api: FakeAPI, client: InferenceClient, kind: str
) -> None:
    path = f"/v1alpha1/inference/{kind}s"
    store = api.gateways if kind == "gateway" else api.deployments

    def lands_but_the_answer_is_lost(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        store[body["id"]] = {kind: {"spec": {"id": body["id"], "name": body["name"]}}}
        return httpx.Response(503, json=error_json("unavailable", "connection reset"))

    api.queue("POST", path, lands_but_the_answer_is_lost, ALREADY_EXISTS)
    created = getattr(client, f"create_{kind}")({"name": "run-1"})
    first, retry = posted(api, path)
    assert first["id"] == retry["id"] == created.id and created.name == "run-1"


@pytest.mark.parametrize("kind", ["gateway", "deployment"])
def test_a_create_whose_name_is_taken_says_so(
    api: FakeAPI, client: InferenceClient, kind: str
) -> None:
    path = f"/v1alpha1/inference/{kind}s"
    api.queue("POST", path, ALREADY_EXISTS)
    with pytest.raises(AlreadyExistsError, match=f"a {kind} named 'run-1' already exists"):
        getattr(client, f"create_{kind}")({"name": "run-1"})
    assert len(posted(api, path)) == 1


def test_a_create_under_an_id_another_deployment_has_says_so(
    api: FakeAPI, client: InferenceClient
) -> None:
    api.queue("POST", DEPLOYMENTS, ALREADY_EXISTS)
    with pytest.raises(AlreadyExistsError, match="already exists, named 'qwen3-8b-rl'"):
        client.create_deployment({"id": DEPLOYMENT_ID, "name": "another-name"})


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"CW_API_TOKEN": "docs-token"}, "docs-token"),
        ({"COREWEAVE_API_TOKEN": "own-token", "CW_API_TOKEN": "docs-token"}, "own-token"),
        ({"COREWEAVE_API_TOKEN": "  ", "CW_API_TOKEN": "docs-token"}, "docs-token"),
        ({"COREWEAVE_API_TOKEN": " padded\n"}, "padded"),
    ],
)
def test_the_token_comes_from_either_variable(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: str
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    with InferenceClient(transport=transport) as c:
        assert c._http.headers["Authorization"] == f"Bearer {expected}"


def test_non_json_success_body_is_retried(api: FakeAPI, client: InferenceClient) -> None:
    path = "/v1alpha1/inference/hotLoads/hl1"
    api.hot_loads["hl1"] = hot_load_json("hl1", "s", "HOT_LOAD_STATE_COMPLETED")
    api.queue("GET", path, httpx.Response(200, text="<html>captive portal</html>"))
    assert client.get_hot_load("hl1").completed
    assert sum(1 for r in api.requests if r.url.path == path) == 2


def test_pagination_stops_on_a_repeated_token(api: FakeAPI, client: InferenceClient) -> None:
    path = "/v1alpha1/inference/hotLoads"
    page = {
        "hotLoads": [hot_load_json("a", "s1", "HOT_LOAD_STATE_COMPLETED")],
        "nextPageToken": "t",
    }
    api.queue("GET", path, *[httpx.Response(200, json=page)] * 3)
    from cwtrain.errors import APIError

    with pytest.raises(APIError, match="twice"):
        list(client.list_hot_loads(DEPLOYMENT_ID))
