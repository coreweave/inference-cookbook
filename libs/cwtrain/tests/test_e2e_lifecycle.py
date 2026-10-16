# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from unittest.mock import Mock

import httpx
import pytest

from cwtrain.errors import NotFoundError
from cwtrain.infra.client import InferenceClient
from e2e import hotload, storage

STATE = {
    "deployment_id": "dep",
    "gateway_id": "gw",
    "name": "test",
    "bucket": "test-e2e-bucket",
    "prefix": "cookbook/cwtrain-e2e/20261009-1912-" + "a" * 8 + "/",
}


@pytest.fixture(autouse=True)
def configure_bucket(monkeypatch):
    monkeypatch.setenv("CWTRAIN_E2E_BUCKET", STATE["bucket"])


@pytest.mark.parametrize(
    "failure", [None, "bucket", "stage", "gateway", "deployment", "publish", "verify"]
)
def test_cleanup_on_success_and_each_failure(monkeypatch, failure):
    head_bucket = Mock()
    stage_model = Mock(return_value="/checkpoint")
    monkeypatch.setattr(hotload, "bucket_client", Mock(return_value=Mock(head_bucket=head_bucket)))
    monkeypatch.setattr(hotload, "stage_model", stage_model)
    client = Mock()
    loader = Mock()
    loader.publish.return_value.id = "hl"
    loader.publish.return_value.identity = "test-checkpoint"
    loader.publish.return_value.deployment_id = "dep"
    loader.current_identity.return_value = "test-checkpoint"
    monkeypatch.setattr(hotload, "HotLoader", Mock(return_value=loader))
    monkeypatch.setattr(hotload.lifecycle, "wait_for_gateway", Mock())
    wait_for_deployment = Mock()
    monkeypatch.setattr(hotload.lifecycle, "wait_for_deployment", wait_for_deployment)
    cleanup = Mock()
    monkeypatch.setattr(hotload, "cleanup", cleanup)
    targets = {
        "bucket": head_bucket,
        "stage": stage_model,
        "gateway": client.create_gateway,
        "deployment": client.create_deployment,
        "publish": loader.publish,
        "verify": loader.verify_served,
    }
    if failure:
        targets[failure].side_effect = RuntimeError("injected")
        with pytest.raises(RuntimeError, match="injected"):
            hotload.run_resources(client, STATE, {}, "zone")
    else:
        hotload.run_resources(client, STATE, {}, "zone")
        # an identity of this run's own, never one an earlier run used
        loader.publish.assert_called_once_with("/checkpoint", "test-checkpoint")
        loader.verify_served.assert_called_once_with("hl", "test-checkpoint")
        wait_for_deployment.assert_called_once_with(client, "dep", timeout=1200)
    cleanup.assert_called_once_with(client, STATE)


def test_cleanup_attempts_gateway_even_when_deployment_delete_fails(monkeypatch):
    delete_run_objects = Mock()
    monkeypatch.setattr(hotload, "delete_run_objects", delete_run_objects)
    client = Mock()
    client.delete_deployment.side_effect = RuntimeError("unavailable")
    client.delete_gateway.side_effect = NotFoundError("not_found", "gone")
    with pytest.raises(RuntimeError, match="deployment dep"):
        hotload.cleanup(client, STATE, timeout=0)
    client.delete_gateway.assert_called_once_with("gw")
    delete_run_objects.assert_called_once_with(STATE["bucket"], STATE["prefix"], timeout=0)


def test_cleanup_waits_longer_for_the_deployment(monkeypatch):
    waits = {}
    for kind in ("deployment", "gateway"):
        monkeypatch.setattr(
            hotload.lifecycle,
            f"delete_{kind}",
            lambda client, resource_id, timeout, kind=kind: waits.setdefault(kind, timeout),
        )
    monkeypatch.setattr(hotload, "delete_run_objects", Mock())
    hotload.cleanup(Mock(), STATE, timeout=300)
    assert waits == {"deployment": 600, "gateway": 300}


def test_cleanup_waits_for_deletion_and_is_repeatable(monkeypatch):
    client = Mock()
    monkeypatch.setattr(hotload, "delete_run_objects", Mock())
    missing = NotFoundError("not_found", "gone")
    client.get_deployment.side_effect = [Mock(), missing]
    client.get_gateway.side_effect = missing
    hotload.cleanup(client, STATE)
    assert client.delete_deployment.call_count == 1 and client.get_deployment.call_count == 2
    client.delete_deployment.side_effect = missing
    client.delete_gateway.side_effect = missing
    hotload.cleanup(client, STATE)


def test_gateway_wire_contract():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"gateway": {"spec": {"id": "gw"}}})

    with InferenceClient(token="test", transport=httpx.MockTransport(respond)) as client:
        assert client.create_gateway({"id": "gw"}).id == "gw"
        assert client.get_gateway("gw").id == "gw"
        client.delete_gateway("gw")
    assert [(r.method, r.url.path) for r in requests] == [
        ("POST", "/v1alpha1/inference/gateways"),
        ("GET", "/v1alpha1/inference/gateways/gw"),
        ("DELETE", "/v1alpha1/inference/gateways/gw"),
    ]


@pytest.mark.parametrize("zone_input", [None, "", "  ", "us-west-04a", " US-WEST-04A "])
def test_state_saved_before_provisioning(monkeypatch, tmp_path, zone_input):
    import json

    state_path = tmp_path / "state.json"
    monkeypatch.setattr("sys.argv", ["hotload", "--state", str(state_path)])
    monkeypatch.setenv("COREWEAVE_API_TOKEN", "test")
    monkeypatch.setenv(
        "CWTRAIN_E2E_DEPLOYMENT_JSON",
        json.dumps(
            {
                "runtime": {"engine": "dynamo-vllm"},
                "resources": {"gpuCount": 1},
                "autoscaling": {"min": 1},
                "traffic": {"weight": 100},
            }
        ),
    )

    def provision(client, state, spec, zone):
        assert json.loads(state_path.read_text()) == state
        assert state["bucket"] == "test-e2e-bucket"
        assert storage.RUN_PREFIX.fullmatch(state["prefix"])
        assert spec["model"]["path"] == f"{state['prefix']}models/base"
        assert spec["hotLoad"]["pathPrefix"] == f"{state['prefix']}hotload"
        assert spec["model"]["bucket"] == state["bucket"]
        assert spec["hotLoad"]["bucket"] == state["bucket"]
        assert zone == "US-WEST-04A"
        assert spec["runtime"]["version"] == hotload.DEFAULT_RUNTIME_VERSION
        raise RuntimeError("create response lost")

    monkeypatch.delenv("CWTRAIN_E2E_RUNTIME_VERSION", raising=False)
    if zone_input is None:
        monkeypatch.delenv("CWTRAIN_E2E_ZONE", raising=False)
    else:
        monkeypatch.setenv("CWTRAIN_E2E_ZONE", zone_input)
    monkeypatch.setattr(hotload, "run_resources", provision)
    with pytest.raises(RuntimeError, match="create response lost"):
        hotload.main()
    with pytest.raises(FileExistsError):
        hotload.main()


def test_cleanup_only_without_state_needs_no_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv", ["hotload", "--state", str(tmp_path / "missing"), "--cleanup-only"]
    )
    monkeypatch.setattr(hotload, "InferenceClient", Mock(side_effect=AssertionError("not needed")))
    hotload.main()


def test_run_cleanup_clears_only_its_objects_and_uploads(monkeypatch):
    from e2e import storage

    prefix = STATE["prefix"]
    own_key = prefix + "models/base/weights"
    other_key = "cookbook/cwtrain-e2e/20261009-1912-" + "b" * 8 + "/models/base/weights"
    objects = {own_key, other_key, "existing/model.safetensors"}
    uploads = {prefix + "hotload/checkpoint/weights": "own", other_key: "other"}
    s3 = Mock()

    def paginator(operation):
        def pages(**kwargs):
            assert kwargs == {"Bucket": STATE["bucket"], "Prefix": prefix}
            if operation == "list_objects_v2":
                return [{"Contents": [{"Key": key} for key in objects if key.startswith(prefix)]}]
            assert operation == "list_multipart_uploads"
            return [
                {
                    "Uploads": [
                        {"Key": key, "UploadId": value}
                        for key, value in uploads.items()
                        if key.startswith(prefix)
                    ]
                }
            ]

        return Mock(paginate=pages)

    def delete(**kwargs):
        objects.difference_update(item["Key"] for item in kwargs["Delete"]["Objects"])
        return {}

    def abort(**kwargs):
        del uploads[kwargs["Key"]]

    s3.get_paginator.side_effect = paginator
    s3.delete_objects.side_effect = delete
    s3.abort_multipart_upload.side_effect = abort
    monkeypatch.setattr(storage, "bucket_client", Mock(return_value=s3))
    monkeypatch.setattr(storage.time, "sleep", Mock())
    storage.delete_run_objects(STATE["bucket"], prefix, timeout=0)
    storage.time.sleep.assert_not_called()
    storage.delete_run_objects(STATE["bucket"], prefix, timeout=0)
    assert objects == {other_key, "existing/model.safetensors"}
    assert uploads == {other_key: "other"}
    s3.delete_bucket.assert_not_called()
    s3.put_bucket_policy.assert_not_called()


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "cookbook/",
        "cwtrain-e2e/20261009-1912-" + "a" * 8 + "/",
        "cookbook/cwtrain-e2e/",
        "models/",
        "cookbook/cwtrain-e2e/20261009-1912-" + "a" * 8,
        "cookbook/cwtrain-e2e/20261009-1912/",
        "cookbook/cwtrain-e2e/20261009-1912-" + "a" * 8 + "/models/",
        "cookbook/cwtrain-e2e/" + "a" * 32 + "/",
    ],
)
def test_cleanup_rejects_unsafe_prefix_before_storage_access(monkeypatch, prefix):
    from e2e import storage

    client = Mock()
    monkeypatch.setattr(storage, "bucket_client", client)
    with pytest.raises(ValueError, match="Refusing"):
        storage.delete_run_objects(STATE["bucket"], prefix, timeout=0)
    client.assert_not_called()


def test_run_prefix_starts_with_the_utc_minute():
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 10, 9, 12, 7, tzinfo=timezone(timedelta(hours=-7)))
    prefixes = {storage.new_run_prefix(start) for _ in range(20)}
    assert all(p.startswith("cookbook/cwtrain-e2e/20261009-1907-") for p in prefixes)
    assert all(storage.RUN_PREFIX.fullmatch(p) for p in prefixes)
    assert len(prefixes) == 20  # runs that start in the same minute still get their own prefix


def test_cleanup_rejects_wrong_bucket(monkeypatch):
    from e2e import storage

    client = Mock()
    monkeypatch.setattr(storage, "bucket_client", client)
    with pytest.raises(ValueError, match="Refusing"):
        storage.delete_run_objects("another-bucket", STATE["prefix"])
    client.assert_not_called()


@pytest.mark.parametrize("operation", ["objects", "uploads"])
def test_cleanup_rejects_out_of_prefix_listing(monkeypatch, operation):
    from e2e import storage

    s3 = Mock()
    pages = (
        [[{"Contents": [{"Key": "unrelated/weights"}]}]]
        if operation == "objects"
        else [[{}], [{"Uploads": [{"Key": "unrelated/weights", "UploadId": "upload"}]}]]
    )
    s3.get_paginator.return_value.paginate.side_effect = pages
    monkeypatch.setattr(storage, "bucket_client", Mock(return_value=s3))
    with pytest.raises(ValueError, match="outside the run prefix"):
        storage.delete_run_objects(STATE["bucket"], STATE["prefix"], timeout=0)
    s3.delete_objects.assert_not_called()
    s3.abort_multipart_upload.assert_not_called()
    s3.delete_bucket.assert_not_called()


def test_run_cleanup_propagates_partial_delete_errors(monkeypatch):
    from e2e import storage

    s3 = Mock()
    s3.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": STATE["prefix"] + "weights"}]}
    ]
    s3.delete_objects.return_value = {"Errors": [{"Code": "AccessDenied"}]}
    monkeypatch.setattr(storage, "bucket_client", Mock(return_value=s3))
    with pytest.raises(RuntimeError, match="object deletion failures"):
        storage.delete_run_objects(STATE["bucket"], STATE["prefix"], timeout=0)
    s3.delete_bucket.assert_not_called()


def test_stage_model_uploads_inside_run_prefix(monkeypatch, tmp_path):
    import sys

    from e2e import storage

    download = Mock(return_value=str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", Mock(snapshot_download=download))
    snapshot = Mock()
    s3 = Mock()
    upload = Mock()
    monkeypatch.setattr(storage, "inspect_snapshot", Mock(return_value=snapshot))
    monkeypatch.setattr(storage, "bucket_client", Mock(return_value=s3))
    monkeypatch.setattr(storage, "upload_snapshot", upload)
    assert storage.stage_model(STATE["bucket"], STATE["prefix"], str(tmp_path)) == str(tmp_path)
    upload.assert_called_once_with(
        snapshot,
        bucket=STATE["bucket"],
        path_prefix=STATE["prefix"] + "models",
        identity="base",
        s3=s3,
    )


@pytest.mark.parametrize("engine", ["vllm", "sglang", None])
def test_reject_unsupported_engine_before_provisioning(monkeypatch, tmp_path, engine):
    import json

    state = tmp_path / "state.json"
    monkeypatch.setattr("sys.argv", ["hotload", "--state", str(state)])
    monkeypatch.setenv(
        "CWTRAIN_E2E_DEPLOYMENT_JSON",
        json.dumps(
            {
                "runtime": {"engine": engine},
                "resources": {"gpuCount": 1},
                "autoscaling": {"min": 1},
                "traffic": {"weight": 100},
            }
        ),
    )
    monkeypatch.setattr(hotload, "InferenceClient", Mock(side_effect=AssertionError("not needed")))
    with pytest.raises(ValueError, match="runtime.engine=dynamo-vllm"):
        hotload.main()
    assert not state.exists()


@pytest.mark.parametrize("code", ["NoSuchBucket", "404"])
def test_missing_shared_bucket_is_cleanup_failure(monkeypatch, code):
    from botocore.exceptions import ClientError

    from e2e import storage

    s3 = Mock()
    s3.get_paginator.return_value.paginate.side_effect = ClientError(
        {"Error": {"Code": code}}, "ListObjectsV2"
    )
    monkeypatch.setattr(storage, "bucket_client", Mock(return_value=s3))
    with pytest.raises(ClientError):
        storage.delete_run_objects(STATE["bucket"], STATE["prefix"], timeout=0)


def test_bucket_configuration_required(monkeypatch):
    from e2e import storage

    monkeypatch.delenv("CWTRAIN_E2E_BUCKET")
    with pytest.raises(ValueError, match="CWTRAIN_E2E_BUCKET"):
        storage.validate_location(STATE["bucket"], STATE["prefix"])


def test_bucket_configuration_can_change(monkeypatch):
    from e2e import storage

    monkeypatch.setenv("CWTRAIN_E2E_BUCKET", "another-org-bucket")
    storage.validate_location("another-org-bucket", STATE["prefix"])
    with pytest.raises(ValueError, match="Refusing"):
        storage.validate_location(STATE["bucket"], STATE["prefix"])
