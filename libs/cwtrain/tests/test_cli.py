# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Offline checks for the CLI's output streams, validation and failure boundary."""

import json
import logging
import os
import subprocess
import sys
from unittest.mock import Mock

import httpx
import pytest
from boto3.exceptions import RetriesExceededError, S3UploadFailedError
from botocore.exceptions import (
    ClientError,
    CredentialRetrievalError,
    EndpointConnectionError,
    NoCredentialsError,
    ParamValidationError,
    PartialCredentialsError,
    ProfileNotFound,
    ReadTimeoutError,
)

from cwtrain import cli
from cwtrain.hotload.errors import SnapshotError
from cwtrain.infra.client import InferenceClient
from tests.conftest import (
    DEPLOYMENT_ID,
    FakeS3,
    hot_load_json,
    write_delta_checkpoint,
    write_full_checkpoint,
    write_single_file_checkpoint,
)


@pytest.mark.parametrize("token", [None, "", "  "])
def test_missing_token(monkeypatch, capsys, token):
    monkeypatch.delenv("COREWEAVE_API_TOKEN", raising=False)
    if token is not None:
        monkeypatch.setenv("COREWEAVE_API_TOKEN", token)
    assert cli.main(["hotload", "list"]) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert "COREWEAVE_API_TOKEN" in out.err and "--token before hotload" in out.err
    assert "Traceback" not in out.err


def test_module_exit_code():
    env = dict(os.environ)
    env.pop("COREWEAVE_API_TOKEN", None)
    result = subprocess.run(
        [sys.executable, "-m", "cwtrain.cli", "hotload", "list"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


COMMANDS = [
    (["publish", "--deployment", "d", "--checkpoint", "p", "--identity", "i"], "--timeout"),
    (["trigger", "--deployment", "d", "--identity", "i"], "--timeout"),
    (["wait", "h"], "--timeout"),
    (["publish", "--deployment", "d", "--checkpoint", "p", "--identity", "i"], "--workers"),
    (["list"], "--limit"),
]


@pytest.mark.parametrize("command,option", COMMANDS)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "bad", "1e999"])
def test_invalid_numbers(monkeypatch, capsys, command, option, value):
    run = Mock(side_effect=AssertionError("must not run"))
    monkeypatch.setattr(cli, "_run", run)
    with pytest.raises(SystemExit) as exc:
        cli.main(["hotload", *command, f"{option}={value}"])
    assert exc.value.code == 2
    out = capsys.readouterr()
    assert out.out == "" and option in out.err and "greater than zero" in out.err
    run.assert_not_called()


@pytest.mark.parametrize("command,option", COMMANDS)
def test_valid_numbers(command, option):
    value = "0.25" if option == "--timeout" else "1"
    args = cli.build_parser().parse_args(["hotload", *command, option, value])
    assert getattr(args, option[2:]) == float(value)


@pytest.mark.parametrize("option", ["--workers", "--limit"])
def test_fractional_counts(option):
    command = next(c for c, o in COMMANDS if o == option)
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["hotload", *command, option, "1.5"])
    assert exc.value.code == 2


@pytest.fixture
def offline_api(monkeypatch, api):
    monkeypatch.setenv("COREWEAVE_API_TOKEN", "env-token")
    monkeypatch.setattr(
        cli,
        "InferenceClient",
        lambda token, **kw: InferenceClient(token, transport=httpx.MockTransport(api.handle), **kw),
    )
    return api


@pytest.mark.parametrize("explicit", [False, True])
def test_list_json_limit_and_token(offline_api, capsys, explicit):
    for i in range(3):
        offline_api.hot_loads[str(i)] = hot_load_json(
            str(i), f"step_{i}", "HOT_LOAD_STATE_COMPLETED"
        )
    prefix = ["--token", "explicit-token"] if explicit else []
    assert cli.main([*prefix, "hotload", "list", "--limit", "1"]) == 0
    out = capsys.readouterr()
    items = json.loads(out.out)
    assert len(items) == 1 and items[0]["status"]["state"] == "COMPLETED"
    assert "raw" not in items[0] and out.err == ""
    assert offline_api.requests[-1].headers["Authorization"] == (
        "Bearer explicit-token" if explicit else "Bearer env-token"
    )


def test_inspect_without_credentials(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("COREWEAVE_API_TOKEN", raising=False)
    checkpoint = write_full_checkpoint(tmp_path / "checkpoint")
    assert cli.main(["hotload", "inspect", str(checkpoint)]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["path"] == str(checkpoint)
    assert out.err == ""


STORAGE_ERRORS = [
    (NoCredentialsError(), "AWS_ACCESS_KEY_ID"),
    (
        PartialCredentialsError(provider="env", cred_var="AWS_SECRET_ACCESS_KEY"),
        "AWS_SECRET_ACCESS_KEY",
    ),
    (CredentialRetrievalError(provider="test", error_msg="unavailable"), "AWS profile"),
    (ProfileNotFound(profile="missing"), "AWS profile"),
    (ClientError({"Error": {"Code": "AccessDenied"}}, "ListObjectsV2"), "AccessDenied"),
    (ClientError({"Error": {"Code": "NoSuchBucket"}}, "ListObjectsV2"), "NoSuchBucket"),
    (EndpointConnectionError(endpoint_url="https://storage.test"), "CW_S3_ENDPOINT"),
    (ReadTimeoutError(endpoint_url="https://storage.test"), "network"),
    (S3UploadFailedError("upload denied"), "bucket permissions"),
    (RetriesExceededError(Exception("retries exhausted")), "retry"),
]


@pytest.mark.parametrize("error,hint", STORAGE_ERRORS)
@pytest.mark.parametrize("stage", ["list_objects_v2", "upload_file"])
def test_publish_storage_failure(offline_api, monkeypatch, tmp_path, capsys, error, hint, stage):
    import cwtrain.hotload.runner as runner

    checkpoint = write_full_checkpoint(tmp_path / "checkpoint")
    s3 = FakeS3()
    monkeypatch.setattr(s3, stage, Mock(side_effect=error))
    monkeypatch.setattr(runner, "s3_client", lambda: s3)
    assert (
        cli.main(
            [
                "hotload",
                "publish",
                "--deployment",
                DEPLOYMENT_ID,
                "--checkpoint",
                str(checkpoint),
                "--identity",
                "step_1",
                "--no-wait",
            ]
        )
        == 1
    )
    out = capsys.readouterr()
    assert out.out == "" and hint in out.err and "Traceback" not in out.err
    assert not any(r.method == "POST" for r in offline_api.requests)


@pytest.mark.parametrize("verbose", [False, True])
def test_operational_debug_traceback(monkeypatch, capsys, verbose):
    monkeypatch.setattr(cli, "_run", Mock(side_effect=SnapshotError("broken checkpoint")))
    assert cli.main([*(["--verbose"] if verbose else []), "hotload", "list"]) == 1
    out = capsys.readouterr()
    assert out.out == "" and "error: broken checkpoint" in out.err
    assert ("Traceback" in out.err) == verbose


@pytest.mark.parametrize(
    "error", [ValueError("bug"), TypeError("bug"), ParamValidationError(report="bug")]
)
def test_programming_errors_propagate(monkeypatch, capsys, error):
    monkeypatch.setattr(cli, "_run", Mock(side_effect=error))
    with pytest.raises(type(error)):
        cli.main(["hotload", "list"])
    assert capsys.readouterr().out == ""


def test_publish_success_json(offline_api, monkeypatch, tmp_path, capsys):
    import cwtrain.hotload.runner as runner

    monkeypatch.setattr(runner, "s3_client", FakeS3)
    checkpoint = write_full_checkpoint(tmp_path / "checkpoint")
    assert (
        cli.main(
            [
                "hotload",
                "publish",
                "--deployment",
                DEPLOYMENT_ID,
                "--checkpoint",
                str(checkpoint),
                "--identity",
                "step_1",
                "--no-wait",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["spec"]["identity"] == "step_1"
    assert result["status"]["state"] == "PENDING"


def test_publish_single_file_checkpoint(offline_api, monkeypatch, tmp_path, capsys):
    import cwtrain.hotload.runner as runner

    s3 = FakeS3()
    monkeypatch.setattr(runner, "s3_client", lambda: s3)
    checkpoint = write_single_file_checkpoint(tmp_path / "checkpoint")
    assert cli.main(["hotload", "inspect", str(checkpoint)]) == 0
    assert json.loads(capsys.readouterr().out)["index_generated"] is True
    argv = ["--deployment", DEPLOYMENT_ID, "--checkpoint", str(checkpoint), "--identity", "sft-v1"]
    assert cli.main(["hotload", "publish", *argv, "--no-wait"]) == 0
    assert s3.uploads[-1] == ("<bytes>", "ckpts", "runs/exp1/sft-v1/model.safetensors.index.json")
    body = json.loads(next(r for r in offline_api.requests if r.method == "POST").content)
    assert body["type"] == "SNAPSHOT_TYPE_FULL"


@pytest.mark.parametrize("state,code", [("COMPLETED", 0), ("FAILED", 1)])
def test_wait_terminal_exit_code(offline_api, capsys, state, code):
    offline_api.hot_loads["h"] = hot_load_json("h", "step_1", f"HOT_LOAD_STATE_{state}")
    assert cli.main(["hotload", "wait", "h"]) == code
    out = capsys.readouterr()
    if code == 0:
        assert json.loads(out.out)["status"]["state"] == state
        assert out.err == ""
    else:
        assert out.out == "" and "FAILED" in out.err and "Traceback" not in out.err


@pytest.mark.parametrize(
    ("delta", "reset"), [(False, False), (False, True), (True, False), (True, True)]
)
def test_trigger_full_or_delta(offline_api, capsys, delta, reset):
    flags = ["--previous-identity", "step_1"] if delta else []
    flags += ["--reset-prompt-cache"] if reset else []
    argv = ["--deployment", DEPLOYMENT_ID, "--identity", "step_2", "--no-wait", *flags]
    assert cli.main(["hotload", "trigger", *argv]) == 0
    body = json.loads(next(r for r in offline_api.requests if r.method == "POST").content)
    if delta:
        assert body["type"] == "SNAPSHOT_TYPE_INCREMENTAL"
        assert body["incrementalSnapshotMetadata"]["previousSnapshotIdentity"] == "step_1"
    else:
        assert body["type"] == "SNAPSHOT_TYPE_FULL"
        assert "incrementalSnapshotMetadata" not in body
    if reset:  # independent of the delta form
        assert body["promptCachePolicy"] == "PROMPT_CACHE_POLICY_RESET_ALL"
    else:
        assert "promptCachePolicy" not in body
    assert json.loads(capsys.readouterr().out)["spec"]["identity"] == "step_2"


def test_publish_delta_with_previous_identity(offline_api, monkeypatch, tmp_path, capsys):
    import cwtrain.hotload.runner as runner

    monkeypatch.setattr(runner, "s3_client", FakeS3)
    delta = write_delta_checkpoint(tmp_path / "weight_v000002", version=2)
    argv = ["--deployment", DEPLOYMENT_ID, "--checkpoint", str(delta), "--identity", "run-v2"]
    assert (
        cli.main(["hotload", "publish", *argv, "--previous-identity", "run-v1", "--no-wait"]) == 0
    )
    body = json.loads(next(r for r in offline_api.requests if r.method == "POST").content)
    assert body["type"] == "SNAPSHOT_TYPE_INCREMENTAL"
    assert body["incrementalSnapshotMetadata"]["previousSnapshotIdentity"] == "run-v1"


def test_api_failure(offline_api, capsys):
    offline_api.queue(
        "GET",
        "/v1alpha1/inference/hotLoads",
        httpx.Response(401, json={"code": "unauthenticated", "message": "invalid token"}),
    )
    assert cli.main(["hotload", "list"]) == 1
    out = capsys.readouterr()
    assert out.out == "" and "unauthenticated: invalid token" in out.err
    assert "Traceback" not in out.err


def test_storage_debug_details(monkeypatch, capsys):
    error = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "provider detail"}}, "PutObject"
    )
    monkeypatch.setattr(cli, "_run", Mock(side_effect=error))
    assert cli.main(["--verbose", "hotload", "list"]) == 1
    out = capsys.readouterr()
    assert out.out == "" and "provider detail" in out.err and "Traceback" in out.err


CHATTY_LOGGERS = ("httpx", "httpcore", "botocore", "boto3", "s3transfer")


@pytest.fixture
def chatty_logger_levels():
    levels = {name: logging.getLogger(name).level for name in CHATTY_LOGGERS}
    yield
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


@pytest.mark.parametrize("verbose", [False, True])
def test_http_and_storage_library_logs_only_with_verbose(
    offline_api, capsys, caplog, chatty_logger_levels, verbose
):
    caplog.set_level(logging.INFO)
    offline_api.hot_loads["hl1"] = hot_load_json("hl1", "step_1", "HOT_LOAD_STATE_COMPLETED")
    assert cli.main([*(["--verbose"] if verbose else []), "hotload", "wait", "hl1"]) == 0
    requests = [r for r in caplog.records if r.name == "httpx" and r.levelno == logging.INFO]
    if verbose:
        assert any(r.getMessage().startswith("HTTP Request: GET") for r in requests)
    else:
        assert requests == []
        assert all(
            logging.getLogger(name).getEffectiveLevel() == logging.WARNING
            for name in CHATTY_LOGGERS
        )
    # cwtrain's own progress line is there either way
    assert any(r.name == "cwtrain.hotload.runner" for r in caplog.records)
    assert json.loads(capsys.readouterr().out)["status"]["state"] == "COMPLETED"


@pytest.mark.parametrize(
    "index,message",
    [
        ([], "index must be a JSON object"),
        ({"metadata": []}, "metadata must be a JSON object"),
        ({"weight_map": []}, "weight_map must be a JSON object"),
        ({"weight_map": {"layer.weight": "missing.safetensors"}}, "missing.safetensors"),
    ],
)
def test_inspect_reports_actionable_index_errors(tmp_path, capsys, index, message):
    checkpoint = write_full_checkpoint(tmp_path / "checkpoint")
    path = checkpoint / "model.safetensors.index.json"
    path.write_text(json.dumps(index))
    assert cli.main(["hotload", "inspect", str(checkpoint)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("error:")
    assert str(path) in output.err and message in output.err
    assert "Traceback" not in output.err


# ------------------------------------------------------------------------- deployment commands

GATEWAY = "22222222-2222-4222-8222-222222222222"
DEPLOYMENTS = "/v1alpha1/inference/deployments"


def deployment_request(tmp_path, **changes):
    body = {
        "name": "qwen3-4b-hotload",
        "gatewayIds": [GATEWAY],
        "model": {"name": "qwen3-4b-hotload", "bucket": "models", "path": "Qwen3-4B"},
        "hotLoad": {
            "bucket": "ckpts",
            "pathPrefix": "rl",
            "transitionMode": "TRANSITION_MODE_ASYNC",
        },
    }
    body.update(changes)
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(body))
    return path


def answer_as_created(api, status="STATUS_CREATING"):
    """POST /deployments answers with the request, as the API does, and keeps it for GETs."""

    def created(request):
        body = json.loads(request.content)
        deployment = {"spec": body, "status": {"status": status}}
        api.deployments[body["id"]] = {"deployment": deployment}
        return httpx.Response(200, json={"deployment": deployment})

    api.queue("POST", DEPLOYMENTS, created)


@pytest.fixture
def gateway_api(offline_api):
    offline_api.gateways[GATEWAY] = {
        "gateway": {
            "spec": {"id": GATEWAY, "name": "rl-gateway"},
            "status": {"status": "STATUS_READY", "endpoints": ["https://rl-gateway.example"]},
        }
    }
    return offline_api


def test_deployment_create_prints_what_a_run_needs(gateway_api, tmp_path, capsys, caplog):
    caplog.set_level(logging.INFO, logger="cwtrain")
    answer_as_created(gateway_api)
    assert cli.main(["deployment", "create", "--file", str(deployment_request(tmp_path))]) == 0
    out = capsys.readouterr()
    shown = json.loads(out.out)
    (sent,) = [json.loads(r.content) for r in gateway_api.requests if r.method == "POST"]
    assert shown["id"] == sent["id"] and f"as {sent['id']}" in caplog.text
    assert shown["status"] == "CREATING" and shown["model_name"] == "qwen3-4b-hotload"
    assert shown["hot_load"]["path_prefix"] == "rl"
    assert shown["endpoints"] == ["https://rl-gateway.example"]
    assert shown["rollout_base_url"] == "https://rl-gateway.example/v1"


def test_deployment_create_waits_for_ready(gateway_api, tmp_path, capsys, caplog):
    caplog.set_level(logging.INFO, logger="cwtrain")
    answer_as_created(gateway_api)
    gateway_api.queue(
        "GET",
        f"{DEPLOYMENTS}/{DEPLOYMENT_ID}",
        httpx.Response(
            200,
            json={
                "deployment": {
                    "spec": {"id": DEPLOYMENT_ID, "gatewayIds": [GATEWAY]},
                    "status": {"status": "STATUS_CREATING"},
                }
            },
        ),
        httpx.Response(
            200,
            json={
                "deployment": {
                    "spec": {"id": DEPLOYMENT_ID, "gatewayIds": [GATEWAY]},
                    "status": {"status": "STATUS_READY"},
                }
            },
        ),
    )
    request = deployment_request(tmp_path, id=DEPLOYMENT_ID)
    assert cli.main(["deployment", "create", "-f", str(request), "--wait"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "READY"
    assert "is CREATING; waiting for READY" in caplog.text


def test_deployment_create_prints_the_deployment_when_the_wait_fails(gateway_api, tmp_path, capsys):
    unmet = {"type": "Ready", "status": "STATUS_FALSE", "reason": "Failed", "message": "no weights"}
    spec = {"id": DEPLOYMENT_ID, "gatewayIds": [GATEWAY]}
    created = {"deployment": {"spec": spec, "status": {"status": "STATUS_CREATING"}}}
    failed = {
        "deployment": {"spec": spec, "status": {"status": "STATUS_FAILED", "conditions": [unmet]}}
    }
    gateway_api.queue("POST", DEPLOYMENTS, httpx.Response(200, json=created))
    gateway_api.deployments[DEPLOYMENT_ID] = failed  # what every later GET reads
    request = deployment_request(tmp_path, id=DEPLOYMENT_ID)
    assert cli.main(["deployment", "create", "-f", str(request), "--wait"]) == 1
    out = capsys.readouterr()
    shown = json.loads(out.out)
    assert shown["id"] == DEPLOYMENT_ID and shown["status"] == "FAILED"
    assert shown["rollout_base_url"] == "https://rl-gateway.example/v1"
    assert "is FAILED: Ready (Failed): no weights" in out.err


@pytest.mark.parametrize(
    "content,message",
    [
        (None, "cannot read"),
        ("{", "is not valid JSON"),
        ("[]", "must hold a JSON object"),
        (
            json.dumps({"name": "x", "gatewayIds": ["<gateway uuid in one zone>"]}),
            "placeholders: gatewayIds[0] = '<gateway uuid in one zone>'",
        ),
        (  # as in the NeMo Gym recipe's template: dots and slashes inside the brackets
            json.dumps({"model": {"path": "<path to Nemotron 3.5 Lightning BF16 checkpoint>"}}),
            "model.path = '<path to Nemotron 3.5 Lightning BF16 checkpoint>'",
        ),
        (
            json.dumps({"hotLoad": {"pathPrefix": "<bucket/prefix for snapshots>"}}),
            "hotLoad.pathPrefix = '<bucket/prefix for snapshots>'",
        ),
    ],
)
def test_deployment_create_refuses_a_request_it_cannot_send(
    offline_api, tmp_path, capsys, content, message
):
    path = tmp_path / "deployment.json"
    if content is not None:
        path.write_text(content)
    assert cli.main(["deployment", "create", "--file", str(path)]) == 1
    out = capsys.readouterr()
    assert out.out == "" and message in out.err and "Traceback" not in out.err
    assert offline_api.requests == []


def test_deployment_create_warns_without_hot_load(gateway_api, tmp_path, caplog):
    answer_as_created(gateway_api)
    request = deployment_request(tmp_path, hotLoad=None)
    assert cli.main(["deployment", "create", "--file", str(request)]) == 0
    assert "hot-loading cannot be enabled after creation" in caplog.text


def test_deployment_create_sends_tags_in_the_engine_options(gateway_api, tmp_path, capsys):
    runtime = {
        "engine": "dynamo-vllm",
        "version": "0.30.0",
        "engineConfig": {"reasoning-parser": "<think>"},
        "engineEnv": {"STOP_TOKEN": "<s>"},
    }
    answer_as_created(gateway_api)
    request = deployment_request(tmp_path, runtime=runtime)
    assert cli.main(["deployment", "create", "--file", str(request)]) == 0
    (sent,) = [json.loads(r.content) for r in gateway_api.requests if r.method == "POST"]
    assert sent["runtime"] == runtime

    # a placeholder anywhere else is still refused
    gateway_api.requests.clear()
    request = deployment_request(tmp_path, runtime={**runtime, "version": "<runtime version>"})
    assert cli.main(["deployment", "create", "--file", str(request)]) == 1
    err = capsys.readouterr().err
    assert "runtime.version = '<runtime version>'" in err and "engineConfig" not in err
    assert gateway_api.requests == []


@pytest.mark.parametrize("status,code", [(403, "permission_denied"), (404, "not_found")])
def test_deployment_create_prints_the_deployment_when_its_gateway_cannot_be_read(
    offline_api, tmp_path, capsys, caplog, status, code
):
    answer_as_created(offline_api)
    offline_api.queue(
        "GET",
        f"/v1alpha1/inference/gateways/{GATEWAY}",
        httpx.Response(status, json={"code": code, "message": "cannot read the gateway"}),
    )
    assert cli.main(["deployment", "create", "--file", str(deployment_request(tmp_path))]) == 0
    shown = json.loads(capsys.readouterr().out)
    (sent,) = [json.loads(r.content) for r in offline_api.requests if r.method == "POST"]
    assert shown["id"] == sent["id"]
    assert shown["endpoints"] == [] and shown["rollout_base_url"] is None
    assert f"cannot read the gateways of deployment {sent['id']}" in caplog.text


def test_deployment_create_names_the_deployment_that_has_the_name(offline_api, tmp_path, capsys):
    taken = {"code": "already_exists", "message": "resource already exists"}
    offline_api.queue("POST", DEPLOYMENTS, httpx.Response(409, json=taken))
    existing = {"spec": {"id": "33333333-3333-4333-8333-333333333333", "name": "qwen3-4b-hotload"}}
    offline_api.queue("GET", DEPLOYMENTS, httpx.Response(200, json={"items": [existing]}))
    assert cli.main(["deployment", "create", "--file", str(deployment_request(tmp_path))]) == 1
    err = capsys.readouterr().err
    assert "'qwen3-4b-hotload' already exists as 33333333-3333-4333-8333-333333333333" in err


def test_deployment_get_with_the_token_from_the_docs(gateway_api, monkeypatch, capsys):
    monkeypatch.delenv("COREWEAVE_API_TOKEN")
    monkeypatch.setenv("CW_API_TOKEN", "docs-token")
    gateway_api.deployments[DEPLOYMENT_ID]["deployment"]["spec"]["gatewayIds"] = [GATEWAY]
    assert cli.main(["deployment", "get", DEPLOYMENT_ID]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["id"] == DEPLOYMENT_ID and shown["status"] == "READY"
    assert shown["rollout_base_url"] == "https://rl-gateway.example/v1"
    assert gateway_api.requests[0].headers["Authorization"] == "Bearer docs-token"


@pytest.mark.parametrize(
    "args,result",
    [([], "deleting"), (["--wait"], "deleted")],
)
def test_deployment_delete(offline_api, capsys, args, result):
    def delete(_request):
        del offline_api.deployments[DEPLOYMENT_ID]
        return httpx.Response(200, json={})

    offline_api.queue("DELETE", f"{DEPLOYMENTS}/{DEPLOYMENT_ID}", delete)
    assert cli.main(["deployment", "delete", DEPLOYMENT_ID, *args]) == 0
    assert json.loads(capsys.readouterr().out) == {"id": DEPLOYMENT_ID, "result": result}


@pytest.mark.parametrize("wait", [[], ["--wait"]])
def test_deployment_delete_of_a_missing_deployment(offline_api, capsys, wait):
    missing = "44444444-4444-4444-8444-444444444444"
    gone = httpx.Response(404, json={"code": "not_found", "message": "deployment not found"})
    offline_api.queue("DELETE", f"{DEPLOYMENTS}/{missing}", gone, gone)
    assert cli.main(["deployment", "delete", missing, *wait]) == 1
    assert f"deployment {missing} not found" in capsys.readouterr().err
    assert cli.main(["deployment", "delete", missing, *wait, "--ignore-not-found"]) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "not_found"


def test_deployment_parameters_for_gateways(offline_api, capsys):
    offline_api.queue(
        "GET", f"{DEPLOYMENTS}/parameters", httpx.Response(200, json={"runtimeParameters": {}})
    )
    assert cli.main(["deployment", "parameters", "--gateway", "a", "--gateway", "b"]) == 0
    assert json.loads(capsys.readouterr().out) == {"runtimeParameters": {}}
    assert offline_api.requests[-1].url.params.get_list("gatewayIds") == ["a", "b"]


def test_deployment_commands_need_a_token(monkeypatch, capsys):
    assert cli.main(["deployment", "get", DEPLOYMENT_ID]) == 1
    err = capsys.readouterr().err
    assert "COREWEAVE_API_TOKEN (or CW_API_TOKEN)" in err and "--token before deployment" in err


@pytest.mark.parametrize("command", [["create", "-f", "x.json"], ["delete", "d"]])
def test_deployment_timeouts_must_be_positive(capsys, command):
    with pytest.raises(SystemExit) as exc:
        cli.main(["deployment", *command, "--timeout", "0"])
    assert exc.value.code == 2 and "greater than zero" in capsys.readouterr().err
