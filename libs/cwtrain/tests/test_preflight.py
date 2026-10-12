# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""The snapshot access check: the trainer's permissions, nothing left behind."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import httpx
import pytest
from botocore.exceptions import ClientError, NoCredentialsError

import cwtrain.hotload.runner as runner
from cwtrain import cli
from cwtrain.errors import CWTrainError
from cwtrain.hotload import HotLoader, SnapshotAccessError, check_snapshot_access
from cwtrain.infra.client import InferenceClient
from cwtrain.integrations import preflight
from tests.conftest import DEPLOYMENT_ID, deployment_json


class ProbeS3:
    """Records the check's calls. ``fail`` maps an operation to the error code it returns.

    Any other S3 call, such as a put or a delete, fails the test.
    """

    def __init__(self, fail: dict[str, str] | None = None, head_code: str = "404") -> None:
        self.fail = fail or {}
        self.head_code = head_code
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _call(self, op: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((op, kwargs))
        if op in self.fail:
            raise ClientError({"Error": {"Code": self.fail[op], "Message": "no"}}, op)

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self._call("list_objects_v2", kwargs)
        return {"KeyCount": 0, "IsTruncated": False}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self._call("head_object", kwargs)
        raise ClientError({"Error": {"Code": self.head_code, "Message": "Not Found"}}, "HeadObject")

    def create_multipart_upload(self, **kwargs: Any) -> dict[str, Any]:
        self._call("create_multipart_upload", kwargs)
        return {"UploadId": "upload-1"}

    def abort_multipart_upload(self, **kwargs: Any) -> dict[str, Any]:
        self._call("abort_multipart_upload", kwargs)
        return {}

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the check must not call {name}")

    def ops(self) -> list[str]:
        return [op for op, _ in self.calls]


def test_the_check_lists_reads_and_aborts_its_own_upload() -> None:
    s3 = ProbeS3()
    assert check_snapshot_access(s3, "ckpts", "hotload") == "s3://ckpts/hotload/"
    assert s3.ops() == [
        "list_objects_v2",
        "head_object",
        "create_multipart_upload",
        "abort_multipart_upload",
    ]
    listed, head, create, abort = (kwargs for _, kwargs in s3.calls)
    assert listed == {"Bucket": "ckpts", "Prefix": "hotload/", "MaxKeys": 1}
    assert head["Key"].startswith("hotload/_preflight/") and head["Bucket"] == "ckpts"
    assert create == head  # the upload goes to the key that does not exist
    assert abort == {**head, "UploadId": "upload-1"}


def test_each_check_uses_a_new_key() -> None:
    first, second = ProbeS3(), ProbeS3()
    check_snapshot_access(first, "ckpts", "hotload")
    check_snapshot_access(second, "ckpts", "hotload")
    assert first.calls[1][1]["Key"] != second.calls[1][1]["Key"]


@pytest.mark.parametrize("prefix", ["/hotload/", "hotload/", "/hotload"])
def test_the_prefix_ignores_outer_slashes(prefix: str) -> None:
    s3 = ProbeS3()
    assert check_snapshot_access(s3, "ckpts", prefix) == "s3://ckpts/hotload/"
    assert s3.calls[0][1]["Prefix"] == "hotload/"
    assert s3.calls[1][1]["Key"].startswith("hotload/_preflight/")


@pytest.mark.parametrize("prefix", ["", "/"])
def test_an_empty_prefix_checks_the_top_of_the_bucket(prefix: str) -> None:
    s3 = ProbeS3()
    assert check_snapshot_access(s3, "ckpts", prefix) == "s3://ckpts/"
    assert s3.calls[0][1]["Prefix"] == ""
    assert s3.calls[1][1]["Key"].startswith("_preflight/")


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_a_missing_key_counts_as_readable(code: str) -> None:
    s3 = ProbeS3(head_code=code)
    check_snapshot_access(s3, "ckpts", "hotload")
    assert s3.ops()[-1] == "abort_multipart_upload"


@pytest.mark.parametrize(
    ("op", "action", "calls"),
    [
        ("list_objects_v2", "s3:ListBucket", []),
        ("create_multipart_upload", "s3:PutObject", ["list_objects_v2", "head_object"]),
        (
            "abort_multipart_upload",
            "s3:AbortMultipartUpload",
            ["list_objects_v2", "head_object", "create_multipart_upload"],
        ),
    ],
)
def test_a_refused_call_names_its_permission(op: str, action: str, calls: list[str]) -> None:
    s3 = ProbeS3(fail={op: "AccessDenied"})
    with pytest.raises(SnapshotAccessError) as exc:
        check_snapshot_access(s3, "ckpts", "hotload")
    assert (exc.value.action, exc.value.code, exc.value.uri) == (
        action,
        "AccessDenied",
        "s3://ckpts/hotload/",
    )
    message = str(exc.value)
    assert action in message and "AccessDenied" in message and "s3://ckpts/hotload/" in message
    assert "AllowTrainer" in message and "s3:AbortMultipartUpload" in message
    assert s3.ops() == [*calls, op]  # nothing runs after the refused call


@pytest.mark.parametrize("code", ["403", "AccessDenied"])
def test_a_forbidden_read_names_get_object(code: str) -> None:
    s3 = ProbeS3(head_code=code)
    with pytest.raises(SnapshotAccessError, match="s3:GetObject"):
        check_snapshot_access(s3, "ckpts", "hotload")
    assert s3.ops() == ["list_objects_v2", "head_object"]


def test_missing_credentials_are_not_reported_as_a_denial() -> None:
    s3 = ProbeS3()
    s3.list_objects_v2 = lambda **_kw: (_ for _ in ()).throw(NoCredentialsError())
    with pytest.raises(NoCredentialsError):
        check_snapshot_access(s3, "ckpts", "hotload")


def test_the_loader_checks_the_deployments_bucket_and_prefix(client: InferenceClient) -> None:
    s3 = ProbeS3()
    assert HotLoader(client, DEPLOYMENT_ID, s3=s3).check_snapshot_access() == (
        "s3://ckpts/runs/exp1/"
    )
    assert s3.calls[0][1] == {"Bucket": "ckpts", "Prefix": "runs/exp1/", "MaxKeys": 1}


# ------------------------------------------------------------------ the trainer Job's module


def env(**extra: str) -> dict[str, str]:
    return {"CWTRAIN_HOTLOAD_DEPLOYMENT_ID": DEPLOYMENT_ID, "COREWEAVE_API_TOKEN": "t", **extra}


def test_the_module_checks_the_deployment_the_environment_names(api) -> None:
    s3 = ProbeS3()
    uri = preflight.check(
        env(COREWEAVE_API_URL="https://api.test", CWTRAIN_API_HEADERS='{"X-Org": "o"}'),
        s3=s3,
        transport=httpx.MockTransport(api.handle),
    )
    assert uri == "s3://ckpts/runs/exp1/" and s3.ops()[-1] == "abort_multipart_upload"
    request = api.requests[-1]
    assert request.url.host == "api.test" and request.url.path.endswith(DEPLOYMENT_ID)
    assert request.headers["Authorization"] == "Bearer t" and request.headers["X-Org"] == "o"


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"COREWEAVE_API_TOKEN": "t"}, "CWTRAIN_HOTLOAD_DEPLOYMENT_ID is required"),
        ({"CWTRAIN_HOTLOAD_DEPLOYMENT_ID": DEPLOYMENT_ID}, "API token is required"),
        (env(CWTRAIN_API_HEADERS="not json"), "CWTRAIN_API_HEADERS must be a JSON object"),
        (env(CWTRAIN_API_INSECURE="maybe"), "CWTRAIN_API_INSECURE must be a boolean"),
    ],
)
def test_the_module_names_a_missing_or_bad_setting(environ, message: str) -> None:
    with pytest.raises(CWTrainError, match=message):
        preflight.check(environ, s3=ProbeS3())


def test_the_module_refuses_a_deployment_without_hot_load(api) -> None:
    api.deployments[DEPLOYMENT_ID] = deployment_json(hot_load=False)
    s3 = ProbeS3()
    with pytest.raises(CWTrainError, match="without a hot_load section"):
        preflight.check(env(), s3=s3, transport=httpx.MockTransport(api.handle))
    assert s3.calls == []


@pytest.fixture
def offline(monkeypatch, api):
    """Point the module and the CLI at the fake API, and the loader at a ProbeS3."""
    s3 = ProbeS3()
    for module in (preflight, cli):
        real = module.InferenceClient
        monkeypatch.setattr(
            module,
            "InferenceClient",
            lambda *a, real=real, **kw: real(*a, transport=httpx.MockTransport(api.handle), **kw),
        )
    monkeypatch.setattr(runner, "s3_client", lambda: s3)
    return s3


def test_the_module_prints_one_line_on_success(offline, capsys) -> None:
    assert preflight.main(env()) == 0
    out = capsys.readouterr()
    assert out.out == (
        "snapshot access OK: s3://ckpts/runs/exp1/ (list, read, multipart write and abort)\n"
    )
    assert out.err == ""


def test_the_module_explains_a_refusal_without_a_traceback(offline, capsys) -> None:
    offline.fail["create_multipart_upload"] = "AccessDenied"
    assert preflight.main(env()) == 1
    out = capsys.readouterr()
    assert out.out == "" and "Traceback" not in out.err
    assert out.err.startswith("snapshot access check failed: ") and "s3:PutObject" in out.err


def test_the_module_runs_with_python_dash_m() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cwtrain.integrations.preflight"],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and result.stdout == ""
    assert "CWTRAIN_HOTLOAD_DEPLOYMENT_ID is required" in result.stderr
    assert "Traceback" not in result.stderr


# ------------------------------------------------------------------ cwtrain hotload check


def test_the_cli_check_prints_the_prefix(offline, monkeypatch, capsys) -> None:
    monkeypatch.setenv("COREWEAVE_API_TOKEN", "t")
    assert cli.main(["hotload", "check", "--deployment", DEPLOYMENT_ID]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "deployment": DEPLOYMENT_ID,
        "bucket": "ckpts",
        "path_prefix": "runs/exp1",
        "uri": "s3://ckpts/runs/exp1/",
        "result": "ok",
    }


def test_the_cli_check_fails_on_a_refusal(offline, monkeypatch, capsys) -> None:
    monkeypatch.setenv("COREWEAVE_API_TOKEN", "t")
    offline.head_code = "AccessDenied"
    assert cli.main(["hotload", "check", "--deployment", DEPLOYMENT_ID]) == 1
    out = capsys.readouterr()
    assert out.out == "" and "s3:GetObject" in out.err and "Traceback" not in out.err


def test_the_cli_check_needs_a_deployment() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["hotload", "check"])
    assert exc.value.code == 2
