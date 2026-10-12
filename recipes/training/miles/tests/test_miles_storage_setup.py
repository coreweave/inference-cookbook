# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Object storage setup: make-storage-secret.sh and make-bucket-job.yaml need the bucket's zone, and
the Job attaches the example bucket policy.

Every training recipe carries identical copies of these files (CI checks it), so this tests the
miles ones.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
K8S = RECIPE / "k8s"
PLATFORM_ARN = "arn:aws:iam::cw4637:coreweave/uvAGGQSxxXeeQBJzcGsD9"
# the trainer runs as a Kubernetes service account (Workload Identity Federation)
TRAINER_ARN = "arn:aws:iam::org:role/https://oidc.example/id/c1:system:serviceaccount:ns:trainer-sa"

FAKE_KUBECTL = """#!/bin/sh
# 'create ... --dry-run' prints the config file it was given; 'apply -f -' saves what it reads
for arg in "$@"; do
  case "$arg" in
    --from-file=config=*) cat "${arg#--from-file=config=}" ;;
    --from-file=credentials=*) cat "${arg#--from-file=credentials=}" ;;
  esac
done
case " $* " in *" apply "*) cat > "$CAPTURE" ;; esac
"""

FAKE_BOTO3 = """import json, os, types


class _Exceptions:
    class BucketAlreadyOwnedByYou(Exception):
        pass

    class BucketAlreadyExists(Exception):
        pass


class _Client:
    exceptions = _Exceptions

    def __init__(self, endpoint):
        self.meta = types.SimpleNamespace(endpoint_url=endpoint)

    def __getattr__(self, name):
        def call(**kwargs):
            _record(name, {"endpoint": self.meta.endpoint_url, **kwargs})
            if name == "get_bucket_policy":
                code = os.environ.get("FAKE_POLICY_ERROR", "NoSuchBucketPolicy")
                if os.environ.get("FAKE_POLICY"):
                    return {"Policy": os.environ["FAKE_POLICY"]}
                raise _ClientError(code)
            return {"UploadId": "u"}

        return call


class _ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def _record(name, kwargs):
    with open(os.environ["CALLS"], "a") as f:
        f.write(json.dumps([name, kwargs], default=repr) + "\\n")


def client(service, endpoint_url=None, config=None, region_name=None, **kwargs):
    # like botocore: an empty endpoint is invalid, a configured one (FAKE_CONFIGURED_ENDPOINT, the
    # storage Secret's endpoint_url) applies unless ignored, and AWS's own, which names the
    # region, is the last resort
    if endpoint_url == "":
        raise ValueError("Invalid endpoint: ")
    ignoring = getattr(config, "kwargs", {}).get("ignore_configured_endpoint_urls", False)
    configured = None if ignoring else os.environ.get("FAKE_CONFIGURED_ENDPOINT")
    endpoint = endpoint_url or configured or f"https://s3.{region_name}.amazonaws.com"
    _record("client", {"endpoint": endpoint, "region_name": region_name, **kwargs})
    return _Client(endpoint)
"""


def make_secret(
    tmp_path: Path, *args: str, keys: bool = True, **env: str
) -> tuple[subprocess.CompletedProcess, str | None]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(FAKE_KUBECTL)
    kubectl.chmod(0o755)
    applied = tmp_path / "applied"
    result = subprocess.run(
        ["bash", str(K8S / "make-storage-secret.sh"), "ctx", "ns", "storage", *args],
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "CAPTURE": str(applied),
            **({"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"} if keys else {}),
            **env,
        },
        capture_output=True,
        text=True,
    )
    return result, applied.read_text() if applied.exists() else None


@pytest.mark.parametrize(
    ("args", "env", "region"),
    [
        ((), {"AWS_REGION": "US-WEST-04A"}, "US-WEST-04A"),
        ((), {"AWS_DEFAULT_REGION": "EU-SOUTH-04A"}, "EU-SOUTH-04A"),
        ((), {"AWS_REGION": "US-WEST-04A", "AWS_DEFAULT_REGION": "EU-SOUTH-04A"}, "US-WEST-04A"),
        (("https://cwobject.com", "RNO2A"), {"AWS_REGION": "US-WEST-04A"}, "RNO2A"),
    ],
)
def test_the_secret_carries_the_configured_region(tmp_path: Path, args, env, region) -> None:
    result, applied = make_secret(tmp_path, *args, **env)
    assert result.returncode == 0, result.stderr
    assert f"region = {region}\n" in applied


def test_the_secret_needs_a_region(tmp_path: Path) -> None:
    result, applied = make_secret(tmp_path)
    assert result.returncode != 0 and "Availability Zone" in result.stderr
    for source in ("AWS_REGION", "AWS_DEFAULT_REGION", "region argument"):
        assert source in result.stderr
    assert applied is None  # nothing reached the cluster


def bucket_job_script() -> str:
    lines = (K8S / "make-bucket-job.yaml").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "python - <<'PY'") + 1
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "PY")
    return textwrap.dedent("\n".join(lines[start:end])) + "\n"


def run_bucket_job(
    tmp_path: Path,
    zone: str,
    endpoint: str = "https://cwobject.com",
    configured: str = "",
    **overrides: str,
) -> tuple[subprocess.CompletedProcess, list]:
    fake = tmp_path / "fake"
    (fake / "boto3").mkdir(parents=True)
    (fake / "boto3" / "__init__.py").write_text(FAKE_BOTO3)
    (fake / "botocore").mkdir()
    (fake / "botocore" / "__init__.py").write_text("")
    (fake / "botocore" / "config.py").write_text(
        "class Config:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n"
        "    def merge(self, other):\n"
        "        return Config(**{**self.kwargs, **other.kwargs})\n"
    )
    script = tmp_path / "make_bucket.py"
    script.write_text(bucket_job_script())
    calls = tmp_path / "calls"
    result = subprocess.run(
        [sys.executable, str(script)],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": str(fake),
            "CALLS": str(calls),
            "BUCKET": "my-checkpoints",
            "BUCKET_ZONE": zone,
            "PLATFORM_PRINCIPAL_ARN": PLATFORM_ARN,
            "ORG_ID": "org",
            "TRAINER_PRINCIPAL_ARN": TRAINER_ARN,
            "ADMIN_PRINCIPAL_ARN": "arn:aws:iam::org:coreweave/admin",
            "BUCKET_POLICY_TEMPLATE": str(RECIPE / "bucket-policy.example.json"),
            "HOTLOAD_PREFIX": "hotload",
            "MODEL_PREFIX": "models",
            "CW_S3_ENDPOINT": endpoint,
            **({"FAKE_CONFIGURED_ENDPOINT": configured} if configured else {}),
            **overrides,
        },
        capture_output=True,
        text=True,
    )
    recorded = (
        [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    )
    return result, recorded


def test_the_bucket_job_signs_for_the_bucket_zone(tmp_path: Path) -> None:
    result, calls = run_bucket_job(tmp_path, "US-WEST-04A")
    assert result.returncode == 0, result.stderr
    clients = [kwargs for name, kwargs in calls if name == "client"]
    (create,) = [kwargs for name, kwargs in calls if name == "create_bucket"]
    assert {client["region_name"] for client in clients} == {"US-WEST-04A"}
    assert create["CreateBucketConfiguration"] == {"LocationConstraint": "US-WEST-04A"}


@pytest.mark.parametrize(
    ("endpoint", "configured", "used"),
    [
        ("http://cwlota.com", "https://other.example", "http://cwlota.com"),  # CW_S3_ENDPOINT first
        ("", "http://cwlota.com", "http://cwlota.com"),  # else the storage Secret's config
        # even an AWS URL that boto3 would also pick for the zone on its own
        ("", "https://s3.US-WEST-04A.amazonaws.com", "https://s3.US-WEST-04A.amazonaws.com"),
        (" ", "", "https://cwobject.com"),  # else the primary CoreWeave endpoint
    ],
)
def test_the_bucket_job_takes_the_endpoint_in_the_trainers_order(
    tmp_path: Path, endpoint, configured, used
) -> None:
    result, calls = run_bucket_job(tmp_path, "US-WEST-04A", endpoint, configured)
    assert result.returncode == 0, result.stderr
    (create,) = [kwargs for name, kwargs in calls if name == "create_bucket"]
    assert create["endpoint"] == used
    if endpoint.strip():  # CW_S3_ENDPOINT decides, so the AWS configuration is not even probed
        assert len([name for name, _ in calls if name == "client"]) == 1


def test_the_bucket_job_needs_a_zone(tmp_path: Path) -> None:
    result, calls = run_bucket_job(tmp_path, "")
    assert result.returncode != 0 and "BUCKET_ZONE" in result.stderr
    assert calls == []  # no client, no bucket


def filled_example_policy() -> dict:
    text = (RECIPE / "bucket-policy.example.json").read_text()
    for placeholder, value in (
        ("[BUCKET-NAME]", "my-checkpoints"),
        ("[ORG-ID]", "org"),
        ("[HOT-LOAD-PREFIX]", "hotload"),
        ("[OIDC-ISSUER-URL]", "https://oidc.example/id/c1"),
        ("[NAMESPACE]", "ns"),
        ("[SERVICE-ACCOUNT]", "trainer-sa"),
        ("[ADMIN-USER-ID]", "admin"),
    ):
        text = text.replace(placeholder, value)
    return json.loads(text)


def test_the_bucket_job_attaches_the_example_policy(tmp_path: Path) -> None:
    result, calls = run_bucket_job(tmp_path, "US-WEST-04A")
    assert result.returncode == 0, result.stderr
    (attached,) = [kwargs for name, kwargs in calls if name == "put_bucket_policy"]
    assert json.loads(attached["Policy"]) == filled_example_policy()


def test_only_the_admin_may_delete_and_the_trainer_writes_only_under_the_hot_load_prefix() -> None:
    statements = {st["Sid"]: st for st in filled_example_policy()["Statement"]}
    for sid, st in statements.items():
        assert "s3:PutBucketPolicy" not in st["Action"] and "s3:*" not in st["Action"], sid
        if sid != "AllowAdmin":
            assert "s3:DeleteObject" not in st["Action"], sid
        writes = {"s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"} & set(st["Action"])
        assert not writes or st["Principal"] != "*", sid  # no org-wide writes
    trainer = statements["AllowTrainer"]
    assert trainer["Principal"] == {"AWS": [TRAINER_ARN]}
    assert set(trainer["Action"]) == {"s3:PutObject", "s3:AbortMultipartUpload"}
    assert trainer["Resource"] == ["arn:aws:s3:::my-checkpoints/hotload/*"]


@pytest.mark.parametrize(
    "unset", ["TRAINER_PRINCIPAL_ARN", "ADMIN_PRINCIPAL_ARN", "HOTLOAD_PREFIX"]
)
def test_the_bucket_job_needs_both_principals_and_the_prefix(tmp_path: Path, unset) -> None:
    result, calls = run_bucket_job(tmp_path, "US-WEST-04A", **{unset: " "})
    assert result.returncode != 0 and unset in result.stderr
    assert calls == []  # no client, no bucket


def attached_policy(calls: list) -> dict:
    (attached,) = [kwargs for name, kwargs in calls if name == "put_bucket_policy"]
    return {st["Sid"]: st for st in json.loads(attached["Policy"])["Statement"]}


def test_running_the_bucket_job_again_keeps_earlier_principals_and_other_statements(
    tmp_path: Path,
) -> None:
    # a second recipe or user must not lock out the trainer and admin of the first, nor drop a
    # statement that someone else wrote
    earlier = filled_example_policy()
    statements = {st["Sid"]: st for st in earlier["Statement"]}
    statements["AllowTrainer"]["Principal"] = {"CW": ["arn:aws:iam::org:coreweave/first-trainer"]}
    statements["AllowAdmin"]["Principal"] = {"CW": ["arn:aws:iam::org:coreweave/first-admin"]}
    keep = {"Sid": "Theirs", "Effect": "Allow", "Principal": "*", "Action": ["s3:GetObject"],
            "Resource": ["arn:aws:s3:::my-checkpoints/public/*"]}  # fmt: skip
    earlier["Statement"].append(keep)
    result, calls = run_bucket_job(tmp_path, "US-WEST-04A", FAKE_POLICY=json.dumps(earlier))
    assert result.returncode == 0, result.stderr
    policy = attached_policy(calls)
    assert policy["AllowTrainer"]["Principal"] == {
        "AWS": [TRAINER_ARN],
        "CW": ["arn:aws:iam::org:coreweave/first-trainer"],
    }
    assert policy["AllowAdmin"]["Principal"] == {
        "CW": ["arn:aws:iam::org:coreweave/admin", "arn:aws:iam::org:coreweave/first-admin"]
    }
    assert policy["AllowTrainer"]["Resource"] == ["arn:aws:s3:::my-checkpoints/hotload/*"]
    assert policy["Theirs"] == keep


def test_the_bucket_job_stops_when_it_cannot_read_the_current_policy(tmp_path: Path) -> None:
    # overwriting a policy it could not read could lock out other users of the bucket
    result, calls = run_bucket_job(tmp_path, "US-WEST-04A", FAKE_POLICY_ERROR="AccessDenied")
    assert result.returncode != 0 and "AccessDenied" in result.stderr
    assert "put_bucket_policy" not in [name for name, _ in calls]


def test_the_bucket_job_puts_a_console_user_under_cw(tmp_path: Path) -> None:
    user = "arn:aws:iam::org:coreweave/trainer-user"
    result, calls = run_bucket_job(tmp_path, "US-WEST-04A", TRAINER_PRINCIPAL_ARN=user)
    assert result.returncode == 0, result.stderr
    assert attached_policy(calls)["AllowTrainer"]["Principal"] == {"CW": [user]}


def test_the_bucket_job_warns_when_the_trainer_is_the_admin(tmp_path: Path) -> None:
    admin = "arn:aws:iam::org:coreweave/admin"
    result, _ = run_bucket_job(tmp_path, "US-WEST-04A", TRAINER_PRINCIPAL_ARN=admin)
    assert result.returncode == 0, result.stderr
    assert "WARNING: TRAINER_PRINCIPAL_ARN is ADMIN_PRINCIPAL_ARN" in result.stdout
    separate, _ = run_bucket_job(tmp_path / "separate", "US-WEST-04A")
    assert separate.returncode == 0 and "WARNING" not in separate.stdout


def test_a_workload_identity_secret_holds_no_key(tmp_path: Path) -> None:
    result, applied = make_secret(tmp_path, keys=False, WORKLOAD_IDENTITY="1", AWS_REGION="US-WEST-04A")
    assert result.returncode == 0, result.stderr
    assert "region = US-WEST-04A\n" in applied and "aws_access_key_id" not in applied
    assert "no key" in result.stdout


def test_a_secret_without_a_key_needs_workload_identity(tmp_path: Path) -> None:
    result, applied = make_secret(tmp_path, keys=False, AWS_REGION="US-WEST-04A")
    assert result.returncode != 0 and "WORKLOAD_IDENTITY=1" in result.stderr
    assert applied is None  # nothing reached the cluster
