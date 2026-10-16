# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Run-owned prefixes in an existing CAIOS bucket. Credentials come from CWIC credential_process."""

import os
import re
import time
import uuid
from datetime import UTC, datetime

from botocore.exceptions import ClientError

from cwtrain.hotload.snapshot import inspect_snapshot, upload_snapshot
from cwtrain.infra.storage import s3_client


def configured_bucket() -> str:
    bucket = os.environ.get("CWTRAIN_E2E_BUCKET", "").strip()
    if not bucket:
        raise ValueError("Set CWTRAIN_E2E_BUCKET to an existing bucket")
    return bucket


def bucket_client():
    return s3_client(region_name=(os.environ.get("AWS_REGION") or "us-west-04a").upper())


# <UTC start, YYYYMMDD-HHMM>-<random>: sorts by start time, and stays unique within a minute,
# because cleanup deletes everything under the prefix.
RUN_PREFIX = re.compile(r"cookbook/cwtrain-e2e/[0-9]{8}-[0-9]{4}-[0-9a-f]{8}/")


def new_run_prefix(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%d-%H%M")
    return f"cookbook/cwtrain-e2e/{stamp}-{uuid.uuid4().hex[:8]}/"


def validate_location(bucket: str, prefix: str) -> None:
    if bucket != configured_bucket() or not RUN_PREFIX.fullmatch(prefix):
        raise ValueError("Refusing storage access outside the shared bucket's E2E run prefix")


def stage_model(bucket: str, prefix: str, directory: str) -> str:
    from huggingface_hub import snapshot_download

    validate_location(bucket, prefix)
    checkpoint = snapshot_download(
        repo_id=os.environ.get("CWTRAIN_E2E_HF_MODEL") or "Qwen/Qwen3-0.6B",
        revision=os.environ.get("CWTRAIN_E2E_HF_REVISION") or "main",
        local_dir=directory,
        ignore_patterns=["original/*", "*.pth", "*.gguf", "*.bin"],
    )
    upload_snapshot(
        inspect_snapshot(checkpoint),
        bucket=bucket,
        path_prefix=f"{prefix}models",
        identity="base",
        s3=bucket_client(),
    )
    return checkpoint


def delete_run_objects(bucket: str, prefix: str, timeout: float = 300) -> None:
    # Require the full generated prefix, including its trailing slash. A state file without a
    # prefix must never cause a bucket-wide delete.
    validate_location(bucket, prefix)
    s3 = bucket_client()
    deadline = time.monotonic() + timeout
    first_pass = True
    while True:
        try:
            objects = [
                {"Key": item["Key"]}
                for page in s3.get_paginator("list_objects_v2").paginate(
                    Bucket=bucket, Prefix=prefix
                )
                for item in page.get("Contents", [])
            ]
            if any(not item["Key"].startswith(prefix) for item in objects):
                raise ValueError("Storage returned an object outside the run prefix")
            uploads = [
                item
                for page in s3.get_paginator("list_multipart_uploads").paginate(
                    Bucket=bucket, Prefix=prefix
                )
                for item in page.get("Uploads", [])
            ]
            if any(not item["Key"].startswith(prefix) for item in objects + uploads):
                raise ValueError("Storage returned an object or upload outside the run prefix")
            if not objects and not uploads:
                return
            if not first_pass:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Objects or uploads remain in {bucket}/{prefix}")
                time.sleep(5)
            first_pass = False
            for offset in range(0, len(objects), 1000):
                result = s3.delete_objects(
                    Bucket=bucket, Delete={"Objects": objects[offset : offset + 1000]}
                )
                if result.get("Errors"):
                    raise RuntimeError("CAIOS reported object deletion failures")
            for upload in uploads:
                s3.abort_multipart_upload(
                    Bucket=bucket, Key=upload["Key"], UploadId=upload["UploadId"]
                )
            # Confirm absence immediately, even if deletion used the remaining time.
            continue
        except (ValueError, TimeoutError):
            raise
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "NoSuchBucket":
                raise
            if time.monotonic() >= deadline:
                raise
        except Exception:
            if time.monotonic() >= deadline:
                raise
        time.sleep(5)
