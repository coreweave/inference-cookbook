# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cwtrain>=1.0.3,<2",
# ]
# ///

"""Laptop-only Dedicated Inference driver for hotload_notebook.py.

    uv run training/examples/hotload_laptop.py --run-id molab-demo-001 \
        --bucket YOUR_BUCKET --zone US-EAST-02A --storage-auth cwic

All management and hot-load control originate here. Molab samples through a W&B-authenticated
gateway; snapshots and control acknowledgments travel through CAIOS.
See hotload-notebook.md for credentials and lifecycle.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import time
import uuid
from pathlib import Path

LOG = logging.getLogger("hotload-laptop")
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def validate_run_id(value):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", value):
        raise ValueError("run ID must be 1–63 lowercase letters, digits or hyphens")
    return value


class Mailbox:
    """One prefix per run; JSON markers commit completed uploads, never partial snapshots."""

    def __init__(self, s3, bucket, run_id, *, session=None, timeout=1800, poll_interval=2):
        self.s3, self.bucket = s3, bucket
        self.run_id = validate_run_id(run_id)
        self.prefix = f"cwtrain-molab/{run_id}"
        self.session = session
        self.timeout, self.poll_interval = timeout, poll_interval

    def read(self, relative):
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}/{relative}")
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        with response["Body"] as body:
            return json.load(body)

    def check(self, message, step=None):
        if (message.get("schema"), message.get("run_id")) != (1, self.run_id):
            raise ValueError("message belongs to a different run or protocol version")
        if self.session is not None and message.get("session") != self.session:
            raise ValueError("trainer session changed; restart both sides with a fresh run ID")
        if step is not None and message.get("step") != step:
            raise ValueError("message step does not match the outstanding request")
        return message

    def put(self, relative, **fields):
        if not self.session:
            raise ValueError("wait for the trainer's ready marker first")
        message = {"schema": 1, "run_id": self.run_id, "session": self.session, **fields}
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/{relative}",
            Body=json.dumps(message, allow_nan=False).encode(),
            ContentType="application/json",
        )

    def wait(self, relative, step=None):
        deadline = time.monotonic() + self.timeout
        while True:
            message = self.read(relative)
            if message is not None:
                return self.check(message, step)
            error = self.read("control/trainer-error.json")
            if error is not None:
                self.check(error)
                raise RuntimeError(
                    f"Molab training failed at step {error.get('step')}; see the notebook"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"waiting for s3://{self.bucket}/{self.prefix}/{relative}")
            time.sleep(self.poll_interval)

    def ready(self):
        ready = self.wait("ready.json")
        if not ready.get("session"):
            raise ValueError("trainer ready marker has no session")
        if (ready.get("model"), ready.get("revision"), ready.get("root_identity")) != (
            MODEL,
            REVISION,
            f"{self.run_id}-v000000",
        ):
            raise ValueError("trainer model, revision or chain root does not match this driver")
        self.session = ready["session"]
        return ready


def accept_result(mailbox, result, step, previous_identity):
    """Validate the chain and uploaded index before any management-side hot-load request."""
    mailbox.check(result, step)
    if (
        result.get("previous_identity") != previous_identity
        or type(result.get("empty")) is not bool
    ):
        raise ValueError("trainer result has the wrong predecessor or empty-delta flag")
    expected = previous_identity if result["empty"] else f"{mailbox.run_id}-v{step:06d}"
    if result.get("identity") != expected:
        raise ValueError("trainer result has the wrong snapshot identity")
    if not result["empty"]:
        index = mailbox.read(f"hotload/{expected}/model.safetensors.index.json")
        if index is None:
            raise ValueError("trainer result was published before its snapshot index")
        meta = index.get("metadata", {})
        if (
            meta.get("version"),
            meta.get("base_version"),
            meta.get("delta_encoding"),
            meta.get("compression_format"),
            meta.get("checksum_format"),
        ) != (expected, previous_identity, "xor", "zstd", "adler32"):
            raise ValueError("uploaded delta has an unexpected version or encoding")
        shards = set(index.get("weight_map", {}).values())
        if not shards:
            raise ValueError("nonempty delta has no shards")
        for shard in shards:
            if not isinstance(shard, str) or "/" in shard or not shard.endswith(".safetensors"):
                raise ValueError("invalid snapshot shard name")
            mailbox.s3.head_object(
                Bucket=mailbox.bucket, Key=f"{mailbox.prefix}/hotload/{expected}/{shard}"
            )
    return expected


def finish_step(mailbox, loader, step, previous_identity, *, step_started=None):
    from cwtrain.hotload import PromptCachePolicy

    result = mailbox.wait(f"results/{step:06d}.json", step)
    identity = accept_result(mailbox, result, step, previous_identity)
    hotload_started = time.monotonic()
    if not result["empty"]:
        hot_load = loader.trigger(
            identity,
            previous_identity=previous_identity,
            prompt_cache_policy=PromptCachePolicy.RESET_ALL,
        )
        loader.wait(hot_load.id)
        loader.verify_served(hot_load.id, identity)
    hotload_s = time.monotonic() - hotload_started if not result["empty"] else 0.0
    # This is the barrier: Molab cannot consume the next batch until every replica is verified.
    metrics = dict(result["metrics"])
    metrics["hotload_s"] = hotload_s
    if step_started is not None:
        metrics["step_s"] = time.monotonic() - step_started
    mailbox.put(
        f"acks/{step:06d}.json",
        step=step,
        identity=identity,
        metrics={k: metrics[k] for k in ("hotload_s", "step_s") if k in metrics},
    )
    LOG.info("step %d served as %s; metrics=%s", step, identity, metrics)
    return identity


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def cleanup(client, state, state_path):
    from cwtrain.infra import lifecycle

    # IDs are persisted before create requests, so even an interrupted create can be recovered.
    if state.get("deployment_id"):
        lifecycle.delete_deployment(client, state["deployment_id"])
        state["deployment_id"] = None
        save_state(state_path, state)
    if state.get("created_gateway") and state.get("gateway_id"):
        lifecycle.delete_gateway(client, state["gateway_id"])
        state["gateway_id"] = None
        save_state(state_path, state)


def create_deployment(client, mailbox, args, state, state_path):
    from cwtrain.infra import lifecycle

    # Create a dedicated W&B-authenticated gateway; never reuse one with platform-key auth.
    state.update(gateway_id=str(uuid.uuid4()), created_gateway=True)
    save_state(state_path, state)
    client.create_gateway(
        {
            "id": state["gateway_id"],
            "name": args.run_id,
            "zones": [args.region.upper()],
            "weightsAndBiasesAuth": {},
            "bodyBasedRouting": {"apiType": "API_TYPE_OPENAI"},
        }
    )
    lifecycle.wait_for_gateway(client, state["gateway_id"], timeout=args.timeout)
    state["deployment_id"] = str(uuid.uuid4())
    save_state(state_path, state)
    client.create_deployment(
        {
            "id": state["deployment_id"],
            "name": args.run_id,
            "gatewayIds": [state["gateway_id"]],
            "runtime": {
                "engine": "dynamo-vllm",
                "version": args.runtime,
                "engineConfig": {"max-model-len": "4096", "gpu-memory-utilization": "0.80"},
            },
            "resources": {"instanceType": args.instance_type, "gpuCount": 1},
            "model": {
                "name": args.run_id,
                "bucket": args.bucket,
                "path": f"{mailbox.prefix}/models/base",
            },
            "autoscaling": {"min": 1, "max": 1, "concurrency": 64},
            "traffic": {"weight": 100},
            "hotLoad": {
                "bucket": args.bucket,
                "pathPrefix": f"{mailbox.prefix}/hotload",
                "transitionMode": "TRANSITION_MODE_ASYNC",
            },
        }
    )
    return lifecycle.wait_for_deployment(client, state["deployment_id"], timeout=args.timeout)


def configure_cwic(state_dir, zone):
    binary = shutil.which("cwic")
    if not binary:
        raise RuntimeError("install cwic on the laptop, or use --storage-auth aws")
    config = state_dir / "aws-config"
    state_dir.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f"[default]\nregion = {zone.upper()}\n"
        f"credential_process = {shlex.quote(binary)} auth accesskey api-token\n"
    )
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        os.environ.pop(key, None)
    os.environ.update(
        AWS_CONFIG_FILE=str(config.resolve()),
        CWIC_CACHE_DIR=str((state_dir / "cwic-cache").resolve()),
        AWS_EC2_METADATA_DISABLED="true",
    )


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", required=True, type=validate_run_id)
    p.add_argument("--bucket", default=os.environ.get("CWTRAIN_BUCKET"))
    p.add_argument("--zone", default=os.environ.get("AWS_REGION", "US-EAST-02A"))
    p.add_argument("--region", default=os.environ.get("CWTRAIN_REGION", "US-WEST-04A"))
    p.add_argument("--instance-type", default=os.environ.get("CWTRAIN_INSTANCE_TYPE", "h100-2x"))
    p.add_argument("--runtime", default=os.environ.get("CWTRAIN_RUNTIME_VERSION", "0.32.0+crwv.5"))
    p.add_argument("--steps", type=int, choices=range(1, 11), default=3)
    p.add_argument(
        "--timeout", type=float, default=1800, help="seconds per deployment or mailbox wait"
    )
    p.add_argument("--storage-auth", choices=("aws", "cwic"), default="aws")
    p.add_argument("--state-dir", type=Path, default=Path(".cwtrain-laptop"))
    p.add_argument(
        "--keep-deployment",
        action="store_true",
        help="keep created resources on exit; cleanup explicitly later",
    )
    p.add_argument(
        "--cleanup",
        action="store_true",
        help="delete only resources recorded in this run's local state",
    )
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = (
        os.environ.get("COREWEAVE_API_TOKEN") or os.environ.get("COREWEAVE_API_KEY") or ""
    ).strip()
    if not token:
        raise ValueError("set COREWEAVE_API_TOKEN (or COREWEAVE_API_KEY) on the laptop only")
    os.environ["COREWEAVE_API_TOKEN"] = token
    from cwtrain import HotLoader, InferenceClient
    from cwtrain.infra import lifecycle

    state_dir = args.state_dir / args.run_id
    state_path = state_dir / "state.json"
    with InferenceClient(token=token) as client:
        if args.cleanup:
            cleanup(client, json.loads(state_path.read_text()), state_path)
            return
        if not args.bucket:
            raise ValueError("--bucket or CWTRAIN_BUCKET is required")
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        if state_path.exists():
            raise ValueError(
                "run ID already has local state; use --cleanup if needed and choose a fresh run ID"
            )
        if args.storage_auth == "cwic":
            configure_cwic(state_dir, args.zone)
        import boto3
        from botocore.config import Config

        s3 = boto3.Session().client(
            "s3",
            endpoint_url="https://cwobject.com",
            region_name=args.zone.upper(),
            config=Config(
                s3={"addressing_style": "virtual"},
                connect_timeout=15,
                read_timeout=120,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        mailbox = Mailbox(s3, args.bucket, args.run_id, timeout=args.timeout)
        LOG.info("waiting for Molab to upload s3://%s/%s/ready.json", args.bucket, mailbox.prefix)
        ready = mailbox.ready()
        state = dict(
            run_id=args.run_id,
            session=mailbox.session,
            bucket=args.bucket,
            gateway_id=None,
            deployment_id=None,
            created_gateway=False,
            step=0,
        )
        save_state(state_path, state)
        try:
            deployment = create_deployment(client, mailbox, args, state, state_path)
            endpoints = lifecycle.endpoints(client, deployment)
            if not endpoints:
                raise RuntimeError("ready deployment has no gateway endpoint")
            loader = HotLoader(client, deployment.id, poll_interval=2, timeout=args.timeout)
            identity = ready["root_identity"]
            root = loader.trigger(identity)
            loader.wait(root.id)
            loader.verify_served(root.id, identity)
            for step in range(1, args.steps + 1):
                started = time.monotonic()
                mailbox.put(
                    f"requests/{step:06d}.json",
                    step=step,
                    previous_identity=identity,
                    rollout_base_url=lifecycle.rollout_base_url(endpoints[0]),
                    model=deployment.model_name,
                )
                identity = finish_step(mailbox, loader, step, identity, step_started=started)
                state.update(step=step, identity=identity)
                save_state(state_path, state)
            mailbox.put("control/done.json", step=args.steps, identity=identity)
        except BaseException:
            # Never copy raw API errors (or credentials) into S3 control messages.
            try:
                mailbox.put("control/error.json", step=state["step"])
            except Exception:
                LOG.exception("could not notify the trainer of driver failure")
            raise
        finally:
            if not args.keep_deployment:
                cleanup(client, state, state_path)
            else:
                LOG.warning("resources retained; run this command with --cleanup when finished")


if __name__ == "__main__":
    main()
