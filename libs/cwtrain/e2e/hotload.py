# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Opt-in live service test. Run from libs/cwtrain with python -m e2e.hotload."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

from cwtrain.hotload import HotLoader
from cwtrain.infra import InferenceClient, lifecycle
from e2e.storage import (
    bucket_client,
    configured_bucket,
    delete_run_objects,
    new_run_prefix,
    stage_model,
)

log = logging.getLogger(__name__)

#: Default dynamo-vllm runtime. The ModelExpress (MX) client that performs hot-loads ships inside
#: the engine build, so the ``+crwv.N`` suffix also selects the MX client version.
DEFAULT_RUNTIME_VERSION = "0.30.0+crwv.5"


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} before running the live E2E test")
    return value


def cleanup(client: InferenceClient, state: dict, timeout: float = 300) -> None:
    """Try all resources, deployment first; confirm deletion rather than acceptance."""
    failures = []
    # Deleting a deployment with a serving replica has taken over five minutes, so it gets twice
    # the time of the gateway and the objects.
    for kind, delete, wait in (
        ("deployment", lifecycle.delete_deployment, 2 * timeout),
        ("gateway", lifecycle.delete_gateway, timeout),
    ):
        resource_id = state[f"{kind}_id"]
        try:
            if delete(client, resource_id, timeout=wait):
                log.info("Deleted %s %s", kind, resource_id)
            else:
                log.info("%s %s was already gone", kind.capitalize(), resource_id)
        except Exception as exc:
            # A failure must not stop the cleanup of the other resources.
            log.warning("Cleanup of %s %s: %s", kind, resource_id, type(exc).__name__)
            failures.append(f"{kind} {resource_id}")
    if state.get("bucket"):
        try:
            delete_run_objects(state["bucket"], state.get("prefix", ""), timeout=timeout)
        except Exception as exc:
            log.warning("Storage cleanup failed: %s", exc)
            failures.append(f"objects in {state['bucket']}/{state.get('prefix', '')}")
    if failures:
        raise RuntimeError("Cleanup incomplete; retry --cleanup-only for " + ", ".join(failures))


def exercise(client: InferenceClient, state: dict, spec: dict, checkpoint: str, zone: str) -> None:
    # The platform rejects a hot-load identity that another deployment in the organization used.
    # A per-run identity keeps the next run from failing.
    identity = f"{state['name']}-checkpoint"
    client.create_gateway(
        {
            "id": state["gateway_id"],
            "name": state["name"],
            "zones": [zone],
            "coreWeaveAuth": {},
            "bodyBasedRouting": {"apiType": "API_TYPE_OPENAI"},
        }
    )
    lifecycle.wait_for_gateway(client, state["gateway_id"])
    client.create_deployment(
        {
            **spec,
            "id": state["deployment_id"],
            "name": state["name"],
            "gatewayIds": [state["gateway_id"]],
        }
    )
    # If the deployment does not become READY, lifecycle logs its unmet conditions.
    lifecycle.wait_for_deployment(client, state["deployment_id"], timeout=1200)
    loader = HotLoader(client, state["deployment_id"], timeout=900)
    completed = loader.publish(checkpoint, identity)
    if completed.identity != identity or completed.deployment_id != state["deployment_id"]:
        raise AssertionError("Hot-load response does not match this test")
    loader.verify_served(completed.id, identity)
    if loader.current_identity() != identity:
        raise AssertionError("Completed hot-load missing from deployment history")
    log.info("Verified hot-load %s and all replicas", completed.id)


def run_resources(client: InferenceClient, state: dict, spec: dict, zone: str) -> None:
    try:
        bucket_client().head_bucket(Bucket=state["bucket"])
        with tempfile.TemporaryDirectory(prefix="cwtrain-e2e-") as directory:
            checkpoint = stage_model(state["bucket"], state["prefix"], directory)
            exercise(client, state, spec, checkpoint, zone)
    finally:
        # Includes partially successful staging and inference failures.
        cleanup(client, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--cleanup-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.cleanup_only and not args.state.exists():
        log.info("No resource state; provisioning never started")
        return
    if args.cleanup_only:
        with InferenceClient() as client:
            cleanup(client, json.loads(args.state.read_text()))
        return
    # Validate all inputs before allocating resources. Never log the deployment configuration.
    spec = json.loads(required("CWTRAIN_E2E_DEPLOYMENT_JSON"))
    for field in ("runtime", "resources", "autoscaling", "traffic"):
        if not isinstance(spec.get(field), dict) or not spec[field]:
            raise ValueError(f"Deployment configuration requires {field}")
    if spec["runtime"].get("engine") != "dynamo-vllm":
        raise ValueError("E2E hot-load requires runtime.engine=dynamo-vllm")
    spec["runtime"]["version"] = (
        os.environ.get("CWTRAIN_E2E_RUNTIME_VERSION", "").strip() or DEFAULT_RUNTIME_VERSION
    )
    if spec.get("disabled"):
        raise ValueError("E2E deployment must be enabled")
    if spec["autoscaling"].get("min", 0) < 1:
        raise ValueError("E2E deployment requires at least one replica")
    bucket = configured_bucket()
    zone = (os.environ.get("CWTRAIN_E2E_ZONE", "").strip() or "US-WEST-04A").upper()
    log.info("Runtime: dynamo-vllm %s", spec["runtime"]["version"])
    with InferenceClient() as client:
        state = {
            "gateway_id": str(uuid.uuid4()),
            "deployment_id": str(uuid.uuid4()),
            "name": "cwtrain-e2e-" + uuid.uuid4().hex[:16],
            "bucket": bucket,
            "prefix": new_run_prefix(),
        }
        # Save client-generated IDs BEFORE either POST, including ambiguous network failures.
        # Exclusive creation prevents accidentally overwriting a previous run's cleanup IDs.
        with args.state.open("x") as stream:
            json.dump(state, stream)
        log.info("Cleanup state: %s (resources: %s)", args.state, state)
        spec["model"] = {
            "name": state["name"],
            "bucket": state["bucket"],
            "path": f"{state['prefix']}models/base",
        }
        spec["hotLoad"] = {
            "bucket": state["bucket"],
            "pathPrefix": f"{state['prefix']}hotload",
            "transitionMode": "TRANSITION_MODE_ASYNC",
        }
        run_resources(client, state, spec, zone)


if __name__ == "__main__":
    main()
