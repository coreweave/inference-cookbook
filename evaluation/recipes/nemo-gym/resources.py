# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Create, inspect and remove a temporary Gym deployment through cwtrain."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from cwtrain.errors import NotFoundError
from cwtrain.infra import InferenceClient, lifecycle


def provision(
    client, state_path: Path, spec: dict, organization: str, zone: str
) -> dict:
    zone = zone.upper()
    if zone not in client.get_gateway_parameters().get("zones", []):
        raise ValueError(f"unavailable gateway zone: {zone}")
    # Verify the token's org before mutation when the account has existing gateways.
    gateways = client.list_gateways()
    if gateways and any(g.organization_id != organization for g in gateways):
        raise ValueError(
            "gateway inventory does not belong to the requested organization"
        )
    gateway_id, deployment_id = str(uuid.uuid4()), str(uuid.uuid4())
    name = f"gym-nemotron-{gateway_id[:8]}"
    state = {
        "gateway_id": gateway_id,
        "deployment_id": deployment_id,
        "name": name,
        "organization": organization,
        "zone": zone,
        "model": spec["model"]["name"],
    }
    # Journal IDs before POST so an interrupted or ambiguous create can still be cleaned up.
    # Exclusive creation prevents overwriting the only record of an earlier run.
    with state_path.open("x") as f:
        json.dump(state, f, indent=2)
    gateway = client.create_gateway(
        {
            "id": gateway_id,
            "name": name,
            "zones": [zone],
            "coreWeaveAuth": {},
            "bodyBasedRouting": {"apiType": "API_TYPE_OPENAI"},
        }
    )
    if gateway.organization_id != organization:
        raise ValueError(
            "created gateway belongs to a different organization; run cleanup"
        )
    params = client.get_deployment_parameters([gateway_id])
    runtime = spec["runtime"]
    versions = params["runtimeParameters"]["runtimeVersions"].get(runtime["engine"], {})
    if runtime["version"] not in versions.get("versions", []):
        raise ValueError(
            "runtime version unavailable in the selected zone; run cleanup"
        )
    if (
        spec["resources"]["instanceType"]
        not in params["resourceParameters"]["instanceTypes"]
    ):
        raise ValueError("instance type unavailable in the selected zone; run cleanup")
    client.create_deployment(
        dict(spec, id=deployment_id, name=name, gatewayIds=[gateway_id])
    )
    return state


def status(client, state: dict) -> dict:
    gateway = client.get_gateway(state["gateway_id"])
    deployment = client.get_deployment(state["deployment_id"])
    return {
        "gateway_status": gateway.status.name,
        "deployment_status": deployment.status.name,
        "endpoints": list(gateway.endpoints),
        "model": deployment.model_name,
        "ready": gateway.ready and deployment.ready,
        "conditions": deployment.raw.get("status", {}).get("conditions", []),
    }


def cleanup(client, state: dict, timeout: float = 300) -> None:
    # Delete only this run's journaled IDs, deployment first. Keep the journal for retries.
    for kind, delete in (
        ("deployment", lifecycle.delete_deployment),
        ("gateway", lifecycle.delete_gateway),
    ):
        resource_id = state[f"{kind}_id"]
        try:
            resource = getattr(client, f"get_{kind}")(resource_id)
        except NotFoundError:
            continue
        if resource.name != state["name"]:
            raise ValueError(
                f"refusing to delete {kind}: name differs from run journal"
            )
        # waits until the resource is gone; on a timeout, rerun cleanup
        delete(client, resource_id, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("provision", "status", "cleanup", "parameters")
    )
    parser.add_argument("--state", type=Path)
    parser.add_argument("--deployment-spec", type=Path)
    parser.add_argument("--organization")
    parser.add_argument("--zone", default="US-WEST-04A")
    parser.add_argument("--gateway-id", help="Scope parameters to an existing gateway")
    args = parser.parse_args()
    if args.action != "parameters" and args.state is None:
        parser.error("--state is required")
    if args.action == "provision" and not (args.organization and args.deployment_spec):
        parser.error("provision requires --organization and --deployment-spec")
    with InferenceClient() as client:
        if args.action == "parameters":
            result = client.get_deployment_parameters(
                [args.gateway_id] if args.gateway_id else None
            )
        elif args.action == "provision":
            result = provision(
                client,
                args.state,
                json.loads(args.deployment_spec.read_text()),
                args.organization,
                args.zone,
            )
        else:
            state = json.loads(args.state.read_text())
            if args.action == "cleanup":
                cleanup(client, state)
                result = {"cleanup": "confirmed absent", "name": state["name"]}
            else:
                result = status(client, state)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
