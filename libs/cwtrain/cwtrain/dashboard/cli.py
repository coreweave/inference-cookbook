# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Dashboard commands use W&B credentials, independently of inference credentials."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cwtrain.dashboard.catalog import SUPPORTED_FRAMEWORKS
from cwtrain.dashboard.workspace import (
    build_workspace,
    default_state_file,
    definition,
    save_managed,
)
from cwtrain.errors import CWTrainError


def add_parser(subparsers: Any) -> None:
    commands = subparsers.add_parser(
        "dashboard", help="native W&B post-training views"
    ).add_subparsers(dest="cmd", required=True)
    setup = commands.add_parser("setup", help="create or refresh a managed W&B saved view")
    setup.add_argument("--entity", required=True)
    setup.add_argument("--project", required=True)
    setup.add_argument("--framework", choices=SUPPORTED_FRAMEWORKS, required=True)
    setup.add_argument(
        "--dry-run", action="store_true", help="print the offline metric specification"
    )
    setup.add_argument("--state-file", type=Path, default=None)
    timeline = commands.add_parser(
        "timeline", help="export an interactive local timeline prototype"
    )
    timeline.add_argument("--run", required=True, help="entity/project/run-id")
    timeline.add_argument("--output", type=Path, required=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cmd == "setup" and args.dry_run:
        return {"entity": args.entity, "project": args.project, **definition(args.framework)}
    try:
        import wandb

        sdk_errors = ()
        if args.cmd == "setup":
            from wandb_workspaces.workspaces.errors import SpecVersionError, UnsupportedViewError

            sdk_errors = (SpecVersionError, UnsupportedViewError)
    except ImportError as exc:
        raise CWTrainError(
            "cwtrain installation is incomplete; reinstall with: pip install --upgrade cwtrain"
        ) from exc
    try:
        api = wandb.Api(timeout=30)
        if args.cmd == "timeline":
            from cwtrain.dashboard.timeline import export_html

            if len(args.run.strip("/").split("/")) != 3:
                raise CWTrainError("--run must be entity/project/run-id")
            source = api.run(args.run)
            # No keys filter: older sparse history rows need not contain every lane.
            page = export_html(
                source.scan_history(page_size=1000), title=f"Post-training · {source.name}"
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(page, encoding="utf-8")
            return {"output": str(args.output.resolve()), "run": args.run, "prototype": True}

        candidates = itertools.islice(
            api.runs(
                f"{args.entity}/{args.project}",
                filters={"jobType": "train"},
                order="-created_at",
                per_page=50,
            ),
            200,
        )
        # A newly tagged run can be returned by ID but missed by the server's tag
        # index. Read the actual tags so recent runs are not silently excluded.
        runs = list(
            itertools.islice(
                (source for source in candidates if args.framework in source.tags),
                10,
            )
        )
        if not runs:
            raise CWTrainError(
                f"no runs tagged {args.framework!r} among the latest 200 train runs "
                f"in {args.entity}/{args.project}; "
                "tag the intended runs or use --dry-run to inspect the full specification"
            )
        metrics, systems = set(), set()
        for source in runs:
            metrics.update(dict(source.summary))
            # This is key discovery, not chart data; native charts query their own history.
            for row in source.history(stream="system", samples=20, pandas=False):
                systems.update(row)
        plan = definition(args.framework, metric_keys=metrics, system_keys=systems)
        plan["included_runs"] = [source.id for source in runs]
        workspace = build_workspace(args.entity, args.project, plan)
        result = save_managed(
            workspace,
            plan,
            path=args.state_file or default_state_file(),
            api_host=api.settings["base_url"],
            username=api.viewer.username,
            app_url=runs[0].url,
        )
        # The workspace SDK drops reverse-proxy path prefixes in its generated URL.
        # Retain its loadable URL in state, but return the run's canonical browser route.
        result["url"] = runs[0].url.rsplit("/runs/", 1)[0] + "?" + urlsplit(result["url"]).query
        return {
            **result,
            "framework": args.framework,
            "discovered_runs": [source.id for source in runs],
            "panels": len(plan["panels"]),
            "omitted_metrics": plan["omitted_metrics"],
        }
    except CWTrainError:
        raise
    except (wandb.Error, OSError, ValueError, KeyError, *sdk_errors) as exc:
        # SDK/network exceptions can include request details: keep the normal CLI error concise.
        raise CWTrainError(
            f"dashboard operation failed ({type(exc).__name__}); check W&B access, "
            "WANDB_BASE_URL, and the state/output path; use --verbose for details"
        ) from exc
