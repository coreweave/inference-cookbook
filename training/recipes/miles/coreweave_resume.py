# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Validate a fresh DI publication chain before resuming native Miles state."""

from cwtrain.errors import CWTrainError
from cwtrain.hotload import HotLoader, PromptCachePolicy
from cwtrain.hotload.snapshot import snapshot_prefix
from cwtrain.infra.client import InferenceClient
from cwtrain.integrations.slime import HookConfig


def validate_resume(args) -> None:
    """Keep native optimizer/data recovery separate from the new transport version sequence.

    Miles seeds the delta encoder from the original HF bytes, then publishes the loaded
    trainer weights before generating a rollout. A restart must use a new publication
    identity so an earlier transport version cannot be mistaken for the restored model.
    """
    if (args.rollout_num_gpus != 0
            or args.custom_update_weight_post_write_path != "cwtrain.integrations.slime.hook"):
        raise CWTrainError("Native disk-delta resume requires the CoreWeave hook and zero local rollout GPUs")
    config = HookConfig.from_env()
    if not (config.publish_baseline and config.wait and config.verify and config.mirror_enabled):
        raise CWTrainError("Native resume requires baseline publication, WAIT=1, VERIFY=1, and the weight mirror")
    if config.prompt_cache_policy is not PromptCachePolicy.RESET_ALL:
        raise CWTrainError("Native resume requires CWTRAIN_HOTLOAD_PROMPT_CACHE_POLICY=reset_all")
    prefix = f"{config.run_id}-v"
    with InferenceClient(base_url=config.api_base_url, headers=config.api_headers or None,
                         verify=not config.api_insecure) as client:
        if any(item.identity.startswith(prefix) for item in client.list_hot_loads(config.deployment_id)):
            raise CWTrainError("Native resume requires a fresh CWTRAIN_HOTLOAD_RUN_ID; this one has hot-load history")
        loader = HotLoader(client, config.deployment_id)
        storage = loader.deployment.hot_load
        objects = loader.s3.list_objects_v2(
            Bucket=storage.bucket, Prefix=snapshot_prefix(storage.path_prefix, prefix), MaxKeys=1,
        )
        if objects.get("Contents"):
            raise CWTrainError("Native resume requires a fresh CWTRAIN_HOTLOAD_RUN_ID; this one has stored snapshots")
