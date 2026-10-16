# Dedicated Inference hot-load E2E

The `hotload E2E` workflow runs the hot-load lifecycle against the live
[Dedicated Inference API](https://docs.coreweave.com/products/inference/reference/api-overview).
It does these steps:

1. Download a Hugging Face model. Use `cwtrain` to stage it under a unique run prefix in an
   existing AI Object Storage bucket, set by `CWTRAIN_E2E_BUCKET`.
2. Create a gateway and a deployment with `InferenceClient`.
3. Wait until both are READY with `cwtrain.infra.lifecycle`. If the deployment does not get there,
   it logs the unmet conditions.
4. Call this checkout's `HotLoader.publish()` to upload and hot-load a full checkpoint.
5. Verify the completed hot-load, the identity that every replica serves, and the deployment's
   hot-load history.

The checkpoint has the same weights as the initial model. The test covers the service lifecycle,
not training or generation quality.

## GitHub configuration

Create the `hotload-e2e` GitHub environment and configure:

| Kind | Name | Value |
| --- | --- | --- |
| Secret | `CWTRAIN_E2E_API_TOKEN` | API token with the Inference Admin role and the storage permissions described below, in an organization with RL Rollouts enabled |
| Secret | `CWTRAIN_E2E_DEPLOYMENT_JSON` | Deployment runtime and resources configuration, described below |
| Secret | `CWTRAIN_E2E_HF_TOKEN` | Optional, for a gated or private Hugging Face model |
| Variable | `CWTRAIN_E2E_BUCKET` | Required: an existing bucket that grants the inference service read access under `cookbook/*` |
| Variable | `CWTRAIN_E2E_RUNTIME_VERSION` | `dynamo-vllm` version, `0.30.0+crwv.5` by default; overrides `runtime.version` in the deployment configuration |
| Variable | `CWTRAIN_E2E_ZONE` | `us-west-04a` by default; not a credential |
| Variable | `CWTRAIN_E2E_HF_MODEL` | `Qwen/Qwen3-0.6B` by default; choose a model your runtime supports |
| Variable | `CWTRAIN_E2E_HF_REVISION` | Model revision, `main` by default; pin a commit for reproducibility |

Enter the secrets in GitHub Settings → Environments → hotload-e2e. Restrict the environment's
branches to the default branch, which hosts the comment-triggered workflow.

`cwic` exchanges the API token for temporary AI Object Storage credentials through the AWS
`credential_process`, so you need no long-lived storage keys. The token must be able to:

- get those credentials and access the existing bucket;
- list, read, write and delete objects under `cookbook/cwtrain-e2e/`;
- list and abort multipart uploads under that prefix.

The token needs no permission to create or delete buckets or to change bucket policies. The runner
never changes the bucket's policy. So the bucket must already give the inference service read
access to `cookbook/*`, where the staged model and the hot-load snapshots go.

The E2E bucket keeps its own policy. Its token is the admin and the trainer at once: it stages the
model, publishes snapshots and deletes its run prefix afterwards. So do not apply the recipes'
bucket policy, [`k8s/bucket-policy.example.json`](../../../k8s/bucket-policy.example.json), to this
bucket: its `AllowTrainer` statement cannot delete. The workflow pins
`cwic` to v1.44.0 and verifies the checksum of its release archive.

For `CWTRAIN_E2E_DEPLOYMENT_JSON`, use the `runtime`, `resources`, `autoscaling` and `traffic`
sections of `training/examples/miles-grpo-qwen3-4b/deployment.example.json`. Use engine `dynamo-vllm` and an
instance type that is available in the zone. The runner sets `runtime.version` from
`CWTRAIN_E2E_RUNTIME_VERSION`. The ModelExpress (MX) client that performs hot-loads is built into
the engine, so the `+crwv.N` suffix of the runtime version also selects the MX client version;
there is no separate client version to set. At least one replica is required. Where supported, set `min` and `max` to 1
for a smaller smoke test. The runner supplies `id`, `name`, `gatewayIds`, `model` and `hotLoad`,
with fresh resource IDs and the run's paths in the bucket. It keeps the other settings. It creates
no capacity claim, so GPU capacity and a compatible runtime must be available.

The runner uploads the initial model to `cookbook/cwtrain-e2e/<run-id>/models/base/`.
`HotLoader.publish()` uploads the checkpoint to
`cookbook/cwtrain-e2e/<run-id>/hotload/<name>-checkpoint/` in the same bucket. `<name>` is the
run's `cwtrain-e2e-...` resource name, so each run hot-loads its own identity. `<run-id>` is the
UTC minute the run started plus a random suffix, such as `20261009-1912-3f9a1c2e`, so the bucket
lists runs in start order. Download and upload times increase with model size, so choose a small
compatible model for hosted GitHub runners.

## Trigger a run

Post this exact comment on an open pull request:

```text
/hotload-e2e
```

- The commenter must have write, maintain or admin access to the repository. The workflow checks
  this through GitHub's API before it touches the environment's secrets.
- The workflow accepts only pull requests from a branch of this repository. It rejects pull
  requests from forks.
- The run checks out the pull request's head commit as resolved during authorization, so later
  pushes do not change the code under test. **Review that code before you trigger a run with
  secrets.**
- The comment is the only trigger. Pushes, schedules, pull request updates and manual dispatch do
  not start a run. New commits need a new `/hotload-e2e` comment. GitHub delivers `issue_comment`
  events only to workflows on the default branch.
- Runs are serialized. A newer queued run can replace an older pending one. An active run is not
  cancelled.

Results are under Actions → hotload E2E. The authorization job's summary records the pull request
number and the tested commit. After each authorized run, a separate job comments on the pull
request with the result, the tested commit and a link to the run. That job does not check out pull
request code or use the environment's secrets. A pass includes cleanup. A failure can come from
setup, the test, or cleanup. If GitHub stops the workflow before the reporting job runs, a
cancellation can go unreported.

## Cleanup and recovery

Before it creates anything, the runner writes a state file. The file holds fresh resource UUIDs,
the bucket name and a unique `cookbook/cwtrain-e2e/<run-id>/` prefix. So cleanup works even when a
create response is lost. The runner's `finally` block does the cleanup with
`cwtrain.infra.lifecycle`:

1. Delete the deployment, and wait until it is confirmed gone.
2. Delete the gateway, and wait until it is confirmed gone.
3. Delete every object and incomplete multipart upload under the run's exact prefix.

The cleanup does not touch the bucket, its policy, or anything outside the run prefix. It uses
normal deletes. In a versioned bucket, older versions follow the bucket's retention policy. A
failure on one resource does not stop the cleanup of the others. Incomplete cleanup fails the
test.

A separate `always()` step repeats the cleanup after success or failure, including a step timeout
or an ordinary cancellation. The job reserves time for this step. A lost runner, a forced
cancellation or a long API outage can still prevent cleanup. Gateway and deployment names start
with `cwtrain-e2e-`. The logs include their IDs and the storage prefix for manual recovery. Keep
the state file to retry cleanup. A new run needs a new state file name. Cleanup refuses a state
file without a valid run prefix. Remove the resources of such a file by hand.

## Running it locally

1. Install `cwic`.
2. From `libs/cwtrain`, run `uv pip install -e . 'huggingface_hub>=0.26,<2'`.
3. Set `CWTRAIN_E2E_BUCKET`, `CWTRAIN_E2E_DEPLOYMENT_JSON` and the API token
   (`COREWEAVE_API_TOKEN` or `CW_API_TOKEN`).
4. Configure an AWS profile with `credential_process = cwic auth accesskey api-token`.
5. Export `AWS_REGION=US-WEST-04A`. When you use another zone, also set `CWTRAIN_E2E_ZONE`.
6. From `libs/cwtrain`, run the test. Cleanup is safe to repeat, and fails if it cannot confirm
   that every resource is deleted.

```bash
python -m e2e.hotload --state /tmp/hotload-e2e.json
# Clean up only:
python -m e2e.hotload --state /tmp/hotload-e2e.json --cleanup-only
```

The normal `pytest` suite tests this runner with mocks. It never creates live resources.
