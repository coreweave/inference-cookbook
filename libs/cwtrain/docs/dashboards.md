# Post-training dashboards in W&B

`cwtrain dashboard setup` creates a managed W&B workspace view for a cookbook framework.
The panels cover training, rollouts, hot-load, available validation metrics, and trainer system
telemetry. `cwtrain dashboard timeline` separately exports one run's phase timeline to local HTML.

## Install and authenticate

The dashboard commands are planned for the next release; they are not included in
`cwtrain 1.0.2`. Once released, install or upgrade the standard package:

```bash
python -m pip install --upgrade cwtrain
```

The package includes the dashboard CLI and its W&B dependencies.

Authenticate with `wandb login` or supply `WANDB_API_KEY` in your environment. Set `WANDB_BASE_URL`
when using your organization's W&B deployment instead of the default cloud endpoint. These
commands use W&B credentials; they do not require `COREWEAVE_API_TOKEN` or object-storage keys.

## Preview and create a view

Preview the specification without a W&B connection:

```bash
cwtrain dashboard setup --entity ENTITY --project PROJECT --framework miles --dry-run
```

The dry run is offline and lists the full recipe catalogue. It does not discover which metrics
exist in the project or create a view. Supported frameworks are `miles`, `slime`, `nemo-rl`, and
`axolotl`.

Create the view:

```bash
cwtrain dashboard setup --entity ENTITY --project PROJECT --framework miles
```

Setup checks the newest 200 runs with job type `train` and selects up to ten whose recorded tags
include the selected framework. It reads tags directly because newly tagged runs can be missing
from W&B's tag index. Metric keys found in those runs determine which panels are available. Keep
the recipes' default framework tags and job type: runs with those values overridden will not
match this selection. The command prints the URL of the resulting managed view, such as
**Post-training — Miles**.

The native filter includes the framework tag plus explicit IDs only for discovered runs missing
from W&B's tag index. Ordinary new tagged runs are included automatically. If a new run needs an
explicit ID not already installed in the view, setup stops instead of silently leaving it out.

Run setup for each framework you want to inspect. The views retain each framework's metric names
and step axes; they do not combine unlike losses or rewards into a cross-framework ranking.

### Rerun without overwriting personal changes

Setup records the view URL and a fingerprint in a local state file. Its default location is
`~/.local/state/cwtrain/dashboards.json`; select another location with `--state-file`:

```bash
cwtrain dashboard setup --entity ENTITY --project PROJECT --framework miles \
  --state-file ./dashboard-state.json
```

Reuse the same state file for subsequent runs. An unchanged definition reuses the saved URL
without writing to W&B. Setup stops if it detects edits to the recorded view or if the generated
definition changes. Keep the state file when moving this workflow to another machine.

The supported W&B workspace SDK does not expose or preserve all native UI settings. Setup
therefore never replaces an existing saved view, even when the SDK-visible fingerprint is
unchanged. Edit an existing view directly in W&B. To deliberately create another view, use a
different state file and retain the original view and state.

If a save fails with an unknown outcome, setup keeps a pending record and stops subsequent
attempts. The server may already have created the view even if its response was lost.
Inspect W&B and the recorded state before recovering; keep the state file and do not switch to a
different one to retry, since that can create a duplicate view.

## What the panels mean

The versioned [metric catalogue](../cwtrain/dashboard/catalog.py) defines the panel titles, exact
metric keys, native axes, and units. Its current specification version is 1. Values are not
rescaled: bytes remain bytes, fractions remain fractions, and metrics already expressed as
percentages remain percentages.

Native charts use W&B's full-fidelity visualization with smoothing and outlier exclusion disabled.
The default view shows up to five runs, ordered newest first; each run remains a separate series.

| Framework | Training axis | Rollout and hot-load axis | Evaluation |
| --- | --- | --- | --- |
| miles / slime | `train/step` for optimizer metrics; `rollout/step` for reward | `rollout/step`; training-step mismatch diagnostics retain `train/step` | No evaluation series assumed |
| NeMo-RL | Explicit logged `_step` | `_step`; hot-load metrics use `train/hotload/` | `validation/accuracy` and `validation/avg_length`, when logged |
| Axolotl | `train/global_step` | Publish metrics at the final global step; no rollout series | No evaluation series assumed |

Some distinctions matter when reading a panel:

- **Reward variation:** miles' `rollout/zero_std/*_percentage` keys contain fractions despite their
  names. Slime logs counts without a recorded total-group denominator. NeMo's
  `train/baseline_reward/pct_*` keys already contain percentages of baseline entries; they are not
  the same group statistic.
- **Policy agreement:** update-policy KL and trainer/generation mismatch answer different
  questions. NeMo's `train/gen_kl_error` and `train/policy_kl_error` describe the latter. The signed
  `train/ppo_kl` estimate can be negative. Miles' sometimes-uncomputed, zero-valued
  `train/entropy_loss` is omitted rather than presented as measured entropy.
- **Reference-policy KL:** Miles' Training section puts `train/ppo_kl` first and, when logged,
  `train/kl_loss` next. The latter is the reference-policy estimate before multiplication by
  `kl_loss_coef`, not the weighted penalty. Miles computes and logs it only with `use_kl_loss`.
  The current cookbook launcher leaves that flag disabled and sets `kl_loss_coef=0`, so an
  absent reference-KL curve is an unavailable measurement, not zero. Dashboard setup does not
  change training flags or reconstruct missing reference log-probabilities.
- **Request latency:** the p50, p95, and maximum time successful requests through the whole
  response, including deployment queue time. They do not measure time to first token or all retry
  and backoff time. Retry counts are not failure rates.
- **Hot-load:** activation wall time includes waiting enabled by the recipe. Replica install
  times are nested measurements, not extra durations to add to that wall time.
  `snapshot_bytes` describes the local full-checkpoint or compressed-delta artifact, not exact
  network traffic. Missing tensor counts on full snapshots are not zeros.
- **GPU telemetry:** device indices are local to a telemetry source. Multiple W&B writers may
  monitor the same devices; their metrics stay separate. System charts use event timestamps,
  which can be aligned with the timeline. They describe trainer-side telemetry, not remote
  inference-deployment GPUs.
- **CPU and RAM:** reported process CPU is not host-wide utilization. W&B SDK 0.28/0.30 CPU values
  are process-lifetime averages, not instantaneous samples. RAM utilization uses container/cgroup
  memory when available, otherwise host memory. GPU memory is recorded separately in bytes.

The setup can only use what the runs logged. No panel fabricates missing observations, reward
histograms, per-sample reward/length correlations, completion-error rates, time to first token, or
inter-token latency. NeMo generation-throughput fields that are unusable for this backend are
excluded. Separate `job_type=eval` runs have their own schema; this version's Evaluation section
shows NeMo's in-training validation.

## Export a run timeline

```bash
cwtrain dashboard timeline --run ENTITY/PROJECT/RUN_ID --output timeline.html
```

The command scans the run's unsampled history and extracts `timeline/<phase>/start` and `/end`
spans with `timeline/step`. Open `timeline.html` in a browser. Drag across the overview to inspect
a time interval; double-click to return to the full run. The file embeds the recorded spans and
loads a pinned Vega runtime from a CDN, so rendering needs internet access for that runtime. It
does not embed W&B credentials or fetch run data in the browser.

The overview shows per-phase interval-union coverage in fixed bins. A dense selection uses those
same labeled bins; zooming to at most 1,500 spans reveals the original intervals. Repeated update
spans remain separate. Short positive spans use markers when narrower than a pixel-scale bar;
zero-duration spans use ticks. Hover to inspect exact timestamps and durations.

Replica lanes are nested inside hot-load. Gaps between recorded phases remain unexplained time,
not inferred GPU idleness. The timeline's origin is the first recorded phase, which need not
coincide with the start of the training job or the W&B run.

This is a local timeline prototype. Its overview selection controls its own detail view; it does
not create a global time brush across native W&B workspace panels.
