# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

from __future__ import annotations

import json
import math
import re

import pytest

from cwtrain.dashboard.timeline import (
    _coverage,
    build_timeline,
    export_html,
    extract_spans,
    timeline_spec,
)


def row(phase, start, end, step=1):
    return {f"timeline/{phase}/start": start, f"timeline/{phase}/end": end, "timeline/step": step}


def test_preserves_repeated_updates_zeros_and_absolute_timestamps():
    rows = [
        row("update", 1700000000, 1700000001),
        row("update", 1700000002, 1700000003),
        row("hotload_peers", 1700000004, 1700000004),
    ]
    for data in rows:
        data.update({"_step": 40, "train/step": 9, "unrelated_secret": "not exported"})
    spans = extract_spans(iter(rows))
    assert [span["id"] for span in spans] == ["0:update", "1:update", "2:hotload_peers"]
    assert [span["row"] for span in spans] == [0, 1, 2]
    assert [span["step"] for span in spans] == [1, 1, 1]
    assert [span["duration"] for span in spans] == [1, 1, 0]
    assert spans[0]["start"] == 1700000000
    assert "unrelated_secret" not in json.dumps(spans)
    assert rows[0]["unrelated_secret"] == "not exported"


@pytest.mark.parametrize(
    "start,end",
    [
        (None, 2),
        (1, None),
        (math.nan, 2),
        (1, math.inf),
        (-math.inf, 2),
        (2, 1),
        ("1", 2),
        (True, 2),
        (1, False),
        (10**1000, 10**1000),
        (-1e308, 1e308),
    ],
)
def test_discards_invalid_spans_without_converting_missing_values_to_zero(start, end):
    assert extract_spans([row("update", start, end)]) == []


def test_missing_phase_pairs_and_unknown_phases_are_ignored():
    assert extract_spans([{"timeline/update/start": 1}, row("unrecognized", 1, 2)]) == []


@pytest.mark.parametrize("step", [None, math.nan, math.inf, True, "7"])
def test_invalid_timeline_step_does_not_fall_back_to_native_trainer_step(step):
    data = {**row("upload", 1, 2, step), "train/step": 7, "rollout/step": 3}
    assert extract_spans([data])[0]["step"] is None


def test_union_coverage_counts_overlapping_spans_once_and_keeps_zero_events():
    spans = extract_spans(
        [row("update", 100, 104), row("update", 102, 106), row("update", 103, 103)]
    )
    bins = _coverage(spans, bins=3)
    assert [b["coverage"] for b in bins] == [1, 1, 1]
    assert sum(b["occupied_seconds"] for b in bins) == 6
    assert [b["count"] for b in bins] == [1, 3, 1]
    assert [b["zero_count"] for b in bins] == [0, 1, 0]
    assert [(b["start_seconds"], b["end_seconds"]) for b in bins] == [(0, 2), (2, 4), (4, 6)]


def test_coverage_preserves_gaps_and_nested_phase_measurements():
    spans = extract_spans(
        [row("hotload", 0, 1), row("hotload", 3, 4), row("hotload_first_replica", 0, 1)]
    )
    bins = _coverage(spans, bins=2)
    parent = [b for b in bins if b["phase"] == "hotload"]
    child = [b for b in bins if b["phase"] == "hotload_first_replica"]
    assert [b["coverage"] for b in parent] == [0.5, 0.5]
    assert sum(b["occupied_seconds"] for b in parent) == 2
    assert sum(b["occupied_seconds"] for b in child) == 1


def test_zero_only_run_has_counts_without_fabricated_duration():
    bins = _coverage(extract_spans([row("hotload_peers", 10, 10)]))
    assert len(bins) == 1
    assert bins[0]["coverage"] == 0
    assert bins[0]["zero_count"] == 1


def test_built_spec_keeps_raw_data_and_uses_relative_chart_fields():
    spec = build_timeline([row("update", 1700000000, 1700000001)])
    sources = {source["name"]: source for source in spec["data"]}
    assert sources["spans"]["values"][0]["start"] == 1700000000
    assert sources["coverage"]["values"][0]["start_seconds"] == 0
    assert any(t.get("expr") == "datum.start-origin" for t in sources["relative"]["transform"])
    assert spec["$schema"].endswith("/vega/v5.json")
    assert "NaN" not in json.dumps(spec, allow_nan=False)


def test_empty_spec_is_portable_and_each_call_is_independent():
    first, second = timeline_spec(), timeline_spec()
    first["data"][0]["values"].append({"test": True})
    assert second["data"][0]["values"] == []
    assert build_timeline([])["data"][0]["values"] == []
    assert all("url" not in source for source in second["data"])


def test_html_escapes_untrusted_title_and_contains_only_relevant_inline_data():
    title = '</title><script>alert("x")</script> & __SPEC__'
    rendered = export_html([{**row("update", 1, 2), "api_key": "secret"}], title=title)
    assert '<script>alert("x")</script>' not in rendered
    assert "&lt;/title&gt;" in rendered
    assert "&amp; __SPEC__" in rendered
    assert "secret" not in rendered
    assert "vega@5.33.0/build/vega.min.js" in rendered
    payload = re.search(
        r'<script type="application/json" id="timeline-spec">(.*?)</script>', rendered, re.S
    )
    assert payload is not None
    assert json.loads(payload.group(1))["data"][0]["values"][0]["phase"] == "update"
    assert "renderer:'canvas'" in rendered


def test_coverage_render_size_is_bounded_by_bins_not_input_row_count():
    # Tiny fixture, exercising the same count bound without a large synthetic run in the repo.
    spans = extract_spans(row("update", i, i + 0.5, i) for i in range(20))
    bins = _coverage(spans, bins=4)
    assert len(spans) == 20 and len(bins) <= 4
    assert all(0 <= b["coverage"] <= 1 for b in bins)
