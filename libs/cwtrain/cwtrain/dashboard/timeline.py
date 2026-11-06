# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Local Vega feasibility export for the timeline already present in run history.

This module neither imports W&B nor starts a run. Feed it an *unsampled* history
iterator. Do not request every timeline key at once from older runs: keyed history
can omit rows that lack any requested key.

The overview uses fixed whole-run bins of interval-union coverage, not summed
durations. Dense detail reuses those explicitly labeled bins; once a brush contains
at most 1,500 spans, it displays the original intervals. This is a bounded-rendering
prototype, not a native W&B workspace-wide brush or a multiresolution query service.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

from cwtrain.integrations.tracking import LANES, TIMELINE_STEP

_DETAIL_LIMIT = 1500
_OVERVIEW_BINS = 256
_REPLICAS = ("hotload_first_replica", "hotload_peers")
_LABELS = {lane: lane.replace("_", " ").capitalize() for lane in LANES}
_LABELS.update(hotload="Hot-load", hotload_first_replica="First replica", hotload_peers="Peers")


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def extract_spans(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Preserve every valid span, including repeated phases and zero durations.

    ``row`` is the zero-based input position, not W&B's possibly repeated ``_step``.
    IDs are stable for the same ordered input. ``step`` is ``timeline/step`` or null;
    it is never inferred from a trainer's different native step axis. Absolute Unix
    timestamps are retained. Null, non-finite, reversed and nonnumeric spans are skipped.
    """
    spans = []
    for index, row in enumerate(rows):
        step = row.get(TIMELINE_STEP)
        if not _finite(step):
            step = None
        for phase in LANES:
            start, end = (row.get(f"timeline/{phase}/{edge}") for edge in ("start", "end"))
            if not _finite(start) or not _finite(end) or end < start:
                continue
            duration = end - start
            if not _finite(duration):
                continue
            spans.append(
                {
                    "id": f"{index}:{phase}",
                    "row": index,
                    "step": step,
                    "phase": phase,
                    "start": start,
                    "end": end,
                    "duration": duration,
                }
            )
    return spans


def _coverage(spans: list[dict[str, Any]], bins: int = _OVERVIEW_BINS) -> list[dict[str, Any]]:
    """Fixed-bin per-phase union coverage; zero events contribute counts, not time."""
    if not spans:
        return []
    origin = min(s["start"] for s in spans)
    duration = max(max(s["end"] for s in spans) - origin, 0.001)
    bin_width = duration / bins
    result = []
    for phase in LANES:
        intervals = sorted(
            (s["start"] - origin, s["end"] - origin) for s in spans if s["phase"] == phase
        )
        if not intervals:
            continue
        occupied, counts, zeros = [0.0] * bins, [0] * bins, [0] * bins
        union: list[list[float]] = []
        for start, end in intervals:
            first = min(bins - 1, max(0, int(start / bin_width)))
            if start == end:
                counts[first] += 1
                zeros[first] += 1
                continue
            last = min(bins - 1, max(first, math.ceil(end / bin_width) - 1))
            for i in range(first, last + 1):
                counts[i] += 1
            if union and start <= union[-1][1]:
                union[-1][1] = max(union[-1][1], end)
            else:
                union.append([start, end])
        for start, end in union:
            first = min(bins - 1, max(0, int(start / bin_width)))
            last = min(bins - 1, max(first, math.ceil(end / bin_width) - 1))
            for i in range(first, last + 1):
                occupied[i] += max(0, min(end, (i + 1) * bin_width) - max(start, i * bin_width))
        for i, count in enumerate(counts):
            if count:
                result.append(
                    {
                        "phase": phase,
                        "start_seconds": i * bin_width,
                        "end_seconds": (i + 1) * bin_width,
                        "coverage": min(1.0, occupied[i] / bin_width),
                        "occupied_seconds": occupied[i],
                        "count": count,
                        "zero_count": zeros[i],
                    }
                )
    return result


def _filter(expression: str) -> dict[str, str]:
    return {"type": "filter", "expr": expression}


def _mark(kind: str, data: str | None, update: dict[str, Any], **extra: Any) -> dict[str, Any]:
    mark = {"type": kind, "encode": {"update": update}, **extra}
    if data:
        mark["from"] = {"data": data}
    return mark


def timeline_spec() -> dict[str, Any]:
    """Return a standalone Vega v5 spec with an injectable ``spans`` data source.

    ``build_timeline`` also fills ``coverage`` for bounded overview rendering.
    Injecting only extracted spans works, but the overview then renders raw spans.
    There are no W&B-specific data-query assumptions in this portable spec.
    """
    phase_filter = "expandReplicas || indexof(replicaPhases, datum.phase) < 0"
    overlap = "datum.end_seconds >= detailDomain[0] && datum.start_seconds <= detailDomain[1]"
    color = {"scale": "color", "field": "phase"}
    span_tip = {
        "signal": "{'Phase': phaseLabels[datum.phase], 'Timeline step': datum.step, "
        "'History row': datum.row, "
        "'Start (local time)': timeFormat(toDate(datum.start * 1000), '%Y-%m-%d %H:%M:%S.%L %Z'), "
        "'End (local time)': timeFormat(toDate(datum.end * 1000), '%Y-%m-%d %H:%M:%S.%L %Z'), "
        "'Duration': datum.duration_label}"
    }
    coverage_tip = {
        "signal": "{'Phase': phaseLabels[datum.phase], "
        "'Full bin start (elapsed s)': datum.start_seconds, "
        "'Full bin end (elapsed s)': datum.end_seconds, 'Coverage (%)': datum.coverage * 100, "
        "'Union occupied (s)': datum.occupied_seconds, 'Contributing spans': datum.count, "
        "'Zero-duration events': datum.zero_count}"
    }
    spec = {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "description": "Elapsed-time overview and detail of original cwtrain phase spans.",
        "width": 900,
        "padding": {"left": 112, "right": 20, "top": 22, "bottom": 42},
        "autosize": {"type": "fit-x", "contains": "padding"},
        "background": "#ffffff",
        "config": {"axis": {"labelFontSize": 11, "titleFontSize": 12, "gridColor": "#e5e7eb"}},
        "signals": [
            {"name": "phaseOrder", "value": list(LANES)},
            {"name": "phaseLabels", "value": _LABELS},
            {"name": "replicaPhases", "value": list(_REPLICAS)},
            {
                "name": "expandReplicas",
                "value": False,
                "bind": {"input": "checkbox", "name": "Expand replicas "},
            },
            {
                "name": "stepFilter",
                "value": -1,
                "bind": {"input": "number", "name": "Focus timeline step (−1 = all) ", "min": -1},
                "on": [{"events": "@overview_hit:pointerdown", "update": "-1"}],
            },
            {
                "name": "selectedStep",
                "update": "stepFilter === '' || stepFilter === null ? -1 : toNumber(stepFilter)",
            },
            {
                "name": "origin",
                "update": "length(data('extent')) && "
                "isValid(data('extent')[0].start) ? data('extent')[0].start : 0",
            },
            {
                "name": "fullDuration",
                "update": "length(data('extent')) && "
                "isValid(data('extent')[0].end) ? max(0.001, data('extent')[0].end-origin) : 1",
            },
            {"name": "overviewHeight", "update": "max(1, length(data('phases'))) * 9"},
            {"name": "detailTop", "update": "overviewHeight + 76"},
            {"name": "detailHeight", "update": "max(1, length(data('phases'))) * 28"},
            {"name": "height", "update": "detailTop + detailHeight"},
            {
                "name": "brush",
                "value": [0, 0],
                "on": [
                    {
                        "events": "@overview_hit:pointerdown",
                        "update": "[clamp(x(), 0, width), clamp(x(), 0, width)]",
                    },
                    {
                        "events": "[@overview_hit:pointerdown, window:pointerup] "
                        "> window:pointermove!",
                        "update": "[brush[0], clamp(x(), 0, width)]",
                    },
                    {"events": "@overview_hit:dblclick", "update": "[0, 0]"},
                    {"events": {"signal": "width"}, "update": "[0, 0]"},
                ],
            },
            {
                "name": "detailDomain",
                "update": "selectedStep >= 0 && length(data('selected_step')) && "
                "isValid(data('selected_step')[0].start) ? [data('selected_step')[0].start, "
                "max(data('selected_step')[0].start + 0.001, data('selected_step')[0].end)] : "
                "abs(brush[1]-brush[0]) < 2 ? "
                "[0, fullDuration] : [invert('overview_x', min(brush[0], brush[1])), "
                "invert('overview_x', max(brush[0], brush[1]))]",
            },
            {
                "name": "dense",
                "update": f"selectedStep < 0 && length(data('in_window')) > {_DETAIL_LIMIT} "
                "&& length(data('coverage')) > 0",
            },
        ],
        "data": [
            {"name": "spans", "values": []},
            {"name": "coverage", "values": []},
            {
                "name": "extent",
                "source": "spans",
                "transform": [
                    {
                        "type": "aggregate",
                        "fields": ["start", "end"],
                        "ops": ["min", "max"],
                        "as": ["start", "end"],
                    }
                ],
            },
            {
                "name": "phases",
                "source": "spans",
                "transform": [
                    _filter(phase_filter),
                    {"type": "aggregate", "groupby": ["phase"]},
                    {"type": "formula", "expr": "indexof(phaseOrder, datum.phase)", "as": "order"},
                    {"type": "collect", "sort": {"field": "order"}},
                ],
            },
            {
                "name": "relative",
                "source": "spans",
                "transform": [
                    _filter(phase_filter),
                    {"type": "formula", "expr": "datum.start-origin", "as": "start_seconds"},
                    {"type": "formula", "expr": "datum.end-origin", "as": "end_seconds"},
                    {
                        "type": "formula",
                        "expr": "round((datum.end-datum.start)*1000)/1000",
                        "as": "duration_seconds_rounded",
                    },
                    {
                        "type": "formula",
                        "expr": (
                            "datum.end>datum.start && datum.duration_seconds_rounded<0.001 "
                            "? '<0.001 s' : "
                            "(datum.duration_seconds_rounded>=86400?"
                            "floor(datum.duration_seconds_rounded/86400)+' d ':'')+"
                            "(datum.duration_seconds_rounded>=3600?"
                            "floor(datum.duration_seconds_rounded%86400/3600)+' h ':'')+"
                            "(datum.duration_seconds_rounded>=60?"
                            "floor(datum.duration_seconds_rounded%3600/60)+' min ':'')+"
                            "format(datum.duration_seconds_rounded%60,'.3~f')+' s'"
                        ),
                        "as": "duration_label",
                    },
                ],
            },
            {
                "name": "overview_spans",
                "source": "relative",
                "transform": [_filter("!length(data('coverage'))")],
            },
            {
                "name": "overview_coverage",
                "source": "coverage",
                "transform": [_filter(phase_filter)],
            },
            {
                "name": "selected_step",
                "source": "relative",
                "transform": [
                    _filter("selectedStep >= 0 && datum.step === selectedStep"),
                    {
                        "type": "aggregate",
                        "fields": ["start_seconds", "end_seconds"],
                        "ops": ["min", "max"],
                        "as": ["start", "end"],
                    },
                ],
            },
            {
                "name": "in_window",
                "source": "relative",
                "transform": [
                    _filter(overlap),
                    _filter("selectedStep < 0 || datum.step === selectedStep"),
                ],
            },
            {"name": "detail_spans", "source": "in_window", "transform": [_filter("!dense")]},
            {
                "name": "detail_coverage",
                "source": "overview_coverage",
                "transform": [_filter("dense"), _filter(overlap)],
            },
            {
                "name": "coverage_zero",
                "source": "detail_coverage",
                "transform": [_filter("datum.coverage === 0 && datum.zero_count > 0")],
            },
            {
                "name": "bars",
                "source": "detail_spans",
                "transform": [
                    _filter(
                        "datum.duration > 0 && scale('detail_x', datum.end_seconds) - "
                        "scale('detail_x', datum.start_seconds) >= 2"
                    )
                ],
            },
            {
                "name": "small",
                "source": "detail_spans",
                "transform": [
                    _filter(
                        "datum.duration > 0 && scale('detail_x', datum.end_seconds) - "
                        "scale('detail_x', datum.start_seconds) < 2"
                    )
                ],
            },
            {
                "name": "zero",
                "source": "detail_spans",
                "transform": [_filter("datum.duration === 0")],
            },
        ],
        "scales": [
            {
                "name": "overview_x",
                "type": "linear",
                "domain": {"signal": "[0,fullDuration]"},
                "range": "width",
                "zero": False,
            },
            {
                "name": "detail_x",
                "type": "linear",
                "domain": {"signal": "detailDomain"},
                "range": "width",
                "zero": False,
            },
            {
                "name": "overview_y",
                "type": "band",
                "domain": {"data": "phases", "field": "phase"},
                "range": {"signal": "[0,overviewHeight]"},
                "padding": 0.15,
            },
            {
                "name": "detail_y",
                "type": "band",
                "domain": {"data": "phases", "field": "phase"},
                "range": {"signal": "[detailTop,height]"},
                "padding": 0.3,
            },
            {
                "name": "color",
                "type": "ordinal",
                "domain": list(LANES),
                "range": [
                    "#20845c",
                    "#9463b5",
                    "#287ca2",
                    "#9463b5",
                    "#9463b5",
                    "#9463b5",
                    "#287ca2",
                    "#9463b5",
                    "#287ca2",
                    "#b26721",
                    "#b26721",
                    "#b26721",
                ],
            },
        ],
        "axes": [
            {
                "orient": "bottom",
                "scale": "overview_x",
                "offset": {"signal": "overviewHeight-height"},
                "tickCount": {"signal": "width < 400 ? 3 : 6"},
                "title": {
                    "signal": "width < 400 ? 'Whole run (s)' : 'Whole run · elapsed seconds'"
                },
            },
            {
                "orient": "bottom",
                "scale": "detail_x",
                "title": {
                    "signal": "width < 400 ? 'Elapsed (s)' : 'Elapsed seconds from first span'"
                },
                "tickCount": {"signal": "width < 400 ? 3 : 6"},
            },
            {
                "orient": "left",
                "scale": "detail_y",
                "ticks": False,
                "domain": False,
                "encode": {"labels": {"update": {"text": {"signal": "phaseLabels[datum.value]"}}}},
            },
        ],
        "marks": [],
    }
    marks = spec["marks"]
    marks.append(
        _mark(
            "rect",
            "overview_spans",
            {
                "x": {"scale": "overview_x", "field": "start_seconds"},
                "x2": {"scale": "overview_x", "field": "end_seconds"},
                "y": {"scale": "overview_y", "field": "phase"},
                "height": {"scale": "overview_y", "band": 1},
                "fill": color,
            },
            interactive=False,
        )
    )
    marks.append(
        _mark(
            "rect",
            "overview_coverage",
            {
                "x": {"scale": "overview_x", "field": "start_seconds"},
                "x2": {"scale": "overview_x", "field": "end_seconds"},
                "y": {"scale": "overview_y", "field": "phase"},
                "height": {"scale": "overview_y", "band": 1},
                "fill": color,
                "fillOpacity": {"field": "coverage"},
            },
            interactive=False,
        )
    )
    marks.append(
        _mark(
            "rect",
            None,
            {
                "x": {"signal": "min(brush[0],brush[1])"},
                "x2": {"signal": "max(brush[0],brush[1])"},
                "y": {"value": 0},
                "height": {"signal": "overviewHeight"},
                "fill": {"value": "#374151"},
                "fillOpacity": {"value": 0.15},
            },
            interactive=False,
        )
    )
    marks.append(
        _mark(
            "rect",
            None,
            {
                "x": {"value": 0},
                "y": {"value": 0},
                "width": {"signal": "width"},
                "height": {"signal": "overviewHeight"},
                "fill": {"value": "transparent"},
                "cursor": {"value": "crosshair"},
            },
            name="overview_hit",
        )
    )
    marks.append(
        _mark(
            "text",
            None,
            {
                "x": {"value": 0},
                "y": {"signal": "detailTop-14"},
                "fill": {"value": "#4b5563"},
                "fontSize": {"value": 11},
                "text": {
                    "signal": "!length(data('spans')) ? 'No valid timeline spans' : "
                    "width < 400 ? (dense ? 'Coverage · zoom in' : "
                    "length(data('detail_spans')) + ' spans') : "
                    "dense ? 'Fixed-bin union coverage · brush closer for exact spans' : "
                    "length(data('detail_spans')) + "
                    "' spans · diamonds <2 px · ticks = zero duration'"
                },
            },
        )
    )
    marks.append(
        _mark(
            "rect",
            "detail_coverage",
            {
                "x": {"signal": "max(0,scale('detail_x',datum.start_seconds))"},
                "x2": {"signal": "min(width,scale('detail_x',datum.end_seconds))"},
                "y": {"scale": "detail_y", "field": "phase"},
                "height": {"scale": "detail_y", "band": 1},
                "fill": color,
                "fillOpacity": {"field": "coverage"},
                "tooltip": coverage_tip,
            },
        )
    )
    marks.append(
        _mark(
            "rule",
            "coverage_zero",
            {
                "x": {"signal": "scale('detail_x',(datum.start_seconds+datum.end_seconds)/2)"},
                "y": {"scale": "detail_y", "field": "phase", "band": 0.25},
                "y2": {"scale": "detail_y", "field": "phase", "band": 0.75},
                "stroke": color,
                "strokeWidth": {"value": 1},
                "tooltip": coverage_tip,
            },
            clip=True,
        )
    )
    marks.append(
        _mark(
            "rect",
            "bars",
            {
                "x": {"signal": "max(0,scale('detail_x',datum.start_seconds))"},
                "x2": {"signal": "min(width,scale('detail_x',datum.end_seconds))"},
                "y": {"scale": "detail_y", "field": "phase"},
                "height": {"scale": "detail_y", "band": 1},
                "fill": color,
                "tooltip": span_tip,
            },
        )
    )
    marks.append(
        _mark(
            "symbol",
            "small",
            {
                "x": {"signal": "scale('detail_x',(datum.start_seconds+datum.end_seconds)/2)"},
                "y": {"scale": "detail_y", "field": "phase", "band": 0.5},
                "shape": {"value": "diamond"},
                "size": {"value": 25},
                "fill": color,
                "tooltip": span_tip,
            },
            clip=True,
        )
    )
    marks.append(
        _mark(
            "rule",
            "zero",
            {
                "x": {"scale": "detail_x", "field": "start_seconds"},
                "y": {"scale": "detail_y", "field": "phase"},
                "y2": {"scale": "detail_y", "field": "phase", "band": 1},
                "stroke": color,
                "strokeWidth": {"value": 2},
                "tooltip": span_tip,
            },
        )
    )
    return spec


def build_timeline(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the portable Vega spec, retaining raw spans and bounded coverage data."""
    spans = extract_spans(rows)
    spec = timeline_spec()
    spec["data"][0]["values"] = spans
    spec["data"][1]["values"] = _coverage(spans)
    return spec


def export_html(rows: Iterable[Mapping[str, Any]], title: str = "cwtrain timeline") -> str:
    """Export inline data and a pinned CDN Vega runtime; no W&B credentials or API calls.

    The file needs internet access for its JavaScript runtime, not for run data.
    Untrusted labels are HTML-escaped; JSON is inert and cannot terminate its script tag.
    """
    spec_json = json.dumps(build_timeline(rows), allow_nan=False, separators=(",", ":"))
    spec_json = spec_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    safe_title = html.escape(title, quote=True)
    return _HTML.replace("__SPEC__", spec_json).replace("__TITLE__", safe_title)


_HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>__TITLE__</title>
<style>
body{margin:0;background:#fff;color:#1f2937;font:14px system-ui,sans-serif;color-scheme:light}
main{max-width:1200px;margin:auto;padding:20px}h1{font-size:20px;font-weight:500}
p{color:#4b5563;line-height:1.5}#chart{width:100%}canvas{display:block;max-width:100%}
.vega-bindings{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}
.vega-bind{display:flex;align-items:center;gap:8px}input[type=number]{width:90px}
#tooltip{position:fixed;pointer-events:none;z-index:2;background:#fff;border:1px solid #d1d5db;
padding:10px;white-space:pre-wrap;font-size:12px;max-width:min(380px,85vw);overflow-wrap:anywhere}
</style><main><h1>__TITLE__</h1>
<p>Local timeline feasibility export. Drag across the overview to inspect a time range;
double-click it to reset. Coverage counts overlapping intervals once within each phase.
Replica lanes are nested measurements, not additive time.</p>
<div id="chart"></div><p id="error" role="alert"></p><div id="tooltip" hidden></div></main>
<script type="application/json" id="timeline-spec">__SPEC__</script>
<script src="https://cdn.jsdelivr.net/npm/vega@5.33.0/build/vega.min.js"></script>
<script>
(async()=>{
  const chart=document.getElementById('chart'),tip=document.getElementById('tooltip');
  try {
    const spec=JSON.parse(document.getElementById('timeline-spec').textContent);
    const view=new vega.View(vega.parse(spec),{renderer:'canvas',hover:true}).initialize(chart);
    view.tooltip((handler,event,item,value)=>{
      if(value==null){tip.hidden=true;return;}
      tip.textContent=typeof value==='object'
        ?Object.entries(value).map(([k,v])=>k+': '+v).join('\\n'):String(value);
      tip.hidden=false;
      tip.style.left=Math.max(4,Math.min(event.clientX+12,innerWidth-tip.offsetWidth-8))+'px';
      tip.style.top=Math.max(4,Math.min(event.clientY+12,innerHeight-tip.offsetHeight-8))+'px';
    });
    await view.width(chart.clientWidth).runAsync();
    let lastWidth=chart.clientWidth;
    new ResizeObserver(()=>{if(chart.clientWidth!==lastWidth){lastWidth=chart.clientWidth;
      view.width(lastWidth).resize().runAsync();}}).observe(chart);
  } catch(error){
    document.getElementById('error').textContent='Unable to render timeline: '+error.message;
  }
})();
</script></html>"""
