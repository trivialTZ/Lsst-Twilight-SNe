"""Paths, logging, and window summary helpers.

Responsibility
--------------
- Construct output paths for CSVs.
- Format windows for logs (UTC and local variants).
- Compute and log per-window/night summaries with exact CSV columns.
"""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

from ..astro_utils import _local_timezone_from_location
from ..config import PlannerConfig
from ..priority import PriorityTracker  # only for filter groups
from .selection import _policy_filters_mid
from ..simlib_writer import SimlibWriter
import gzip


def _fmt_window(start: datetime | None, end: datetime | None, tz_local: tzinfo) -> str:
    if start is None or end is None:
        return "na"
    start_utc = pd.Timestamp(start).tz_convert("UTC")
    end_utc = pd.Timestamp(end).tz_convert("UTC")
    start_local = start_utc.tz_convert(tz_local)
    end_local = end_utc.tz_convert(tz_local)
    offset_td = start_local.utcoffset() or timedelta(0)
    sign = "+" if offset_td >= timedelta(0) else "-"
    offset_td = abs(offset_td)
    hours = int(offset_td.total_seconds() // 3600)
    minutes = int((offset_td.total_seconds() % 3600) // 60)
    offset_str = f"UTC{sign}{hours:02d}:{minutes:02d}"
    return (
        f"{start_utc.isoformat()} \u2192 {end_utc.isoformat()} (local "
        f"{start_local.strftime('%H:%M')} \u2192 {end_local.strftime('%H:%M')} {offset_str})"
    )


def _fmt_window_local(start: datetime | None, end: datetime | None, tz_local: tzinfo) -> str:
    if start is None or end is None:
        return "na"
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    start_local = start_ts.tz_convert(tz_local)
    end_local = end_ts.tz_convert(tz_local)
    offset_td = start_local.utcoffset() or timedelta(0)
    sign = "+" if offset_td >= timedelta(0) else "-"
    offset_td = abs(offset_td)
    hours = int(offset_td.total_seconds() // 3600)
    minutes = int((offset_td.total_seconds() % 3600) // 60)
    offset_str = f"UTC{sign}{hours:02d}:{minutes:02d}"
    return (
        f"local {start_local.strftime('%H:%M')} \u2192 {end_local.strftime('%H:%M')} {offset_str}"
    )


def _mid_sun_alt_of_window(window: dict, site: EarthLocation) -> float:
    mid = window["start"] + (window["end"] - window["start"]) / 2
    return float(
        get_sun(Time(mid))
        .transform_to(AltAz(location=site, obstime=Time(mid)))
        .alt.to(u.deg)
        .value
    )


def _path_plan(outdir: str | Path, start: datetime, end: datetime, run_label: str | None = None) -> Path:
    label = f"{run_label}_" if run_label else ""
    return Path(outdir) / f"lsst_twilight_plan_{label}{start.isoformat()}_to_{end.isoformat()}.csv"


def _path_summary(outdir: str | Path, start: datetime, end: datetime, run_label: str | None = None) -> Path:
    label = f"{run_label}_" if run_label else ""
    return Path(outdir) / f"lsst_twilight_summary_{label}{start.isoformat()}_to_{end.isoformat()}.csv"


def _path_sequence(outdir: str | Path, start: datetime, end: datetime, run_label: str | None = None) -> Path:
    label = f"{run_label}_" if run_label else ""
    return Path(outdir) / f"lsst_twilight_sequence_true_{label}{start.isoformat()}_to_{end.isoformat()}.csv"


def _log_day_status(
    day_iso: str,
    eligible: int,
    visible: int,
    planned_total: int,
    evening_start: datetime | None,
    evening_end: datetime | None,
    morning_start: datetime | None,
    morning_end: datetime | None,
    tz_local: tzinfo,
    verbose: bool,
    window_usage: dict[str, dict] | None = None,
) -> None:
    if not verbose:
        return
    print(f"{day_iso}: eligible={eligible} visible={visible} planned_total={planned_total}")
    usage = window_usage or {}

    def _print_window(label: str, start: datetime | None, end: datetime | None) -> None:
        window_str = _fmt_window_local(start, end, tz_local)
        print(f"  {label}_twilight: {window_str}")
        metrics = usage.get(label)
        if not metrics:
            return
        parts: list[str] = []
        window_use = metrics.get("window_use_pct")
        if window_use is not None:
            parts.append(f"Time use: {window_use:.1f}%")
        observing = metrics.get("observing_s")
        if observing is not None:
            parts.append(f"Observing time: {observing:.1f}s")
        filter_change = metrics.get("filter_change_s")
        if filter_change is not None:
            parts.append(f"Filter change time: {filter_change:.1f}s")
        dp_time = metrics.get("dp_time_s")
        if dp_time is not None:
            parts.append(f"DP batch time used: {dp_time:.1f}s")
        backfill_time = metrics.get("backfill_time_s")
        if backfill_time is not None:
            parts.append(f"backfill time used: {backfill_time:.1f}s")
        filters_used = metrics.get("filters_used")
        if filters_used:
            parts.append(f"Filters used: {filters_used}")
        if parts:
            print("    " + ", ".join(parts))

        dbg = metrics.get("debug") if isinstance(metrics, dict) else None
        if not dbg:
            return
        pair_counts = dbg.get("pair_counts_by_filter", {})
        if pair_counts:
            total_pairs = sum(int(v) for v in pair_counts.values())
            pairs_str = ", ".join(f"{f}:{int(n)}" for f, n in sorted(pair_counts.items()))
            print(f"    pairs: total={total_pairs} ({pairs_str})")
        dp_plan = dbg.get("dp_plan", [])
        if dp_plan:
            plan_str = ", ".join(f"{f}×{int(c)}" for f, c in dp_plan)
            print(f"    DP plan: {plan_str}")
        dp_exec = dbg.get("dp_scheduled_by_filter", {})
        if dp_exec:
            dp_exec_str = ", ".join(f"{f}:{int(n)}" for f, n in sorted(dp_exec.items()))
            print(f"    DP executed: {sum(int(v) for v in dp_exec.values())} ({dp_exec_str})")
        dp_rej = dbg.get("dp_rejected_by_reason", {})
        if dp_rej:
            rej_str = ", ".join(f"{k}:{int(v)}" for k, v in sorted(dp_rej.items()))
            print(f"    DP rejects: {rej_str}")
        bf_order = dbg.get("backfill_order", [])
        if bf_order:
            print(f"    Backfill order: {', '.join(bf_order)}")
        bf_exec = dbg.get("backfill_scheduled_by_filter", {})
        if bf_exec:
            bf_exec_str = ", ".join(f"{f}:{int(n)}" for f, n in sorted(bf_exec.items()))
            print(f"    Backfill executed: {sum(int(v) for v in bf_exec.values())} ({bf_exec_str})")
        rep_exec = dbg.get("repeat_scheduled_by_filter", {})
        if rep_exec:
            rep_exec_str = ", ".join(f"{f}:{int(n)}" for f, n in sorted(rep_exec.items()))
            print(f"    Repeats executed: {sum(int(v) for v in rep_exec.values())} ({rep_exec_str})")
        relax_exec = dbg.get("relax_scheduled_by_filter", {})
        if relax_exec:
            relax_exec_str = ", ".join(f"{f}:{int(n)}" for f, n in sorted(relax_exec.items()))
            print(f"    Relaxed backfill executed: {sum(int(v) for v in relax_exec.values())} ({relax_exec_str})")

    _print_window("evening", evening_start, evening_end)
    _print_window("morning", morning_start, morning_end)


def _build_window_summary_row(
    day_iso,
    window_label,
    win,
    idx_w,
    ws_summary,
    cap_s,
    cap_source,
    cfg: PlannerConfig,
    site: EarthLocation,
    pernight_rows_for_window,
) -> dict:
    tz_local = _local_timezone_from_location(site)
    window_str = _fmt_window(win["start"], win["end"], tz_local)
    start_utc, _, after_arrow = window_str.partition(" \u2192 ")
    end_utc, _, _ = after_arrow.partition(" ")
    dur_s = (win["end"] - win["start"]).total_seconds()
    mid = win["start"] + (win["end"] - win["start"]) / 2
    mid_utc = pd.Timestamp(mid).tz_convert("UTC").isoformat()
    sun_alt_mid = _mid_sun_alt_of_window(win, site)
    policy_filters_mid_csv = ",".join(_policy_filters_mid(sun_alt_mid, cfg))
    alts = [r["alt_deg"] for r in pernight_rows_for_window]
    guard_s_total = float(sum(r.get("guard_s", 0.0) for r in pernight_rows_for_window))
    guard_count = int(sum(1 for r in pernight_rows_for_window if r.get("inter_exposure_guard_enforced")))
    ids = [r.get("SN") for r in pernight_rows_for_window if r.get("SN") is not None]
    unique_targets_observed = len(set(ids))
    n_planned = len(pernight_rows_for_window)
    repeat_fraction = ((n_planned - unique_targets_observed) / n_planned if n_planned else 0.0)
    if getattr(cfg, "cadence_enable", True) and getattr(cfg, "cadence_per_filter", True):
        cad_rows = [
            r
            for r in pernight_rows_for_window
            if pd.notna(r.get("cadence_days_since")) and r.get("cadence_gate_passed") is True
        ]
        cad_by_filter: Dict[str, List[float]] = {}
        for r in cad_rows:
            cad_by_filter.setdefault(r["filter"], []).append(float(r["cadence_days_since"]))
        cad_median_abs_err_by_filter: Dict[str, float] = {}
        cad_within_pct_by_filter: Dict[str, float] = {}
        target = getattr(cfg, "cadence_days_target", 3.0)
        tol = getattr(cfg, "cadence_days_tolerance", 0.5)
        for filt, vals in cad_by_filter.items():
            diffs = [abs(v - target) for v in vals]
            if diffs:
                cad_median_abs_err_by_filter[filt] = float(np.median(diffs))
                within = [abs(v - target) <= tol for v in vals]
                cad_within_pct_by_filter[filt] = 100.0 * (sum(within) / len(vals))
        all_vals = [v for vals in cad_by_filter.values() for v in vals]
        cad_median_abs_err_all = (float(np.median([abs(v - target) for v in all_vals])) if all_vals else np.nan)
        cad_within_pct_all = (100.0 * (sum(abs(v - target) <= tol for v in all_vals) / len(all_vals)) if all_vals else np.nan)
        cad_median_abs_err_by_filter_csv = ",".join(f"{k}:{round(v,2)}" for k, v in sorted(cad_median_abs_err_by_filter.items()))
        cad_within_pct_by_filter_csv = ",".join(f"{k}:{round(v,1)}" for k, v in sorted(cad_within_pct_by_filter.items()))
    else:
        cad_median_abs_err_by_filter_csv = ""
        cad_within_pct_by_filter_csv = ""
        cad_median_abs_err_all = np.nan
        cad_within_pct_all = np.nan
    return {
        "date": day_iso,
        "twilight_window": window_label,
        "n_candidates": int(ws_summary["n_candidates"]),
        "n_planned": int(n_planned),
        "unique_targets_observed": int(unique_targets_observed),
        "repeat_fraction": round(repeat_fraction, 3),
        "sum_time_s": round(ws_summary["window_sum"], 1),
        "dp_time_s": round(float(ws_summary.get("dp_time_s", 0.0)), 1),
        "backfill_time_s": round(float(ws_summary.get("backfill_time_s", 0.0)), 1),
        "window_cap_s": int(cap_s),
        "swap_count": int(ws_summary.get("swap_count", 0)),
        "internal_filter_changes": int(ws_summary["internal_changes"]),
        "filter_change_s_total": round(ws_summary["window_filter_change_s"], 1),
        "inter_exposure_guard_s": round(guard_s_total, 1),
        "inter_exposure_guard_count": guard_count,
        "mean_slew_s": (float(np.mean(ws_summary["window_slew_times"])) if ws_summary["window_slew_times"] else 0.0),
        "median_airmass": (float(np.median(ws_summary["window_airmasses"])) if ws_summary["window_airmasses"] else 0.0),
        "loaded_filters": ",".join(cfg.filters),
        "filters_used_csv": ws_summary["used_filters_csv"],
        "window_start_utc": start_utc,
        "window_end_utc": end_utc,
        "window_duration_s": int(dur_s),
        "window_mid_utc": mid_utc,
        "sun_alt_mid_deg": round(sun_alt_mid, 2),
        "policy_filters_mid_csv": policy_filters_mid_csv,
        "window_utilization": round(ws_summary["window_sum"] / max(1.0, dur_s), 4),
        "cap_utilization": round(ws_summary["window_sum"] / max(1.0, cap_s), 4),
        "cap_source": cap_source,
        "median_sky_mag_arcsec2": (float(np.median(ws_summary["window_skymags"])) if ws_summary["window_skymags"] else np.nan),
        "median_alt_deg": float(np.median(alts)) if alts else np.nan,
        "cad_median_abs_err_by_filter_csv": cad_median_abs_err_by_filter_csv,
        "cad_within_pct_by_filter_csv": cad_within_pct_by_filter_csv,
        "cad_median_abs_err_all_d": (round(cad_median_abs_err_all, 3) if not np.isnan(cad_median_abs_err_all) else np.nan),
        "cad_within_pct_all": (round(cad_within_pct_all, 1) if not np.isnan(cad_within_pct_all) else np.nan),
        "quota_assigned": ws_summary.get("quota_assigned"),
        "n_candidates_pre_cap": ws_summary.get("n_candidates_pre_cap"),
        "n_candidates_post_cap": ws_summary.get("n_candidates_post_cap"),
    }


def create_simlib_writer(cfg: PlannerConfig, header) -> SimlibWriter:
    """Factory returning a SIMLIB writer with header pre-written.

    Uses gzip compression automatically when the output filename ends with
    ".gz". This keeps the scheduler logic minimal (no need to call
    write_header() separately).
    """
    if not getattr(cfg, "simlib_out", None):  # pragma: no cover - defensive
        raise ValueError("cfg.simlib_out must be set to create a SIMLIB writer")
    outfile = str(cfg.simlib_out)
    if outfile.endswith(".gz"):
        fp = gzip.open(outfile, "wt")
    else:
        fp = open(outfile, "w")
    writer = SimlibWriter(fp, header)
    writer.write_header()
    return writer
