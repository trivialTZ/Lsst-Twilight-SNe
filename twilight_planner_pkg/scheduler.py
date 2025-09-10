"""Twilight scheduling pipeline for LSST supernova follow-up.

The module exposes :func:`plan_twilight_range_with_caps`, the main entry point
that reads a candidate CSV, prepares evening and morning windows, schedules
visits subject to Sun-altitude filter policies, and writes per-night plan,
summary, and sequence CSVs plus optional SIMLIB files.  Targets are batched by
their first filter, routed with a filter-aware slew cost, and charged carousel
overheads for cross-target swaps.  Moon separation and Sun altitude constraints
gate filter availability at each twilight window.

Overhead values follow Rubin Observatory technical notes (slew, readout, and
filter change times), and airmass calculations use the Kasten & Young (1989)
formula.  Exposure times may be overridden by
``PlannerConfig.sun_alt_exposure_ladder`` to shorten visits in bright twilight.
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Dict, List

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time

# mypy: ignore-errors


try:
    import ipywidgets as _ipyw  # noqa: F401
    from tqdm.notebook import tqdm as _tqdm  # type: ignore

    tqdm = _tqdm
except Exception:  # pragma: no cover - fallback
    from tqdm.auto import tqdm

from .astro_utils import (
    _best_time_with_moon,
    _local_timezone_from_location,
    airmass_from_alt_deg,
    allowed_filters_for_sun_alt,
    precompute_window_ephemerides,
    choose_filters_with_cap,
    great_circle_sep_deg,
    parse_sn_type_to_window_days,
    pick_first_filter_for_target,
    slew_time_seconds,
    twilight_windows_for_local_night,
)
from .config import PlannerConfig
from .constraints import effective_min_sep
from .filter_policy import allowed_filters_for_window
from .io_utils import extract_current_mags, standardize_columns
from .photom_rubin import PhotomConfig, compute_epoch_photom
from .priority import PriorityTracker
from .simlib_writer import SimlibHeader, SimlibWriter
from .sky_model import (
    RubinSkyProvider,
    SimpleSkyProvider,
    SkyModelConfig,
    sky_mag_arcsec2,
)

warnings.filterwarnings("ignore", message=".*get_moon.*deprecated.*")
warnings.filterwarnings(
    "ignore", message=".*transforming other coordinates from <GCRS Frame.*>"
)
warnings.filterwarnings(
    "ignore",
    message="Angular separation can depend on the direction of the transformation",
)


def _fmt_window(start: datetime | None, end: datetime | None, tz_local: tzinfo) -> str:
    """Format a twilight window in UTC and local time.

    Returns ``"na"`` if either bound is ``None``.
    """
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
        f"{start_utc.isoformat()} \u2192 {end_utc.isoformat()} "
        f"(local {start_local.strftime('%H:%M')} \u2192 "
        f"{end_local.strftime('%H:%M')} {offset_str})"
    )


def _mid_sun_alt_of_window(window: dict, site: EarthLocation) -> float:
    """Return Sun altitude in degrees at the midpoint of ``window``."""
    mid = window["start"] + (window["end"] - window["start"]) / 2
    return float(
        get_sun(Time(mid))
        .transform_to(AltAz(location=site, obstime=Time(mid)))
        .alt.to(u.deg)
        .value
    )


def _policy_filters_mid(sun_alt_deg: float, cfg: PlannerConfig) -> list[str]:
    """Allowed filters at ``sun_alt_deg`` according to policy."""
    return allowed_filters_for_sun_alt(sun_alt_deg, cfg)


def _path_plan(
    outdir: str | Path,
    start: datetime,
    end: datetime,
    run_label: str | None = None,
) -> Path:
    """Return output path for the per-SN plan CSV."""
    label = f"{run_label}_" if run_label else ""
    return (
        Path(outdir)
        / f"lsst_twilight_plan_{label}{start.isoformat()}_to_{end.isoformat()}.csv"
    )


def _path_summary(
    outdir: str | Path,
    start: datetime,
    end: datetime,
    run_label: str | None = None,
) -> Path:
    """Return output path for the nightly/window summary CSV."""
    label = f"{run_label}_" if run_label else ""
    return (
        Path(outdir)
        / f"lsst_twilight_summary_{label}{start.isoformat()}_to_{end.isoformat()}.csv"
    )


def _path_sequence(
    outdir: str | Path,
    start: datetime,
    end: datetime,
    run_label: str | None = None,
) -> Path:
    """Return output path for the true on-sky sequence CSV."""
    label = f"{run_label}_" if run_label else ""
    return (
        Path(outdir)
        / f"lsst_twilight_sequence_true_{label}{start.isoformat()}_to_{end.isoformat()}.csv"
    )


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
) -> None:
    """Print nightly eligibility and twilight window information."""
    if not verbose:
        return
    print(
        f"{day_iso}: eligible={eligible} visible={visible} "
        f"planned_total={planned_total}"
    )
    print(
        f"  evening_twilight: " f"{_fmt_window(evening_start, evening_end, tz_local)}"
    )
    print(
        f"  morning_twilight: " f"{_fmt_window(morning_start, morning_end, tz_local)}"
    )


def _cap_candidates_per_window(
    df_sorted: pd.DataFrame,
    cfg: PlannerConfig,
    evening_cap_s: float,
    morning_cap_s: float,
) -> tuple[pd.DataFrame, dict]:
    """Limit candidates per twilight window based on time caps.

    Returns the capped DataFrame along with diagnostics describing the assigned
    quotas and pre/post counts. If ``cfg.max_sn_per_night`` is ``inf`` or
    ``None``, ``df_sorted`` is returned unchanged with diagnostics noting the
    counts.
    """

    limit = getattr(cfg, "max_sn_per_night", None)
    diag = {
        "quota_evening": None,
        "quota_morning": None,
        "pre_total": int(len(df_sorted)),
        "post_total": None,
    }
    if limit is None or (isinstance(limit, float) and math.isinf(limit)):
        out = df_sorted
        diag["post_total"] = int(len(out))
        return out, diag

    total_max = int(limit)
    if total_max <= 0:
        out = df_sorted.iloc[0:0].copy()
        diag["quota_evening"] = 0
        diag["quota_morning"] = 0
        diag["post_total"] = 0
        return out, diag

    if "best_window_index" not in df_sorted.columns:
        out = df_sorted.head(total_max).copy()
        diag["post_total"] = int(len(out))
        return out, diag

    e_cap = max(float(evening_cap_s or 0.0), 0.0)
    m_cap = max(float(morning_cap_s or 0.0), 0.0)
    total_cap = e_cap + m_cap
    if total_cap <= 0:
        out = df_sorted.head(total_max).copy()
        diag["post_total"] = int(len(out))
        return out, diag

    q_e = int(round(total_max * (e_cap / total_cap))) if e_cap > 0 else 0
    q_m = total_max - q_e
    diag["quota_evening"] = q_e
    diag["quota_morning"] = q_m

    df_e = df_sorted[df_sorted["best_window_index"] == 0].head(q_e)
    df_m = df_sorted[df_sorted["best_window_index"] == 1].head(q_m)

    short_e = q_e - len(df_e)
    short_m = q_m - len(df_m)
    if short_e > 0:
        topup = df_sorted[df_sorted["best_window_index"] == 1].iloc[
            len(df_m) : len(df_m) + short_e
        ]
        df_m = pd.concat([df_m, topup], ignore_index=True)
    if short_m > 0:
        topup = df_sorted[df_sorted["best_window_index"] == 0].iloc[
            len(df_e) : len(df_e) + short_m
        ]
        df_e = pd.concat([df_e, topup], ignore_index=True)

    capped = pd.concat([df_e, df_m], ignore_index=True)
    capped = capped.sort_values(
        by=["priority_score", "max_alt_deg"], ascending=[False, False], kind="stable"
    )
    diag["post_total"] = int(len(capped))
    return capped, diag


def _emit_simlib(
    writer: SimlibWriter,
    libid_counter: int,
    name: str,
    ra_deg: float,
    dec_deg: float,
    epochs: list[dict],
) -> int:
    """Write one SIMLIB block and return the incremented libid counter."""
    writer.start_libid(
        libid_counter,
        ra_deg,
        dec_deg,
        nobs=len(epochs),
        comment=name,
    )
    libid_counter += 1
    for epoch in epochs:
        writer.add_epoch(**epoch)
    writer.end_libid()
    return libid_counter


def _prepare_window_candidates(
    group_df: pd.DataFrame,
    window: dict,
    idx_w: int,
    cfg: PlannerConfig,
    site: EarthLocation,
    tracker: PriorityTracker,
    mag_lookup: dict,
    current_filter_by_window: dict[int, str],
    cad_on: bool,
    cad_tgt: float,
    cad_sig: float,
    cad_wt: float,
    cad_first: float,
) -> tuple[list[dict], bool]:
    """Build candidate target dicts for a twilight window.

    Returns the candidate list and a flag indicating whether an exposure
    override from ``cfg.sun_alt_exposure_ladder`` was applied.
    """
    mid_sun_alt = _mid_sun_alt_of_window(window, site)
    current_time_utc = pd.Timestamp(window["start"]).tz_convert("UTC")
    override_applied = False
    for low, high, exp_dict in cfg.sun_alt_exposure_ladder:
        if low < mid_sun_alt <= high:
            override_exp = cfg.exposure_by_filter.copy()
            override_exp.update(exp_dict)
            cfg.exposure_by_filter = override_exp
            override_applied = True
            break

    candidates: list[dict] = []
    for _, row in group_df.iterrows():
        allowed = allowed_filters_for_window(
            mag_lookup.get(row["Name"], {}),
            mid_sun_alt,
            row["_moon_alt"],
            row["_moon_phase"],
            row["_moon_sep"],
            airmass_from_alt_deg(row["max_alt_deg"]),
            cfg.fwhm_eff or 0.7,
        )
        allowed = [f for f in allowed if f in cfg.filters]
        allowed_policy = _policy_filters_mid(mid_sun_alt, cfg)
        allowed = [f for f in allowed if f in allowed_policy]
        moon_sep_ok = {
            f: row["_moon_sep"]
            >= effective_min_sep(
                f,
                row["_moon_alt"],
                row["_moon_phase"],
                cfg.min_moon_sep_by_filter,
            )
            for f in allowed
        }
        policy_allowed = [f for f in allowed if moon_sep_ok.get(f, False)]
        if not policy_allowed:
            continue
        first = pick_first_filter_for_target(
            row["Name"],
            row.get("SN_type_raw"),
            tracker,
            policy_allowed,
            cfg,
            sun_alt_deg=mid_sun_alt,
            moon_sep_ok=moon_sep_ok,
            current_mag=mag_lookup.get(row["Name"]),
            current_filter=current_filter_by_window.get(idx_w),
        )
        if first is None:
            continue
        now_mjd_for_bonus = Time(current_time_utc).mjd
        if cad_on:
            rest = sorted(
                [f for f in policy_allowed if f != first],
                key=lambda f: tracker.compute_filter_bonus(
                    row["Name"],
                    f,
                    now_mjd_for_bonus,
                    cad_tgt,
                    cad_sig,
                    cad_wt,
                    cad_first,
                    cfg.cosmo_weight_by_filter,
                    cfg.color_target_pairs,
                    cfg.color_window_days,
                    cfg.color_alpha,
                    cfg.first_epoch_color_boost,
                ),
                reverse=True,
            )
        else:
            rest = [f for f in policy_allowed if f != first]
        cand = {
            "Name": row["Name"],
            "RA_deg": row["RA_deg"],
            "Dec_deg": row["Dec_deg"],
            "best_time_utc": row["best_time_utc"],
            "max_alt_deg": row["max_alt_deg"],
            "priority_score": row["priority_score"],
            "first_filter": first,
            "sn_type": row.get("SN_type_raw"),
            "allowed": [first] + rest,
            "policy_allowed": policy_allowed,
            "moon_sep_ok": moon_sep_ok,
            "moon_sep": float(row["_moon_sep"]),
        }
        candidates.append(cand)

    return candidates, override_applied


def _attempt_schedule_one(
    target_dict,
    window_state,
    cfg: PlannerConfig,
    site: EarthLocation,
    tracker: PriorityTracker,
    phot_cfg: PhotomConfig,
    sky_cfg: SkyModelConfig,
    sky_provider,
    writer_or_none,
) -> tuple[bool, dict]:
    """Attempt to schedule a single target within a twilight window.

    Parameters
    ----------
    target_dict : dict
        Candidate target information with filter options and metadata.
    window_state : dict
        Per-window state such as current time, filter, and accumulated time.

    Returns
    -------
    scheduled : bool
        ``True`` if the target was scheduled.
    effects : dict
        Dict capturing rows, epochs, time usage, and state deltas.
    """

    if writer_or_none is not None:  # pragma: no cover - placeholder
        pass

    allow_defer: bool = window_state.get("allow_defer", True)
    deferred: list[dict] = window_state.get("deferred", [])
    window_sum: float = window_state.get("window_sum", 0.0)
    cap_s: float = window_state.get("cap_s", 0.0)
    state: str | None = window_state.get("state")
    prev: dict | None = window_state.get("prev")
    current_time_utc: pd.Timestamp = window_state["current_time_utc"]
    order_in_window: int = window_state.get("order_in_window", 0)
    day = window_state["day"]
    window_label_out: str = window_state["window_label_out"]
    mag_lookup: dict = window_state["mag_lookup"]

    t = target_dict
    sep = (
        0.0
        if prev is None
        else great_circle_sep_deg(
            prev["RA_deg"], prev["Dec_deg"], t["RA_deg"], t["Dec_deg"]
        )
    )
    cfg.current_mag_by_filter = mag_lookup.get(t["Name"])
    cfg.current_alt_deg = t["max_alt_deg"]
    cfg.current_mjd = (
        Time(t["best_time_utc"]).mjd
        if isinstance(t["best_time_utc"], (datetime, pd.Timestamp))
        else None
    )
    now_mjd = Time(current_time_utc).mjd
    cad_on = getattr(cfg, "cadence_enable", True) and getattr(
        cfg, "cadence_per_filter", True
    )
    cad_tgt = getattr(cfg, "cadence_days_target", 3.0)
    cad_jit = getattr(cfg, "cadence_jitter_days", 0.25)
    cad_sig = getattr(cfg, "cadence_bonus_sigma_days", 0.5)
    cad_wt = getattr(cfg, "cadence_bonus_weight", 0.25)
    cad_first = getattr(cfg, "cadence_first_epoch_bonus_weight", 0.0)
    if cad_on:
        gated = [
            f
            for f in t["policy_allowed"]
            if tracker.cadence_gate(t["Name"], f, now_mjd, cad_tgt, cad_jit)
        ]
        if not gated:
            if allow_defer:
                deferred.append(t)
            return False, {}

        def _bonus(f: str) -> float:
            return tracker.compute_filter_bonus(
                t["Name"],
                f,
                now_mjd,
                cad_tgt,
                cad_sig,
                cad_wt,
                cad_first,
                cfg.cosmo_weight_by_filter,
                cfg.color_target_pairs,
                cfg.color_window_days,
                cfg.color_alpha,
                cfg.first_epoch_color_boost,
            )

        first = (
            t["first_filter"] if t["first_filter"] in gated else max(gated, key=_bonus)
        )
        rest = sorted([f for f in gated if f != first], key=_bonus, reverse=True)
        filters_pref = [first]
        if int(cfg.max_filters_per_visit) >= 2 and rest:
            opp = tracker.RED if first in tracker.BLUE else tracker.BLUE
            second = next((f for f in rest if f in opp), rest[0])
            filters_pref.append(second)
            rest = [f for f in rest if f != second]
        filters_pref.extend(rest)
        filters_pref = filters_pref[: int(cfg.max_filters_per_visit)]
    else:
        filters_pref = [t["first_filter"]] + [
            f for f in t["allowed"] if f != t["first_filter"]
        ]
        if int(cfg.max_filters_per_visit) >= 2 and len(filters_pref) > 1:
            first = filters_pref[0]
            rest = filters_pref[1:]
            opp = tracker.RED if first in tracker.BLUE else tracker.BLUE
            second = next((f for f in rest if f in opp), rest[0])
            filters_pref = [first, second] + [f for f in rest if f != second]
        filters_pref = filters_pref[: int(cfg.max_filters_per_visit)]
    filters_used, timing = choose_filters_with_cap(
        filters_pref,
        sep,
        cfg.per_sn_cap_s,
        cfg,
        current_filter=state,
        max_filters_per_visit=cfg.max_filters_per_visit,
    )
    if not filters_used:
        return False, {}
    natural_gap_first = max(timing["slew_s"], cfg.readout_s) + timing.get(
        "cross_filter_change_s", 0.0
    )
    guard_first_s = (
        0.0
        if window_sum == 0.0
        else max(0.0, cfg.inter_exposure_min_s - natural_gap_first)
    )
    internal_gap = cfg.readout_s + (
        cfg.filter_change_s if len(filters_used) > 1 else 0.0
    )
    guard_internal_per = max(0.0, cfg.inter_exposure_min_s - internal_gap)
    guard_internal_total = guard_internal_per * max(0, len(filters_used) - 1)
    guard_s = guard_first_s + guard_internal_total
    elapsed_overhead = (
        max(timing["slew_s"], cfg.readout_s)
        + timing.get("filter_changes_s", 0.0)
        + timing.get("cross_filter_change_s", 0.0)
    )
    total_with_guard = elapsed_overhead + timing["exposure_s"] + guard_s
    if window_sum + total_with_guard > cap_s:
        return False, {}

    timing["total_s"] = total_with_guard
    timing["guard_s"] = guard_s
    timing["elapsed_overhead_s"] = elapsed_overhead
    timing["guard_first_s"] = guard_first_s
    timing["guard_internal_s"] = guard_internal_total

    preferred_utc = (
        pd.Timestamp(t["best_time_utc"]).tz_convert("UTC")
        if isinstance(t["best_time_utc"], pd.Timestamp)
        else pd.Timestamp(t["best_time_utc"]).tz_localize("UTC")
    )
    sn_start_utc = current_time_utc
    sn_end_utc = sn_start_utc + pd.to_timedelta(timing["total_s"], unit="s")
    visit_mjd = Time(sn_start_utc).mjd

    order = order_in_window + 1
    sequence_row = {
        "date": day.date().isoformat(),
        "twilight_window": window_label_out,
        "order_in_window": int(order),
        "SN": t["Name"],
        "RA_deg": round(t["RA_deg"], 6),
        "Dec_deg": round(t["Dec_deg"], 6),
        "filters_used_csv": ",".join(filters_used),
        "preferred_best_utc": preferred_utc.isoformat(),
        "sn_start_utc": sn_start_utc.isoformat(),
        "sn_end_utc": sn_end_utc.isoformat(),
        "total_time_s": round(timing.get("total_s", 0.0), 2),
        "slew_s": round(timing.get("slew_s", 0.0), 2),
        "readout_s": round(timing.get("readout_s", 0.0), 2),
        "filter_changes_s": round(timing.get("filter_changes_s", 0.0), 2),
        "cross_filter_change_s": round(timing.get("cross_filter_change_s", 0.0), 2),
        "guard_s": round(timing.get("guard_s", 0.0), 2),
        "elapsed_overhead_s": round(timing.get("elapsed_overhead_s", 0.0), 2),
    }

    pernight_rows: list[dict] = []
    epochs: list[dict] = []
    sky_mags: list[float] = []
    air = 0.0
    for f in filters_used:
        exp_s = timing.get("exp_times", {}).get(f, cfg.exposure_by_filter.get(f, 0.0))
        flags = timing.get("flags_by_filter", {}).get(f, set())
        alt_deg = float(t["max_alt_deg"])
        air = airmass_from_alt_deg(alt_deg)
        mjd = visit_mjd
        if sky_provider:
            sky_mag = sky_provider.sky_mag(
                mjd, t["RA_deg"], t["Dec_deg"], f, airmass_from_alt_deg(alt_deg)
            )
        else:
            sky_mag = sky_mag_arcsec2(f, sky_cfg)
        eph = compute_epoch_photom(f, exp_s, alt_deg, sky_mag, phot_cfg)
        sky_mags.append(sky_mag)
        if writer_or_none:
            epochs.append(
                {
                    "mjd": mjd,
                    "band": f,
                    "gain": eph.GAIN,
                    "rdnoise": eph.RDNOISE,
                    "skysig": eph.SKYSIG,
                    "nea": eph.NEA_pix,
                    "zpavg": eph.ZPTAVG,
                    "zperr": eph.ZPTERR,
                    "mag": -99.0,
                }
            )
        start_utc = sn_start_utc
        total_s = round(timing["total_s"], 2)
        end_utc = sn_end_utc
        days_since = tracker.days_since(t["Name"], f, visit_mjd)
        gate_passed = (
            tracker.cadence_gate(t["Name"], f, visit_mjd, cad_tgt, cad_jit)
            if cad_on
            else True
        )
        best_start_utc = (
            pd.Timestamp(t["best_time_utc"]).tz_convert("UTC")
            if isinstance(t["best_time_utc"], pd.Timestamp)
            else pd.Timestamp(t["best_time_utc"]).tz_localize("UTC")
        )
        row = {
            "date": day.date().isoformat(),
            "twilight_window": window_label_out,
            "SN": t["Name"],
            "RA_deg": round(t["RA_deg"], 6),
            "Dec_deg": round(t["Dec_deg"], 6),
            "best_twilight_time_utc": best_start_utc.isoformat(),
            "visit_start_utc": start_utc.isoformat(),
            "sn_end_utc": end_utc.isoformat(),
            "filter": f,
            "t_exp_s": round(exp_s, 1),
            "airmass": round(air, 3),
            "alt_deg": round(alt_deg, 2),
            "sky_mag_arcsec2": round(sky_mag, 2),
            "moon_sep": round(float(t.get("moon_sep", np.nan)), 2),
            "ZPT": round(eph.ZPTAVG, 3),
            "SKYSIG": round(eph.SKYSIG, 3),
            "NEA_pix": round(eph.NEA_pix, 2),
            "RDNOISE": round(eph.RDNOISE, 2),
            "GAIN": round(eph.GAIN, 2),
            "saturation_guard_applied": "sat_guard" in flags,
            "warn_nonlinear": "warn_nonlinear" in flags,
            "priority_score": round(float(t["priority_score"]), 2),
            "cadence_days_since": (
                round(days_since, 3) if days_since is not None else np.nan
            ),
            "cadence_target_d": cad_tgt,
            "cadence_gate_passed": bool(gate_passed),
            "slew_s": round(timing["slew_s"], 2) if f == filters_used[0] else 0.0,
            "cross_filter_change_s": (
                round(timing.get("cross_filter_change_s", 0.0), 2)
                if f == filters_used[0]
                else 0.0
            ),
            "filter_changes_s": (
                round(timing.get("filter_changes_s", 0.0), 2)
                if f == filters_used[0]
                else 0.0
            ),
            "readout_s": round(timing["readout_s"], 2) if f == filters_used[0] else 0.0,
            "exposure_s": (
                round(timing["exposure_s"], 2) if f == filters_used[0] else 0.0
            ),
            "guard_s": (
                round(timing.get("guard_s", 0.0), 2) if f == filters_used[0] else 0.0
            ),
            "inter_exposure_guard_enforced": (
                bool(timing.get("guard_s", 0.0) > 0.0)
                if f == filters_used[0]
                else False
            ),
            "total_time_s": total_s if f == filters_used[0] else 0.0,
            "elapsed_overhead_s": (
                round(timing.get("elapsed_overhead_s", 0.0), 2)
                if f == filters_used[0]
                else 0.0
            ),
        }
        pernight_rows.append(row)

    swap_count_delta = 0
    if state is not None and filters_used:
        if state != filters_used[0]:
            swap_count_delta = 1
    if filters_used:
        state = filters_used[-1]

    summary_updates = {
        "window_filter_change_s_delta": timing.get("filter_changes_s", 0.0)
        + timing.get("cross_filter_change_s", 0.0),
        "internal_changes_delta": max(0, len(filters_used) - 1),
        "slew_time": timing["slew_s"],
        "airmass": air,
        "sky_mags": sky_mags,
        "swap_count_delta": swap_count_delta,
    }
    tracker.record_detection(
        t["Name"], timing["exposure_s"], filters_used, mjd=visit_mjd
    )

    effects = {
        "pernight_rows": pernight_rows,
        "sequence_rows": [sequence_row],
        "simlib_epochs": epochs,
        "time_used_s": total_with_guard,
        "state_updates": {
            "current_time_utc": sn_end_utc,
            "state_filter": state,
            "prev_target": t,
            "filters_used_set_delta": set(filters_used),
            "order_in_window": order,
            "summary_updates": summary_updates,
        },
    }

    return True, effects


def _build_window_summary_row(
    day_iso,
    window_label,
    win,
    idx_w,
    ws_summary,
    cap_s,
    cap_source,
    cfg,
    site,
    pernight_rows_for_window,
) -> dict:
    """Return summary metrics for a window with exact CSV keys."""

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
    guard_count = int(
        sum(
            1
            for r in pernight_rows_for_window
            if r.get("inter_exposure_guard_enforced")
        )
    )
    ids = [r.get("SN") for r in pernight_rows_for_window if r.get("SN") is not None]
    unique_targets_observed = len(set(ids))
    n_planned = len(pernight_rows_for_window)
    repeat_fraction = (
        (n_planned - unique_targets_observed) / n_planned if n_planned else 0.0
    )
    if getattr(cfg, "cadence_enable", True) and getattr(
        cfg, "cadence_per_filter", True
    ):
        cad_rows = [
            r
            for r in pernight_rows_for_window
            if pd.notna(r.get("cadence_days_since"))
            and r.get("cadence_gate_passed") is True
        ]
        cad_by_filter: Dict[str, List[float]] = {}
        for r in cad_rows:
            cad_by_filter.setdefault(r["filter"], []).append(
                float(r["cadence_days_since"])
            )
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
        cad_median_abs_err_all = (
            float(np.median([abs(v - target) for v in all_vals]))
            if all_vals
            else np.nan
        )
        cad_within_pct_all = (
            100.0 * (sum(abs(v - target) <= tol for v in all_vals) / len(all_vals))
            if all_vals
            else np.nan
        )
        cad_median_abs_err_by_filter_csv = ",".join(
            f"{k}:{round(v,2)}" for k, v in sorted(cad_median_abs_err_by_filter.items())
        )
        cad_within_pct_by_filter_csv = ",".join(
            f"{k}:{round(v,1)}" for k, v in sorted(cad_within_pct_by_filter.items())
        )
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
        "window_cap_s": int(cap_s),
        "swap_count": int(ws_summary.get("swap_count", 0)),
        "internal_filter_changes": int(ws_summary["internal_changes"]),
        "filter_change_s_total": round(ws_summary["window_filter_change_s"], 1),
        "inter_exposure_guard_s": round(guard_s_total, 1),
        "inter_exposure_guard_count": guard_count,
        "mean_slew_s": (
            float(np.mean(ws_summary["window_slew_times"]))
            if ws_summary["window_slew_times"]
            else 0.0
        ),
        "median_airmass": (
            float(np.median(ws_summary["window_airmasses"]))
            if ws_summary["window_airmasses"]
            else 0.0
        ),
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
        "median_sky_mag_arcsec2": (
            float(np.median(ws_summary["window_skymags"]))
            if ws_summary["window_skymags"]
            else np.nan
        ),
        "median_alt_deg": float(np.median(alts)) if alts else np.nan,
        "cad_median_abs_err_by_filter_csv": cad_median_abs_err_by_filter_csv,
        "cad_within_pct_by_filter_csv": cad_within_pct_by_filter_csv,
        "cad_median_abs_err_all_d": (
            round(cad_median_abs_err_all, 3)
            if not np.isnan(cad_median_abs_err_all)
            else np.nan
        ),
        "cad_within_pct_all": (
            round(cad_within_pct_all, 1) if not np.isnan(cad_within_pct_all) else np.nan
        ),
        "quota_assigned": ws_summary.get("quota_assigned"),
        "n_candidates_pre_cap": ws_summary.get("n_candidates_pre_cap"),
        "n_candidates_post_cap": ws_summary.get("n_candidates_post_cap"),
    }


def plan_twilight_range_with_caps(
    csv_path: str,
    outdir: str,
    start_date: str,
    end_date: str,
    cfg: PlannerConfig,
    run_label: str | None = None,
    verbose: bool = True,
    *,
    stream_per_sn: bool = False,
    stream_sequence: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a twilight observing plan over a date range with per-window time caps.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file listing supernovae.
    outdir : str
        Directory where output CSVs are written.
    start_date : str
        Inclusive start date in ``YYYY-MM-DD`` (UTC).
    end_date : str
        Inclusive end date in ``YYYY-MM-DD`` (UTC).
    cfg : PlannerConfig
        Configuration with site, filter, and timing parameters. Filters outside
        ``cfg.sun_alt_policy`` are excluded at disallowed Sun altitudes.
    run_label : str, optional
        Optional label inserted into output filenames. Defaults to ``"hybrid"``
        (the standard priority strategy). Useful for differentiating multiple
        runs over the same date range.
    verbose : bool, optional
        If ``True``, print progress messages.

    Returns
    -------
    pernight_df : pandas.DataFrame
        Planned observations for each supernova.
    nights_df : pandas.DataFrame
        Summary of time usage for each night and window.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(csv_path)
    df = standardize_columns(raw, cfg)
    mag_lookup = extract_current_mags(df)

    phot_cfg = PhotomConfig(
        pixel_scale_arcsec=cfg.pixel_scale_arcsec,
        zpt1s=cfg.zpt1s or None,
        k_m=cfg.k_m or None,
        fwhm_eff=cfg.fwhm_eff or None,
        read_noise_e=cfg.read_noise_e,
        gain_e_per_adu=cfg.gain_e_per_adu,
        zpt_err_mag=cfg.zpt_err_mag,
        npe_pixel_saturate=cfg.simlib_npe_pixel_saturate,
    )
    site = EarthLocation(
        lat=cfg.lat_deg * u.deg, lon=cfg.lon_deg * u.deg, height=cfg.height_m * u.m
    )
    sky_cfg = SkyModelConfig(
        dark_sky_mag=cfg.dark_sky_mag,
        twilight_delta_mag=cfg.twilight_delta_mag,
    )
    try:
        sky_provider = RubinSkyProvider()
    except Exception:
        sky_provider = SimpleSkyProvider(sky_cfg, site=site)
    cfg.sky_provider = sky_provider

    writer = None
    if cfg.simlib_out:
        hdr = SimlibHeader(
            SURVEY=cfg.simlib_survey,
            FILTERS=cfg.simlib_filters,
            PIXSIZE=cfg.simlib_pixsize,
            NPE_PIXEL_SATURATE=cfg.simlib_npe_pixel_saturate,
            PHOTFLAG_SATURATE=cfg.simlib_photflag_saturate,
            PSF_UNIT=cfg.simlib_psf_unit,
        )
        writer = SimlibWriter(open(cfg.simlib_out, "w"), hdr)
        writer.write_header()
    libid_counter = 1

    filters = list(cfg.filters or [])
    if cfg.carousel_capacity and len(filters) > cfg.carousel_capacity:
        drop = "u" if "u" in filters else filters[-1]
        if verbose:
            print(
                f"WARNING: requesting {len(filters)} filters but carousel holds only {cfg.carousel_capacity}; dropping {drop}."
            )
        filters.remove(drop)
        cfg.filters = filters

    start = pd.to_datetime(start_date, utc=True).date()
    end = pd.to_datetime(end_date, utc=True).date()

    label = run_label or "hybrid"
    pernight_path = _path_plan(outdir, pd.to_datetime(start_date, utc=True), pd.to_datetime(end_date, utc=True), label)
    nights_path = _path_summary(outdir, pd.to_datetime(start_date, utc=True), pd.to_datetime(end_date, utc=True), label)
    seq_path = _path_sequence(outdir, pd.to_datetime(start_date, utc=True), pd.to_datetime(end_date, utc=True), label)

    pernight_rows: List[Dict] = [] if not stream_per_sn else []  # day-batched if streaming
    nights_rows: List[Dict] = []
    # ---- NEW: true sequential execution order collector (one row per SN visit) ----
    sequence_rows: list[dict] = [] if not stream_sequence else []  # day-batched if streaming
    # Streaming state
    per_sn_header_written = False
    seq_header_written = False
    per_sn_count = 0
    seq_count = 0
    nights = pd.date_range(start, end, freq="D")
    nights_iter = tqdm(nights, desc="Nights", unit="night", leave=True)
    tracker = PriorityTracker(
        hybrid_detections=cfg.hybrid_detections,
        hybrid_exposure_s=cfg.hybrid_exposure_s,
        lc_detections=cfg.lc_detections,
        lc_exposure_s=cfg.lc_exposure_s,
        unique_lookback_days=cfg.unique_lookback_days,
    )
    tz_local = _local_timezone_from_location(site)

    for day_idx, day in enumerate(nights_iter):
        # day-batched collectors (used when streaming)
        day_pernight_rows: List[Dict] = []
        day_sequence_rows: List[Dict] = []
        seen_ids: set[str] = set()
        # Here 'day' is interpreted as *local* civil date of the evening block
        windows = twilight_windows_for_local_night(
            day.date(),
            site,
            cfg.twilight_sun_alt_min_deg,
            cfg.twilight_sun_alt_max_deg,
        )
        if not windows:
            continue
        evening_start: datetime | None = None
        evening_end: datetime | None = None
        morning_start: datetime | None = None
        morning_end: datetime | None = None
        for w in windows:
            if w.get("label") == "evening":
                evening_start = w["start"]
                evening_end = w["end"]
            elif w.get("label") == "morning":
                morning_start = w["start"]
                morning_end = w["end"]
        if cfg.evening_twilight:
            hh, mm = map(int, cfg.evening_twilight.split(":"))
            dt_local = datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz_local)
            evening_start = dt_local.astimezone(timezone.utc)
        if cfg.morning_twilight:
            hh, mm = map(int, cfg.morning_twilight.split(":"))
            d2 = day + pd.Timedelta(days=1)
            dt_local = datetime(d2.year, d2.month, d2.day, hh, mm, tzinfo=tz_local)
            morning_start = dt_local.astimezone(timezone.utc)
        # Conservative baseline Moon separation used while sampling best times.
        # Detailed per-filter checks with altitude/phase scaling are applied later via effective_min_sep.
        vals: list[float] = []
        if getattr(cfg, "min_moon_sep_by_filter", None) and getattr(
            cfg, "filters", None
        ):
            try:
                vals = [cfg.min_moon_sep_by_filter.get(f, 0.0) for f in cfg.filters]
            except Exception:
                vals = []
        # Coarse gate should be permissive; strict, per-filter checks happen later.
        req_sep = min(vals) if vals else 0.0

        current_filter_by_window: Dict[int, str | None] = {}
        swap_count_by_window: Dict[int, int] = {}
        window_caps: Dict[int, float] = {}
        window_labels: Dict[int, str | None] = {}
        cap_source_by_window: Dict[int, str] = {}
        evening_idx: int | None = None
        morning_idx: int | None = None
        for idx_w, w in enumerate(windows):
            current_filter_by_window[idx_w] = cfg.start_filter
            swap_count_by_window[idx_w] = 0
            label = w.get("label")
            window_labels[idx_w] = label
            if label == "morning":
                morning_idx = idx_w
                cap = cfg.morning_cap_s
                if cap == "auto":
                    cap = (w["end"] - w["start"]).total_seconds()
                    cap_source_by_window[idx_w] = "window_duration"
                else:
                    cap_source_by_window[idx_w] = "morning_cap_s"
                window_caps[idx_w] = float(cap)
            elif label == "evening":
                evening_idx = idx_w
                cap = cfg.evening_cap_s
                if cap == "auto":
                    cap = (w["end"] - w["start"]).total_seconds()
                    cap_source_by_window[idx_w] = "window_duration"
                else:
                    cap_source_by_window[idx_w] = "evening_cap_s"
                window_caps[idx_w] = float(cap)
            else:
                window_caps[idx_w] = 0.0
                cap_source_by_window[idx_w] = "none"

        evening_cap_s_val = window_caps.get(evening_idx, 0.0)
        morning_cap_s_val = window_caps.get(morning_idx, 0.0)

        # Discovery cutoff at 23:59:59 *local* on the night’s evening date
        cutoff_local = datetime(
            day.year, day.month, day.day, 23, 59, 59, tzinfo=tz_local
        )
        cutoff = cutoff_local.astimezone(timezone.utc)
        if "typical_lifetime_days" in df.columns:
            lifetime_days_each = df["typical_lifetime_days"]
        else:
            lifetime_days_each = df["SN_type_raw"].apply(
                lambda t: parse_sn_type_to_window_days(t, cfg)
            )
        min_allowed_disc_each = cutoff - lifetime_days_each.apply(
            lambda d: timedelta(days=int(d))
        )
        has_disc = df["discovery_datetime"].notna()
        subset = df[
            has_disc
            & (df["discovery_datetime"] <= cutoff)
            & (df["discovery_datetime"] >= min_allowed_disc_each)
        ].copy()
        if subset.empty:
            _log_day_status(
                day.date().isoformat(),
                0,
                0,
                0,
                evening_start,
                evening_end,
                morning_start,
                morning_end,
                tz_local,
                verbose,
            )
            continue

        best_alts, best_times, best_winidx = [], [], []
        # Precompute per-window ephemerides shared across all targets
        labeled = [
            (i, w)
            for i, w in enumerate(windows)
            if w.get("label") in ("morning", "evening")
        ]
        precomp_by_idx: dict[int, dict] = {}
        for idx_w, w in labeled:
            precomp_by_idx[idx_w] = precompute_window_ephemerides(
                (w["start"], w["end"]), site, cfg.twilight_step_min
            )
        for _, row in subset.iterrows():
            sc = SkyCoord(row["RA_deg"] * u.deg, row["Dec_deg"] * u.deg, frame="icrs")
            max_alt, max_time, max_idx = -999.0, None, None
            best_moon = (float("nan"), float("nan"), float("nan"))
            for idx_w, w in labeled:
                alt_deg, t_utc, moon_alt_deg, moon_phase, moon_sep_deg = (
                    _best_time_with_moon(
                        sc,
                        (w["start"], w["end"]),
                        site,
                        cfg.twilight_step_min,
                        cfg.min_alt_deg,
                        req_sep,
                        precomputed=precomp_by_idx.get(idx_w),
                    )
                )
                if alt_deg > max_alt:
                    max_alt, max_time, max_idx = alt_deg, t_utc, idx_w
                    best_moon = (moon_alt_deg, moon_phase, moon_sep_deg)
            best_alts.append(max_alt)
            best_times.append(max_time if max_time is not None else pd.NaT)
            best_winidx.append(max_idx if max_time is not None else -1)
            subset.loc[_, "_moon_alt"] = best_moon[0]
            subset.loc[_, "_moon_phase"] = best_moon[1]
            subset.loc[_, "_moon_sep"] = best_moon[2]

        subset["max_alt_deg"] = best_alts
        subset["best_time_utc"] = best_times
        subset["best_window_index"] = best_winidx

        visible = subset[
            (subset["best_time_utc"].notna())
            & (subset["max_alt_deg"] >= cfg.min_alt_deg)
            & (subset["best_window_index"] >= 0)
        ].copy()
        if visible.empty:
            _log_day_status(
                day.date().isoformat(),
                len(subset),
                0,
                0,
                evening_start,
                evening_end,
                morning_start,
                morning_end,
                tz_local,
                verbose,
            )
            continue

        visible["priority_score"] = visible.apply(
            lambda r: tracker.score(
                r["Name"],
                r.get("SN_type_raw"),
                cfg.priority_strategy,
                now_mjd=(
                    Time(r["best_time_utc"]).mjd
                    if pd.notna(r.get("best_time_utc"))
                    else None
                ),
            ),
            axis=1,
        )
        if cfg.priority_strategy == "unique_first":
            thr = float(getattr(cfg, "unique_first_drop_threshold", 0.0))
            visible = visible[visible["priority_score"] > thr].copy()
        visible.sort_values(
            ["priority_score", "max_alt_deg"], ascending=[False, False], inplace=True
        )
        pre_counts = (
            visible.groupby("best_window_index").size().to_dict()
            if "best_window_index" in visible.columns
            else {}
        )
        top_global, cap_diag = _cap_candidates_per_window(
            visible, cfg, evening_cap_s_val, morning_cap_s_val
        )
        post_counts = (
            top_global.groupby("best_window_index").size().to_dict()
            if "best_window_index" in top_global.columns
            else {}
        )

        for idx_w in sorted(set(top_global["best_window_index"].values)):
            group = top_global[top_global["best_window_index"] == idx_w].copy()
            if group.empty:
                continue
            # Skip unlabeled (previous/next day) twilight windows entirely
            if window_labels.get(idx_w) is None:
                continue

            win = windows[idx_w]
            window_label = window_labels.get(idx_w)
            window_label_out = window_label if window_label else f"W{idx_w}"
            cad_on = getattr(cfg, "cadence_enable", True) and getattr(
                cfg, "cadence_per_filter", True
            )
            cad_tgt = getattr(cfg, "cadence_days_target", 3.0)
            cad_jit = getattr(cfg, "cadence_jitter_days", 0.25)
            cad_sig = getattr(cfg, "cadence_bonus_sigma_days", 0.5)
            cad_wt = getattr(cfg, "cadence_bonus_weight", 0.25)
            cad_first = getattr(cfg, "cadence_first_epoch_bonus_weight", 0.0)

            original_exp = cfg.exposure_by_filter
            candidates, override_applied = _prepare_window_candidates(
                group,
                win,
                idx_w,
                cfg,
                site,
                tracker,
                mag_lookup,
                current_filter_by_window,
                cad_on,
                cad_tgt,
                cad_sig,
                cad_wt,
                cad_first,
            )
            if not candidates:
                if override_applied:
                    cfg.exposure_by_filter = original_exp
                continue

            pal = (
                cfg.palette_evening
                if window_label_out.startswith("evening")
                else cfg.palette_morning
            )
            rot = day_idx % max(cfg.palette_rotation_days, 1)
            pal_rot = pal[rot:] + pal[:rot]
            available = {c["first_filter"] for c in candidates}
            batch_order = [f for f in pal_rot if f in available]
            batch_order += [
                f
                for f in ["y", "z", "i", "r", "g", "u"]
                if f in available and f not in batch_order
            ]

            cap_s = window_caps.get(idx_w, 0.0)
            window_sum = 0.0
            prev = None
            internal_changes = 0
            window_filter_change_s = 0.0
            window_slew_times: List[float] = []
            window_airmasses: List[float] = []
            window_skymags: List[float] = []
            filters_used_set: set[str] = set()
            state = current_filter_by_window.get(idx_w)

            deferred: list[dict] = []

            # Collect rows only for this window (avoids scanning global history)
            pernight_rows_window: List[Dict] = []
            sequence_rows_window: List[Dict] = []

            # ---- NEW: per-window running clock (UTC) for true order ----
            current_time_utc = pd.Timestamp(win["start"]).tz_convert("UTC")
            order_in_window = 0

            def _attempt_schedule(t: dict, allow_defer: bool = True) -> bool:
                nonlocal window_sum, prev, internal_changes, window_filter_change_s
                nonlocal state, window_slew_times, window_airmasses, window_skymags
                nonlocal filters_used_set, current_time_utc, order_in_window, libid_counter
                window_state = {
                    "allow_defer": allow_defer,
                    "deferred": deferred,
                    "window_sum": window_sum,
                    "cap_s": cap_s,
                    "state": state,
                    "prev": prev,
                    "current_time_utc": current_time_utc,
                    "order_in_window": order_in_window,
                    "day": day,
                    "window_label_out": window_label_out,
                    "mag_lookup": mag_lookup,
                }
                scheduled, effects = _attempt_schedule_one(
                    t,
                    window_state,
                    cfg,
                    site,
                    tracker,
                    phot_cfg,
                    sky_cfg,
                    sky_provider,
                    writer,
                )
                if not scheduled:
                    return False
                window_sum += effects["time_used_s"]
                updates = effects["state_updates"]
                current_time_utc = updates["current_time_utc"]
                state = updates["state_filter"]
                prev = updates["prev_target"]
                order_in_window = updates["order_in_window"]
                current_filter_by_window[idx_w] = state
                filters_used_set.update(updates["filters_used_set_delta"])
                summary = updates["summary_updates"]
                swap_count_by_window[idx_w] = swap_count_by_window.get(
                    idx_w, 0
                ) + summary.get("swap_count_delta", 0)
                internal_changes += summary["internal_changes_delta"]
                window_filter_change_s += summary["window_filter_change_s_delta"]
                window_slew_times.append(summary["slew_time"])
                window_airmasses.append(summary["airmass"])
                window_skymags.extend(summary["sky_mags"])
                # Accumulate rows globally (for non-streaming) and per-window/day
                if not stream_per_sn:
                    pernight_rows.extend(effects["pernight_rows"])
                pernight_rows_window.extend(effects["pernight_rows"])
                day_pernight_rows.extend(effects["pernight_rows"])  # day batch
                if not stream_sequence:
                    sequence_rows.extend(effects["sequence_rows"])
                sequence_rows_window.extend(effects["sequence_rows"])
                day_sequence_rows.extend(effects["sequence_rows"])  # day batch
                if writer and effects["simlib_epochs"]:
                    libid_counter = _emit_simlib(
                        writer,
                        libid_counter,
                        t["Name"],
                        t["RA_deg"],
                        t["Dec_deg"],
                        effects["simlib_epochs"],
                    )
                return True

            for filt in batch_order:
                batch = [
                    c
                    for c in candidates
                    if c["first_filter"] == filt and c["Name"] not in seen_ids
                ]
                min_amort = cfg.filter_change_s / max(cfg.swap_amortize_min, 1)
                while batch:
                    time_left = float(cap_s - window_sum)
                    exp_s = float(cfg.exposure_by_filter.get(filt, 0.0))
                    # conservative per-visit wall time used for swap amortization
                    est_visit_s = (
                        max(cfg.inter_exposure_min_s, cfg.readout_s + exp_s)
                        + cfg.slew_small_time_s
                        + cfg.slew_settle_s
                    )
                    k_time = max(1, int(time_left // max(est_visit_s, 1.0)))
                    k = max(1, min(len(batch), k_time))
                    amortized_penalty = cfg.filter_change_s / k
                    # select next target based on filter-aware cost
                    costs: List[float] = []
                    for t in batch:
                        now_mjd_cost = Time(current_time_utc).mjd
                        gated_for_cost = [
                            f
                            for f in t.get("policy_allowed", t["allowed"])
                            if (not cad_on)
                            or tracker.cadence_gate(
                                t["Name"], f, now_mjd_cost, cad_tgt, cad_jit
                            )
                        ]
                        if gated_for_cost:

                            def _bonus_cost(f: str) -> float:
                                return tracker.compute_filter_bonus(
                                    t["Name"],
                                    f,
                                    now_mjd_cost,
                                    cad_tgt,
                                    cad_sig,
                                    cad_wt,
                                    cad_first,
                                    cfg.cosmo_weight_by_filter,
                                    cfg.color_target_pairs,
                                    cfg.color_window_days,
                                    cfg.color_alpha,
                                    cfg.first_epoch_color_boost,
                                )

                            first_tmp = (
                                t["first_filter"]
                                if t["first_filter"] in gated_for_cost
                                else max(gated_for_cost, key=_bonus_cost)
                            )
                        else:
                            first_tmp = None
                        sep = (
                            0.0
                            if prev is None
                            else great_circle_sep_deg(
                                prev["RA_deg"],
                                prev["Dec_deg"],
                                t["RA_deg"],
                                t["Dec_deg"],
                            )
                        )
                        cost = slew_time_seconds(
                            sep,
                            small_deg=cfg.slew_small_deg,
                            small_time=cfg.slew_small_time_s,
                            rate_deg_per_s=cfg.slew_rate_deg_per_s,
                            settle_s=cfg.slew_settle_s,
                        )
                        if first_tmp is None:
                            cost = 1e9
                        elif state is not None and state != first_tmp:
                            if swap_count_by_window.get(idx_w, 0) >= int(
                                getattr(cfg, "max_swaps_per_window", 999)
                            ):
                                cost = 1e9
                            else:
                                scale = 1.0
                                if (
                                    tracker.cosmology_boost(
                                        t["Name"],
                                        first_tmp,
                                        now_mjd_cost,
                                        cfg.color_target_pairs,
                                        cfg.color_window_days,
                                        cfg.color_alpha,
                                    )
                                    > 1.0
                                ):
                                    scale = cfg.swap_cost_scale_color
                                penalty = max(amortized_penalty, min_amort) * scale
                                cost += penalty
                        costs.append(cost)
                    j = int(np.argmin(costs))
                    t = batch.pop(j)
                    sn_id = t["Name"]
                    if sn_id in seen_ids:
                        continue
                    if _attempt_schedule(t):
                        seen_ids.add(sn_id)

            progress = True
            while progress and deferred and window_sum < cap_s:
                progress = False
                # iterate over a snapshot so we can remove items during iteration
                for t in list(deferred):
                    sn_id = t["Name"]
                    if sn_id in seen_ids:
                        deferred.remove(t)
                        continue
                    if _attempt_schedule(t, allow_defer=False):
                        deferred.remove(t)
                        seen_ids.add(sn_id)
                        progress = True
            used_filters_csv = ",".join(sorted(filters_used_set))
            win = windows[idx_w]
            # Build window-local rows without scanning full history
            pernight_rows_for_window = pernight_rows_window
            start_mjd = Time(win["start"]).mjd if win["start"] else 0.0
            end_mjd = Time(win["end"]).mjd if win["end"] else 0.0
            color_pairs = 0
            color_div = 0
            multi_visit = 0
            if pernight_rows_for_window:
                # Count per-SN blue/red visits using the rows scheduled in this window
                by_sn: dict[str, dict] = {}
                for r in pernight_rows_for_window:
                    name = r.get("SN") or r.get("Name")
                    if not name:
                        continue
                    filt = r.get("filter")
                    if not filt:
                        continue
                    d = by_sn.setdefault(name, {"blue": 0, "red": 0, "n": 0})
                    d["n"] += 1
                    if filt in tracker.BLUE:
                        d["blue"] += 1
                    elif filt in tracker.RED:
                        d["red"] += 1
                for name, d in by_sn.items():
                    if d["n"] >= 2:
                        multi_visit += 1
                    pairs = min(d["blue"], d["red"])
                    if pairs > 0:
                        color_div += 1
                        color_pairs += pairs
            pct_diff = 100.0 * color_div / max(multi_visit, 1)
            ws_summary = {
                "window_sum": window_sum,
                "swap_count": swap_count_by_window.get(idx_w, 0),
                "internal_changes": internal_changes,
                "window_filter_change_s": window_filter_change_s,
                "window_slew_times": window_slew_times,
                "window_airmasses": window_airmasses,
                "window_skymags": window_skymags,
                "used_filters_csv": used_filters_csv,
                "n_candidates": len(group),
                "quota_assigned": (
                    cap_diag.get("quota_evening")
                    if idx_w == evening_idx
                    else cap_diag.get("quota_morning") if idx_w == morning_idx else None
                ),
                "n_candidates_pre_cap": pre_counts.get(idx_w, 0),
                "n_candidates_post_cap": post_counts.get(idx_w, 0),
                "color_pairs": color_pairs,
                "color_diversity": color_div,
                "pct_sne_with_blue_red_pair": pct_diff,
            }
            cap_source = cap_source_by_window.get(idx_w, "none")
            nights_rows.append(
                _build_window_summary_row(
                    day.date().isoformat(),
                    window_label_out,
                    win,
                    idx_w,
                    ws_summary,
                    cap_s,
                    cap_source,
                    cfg,
                    site,
                    pernight_rows_for_window,
                )
            )
            if override_applied:
                cfg.exposure_by_filter = original_exp

        planned_today = [
            r for r in (pernight_rows if not stream_per_sn else day_pernight_rows)
            if r["date"] == day.date().isoformat()
        ]
        _log_day_status(
            day.date().isoformat(),
            len(subset),
            len(visible),
            len(planned_today),
            evening_start,
            evening_end,
            morning_start,
            morning_end,
            tz_local,
            verbose,
        )

        # Stream to disk at end of day to cap memory
        if stream_per_sn and day_pernight_rows:
            df_day = pd.DataFrame(day_pernight_rows)
            df_day.to_csv(
                pernight_path,
                mode="a",
                header=not per_sn_header_written,
                index=False,
            )
            per_sn_header_written = True
            per_sn_count += len(df_day)
            day_pernight_rows.clear()
        if stream_sequence and day_sequence_rows:
            df_seq_day = pd.DataFrame(day_sequence_rows)
            df_seq_day.to_csv(
                seq_path,
                mode="a",
                header=not seq_header_written,
                index=False,
            )
            seq_header_written = True
            seq_count += len(df_seq_day)
            day_sequence_rows.clear()

    pernight_df = pd.DataFrame(pernight_rows) if not stream_per_sn else pd.DataFrame()
    nights_df = pd.DataFrame(nights_rows)
    # Write final outputs
    if not stream_per_sn:
        pernight_df.to_csv(pernight_path, index=False)
    nights_df.to_csv(nights_path, index=False)
    seq_df = pd.DataFrame(sequence_rows) if not stream_sequence else pd.DataFrame()
    if not seq_df.empty:
        seq_df.to_csv(seq_path, index=False)
    if writer:
        writer.close()
    print("Wrote:")
    print(f"  {pernight_path}")
    print(f"  {nights_path}")
    if (not seq_df.empty) or seq_header_written:
        print(f"  true-sequence: {seq_path}")
    label_hint = f"{label}_"
    print(
        "NOTE: per-SN plan CSV is best-in-theory (keyed to best_time_utc), may overlap.\n"
        f"      Use lsst_twilight_sequence_true_{label_hint}*.csv for the serialized, on-sky order."
    )
    rows_per_sn = len(pernight_df) if not stream_per_sn else per_sn_count
    print(f"Rows: per-SN={rows_per_sn}, nights*windows={len(nights_df)}")
    return pernight_df, nights_df
