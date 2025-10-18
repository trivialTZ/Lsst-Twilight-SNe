"""Single-target execution inside a twilight window.

Note: cfg.current_* side effects are intentionally preserved for performance
and to avoid behavior drift. Several downstream helpers read these fields.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from ..astro_utils import (
    airmass_from_alt_deg,
    choose_filters_with_cap,
    great_circle_sep_deg,
)
from ..config import PlannerConfig
from ..photom_rubin import PhotomConfig, compute_epoch_photom
from ..priority import PriorityTracker
from ..sky_model import SkyModelConfig, sky_mag_arcsec2


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
    single_filter_mode: bool = bool(window_state.get("single_filter_mode", False))
    relax_cadence: bool = window_state.get("relax_cadence", False)
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
    # Persist current target coordinates for downstream sky model usage
    cfg.current_ra_deg = float(t["RA_deg"]) if pd.notna(t.get("RA_deg", None)) else None
    cfg.current_dec_deg = float(t["Dec_deg"]) if pd.notna(t.get("Dec_deg", None)) else None
    cfg.current_mjd = (
        Time(t["best_time_utc"]).mjd
        if isinstance(t["best_time_utc"], (datetime, pd.Timestamp))
        else None
    )
    cfg.current_redshift = (
        float(t.get("redshift")) if t.get("redshift") is not None else None
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
    # Per-target cadence override for low-z Ia (smaller gate allowed)
    cad_tgt_eff = cad_tgt
    try:
        from .selection import _is_low_z_ia  # local import to avoid cycles

        if _is_low_z_ia(t.get("sn_type"), t.get("redshift"), cfg):
            override = getattr(cfg, "low_z_ia_cadence_days_target", None)
            if isinstance(override, (int, float)) and override > 0.0:
                cad_tgt_eff = float(override)
    except Exception:
        pass
    # Always execute single-filter visits (no automatic color pairing).
    cap_per_visit = 1

    if cad_on:
        # Re-evaluate cadence at execution time (see detailed comments in original).
        if relax_cadence:
            gated = list(t["policy_allowed"])  # ignore cadence gate in relaxed mode
        else:
            gated = [
                f
                for f in t.get("policy_allowed", t["allowed"])
                if tracker.cadence_gate(t["Name"], f, now_mjd, cad_tgt_eff, cad_jit)
            ]
        if not gated:
            if allow_defer:
                deferred.append(t)
            return False, {"failure_reason": "cadence_gate_reject"}
    else:
        gated = list(t.get("policy_allowed", t["allowed"]))

    if state is not None:
        first_tmp = state if state in gated else None
    else:
        s = list(reversed(cfg.filters))
        first_tmp = next((f for f in s if f in gated), None)
    if first_tmp is None:
        if allow_defer:
            deferred.append(t)
        return False, {"failure_reason": "no_entry_filter"}

    single_filter_mode = bool(single_filter_mode)
    if single_filter_mode:
        filters_pref = [first_tmp]
    else:
        # Prefer the best-first filter when not single-filtering.
        def _bonus(f: str) -> float:
            return tracker.cosmology_boost(
                t["Name"],
                f,
                now_mjd,
                cfg.color_target_pairs,
                cfg.color_window_days,
                cfg.color_alpha,
            )

        first = t["first_filter"]
        if first in gated:
            filters_pref = [first]
        else:
            filters_pref = [max(gated, key=_bonus)]

    effective_current_filter = first_tmp if single_filter_mode else state
    filters_used, timing = choose_filters_with_cap(
        filters_pref,
        sep,
        cfg.per_sn_cap_s,
        cfg,
        current_filter=effective_current_filter,
        filters_per_visit_cap=cap_per_visit,
    )
    if not filters_used:
        return False, {"failure_reason": "no_viable_filter"}
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
        max(timing["slew_s"], cfg.readout_s) + timing.get("filter_changes_s", 0.0)
    )
    total_with_guard = elapsed_overhead + timing["exposure_s"] + guard_s
    if window_sum + total_with_guard > cap_s:
        return False, {"failure_reason": "capacity_exhausted"}

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
        "total_time_s": round(timing["total_s"], 2),
        "slew_s": round(timing["slew_s"], 2),
        "readout_s": round(timing["readout_s"], 2),
        "filter_changes_s": round(timing["filter_changes_s"], 2),
        "cross_filter_change_s": round(timing.get("cross_filter_change_s", 0.0), 2),
        "guard_s": round(timing["guard_s"], 2),
        "elapsed_overhead_s": round(timing.get("elapsed_overhead_s", 0.0), 2),
    }

    pernight_rows: list[dict] = []
    epochs: list[dict] = []
    sky_mags: list[float] = []
    # Altitude at the actual visit start time
    try:
        sc_now = SkyCoord(float(t["RA_deg"]) * u.deg, float(t["Dec_deg"]) * u.deg, frame="icrs")
        alt_now_deg = float(
            sc_now.transform_to(AltAz(obstime=Time(sn_start_utc), location=site)).alt.deg
        )
    except Exception:
        alt_now_deg = float(t.get("max_alt_deg", np.nan))
    air = airmass_from_alt_deg(alt_now_deg)
    # Update current context for any downstream consumers
    cfg.current_alt_deg = alt_now_deg
    cfg.current_ra_deg = float(t["RA_deg"]) if pd.notna(t.get("RA_deg", None)) else None
    cfg.current_dec_deg = float(t["Dec_deg"]) if pd.notna(t.get("Dec_deg", None)) else None
    cfg.current_mjd = visit_mjd
    for f in filters_used:
        exp_s = timing.get("exp_times", {}).get(f, cfg.exposure_by_filter.get(f, 0.0))
        flags = timing.get("flags_by_filter", {}).get(f, set())
        mjd = visit_mjd
        if sky_provider:
            sky_mag = sky_provider.sky_mag(mjd, t["RA_deg"], t["Dec_deg"], f, air)
        else:
            sky_mag = sky_mag_arcsec2(f, sky_cfg)
        eph = compute_epoch_photom(f, exp_s, alt_now_deg, sky_mag, phot_cfg)
        sky_mags.append(sky_mag)
        if writer_or_none:
            epochs.append(
                {
                    "mjd": mjd,
                    "band": f,
                    "gain": eph.GAIN,
                    "noise": eph.RDNOISE,
                    "skysig": eph.SKYSIG,
                    "psf1": eph.PSF1_pix,
                    "psf2": eph.PSF2_pix,
                    "psfratio": eph.PSFRATIO,
                    "zpavg": eph.ZPTAVG,
                    "zperr": eph.ZPTERR,
                    "mag": -99.0,
                    "nexpose": 1,
                }
            )
        start_utc = sn_start_utc
        total_s = round(timing["total_s"], 2)
        end_utc = sn_end_utc
        days_since = tracker.days_since(t["Name"], f, visit_mjd)
        gate_passed = (
            tracker.cadence_gate(t["Name"], f, visit_mjd, cad_tgt_eff, cad_jit)
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
            "alt_deg": round(alt_now_deg, 2),
            "sky_mag_arcsec2": round(sky_mag, 2),
            "moon_sep": round(float(t.get("moon_sep", np.nan)), 2),
            "ZPT": round(eph.ZPTAVG, 3),
            "SKYSIG": round(eph.SKYSIG, 3),
            "NEA_pix": round(eph.NEA_pix, 2),
            "RDNOISE": round(eph.RDNOISE, 2),
            "GAIN": round(eph.GAIN, 2),
            "PSF1_pix": round(eph.PSF1_pix, 3),
            "PSF2_pix": round(eph.PSF2_pix, 3),
            "PSFRATIO": round(eph.PSFRATIO, 3),
            "saturation_guard_applied": "sat_guard" in flags,
            "warn_nonlinear": "warn_nonlinear" in flags,
            "priority_score": round(float(t["priority_score"]), 2),
            "cadence_days_since": (
                round(days_since, 3) if days_since is not None else np.nan
            ),
            "cadence_target_d": cad_tgt_eff,
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
    if not single_filter_mode:
        if state is not None and filters_used:
            if state != filters_used[0]:
                swap_count_delta = 1
    if filters_used:
        state = filters_used[-1]

    summary_updates = {
        "window_filter_change_s_delta": timing.get("filter_changes_s", 0.0),
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

