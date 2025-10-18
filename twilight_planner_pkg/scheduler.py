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

import itertools
import math
import warnings
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

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
    _local_timezone_from_location,
    airmass_from_alt_deg,
    choose_filters_with_cap,
    compute_capped_exptime,
    parse_sn_type_to_window_days,
    precompute_window_ephemerides,
    slew_time_seconds,
)
from .config import PlannerConfig
from .constraints import effective_min_sep
from .filter_policy import _m5_scale
from .io_utils import standardize_columns
from .photom_rubin import PhotomConfig, compute_epoch_photom
from .priority import PriorityTracker
from .simlib_writer import SimlibHeader
from .sky_model import (
    RubinSkyProvider,
    SimpleSkyProvider,
    SkyModelConfig,
    sky_mag_arcsec2,
)
from .twilight_sched.caching import (  # noqa: F401
    _CACHE_TOKEN,
    _SKY_CACHE,
    _M5_CACHE,
    _cached_sky_mag,
    _cached_m5_at_time,
    _finite_or_none,
    _m5best_cache_prune_by_day,
    bump_window_token,
    prune_m5best_by_day,
)
from .twilight_sched.scoring_pairs import (  # noqa: F401
    _candidate_coord,
    _compute_capped_exptime_for_pair,
    _estimate_m5_for_pair,
    _pair_score,
    _build_global_pairs_for_window,
)
from .twilight_sched.dp_planner import (
    _plan_batches_by_dp,
    _generate_filter_sequences,
    _prefix_scores,
)
from .twilight_sched.selection import _policy_filters_mid, _is_low_z_ia, get_sun_alt_deg_cached
from .twilight_sched.cost import (
    _visit_unit_time,
    _residual_slew_cost,
    _opportunity_cost_seconds,
    has_room_for_any_visit,
    min_exposure_in_palette,
    opportunity_cost_seconds_cached,
)
from .twilight_sched.executor import _attempt_schedule_one
from .twilight_sched.summaries import (
    _fmt_window,
    _fmt_window_local,
    _mid_sun_alt_of_window,
    _path_plan,
    _path_summary,
    _path_sequence,
    _log_day_status,
    _build_window_summary_row,
    create_simlib_writer,
)
from .twilight_sched.types import PairItem

# --- Re-export refactored internals so tests can import them from this module ---
__all__ = [
    "plan_twilight_range_with_caps",
    "_log_day_status",
    "_fmt_window",
    "_fmt_window_local",
    "_mid_sun_alt_of_window",
    "_prefix_scores",
    "_generate_filter_sequences",
    "_plan_batches_by_dp",
    "_build_global_pairs_for_window",
    "_prepare_window_candidates",
    "_policy_filters_mid",
    "_is_low_z_ia",
    "_visit_unit_time",
    "_residual_slew_cost",
    "_CACHE_TOKEN",
    "_SKY_CACHE",
    "_M5_CACHE",
    "PairItem",
    "_candidate_coord",
    "_compute_capped_exptime_for_pair",
    "_estimate_m5_for_pair",
    "_pair_score",
    "_build_window_summary_row",
    "_path_plan",
    "_path_summary",
    "_path_sequence",
    "_cached_sky_mag",
    "_cached_m5_at_time",
    "_finite_or_none",
    "_m5best_cache_prune_by_day",
    "prune_m5best_by_day",
    "bump_window_token",
    "_opportunity_cost_seconds",
]

# --- Monkeypatchable symbols (re-exported) ---
from .astro_utils import (
    twilight_windows_for_local_night as twilight_windows_for_local_night,
    great_circle_sep_deg as great_circle_sep_deg,
)
try:
    # Prefer the public alias when available to preserve monkeypatching surface.
    from .astro_utils import best_time_with_moon as _best_time_with_moon  # type: ignore
except Exception:
    from .astro_utils import _best_time_with_moon as _best_time_with_moon  # type: ignore
from .filter_policy import (
    allowed_filters_for_window as allowed_filters_for_window,
)
from .astro_utils import (
    allowed_filters_for_sun_alt as allowed_filters_for_sun_alt,
    pick_first_filter_for_target as pick_first_filter_for_target,
)

__all__ += [
    "twilight_windows_for_local_night",
    "_best_time_with_moon",
    "great_circle_sep_deg",
    "allowed_filters_for_window",
    "allowed_filters_for_sun_alt",
    "pick_first_filter_for_target",
]

PER_SN_COLUMNS = [
    "date",
    "twilight_window",
    "SN",
    "RA_deg",
    "Dec_deg",
    "best_twilight_time_utc",
    "visit_start_utc",
    "sn_end_utc",
    "filter",
    "t_exp_s",
    "guard_s",
    "airmass",
    "alt_deg",
    "sky_mag_arcsec2",
    "moon_sep",
    "ZPT",
    "SKYSIG",
    "NEA_pix",
    "RDNOISE",
    "GAIN",
    "PSF1_pix",
    "PSF2_pix",
    "PSFRATIO",
    "saturation_guard_applied",
    "warn_nonlinear",
    "priority_score",
    "cadence_days_since",
    "cadence_target_d",
    "cadence_gate_passed",
    "slew_s",
    "cross_filter_change_s",
    "filter_changes_s",
    "readout_s",
    "exposure_s",
    "inter_exposure_guard_enforced",
    "total_time_s",
    "elapsed_overhead_s",
]

warnings.filterwarnings("ignore", message=".*get_moon.*deprecated.*")
warnings.filterwarnings(
    "ignore", message=".*transforming other coordinates from <GCRS Frame.*>"
)
warnings.filterwarnings(
    "ignore",
    message="Angular separation can depend on the direction of the transformation",
)
warnings.filterwarnings(
    "ignore", message="Extrapolating twilight beyond a sun altitude of -11 degrees"
)

# Cache singletons and helpers moved to twilight_sched.caching

 



# moved to twilight_sched.scoring_pairs


# moved to twilight_sched.scoring_pairs


# moved to twilight_sched.scoring_pairs


# moved to twilight_sched.dp_planner


## moved to twilight_sched.summaries


## moved to twilight_sched.summaries


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
    redshift: float | None = None,
) -> int:
    """Write one SIMLIB block and return the incremented libid counter."""
    writer.start_libid(
        libid_counter,
        ra_deg,
        dec_deg,
        nobs=len(epochs),
        comment=name,
        redshift=redshift,
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
    phot_cfg: PhotomConfig,
    sky_provider,
    sky_cfg: SkyModelConfig,
) -> tuple[list[dict], bool]:
    """Build candidate target dicts for a twilight window.

    Returns the candidate list and a flag indicating whether an exposure
    override from ``cfg.sun_alt_exposure_ladder`` was applied.
    """
    # Compute mid-window Sun altitude using this module's get_sun so tests can
    # monkeypatch scheduler.get_sun deterministically.
    mid = window["start"] + (window["end"] - window["start"]) / 2
    try:
        mid_sun_alt = float(
            get_sun(Time(mid))
            .transform_to(AltAz(location=site, obstime=Time(mid)))
            .alt.to(u.deg)
            .value
        )
    except Exception:
        mid_sun_alt = _mid_sun_alt_of_window(window, site)
    current_time_utc = pd.Timestamp(window["start"]).tz_convert("UTC")
    # Respect an explicitly empty ladder (do not fallback), and only warn
    # if both ladders are configured (non-empty) to clarify precedence.
    ladder_sun = getattr(cfg, "sun_alt_exposure_ladder", None)
    ladder_exp = getattr(cfg, "exposure_time_ladder", None)
    if ladder_sun is None:
        ladder = ladder_exp
    else:
        ladder = ladder_sun
        # If both ladders are non-empty lists, warn which one takes precedence.
        try:
            import warnings as _warnings
            if (
                isinstance(ladder_sun, list)
                and len(ladder_sun) > 0
                and isinstance(ladder_exp, list)
                and len(ladder_exp) > 0
            ):
                _warnings.warn(
                    "Both sun_alt_exposure_ladder and exposure_time_ladder are set; using sun_alt_exposure_ladder.",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception:
            pass
    override_applied = False
    original_exp: Optional[dict] = None
    if ladder:
        for low, high, overrides in ladder:
            if low < mid_sun_alt <= high:
                original_exp = dict(cfg.exposure_by_filter)
                new_exp = dict(cfg.exposure_by_filter)
                new_exp.update(overrides)
                cfg.exposure_by_filter = new_exp
                override_applied = True
                break

    candidates: list[dict] = []
    for _, row in group_df.iterrows():
        # Optionally evaluate policy at each target's best time instead of the window midpoint.
        sun_alt_for_policy = mid_sun_alt
        best_time_mjd = None
        if getattr(cfg, "filter_policy_use_best_time_alt", False) and pd.notna(
            row.get("best_time_utc")
        ):
            try:
                t_best = row["best_time_utc"]
                sun_alt_for_policy = get_sun_alt_deg_cached(
                    t_best,
                    site,
                    step_minutes=int(getattr(cfg, "policy_sun_alt_minutes", 1) or 1),
                )
                t_best_ts = (
                    pd.Timestamp(t_best).tz_convert("UTC")
                    if isinstance(t_best, pd.Timestamp)
                    else pd.Timestamp(t_best).tz_localize("UTC")
                )
                best_time_mjd = float(Time(t_best_ts.to_pydatetime()).mjd)
            except Exception:
                sun_alt_for_policy = mid_sun_alt
                best_time_mjd = None
        if best_time_mjd is None:
            try:
                bt = row.get("best_time_utc")
                if pd.notna(bt):
                    bt_ts = (
                        pd.Timestamp(bt).tz_convert("UTC")
                        if isinstance(bt, pd.Timestamp)
                        else pd.Timestamp(bt).tz_localize("UTC")
                    )
                    best_time_mjd = float(Time(bt_ts.to_pydatetime()).mjd)
            except Exception:
                best_time_mjd = None

        ra_val = row.get("RA_deg")
        ra_float = float(ra_val) if pd.notna(ra_val) else None
        dec_val = row.get("Dec_deg")
        dec_float = float(dec_val) if pd.notna(dec_val) else None

        allowed = allowed_filters_for_window(
            mag_lookup.get(row["Name"], {}),
            sun_alt_for_policy,
            row["_moon_alt"],
            row["_moon_phase"],
            row["_moon_sep"],
            airmass_from_alt_deg(row["max_alt_deg"]),
            phot_cfg.fwhm_eff.get("r", 0.83) if phot_cfg.fwhm_eff else 0.83,
            exposure_by_filter=cfg.exposure_by_filter,
            phot_cfg=phot_cfg,
            sky_provider=sky_provider,
            sky_cfg=sky_cfg,
            mjd=Time(current_time_utc).mjd,
            ra_deg=ra_float,
            dec_deg=dec_float,
            sky_lookup=_cached_sky_mag,
        )
        allowed = [f for f in allowed if f in cfg.filters]
        # Apply sun-alt policy via monkeypatchable alias; if intersection is empty,
        # fall back to feasibility-allowed filters to avoid discarding the target.
        if getattr(cfg, "sun_alt_policy", None):
            pol = allowed_filters_for_sun_alt(sun_alt_for_policy, cfg)
        else:
            pol = list(cfg.filters)
        policy_intersection = [f for f in allowed if f in pol]
        allowed = policy_intersection if policy_intersection else list(allowed)
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
        now_mjd_for_bonus = Time(current_time_utc).mjd
        first = pick_first_filter_for_target(
            row["Name"],
            row.get("SN_type_raw"),
            tracker,
            policy_allowed,
            cfg,
            sun_alt_deg=sun_alt_for_policy,
            moon_sep_ok=moon_sep_ok,
            current_mag=mag_lookup.get(row["Name"]),
            current_filter=current_filter_by_window.get(idx_w),
            now_mjd=now_mjd_for_bonus,
        )
        if first is None:
            continue
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
                    diversity_enable=getattr(cfg, "diversity_enable", False),
                    diversity_target_per_filter=getattr(cfg, "diversity_target_per_filter", 1),
                    diversity_window_days=getattr(cfg, "diversity_window_days", 5.0),
                    diversity_alpha=getattr(cfg, "diversity_alpha", 0.3),
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
            "best_time_mjd": best_time_mjd,
            "max_alt_deg": row["max_alt_deg"],
            "priority_score": row["priority_score"],
            "redshift": (
                float(row.get("redshift"))
                if pd.notna(row.get("redshift", np.nan))
                else None
            ),
            "first_filter": first,
            "sn_type": row.get("SN_type_raw"),
            "allowed": [first] + rest,
            "policy_allowed": policy_allowed,
            "moon_sep_ok": moon_sep_ok,
            "moon_sep": float(row["_moon_sep"]),
            "moon_alt": float(row["_moon_alt"]),
            "moon_phase": float(row["_moon_phase"]),
            "sun_alt_policy": float(sun_alt_for_policy),
            "airmass": float(airmass_from_alt_deg(row["max_alt_deg"])),
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
    # When forced (allow_defer=False upstream), evaluate cadence at the later of
    # current time and the target's preferred best_time_utc so eligible items
    # can be promoted once their window-appropriate time is reached.
    now_mjd_for_cad = now_mjd
    try:
        # Determine if upstream requested forced scheduling by presence in window_state;
        # here we infer via allow_defer flag available in closure.
        forced_mode = not allow_defer
    except Exception:
        forced_mode = False
    if forced_mode and pd.notna(t.get("best_time_utc")):
        try:
            bt = t["best_time_utc"]
            bt_ts = (
                pd.Timestamp(bt).tz_convert("UTC")
                if isinstance(bt, pd.Timestamp)
                else pd.Timestamp(bt).tz_localize("UTC")
            )
            bt_mjd = float(Time(bt_ts.to_pydatetime()).mjd)
            if bt_mjd > now_mjd_for_cad:
                now_mjd_for_cad = bt_mjd
        except Exception:
            pass
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
        if _is_low_z_ia(t.get("sn_type"), t.get("redshift"), cfg):
            override = getattr(cfg, "low_z_ia_cadence_days_target", None)
            if isinstance(override, (int, float)) and override > 0.0:
                cad_tgt_eff = float(override)
    except Exception:
        pass
    # Allow multi-filter visits up to configured cap
    try:
        cap_per_visit = int(getattr(cfg, "filters_per_visit_cap", 1))
    except Exception:
        cap_per_visit = 1
    cap_per_visit = max(1, cap_per_visit)

    if cad_on:
        # Re-evaluate cadence at execution time:
        # - The DP step used a snapshot of time when building pairs, but actual
        #   execution time (now_mjd) advances as we schedule/defer items.
        # - Cadence eligibility depends on the actual observation time and the
        #   last-visit history per filter, so a target can flip from gated-out
        #   to permitted (or vice versa) between planning and execution.
        # - In relaxed mode we intentionally ignore cadence gating to utilize
        #   leftover time near the end of a window/backfill, trading strict
        #   cadence for increased utilization.
        if relax_cadence:
            gated = list(t["policy_allowed"])  # ignore cadence gate in relaxed mode
        else:
            gated = [
                f
                for f in t["policy_allowed"]
                if tracker.cadence_gate(t["Name"], f, now_mjd_for_cad, cad_tgt_eff, cad_jit)
            ]
            if not gated:
                if allow_defer:
                    deferred.append(t)
                return False, {"failure_reason": "cadence_gate_failed"}

        def _bonus(f: str) -> float:
            return tracker.compute_filter_bonus(
                t["Name"],
                f,
                now_mjd_for_cad,
                cad_tgt_eff,
                cad_sig,
                cad_wt,
                cad_first,
                cfg.cosmo_weight_by_filter,
                cfg.color_target_pairs,
                cfg.color_window_days,
                cfg.color_alpha,
                cfg.first_epoch_color_boost,
                diversity_enable=getattr(cfg, "diversity_enable", False),
                diversity_target_per_filter=getattr(cfg, "diversity_target_per_filter", 1),
                diversity_window_days=getattr(cfg, "diversity_window_days", 5.0),
                diversity_alpha=getattr(cfg, "diversity_alpha", 0.3),
            )

        first = (
            t["first_filter"] if t["first_filter"] in gated else max(gated, key=_bonus)
        )
        rest = [f for f in gated if f != first]
        filters_pref = [first] + rest
    else:
        first = t["first_filter"]
        rest = [f for f in t.get("policy_allowed", []) if f != first]
        filters_pref = [first] + rest
    # When single_filter_mode is active we assume the carousel is already at the
    # desired filter for the visit (no cross-filter change at segment entry), so
    # we pass current_filter as the target's first_filter to suppress cross cost.
    effective_current_filter = (
        t["first_filter"] if single_filter_mode else state
    )
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
    # filter_changes_s already includes cross + internal; don't add cross again
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
    # Do not count swaps for primary DP batches executing in single-filter mode
    # (we assume the carousel is already at the requested filter for the batch).
    if not single_filter_mode:
        if state is not None and filters_used:
            if state != filters_used[0]:
                swap_count_delta = 1
    if filters_used:
        state = filters_used[-1]

    # Accumulate total filter-change time once (cross + internal)
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
        "dp_time_s": round(float(ws_summary.get("dp_time_s", 0.0)), 1),
        "backfill_time_s": round(float(ws_summary.get("backfill_time_s", 0.0)), 1),
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
    # Opt-in lightweight timing (prints only if cfg.debug_timing=True)
    try:
        import time as _time  # local alias
    except Exception:  # pragma: no cover
        _time = None  # type: ignore
    timing_enabled = bool(getattr(cfg, "debug_timing", False)) and (_time is not None)
    class _Timer:
        def __init__(self, enabled: bool):
            self.enabled = enabled
            self.last = _time.monotonic() if _time else 0.0
        def mark(self, label: str):
            if not self.enabled or _time is None:
                return
            now = _time.monotonic()
            dt = now - self.last
            print(f"[timing] {label}: {dt:.3f}s")
            self.last = now
        def reset(self):
            if _time is not None:
                self.last = _time.monotonic()
    _t = _Timer(timing_enabled)
    raw = pd.read_csv(csv_path)
    _t.mark("read_csv")
    df = standardize_columns(raw, cfg)
    _t.mark("standardize_columns")
    # Optional: restrict to Ia-like types
    try:
        if getattr(cfg, "only_ia", False):
            types_norm = df["SN_type_raw"].astype(str).str.lower()
            df = df[types_norm.str.contains("ia", na=False)].copy()
    except Exception:
        pass
    # Build per-target per-band magnitudes with discovery fallback so that
    # saturation capping can use source brightness even when band mags are
    # missing in the input catalog.
    from .io_utils import build_mag_lookup_with_fallback

    mag_lookup = build_mag_lookup_with_fallback(df, cfg)
    _t.mark("build_mag_lookup_with_fallback")

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
        sky_provider = RubinSkyProvider(site=site)
    except Exception:
        sky_provider = SimpleSkyProvider(sky_cfg, site=site)
    cfg.sky_provider = sky_provider
    _t.mark("init_sky_provider")

    # Opportunity cost helper now in twilight_sched.cost (identical math)

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
        writer = create_simlib_writer(cfg, hdr)
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
    pernight_path = _path_plan(
        outdir,
        pd.to_datetime(start_date, utc=True),
        pd.to_datetime(end_date, utc=True),
        label,
    )
    nights_path = _path_summary(
        outdir,
        pd.to_datetime(start_date, utc=True),
        pd.to_datetime(end_date, utc=True),
        label,
    )
    seq_path = _path_sequence(
        outdir,
        pd.to_datetime(start_date, utc=True),
        pd.to_datetime(end_date, utc=True),
        label,
    )

    pernight_rows: List[Dict] = (
        [] if not stream_per_sn else []
    )  # day-batched if streaming
    nights_rows: List[Dict] = []
    # ---- NEW: true sequential execution order collector (one row per SN visit) ----
    sequence_rows: list[dict] = (
        [] if not stream_sequence else []
    )  # day-batched if streaming
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
        _t.reset()
        if timing_enabled:
            print(f"[timing] === {day.date()} start ===")
        # day-batched collectors (used when streaming)
        day_pernight_rows: List[Dict] = []
        day_sequence_rows: List[Dict] = []
        seen_ids: set[str] = set()
        # Optional finite nightly cap enforced during scheduling for backfill
        nightly_cap: int | None = None
        _lim = getattr(cfg, "max_sn_per_night", None)
        try:
            _lim_f = float(_lim) if _lim is not None else float("inf")
        except Exception:
            _lim_f = float("inf")
        if math.isfinite(_lim_f):
            nightly_cap = int(_lim_f)
        # Here 'day' is interpreted as *local* civil date of the evening block
        windows = twilight_windows_for_local_night(
            day.date(),
            site,
            cfg.twilight_sun_alt_min_deg,
            cfg.twilight_sun_alt_max_deg,
        )
        _t.mark("twilight_windows_for_local_night")
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
        window_usage_for_log: dict[str, dict] = {}
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
        has_disc = df["discovery_datetime"].notna()
        if getattr(cfg, "limit_by_typical_lifetime", False):
            if "typical_lifetime_days" in df.columns:
                lifetime_days_each = df["typical_lifetime_days"]
            else:
                lifetime_days_each = df["SN_type_raw"].apply(
                    lambda t: parse_sn_type_to_window_days(t, cfg)
                )
            min_allowed_disc_each = cutoff - lifetime_days_each.apply(
                lambda d: timedelta(days=int(d))
            )
            subset = df[
                has_disc
                & (df["discovery_datetime"] <= cutoff)
                & (df["discovery_datetime"] >= min_allowed_disc_each)
            ].copy()
        else:
            # Be permissive by default: anything discovered up to the cutoff is eligible.
            subset = df[has_disc & (df["discovery_datetime"] <= cutoff)].copy()
        _t.mark("discovery_window_filtering")
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
        _t.mark("precompute_window_ephemerides")
        for _, row in subset.iterrows():
            sc = SkyCoord(row["RA_deg"] * u.deg, row["Dec_deg"] * u.deg, frame="icrs")
            max_alt, max_time, max_idx = -999.0, None, None
            best_moon = (float("nan"), float("nan"), float("nan"))
            for idx_w, w in labeled:
                # Call _best_time_with_moon with backward-compatibility for tests
                # that monkeypatch a version without the 'precomputed' kwarg.
                try:
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
                except TypeError:
                    alt_deg, t_utc, moon_alt_deg, moon_phase, moon_sep_deg = (
                        _best_time_with_moon(
                            sc,
                            (w["start"], w["end"]),
                            site,
                            cfg.twilight_step_min,
                            cfg.min_alt_deg,
                            req_sep,
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
        _t.mark("best_time_scan_with_moon")

        visible = subset[
            (subset["best_time_utc"].notna())
            & (subset["max_alt_deg"] >= cfg.min_alt_deg)
            & (subset["best_window_index"] >= 0)
        ].copy()
        _t.mark("visible_subset")
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
        _t.mark("priority_scoring_base")
        # Optional low-redshift boost for tie-breaking under the default (hybrid) strategy
        if (
            getattr(cfg, "redshift_boost_enable", True)
            and str(cfg.priority_strategy).lower() == "hybrid"
            and "redshift" in visible.columns
        ):
            z_ref = float(getattr(cfg, "redshift_low_ref", 0.1))
            z_max = float(getattr(cfg, "redshift_boost_max", 1.2))

            def _apply_zboost(row):
                base = float(row["priority_score"])
                if base <= 0.0:
                    return base
                zval = row.get("redshift")
                boost = tracker.redshift_boost(zval, z_ref=z_ref, max_boost=z_max)
                return float(base * boost)

            visible["priority_score"] = visible.apply(_apply_zboost, axis=1)
            _t.mark("priority_scoring_zboost")
        # Stronger low-z Ia multiplier (configurable; disabled if =1.0)
        if (
            float(getattr(cfg, "low_z_ia_priority_multiplier", 1.0)) > 1.0
            and "redshift" in visible.columns
        ):
            mult = float(getattr(cfg, "low_z_ia_priority_multiplier", 1.0))

            def _apply_lowz_ia(row):
                base = float(row["priority_score"])
                if base <= 0.0:
                    return base
                if _is_low_z_ia(row.get("SN_type_raw"), row.get("redshift"), cfg):
                    return float(base * mult)
                return base

            visible["priority_score"] = visible.apply(_apply_lowz_ia, axis=1)
            _t.mark("priority_scoring_lowz_ia")
        if cfg.priority_strategy == "unique_first":
            thr = float(getattr(cfg, "unique_first_drop_threshold", 0.0))
            visible = visible[visible["priority_score"] > thr].copy()
        visible.sort_values(
            ["priority_score", "max_alt_deg"], ascending=[False, False], inplace=True
        )
        _t.mark("sort_visible")
        pre_counts = (
            visible.groupby("best_window_index").size().to_dict()
            if "best_window_index" in visible.columns
            else {}
        )
        top_global, cap_diag = _cap_candidates_per_window(
            visible, cfg, evening_cap_s_val, morning_cap_s_val
        )
        _t.mark("cap_candidates_per_window")
        post_counts = (
            top_global.groupby("best_window_index").size().to_dict()
            if "best_window_index" in top_global.columns
            else {}
        )
        # Build a backfill pool: visible candidates not in the capped set
        # These are considered later if deferrals or cadence gates leave
        # unused time in a window.
        backfill_pool = visible[~visible.index.isin(top_global.index)].copy()
        # Consider any window that has either primary or backfill candidates
        _win_idxs_primary = set(top_global["best_window_index"].values)
        _win_idxs_backfill = set(backfill_pool["best_window_index"].values)
        for idx_w in sorted(_win_idxs_primary | _win_idxs_backfill):
            group = top_global[top_global["best_window_index"] == idx_w].copy()
            # Skip unlabeled (previous/next day) twilight windows entirely
            if window_labels.get(idx_w) is None:
                continue
            win = windows[idx_w]
            win_start_utc = pd.Timestamp(win["start"]).tz_convert("UTC")
            win_start_mjd = float(Time(win_start_utc.to_pydatetime()).mjd)
            bump_window_token()
            prune_m5best_by_day(cfg, win_start_mjd)
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
            # Primary candidates (subject to nightly cap pre-selection)
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
                phot_cfg,
                cfg.sky_provider,
                sky_cfg,
            )
            _t.mark(f"prepare_window_candidates[{window_label_out}]")
            # Backfill pool for this window (lower-priority visibles)
            backfill_group = backfill_pool[
                backfill_pool["best_window_index"] == idx_w
            ].copy()
            backfill_candidates: list[dict] = []
            if not backfill_group.empty:
                bfill_list, _ov2 = _prepare_window_candidates(
                    backfill_group,
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
                    phot_cfg,
                    cfg.sky_provider,
                    sky_cfg,
                )
                backfill_candidates = bfill_list
                _t.mark(f"prepare_backfill_candidates[{window_label_out}]")
            # If both are empty, skip this window
            if not candidates and not backfill_candidates:
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
            # Optional per-window first-filter cycle override
            first_override: str | None = None
            if getattr(cfg, "first_filter_cycle_enable", False):
                if window_label_out.startswith("morning"):
                    cyc = list(getattr(cfg, "first_filter_cycle_morning", []))
                    if cyc:
                        first_override = cyc[day_idx % len(cyc)]
                else:
                    cyc = list(getattr(cfg, "first_filter_cycle_evening", []))
                    if cyc:
                        first_override = cyc[day_idx % len(cyc)]
            assignments: dict[str, str] = {}
            dp_filter_sequence: list[str] | None = None
            dp_counts: list[int] | None = None
            # Early debug flag so we can use it while building DP pairs/plan
            _dbg_enabled = bool(getattr(cfg, "debug_planner", False))
            if candidates:
                cadence_params_base = {
                    "cad_on": cad_on,
                    "cad_tgt": cad_tgt,
                    "cad_sig": cad_sig,
                    "cad_wt": cad_wt,
                    "cad_first": cad_first,
                    "cad_jit": cad_jit,
                    "diversity_enable": getattr(cfg, "diversity_enable", False),
                    "diversity_target": getattr(cfg, "diversity_target_per_filter", 1),
                    "diversity_window": getattr(cfg, "diversity_window_days", 5.0),
                    "diversity_alpha": getattr(cfg, "diversity_alpha", 0.3),
                }
                now_mjd_window = Time(
                    pd.Timestamp(win["start"]).tz_convert("UTC")
                ).mjd
                per_filter_pairs = _build_global_pairs_for_window(
                    candidates,
                    cfg,
                    tracker,
                    now_mjd_window,
                    None,
                    mag_lookup,
                    cadence_params_base.copy(),
                    phot_cfg,
                    sky_cfg,
                    cfg.sky_provider,
                )
                _t.mark(f"build_global_pairs[{window_label_out}]")
                if _dbg_enabled and per_filter_pairs:
                    _dbg_pairs_by_filter = {f: len(v) for f, v in per_filter_pairs.items()}
                if per_filter_pairs:
                    prefix_scores = _prefix_scores(per_filter_pairs)
                    forced_first = (
                        first_override if first_override in per_filter_pairs else None
                    )
                    plan_seq, plan_counts, _ = _plan_batches_by_dp(
                        per_filter_pairs,
                        prefix_scores,
                        window_caps.get(idx_w, 0.0),
                        cfg,
                        forced_first,
                    )
                    _t.mark(f"dp_plan_batches[{window_label_out}]")
                    if plan_seq and plan_counts:
                        # Safety clamp: if swaps are disallowed, execute only the first segment
                        # to respect max_swaps_per_window while still scheduling something.
                        try:
                            _limit_swaps = int(getattr(cfg, "max_swaps_per_window", 999))
                        except Exception:
                            _limit_swaps = 999
                        if _limit_swaps <= 0 and len(plan_seq) > 1:
                            plan_seq = plan_seq[:1]
                            plan_counts = plan_counts[:1]
                        dp_filter_sequence = list(plan_seq)
                        dp_counts = list(plan_counts)
                        if _dbg_enabled:
                            _dbg_dp_plan = list(zip(dp_filter_sequence, dp_counts))
                        assigned_per_filter: Counter = Counter()
                        for filt, take in zip(plan_seq, plan_counts):
                            if take <= 0:
                                continue
                            pool = per_filter_pairs.get(filt, [])
                            for item in pool:
                                if item.name in assignments:
                                    continue
                                assignments[item.name] = filt
                                assigned_per_filter[filt] += 1
                                if assigned_per_filter[filt] >= take:
                                    break
                        if assignments:
                            selected_candidates: list[dict] = []
                            deferred_candidates: list[dict] = []
                            for cand in candidates:
                                name = cand["Name"]
                                assigned = assignments.get(name)
                                if assigned:
                                    policy_allowed = list(
                                        dict.fromkeys(cand.get("policy_allowed", []))
                                    )
                                    reorder = [assigned] + [
                                        f for f in policy_allowed if f != assigned
                                    ]
                                    # Preserve any extra filters from 'allowed'
                                    extra_allowed = [
                                        f
                                        for f in cand.get("allowed", [])
                                        if f not in reorder
                                    ]
                                    cand["first_filter"] = assigned
                                    cand["policy_allowed"] = reorder
                                    cand["allowed"] = reorder + extra_allowed
                                    selected_candidates.append(cand)
                                else:
                                    deferred_candidates.append(cand)
                            if selected_candidates:
                                candidates = selected_candidates
                                if deferred_candidates:
                                    existing_names = {c["Name"] for c in backfill_candidates}
                                    for cand in deferred_candidates:
                                        if cand["Name"] not in existing_names:
                                            backfill_candidates.append(cand)
                                            existing_names.add(cand["Name"])
            if dp_filter_sequence:
                batch_order = dp_filter_sequence
            else:
                available = {c["first_filter"] for c in candidates}
                counts = Counter(c["first_filter"] for c in candidates)
                batch_order = sorted(
                    available,
                    key=lambda f: (
                        -counts.get(f, 0),
                        pal_rot.index(f) if f in pal_rot else 99,
                    ),
                )
                if first_override in batch_order:
                    batch_order.remove(first_override)
                    batch_order.insert(0, first_override)

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
            # Aggregate SIMLIB write time per window for diagnostics
            simlib_write_s: float = 0.0

            deferred: list[dict] = []
            # Allow a single extra carousel swap during backfill if otherwise
            # we'd leave time unused because no backfill candidates match the
            # current carousel filter. Keeps change minimal and scoped.
            backfill_extra_swap_used = False
            # Track per-window revisits: allow configurable repeats per SN
            # (default remains one repeat: primary + one extra color)
            repeat_counts: dict[str, int] = {}

            # Collect rows only for this window (avoids scanning global history)
            pernight_rows_window: List[Dict] = []
            sequence_rows_window: List[Dict] = []

            # ---- NEW: per-window running clock (UTC) for true order ----
            current_time_utc = pd.Timestamp(win["start"]).tz_convert("UTC")
            order_in_window = 0
            # ---- Accounting: track DP vs backfill time usage ----
            dp_time_used: float = 0.0
            backfill_time_used: float = 0.0
            current_phase: str = "primary"

            # Debug instrumentation (pairs, DP plan, execution counts)
            if _dbg_enabled:
                from collections import Counter as _Counter
                # Only initialize counters; keep any earlier pair/plan info
                try:
                    _dbg_pairs_by_filter  # type: ignore[name-defined]
                except NameError:
                    _dbg_pairs_by_filter = None  # type: ignore[assignment]
                try:
                    _dbg_dp_plan  # type: ignore[name-defined]
                except NameError:
                    _dbg_dp_plan = None  # type: ignore[assignment]
                _dbg_dp_sched_counts = _Counter()
                _dbg_dp_reject_reasons = _Counter()
                _dbg_backfill_sched_counts = _Counter()
                _dbg_repeat_sched_counts = _Counter()
                _dbg_relax_sched_counts = _Counter()
                _dbg_backfill_order = []

            def _attempt_schedule(
                t: dict,
                allow_defer: bool = True,
                *,
                single_filter_mode: bool | None = None,
                state_for_call: str | None = None,
            ) -> bool:
                nonlocal window_sum, prev, internal_changes, window_filter_change_s
                nonlocal state, window_slew_times, window_airmasses, window_skymags
                nonlocal filters_used_set, current_time_utc, order_in_window, libid_counter
                nonlocal simlib_write_s
                nonlocal dp_time_used, backfill_time_used, current_phase
                sfm = bool(single_filter_mode) if single_filter_mode is not None else False
                window_state = {
                    "allow_defer": allow_defer,
                    "deferred": deferred,
                    "window_sum": window_sum,
                    "cap_s": cap_s,
                    # In single-filter DP batch mode we treat the carousel as already
                    # positioned at the batch filter; pass that here to suppress any
                    # cross-filter change cost inside the batch.
                    "state": (state_for_call if sfm and state_for_call else state),
                    "prev": prev,
                    "current_time_utc": current_time_utc,
                    "order_in_window": order_in_window,
                    "day": day,
                    "window_label_out": window_label_out,
                    "mag_lookup": mag_lookup,
                    "single_filter_mode": sfm,
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
                    if _dbg_enabled:
                        reason = None
                        try:
                            reason = effects.get("failure_reason")
                        except Exception:
                            reason = None
                        if reason:
                            # Attribute to DP or backfill based on current phase
                            if current_phase == "primary":
                                _dbg_dp_reject_reasons[reason] += 1
                    return False
                window_sum += effects["time_used_s"]
                # Attribute time to DP (primary) or backfill phases
                if current_phase == "primary" and dp_filter_sequence is not None:
                    dp_time_used += effects["time_used_s"]
                else:
                    backfill_time_used += effects["time_used_s"]
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
                    _sim_t0 = _time.monotonic() if timing_enabled else None
                    libid_counter = _emit_simlib(
                        writer,
                        libid_counter,
                        t["Name"],
                        t["RA_deg"],
                        t["Dec_deg"],
                        effects["simlib_epochs"],
                        redshift=(
                            float(t.get("redshift"))
                            if t.get("redshift") is not None
                            else None
                        ),
                    )
                    if timing_enabled and _sim_t0 is not None:
                        simlib_write_s += max(0.0, (_time.monotonic() - _sim_t0))
                return True

            for idx_filt, filt in enumerate(batch_order):
                if nightly_cap is not None and len(seen_ids) >= nightly_cap:
                    break
                batch = [
                    c
                    for c in candidates
                    if c["first_filter"] == filt and c["Name"] not in seen_ids
                ]
                min_amort = cfg.filter_change_s / max(cfg.swap_amortize_min, 1)
                dp_limit = dp_counts[idx_filt] if (dp_counts and idx_filt < len(dp_counts)) else None
                exp_s = float(cfg.exposure_by_filter.get(filt, 0.0))
                est_visit_s_base = float(exp_s) + float(cfg.inter_exposure_min_s)
                segment_swap_charged = True
                if dp_filter_sequence is not None:
                    segment_swap_charged = idx_filt == 0
                    if idx_filt == 0 and state != filt:
                        state = filt
                if dp_limit is not None and idx_filt > 0:
                    payoff_thresh = (
                        float(cfg.min_batch_payoff_s)
                        if getattr(cfg, "min_batch_payoff_s", None) is not None
                        else float(cfg.filter_change_s)
                    )
                    if est_visit_s_base * max(dp_limit, 1) < payoff_thresh:
                        continue
                scheduled_in_segment = 0
                while batch:
                    if nightly_cap is not None and len(seen_ids) >= nightly_cap:
                        break
                    if dp_limit is not None and scheduled_in_segment >= dp_limit:
                        break
                    time_left = float(cap_s - window_sum)
                    # conservative per-visit wall time used for swap amortization
                    est_visit_s = est_visit_s_base
                    k_time = max(1, int(time_left // max(est_visit_s, 1.0)))
                    k = max(1, min(len(batch), k_time))
                    amortized_penalty = cfg.filter_change_s / k
                    # select next target based on filter-aware cost
                    # Hoist invariants for this selection round
                    now_mjd_cost = Time(current_time_utc).mjd
                    costs: List[float] = []
                    reasons_for_batch: dict[str, str] = {}
                    # Pre-filter by cadence once as a fast reject; low-z Ia keep for per-target override
                    if cad_on:
                        def _cad_ok_any(tt: dict) -> bool:
                            try:
                                if _is_low_z_ia(tt.get("sn_type"), tt.get("redshift"), cfg):
                                    return True
                            except Exception:
                                pass
                            allowed_list = tt.get("policy_allowed", tt.get("allowed", [])) or []
                            return any(
                                tracker.cadence_gate(tt["Name"], f, now_mjd_cost, cad_tgt, cad_jit)
                                for f in allowed_list
                            )
                        batch = [tt for tt in batch if _cad_ok_any(tt)]
                    for t in batch:
                        lowz_ia_t = False
                        try:
                            lowz_ia_t = _is_low_z_ia(
                                t.get("sn_type"), t.get("redshift"), cfg
                            )
                        except Exception:
                            lowz_ia_t = False
                        cad_tgt_for_t = (
                            float(getattr(cfg, "low_z_ia_cadence_days_target", cad_tgt))
                            if (
                                lowz_ia_t
                                and isinstance(
                                    getattr(cfg, "low_z_ia_cadence_days_target", None),
                                    (int, float),
                                )
                            )
                            else cad_tgt
                        )
                        gated_for_cost = [
                            f
                            for f in t.get("policy_allowed", t["allowed"])
                            if (not cad_on)
                            or tracker.cadence_gate(
                                t["Name"], f, now_mjd_cost, cad_tgt_for_t, cad_jit
                            )
                        ]
                        hard_batches = dp_filter_sequence is not None
                        if hard_batches:
                            # In DP hard-batch mode we require the batch filter to be
                            # cadence-eligible now; otherwise treat as ineligible.
                            if filt not in gated_for_cost:
                                first_tmp = None
                                if _dbg_enabled:
                                    reasons_for_batch[t["Name"]] = "cadence_ineligible_now"
                            else:
                                first_tmp = filt
                        elif gated_for_cost:

                            def _bonus_cost(f: str) -> float:
                                return tracker.compute_filter_bonus(
                                    t["Name"],
                                    f,
                                    now_mjd_cost,
                                    cad_tgt_for_t,
                                    cad_sig,
                                    cad_wt,
                                    cad_first,
                                    cfg.cosmo_weight_by_filter,
                                    cfg.color_target_pairs,
                                    cfg.color_window_days,
                                    cfg.color_alpha,
                                    cfg.first_epoch_color_boost,
                                    diversity_enable=getattr(cfg, "diversity_enable", False),
                                    diversity_target_per_filter=getattr(cfg, "diversity_target_per_filter", 1),
                                    diversity_window_days=getattr(cfg, "diversity_window_days", 5.0),
                                    diversity_alpha=getattr(cfg, "diversity_alpha", 0.3),
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
                        slew_s = slew_time_seconds(
                            sep,
                            small_deg=cfg.slew_small_deg,
                            small_time=cfg.slew_small_time_s,
                            rate_deg_per_s=cfg.slew_rate_deg_per_s,
                            settle_s=cfg.slew_settle_s,
                        )
                        if first_tmp is None:
                            cost = 1e9
                        else:
                            opp_s = opportunity_cost_seconds_cached(
                                t,
                                first_tmp,
                                est_visit_s,
                                now_mjd_cost,
                                cfg,
                                phot_cfg,
                                sky_provider,
                                sky_cfg,
                                site,
                                minutes=int(getattr(cfg, "primary_m5_minutes", 1) or 1),
                            )
                            w_slew = float(getattr(cfg, "inbatch_slew_weight", 0.3))
                            cost = opp_s + w_slew * float(slew_s)
                            # In hard-batch mode ignore carousel state mismatch and any
                            # swap penalty; otherwise apply original penalty logic.
                            hard_batches = dp_filter_sequence is not None
                            if (not hard_batches) and (state is not None and state != first_tmp):
                                # Permit one initial alignment swap even when the configured
                                # per-window limit is zero. Subsequent swaps obey the limit.
                                _limit = int(getattr(cfg, "max_swaps_per_window", 999))
                                _sofar = int(swap_count_by_window.get(idx_w, 0))
                                _would_swap = True
                                _allow_initial = (_limit == 0 and _sofar == 0 and _would_swap)
                                if _sofar >= _limit and not _allow_initial:
                                    cost = 1e9
                                    if _dbg_enabled:
                                        reasons_for_batch[t["Name"]] = "swap_limit_reached"
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
                    if (not costs) or (min(costs) >= 1e8):
                        if batch:
                            existing_names = {c["Name"] for c in backfill_candidates}
                            for t_rem in batch:
                                if t_rem["Name"] not in existing_names:
                                    backfill_candidates.append(t_rem)
                                    existing_names.add(t_rem["Name"])
                                if _dbg_enabled:
                                    _dbg_dp_reject_reasons[reasons_for_batch.get(t_rem["Name"], "deferred")] += 1
                        batch.clear()
                        break
                    j = int(np.argmin(costs))
                    if (
                        dp_filter_sequence is not None
                        and not segment_swap_charged
                        and (state is None or state != filt)
                    ):
                        swap_dur = float(cfg.filter_change_s)
                        window_filter_change_s += swap_dur
                        window_sum += swap_dur
                        current_time_utc = current_time_utc + pd.to_timedelta(
                            swap_dur, unit="s"
                        )
                        state = filt
                        swap_count_by_window[idx_w] = swap_count_by_window.get(idx_w, 0) + 1
                        segment_swap_charged = True
                    t = batch.pop(j)
                    sn_id = t["Name"]
                    if sn_id in seen_ids:
                        continue
                    # DP hard-batch: always execute as single-filter visits,
                    # assuming the carousel is already at the segment filter.
                    if dp_filter_sequence is not None:
                        scheduled_ok = _attempt_schedule(
                            t,
                            single_filter_mode=True,
                            state_for_call=filt,
                        )
                    else:
                        scheduled_ok = _attempt_schedule(t)
                    if scheduled_ok:
                        seen_ids.add(sn_id)
                        scheduled_in_segment += 1
                        if _dbg_enabled:
                            _dbg_dp_sched_counts[filt] += 1
            # Merge deferred DP items into the backfill pool; handle after DP completes.
            if deferred:
                existing_names = {c["Name"] for c in backfill_candidates}
                for tdef in deferred:
                    if tdef["Name"] not in existing_names:
                        backfill_candidates.append(tdef)
                        existing_names.add(tdef["Name"])
                deferred.clear()

            _t.mark(f"schedule_primary_and_deferred[{window_label_out}]")
            # If time remains and the nightly cap allows, try backfilling with
            # additional visible candidates not in the primary capped set.
            available_bf = (
                {c["first_filter"] for c in backfill_candidates} if backfill_candidates else set()
            )
            if (
                window_sum < cap_s
                and (nightly_cap is None or len(seen_ids) < nightly_cap)
                and backfill_candidates
                and has_room_for_any_visit(window_sum, cap_s, cfg, palette=available_bf)
            ):
                current_phase = "backfill"
                # Rebuild palette for backfill candidates (can differ from primary)
                available_bf = {c["first_filter"] for c in backfill_candidates}
                counts_bf = Counter(c["first_filter"] for c in backfill_candidates)
                batch_order_bf = sorted(
                    available_bf,
                    key=lambda f: (
                        -counts_bf.get(f, 0),
                        pal_rot.index(f) if f in pal_rot else 99,
                    ),
                )
                if first_override in batch_order_bf:
                    batch_order_bf.remove(first_override)
                    batch_order_bf.insert(0, first_override)
                # Prefer current carousel state to avoid an immediate swap
                if state is not None and state in batch_order_bf:
                    try:
                        batch_order_bf.remove(state)
                    except ValueError:
                        pass
                    batch_order_bf.insert(0, state)

                # Check if any backfill candidate can run without a swap under
                # current cadence gating and carousel state. If none, we permit
                # one extra swap beyond max_swaps_per_window to utilize time.
                try:
                    now_mjd_bf = Time(current_time_utc).mjd
                except Exception:
                    now_mjd_bf = None

                # During backfill, the schedule has progressed and the current
                # time/state may differ from the DP planning snapshot. Re-check
                # cadence here so we only pull candidates that are cadence-eligible
                # at the exact backfill time (now_mjd_bf). This avoids scheduling
                # a pair that was eligible earlier (or in a different order) but
                # is no longer eligible after prior visits/deferrals.
                def _passes_cad(t: dict) -> bool:
                    if not cad_on or now_mjd_bf is None:
                        return True
                    lowz_ia_t = False
                    try:
                        lowz_ia_t = _is_low_z_ia(
                            t.get("sn_type"), t.get("redshift"), cfg
                        )
                    except Exception:
                        lowz_ia_t = False
                    cad_tgt_for_t = (
                        float(getattr(cfg, "low_z_ia_cadence_days_target", cad_tgt))
                        if (
                            lowz_ia_t
                            and isinstance(
                                getattr(cfg, "low_z_ia_cadence_days_target", None),
                                (int, float),
                            )
                        )
                        else cad_tgt
                    )
                    for f in t.get("policy_allowed", t["allowed"]):
                        if tracker.cadence_gate(
                            t["Name"], f, now_mjd_bf, cad_tgt_for_t, cad_jit
                        ):
                            return True
                    return False

                has_state_backfill = False
                if state is not None:
                    for t in backfill_candidates:
                        if t["Name"] in seen_ids:
                            continue
                        if t.get("first_filter") == state and _passes_cad(t):
                            has_state_backfill = True
                            break
                # If state is None we don't count this as requiring a swap; be conservative
                allow_extra_swap = state is not None and (not has_state_backfill)

                for filt in batch_order_bf:
                    if nightly_cap is not None and len(seen_ids) >= nightly_cap:
                        break
                    if _dbg_enabled:
                        if (not _dbg_backfill_order) or _dbg_backfill_order[-1] != filt:
                            _dbg_backfill_order.append(filt)
                    batch = [
                        c
                        for c in backfill_candidates
                        if c["first_filter"] == filt and c["Name"] not in seen_ids
                    ]
                    min_amort = cfg.filter_change_s / max(cfg.swap_amortize_min, 1)
                    while batch and window_sum < cap_s:
                        if nightly_cap is not None and len(seen_ids) >= nightly_cap:
                            break
                        time_left = float(cap_s - window_sum)
                        exp_s = float(cfg.exposure_by_filter.get(filt, 0.0))
                        est_visit_s = float(exp_s) + float(cfg.inter_exposure_min_s)
                        k_time = max(1, int(time_left // max(est_visit_s, 1.0)))
                        k = max(1, min(len(batch), k_time))
                        amortized_penalty = cfg.filter_change_s / k
                        # Hoist invariants and pre-filter by cadence once (keep low-z Ia)
                        now_mjd_cost = Time(current_time_utc).mjd
                        if cad_on:
                            def _cad_ok_any_bf(tt: dict) -> bool:
                                try:
                                    if _is_low_z_ia(tt.get("sn_type"), tt.get("redshift"), cfg):
                                        return True
                                except Exception:
                                    pass
                                allowed_list = tt.get("policy_allowed", tt.get("allowed", [])) or []
                                return any(
                                    tracker.cadence_gate(tt["Name"], f, now_mjd_cost, cad_tgt, cad_jit)
                                    for f in allowed_list
                                )
                            batch = [tt for tt in batch if _cad_ok_any_bf(tt)]
                        costs: List[float] = []
                        for t in batch:
                            lowz_ia_t = False
                            try:
                                lowz_ia_t = _is_low_z_ia(
                                    t.get("sn_type"), t.get("redshift"), cfg
                                )
                            except Exception:
                                lowz_ia_t = False
                            cad_tgt_for_t = (
                                float(
                                    getattr(
                                        cfg, "low_z_ia_cadence_days_target", cad_tgt
                                    )
                                )
                                if (
                                    lowz_ia_t
                                    and isinstance(
                                        getattr(
                                            cfg, "low_z_ia_cadence_days_target", None
                                        ),
                                        (int, float),
                                    )
                                )
                                else cad_tgt
                            )
                            gated_for_cost = [
                                f
                                for f in t.get("policy_allowed", t["allowed"])
                                if (not cad_on)
                                or tracker.cadence_gate(
                                    t["Name"], f, now_mjd_cost, cad_tgt_for_t, cad_jit
                                )
                            ]
                            if gated_for_cost:

                                def _bonus_cost(f: str) -> float:
                                    return tracker.compute_filter_bonus(
                                        t["Name"],
                                        f,
                                        now_mjd_cost,
                                        cad_tgt_for_t,
                                        cad_sig,
                                        cad_wt,
                                        cad_first,
                                        cfg.cosmo_weight_by_filter,
                                        cfg.color_target_pairs,
                                        cfg.color_window_days,
                                        cfg.color_alpha,
                                        cfg.first_epoch_color_boost,
                                        diversity_enable=getattr(cfg, "diversity_enable", False),
                                        diversity_target_per_filter=getattr(cfg, "diversity_target_per_filter", 1),
                                        diversity_window_days=getattr(cfg, "diversity_window_days", 5.0),
                                        diversity_alpha=getattr(cfg, "diversity_alpha", 0.3),
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
                            slew_s = slew_time_seconds(
                                sep,
                                small_deg=cfg.slew_small_deg,
                                small_time=cfg.slew_small_time_s,
                                rate_deg_per_s=cfg.slew_rate_deg_per_s,
                                settle_s=cfg.slew_settle_s,
                            )
                            if first_tmp is None:
                                cost = 1e9
                            else:
                                opp_s = opportunity_cost_seconds_cached(
                                    t,
                                    first_tmp,
                                    est_visit_s,
                                    now_mjd_cost,
                                    cfg,
                                    phot_cfg,
                                    sky_provider,
                                    sky_cfg,
                                    site,
                                    minutes=int(getattr(cfg, "backfill_m5_minutes", 1) or 1),
                                )
                                w_slew = float(getattr(cfg, "inbatch_slew_weight", 0.3))
                                cost = opp_s + w_slew * float(slew_s)
                                if state is not None and state != first_tmp:
                                    limit = int(getattr(cfg, "max_swaps_per_window", 999))
                                    if swap_count_by_window.get(idx_w, 0) >= limit and not (
                                        allow_extra_swap and not backfill_extra_swap_used
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
                        # Guard against empty or invalid cost lists which can occur
                        # if cadence gating filters the batch to zero within-loop.
                        if (not costs) or (min(costs) >= 1e8):
                            # Nothing viable to schedule in this batch/filter now.
                            break
                        j = int(np.argmin(costs))
                        t = batch.pop(j)
                        sn_id = t["Name"]
                        if sn_id in seen_ids:
                            continue
                        if _attempt_schedule(t):
                            seen_ids.add(sn_id)
                            # Mark that we've consumed the one-time extra swap allowance
                            # if the swap counter moved beyond the nominal limit.
                            limit = int(getattr(cfg, "max_swaps_per_window", 999))
                            if swap_count_by_window.get(idx_w, 0) > limit:
                                backfill_extra_swap_used = True
                            if _dbg_enabled:
                                _dbg_backfill_sched_counts[filt] += 1

                # Final attempt to flush any deferred items created while
                # backfilling, as long as time and the nightly cap allow.
                progress = True
                while (
                    progress
                    and deferred
                    and window_sum < cap_s
                    and (nightly_cap is None or len(seen_ids) < nightly_cap)
                ):
                    progress = False
                    for t in list(deferred):
                        sn_id = t["Name"]
                        if sn_id in seen_ids:
                            deferred.remove(t)
                            continue
                        if _attempt_schedule(t, allow_defer=False):
                            deferred.remove(t)
                            seen_ids.add(sn_id)
                            progress = True

            # If time still remains, allow a limited second visit to targets
            # already observed earlier tonight (e.g., to take a complementary
            # color in a different filter). This considers both primary and
            # backfill pools and uses the same cadence and cost logic.
            # NEW: skip repeats unless the leftover time can fit (inter_exposure_min + min exposure)
            available_rep = (
                {c["first_filter"] for c in candidates} | {c["first_filter"] for c in backfill_candidates}
            ) if (candidates or backfill_candidates) else set()
            if (
                window_sum < cap_s
                and (nightly_cap is None or len(seen_ids) < nightly_cap)
                and (candidates or backfill_candidates)
                and has_room_for_any_visit(window_sum, cap_s, cfg, palette=available_rep)
            ):
                current_phase = "backfill"
                repeat_candidates = [
                    c for c in (candidates + backfill_candidates) if c["Name"] in seen_ids
                ]
                if repeat_candidates:
                    try:
                        now_mjd_rep = Time(current_time_utc).mjd
                    except Exception:
                        now_mjd_rep = None
                    rep_step_min = max(1, int(getattr(cfg, "repeats_m5_minutes", 1)))
                    m5_now_cache: dict[tuple[str, str, float], float] = {}
                    m5_best_cache: dict[tuple[str, str, float], float] = {}

                    def _rep_target_key(t: dict) -> str | None:
                        for field in ("Name", "name", "ID", "id"):
                            val = t.get(field)
                            if val is None:
                                continue
                            try:
                                text = str(val).strip()
                            except Exception:
                                continue
                            if text:
                                return text
                        return None

                    def _rep_minute_bin(mjd: float | None) -> float | None:
                        if mjd is None:
                            return None
                        try:
                            mjd_val = float(mjd)
                        except Exception:
                            return None
                        bucket = math.floor((mjd_val * 1440.0) / rep_step_min)
                        return bucket * rep_step_min / 1440.0

                    def _rep_m5_now(t: dict, filt: str, mjd: float | None) -> float | None:
                        bucket = _rep_minute_bin(mjd)
                        if bucket is None:
                            return None
                        tgt_key = _rep_target_key(t)
                        if not tgt_key:
                            return None
                        cache_key = (tgt_key, filt, bucket)
                        cached = m5_now_cache.get(cache_key)
                        if cached is not None:
                            return cached
                        val = _cached_m5_at_time(
                            target=t,
                            filt=filt,
                            cfg=cfg,
                            phot_cfg=phot_cfg,
                            sky_provider=sky_provider,
                            sky_cfg=sky_cfg,
                            mjd=bucket,
                            minutes=rep_step_min,
                            site=site,
                            tag_best=False,
                        )
                        if val is None:
                            return None
                        m5_now_cache[cache_key] = float(val)
                        return m5_now_cache[cache_key]

                    def _rep_m5_best(
                        t: dict, filt: str, m5_now_val: float | None
                    ) -> float | None:
                        best_mjd = t.get("best_time_mjd")
                        if best_mjd is None:
                            best_ts = t.get("best_time_utc")
                            if best_ts is None:
                                return m5_now_val
                            try:
                                if isinstance(best_ts, pd.Timestamp):
                                    best_mjd = Time(best_ts.tz_convert("UTC")).mjd
                                else:
                                    best_mjd = Time(pd.Timestamp(best_ts).tz_localize("UTC")).mjd
                            except Exception:
                                return m5_now_val
                        try:
                            best_mjd_val = float(best_mjd)
                        except Exception:
                            return m5_now_val
                        bucket = _rep_minute_bin(best_mjd_val)
                        if bucket is None:
                            return m5_now_val
                        tgt_key = _rep_target_key(t)
                        if not tgt_key:
                            return m5_now_val
                        cache_key = (tgt_key, filt, bucket)
                        cached = m5_best_cache.get(cache_key)
                        if cached is not None:
                            return cached
                        val = _cached_m5_at_time(
                            target=t,
                            filt=filt,
                            cfg=cfg,
                            phot_cfg=phot_cfg,
                            sky_provider=sky_provider,
                            sky_cfg=sky_cfg,
                            mjd=bucket,
                            minutes=rep_step_min,
                            site=site,
                            tag_best=True,
                        )
                        if val is None:
                            return m5_now_val
                        m5_best_cache[cache_key] = float(val)
                        return m5_best_cache[cache_key]

                    def _rep_opp_seconds(
                        est_visit_s: float, m5_now_val: float | None, m5_best_val: float | None
                    ) -> float:
                        if m5_now_val is None or m5_best_val is None:
                            return 0.0
                        delta_m = max(0.0, float(m5_best_val) - float(m5_now_val))
                        opp = float(est_visit_s) * (10 ** (0.8 * delta_m) - 1.0)
                        cap = float(getattr(cfg, "inbatch_cost_cap_s", 600.0))
                        return float(min(max(0.0, opp), cap))

                    available_rep = {c["first_filter"] for c in repeat_candidates}
                    counts_rep = Counter(c["first_filter"] for c in repeat_candidates)
                    batch_order_rep = sorted(
                        available_rep,
                        key=lambda f: (
                            -counts_rep.get(f, 0),
                            pal_rot.index(f) if f in pal_rot else 99,
                        ),
                    )
                    if first_override in batch_order_rep:
                        batch_order_rep.remove(first_override)
                        batch_order_rep.insert(0, first_override)
                    # Prefer current carousel state to minimize swap entering repeats
                    if state is not None and state in batch_order_rep:
                        try:
                            batch_order_rep.remove(state)
                        except ValueError:
                            pass
                        batch_order_rep.insert(0, state)
                    # Determine if a swap is unavoidable for repeats under current state
                    # For repeat visits within the same window, cadence must be
                    # assessed again at the repeat decision point. The effective
                    # "now" (now_mjd_rep) has advanced and earlier actions may
                    # have updated last-visit times, so eligibility can change.
                    # Re-checking here keeps repeats consistent with per-filter
                    # cadence policy while permitting relaxed behavior elsewhere
                    # when configured.
                    def _passes_cad_rep(t: dict) -> bool:
                        if not cad_on or now_mjd_rep is None:
                            return True
                        lowz_ia_t = False
                        try:
                            lowz_ia_t = _is_low_z_ia(
                                t.get("sn_type"), t.get("redshift"), cfg
                            )
                        except Exception:
                            lowz_ia_t = False
                        cad_tgt_for_t = (
                            float(getattr(cfg, "low_z_ia_cadence_days_target", cad_tgt))
                            if (
                                lowz_ia_t
                                and isinstance(
                                    getattr(cfg, "low_z_ia_cadence_days_target", None),
                                    (int, float),
                                )
                            )
                            else cad_tgt
                        )
                        for f in t.get("policy_allowed", t["allowed"]):
                            if tracker.cadence_gate(
                                t["Name"], f, now_mjd_rep, cad_tgt_for_t, cad_jit
                            ):
                                return True
                        return False

                    has_state_repeat = False
                    if state is not None:
                        for t in repeat_candidates:
                            if t.get("first_filter") == state and _passes_cad_rep(t):
                                has_state_repeat = True
                                break
                    allow_extra_swap_rep = state is not None and (not has_state_repeat)

                    for filt in batch_order_rep:
                        if nightly_cap is not None and len(seen_ids) >= nightly_cap:
                            break
                        batch = []
                        for c in repeat_candidates:
                            if c["first_filter"] != filt:
                                continue
                            name_c = c["Name"]
                            # Determine per-target allowed repeats (default 1)
                            lowz = False
                            try:
                                lowz = _is_low_z_ia(
                                    c.get("sn_type"), c.get("redshift"), cfg
                                )
                            except Exception:
                                lowz = False
                            allowed_rep = 1
                            if lowz:
                                rep_conf = getattr(
                                    cfg, "low_z_ia_repeats_per_window", None
                                )
                                if isinstance(rep_conf, int) and rep_conf >= 1:
                                    allowed_rep = int(rep_conf)
                            if (
                                repeat_counts.get(name_c, 0) < allowed_rep
                                and _passes_cad_rep(c)
                            ):
                                batch.append(c)
                        min_amort = cfg.filter_change_s / max(cfg.swap_amortize_min, 1)
                        while batch and window_sum < cap_s:
                            if nightly_cap is not None and len(seen_ids) >= nightly_cap:
                                break
                            time_left = float(cap_s - window_sum)
                            exp_s = float(cfg.exposure_by_filter.get(filt, 0.0))
                            est_visit_s = float(exp_s) + float(cfg.inter_exposure_min_s)
                            k_time = max(1, int(time_left // max(est_visit_s, 1.0)))
                            k = max(1, min(len(batch), k_time))
                            amortized_penalty = cfg.filter_change_s / k
                            costs: List[float] = []
                            for t in batch:
                                allowed_for_cost = t.get("policy_allowed", t["allowed"]) or []
                                now_mjd_for_cost = now_mjd_rep
                                if now_mjd_for_cost is None:
                                    try:
                                        now_mjd_for_cost = Time(current_time_utc).mjd
                                    except Exception:
                                        now_mjd_for_cost = 0.0
                                if cad_on and now_mjd_rep is not None:
                                    gated_for_cost = [
                                        f
                                        for f in allowed_for_cost
                                        if tracker.cadence_gate(
                                            t["Name"], f, now_mjd_rep, cad_tgt, cad_jit
                                        )
                                    ]
                                else:
                                    gated_for_cost = list(allowed_for_cost)
                                if gated_for_cost:

                                    def _bonus_cost_rep(f: str) -> float:
                                        return tracker.compute_filter_bonus(
                                            t["Name"],
                                            f,
                                            now_mjd_for_cost,
                                            cad_tgt,
                                            cad_sig,
                                            cad_wt,
                                            cad_first,
                                            cfg.cosmo_weight_by_filter,
                                            cfg.color_target_pairs,
                                            cfg.color_window_days,
                                            cfg.color_alpha,
                                            cfg.first_epoch_color_boost,
                                            diversity_enable=getattr(cfg, "diversity_enable", False),
                                            diversity_target_per_filter=getattr(cfg, "diversity_target_per_filter", 1),
                                            diversity_window_days=getattr(cfg, "diversity_window_days", 5.0),
                                            diversity_alpha=getattr(cfg, "diversity_alpha", 0.3),
                                        )

                                    first_tmp = (
                                        t["first_filter"]
                                        if t["first_filter"] in gated_for_cost
                                        else max(gated_for_cost, key=_bonus_cost_rep)
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
                                slew_s = slew_time_seconds(
                                    sep,
                                    small_deg=cfg.slew_small_deg,
                                    small_time=cfg.slew_small_time_s,
                                    rate_deg_per_s=cfg.slew_rate_deg_per_s,
                                    settle_s=cfg.slew_settle_s,
                                )
                                if first_tmp is None:
                                    cost = 1e9
                                else:
                                    m5_now_val = _rep_m5_now(t, first_tmp, now_mjd_for_cost)
                                    m5_best_val = _rep_m5_best(t, first_tmp, m5_now_val)
                                    opp_s = _rep_opp_seconds(est_visit_s, m5_now_val, m5_best_val)
                                    w_slew = float(getattr(cfg, "inbatch_slew_weight", 0.3))
                                    cost = opp_s + w_slew * float(slew_s)
                                    if state is not None and state != first_tmp:
                                        limit = int(
                                            getattr(cfg, "max_swaps_per_window", 999)
                                        )
                                        if swap_count_by_window.get(
                                            idx_w, 0
                                        ) >= limit and not (
                                            allow_extra_swap_rep
                                            and not backfill_extra_swap_used
                                        ):
                                            cost = 1e9
                                        else:
                                            scale = 1.0
                                            if (
                                                tracker.cosmology_boost(
                                                    t["Name"],
                                                    first_tmp,
                                                    now_mjd_for_cost,
                                                    cfg.color_target_pairs,
                                                    cfg.color_window_days,
                                                    cfg.color_alpha,
                                                )
                                                > 1.0
                                            ):
                                                scale = cfg.swap_cost_scale_color
                                            penalty = (
                                                max(amortized_penalty, min_amort) * scale
                                            )
                                            cost += penalty
                                costs.append(cost)
                            j = int(np.argmin(costs))
                            t = batch.pop(j)
                            sn_id = t["Name"]
                            # Per-target allowed repeats (default 1)
                            lowz = False
                            try:
                                lowz = _is_low_z_ia(
                                    t.get("sn_type"), t.get("redshift"), cfg
                                )
                            except Exception:
                                lowz = False
                            allowed_rep = 1
                            if lowz:
                                rep_conf = getattr(
                                    cfg, "low_z_ia_repeats_per_window", None
                                )
                                if isinstance(rep_conf, int) and rep_conf >= 1:
                                    allowed_rep = int(rep_conf)
                            if repeat_counts.get(sn_id, 0) >= allowed_rep:
                                continue
                            if _attempt_schedule(t):
                                repeat_counts[sn_id] = repeat_counts.get(sn_id, 0) + 1
                                # Reuse the same one-time extra swap allowance across backfill/repeats
                                limit = int(getattr(cfg, "max_swaps_per_window", 999))
                                if swap_count_by_window.get(idx_w, 0) > limit:
                                    backfill_extra_swap_used = True
                                if _dbg_enabled:
                                    _dbg_repeat_sched_counts[filt] += 1
            _t.mark(f"schedule_repeats[{window_label_out}]")
            _t.mark(f"schedule_backfill[{window_label_out}]")

            # Final last-resort relaxed backfill: only if time remains and
            # otherwise we'd leave the window under-utilized. This ignores
            # cadence gating but keeps swap and other constraints. Only targets
            # not yet observed in this window are considered to avoid
            # exceeding repeat policies here.
            if (
                getattr(cfg, "backfill_relax_cadence", False)
                and window_sum < cap_s
                and (nightly_cap is None or len(seen_ids) < nightly_cap)
                and (candidates or backfill_candidates)
            ):
                current_phase = "backfill"
                # Permit relaxed mode to revisit already observed targets, but
                # keep unseen SNe ahead of repeats when picking the next visit.
                relaxed_pool = list(candidates + backfill_candidates)
                if relaxed_pool:
                    available_relax = {c["first_filter"] for c in relaxed_pool}
                    counts_relax = Counter(
                        c["first_filter"] for c in relaxed_pool
                    )
                    batch_order_relax = sorted(
                        available_relax,
                        key=lambda f: (
                            -counts_relax.get(f, 0),
                            pal_rot.index(f) if f in pal_rot else 99,
                        ),
                    )
                    # Prefer current state to avoid an immediate swap in relaxed mode
                    if state is not None and state in batch_order_relax:
                        try:
                            batch_order_relax.remove(state)
                        except ValueError:
                            pass
                        batch_order_relax.insert(0, state)
                    if first_override in batch_order_relax:
                        batch_order_relax.remove(first_override)
                        batch_order_relax.insert(0, first_override)
                    for filt in batch_order_relax:
                        if nightly_cap is not None and len(seen_ids) >= nightly_cap:
                            break
                        batch = [c for c in relaxed_pool if c["first_filter"] == filt]
                        min_amort = cfg.filter_change_s / max(cfg.swap_amortize_min, 1)
                        while batch and window_sum < cap_s:
                            if nightly_cap is not None and len(seen_ids) >= nightly_cap:
                                break
                            time_left = float(cap_s - window_sum)
                            exp_s = float(cfg.exposure_by_filter.get(filt, 0.0))
                            est_visit_s = float(exp_s) + float(cfg.inter_exposure_min_s)
                            k_time = max(1, int(time_left // max(est_visit_s, 1.0)))
                            k = max(1, min(len(batch), k_time))
                            amortized_penalty = cfg.filter_change_s / k
                            costs: List[float] = []
                            idx_candidates: List[int] = []
                            has_unseen = any(
                                c["Name"] not in seen_ids for c in batch
                            )

                            for idx_item, t in enumerate(batch):
                                now_mjd_cost = Time(current_time_utc).mjd
                                # In relaxed mode, ignore cadence gating when
                                # choosing the first filter for cost.
                                allowed_for_cost = (
                                    t.get("policy_allowed", t["allowed"]) or []
                                )

                                def _bonus_cost_rel(f: str) -> float:
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
                                        diversity_enable=getattr(cfg, "diversity_enable", False),
                                        diversity_target_per_filter=getattr(cfg, "diversity_target_per_filter", 1),
                                        diversity_window_days=getattr(cfg, "diversity_window_days", 5.0),
                                        diversity_alpha=getattr(cfg, "diversity_alpha", 0.3),
                                    )

                                if allowed_for_cost:
                                    first_tmp = (
                                        t["first_filter"]
                                        if t["first_filter"] in allowed_for_cost
                                        else max(allowed_for_cost, key=_bonus_cost_rel)
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
                                slew_s = slew_time_seconds(
                                    sep,
                                    small_deg=cfg.slew_small_deg,
                                    small_time=cfg.slew_small_time_s,
                                    rate_deg_per_s=cfg.slew_rate_deg_per_s,
                                    settle_s=cfg.slew_settle_s,
                                )
                                if first_tmp is None:
                                    cost = 1e9
                                else:
                                    now_mjd_cost = Time(current_time_utc).mjd
                                    opp_s = _opportunity_cost_seconds(
                                        t,
                                        first_tmp,
                                        est_visit_s,
                                        now_mjd_cost,
                                        cfg,
                                        phot_cfg,
                                        sky_provider,
                                        sky_cfg,
                                        site,
                                    )
                                    w_slew = float(getattr(cfg, "inbatch_slew_weight", 0.3))
                                    cost = opp_s + w_slew * float(slew_s)
                                    if state is not None and state != first_tmp:
                                        limit = int(
                                            getattr(cfg, "max_swaps_per_window", 999)
                                        )
                                        if swap_count_by_window.get(idx_w, 0) >= limit:
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
                                            penalty = (
                                                max(amortized_penalty, min_amort) * scale
                                            )
                                            cost += penalty
                                sn_id_cost = t["Name"]
                                already_seen = sn_id_cost in seen_ids
                                if already_seen:
                                    if has_unseen:
                                        continue

                                    lowz = False
                                    try:
                                        lowz = _is_low_z_ia(
                                            t.get("sn_type"), t.get("redshift"), cfg
                                        )
                                    except Exception:
                                        lowz = False
                                    allowed_rep = 1
                                    if lowz:
                                        rep_conf = getattr(
                                            cfg, "low_z_ia_repeats_per_window", None
                                        )
                                        if isinstance(rep_conf, int) and rep_conf >= 1:
                                            allowed_rep = int(rep_conf)
                                    if repeat_counts.get(sn_id_cost, 0) >= allowed_rep:
                                        continue

                                idx_candidates.append(idx_item)
                                costs.append(cost)

                            if not costs:
                                break

                            j_local = int(np.argmin(costs))
                            j = idx_candidates[j_local]
                            t = batch.pop(j)
                            sn_id = t["Name"]

                            already_seen_main = sn_id in seen_ids
                            if already_seen_main:
                                lowz = False
                                try:
                                    lowz = _is_low_z_ia(
                                        t.get("sn_type"), t.get("redshift"), cfg
                                    )
                                except Exception:
                                    lowz = False
                                allowed_rep = 1
                                if lowz:
                                    rep_conf = getattr(
                                        cfg, "low_z_ia_repeats_per_window", None
                                    )
                                    if isinstance(rep_conf, int) and rep_conf >= 1:
                                        allowed_rep = int(rep_conf)
                                if repeat_counts.get(sn_id, 0) >= allowed_rep:
                                    continue

                            # Set relaxed cadence flag in the per-call state
                            def _attempt_schedule_relaxed(tt: dict) -> bool:
                                nonlocal window_sum, prev, internal_changes, window_filter_change_s
                                nonlocal state, window_slew_times, window_airmasses, window_skymags
                                nonlocal filters_used_set, current_time_utc, order_in_window, libid_counter
                                nonlocal simlib_write_s
                                window_state = {
                                    "allow_defer": False,
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
                                    "relax_cadence": True,
                                }
                                scheduled, effects = _attempt_schedule_one(
                                    tt,
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
                                filters_used_set.update(
                                    updates["filters_used_set_delta"]
                                )
                                summary = updates["summary_updates"]
                                swap_count_by_window[idx_w] = swap_count_by_window.get(
                                    idx_w, 0
                                ) + summary.get("swap_count_delta", 0)
                                internal_changes += summary["internal_changes_delta"]
                                window_filter_change_s += summary[
                                    "window_filter_change_s_delta"
                                ]
                                window_slew_times.append(summary["slew_time"])
                                window_airmasses.append(summary["airmass"])
                                window_skymags.extend(summary["sky_mags"])
                                if not stream_per_sn:
                                    pernight_rows.extend(effects["pernight_rows"])
                                pernight_rows_window.extend(effects["pernight_rows"])
                                day_pernight_rows.extend(
                                    effects["pernight_rows"]
                                )  # day batch
                                if not stream_sequence:
                                    sequence_rows.extend(effects["sequence_rows"])
                                sequence_rows_window.extend(effects["sequence_rows"])
                                day_sequence_rows.extend(
                                    effects["sequence_rows"]
                                )  # day batch
                                if writer and effects["simlib_epochs"]:
                                    _sim_t0b = _time.monotonic() if timing_enabled else None
                                    libid_counter = _emit_simlib(
                                        writer,
                                        libid_counter,
                                        tt["Name"],
                                        tt["RA_deg"],
                                        tt["Dec_deg"],
                                        effects["simlib_epochs"],
                                        redshift=(
                                            float(tt.get("redshift"))
                                            if tt.get("redshift") is not None
                                            else None
                                        ),
                                    )
                                    if timing_enabled and _sim_t0b is not None:
                                        simlib_write_s += max(0.0, (_time.monotonic() - _sim_t0b))
                                return True

                            if _attempt_schedule_relaxed(t):
                                if already_seen_main:
                                    repeat_counts[sn_id] = repeat_counts.get(sn_id, 0) + 1
                                seen_ids.add(sn_id)
                                if _dbg_enabled:
                                    _dbg_relax_sched_counts[filt] += 1
            _t.mark(f"backfill_relaxed[{window_label_out}]")
            used_filters_csv = ",".join(sorted(filters_used_set))
            win = windows[idx_w]
            # Build window-local rows without scanning full history
            pernight_rows_for_window = pernight_rows_window
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
                "dp_time_s": dp_time_used,
                "backfill_time_s": backfill_time_used,
                "n_candidates": len(group),
                "simlib_write_s": simlib_write_s,
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
            summary_row = _build_window_summary_row(
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
            nights_rows.append(summary_row)
            _t.mark(f"window_complete[{window_label_out}]")
            key_for_log: str | None = None
            if window_label_out.startswith("evening"):
                key_for_log = "evening"
            elif window_label_out.startswith("morning"):
                key_for_log = "morning"
            if key_for_log:
                util_raw = summary_row.get("window_utilization")
                observing_raw = summary_row.get("sum_time_s")
                filter_change_raw = summary_row.get("filter_change_s_total")
                filters_used_raw = summary_row.get("filters_used_csv")
                dp_time_raw = summary_row.get("dp_time_s")
                backfill_time_raw = summary_row.get("backfill_time_s")
                window_use_pct = None
                try:
                    if util_raw is not None and not pd.isna(util_raw):
                        window_use_pct = float(util_raw) * 100.0
                except Exception:
                    window_use_pct = None
                observing_s = None
                try:
                    if observing_raw is not None and not pd.isna(observing_raw):
                        observing_s = float(observing_raw)
                except Exception:
                    observing_s = None
                filter_change_s = None
                try:
                    if filter_change_raw is not None and not pd.isna(filter_change_raw):
                        filter_change_s = float(filter_change_raw)
                except Exception:
                    filter_change_s = None
                dp_time_s = None
                try:
                    if dp_time_raw is not None and not pd.isna(dp_time_raw):
                        dp_time_s = float(dp_time_raw)
                except Exception:
                    dp_time_s = None
                backfill_time_s = None
                try:
                    if backfill_time_raw is not None and not pd.isna(backfill_time_raw):
                        backfill_time_s = float(backfill_time_raw)
                except Exception:
                    backfill_time_s = None
                filters_used: str | None
                if isinstance(filters_used_raw, str) and filters_used_raw.strip():
                    filters_used = filters_used_raw
                else:
                    filters_used = None
                window_usage_for_log[key_for_log] = {
                    "window_use_pct": window_use_pct,
                    "observing_s": observing_s,
                    "filter_change_s": filter_change_s,
                    "dp_time_s": dp_time_s,
                    "backfill_time_s": backfill_time_s,
                    "filters_used": filters_used,
                }
                if _dbg_enabled:
                    _dbg_payload: dict = {}
                    if _dbg_pairs_by_filter is not None:
                        _dbg_payload["pair_counts_by_filter"] = _dbg_pairs_by_filter
                    if _dbg_dp_plan is not None:
                        _dbg_payload["dp_plan"] = _dbg_dp_plan
                    if _dbg_dp_sched_counts:
                        _dbg_payload["dp_scheduled_by_filter"] = dict(_dbg_dp_sched_counts)
                    if _dbg_dp_reject_reasons:
                        _dbg_payload["dp_rejected_by_reason"] = dict(_dbg_dp_reject_reasons)
                    if _dbg_backfill_order:
                        _dbg_payload["backfill_order"] = list(_dbg_backfill_order)
                    if _dbg_backfill_sched_counts:
                        _dbg_payload["backfill_scheduled_by_filter"] = dict(
                            _dbg_backfill_sched_counts
                        )
                    if _dbg_repeat_sched_counts:
                        _dbg_payload["repeat_scheduled_by_filter"] = dict(
                            _dbg_repeat_sched_counts
                        )
                    if _dbg_relax_sched_counts:
                        _dbg_payload["relax_scheduled_by_filter"] = dict(
                            _dbg_relax_sched_counts
                        )
                    window_usage_for_log[key_for_log]["debug"] = _dbg_payload
            if override_applied:
                cfg.exposure_by_filter = original_exp

        planned_today = [
            r
            for r in (pernight_rows if not stream_per_sn else day_pernight_rows)
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
            window_usage_for_log,
        )
        _t.mark("log_day_status")

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
        _t.mark("build_summaries_and_stream")
        if timing_enabled:
            print(f"[timing] === {day.date()} end ===")

    if not stream_per_sn:
        if pernight_rows:
            pernight_df = pd.DataFrame(pernight_rows)
            ordered = [c for c in PER_SN_COLUMNS if c in pernight_df.columns]
            ordered.extend(c for c in pernight_df.columns if c not in ordered)
            pernight_df = pernight_df.loc[:, ordered]
        else:
            pernight_df = pd.DataFrame(columns=PER_SN_COLUMNS)
    else:
        pernight_df = pd.DataFrame(columns=PER_SN_COLUMNS)
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
