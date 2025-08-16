"""High-level scheduling routines for LSST twilight supernova observations.

This module batches targets by their first filter, routes within a batch using a
filter-aware slew cost, and accounts for carousel overheads.  Cross-target
filter swaps incur a one-time ``filter_change_s`` penalty, while additional
filters within a visit add internal changes.  The allowed filter set per
twilight window is strictly governed by the Sun-altitude policy in
``PlannerConfig.sun_alt_policy``; filters outside the policy are never scheduled
unless the user supplies a custom policy.  Moon
separation constraints are evaluated in a shared AltAz frame and are ignored
whenever the Moon is below the horizon.

Overhead values follow Rubin Observatory technical notes (slew, readout, and
filter change times), and airmass calculations use the Kasten & Young (1989)
formula.  Exposure times may be overridden by
``PlannerConfig.sun_alt_exposure_ladder`` to shorten visits in bright
twilight.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time

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


def plan_twilight_range_with_caps(
    csv_path: str,
    outdir: str,
    start_date: str,
    end_date: str,
    cfg: PlannerConfig,
    verbose: bool = True,
):
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

    pernight_rows: List[Dict] = []
    nights_rows: List[Dict] = []
    nights = pd.date_range(start, end, freq="D")
    nights_iter = tqdm(nights, desc="Nights", unit="night", leave=True)
    tracker = PriorityTracker(
        hybrid_detections=cfg.hybrid_detections,
        hybrid_exposure_s=cfg.hybrid_exposure_s,
        lc_detections=cfg.lc_detections,
        lc_exposure_s=cfg.lc_exposure_s,
    )
    tz_local = _local_timezone_from_location(site)

    for day in nights_iter:
        # Here 'day' is interpreted as *local* civil date of the evening block
        windows = twilight_windows_for_local_night(
            day.date(),
            site,
            cfg.twilight_sun_alt_min_deg,
            cfg.twilight_sun_alt_max_deg,
        )
        if not windows:
            continue
        evening_tw: datetime | None = None
        morning_tw: datetime | None = None
        for w in windows:
            if w.get("label") == "evening":
                evening_tw = w["start"]
            elif w.get("label") == "morning":
                morning_tw = w["start"]
        if cfg.evening_twilight:
            hh, mm = map(int, cfg.evening_twilight.split(":"))
            dt_local = datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz_local)
            evening_tw = dt_local.astimezone(timezone.utc)
        if cfg.morning_twilight:
            hh, mm = map(int, cfg.morning_twilight.split(":"))
            d2 = day + pd.Timedelta(days=1)
            dt_local = datetime(d2.year, d2.month, d2.day, hh, mm, tzinfo=tz_local)
            morning_tw = dt_local.astimezone(timezone.utc)
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
        for idx_w, w in enumerate(windows):
            current_filter_by_window[idx_w] = cfg.start_filter
            swap_count_by_window[idx_w] = 0
            label = w.get("label")
            window_labels[idx_w] = label
            if label == "morning":
                cap = cfg.morning_cap_s
                if cap == "auto":
                    cap = (w["end"] - w["start"]).total_seconds()
                    cap_source_by_window[idx_w] = "window_duration"
                else:
                    cap_source_by_window[idx_w] = "morning_cap_s"
                window_caps[idx_w] = float(cap)
            elif label == "evening":
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
            if verbose:
                print(f"{day.date().isoformat()}: 0 eligible")
            continue

        best_alts, best_times, best_winidx = [], [], []
        for _, row in subset.iterrows():
            sc = SkyCoord(row["RA_deg"] * u.deg, row["Dec_deg"] * u.deg, frame="icrs")
            max_alt, max_time, max_idx = -999.0, None, None
            best_moon = (float("nan"), float("nan"), float("nan"))
            labeled = [
                (i, w)
                for i, w in enumerate(windows)
                if w.get("label") in ("morning", "evening")
            ]
            for idx_w, w in labeled:
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

        visible = subset[
            (subset["best_time_utc"].notna())
            & (subset["max_alt_deg"] >= cfg.min_alt_deg)
            & (subset["best_window_index"] >= 0)
        ].copy()
        if visible.empty:
            if verbose:
                print(f"{day.date().isoformat()}: 0 visible")
            continue

        visible["priority_score"] = visible.apply(
            lambda r: tracker.score(
                r["Name"], r.get("SN_type_raw"), cfg.priority_strategy
            ),
            axis=1,
        )
        visible.sort_values(
            ["priority_score", "max_alt_deg"], ascending=[False, False], inplace=True
        )
        top_global = visible.head(int(cfg.max_sn_per_night)).copy()

        for idx_w in sorted(set(top_global["best_window_index"].values)):
            group = top_global[top_global["best_window_index"] == idx_w].copy()
            if group.empty:
                continue
            # Skip unlabeled (previous/next day) twilight windows entirely
            if window_labels.get(idx_w) is None:
                continue

            win = windows[idx_w]
            mid = win["start"] + (win["end"] - win["start"]) / 2
            window_label = window_labels.get(idx_w)
            window_label_out = window_label if window_label else f"W{idx_w}"
            sun_alt = (
                get_sun(Time(mid))
                .transform_to(AltAz(location=site, obstime=Time(mid)))
                .alt.to(u.deg)
                .value
            )
            # Apply exposure-time ladder: override baseline exposures if needed
            original_exp = cfg.exposure_by_filter
            override_exp: Dict[str, float] | None = None
            for low, high, exp_dict in cfg.sun_alt_exposure_ladder:
                if low < sun_alt <= high:
                    override_exp = original_exp.copy()
                    override_exp.update(exp_dict)
                    cfg.exposure_by_filter = override_exp
                    break

            candidates = []
            for _, row in group.iterrows():
                allowed = allowed_filters_for_window(
                    mag_lookup.get(row["Name"], {}),
                    sun_alt,
                    row["_moon_alt"],
                    row["_moon_phase"],
                    row["_moon_sep"],
                    airmass_from_alt_deg(row["max_alt_deg"]),
                    cfg.fwhm_eff or 0.7,
                )
                allowed = [f for f in allowed if f in cfg.filters]
                # Enforce sun_alt_policy: drop filters disallowed at this Sun altitude
                allowed_policy = allowed_filters_for_sun_alt(sun_alt, cfg)
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
                first = pick_first_filter_for_target(
                    row["Name"],
                    row.get("SN_type_raw"),
                    tracker,
                    allowed,
                    cfg,
                    sun_alt_deg=sun_alt,
                    moon_sep_ok=moon_sep_ok,
                    current_mag=mag_lookup.get(row["Name"]),
                    current_filter=current_filter_by_window.get(idx_w),
                )
                if first is None:
                    continue
                cand = {
                    "Name": row["Name"],
                    "RA_deg": row["RA_deg"],
                    "Dec_deg": row["Dec_deg"],
                    "best_time_utc": row["best_time_utc"],
                    "max_alt_deg": row["max_alt_deg"],
                    "priority_score": row["priority_score"],
                    "first_filter": first,
                    "sn_type": row.get("SN_type_raw"),
                    "allowed": allowed,
                    "moon_sep_ok": moon_sep_ok,
                    "moon_sep": float(row["_moon_sep"]),
                }
                candidates.append(cand)
            if not candidates:
                if override_exp is not None:
                    cfg.exposure_by_filter = original_exp
                continue

            batch_order = [
                f
                for f in ["y", "z", "i", "r", "g", "u"]
                if any(c["first_filter"] == f for c in candidates)
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

            for filt in batch_order:
                batch = [c for c in candidates if c["first_filter"] == filt]
                while batch:
                    # select next target based on filter-aware cost
                    costs: List[float] = []
                    first_choices: List[str] = []
                    for t in batch:
                        first_tmp = pick_first_filter_for_target(
                            t["Name"],
                            t.get("sn_type"),
                            tracker,
                            t["allowed"],
                            cfg,
                            sun_alt_deg=sun_alt,
                            moon_sep_ok=t["moon_sep_ok"],
                            current_mag=mag_lookup.get(t["Name"]),
                            current_filter=state,
                        )
                        first_choices.append(first_tmp)
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
                        if state is not None and state != first_tmp:
                            cost += cfg.filter_change_s
                        costs.append(cost)
                    j = int(np.argmin(costs))
                    t = batch.pop(j)
                    first = first_choices.pop(j)
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
                    filters_pref = [first] + [x for x in t["allowed"] if x != first]
                    filters_used, timing = choose_filters_with_cap(
                        filters_pref,
                        sep,
                        cfg.per_sn_cap_s,
                        cfg,
                        current_filter=state,
                        max_filters_per_visit=cfg.max_filters_per_visit,
                    )
                    if window_sum + timing["total_s"] > cap_s:
                        continue
                    window_sum += timing["total_s"]
                    window_filter_change_s += timing.get("filter_changes_s", 0.0)

                    epochs = []
                    for f in filters_used:
                        exp_s = timing.get("exp_times", {}).get(
                            f, cfg.exposure_by_filter.get(f, 0.0)
                        )
                        flags = timing.get("flags_by_filter", {}).get(f, set())
                        alt_deg = float(t["max_alt_deg"])
                        air = airmass_from_alt_deg(alt_deg)
                        mjd = (
                            Time(t["best_time_utc"]).mjd
                            if isinstance(t["best_time_utc"], (datetime, pd.Timestamp))
                            else np.nan
                        )
                        if sky_provider:
                            sky_mag = sky_provider.sky_mag(
                                mjd,
                                t["RA_deg"],
                                t["Dec_deg"],
                                f,
                                airmass_from_alt_deg(alt_deg),
                            )
                        else:
                            sky_mag = sky_mag_arcsec2(f, sky_cfg)
                        eph = compute_epoch_photom(f, exp_s, alt_deg, sky_mag, phot_cfg)
                        window_skymags.append(sky_mag)
                        if writer:
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
                        pernight_rows.append(
                            {
                                "date": day.date().isoformat(),
                                "twilight_window": window_label_out,
                                "SN": t["Name"],
                                "RA_deg": round(t["RA_deg"], 6),
                                "Dec_deg": round(t["Dec_deg"], 6),
                                "best_twilight_time_utc": (
                                    pd.Timestamp(t["best_time_utc"])
                                    .tz_convert("UTC")
                                    .isoformat()
                                    if isinstance(t["best_time_utc"], pd.Timestamp)
                                    else str(t["best_time_utc"])
                                ),
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
                                "slew_s": round(timing["slew_s"], 2),
                                "cross_filter_change_s": round(
                                    timing.get("cross_filter_change_s", 0.0), 2
                                ),
                                "filter_changes_s": round(
                                    timing.get("filter_changes_s", 0.0), 2
                                ),
                                "readout_s": round(timing["readout_s"], 2),
                                "exposure_s": round(timing["exposure_s"], 2),
                                "total_time_s": round(timing["total_s"], 2),
                            }
                        )

                    if writer and epochs:
                        writer.start_libid(
                            libid_counter,
                            t["RA_deg"],
                            t["Dec_deg"],
                            nobs=len(epochs),
                            comment=t["Name"],
                        )
                        libid_counter += 1
                        for epoch in epochs:
                            writer.add_epoch(**epoch)
                        writer.end_libid()
                    if filters_used:
                        if state is not None and filters_used[0] != state:
                            swap_count_by_window[idx_w] += 1
                        current_filter_by_window[idx_w] = filters_used[0]
                        current_filter_by_window[idx_w] = filters_used[-1]
                        state = current_filter_by_window[idx_w]
                        filters_used_set.update(filters_used)
                    internal_changes += max(0, len(filters_used) - 1)
                    window_slew_times.append(timing["slew_s"])
                    window_airmasses.append(air)
                    prev = t
                    tracker.record_detection(
                        t["Name"], timing["exposure_s"], filters_used
                    )

            used_filters_csv = ",".join(sorted(filters_used_set))
            win = windows[idx_w]
            start_utc = pd.Timestamp(win["start"]).tz_convert("UTC").isoformat()
            end_utc = pd.Timestamp(win["end"]).tz_convert("UTC").isoformat()
            dur_s = (win["end"] - win["start"]).total_seconds()
            mid = win["start"] + (win["end"] - win["start"]) / 2
            mid_utc = pd.Timestamp(mid).tz_convert("UTC").isoformat()
            sun_alt_mid = float(
                get_sun(Time(mid))
                .transform_to(AltAz(location=site, obstime=Time(mid)))
                .alt.to(u.deg)
                .value
            )
            policy_filters_mid = ",".join(allowed_filters_for_sun_alt(sun_alt_mid, cfg))
            cap_source = cap_source_by_window.get(idx_w, "none")
            alts = [
                r["alt_deg"]
                for r in pernight_rows
                if r["date"] == day.date().isoformat()
                and r["twilight_window"] == window_label_out
            ]
            nights_rows.append(
                {
                    "date": day.date().isoformat(),
                    "twilight_window": window_label_out,
                    "n_candidates": int(len(group)),
                    "n_planned": int(
                        len(
                            [
                                r
                                for r in pernight_rows
                                if (
                                    r["date"] == day.date().isoformat()
                                    and r["twilight_window"] == window_label_out
                                )
                            ]
                        )
                    ),
                    "sum_time_s": round(window_sum, 1),
                    "window_cap_s": int(cap_s),
                    "swap_count": int(swap_count_by_window.get(idx_w, 0)),
                    "internal_filter_changes": int(internal_changes),
                    "filter_change_s_total": round(window_filter_change_s, 1),
                    "mean_slew_s": (
                        float(np.mean(window_slew_times)) if window_slew_times else 0.0
                    ),
                    "median_airmass": (
                        float(np.median(window_airmasses)) if window_airmasses else 0.0
                    ),
                    "loaded_filters": ",".join(cfg.filters),
                    "filters_used_csv": used_filters_csv,
                    "window_start_utc": start_utc,
                    "window_end_utc": end_utc,
                    "window_duration_s": int(dur_s),
                    "window_mid_utc": mid_utc,
                    "sun_alt_mid_deg": round(sun_alt_mid, 2),
                    "policy_filters_mid_csv": policy_filters_mid,
                    "window_utilization": round(window_sum / max(1.0, dur_s), 4),
                    "cap_utilization": round(window_sum / max(1.0, cap_s), 4),
                    "cap_source": cap_source,
                    "median_sky_mag_arcsec2": (
                        float(np.median(window_skymags)) if window_skymags else np.nan
                    ),
                    "median_alt_deg": (float(np.median(alts)) if alts else np.nan),
                }
            )
            if override_exp is not None:
                cfg.exposure_by_filter = original_exp

        if verbose:
            planned_today = [
                r for r in pernight_rows if r["date"] == day.date().isoformat()
            ]
            filters_csv = ",".join(cfg.filters)
            eve_str = (
                pd.Timestamp(evening_tw).tz_convert("UTC").isoformat()
                if evening_tw
                else "na"
            )
            morn_str = (
                pd.Timestamp(morning_tw).tz_convert("UTC").isoformat()
                if morning_tw
                else "na"
            )
            print(
                f"{day.date().isoformat()}: eligible={len(subset)} visible={len(visible)} planned_total={len(planned_today)} "
                f"evening_twilight={eve_str} morning_twilight={morn_str} filters={filters_csv}"
            )

    pernight_df = pd.DataFrame(pernight_rows)
    nights_df = pd.DataFrame(nights_rows)
    pernight_path = (
        Path(outdir)
        / f"lsst_twilight_plan_{start.isoformat()}_to_{end.isoformat()}.csv"
    )
    nights_path = (
        Path(outdir)
        / f"lsst_twilight_summary_{start.isoformat()}_to_{end.isoformat()}.csv"
    )
    pernight_df.to_csv(pernight_path, index=False)
    nights_df.to_csv(nights_path, index=False)
    if writer:
        writer.close()
    print(f"Wrote:\n  {pernight_path}\n  {nights_path}")
    print(f"Rows: per-SN={len(pernight_df)}, nights*windows={len(nights_df)}")
    return pernight_df, nights_df
