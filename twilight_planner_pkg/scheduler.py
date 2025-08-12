from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
try:
    from tqdm.notebook import tqdm as _tqdm  # type: ignore
    import ipywidgets as _ipyw  # noqa: F401
    tqdm = _tqdm
except Exception:  # pragma: no cover - fallback
    from tqdm.auto import tqdm

import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
import warnings

warnings.filterwarnings("ignore", message=".*get_moon.*deprecated.*")
warnings.filterwarnings(
    "ignore",
    message=".*transforming other coordinates from <GCRS Frame.*>",
)
warnings.filterwarnings(
    "ignore",
    message="Angular separation can depend on the direction of the transformation",
)

from .config import PlannerConfig
from .io_utils import standardize_columns, extract_current_mags
from .astro_utils import (
    twilight_windows_astro,
    great_circle_sep_deg,
    choose_filters_with_cap,
    per_sn_time_seconds,
    parse_sn_type_to_window_days,
    _best_time_with_moon,
    airmass_from_alt_deg,
    allowed_filters_for_sun_alt,
    slew_time_seconds,
    pick_first_filter_for_target,
)
from .priority import PriorityTracker
from .simlib_writer import SimlibWriter, SimlibHeader
from .photom_rubin import PhotomConfig, compute_epoch_photom
from .sky_model import (
    SkyModelConfig,
    sky_mag_arcsec2,
    RubinSkyProvider,
    SimpleSkyProvider,
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
        Configuration with site, filter, and timing parameters.
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
    sky_cfg = SkyModelConfig(
        dark_sky_mag=cfg.dark_sky_mag,
        twilight_delta_mag=cfg.twilight_delta_mag,
    )
    try:
        sky_provider = RubinSkyProvider()
    except Exception:
        sky_provider = SimpleSkyProvider(sky_cfg)
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

    req_sep = (
        max(cfg.min_moon_sep_by_filter.get(f, 0.0) for f in filters)
        if cfg.require_single_time_for_all_filters and cfg.min_moon_sep_by_filter
        else max(cfg.min_moon_sep_by_filter.values()) if cfg.min_moon_sep_by_filter else 0.0
    )

    site = EarthLocation(lat=cfg.lat_deg*u.deg, lon=cfg.lon_deg*u.deg, height=cfg.height_m*u.m)
    start = pd.to_datetime(start_date, utc=True).date()
    end   = pd.to_datetime(end_date,   utc=True).date()

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

    for day in nights_iter:
        day_utc = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        windows = twilight_windows_astro(day_utc, site)
        if not windows:
            continue
        window_caps = {0: cfg.morning_cap_s, 1: cfg.evening_cap_s}
        window_labels = {0: "morning", 1: "evening"}

        cutoff = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)
        lifetime_days_each = df["SN_type_raw"].apply(lambda t: parse_sn_type_to_window_days(t, cfg))
        min_allowed_disc_each = cutoff - lifetime_days_each.apply(lambda d: timedelta(days=int(d)))
        has_disc = df["discovery_datetime"].notna()
        subset = df[has_disc & (df["discovery_datetime"] <= cutoff) & (df["discovery_datetime"] >= min_allowed_disc_each)].copy()
        if subset.empty:
            if verbose:
                print(f"{day.date().isoformat()}: 0 eligible")
            continue

        best_alts, best_times, best_winidx = [], [], []
        for _, row in subset.iterrows():
            sc = SkyCoord(row["RA_deg"]*u.deg, row["Dec_deg"]*u.deg, frame="icrs")
            max_alt, max_time, max_idx = -999.0, None, None
            for idx_w, w in enumerate(windows):
                alt_deg, t_utc = _best_time_with_moon(sc, w, site, cfg.twilight_step_min, cfg.min_alt_deg, req_sep)
                if alt_deg > max_alt:
                    max_alt, max_time, max_idx = alt_deg, t_utc, idx_w
            best_alts.append(max_alt)
            best_times.append(max_time if max_time is not None else pd.NaT)
            best_winidx.append(max_idx if max_time is not None else -1)

        subset["max_alt_deg"] = best_alts
        subset["best_time_utc"] = best_times
        subset["best_window_index"] = best_winidx

        visible = subset[(subset["best_time_utc"].notna()) & (subset["max_alt_deg"] >= cfg.min_alt_deg) & (subset["best_window_index"] >= 0)].copy()
        if visible.empty:
            if verbose:
                print(f"{day.date().isoformat()}: 0 visible")
            continue

        visible["priority_score"] = visible.apply(
            lambda r: tracker.score(r["Name"], r.get("SN_type_raw"), cfg.priority_strategy), axis=1
        )
        visible.sort_values(["priority_score", "max_alt_deg"], ascending=[False, False], inplace=True)
        top_global = visible.head(int(cfg.max_sn_per_night)).copy()

        for idx_w in sorted(set(top_global["best_window_index"].values)):
            group = top_global[top_global["best_window_index"] == idx_w].copy()
            if group.empty:
                continue

            win = windows[idx_w]
            mid = win[0] + (win[1] - win[0]) / 2
            sun_alt = (
                get_sun(Time(mid)).transform_to(AltAz(location=site, obstime=Time(mid))).alt.to(u.deg).value
            )
            allowed = allowed_filters_for_sun_alt(sun_alt, cfg)
            candidates = []
            for _, row in group.iterrows():
                first = pick_first_filter_for_target(
                    row["Name"],
                    row.get("SN_type_raw"),
                    tracker,
                    allowed,
                    cfg,
                    current_filter=None,
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
                }
                candidates.append(cand)
            if not candidates:
                continue

            batch_order = [f for f in ["y", "z", "i", "r", "g", "u"] if any(c["first_filter"] == f for c in candidates) and f in allowed]

            cap_s = window_caps.get(idx_w, 0.0)
            window_sum = 0.0
            prev = None
            current_filter = None
            cross_changes = 0
            internal_changes = 0
            window_slews: List[float] = []
            window_alts: List[float] = []

            for filt in batch_order:
                batch = [c for c in candidates if c["first_filter"] == filt]
                while batch:
                    # select next target based on filter-aware cost
                    costs = []
                    for t in batch:
                        sep = 0.0 if prev is None else great_circle_sep_deg(prev["RA_deg"], prev["Dec_deg"], t["RA_deg"], t["Dec_deg"])
                        cost = slew_time_seconds(
                            sep,
                            small_deg=cfg.slew_small_deg,
                            small_time=cfg.slew_small_time_s,
                            rate_deg_per_s=cfg.slew_rate_deg_per_s,
                            settle_s=cfg.slew_settle_s,
                        )
                        if current_filter is not None and current_filter != t["first_filter"]:
                            cost += cfg.filter_change_time_s
                        costs.append(cost)
                    j = int(np.argmin(costs))
                    t = batch.pop(j)
                    sep = 0.0 if prev is None else great_circle_sep_deg(prev["RA_deg"], prev["Dec_deg"], t["RA_deg"], t["Dec_deg"])
                    cfg.current_mag_by_filter = mag_lookup.get(t["Name"])
                    cfg.current_alt_deg = t["max_alt_deg"]
                    filters_pref = [t["first_filter"]] + [x for x in allowed if x != t["first_filter"]]
                    filters_used, timing = choose_filters_with_cap(
                        filters_pref,
                        sep,
                        cfg.per_sn_cap_s,
                        cfg,
                        current_filter=current_filter,
                        max_filters_per_visit=cfg.max_filters_per_visit,
                    )
                    if window_sum + timing["total_s"] > cap_s:
                        continue
                    window_sum += timing["total_s"]

                    epochs = []
                    for f in filters_used:
                        exp_s = timing.get("exp_times", {}).get(f, cfg.exposure_by_filter.get(f, 0.0))
                        alt_deg = float(t["max_alt_deg"])
                        mjd = (
                            Time(t["best_time_utc"]).mjd
                            if isinstance(t["best_time_utc"], (datetime, pd.Timestamp))
                            else np.nan
                        )
                        if sky_provider:
                            sky_mag = sky_provider.sky_mag(
                                mjd, t["RA_deg"], t["Dec_deg"], f, airmass_from_alt_deg(alt_deg)
                            )
                        else:
                            sky_mag = sky_mag_arcsec2(f, sky_cfg)
                        eph = compute_epoch_photom(f, exp_s, alt_deg, sky_mag, phot_cfg)
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
                                "twilight_window": window_labels.get(idx_w, f"W{idx_w}"),
                                "SN": t["Name"],
                                "RA_deg": round(t["RA_deg"], 6),
                                "Dec_deg": round(t["Dec_deg"], 6),
                                "best_twilight_time_utc": pd.Timestamp(t["best_time_utc"]).tz_convert("UTC").isoformat()
                                if isinstance(t["best_time_utc"], pd.Timestamp)
                                else str(t["best_time_utc"]),
                                "filter": f,
                                "t_exp_s": round(exp_s, 1),
                                "airmass": round(airmass_from_alt_deg(alt_deg), 3),
                                "alt_deg": round(alt_deg, 2),
                                "sky_mag_arcsec2": round(sky_mag, 2),
                                "ZPT": round(eph.ZPTAVG, 3),
                                "SKYSIG": round(eph.SKYSIG, 3),
                                "NEA_pix": round(eph.NEA_pix, 2),
                                "RDNOISE": round(eph.RDNOISE, 2),
                                "GAIN": round(eph.GAIN, 2),
                                "saturation_guard_applied": exp_s < cfg.exposure_by_filter.get(f, exp_s) - 1e-6,
                                "priority_score": round(float(t["priority_score"]), 2),
                                "slew_s": round(timing["slew_s"], 2),
                                "cross_filter_change_s": round(timing.get("cross_filter_change_s", 0.0), 2),
                                "filter_changes_s": round(timing.get("filter_changes_s", 0.0), 2),
                                "readout_s": round(timing["readout_s"], 2),
                                "exposure_s": round(timing["exposure_s"], 2),
                                "total_time_s": round(timing["total_s"], 2),
                            }
                        )

                    if writer and epochs:
                        writer.start_libid(
                            libid_counter, t["RA_deg"], t["Dec_deg"], nobs=len(epochs), comment=t["Name"]
                        )
                        libid_counter += 1
                        for epoch in epochs:
                            writer.add_epoch(**epoch)
                        writer.end_libid()
                    if timing.get("cross_filter_change_s", 0) > 0:
                        cross_changes += 1
                    internal_changes += max(0, len(filters_used) - 1)
                    window_slews.append(sep)
                    window_alts.append(t["max_alt_deg"])
                    prev = t
                    current_filter = filters_used[-1] if filters_used else current_filter
                    tracker.record_detection(t["Name"], timing["exposure_s"], filters_used)

            nights_rows.append({
                "date": day.date().isoformat(),
                "twilight_window": window_labels.get(idx_w, f"W{idx_w}"),
                "n_candidates": int(len(group)),
                "n_planned": int(len([r for r in pernight_rows if (r['date']==day.date().isoformat() and r['twilight_window']==window_labels.get(idx_w, f'W{idx_w}'))])),
                "sum_time_s": round(window_sum, 1),
                "window_cap_s": int(cap_s),
                "cross_filter_changes": int(cross_changes),
                "internal_filter_changes": int(internal_changes),
                "avg_slew_deg": float(np.mean(window_slews)) if window_slews else 0.0,
                "median_alt_deg": float(np.median(window_alts)) if window_alts else 0.0,
                "loaded_filters": ",".join(allowed),
            })

        if verbose:
            planned_today = [r for r in pernight_rows if r["date"] == day.date().isoformat()]
            print(f"{day.date().isoformat()}: eligible={len(subset)} visible={len(visible)} planned_total={len(planned_today)}")

    pernight_df = pd.DataFrame(pernight_rows)
    nights_df = pd.DataFrame(nights_rows)
    pernight_path = Path(outdir) / f"lsst_twilight_plan_{start.isoformat()}_to_{end.isoformat()}.csv"
    nights_path   = Path(outdir) / f"lsst_twilight_summary_{start.isoformat()}_to_{end.isoformat()}.csv"
    pernight_df.to_csv(pernight_path, index=False)
    nights_df.to_csv(nights_path, index=False)
    if writer:
        writer.close()
    print(f"Wrote:\n  {pernight_path}\n  {nights_path}")
    print(f"Rows: per-SN={len(pernight_df)}, nights*windows={len(nights_df)}")
    return pernight_df, nights_df
