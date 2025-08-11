from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

from .config import PlannerConfig
from .io_utils import standardize_columns, extract_current_mags
from .astro_utils import (
    twilight_windows_astro, great_circle_sep_deg, choose_filters_with_cap,
    per_sn_time_seconds, parse_sn_type_to_window_days, _best_time_with_moon, airmass_from_alt_deg
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

    filters = cfg.filters or []
    if cfg.carousel_capacity and len(filters) > cfg.carousel_capacity:
        print(f"WARNING: Requesting {len(filters)} filters but carousel holds only {cfg.carousel_capacity}. Proceeding for planning only.")

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

            targets = group[["Name","RA_deg","Dec_deg","best_time_utc","max_alt_deg","priority_score"]].to_dict(orient="records")
            ordered = [targets.pop(0)]
            while targets:
                last = ordered[-1]
                dists = [great_circle_sep_deg(last["RA_deg"], last["Dec_deg"], t["RA_deg"], t["Dec_deg"]) for t in targets]
                j = int(np.argmin(dists))
                ordered.append(targets.pop(j))

            cap_s = window_caps.get(idx_w, 0.0)
            window_sum = 0.0
            prev = None
            for t in ordered:
                sep = 0.0 if prev is None else great_circle_sep_deg(prev["RA_deg"], prev["Dec_deg"], t["RA_deg"], t["Dec_deg"])
                cfg.current_mag_by_filter = mag_lookup.get(t["Name"])
                cfg.current_alt_deg = t["max_alt_deg"]
                filters_used, timing = choose_filters_with_cap(cfg.filters, sep, cfg.per_sn_cap_s, cfg)
                if window_sum + timing["total_s"] > cap_s:
                    continue
                window_sum += timing["total_s"]
                if writer:
                    writer.start_libid(libid_counter, t["RA_deg"], t["Dec_deg"], comment=t["Name"])
                    libid_counter += 1
                for f in filters_used:
                    exp_s = timing.get("exp_times", {}).get(f, cfg.exposure_by_filter.get(f, 0.0))
                    alt_deg = float(t["max_alt_deg"])
                    mjd = Time(t["best_time_utc"]).mjd if isinstance(t["best_time_utc"], (datetime, pd.Timestamp)) else np.nan
                    if sky_provider:
                        sky_mag = sky_provider.sky_mag(mjd, t["RA_deg"], t["Dec_deg"], f, airmass_from_alt_deg(alt_deg))
                    else:
                        sky_mag = sky_mag_arcsec2(f, sky_cfg)
                    eph = compute_epoch_photom(f, exp_s, alt_deg, sky_mag, phot_cfg)
                    if writer:
                        writer.add_epoch(
                            mjd=mjd,
                            band=f,
                            gain=eph.GAIN,
                            rdnoise=eph.RDNOISE,
                            skysig=eph.SKYSIG,
                            psf1=cfg.fwhm_eff.get(f, 0.83) if cfg.fwhm_eff else 0.83,
                            psf2=0.0,
                            psfrat=0.0,
                            zpavg=eph.ZPTAVG,
                            zperr=eph.ZPTERR,
                            mag=-99.0,
                        )
                    pernight_rows.append({
                        "date": day.date().isoformat(),
                        "twilight_window": window_labels.get(idx_w, f"W{idx_w}"),
                        "SN": t["Name"],
                        "RA_deg": round(t["RA_deg"], 6),
                        "Dec_deg": round(t["Dec_deg"], 6),
                        "best_twilight_time_utc": pd.Timestamp(t["best_time_utc"]).tz_convert("UTC").isoformat() if isinstance(t["best_time_utc"], pd.Timestamp) else str(t["best_time_utc"]),
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
                    })
                prev = t
                tracker.record_detection(t["Name"], timing["exposure_s"], filters_used)

            nights_rows.append({
                "date": day.date().isoformat(),
                "twilight_window": window_labels.get(idx_w, f"W{idx_w}"),
                "n_candidates": int(len(group)),
                "n_planned": int(len([r for r in pernight_rows if (r['date']==day.date().isoformat() and r['twilight_window']==window_labels.get(idx_w, f'W{idx_w}'))])),
                "sum_time_s": round(window_sum, 1),
                "window_cap_s": int(cap_s),
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
