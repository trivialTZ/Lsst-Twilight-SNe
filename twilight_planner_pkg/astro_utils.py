from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Sequence
import numpy as np
import astropy.units as u
import warnings
from astropy.coordinates import (
    SkyCoord, AltAz, EarthLocation, get_sun, get_body
)
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyWarning

try:
    from astropy.coordinates.baseframe import NonRotationTransformationWarning  # most versions
except Exception:
    try:
        from astropy.coordinates.transformations import NonRotationTransformationWarning
    except Exception:
        class NonRotationTransformationWarning(AstropyWarning):
            pass

warnings.filterwarnings("ignore", category=AstropyDeprecationWarning)
warnings.filterwarnings("ignore", category=NonRotationTransformationWarning)
warnings.filterwarnings("ignore",
                        message="Angular separation can depend on the direction of the transformation",
                        category=Warning,
                        module="astropy")

from .config import PlannerConfig


def airmass_from_alt_deg(alt_deg: float) -> float:
    """Convert an altitude angle to airmass.

    Parameters
    ----------
    alt_deg : float
        Altitude above the horizon in degrees.

    Returns
    -------
    float
        Airmass estimated using a secant approximation.
    """

    import numpy as np, math
    z = np.deg2rad(max(0.0, min(90.0, 90.0 - float(alt_deg))))
    return 1.0 / max(1e-4, math.cos(z))

def twilight_windows_astro(date_utc: datetime, loc: EarthLocation) -> List[Tuple[datetime, datetime]]:
    """Compute astronomical twilight windows for a given date and location.

    Parameters
    ----------
    date_utc : datetime
        Reference date in UTC.
    loc : astropy.coordinates.EarthLocation
        Observatory location.

    Returns
    -------
    list[tuple[datetime, datetime]]
        Sorted list of ``(start, end)`` UTC times where ``-18° <`` Sun altitude ``< 0°``.
    """
    start = date_utc.replace(tzinfo=timezone.utc) - timedelta(hours=12)
    times = Time([start + timedelta(minutes=i) for i in range(48*60)])
    altaz = AltAz(obstime=times, location=loc)
    sun_alt = get_sun(times).transform_to(altaz).alt.to(u.deg).value
    mask = (sun_alt > -18.0) & (sun_alt < 0.0)
    edges = np.where(np.diff(mask.astype(int)) != 0)[0]
    segments, prev = [], 0
    for e in edges:
        segments.append((prev, e))
        prev = e + 1
    segments.append((prev, len(mask)-1))
    windows = []
    for a, b in segments:
        if np.any(mask[a:b+1]):
            i0 = a + int(np.argmax(mask[a:b+1]))
            i1 = i0
            while i0 > a and mask[i0-1]:
                i0 -= 1
            while i1 < b and mask[i1+1]:
                i1 += 1
            windows.append((Time(times[i0]).to_datetime(timezone.utc),
                            Time(times[i1]).to_datetime(timezone.utc)))
    windows.sort(key=lambda w: w[0])
    return windows

def great_circle_sep_deg(ra1, dec1, ra2, dec2) -> float:
    """Compute on-sky separation between two coordinates.

    Parameters
    ----------
    ra1, dec1, ra2, dec2 : float
        Coordinates in degrees.

    Returns
    -------
    float
        Great-circle separation in degrees.
    """
    c1 = SkyCoord(ra1*u.deg, dec1*u.deg)
    c2 = SkyCoord(ra2*u.deg, dec2*u.deg)
    return c1.separation(c2).deg

def slew_time_seconds(sep_deg: float, *, small_deg: float, small_time: float,
                      rate_deg_per_s: float, settle_s: float) -> float:
    """Estimate telescope slew time including settle time.

    Parameters
    ----------
    sep_deg : float
        Angular distance between targets in degrees.
    small_deg : float
        Distance below which a fixed small slew time is used.
    small_time : float
        Time in seconds for slews ``<= small_deg``.
    rate_deg_per_s : float
        Slew rate for larger moves.
    settle_s : float
        Additional settling overhead in seconds.

    Returns
    -------
    float
        Total slew time in seconds.
    """
    if sep_deg <= 0:
        return 0.0
    if sep_deg <= small_deg:
        t = small_time
    else:
        t = small_time + (sep_deg - small_deg) / max(rate_deg_per_s, 1e-3)
    return t + settle_s

def per_sn_time_seconds(filters, sep_deg: float, cfg: PlannerConfig):
    """Compute total time budget for observing one supernova.

    Parameters
    ----------
    filters : Sequence[str]
        Filters to use for the target.
    sep_deg : float
        Slew distance from the previous target in degrees.
    cfg : PlannerConfig
        Configuration with exposure and overhead settings.

    Returns
    -------
    tuple
        ``(total_s, slew_s, exposure_s, readout_s, filter_changes_s)`` in seconds.
    """
    slew = slew_time_seconds(
        sep_deg,
        small_deg=cfg.slew_small_deg,
        small_time=cfg.slew_small_time_s,
        rate_deg_per_s=cfg.slew_rate_deg_per_s,
        settle_s=cfg.slew_settle_s,
    )
    exptime = sum(cfg.exposure_by_filter.get(f, 0.0) for f in filters)
    readout = cfg.readout_s * len(filters)
    fchanges = cfg.filter_change_s * max(0, len(filters)-1)
    total = slew + exptime + readout + fchanges
    return total, slew, exptime, readout, fchanges

def choose_filters_with_cap(filters, sep_deg: float, cap_s: float, cfg: PlannerConfig):
    """Select a subset of filters whose total time fits within a cap.

    Parameters
    ----------
    filters : Sequence[str]
        Candidate filters in priority order.
    sep_deg : float
        Slew distance from the previous target in degrees.
    cap_s : float
        Maximum allowed time in seconds.
    cfg : PlannerConfig
        Timing configuration.

    Returns
    -------
    tuple
        ``(used_filters, timing)`` where ``timing`` matches
        :func:`per_sn_time_seconds` keys.
    """
    from .photom_rubin import PhotomConfig, cap_exposure_for_saturation
    from .sky_model import SkyModelConfig, sky_mag_arcsec2

    used: list[str] = []

    if not cfg.allow_filter_changes_in_twilight:
        filters = list(filters)[:1]
    _slew = slew_time_seconds(
        sep_deg,
        small_deg=cfg.slew_small_deg,
        small_time=cfg.slew_small_time_s,
        rate_deg_per_s=cfg.slew_rate_deg_per_s,
        settle_s=cfg.slew_settle_s,
    )

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

    def capped_exp(f: str) -> float:
        base = cfg.exposure_by_filter.get(f, 0.0)
        if cfg.current_mag_by_filter and f in cfg.current_mag_by_filter:
            alt = cfg.current_alt_deg if cfg.current_alt_deg is not None else cfg.min_alt_deg
            if cfg.sky_provider:
                sky = cfg.sky_provider.sky_mag(None, None, None, f, airmass_from_alt_deg(alt))
            else:
                sky = sky_mag_arcsec2(f, sky_cfg)
            base = cap_exposure_for_saturation(
                f,
                base,
                alt,
                cfg.current_mag_by_filter[f],
                sky,
                phot_cfg,
                min_exp_s=1.0,
            )
        return base

    for f in filters:
        trial = used + [f]
        exp_times = {x: capped_exp(x) for x in trial}
        exptime = sum(exp_times.values())
        readout = cfg.readout_s * len(trial)
        fchanges = cfg.filter_change_s * max(0, len(trial) - 1)
        total = _slew + exptime + readout + fchanges
        if total <= cap_s:
            used = trial
        else:
            break

    if not used and filters:
        used = [filters[0]]

    exp_times = {x: capped_exp(x) for x in used}
    exptime = sum(exp_times.values())
    readout = cfg.readout_s * len(used)
    fchanges = cfg.filter_change_s * max(0, len(used) - 1)
    total = _slew + exptime + readout + fchanges
    timing = {
        "total_s": total,
        "slew_s": _slew,
        "exposure_s": exptime,
        "readout_s": readout,
        "filter_changes_s": fchanges,
        "exp_times": exp_times,
    }
    return used, timing

def parse_sn_type_to_window_days(type_str: str, cfg: PlannerConfig) -> int:
    """Estimate the number of days a supernova remains observable.

    Parameters
    ----------
    type_str : str
        Text description of the SN type.
    cfg : PlannerConfig
        Configuration mapping types to typical lifetimes.

    Returns
    -------
    int
        Observation window in days, scaled by ``1.2`` for safety.
    """
    import math
    if not isinstance(type_str, str) or not type_str.strip():
        return int(math.ceil(1.2 * cfg.default_typical_days))
    s = type_str.lower()
    for key, days in cfg.typical_days_by_type.items():
        if str(key).lower() in s:
            return int(math.ceil(1.2 * days))
    return int(math.ceil(1.2 * cfg.default_typical_days))

def _best_time_with_moon(sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg):
    """Find the best time within a window that meets altitude and moon constraints.

    Parameters
    ----------
    sc : astropy.coordinates.SkyCoord
        Target coordinates.
    window : tuple[datetime, datetime]
        Candidate twilight window.
    loc : astropy.coordinates.EarthLocation
        Observatory location.
    step_min : int
        Sampling step size in minutes.
    min_alt_deg : float
        Minimum altitude requirement in degrees.
    min_moon_sep_deg : float
        Minimum separation from the Moon in degrees.

    Returns
    -------
    tuple
        ``(best_alt_deg, best_time_utc)`` where ``best_time_utc`` is a
        ``datetime`` or ``None`` if no suitable time exists.
    """
    t0, t1 = window
    if t1 <= t0:
        return -np.inf, None
    n = 1 + int((t1 - t0).total_seconds() // (step_min * 60))
    ts = Time([t0 + timedelta(minutes=step_min * i) for i in range(n)])
    altaz = AltAz(obstime=ts, location=loc)

    sc_altaz   = sc.transform_to(altaz)
    alt        = sc_altaz.alt.to(u.deg).value
    moon_altaz = get_body("moon", ts, location=loc).transform_to(altaz)

    moon_alt = moon_altaz.alt.to(u.deg).value
    sep_deg  = sc_altaz.separation(moon_altaz).deg

    ok = (alt >= min_alt_deg) & ((moon_alt < 0.0) | (sep_deg >= min_moon_sep_deg))
    if not np.any(ok):
        return -np.inf, None
    j = int(np.argmax(alt * ok))
    return float(alt[j]), ts[j].to_datetime(timezone.utc)
