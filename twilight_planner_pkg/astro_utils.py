from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
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

def twilight_windows_astro(date_utc: datetime, loc: EarthLocation) -> List[Tuple[datetime, datetime]]:
    """Return list of (start_utc, end_utc) for periods when -18° < SunAlt < 0° around a given date."""
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
    c1 = SkyCoord(ra1*u.deg, dec1*u.deg)
    c2 = SkyCoord(ra2*u.deg, dec2*u.deg)
    return c1.separation(c2).deg

def slew_time_seconds(sep_deg: float, *, small_deg: float, small_time: float,
                      rate_deg_per_s: float, settle_s: float) -> float:
    if sep_deg <= 0:
        return 0.0
    if sep_deg <= small_deg:
        t = small_time
    else:
        t = small_time + (sep_deg - small_deg) / max(rate_deg_per_s, 1e-3)
    return t + settle_s

def per_sn_time_seconds(filters, sep_deg: float, cfg: PlannerConfig):
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
    used = []
    _slew = slew_time_seconds(
        sep_deg,
        small_deg=cfg.slew_small_deg,
        small_time=cfg.slew_small_time_s,
        rate_deg_per_s=cfg.slew_rate_deg_per_s,
        settle_s=cfg.slew_settle_s,
    )
    for f in filters:
        trial = used + [f]
        exptime = sum(cfg.exposure_by_filter.get(x, 0.0) for x in trial)
        readout = cfg.readout_s * len(trial)
        fchanges = cfg.filter_change_s * max(0, len(trial)-1)
        total = _slew + exptime + readout + fchanges
        if total <= cap_s:
            used = trial
        else:
            break
    if not used and filters:
        used = [filters[0]]
    total, slew, exptime, readout, fchanges = per_sn_time_seconds(used, sep_deg, cfg)
    return used, {"total_s": total, "slew_s": slew, "exposure_s": exptime, "readout_s": readout, "filter_changes_s": fchanges}

def parse_sn_type_to_window_days(type_str: str, cfg: PlannerConfig) -> int:
    import math
    if not isinstance(type_str, str) or not type_str.strip():
        return int(math.ceil(1.2 * cfg.default_typical_days))
    s = type_str.lower()
    for key, days in cfg.typical_days_by_type.items():
        if str(key).lower() in s:
            return int(math.ceil(1.2 * days))
    return int(math.ceil(1.2 * cfg.default_typical_days))

def _best_time_with_moon(sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg):
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
