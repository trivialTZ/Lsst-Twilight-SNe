from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Sequence
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
    """Convert altitude to relative airmass using Kasten & Young (1989).

    The implementation follows the "Simple" airmass approximation from
    Kasten & Young (Applied Optics, 1989).  It clamps the output to ``>=1`` and
    avoids evaluating below the horizon where the formula would diverge.
    """

    alt = float(alt_deg)
    if alt <= 0.0:
        return float("inf")
    if alt >= 90.0:
        return 1.0
    z = 90.0 - alt
    denom = np.cos(np.deg2rad(z)) + 0.50572 * (96.07995 - z) ** (-1.6364)
    if denom <= 0:
        return float("inf")
    x = 1.0 / denom
    return float(max(x, 1.0))

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


def allowed_filters_for_sun_alt(sun_alt_deg: float, cfg: PlannerConfig) -> List[str]:
    """Return filters permitted for a given Sun altitude.

    The defaults implement a simple twilight policy: the brighter the twilight
    (higher Sun altitude), the redder the allowed filters.  The returned list is
    already intersected with the planner configuration's ``filters``.
    """

    if -18.0 < sun_alt_deg <= -15.0:
        allowed = ["y", "z", "i"]
    elif -15.0 < sun_alt_deg <= -12.0:
        allowed = ["z", "i", "r"]
    elif -12.0 < sun_alt_deg < 0.0:
        allowed = ["i", "z", "y"]
    else:
        allowed = list(cfg.filters)
    return [f for f in allowed if f in cfg.filters]


def pick_first_filter_for_target(
    name: str,
    sn_type: str | None,
    tracker: "PriorityTracker",
    allowed_filters: List[str],
    cfg: PlannerConfig,
    sun_alt_deg: float | None = None,
    moon_sep_ok: Dict[str, bool] | None = None,
    current_mag: Dict[str, float] | None = None,
    current_filter: str | None = None,
) -> str | None:
    """Decide which filter to use first for the SN ``name``.

    Parameters are intentionally lightweight; the caller supplies the set of
    ``allowed_filters`` for the current window and an optional map of
    ``moon_sep_ok`` booleans keyed by filter.  The :class:`PriorityTracker`
    decides whether the target is still in the Hybrid stage (seeking detections
    in two filters) or has been escalated to the light‑curve stage.

    The function prefers filters that have not yet been used on the target when
    in the Hybrid stage.  Once escalated, it chooses the reddest available
    filter that passes the Moon constraint and, if possible, matches the
    current carousel filter to avoid a costly swap.
    """

    try:
        hist = tracker.history.get(name, None)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - tracker not supplied
        hist = None

    candidates = [f for f in allowed_filters if (moon_sep_ok or {}).get(f, True)]
    if not candidates:
        return None

    if hist and not getattr(hist, "escalated", False):
        for f in candidates:
            if f not in getattr(hist, "filters", set()):
                return f

    twilight_pref = ["y", "z", "i", "r", "g", "u"]
    ordered = [f for f in twilight_pref if f in candidates]
    if current_filter in ordered and ordered.index(current_filter) == 0:
        return current_filter
    return ordered[0] if ordered else candidates[0]

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

def per_sn_time_seconds(filters: Sequence[str], sep_deg: float, cfg: PlannerConfig):
    """Compute the total time to execute a visit in the given ``filters``."""

    slew = slew_time_seconds(
        sep_deg,
        small_deg=cfg.slew_small_deg,
        small_time=cfg.slew_small_time_s,
        rate_deg_per_s=cfg.slew_rate_deg_per_s,
        settle_s=cfg.slew_settle_s,
    )
    exposure = sum(cfg.exposure_by_filter.get(f, 0.0) for f in filters)
    readout = cfg.readout_time_s * len(filters)
    fchanges = cfg.filter_change_time_s * max(0, len(filters) - 1)
    total = slew + exposure + readout + fchanges
    return total, slew, exposure, readout, fchanges


def choose_filters_with_cap(
    filters: List[str],
    sep_deg: float,
    cap_s: float,
    cfg: PlannerConfig,
    *,
    current_filter: str | None = None,
    max_filters_per_visit: int | None = None,
) -> tuple[List[str], dict]:
    """Choose a sequence of filters that fits within ``cap_s`` seconds.

    The function accounts for slew time, cross‑target filter changes and
    per‑filter exposures/readouts.  If no filter fits within the cap, the first
    requested filter is returned even if it slightly exceeds the cap.  The
    ``timing`` dictionary contains a breakdown of the time budget with keys:

    ``total_s``
        Total visit time including all overheads.
    ``slew_s``
        Slew time from the previous target.
    ``cross_filter_change_s``
        Carousel change needed before the first exposure.
    ``filter_changes_s``
        Internal changes within the visit (after the first filter).
    ``readout_s``
        Total readout time.
    ``exposure_s``
        Sum of exposure times.
    ``exp_times``
        Mapping of filter to exposure time.
    """

    if max_filters_per_visit is not None:
        max_filters = max_filters_per_visit
    elif getattr(cfg, "allow_filter_changes_in_twilight", False):
        max_filters = len(filters)
    else:
        max_filters = cfg.max_filters_per_visit
    slew = slew_time_seconds(
        sep_deg,
        small_deg=cfg.slew_small_deg,
        small_time=cfg.slew_small_time_s,
        rate_deg_per_s=cfg.slew_rate_deg_per_s,
        settle_s=cfg.slew_settle_s,
    )

    used: List[str] = []
    cross_change = 0.0
    exposure = 0.0
    readout = 0.0
    internal_changes = 0.0

    for f in filters:
        if len(used) >= max_filters:
            break
        exp = cfg.exposure_by_filter.get(f, 0.0)
        trial_cross = cfg.filter_change_time_s if (used == [] and current_filter and f != current_filter) else 0.0
        trial_internal = cfg.filter_change_time_s if used else 0.0
        trial_total = (
            slew
            + cross_change
            + trial_cross
            + internal_changes
            + trial_internal
            + exposure
            + exp
            + readout
            + cfg.readout_time_s
        )
        if trial_total <= cap_s or not used:
            if used == []:
                cross_change = trial_cross
            else:
                internal_changes += trial_internal
            used.append(f)
            exposure += exp
            readout += cfg.readout_time_s
        else:
            break

    if not used and filters:
        used = [filters[0]]
        exp = cfg.exposure_by_filter.get(filters[0], 0.0)
        cross_change = cfg.filter_change_time_s if current_filter and filters[0] != current_filter else 0.0
        exposure = exp
        readout = cfg.readout_time_s

    total = slew + cross_change + internal_changes + exposure + readout
    timing = {
        "total_s": total,
        "slew_s": slew,
        "cross_filter_change_s": cross_change,
        "filter_changes_s": internal_changes,
        "readout_s": readout,
        "exposure_s": exposure,
        "exp_times": {f: cfg.exposure_by_filter.get(f, 0.0) for f in used},
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

def _best_time_with_moon(
    sc: SkyCoord,
    window: Tuple[datetime, datetime],
    loc: EarthLocation,
    step_min: int,
    min_alt_deg: float,
    min_moon_sep_deg: float,
) -> Tuple[float, datetime | None]:
    """Sample a window and return the best observation time.

    Both the target and the Moon are transformed into the same topocentric
    :class:`~astropy.coordinates.AltAz` frame to evaluate the separation.  The
    separation constraint is ignored if the Moon is below the horizon.  The
    function is vectorised and avoids Astropy's angular-separation warnings by
    construction.
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
