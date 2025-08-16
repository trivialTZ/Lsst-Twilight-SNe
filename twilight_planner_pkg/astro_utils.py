"""Astronomy utilities for the LSST twilight planner.

Provides Kasten & Young (1989) airmass, Moon-aware timing helpers using
``astropy.coordinates.get_body`` in a shared AltAz frame, and functions for
filter-aware visit timing.  Moon separation requirements are graded by Moon
altitude and phase rather than simply waived below the horizon.  Slew,
readout, and filter-change overheads reflect Rubin Observatory technical
notes.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, TypedDict

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyWarning

try:
    from astropy.coordinates.baseframe import (  # most versions
        NonRotationTransformationWarning,
    )
except Exception:
    try:
        from astropy.coordinates.transformations import NonRotationTransformationWarning
    except Exception:

        class NonRotationTransformationWarning(AstropyWarning):
            pass


from .config import PlannerConfig
from .constraints import moon_separation_factor

if TYPE_CHECKING:
    from .priority import PriorityTracker

warnings.filterwarnings("ignore", category=AstropyDeprecationWarning)
warnings.filterwarnings("ignore", category=NonRotationTransformationWarning)
warnings.filterwarnings(
    "ignore",
    message="Angular separation can depend on the direction of the transformation",
    category=Warning,
    module="astropy",
)


def validate_coords(
    ra_deg: float, dec_deg: float, eps: float = 1e-6
) -> Tuple[float, float]:
    """Normalize and validate equatorial coordinates.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Right ascension and declination in degrees.
    eps : float, optional
        Tolerance for floating-point jitter at the ``\pm90`` declination bounds.

    Returns
    -------
    tuple(float, float)
        ``(ra_deg_normalized, dec_deg)`` where RA is in ``[0,360)`` and Dec is
        clamped to ``\pm90`` only if within ``eps`` of the boundary.

    Raises
    ------
    ValueError
        If ``dec_deg`` lies outside ``[-90-eps, +90+eps]``.
    """

    ra_norm = (ra_deg % 360.0 + 360.0) % 360.0
    if dec_deg < -90.0 - eps or dec_deg > 90.0 + eps:
        raise ValueError(f"Invalid Dec={dec_deg} deg (must be within [-90, +90])")
    if dec_deg > 90.0:
        dec_deg = 90.0
    if dec_deg < -90.0:
        dec_deg = -90.0
    return ra_norm, dec_deg


def ra_delta_shortest_deg(ra1_deg: float, ra2_deg: float) -> float:
    """Return the absolute shortest RA difference in degrees.

    This properly wraps around the 0/360 boundary such that ``359°`` to
    ``1°`` yields ``2°`` rather than ``-358°``.
    """

    d = ((ra2_deg - ra1_deg + 180.0) % 360.0) - 180.0
    return abs(d)


def airmass_from_alt_deg(alt_deg: float) -> float:
    """Convert altitude to relative airmass using Kasten & Young (1989).

    This uses the "simple" airmass approximation of
    Kasten & Young (1989, *Applied Optics*, 28, 4735).  The function clamps
    the result to ``>=1`` and avoids evaluating below the horizon where the
    formula would formally diverge.
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


class TwilightWindow(TypedDict):
    """Twilight window with classification."""

    start: datetime
    end: datetime
    label: str | None
    # date of the *local night* this window belongs to (evening's civil date)
    night_date: date | None


def twilight_windows_astro(
    date_utc: datetime,
    loc: EarthLocation,
    min_sun_alt_deg: float = PlannerConfig.twilight_sun_alt_min_deg,
    max_sun_alt_deg: float = PlannerConfig.twilight_sun_alt_max_deg,
) -> List[TwilightWindow]:
    """Compute twilight windows for a given date and location.

    Parameters
    ----------
    date_utc : datetime
        Reference date in UTC.
    loc : astropy.coordinates.EarthLocation
        Observatory location.
    min_sun_alt_deg, max_sun_alt_deg : float, optional
        Inclusive Sun-altitude bounds in degrees. Windows are emitted where
        ``min_sun_alt_deg < h_☉ < max_sun_alt_deg``. Defaults reflect
        astronomical twilight (``-18°`` to ``0°``).

    Returns
    -------
    list[TwilightWindow]
        Sorted list of twilight windows within the Sun-altitude range. Each
        window includes a ``label`` of ``"morning"`` or ``"evening"`` when it
        falls on the given ``date_utc``; windows outside that day remain
        unlabeled (``None``).
    """
    start = date_utc.replace(tzinfo=timezone.utc) - timedelta(hours=12)
    times = Time([start + timedelta(minutes=i) for i in range(48 * 60)])
    altaz = AltAz(obstime=times, location=loc)
    sun_alt = get_sun(times).transform_to(altaz).alt.to(u.deg).value
    mask = (sun_alt > min_sun_alt_deg) & (sun_alt < max_sun_alt_deg)
    edges = np.where(np.diff(mask.astype(int)) != 0)[0]
    segments, prev = [], 0
    for e in edges:
        segments.append((prev, e))
        prev = e + 1
    segments.append((prev, len(mask) - 1))
    windows: List[TwilightWindow] = []
    day_end = date_utc + timedelta(days=1)
    for a, b in segments:
        if np.any(mask[a : b + 1]):
            i0 = a + int(np.argmax(mask[a : b + 1]))
            i1 = i0
            while i0 > a and mask[i0 - 1]:
                i0 -= 1
            while i1 < b and mask[i1 + 1]:
                i1 += 1
            start_dt = Time(times[i0]).to_datetime(timezone.utc)
            end_dt = Time(times[i1]).to_datetime(timezone.utc)
            alt_start = float(sun_alt[i0])
            alt_end = float(sun_alt[i1])
            label: str | None = None
            if date_utc <= start_dt < day_end:
                label = "morning" if alt_end > alt_start else "evening"
            windows.append(
                {"start": start_dt, "end": end_dt, "label": label, "night_date": None}
            )
    windows.sort(key=lambda w: w["start"])
    return windows


def _local_timezone_from_location(loc: EarthLocation) -> timezone:
    """Approximate local (solar) timezone from longitude, as a fixed UTC offset.

    Positive offsets are east of Greenwich. West longitudes produce negative offsets.
    This avoids depending on a civil timezone database and is sufficient for
    night-bundling logic.
    """
    offset_minutes = int(round(loc.lon.to(u.deg).value / 15.0 * 60))
    return timezone(timedelta(minutes=offset_minutes))


def twilight_windows_for_local_night(
    date_local: date,
    loc: EarthLocation,
    min_sun_alt_deg: float = PlannerConfig.twilight_sun_alt_min_deg,
    max_sun_alt_deg: float = PlannerConfig.twilight_sun_alt_max_deg,
) -> List[TwilightWindow]:
    """Return twilight windows (evening and/or morning) for a *local* night.

    Parameters
    ----------
    date_local : datetime.date
        Local civil date identifying the evening of the night.
    loc : astropy.coordinates.EarthLocation
        Observatory location.
    min_sun_alt_deg, max_sun_alt_deg : float, optional
        Sun-altitude bounds passed through to :func:`twilight_windows_astro`.
        Defaults match astronomical twilight.

    Returns
    -------
    list[TwilightWindow]
        At most two windows (evening then morning) labeled and stamped with
        ``night_date = date_local``.

    Notes
    -----
    We compute a ±24h UTC sweep around local midnight, find all twilight
    windows and then select:
      - the *evening* window with ``start_local.date() == date_local``
      - the *morning* window with ``start_local.date() == date_local + 1 day``
    Each selected window is labeled and stamped with ``night_date = date_local``.
    """
    # Accept pandas.Timestamp or datetime-like; normalize to datetime.date
    try:
        from datetime import date as _Date

        if hasattr(date_local, "date") and type(date_local) is not _Date:
            date_local = date_local.date()
    except Exception:
        pass

    tz = _local_timezone_from_location(loc)
    local_midnight = datetime(
        date_local.year, date_local.month, date_local.day, tzinfo=tz
    )
    # Center the 48h sweep on the local date's midnight
    date_utc_anchor = local_midnight.astimezone(timezone.utc)
    wins = twilight_windows_astro(
        date_utc_anchor,
        loc,
        min_sun_alt_deg=min_sun_alt_deg,
        max_sun_alt_deg=max_sun_alt_deg,
    )

    def _is_morning(w: TwilightWindow) -> bool:
        # compute Sun alt trend at start/end; morning rises, evening falls
        altaz = AltAz(obstime=Time([w["start"], w["end"]]), location=loc)
        alt = (
            get_sun(Time([w["start"], w["end"]]))
            .transform_to(altaz)
            .alt.to(u.deg)
            .value
        )
        return float(alt[1]) > float(alt[0])

    selected: List[TwilightWindow] = []
    # tag local dates
    for w in wins:
        start_local = w["start"].astimezone(tz)
        if _is_morning(w) and start_local.date() == (date_local + timedelta(days=1)):
            w = dict(w)  # copy
            w["label"] = "morning"
            w["night_date"] = date_local
            selected.append(w)  # keep the one morning of this night (if present)
        elif (not _is_morning(w)) and start_local.date() == date_local:
            w = dict(w)
            w["label"] = "evening"
            w["night_date"] = date_local
            selected.append(w)

    # Sort: evening first, then morning
    selected.sort(key=lambda w: (0 if w["label"] == "evening" else 1, w["start"]))
    return selected


def great_circle_sep_deg(ra1, dec1, ra2, dec2) -> float:
    """Compute on-sky separation between two coordinates."""

    ra1_r, dec1_r, ra2_r, dec2_r = map(np.deg2rad, [ra1, dec1, ra2, dec2])
    d_ra = np.deg2rad(ra_delta_shortest_deg(ra1, ra2))
    d_dec = dec2_r - dec1_r
    sin_ddec2 = np.sin(d_dec / 2.0)
    sin_dra2 = np.sin(d_ra / 2.0)
    a = sin_ddec2**2 + np.cos(dec1_r) * np.cos(dec2_r) * sin_dra2**2
    return float(np.rad2deg(2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))))


def allowed_filters_for_sun_alt(sun_alt_deg: float, cfg: PlannerConfig) -> List[str]:
    """Return filters permitted for a given Sun altitude.

    The mapping is defined by :attr:`PlannerConfig.sun_alt_policy`, a list of
    ``(low, high, filters)`` tuples.  The first tuple with ``low < sun_alt_deg
    <= high`` is selected.  The result is intersected with ``cfg.filters`` so
    that only loaded carousel filters are returned.  If no policy applies, all
    configured filters are allowed.
    """

    allowed: List[str] | None = None
    for low, high, flist in cfg.sun_alt_policy:
        if low < sun_alt_deg <= high:
            allowed = list(flist)
            break
    if allowed is None:
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


def slew_time_seconds(
    sep_deg: float,
    *,
    small_deg: float,
    small_time: float,
    rate_deg_per_s: float,
    settle_s: float,
) -> float:
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
    readout = cfg.readout_s * len(filters)
    fchanges = cfg.filter_change_s * max(0, len(filters) - 1)
    total = slew + exposure + readout + fchanges
    return total, slew, exposure, readout, fchanges


def compute_capped_exptime(band: str, cfg: PlannerConfig) -> tuple[float, set[str]]:
    """Return a saturation-safe exposure time and flags for ``band``.

    The function examines ``cfg.current_mag_by_filter`` and
    ``cfg.current_alt_deg`` for the SN currently under consideration.  If a
    ``cfg.sky_provider`` is available it is queried for the sky brightness.
    When ``cfg.current_mjd`` is set the provider receives it, allowing
    Sun-altitude--aware brightening to drive exposure capping.  These
    parameters are fed into
    :func:`photom_rubin.cap_exposure_for_saturation`, potentially reducing the
    baseline exposure from ``cfg.exposure_by_filter``.  The returned tuple
    includes a set of flags such as ``{"warn_nonlinear"}``.
    When the required context is missing, the baseline exposure and an empty
    flag set are returned.
    """

    base = cfg.exposure_by_filter.get(band, 0.0)
    if (
        cfg.current_mag_by_filter is None
        or band not in cfg.current_mag_by_filter
        or cfg.current_alt_deg is None
    ):
        return base, set()

    from .photom_rubin import PhotomConfig, cap_exposure_for_saturation

    if cfg.sky_provider:
        # Use the actual MJD (if provided) so the Sun-altitude–aware sky model
        # informs capping. Brighter twilight implies shorter safe exposures.
        mjd = getattr(cfg, "current_mjd", None)
        sky_mag = cfg.sky_provider.sky_mag(
            mjd, None, None, band, airmass_from_alt_deg(cfg.current_alt_deg)
        )
    else:
        sky_mag = 21.0

    phot_cfg = PhotomConfig(
        pixel_scale_arcsec=cfg.pixel_scale_arcsec,
        zpt1s=cfg.zpt1s,
        k_m=cfg.k_m,
        fwhm_eff=cfg.fwhm_eff,
        read_noise_e=cfg.read_noise_e,
        gain_e_per_adu=cfg.gain_e_per_adu,
        zpt_err_mag=cfg.zpt_err_mag,
    )
    return cap_exposure_for_saturation(
        band,
        base,
        cfg.current_alt_deg,
        cfg.current_mag_by_filter[band],
        sky_mag,
        phot_cfg,
    )


def choose_filters_with_cap(
    filters: list[str],
    sep_deg: float,
    cap_s: float,
    cfg: PlannerConfig,
    *,
    current_filter: str | None = None,
    max_filters_per_visit: int | None = None,
    use_capped_exposure: bool = True,
) -> tuple[list[str], dict]:
    """Greedily choose filters that fit within ``cap_s`` seconds.

    A cross‑target filter change is charged once if the first selected filter
    differs from ``current_filter``.  Additional in‑visit filter switches incur
    an "internal" change cost for each extra filter.  Exposure times are either
    the baseline values from ``cfg.exposure_by_filter`` or, when
    ``use_capped_exposure`` is true, the saturation‑safe values and flag sets
    returned by :func:`compute_capped_exptime`.
    """

    max_filters = (
        max_filters_per_visit
        if max_filters_per_visit is not None
        else cfg.max_filters_per_visit
    )
    slew = slew_time_seconds(
        sep_deg,
        small_deg=cfg.slew_small_deg,
        small_time=cfg.slew_small_time_s,
        rate_deg_per_s=cfg.slew_rate_deg_per_s,
        settle_s=cfg.slew_settle_s,
    )

    used: list[str] = []
    exp_times: dict[str, float] = {}
    flags_by_filter: dict[str, set[str]] = {}
    cross_change = 0.0
    internal_changes = 0.0

    for f in filters:
        if len(used) >= max_filters:
            break
        if use_capped_exposure:
            exp, flags = compute_capped_exptime(f, cfg)
        else:
            exp, flags = cfg.exposure_by_filter.get(f, 0.0), set()
        trial_len = len(used) + 1
        trial_cross = cross_change or (
            cfg.filter_change_s
            if (current_filter is not None and not used and f != current_filter)
            else 0.0
        )
        trial_internal = cfg.filter_change_s * max(0, trial_len - 1)
        trial_total = (
            slew
            + trial_cross
            + trial_internal
            + sum(exp_times.values())
            + exp
            + cfg.readout_s * trial_len
        )
        if trial_total <= cap_s or not used:
            used.append(f)
            exp_times[f] = exp
            flags_by_filter[f] = flags
            cross_change = trial_cross
            internal_changes = cfg.filter_change_s * max(0, len(used) - 1)
        else:
            break

    if not used and filters:
        f = filters[0]
        if use_capped_exposure:
            exp, flags = compute_capped_exptime(f, cfg)
        else:
            exp, flags = cfg.exposure_by_filter.get(f, 0.0), set()
        exp_times = {f: exp}
        flags_by_filter = {f: flags}
        used = [f]
        cross_change = (
            cfg.filter_change_s
            if current_filter is not None and f != current_filter
            else 0.0
        )

    exposure_s = sum(exp_times.values())
    readout_s = cfg.readout_s * len(used)
    total = slew + cross_change + internal_changes + exposure_s + readout_s
    timing = {
        "total_s": total,
        "slew_s": slew,
        "cross_filter_change_s": cross_change,
        "internal_filter_changes_s": internal_changes,
        "filter_changes_s": cross_change + internal_changes,
        "exposure_s": exposure_s,
        "readout_s": readout_s,
        "exp_times": exp_times,
        # flags keyed by filter, e.g. {'r': {'warn_nonlinear'}}
        "flags_by_filter": flags_by_filter,
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
) -> Tuple[float, datetime | None, float, float, float]:
    """Sample a window and return the best observation time.

    Both the target and the Moon are transformed into the same topocentric
    :class:`~astropy.coordinates.AltAz` frame to evaluate the separation.  The
    separation requirement is scaled by :func:`~twilight_planner_pkg.constraints.moon_separation_factor`,
    which gradually relaxes the minimum separation when the Moon is low or
    below the horizon.  The function is vectorised and avoids Astropy's
    angular-separation warnings by construction.
    """
    t0, t1 = window
    if t1 <= t0:
        return -np.inf, None, float("nan"), float("nan"), float("nan")
    n = 1 + int((t1 - t0).total_seconds() // (step_min * 60))
    ts = Time([t0 + timedelta(minutes=step_min * i) for i in range(n)])
    altaz = AltAz(obstime=ts, location=loc)

    sc_altaz = sc.transform_to(altaz)
    alt = sc_altaz.alt.to(u.deg).value
    moon_altaz = get_body("moon", ts).transform_to(altaz)
    sun_altaz = get_sun(ts).transform_to(altaz)

    moon_alt = moon_altaz.alt.to(u.deg).value
    sep_deg = sc_altaz.separation(moon_altaz).deg
    phase = 0.5 * (1 - np.cos(np.deg2rad(moon_altaz.separation(sun_altaz).deg)))

    factor = moon_separation_factor(moon_alt, phase)
    eff = min_moon_sep_deg * factor
    ok = (alt >= min_alt_deg) & (sep_deg >= eff)
    if not np.any(ok):
        return -np.inf, None, float("nan"), float("nan"), float("nan")
    alt_ok = np.where(ok, alt, -np.inf)
    j = int(np.argmax(alt_ok))
    return (
        float(alt_ok[j]),
        ts[j].to_datetime(timezone.utc),
        float(moon_alt[j]),
        float(phase[j]),
        float(sep_deg[j]),
    )
