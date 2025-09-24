"""Astronomy utilities for the LSST twilight planner.

Provides Kasten & Young (1989) airmass, Moon-aware timing helpers using
``astropy.coordinates.get_body`` in a shared AltAz frame, and functions for
filter-aware visit timing.  Moon separation requirements are graded by Moon
altitude and phase rather than simply waived below the horizon.  Slew,
readout, and filter-change overheads reflect Rubin Observatory technical
notes.
"""

from __future__ import annotations

import math
import warnings
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, TypedDict

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyWarning

# mypy: ignore-errors


try:
    from astropy.coordinates.baseframe import (
        NonRotationTransformationWarning,
    )  # most versions
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


def default_host_mu_obs(band: str, cfg: PlannerConfig) -> float | None:
    """Return a fallback observed host surface brightness for ``band``."""

    if not getattr(cfg, "use_default_host_sb", False):
        return None

    z_context: float | None = None
    for attr in ("current_host_z", "current_redshift"):
        z_val = getattr(cfg, attr, None)
        if z_val is None:
            continue
        try:
            z_candidate = float(z_val)
        except (TypeError, ValueError):
            continue
        if z_candidate > -0.99:
            z_context = z_candidate
            break

    mu_rest_map = getattr(cfg, "default_host_mu_rest_arcsec2_by_filter", None)
    slope_map = getattr(cfg, "default_host_kcorr_slope_by_filter", None) or {}
    if (
        mu_rest_map
        and z_context is not None
        and z_context > 0.0
        and band in mu_rest_map
    ):
        mu_rest = float(mu_rest_map[band])
        slope = float(slope_map.get(band, 0.2))
        mu_obs = mu_rest + 10.0 * math.log10(1.0 + z_context) + slope * z_context
        return mu_obs

    obs_map = getattr(cfg, "default_host_mu_arcsec2_by_filter", None)
    if obs_map and band in obs_map:
        return float(obs_map[band])

    # Final conservative fallback matching historical behaviour.
    return 22.0


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
    now_mjd: float | None = None,
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

    # 0) User-supplied hook (if provided). Prefer strict call signature, then
    #    fall back to a simpler variant for convenience.
    try:
        hook = getattr(cfg, "pick_first_filter", None)
    except Exception:
        hook = None
    if callable(hook):
        context = {
            "tracker": tracker,
            "cfg": cfg,
            "sun_alt_deg": sun_alt_deg,
            "moon_sep_ok": dict(moon_sep_ok or {}),
            "current_mag": dict(current_mag or {}),
            "current_filter": current_filter,
            "best_time_mjd": now_mjd,
        }
        choice: str | None = None
        try:
            choice = hook(name, candidates, cfg, context=context)
        except TypeError:
            try:
                choice = hook(name, candidates, cfg)  # type: ignore[misc]
            except Exception:
                choice = None
        except Exception:
            choice = None
        if isinstance(choice, str) and choice in candidates:
            return choice

    weight_map: Dict[str, float] = {}
    try:
        weight_map = dict(getattr(cfg, "first_filter_bonus_weights", {}) or {})
    except Exception:
        weight_map = {}

    def _weight_for(filt: str) -> float:
        try:
            return float(weight_map.get(filt, 1.0))
        except Exception:
            return 1.0

    def _bonus_for(filt: str) -> float:
        if now_mjd is None:
            return 1.0
        try:
            target_d = float(getattr(cfg, "cadence_days_target", 3.0))
        except Exception:
            target_d = 3.0
        try:
            sigma_d = float(getattr(cfg, "cadence_bonus_sigma_days", 0.5))
        except Exception:
            sigma_d = 0.5
        try:
            cadence_weight = float(getattr(cfg, "cadence_bonus_weight", 0.25))
        except Exception:
            cadence_weight = 0.25
        try:
            first_epoch_weight = float(getattr(cfg, "cadence_first_epoch_bonus_weight", 0.0))
        except Exception:
            first_epoch_weight = 0.0
        try:
            target_pairs = int(getattr(cfg, "color_target_pairs", 2))
        except Exception:
            target_pairs = 2
        try:
            window_days = float(getattr(cfg, "color_window_days", 5.0))
        except Exception:
            window_days = 5.0
        try:
            alpha = float(getattr(cfg, "color_alpha", 0.3))
        except Exception:
            alpha = 0.3
        try:
            first_epoch_boost = float(getattr(cfg, "first_epoch_color_boost", 1.5))
        except Exception:
            first_epoch_boost = 1.5
        diversity_enable = bool(getattr(cfg, "diversity_enable", False))
        try:
            diversity_target = int(getattr(cfg, "diversity_target_per_filter", 1))
        except Exception:
            diversity_target = 1
        try:
            diversity_window = float(getattr(cfg, "diversity_window_days", 5.0))
        except Exception:
            diversity_window = 5.0
        try:
            diversity_alpha = float(getattr(cfg, "diversity_alpha", 0.3))
        except Exception:
            diversity_alpha = 0.3

        cosmo_weights = getattr(cfg, "cosmo_weight_by_filter", {}) or {}

        try:
            bonus = tracker.compute_filter_bonus(
                name,
                filt,
                float(now_mjd),
                target_d,
                sigma_d,
                cadence_weight,
                first_epoch_weight,
                cosmo_weights,
                target_pairs,
                window_days,
                alpha,
                first_epoch_boost,
                diversity_enable=diversity_enable,
                diversity_target_per_filter=diversity_target,
                diversity_window_days=diversity_window,
                diversity_alpha=diversity_alpha,
            )
        except Exception:
            bonus = 1.0
        return float(bonus)

    def _score(filt: str) -> float:
        weight = _weight_for(filt)
        bonus = _bonus_for(filt)
        bias = 1e-6 if current_filter and current_filter == filt else 0.0
        return float(bonus * weight) + bias

    def _ordered(filters: List[str]) -> List[str]:
        if not filters:
            return []
        user_order = []
        try:
            user_order = list(getattr(cfg, "first_filter_order", []) or [])
        except Exception:
            user_order = []
        default_tail = ["y", "z", "i", "r", "g", "u"]
        twilight_pref = user_order + [f for f in default_tail if f not in user_order]
        index_map = {f: i for i, f in enumerate(twilight_pref)}
        try:
            return sorted(
                filters,
                key=lambda f: (-_score(f), index_map.get(f, len(twilight_pref))),
            )
        except Exception:
            return [f for f in twilight_pref if f in filters] or list(filters)

    if hist and not getattr(hist, "escalated", False):
        unseen = [f for f in candidates if f not in getattr(hist, "filters", set())]
        ordered_unseen = _ordered(unseen)
        if ordered_unseen:
            return ordered_unseen[0]

    ordered_candidates = _ordered(candidates)
    if current_filter in ordered_candidates and ordered_candidates.index(current_filter) == 0:
        return current_filter
    return ordered_candidates[0] if ordered_candidates else candidates[0]


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


def _Ez_for_lcdm(z: float, Om: float, Ol: float) -> float:
    """Hubble parameter term :math:`E(z)` for a general ΛCDM cosmology."""

    Ok = 1.0 - Om - Ol
    return math.sqrt(Om * (1.0 + z) ** 3 + Ok * (1.0 + z) ** 2 + Ol)


def distance_modulus_mu(
    z: float,
    H0: float = 70.0,
    Om: float = 0.3,
    Ol: float = 0.7,
) -> float:
    """Return the distance modulus μ(z) using Simpson integration for ΛCDM."""

    zf = float(z)
    if zf <= 0.0:
        return 0.0
    c_km_s = 299_792.458
    steps = 100
    h = zf / steps
    inv_e0 = 1.0 / _Ez_for_lcdm(0.0, Om, Ol)
    inv_en = 1.0 / _Ez_for_lcdm(zf, Om, Ol)
    acc = inv_e0 + inv_en
    for i in range(1, steps):
        zi = i * h
        weight = 4.0 if i % 2 else 2.0
        acc += weight / _Ez_for_lcdm(zi, Om, Ol)
    Dc_Mpc = (c_km_s / H0) * (h / 3.0) * acc
    Dl_Mpc = (1.0 + zf) * Dc_Mpc
    if Dl_Mpc <= 0.0:
        return 0.0
    return 5.0 * math.log10(Dl_Mpc * 1e6 / 10.0)


def peak_mag_from_redshift(
    z: float,
    band: str,
    *,
    MB: float = -19.36,
    alpha: float = 0.14,
    beta: float = 3.1,
    x1: float = 0.0,
    c: float = 0.0,
    K_approx: float = 0.0,
    Ab_MW: float = 0.0,
    Ab_host: float = 0.0,
    H0: float = 70.0,
    Om: float = 0.3,
    Ol: float = 0.7,
) -> float:
    """Conservative SN Ia peak magnitude estimate from redshift.

    ``band`` is accepted for API symmetry; the simplified model applies a
    scalar K-correction via ``K_approx``. Defaults keep ``x1=c=0`` so the
    result errs brightwards for saturation safety.
    """

    zf = float(z)
    if zf <= 0.0:
        mu = 0.0
    else:
        mu = distance_modulus_mu(zf, H0=H0, Om=Om, Ol=Ol)
    M_eff = MB - alpha * x1 + beta * c
    return mu + M_eff + K_approx + Ab_MW + Ab_host


def compute_capped_exptime(band: str, cfg: PlannerConfig) -> tuple[float, set[str]]:
    """Return a saturation-safe exposure time and flags for ``band``.

    The function examines ``cfg.current_mag_by_filter`` and
    ``cfg.current_alt_deg`` for the SN currently under consideration. If a
    ``cfg.sky_provider`` is available it is queried for the sky brightness.
    When ``cfg.current_mjd`` is set the provider receives it, allowing
    Sun-altitude--aware brightening to drive exposure capping.  These
    parameters plus any available host-galaxy context (local surface brightness
    or rest-frame SB + redshift) are fed into
    :func:`photom_rubin.cap_exposure_for_saturation`, potentially reducing the
    baseline exposure from ``cfg.exposure_by_filter``. The returned tuple
    includes a set of flags such as ``{"warn_nonlinear"}``.
    When the required context is missing, the baseline exposure and an empty
    flag set are returned.
    """

    base = cfg.exposure_by_filter.get(band, 0.0)
    if cfg.current_alt_deg is None:
        return base, set()

    src_mag_map = dict((cfg.current_mag_by_filter or {}))
    if band not in src_mag_map:
        try:
            z_val = float(getattr(cfg, "current_redshift", float("nan")))
        except Exception:
            z_val = float("nan")
        if z_val == z_val and z_val > 0.0:
            try:
                H0 = float(getattr(cfg, "H0_km_s_Mpc", 70.0))
                Om = float(getattr(cfg, "Omega_m", 0.3))
                Ol = float(getattr(cfg, "Omega_L", 0.7))
                MB = float(getattr(cfg, "MB_absolute", -19.36))
                alpha = float(getattr(cfg, "SALT2_alpha", 0.14))
                beta = float(getattr(cfg, "SALT2_beta", 3.1))
                K0 = float(getattr(cfg, "Kcorr_approx_mag", 0.0))
                K_by_filter = getattr(cfg, "Kcorr_approx_mag_by_filter", None)
                if isinstance(K_by_filter, dict):
                    K0 = float(K_by_filter.get(str(band).lower(), K0))
                margin = float(getattr(cfg, "peak_extra_bright_margin_mag", 0.3))
                src_mag_map[band] = (
                    peak_mag_from_redshift(
                        z_val,
                        band,
                        MB=MB,
                        alpha=alpha,
                        beta=beta,
                        H0=H0,
                        Om=Om,
                        Ol=Ol,
                        K_approx=K0,
                    )
                    - margin
                )
            except Exception:
                pass
    if band not in src_mag_map:
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
        # Tie saturation thresholds to PlannerConfig's SIMLIB settings when present
        npe_pixel_saturate=int(
            getattr(cfg, "simlib_npe_pixel_saturate", 80_000) or 80_000
        ),
        npe_pixel_warn_nonlinear=int(
            0.8 * (getattr(cfg, "simlib_npe_pixel_saturate", 80_000) or 80_000)
        ),
        use_snana_fA=getattr(cfg, "use_snana_fA", True),
    )
    # Optional host terms if provided via config
    kwargs = {}
    if getattr(cfg, "current_host_mu_arcsec2_by_filter", None) and (
        band in cfg.current_host_mu_arcsec2_by_filter
    ):
        kwargs["mu_host_obs_arcsec2"] = cfg.current_host_mu_arcsec2_by_filter[band]
    elif (
        getattr(cfg, "current_host_mu_rest_arcsec2_by_filter", None)
        and (band in cfg.current_host_mu_rest_arcsec2_by_filter)
        and getattr(cfg, "current_host_z", None) is not None
    ):
        kwargs["mu_host_rest_arcsec2"] = cfg.current_host_mu_rest_arcsec2_by_filter[
            band
        ]
        kwargs["z_host"] = cfg.current_host_z
        if getattr(cfg, "current_host_K_by_filter", None):
            kwargs["K_host"] = cfg.current_host_K_by_filter.get(band, 0.0)
    if getattr(cfg, "current_host_point_mag_by_filter", None) and (
        band in cfg.current_host_point_mag_by_filter
    ):
        kwargs["host_point_mag"] = cfg.current_host_point_mag_by_filter[band]
        if getattr(cfg, "current_host_point_frac", None) is not None:
            kwargs["host_point_frac"] = cfg.current_host_point_frac

    # If no explicit host info is supplied, use default host SB if enabled
    if "mu_host_obs_arcsec2" not in kwargs and "mu_host_rest_arcsec2" not in kwargs:
        mu_def = default_host_mu_obs(band, cfg)
        if mu_def is not None:
            kwargs["mu_host_obs_arcsec2"] = float(mu_def)

    return cap_exposure_for_saturation(
        band,
        base,
        cfg.current_alt_deg,
        src_mag_map[band],
        sky_mag,
        phot_cfg,
        **kwargs,
    )


def choose_filters_with_cap(
    filters: list[str],
    sep_deg: float,
    cap_s: float,
    cfg: PlannerConfig,
    *,
    current_filter: str | None = None,
    filters_per_visit_cap: int | None = None,
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

    if filters_per_visit_cap is None and max_filters_per_visit is not None:
        warnings.warn(
            "max_filters_per_visit argument is deprecated; use filters_per_visit_cap",
            DeprecationWarning,
            stacklevel=2,
        )
        filters_per_visit_cap = max_filters_per_visit
    if filters_per_visit_cap is None:
        filters_per_visit_cap = int(
            getattr(cfg, "filters_per_visit_cap", getattr(cfg, "max_filters_per_visit", 1))
        )
    max_filters = max(1, int(filters_per_visit_cap))
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
    *,
    precomputed: dict | None = None,
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
    if precomputed is None:
        n = 1 + int((t1 - t0).total_seconds() // (step_min * 60))
        ts = Time([t0 + timedelta(minutes=step_min * i) for i in range(n)])
        altaz = AltAz(obstime=ts, location=loc)
        moon_altaz = get_body("moon", ts).transform_to(altaz)
        sun_altaz = get_sun(ts).transform_to(altaz)
        moon_alt = moon_altaz.alt.to(u.deg).value
        phase = 0.5 * (1 - np.cos(np.deg2rad(moon_altaz.separation(sun_altaz).deg)))
        precomputed = {
            "ts": ts,
            "altaz": altaz,
            "moon_altaz": moon_altaz,
            "moon_alt": moon_alt,
            "phase": phase,
        }
    ts = precomputed["ts"]
    altaz = precomputed["altaz"]
    moon_altaz = precomputed["moon_altaz"]
    moon_alt = precomputed["moon_alt"]
    phase = precomputed["phase"]

    sc_altaz = sc.transform_to(altaz)
    alt = sc_altaz.alt.to(u.deg).value
    sep_deg = sc_altaz.separation(moon_altaz).deg

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


def precompute_window_ephemerides(
    window: Tuple[datetime, datetime],
    loc: EarthLocation,
    step_min: int,
) -> dict:
    """Precompute Moon/Sun ephemerides shared by all targets in a window.

    Returns a dict suitable for the ``precomputed`` parameter of
    :func:`_best_time_with_moon`.
    """
    t0, t1 = window
    if t1 <= t0:
        return {}
    n = 1 + int((t1 - t0).total_seconds() // (step_min * 60))
    ts = Time([t0 + timedelta(minutes=step_min * i) for i in range(n)])
    altaz = AltAz(obstime=ts, location=loc)
    moon_altaz = get_body("moon", ts).transform_to(altaz)
    sun_altaz = get_sun(ts).transform_to(altaz)
    moon_alt = moon_altaz.alt.to(u.deg).value
    phase = 0.5 * (1 - np.cos(np.deg2rad(moon_altaz.separation(sun_altaz).deg)))
    return {
        "ts": ts,
        "altaz": altaz,
        "moon_altaz": moon_altaz,
        "moon_alt": moon_alt,
        "phase": phase,
    }
