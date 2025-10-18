"""Caching utilities for sky brightness and m5 estimates.

Responsibility
--------------
- Maintain per-window caches keyed by a window token and minute buckets.
- Maintain a persistent LRU cache of best-time m5 values between windows.
- Provide cached lookups for sky surface brightness (mag/arcsec^2) and m5.
- Expose helpers to bump the per-window cache token and prune the LRU by day.

Hot path
--------
- ``_cached_sky_mag``
- ``_cached_m5_at_time``

Behavior and keys match the original scheduler implementation exactly.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional, Dict

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from ..astro_utils import airmass_from_alt_deg
from ..config import PlannerConfig
from ..filter_policy import _m5_scale
from ..photom_rubin import PhotomConfig
from ..sky_model import SkyModelConfig, sky_mag_arcsec2

# ---------------------------
# Per-window SKY / M5 caches
# ---------------------------
_CACHE_TOKEN: int = 0  # bumped per twilight window
_SKY_CACHE: dict[tuple[int, int, int, str, int, int], float] = {}
_M5_CACHE: dict[tuple[int, str, str, int], float] = {}

# Persistent best-time m5 cache (cross-window)
_M5BEST_LRU: OrderedDict[tuple[str, str, int], float] = OrderedDict()
_M5BEST_DEFAULT_MAX = 30_000
_M5BEST_KEEP_DAYS_DEFAULT = 5


def _mjd_bucket(mjd: float, minutes: int = 1) -> int:
    """Quantize MJD to a per-minute bucket for caching."""
    return int(round(float(mjd) * 1440.0 / max(1, minutes)))


def _mjd_day(mjd: float) -> int:
    """Return integer day bucket for persistent caches."""
    return int(math.floor(float(mjd)))


def _m5best_key(name: str, filt: str, best_mjd: float) -> tuple[str, str, int]:
    return (str(name), str(filt), _mjd_day(best_mjd))


def _m5best_cache_get(name: str | None, filt: str, best_mjd: float) -> Optional[float]:
    """Return cached best-time m5 if present, touching LRU order."""
    if not name:
        return None
    key = _m5best_key(name, filt, best_mjd)
    cached = _M5BEST_LRU.get(key)
    if cached is not None:
        _M5BEST_LRU.move_to_end(key)
    return cached


def _m5best_cache_put(
    cfg: PlannerConfig, name: str, filt: str, best_mjd: float, value: float
) -> None:
    """Insert/update best-time m5 value, trimming LRU size."""
    key = _m5best_key(name, filt, best_mjd)
    _M5BEST_LRU[key] = float(value)
    _M5BEST_LRU.move_to_end(key)
    try:
        max_items = int(getattr(cfg, "m5best_cache_max_items", _M5BEST_DEFAULT_MAX))
    except Exception:
        max_items = _M5BEST_DEFAULT_MAX
    if max_items <= 0:
        _M5BEST_LRU.clear()
        return
    while len(_M5BEST_LRU) > max_items:
        _M5BEST_LRU.popitem(last=False)


def _m5best_cache_prune_by_day(cfg: PlannerConfig, current_mjd: float) -> None:
    """Drop stale best-time m5 entries beyond the configured day horizon."""
    try:
        keep_days = int(
            getattr(cfg, "m5best_cache_keep_days", _M5BEST_KEEP_DAYS_DEFAULT)
        )
    except Exception:
        keep_days = _M5BEST_KEEP_DAYS_DEFAULT
    if keep_days <= 0:
        _M5BEST_LRU.clear()
        return
    min_day = _mjd_day(current_mjd) - keep_days
    for key in list(_M5BEST_LRU.keys()):
        if key[2] < min_day:
            del _M5BEST_LRU[key]


def _finite_or_none(val) -> float | None:
    """Return float(val) if finite; otherwise None."""
    try:
        out = float(val)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _cached_sky_mag(
    *,
    sky_provider,
    sky_cfg: SkyModelConfig,
    ra_deg: float | None,
    dec_deg: float | None,
    band: str,
    airmass: float,
    mjd: float,
    minutes: int = 1,
    sun_alt_deg: float | None = None,
    moon_alt_deg: float | None = None,
    moon_phase: float | None = None,
    moon_sep_deg: float | None = None,
    k_band: float = 0.0,
) -> float:
    """Sky μ (mag/arcsec²) cached per-minute for the target geometry."""

    bucket = _mjd_bucket(mjd, minutes)
    try:
        ra_val = float(ra_deg) if ra_deg is not None else None
    except Exception:
        ra_val = None
    if ra_val is None or not math.isfinite(ra_val):
        ra_q = 0
    else:
        ra_q = int(round(ra_val * 1e4))
    try:
        dec_val = float(dec_deg) if dec_deg is not None else None
    except Exception:
        dec_val = None
    if dec_val is None or not math.isfinite(dec_val):
        dec_q = 0
    else:
        dec_q = int(round(dec_val * 1e4))
    try:
        X_val = float(airmass)
    except Exception:
        X_val = 1.2
    if not math.isfinite(X_val):
        X_val = 1.2
    X_q = int(round(X_val * 100.0))
    key = (_CACHE_TOKEN, bucket, ra_q, band, dec_q, X_q)
    cached = _SKY_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        if sky_provider is None:
            raise RuntimeError("no sky provider configured")
        sky_val = float(
            sky_provider.sky_mag(
                float(mjd),
                float(ra_deg) if ra_deg is not None else None,
                float(dec_deg) if dec_deg is not None else None,
                band,
                X_val,
            )
        )
    except Exception:
        sky_val = float(
            sky_mag_arcsec2(
                band,
                sky_cfg,
                sun_alt_deg=sun_alt_deg,
                moon_alt_deg=moon_alt_deg,
                moon_phase=moon_phase,
                moon_sep_deg=moon_sep_deg,
                airmass=X_val,
                k_band=float(k_band),
            )
        )

    _SKY_CACHE[key] = sky_val
    return sky_val


def _cached_m5_at_time(
    *,
    target: dict,
    filt: str,
    cfg: PlannerConfig,
    phot_cfg: PhotomConfig,
    sky_provider,
    sky_cfg: SkyModelConfig,
    mjd: float,
    minutes: int = 1,
    site: EarthLocation | None = None,
    tag_best: bool = False,
) -> Optional[float]:
    """Return (and cache) m5 for a target/filter at the given MJD."""

    name_fields = (
        target.get("Name"),
        target.get("name"),
        target.get("ID"),
        target.get("id"),
    )
    name: str | None = None
    for val in name_fields:
        if val is None:
            continue
        try:
            text = str(val).strip()
        except Exception:
            continue
        if text:
            name = text
            break
    if not name:
        return None

    bucket = _mjd_bucket(mjd, minutes)
    m5_key = (_CACHE_TOKEN, name, filt, bucket)
    cached = _M5_CACHE.get(m5_key)
    if cached is not None:
        return cached
    if tag_best:
        best_cached = _m5best_cache_get(name, filt, mjd)
        if best_cached is not None:
            return best_cached

    # Inline candidate coord extraction to avoid cross-module dependency
    try:
        ra = float(target.get("RA_deg"))
        dec = float(target.get("Dec_deg"))
    except Exception:
        return None
    if not math.isfinite(ra) or not math.isfinite(dec):
        return None
    ra_deg, dec_deg = ra, dec

    site_loc = site
    if site_loc is None:
        site_loc = getattr(cfg, "_site_location_cache", None)
        if site_loc is None:
            site_loc = EarthLocation(
                lat=float(cfg.lat_deg) * u.deg,
                lon=float(cfg.lon_deg) * u.deg,
                height=float(cfg.height_m) * u.m,
            )
            setattr(cfg, "_site_location_cache", site_loc)

    try:
        obstime = Time(float(mjd), format="mjd", scale="utc")
        altaz_frame = AltAz(obstime=obstime, location=site_loc)
        sc = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")
        alt_deg = float(sc.transform_to(altaz_frame).alt.deg)
    except Exception:
        alt_deg = _finite_or_none(target.get("max_alt_deg")) or float("nan")

    alt_val = _finite_or_none(alt_deg)
    if alt_val is not None:
        airmass = airmass_from_alt_deg(alt_val)
    else:
        alt_fallback = _finite_or_none(target.get("max_alt_deg"))
        if alt_fallback is not None:
            airmass = airmass_from_alt_deg(alt_fallback)
        else:
            airmass = _finite_or_none(target.get("airmass")) or 1.2
    if not math.isfinite(airmass) or airmass <= 0.0:
        airmass = 1.2

    seeing = (
        float(phot_cfg.fwhm_eff.get(filt, 0.83))
        if getattr(phot_cfg, "fwhm_eff", None)
        else 0.83
    )
    k_band = (
        float(phot_cfg.k_m.get(filt, 0.0)) if getattr(phot_cfg, "k_m", None) else 0.0
    )
    t_vis = float(cfg.exposure_by_filter.get(filt, 30.0))

    sun_alt = _finite_or_none(target.get("sun_alt_policy"))
    moon_alt = _finite_or_none(target.get("moon_alt"))
    moon_phase = _finite_or_none(target.get("moon_phase"))
    moon_sep = _finite_or_none(target.get("moon_sep"))

    m_sky = _cached_sky_mag(
        sky_provider=sky_provider,
        sky_cfg=sky_cfg,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        band=filt,
        airmass=airmass,
        mjd=mjd,
        minutes=minutes,
        sun_alt_deg=sun_alt,
        moon_alt_deg=moon_alt,
        moon_phase=moon_phase,
        moon_sep_deg=moon_sep,
        k_band=k_band,
    )
    try:
        m5_val = float(
            _m5_scale(
                filt,
                m_sky_arcsec2=m_sky,
                seeing_fwhm_arcsec=seeing,
                airmass=airmass,
                t_vis_s=t_vis,
                k_m=k_band,
            )
        )
    except Exception:
        return None

    _M5_CACHE[m5_key] = m5_val
    if tag_best:
        _m5best_cache_put(cfg, name, filt, mjd, m5_val)
    return m5_val


def bump_window_token() -> None:
    """Increment the per-window cache token and clear per-window caches."""
    global _CACHE_TOKEN
    _CACHE_TOKEN += 1
    _SKY_CACHE.clear()
    _M5_CACHE.clear()


def prune_m5best_by_day(cfg: PlannerConfig, current_mjd: float) -> None:
    """Prune the persistent m5 LRU cache by day horizon."""
    _m5best_cache_prune_by_day(cfg, current_mjd)

