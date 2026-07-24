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
from typing import Iterable, Optional, Dict

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


def _sky_cache_key(
    *,
    mjd: float,
    minutes: int,
    ra_deg: float | None,
    dec_deg: float | None,
    band: str,
    airmass: float,
) -> tuple[int, int, int, str, int, int]:
    """Build the exact cache key used by scalar and batch sky lookups."""

    bucket = _mjd_bucket(mjd, minutes)
    ra_val = _finite_or_none(ra_deg)
    dec_val = _finite_or_none(dec_deg)
    X_val = _finite_or_none(airmass)
    ra_q = 0 if ra_val is None else int(round(ra_val * 1e4))
    dec_q = 0 if dec_val is None else int(round(dec_val * 1e4))
    if X_val is None:
        X_val = 1.2
    X_q = int(round(X_val * 100.0))
    return (_CACHE_TOKEN, bucket, ra_q, str(band), dec_q, X_q)


def precache_sky_mags(
    *,
    sky_provider,
    mjd: float,
    samples: Iterable[tuple[float, float, float, Iterable[str]]],
    minutes: int = 1,
) -> int:
    """Vectorize one-MJD Rubin sky query and fill the scalar lookup cache.

    Each sample is ``(ra_deg, dec_deg, airmass, requested_bands)``.  Providers
    without a vectorized ``sky_mags`` method are left untouched.  Any batch
    failure also leaves the cache untouched so later scalar calls retain the
    historical fallback behavior.
    """

    batch_method = getattr(sky_provider, "sky_mags", None)
    if not callable(batch_method):
        return 0

    prepared: list[tuple[float, float, float, tuple[str, ...]]] = []
    union_bands: list[str] = []
    seen_bands: set[str] = set()
    for ra_raw, dec_raw, X_raw, bands_raw in samples:
        ra = _finite_or_none(ra_raw)
        dec = _finite_or_none(dec_raw)
        X = _finite_or_none(X_raw)
        if ra is None or dec is None:
            continue
        if X is None:
            X = 1.2
        requested: list[str] = []
        requested_seen: set[str] = set()
        for value in bands_raw:
            band = str(value).strip().lower()
            if not band or band in requested_seen:
                continue
            requested_seen.add(band)
            requested.append(band)
            if band not in seen_bands:
                seen_bands.add(band)
                union_bands.append(band)
        if requested:
            prepared.append((ra, dec, X, tuple(requested)))
    if not prepared or not union_bands:
        return 0

    try:
        values = batch_method(
            float(mjd),
            [item[0] for item in prepared],
            [item[1] for item in prepared],
            union_bands,
            [item[2] for item in prepared],
        )
    except Exception:
        return 0

    pending: list[tuple[tuple[int, int, int, str, int, int], float]] = []
    try:
        for row_index, (ra, dec, X, requested) in enumerate(prepared):
            for band in requested:
                value = float(values[band][row_index])
                key = _sky_cache_key(
                    mjd=float(mjd),
                    minutes=minutes,
                    ra_deg=ra,
                    dec_deg=dec,
                    band=band,
                    airmass=X,
                )
                pending.append((key, value))
    except Exception:
        return 0

    for key, value in pending:
        _SKY_CACHE[key] = value
    return len(pending)


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
    sun_sep_deg: float | None = None,
    sun_az_rel_deg: float | None = None,
    moon_alt_deg: float | None = None,
    moon_phase: float | None = None,
    moon_sep_deg: float | None = None,
    k_band: float = 0.0,
) -> float:
    """Sky μ (mag/arcsec²) cached per-minute for the target geometry."""

    X_val = _finite_or_none(airmass)
    if X_val is None:
        X_val = 1.2
    key = _sky_cache_key(
        mjd=mjd,
        minutes=minutes,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        band=band,
        airmass=X_val,
    )
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
                sun_sep_deg=sun_sep_deg,
                sun_az_rel_deg=sun_az_rel_deg,
                moon_alt_deg=moon_alt_deg,
                moon_phase=moon_phase,
                moon_sep_deg=moon_sep_deg,
                airmass=X_val,
                k_band=float(k_band),
            )
        )

    _SKY_CACHE[key] = sky_val
    return sky_val


def _target_name(target: dict) -> str | None:
    """Return the scheduler's canonical non-empty target identifier."""

    for val in (
        target.get("Name"),
        target.get("name"),
        target.get("ID"),
        target.get("id"),
    ):
        if val is None:
            continue
        try:
            text = str(val).strip()
        except Exception:
            continue
        if text:
            return text
    return None


def _target_airmass_at_time(
    target: dict,
    mjd: float,
    site_loc: EarthLocation,
) -> tuple[float, float, float] | None:
    """Return ``(ra_deg, dec_deg, airmass)`` using the scalar legacy path."""

    try:
        ra_deg = float(target.get("RA_deg"))
        dec_deg = float(target.get("Dec_deg"))
    except Exception:
        return None
    if not math.isfinite(ra_deg) or not math.isfinite(dec_deg):
        return None

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
    return ra_deg, dec_deg, float(airmass)


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

    name = _target_name(target)
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

    geometry = _target_airmass_at_time(target, float(mjd), site_loc)
    if geometry is None:
        return None
    ra_deg, dec_deg, airmass = geometry

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
    sun_sep = _finite_or_none(target.get("sun_sep"))
    sun_az_rel = _finite_or_none(target.get("sun_az_rel"))
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
        sun_sep_deg=sun_sep,
        sun_az_rel_deg=sun_az_rel,
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


def precache_m5_at_times(
    *,
    requests: Iterable[tuple[dict, str, float, int, bool]],
    cfg: PlannerConfig,
    phot_cfg: PhotomConfig,
    sky_provider,
    site: EarthLocation | None = None,
) -> int:
    """Batch Rubin sky calls for pending m5 cache entries.

    Requests are ``(target, filter, mjd, minute_bucket_size, tag_best)``.
    Geometry and the m5 formula intentionally reuse the scalar helpers above;
    only the Rubin sky-model evaluation is vectorized.  Failed or unsupported
    batches are left absent so the normal scalar path handles them later.
    """

    if not bool(getattr(cfg, "sky_batch_cache_enable", True)):
        return 0
    batch_method = getattr(sky_provider, "sky_mags", None)
    if not callable(batch_method):
        return 0

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

    pending: list[dict[str, object]] = []
    seen_m5_keys: set[tuple[int, str, str, int]] = set()
    geometry_cache: dict[tuple[int, float], tuple[float, float, float] | None] = {}
    for target, filt_raw, mjd_raw, minutes_raw, tag_best in requests:
        name = _target_name(target)
        if not name:
            continue
        filt = str(filt_raw).strip().lower()
        try:
            mjd = float(mjd_raw)
            minutes = max(1, int(minutes_raw))
        except Exception:
            continue
        if not math.isfinite(mjd) or not filt:
            continue
        m5_key = (_CACHE_TOKEN, name, filt, _mjd_bucket(mjd, minutes))
        if m5_key in seen_m5_keys or m5_key in _M5_CACHE:
            continue
        if tag_best and _m5best_cache_get(name, filt, mjd) is not None:
            continue
        seen_m5_keys.add(m5_key)

        geometry_key = (id(target), mjd)
        if geometry_key not in geometry_cache:
            geometry_cache[geometry_key] = _target_airmass_at_time(
                target, mjd, site_loc
            )
        geometry = geometry_cache[geometry_key]
        if geometry is None:
            continue
        ra_deg, dec_deg, airmass = geometry
        provider_limit = _finite_or_none(getattr(sky_provider, "airmass_max", None))
        if provider_limit is not None and airmass > provider_limit:
            # The scalar Rubin provider returns its historical fallback for
            # out-of-range points. Leave these few cases to that exact path.
            continue
        pending.append(
            {
                "target": target,
                "name": name,
                "filt": filt,
                "mjd": mjd,
                "minutes": minutes,
                "tag_best": bool(tag_best),
                "m5_key": m5_key,
                "ra": ra_deg,
                "dec": dec_deg,
                "airmass": airmass,
            }
        )

    by_mjd: dict[float, list[dict[str, object]]] = {}
    for record in pending:
        by_mjd.setdefault(float(record["mjd"]), []).append(record)

    stored = 0
    for mjd, records in by_mjd.items():
        samples: list[dict[str, object]] = []
        sample_index: dict[tuple[str, float, float, float], int] = {}
        union_bands: list[str] = []
        for record in records:
            sample_key = (
                str(record["name"]),
                float(record["ra"]),
                float(record["dec"]),
                float(record["airmass"]),
            )
            index = sample_index.get(sample_key)
            if index is None:
                index = len(samples)
                sample_index[sample_key] = index
                samples.append(
                    {
                        "ra": record["ra"],
                        "dec": record["dec"],
                        "airmass": record["airmass"],
                    }
                )
            record["sample_index"] = index
            filt = str(record["filt"])
            if filt not in union_bands:
                union_bands.append(filt)
        try:
            sky_values = batch_method(
                mjd,
                [float(item["ra"]) for item in samples],
                [float(item["dec"]) for item in samples],
                union_bands,
                [float(item["airmass"]) for item in samples],
            )
            completed: list[tuple[dict[str, object], float]] = []
            for record in records:
                filt = str(record["filt"])
                sky_mag = float(sky_values[filt][int(record["sample_index"])])
                seeing = (
                    float(phot_cfg.fwhm_eff.get(filt, 0.83))
                    if getattr(phot_cfg, "fwhm_eff", None)
                    else 0.83
                )
                k_band = (
                    float(phot_cfg.k_m.get(filt, 0.0))
                    if getattr(phot_cfg, "k_m", None)
                    else 0.0
                )
                t_vis = float(cfg.exposure_by_filter.get(filt, 30.0))
                m5_value = float(
                    _m5_scale(
                        filt,
                        m_sky_arcsec2=sky_mag,
                        seeing_fwhm_arcsec=seeing,
                        airmass=float(record["airmass"]),
                        t_vis_s=t_vis,
                        k_m=k_band,
                    )
                )
                completed.append((record, m5_value))
        except Exception:
            continue

        for record, m5_value in completed:
            m5_key = record["m5_key"]
            if not isinstance(m5_key, tuple):
                continue
            _M5_CACHE[m5_key] = m5_value
            if bool(record["tag_best"]):
                _m5best_cache_put(
                    cfg,
                    str(record["name"]),
                    str(record["filt"]),
                    float(record["mjd"]),
                    m5_value,
                )
            stored += 1
    return stored


def bump_window_token() -> None:
    """Increment the per-window cache token and clear per-window caches."""
    global _CACHE_TOKEN
    _CACHE_TOKEN += 1
    _SKY_CACHE.clear()
    _M5_CACHE.clear()


def prune_m5best_by_day(cfg: PlannerConfig, current_mjd: float) -> None:
    """Prune the persistent m5 LRU cache by day horizon."""
    _m5best_cache_prune_by_day(cfg, current_mjd)
