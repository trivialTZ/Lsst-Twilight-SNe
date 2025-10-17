from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Protocol

from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

# mypy: ignore-errors


from .astro_utils import airmass_from_alt_deg


DEFAULT_DARK_SKY_MAG = {
    "u": 23.05,
    "g": 22.25,
    "r": 21.20,
    "i": 20.46,
    "z": 19.61,
    "y": 18.60,
}

@dataclass
class SkyModelConfig:
    """Settings controlling simple sky-brightness estimates."""

    dark_sky_mag: dict | None = None
    twilight_delta_mag: float = 2.5
    use_override: bool = False
    override_mag: Optional[float] = None

    def __post_init__(self) -> None:
        base = {band: float(val) for band, val in DEFAULT_DARK_SKY_MAG.items()}
        if self.dark_sky_mag is None:
            # Zenith dark-sky brightness from SMTN-002 (mag/arcsec^2)
            self.dark_sky_mag = base
            return

        merged = base
        for key, val in self.dark_sky_mag.items():
            try:
                band = str(key).strip()
            except Exception:
                continue
            if not band:
                continue
            try:
                merged[band] = float(val)
            except Exception:
                continue
        self.dark_sky_mag = merged


def sky_mag_arcsec2(
    band: str,
    cfg: SkyModelConfig,
    sun_alt_deg: float | None = None,
    moon_alt_deg: float | None = None,
    moon_phase: float | None = None,
    moon_sep_deg: float | None = None,
    airmass: float = 1.0,
    *,
    k_band: float = 0.2,
) -> float:
    """Return sky brightness for a band in mag/arcsec².

    Parameters
    ----------
    band
        Photometric band name (``"g"``, ``"r"``, ...).
    cfg
        Configuration controlling dark-sky reference magnitudes.
    sun_alt_deg
        Altitude of the Sun in degrees.  If provided the sky brightness is
        brightened when the Sun is above astronomical twilight using a simple
        linear model following Tyson & Gal (1993).  Negative values denote the
        Sun below the horizon.  ``None`` falls back to the fixed
        ``twilight_delta_mag`` offset used historically.

    Returns
    -------
    float
        Sky brightness in mag/arcsec².
    """

    if cfg.use_override and cfg.override_mag is not None:
        return cfg.override_mag

    base = float(cfg.dark_sky_mag.get(band, DEFAULT_DARK_SKY_MAG.get(band, 21.0)))
    if sun_alt_deg is None:
        mu_twilight = base - cfg.twilight_delta_mag
    else:
        sun_alt = float(sun_alt_deg)
        if sun_alt <= -18.0:
            mu_twilight = base
        else:
            alt = min(sun_alt, 0.0)
            factors = {"g": 0.35, "r": 0.30, "i": 0.15, "z": 0.10, "y": 0.05}
            penalty = (alt + 18.0) * factors.get(band, 0.2)
            mu_twilight = base - penalty

    # Convert twilight sky brightness to flux-like units. Use a conventional
    # zero-point for surface-brightness conversions:
    #   mu [mag/arcsec^2] ≈ 26.33 - 2.5 log10(B)
    # Hence B ∝ 10^(-0.4*(mu - 26.33)). The absolute scale cancels when
    # adding moonlight and converting back to mag/arcsec^2.
    SB_ZP = 26.33
    flux_twilight = 10 ** (-0.4 * (mu_twilight - SB_ZP))

    # Moon contribution (Krisciunas & Schaefer 1991). Falls back to 0 if inputs
    # are missing or the Moon is below the horizon.
    flux_moon = 0.0
    if (
        moon_alt_deg is not None
        and moon_alt_deg > 0.0
        and moon_phase is not None
        and moon_phase > 0.0
        and moon_sep_deg is not None
    ):
        rho = max(1e-3, float(abs(moon_sep_deg)))
        # Phase-angle mapping: full Moon α≈0°, new Moon α≈180°.
        # Illuminated fraction f ∈ [0,1] relates via cos α = 2f - 1.
        cos_alpha = max(-1.0, min(1.0, 2.0 * float(moon_phase) - 1.0))
        alpha_deg = math.degrees(math.acos(cos_alpha))
        I_star = 10 ** (-0.4 * (3.84 + 0.026 * abs(alpha_deg) + 4e-9 * alpha_deg**4))
        rho_rad = math.radians(rho)
        f_R = 10 ** 5.36 * (1.06 + math.cos(rho_rad) ** 2)
        if rho < 10.0:
            f_M = 6.2e7 / (rho**2 + 1e-6)
        else:
            f_M = 10 ** (6.15 - rho / 40.0)
        f = f_R + f_M
        X_target = max(1.0, float(airmass))
        try:
            X_moon = airmass_from_alt_deg(float(moon_alt_deg))
        except Exception:
            X_moon = 10.0
        if not math.isfinite(X_moon) or X_moon <= 0:
            X_moon = 10.0
        k = float(k_band)
        # K&S attenuation uses 10^{-0.4 k X_moon} along the Moon path.
        flux_moon = f * I_star * 10 ** (-0.4 * k * X_moon) * (
            1.0 - 10 ** (-0.4 * k * X_target)
        )
    flux_total = flux_twilight + max(0.0, flux_moon)
    return -2.5 * math.log10(max(1e-9, flux_total)) + SB_ZP


class SkyProvider(Protocol):
    """Protocol for sky-brightness providers."""

    def sky_mag(
        self,
        mjd: float | None,
        ra_deg: float | None,
        dec_deg: float | None,
        band: str,
        airmass: float,
    ) -> float: ...


class SimpleSkyProvider:
    """Sky provider using sky_mag_arcsec2 with Sun+Moon at (RA,Dec,MJD).

    When full geometry (MJD, RA/Dec and site) is available, compute Sun and
    Moon terms at the target position and time. If any of those are missing,
    fall back to the historical Sun‑only behavior using the configured
    twilight offset.
    """

    def __init__(self, cfg: SkyModelConfig, site: EarthLocation | None = None):
        self.cfg = cfg
        self.site = site

    def sky_mag(
        self,
        mjd: float | None,
        ra_deg: float | None,
        dec_deg: float | None,
        band: str,
        airmass: float,
    ) -> float:
        # If time/site are missing, keep the simpler Sun‑only path
        if mjd is None or self.site is None or ra_deg is None or dec_deg is None:
            sun_alt_deg: float | None = None
            if mjd is not None and self.site is not None:
                t = Time(mjd, format="mjd", scale="utc")
                sun_alt_deg = float(
                    get_sun(t)
                    .transform_to(AltAz(obstime=t, location=self.site))
                    .alt.deg
                )
            return sky_mag_arcsec2(
                band,
                self.cfg,
                sun_alt_deg=sun_alt_deg,
                airmass=airmass,
            )

        # Full geometry at (RA,Dec,MJD): include Moon altitude, separation and phase
        import numpy as np
        from astropy.coordinates import get_body, SkyCoord
        import astropy.units as u

        t = Time(mjd, format="mjd", scale="utc")
        altaz = AltAz(obstime=t, location=self.site)
        sc = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")
        tgt_altaz = sc.transform_to(altaz)
        sun_altaz = get_sun(t).transform_to(altaz)
        moon_altaz = get_body("moon", t).transform_to(altaz)

        sun_alt_deg = float(sun_altaz.alt.deg)
        moon_alt_deg = float(moon_altaz.alt.deg)
        moon_sep_deg = float(moon_altaz.separation(tgt_altaz).deg)
        # Fractional illumination in [0,1] from Sun–Moon angle
        phase = 0.5 * (1.0 - np.cos(np.deg2rad(moon_altaz.separation(sun_altaz).deg)))

        # Prefer the passed airmass if finite; otherwise compute from current altitude
        try:
            X = float(airmass)
        except Exception:
            X = float("nan")
        if not math.isfinite(X) or X < 1.0:
            X = airmass_from_alt_deg(float(tgt_altaz.alt.deg))

        return sky_mag_arcsec2(
            band,
            self.cfg,
            sun_alt_deg=sun_alt_deg,
            moon_alt_deg=moon_alt_deg,
            moon_phase=phase,
            moon_sep_deg=moon_sep_deg,
            airmass=X,
        )


class RubinSkyProvider:
    """Sky provider wrapping rubin_sim.skybrightness using RA/Dec when available.

    Behavior
    --------
    - If ``mjd``, ``ra_deg`` and ``dec_deg`` are provided, uses
      ``SkyModel.set_ra_dec_mjd`` (preferred; full geometry).
    - Otherwise, falls back to ``SkyModel.set_params`` with
      ``airmass``, ``sun_alt`` and an optional Sun-relative azimuth
      (supports both ``azs`` and ``azRelSun`` signatures across versions).

    With ``mags=True`` the return value is a dict keyed by filter names whose
    values are arrays; we extract the first element.
    """

    def __init__(self, site: EarthLocation | None = None):
        import rubin_sim.skybrightness as sb

        # Try a few constructor signatures observed across versions.
        errors: list[Exception] = []
        self.model = None  # type: ignore[assignment]
        for kwargs in (
            {"mags": True},
            {"mags": True, "filter_names": ["u","g","r","i","z","y"]},
            {},
        ):
            try:
                self.model = sb.SkyModel(**kwargs)
                break
            except Exception as e:  # pragma: no cover - depends on environment
                errors.append(e)
                self.model = None
        if self.model is None:
            # Surface the first error; caller will catch and fall back to toy.
            raise errors[0]
        self.site = site
        # Clip threshold for the parametric twilight component (deg)
        self.sun_alt_min_deg: float = -11.0

    def sky_mag(
        self,
        mjd: float | None,
        ra_deg: float | None,
        dec_deg: float | None,
        band: str,
        airmass: float,
        *,
        sun_alt_deg: float | None = None,
        az_sun_rel_deg: float = 90.0,
    ) -> float:
        import numpy as np
        _catch = warnings.catch_warnings()
        _catch.__enter__()
        warnings.filterwarnings(
            "ignore",
            message="Extrapolating twilight beyond a sun altitude of -11 degrees",
        )
        try:
            # Path A: use full geometry if given
            if mjd is not None and ra_deg is not None and dec_deg is not None:
                # Some versions name arguments lon/lat, others ra/dec
                try:
                    self.model.set_ra_dec_mjd(
                        lon=float(ra_deg), lat=float(dec_deg), mjd=float(mjd),
                        degrees=True, filter_names=[band]
                    )
                except TypeError:
                    self.model.set_ra_dec_mjd(
                        ra=float(ra_deg), dec=float(dec_deg), mjd=float(mjd),
                        degrees=True, filter_names=[band]
                    )
            else:
                # Path B: parameterized twilight point used by our figures
                if sun_alt_deg is None and mjd is not None:
                    # Derive Sun altitude if only MJD is known
                    from astropy.coordinates import AltAz, get_sun
                    from astropy.time import Time
                    import astropy.units as u
                    # Prefer configured site if supplied; else fall back to Rubin site
                    site = self.site or EarthLocation(
                        lat=-30.2446 * u.deg, lon=-70.7494 * u.deg, height=2647 * u.m
                    )
                    t = Time(mjd, format="mjd", scale="utc")
                    sun_alt_deg = float(
                        get_sun(t).transform_to(AltAz(obstime=t, location=site)).alt.deg
                    )
                if sun_alt_deg is None:
                    sun_alt_deg = -12.0
                if sun_alt_deg < self.sun_alt_min_deg:
                    sun_alt_deg = self.sun_alt_min_deg
                # Be resilient to API changes: try common signatures
                set_params_ok = False
                for kwargs in (
                    dict(
                        airmass=float(airmass),
                        sun_alt=float(sun_alt_deg),
                        azs=float(az_sun_rel_deg),
                        degrees=True,
                        filter_names=[band],
                    ),
                    dict(
                        airmass=float(airmass),
                        sun_alt=float(sun_alt_deg),
                        azRelSun=float(az_sun_rel_deg),
                        degrees=True,
                        filter_names=[band],
                    ),
                    dict(
                        airmass=float(airmass),
                        sun_alt=float(sun_alt_deg),
                        degrees=True,
                        filter_names=[band],
                    ),
                ):
                    try:
                        self.model.set_params(**kwargs)
                        set_params_ok = True
                        break
                    except TypeError:
                        continue
                if not set_params_ok:
                    # Last-ditch: minimal required params
                    self.model.set_params(airmass=float(airmass), sun_alt=float(sun_alt_deg))

            mags = self.model.return_mags()
            # rubin_sim returns a dict: { 'g': array([...]), ... }
            arr = mags.get(band, None) if isinstance(mags, dict) else None
            if arr is None:
                # Some versions may return an object with attributes
                try:
                    arr = getattr(mags, band)
                except Exception:
                    return DEFAULT_DARK_SKY_MAG.get(band, 21.0)
            return float(np.asarray(arr).ravel()[0])
        except Exception:
            # Absolute fallback: return a reasonable dark-sky value
            return DEFAULT_DARK_SKY_MAG.get(band, 21.0)
        finally:
            try:
                _catch.__exit__(None, None, None)
            except Exception:
                pass


def rubin_sim_mu_sky_by_sun_alt(
    band: str,
    sun_alt_deg: list[float] | tuple[float, ...] | "np.ndarray",
    airmass: float = 1.2,
) -> "np.ndarray":
    """Vectorized rubin_sim sky μ via set_params/return_mags.

    Safer wrapper that loops over altitudes with ``set_params``. If rubin_sim is
    unavailable, raises ImportError so callers can fall back.
    """
    import numpy as np
    try:
        import rubin_sim.skybrightness as sb
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError("rubin_sim is not installed or import failed") from e

    model = sb.SkyModel(mags=True)
    alts = np.atleast_1d(np.array(sun_alt_deg, dtype=float))
    out = np.full_like(alts, np.nan, dtype=float)
    for i, alt in enumerate(alts):
        try:
            model.set_params(airmass=float(airmass), sun_alt=float(alt), azs=90.0, degrees=True, filter_names=[band])
            mags = model.return_mags()
            arr = mags.get(band) if isinstance(mags, dict) else None
            out[i] = float(np.asarray(arr).ravel()[0]) if arr is not None else np.nan
        except Exception:
            out[i] = np.nan
    # Fill any NaNs with a constant fallback to avoid breaking plots
    if np.any(~np.isfinite(out)):
        fallback = DEFAULT_DARK_SKY_MAG.get(band, 21.0)
        out = np.where(np.isfinite(out), out, fallback)
    return out
