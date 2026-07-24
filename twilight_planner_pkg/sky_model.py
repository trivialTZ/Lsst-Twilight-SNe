from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

# mypy: ignore-errors


from .astro_utils import airmass_from_alt_deg

if TYPE_CHECKING:
    import numpy as np


DEFAULT_DARK_SKY_MAG = {
    "u": 23.05,
    "g": 22.25,
    "r": 21.20,
    "i": 20.46,
    "z": 19.61,
    "y": 18.60,
}


# Rubin ``TwilightInterp`` coefficients after interpolation onto the LSST
# effective wavelengths. Each tuple is:
#   (twilight/dark flux ratio at Sun altitude -12 deg,
#    solar-altitude slope per radian,
#    airmass flux exponent,
#    Sun-facing azimuth flux exponent).
# These reproduce the functional coefficients shipped in rubin_sim 2.6.1.
# Rubin notes that z/y are provisional extrapolations of the empirical fits.
RUBIN_TWILIGHT_PARAMS = {
    "u": (16.000000, 22.962212, 0.285863, 0.299903),
    "g": (6.846105, 23.005221, 0.288207, 0.302541),
    "r": (1.841221, 22.068044, 0.298340, 0.330527),
    "i": (2.263650, 22.782541, 0.298616, 0.318993),
    "z": (2.290000, 24.080000, 0.300000, 0.300000),
    "y": (2.000000, 24.080000, 0.300000, 0.300000),
}

# Rubin's empirical fit used data only through this solar altitude. The
# upstream implementation permits brighter twilight but emits an extrapolation
# warning, so schedules retain an explicit provenance flag above this limit.
TWILIGHT_FIT_MAX_SUN_ALT_DEG = -11.0


@dataclass
class SkyModelConfig:
    """Settings controlling simple sky-brightness estimates."""

    dark_sky_mag: dict | None = None
    twilight_delta_mag: float = 2.5
    twilight_airmass_coeff: float | None = None
    twilight_sunward_azimuth_coeff: float | None = None
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
    sun_sep_deg: float | None = None,
    sun_az_rel_deg: float | None = None,
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
        Altitude of the Sun in degrees. If provided, scattered-sunlight flux
        follows Rubin's empirical ``TwilightInterp`` equation. Negative values
        denote the Sun below the horizon. Values above -11 degrees extrapolate
        beyond the data used for the Rubin fit. ``None`` falls back to the
        fixed ``twilight_delta_mag`` offset used historically.
    sun_sep_deg
        Topocentric target--Sun separation in degrees. It is carried alongside
        the sky estimate for consistent geometry, but the empirical twilight
        brightness term uses Sun-relative azimuth and target airmass.
    sun_az_rel_deg
        Absolute target azimuth relative to the Sun in degrees. Zero points
        toward the Sun and 180 points toward the anti-Sun. When supplied, the
        fallback follows the geometry of Rubin's ``twilight_func`` by making
        the high-airmass Sun-facing hemisphere brighter.

    Returns
    -------
    float
        Sky brightness in mag/arcsec².
    """

    if cfg.use_override and cfg.override_mag is not None:
        return cfg.override_mag

    band = str(band).strip().lower()
    base = float(cfg.dark_sky_mag.get(band, DEFAULT_DARK_SKY_MAG.get(band, 21.0)))

    # Conventional zero-point for surface-brightness conversions:
    #   mu [mag/arcsec^2] = 26.33 - 2.5 log10(B).
    # Only relative fluxes matter when adding twilight and moonlight.
    SB_ZP = 26.33
    flux_dark = 10 ** (-0.4 * (base - SB_ZP))
    if sun_alt_deg is None:
        flux_background = flux_dark * 10 ** (0.4 * float(cfg.twilight_delta_mag))
    else:
        sun_alt = float(sun_alt_deg)
        alt = min(sun_alt, 0.0)
        ratio_at_minus12, slope_per_radian, am_fit, az_fit = (
            RUBIN_TWILIGHT_PARAMS.get(band, (2.0, 23.0, 0.30, 0.30))
        )
        try:
            X_twilight = float(airmass)
        except Exception:
            X_twilight = 1.0
        if not math.isfinite(X_twilight):
            X_twilight = 1.0
        X_twilight = min(max(X_twilight, 1.0), 3.0)
        x_minus_one = X_twilight - 1.0

        am_coeff = (
            float(cfg.twilight_airmass_coeff)
            if cfg.twilight_airmass_coeff is not None
            else am_fit
        )
        az_coeff = (
            float(cfg.twilight_sunward_azimuth_coeff)
            if cfg.twilight_sunward_azimuth_coeff is not None
            else az_fit
        )
        exponent = slope_per_radian * math.radians(alt + 12.0)
        exponent += am_coeff * x_minus_one
        if sun_az_rel_deg is not None:
            az_rel = abs(float(sun_az_rel_deg)) % 360.0
            if az_rel > 180.0:
                az_rel = 360.0 - az_rel
            sunward_cos = max(0.0, math.cos(math.radians(az_rel)))
            exponent += az_coeff * sunward_cos * x_minus_one

        # Rubin models the twilight component in flux. Add it to the
        # configured dark-sky flux rather than linearly subtracting
        # magnitudes; this naturally converges to dark sky in deep
        # twilight and follows the measured ~1 mag/deg bright-twilight
        # slope.
        twilight_to_dark = ratio_at_minus12 * 10**exponent
        flux_background = flux_dark * (1.0 + twilight_to_dark)

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
    flux_total = flux_background + max(0.0, flux_moon)
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

        # Full geometry at (RA,Dec,MJD): include Sun direction plus Moon
        # altitude, separation, and phase.
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
        sun_sep_deg = float(sun_altaz.separation(tgt_altaz).deg)
        sun_az_rel_deg = abs(
            (
                float(tgt_altaz.az.deg)
                - float(sun_altaz.az.deg)
                + 180.0
            )
            % 360.0
            - 180.0
        )
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
            sun_sep_deg=sun_sep_deg,
            sun_az_rel_deg=sun_az_rel_deg,
            moon_alt_deg=moon_alt_deg,
            moon_phase=phase,
            moon_sep_deg=moon_sep_deg,
            airmass=X,
        )


class RubinSkyProvider:
    """Sky provider wrapping rubin_sim.skybrightness.

    Path A (preferred; full physics)
    --------------------------------
    - If ``mjd``, ``ra_deg`` and ``dec_deg`` are provided, call
      ``SkyModel.set_ra_dec_mjd``. rubin_sim then computes airmass, solar/lunar
      geometry, zodiacal light, twilight, etc. The ``airmass`` argument to
      ``sky_mag`` is ignored in this path; true airmass comes from RA/Dec/MJD.

    Path B (parametric twilight for schematic plots)
    -----------------------------------------------
    - If RA/Dec are missing, fall back to ``SkyModel.set_params`` using
      ``airmass``, ``sun_alt`` and a Sun-relative azimuth. This yields an
      average twilight sky rather than a specific observation.

    Notes
    -----
    With ``mags=True`` rubin_sim returns a dict keyed by filter names; we take
    the first element of each array.
    """

    def __init__(
        self,
        site: EarthLocation | None = None,
        *,
        strict: bool = False,
        fallback_cfg: SkyModelConfig | None = None,
    ):
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
        self.strict = strict
        self.fallback_provider = SimpleSkyProvider(
            fallback_cfg or SkyModelConfig(), site=site
        )
        # Twilight parametric range (deg) and airmass bounds used for clamping.
        self.sun_alt_min_deg: float = -20.0
        self.sun_alt_max_deg: float = 15.0
        self.airmass_min: float = 1.0
        self.airmass_max: float = float(getattr(self.model, "airmass_limit", 3.0))

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

        # Clamp airmass to the supported range for the parametric path.
        try:
            X = float(airmass)
        except Exception:
            X = float("nan")
        if not math.isfinite(X):
            if self.strict:
                raise ValueError(f"Invalid airmass value {airmass!r}")
            X = self.airmass_min
        if X < self.airmass_min or X > self.airmass_max:
            if self.strict:
                raise ValueError(
                    f"airmass={X:.3f} outside supported "
                    f"[{self.airmass_min}, {self.airmass_max}]"
                )
            X = min(max(X, self.airmass_min), self.airmass_max)

        try:
            # Path A: use full geometry if given
            if mjd is not None and ra_deg is not None and dec_deg is not None:
                # Some versions name arguments ra/dec, others lon/lat
                try:
                    self.model.set_ra_dec_mjd(
                        ra=float(ra_deg), dec=float(dec_deg), mjd=float(mjd),
                        degrees=True, filter_names=[band]
                    )
                except TypeError:
                    self.model.set_ra_dec_mjd(
                        lon=float(ra_deg), lat=float(dec_deg), mjd=float(mjd),
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
                # Default to ~ESO twilight pivot if still missing.
                if sun_alt_deg is None:
                    sun_alt_deg = -12.0
                try:
                    sun_alt = float(sun_alt_deg)
                except Exception:
                    if self.strict:
                        raise
                    sun_alt = -12.0
                if not math.isfinite(sun_alt):
                    if self.strict:
                        raise ValueError(f"Invalid sun_alt_deg value {sun_alt_deg!r}")
                    sun_alt = -12.0
                if sun_alt < self.sun_alt_min_deg or sun_alt > self.sun_alt_max_deg:
                    if self.strict:
                        raise ValueError(
                            f"sun_alt_deg={sun_alt:.2f} outside "
                            f"[{self.sun_alt_min_deg}, {self.sun_alt_max_deg}]"
                        )
                    sun_alt = min(
                        max(sun_alt, self.sun_alt_min_deg),
                        self.sun_alt_max_deg,
                    )
                # Be resilient to API changes: try common signatures
                set_params_ok = False
                # rubin_sim historically accepted scalar floats for set_params,
                # but NumPy 2.0 turns `ndarray(0d).nonzero()` into a hard error.
                # Passing 1-element arrays avoids triggering that path inside
                # rubin_sim and remains compatible with older NumPy versions.
                X_arr = np.atleast_1d(np.array(X, dtype=float))
                sun_alt_arr = np.atleast_1d(np.array(sun_alt, dtype=float))
                az_arr = np.atleast_1d(np.array(float(az_sun_rel_deg), dtype=float))
                for kwargs in (
                    dict(
                        airmass=X_arr,
                        sun_alt=sun_alt_arr,
                        azs=az_arr,
                        degrees=True,
                        filter_names=[band],
                    ),
                    dict(
                        airmass=X_arr,
                        sun_alt=sun_alt_arr,
                        azRelSun=az_arr,
                        degrees=True,
                        filter_names=[band],
                    ),
                    dict(
                        airmass=X_arr,
                        sun_alt=sun_alt_arr,
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
                    self.model.set_params(airmass=X_arr, sun_alt=sun_alt_arr)

            mags = self.model.return_mags()
            # rubin_sim returns a dict: { 'g': array([...]), ... }
            arr = mags.get(band, None) if isinstance(mags, dict) else None
            if arr is None:
                # Some versions may return an object with attributes
                try:
                    arr = getattr(mags, band)
                except Exception:
                    if self.strict:
                        raise
                    return self.fallback_provider.sky_mag(
                        mjd, ra_deg, dec_deg, band, X
                    )
            return float(np.asarray(arr).ravel()[0])
        except Exception:
            # Fail over to the geometry-aware model. Returning a dark-sky
            # constant here would be dangerously optimistic during twilight.
            if self.strict:
                raise
            return self.fallback_provider.sky_mag(
                mjd, ra_deg, dec_deg, band, X
            )
        finally:
            try:
                _catch.__exit__(None, None, None)
            except Exception:
                pass

    def sky_mags(
        self,
        mjd: float,
        ra_deg,
        dec_deg,
        bands,
        airmass=None,
    ) -> dict[str, "np.ndarray"]:
        """Return full-geometry sky magnitudes for many targets and bands.

        This is the vectorized equivalent of repeated :meth:`sky_mag` calls
        through Path A.  Rubin's full-geometry model derives airmass from
        ``RA/Dec/MJD``, so ``airmass`` is accepted only for API symmetry and
        is intentionally ignored.  Unlike :meth:`sky_mag`, this helper raises
        on failure; callers can then leave their cache empty and fall back to
        the scalar path without changing the scientific behavior.
        """

        import numpy as np

        ra = np.atleast_1d(np.asarray(ra_deg, dtype=float))
        dec = np.atleast_1d(np.asarray(dec_deg, dtype=float))
        try:
            ra, dec = np.broadcast_arrays(ra, dec)
        except ValueError as exc:
            raise ValueError("ra_deg and dec_deg are not broadcast-compatible") from exc
        ra = np.asarray(ra, dtype=float).ravel()
        dec = np.asarray(dec, dtype=float).ravel()
        if ra.size == 0:
            return {}
        if not np.all(np.isfinite(ra)) or not np.all(np.isfinite(dec)):
            raise ValueError("ra_deg and dec_deg must be finite")
        mjd_value = float(mjd)
        if not math.isfinite(mjd_value):
            raise ValueError(f"Invalid MJD value {mjd!r}")

        band_list: list[str] = []
        seen: set[str] = set()
        for value in bands:
            band = str(value).strip().lower()
            if band and band not in seen:
                seen.add(band)
                band_list.append(band)
        if not band_list:
            return {}

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Extrapolating twilight beyond a sun altitude of -11 degrees",
            )
            try:
                self.model.set_ra_dec_mjd(
                    ra=ra,
                    dec=dec,
                    mjd=mjd_value,
                    degrees=True,
                    filter_names=band_list,
                )
            except TypeError:
                self.model.set_ra_dec_mjd(
                    lon=ra,
                    lat=dec,
                    mjd=mjd_value,
                    degrees=True,
                    filter_names=band_list,
                )
            mags = self.model.return_mags()

        result: dict[str, np.ndarray] = {}
        for band in band_list:
            arr = mags.get(band) if isinstance(mags, dict) else getattr(mags, band)
            values = np.asarray(arr, dtype=float).ravel()
            if values.size == 1 and ra.size > 1:
                values = np.full(ra.size, float(values[0]), dtype=float)
            if values.size != ra.size:
                raise ValueError(
                    f"Rubin sky model returned {values.size} {band}-band values "
                    f"for {ra.size} targets"
                )
            result[band] = values.copy()
        return result


def rubin_sim_mu_sky_by_sun_alt(
    band: str,
    sun_alt_deg: list[float] | tuple[float, ...] | "np.ndarray",
    airmass: float = 1.2,
) -> "np.ndarray":
    """Vectorized rubin_sim twilight sky μ via set_params/return_mags.

    This uses parametric geometry (airmass + sun_alt + azRelSun=90°), i.e.
    a typical twilight brightness rather than a specific RA/Dec/MJD. For
    concrete observations use ``RubinSkyProvider.sky_mag`` Path A.
    """
    import numpy as np
    try:
        import rubin_sim.skybrightness as sb
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError("rubin_sim is not installed or import failed") from e

    model = sb.SkyModel(mags=True)
    airmass_min = 1.0
    airmass_max = float(getattr(model, "airmass_limit", 3.0))
    sun_alt_min_deg = -20.0
    sun_alt_max_deg = 15.0

    alts = np.atleast_1d(np.array(sun_alt_deg, dtype=float))
    out = np.full_like(alts, np.nan, dtype=float)

    try:
        X_in = float(airmass)
    except Exception:
        X_in = float("nan")
    if not np.isfinite(X_in):
        X_in = airmass_min
    X_in = min(max(X_in, airmass_min), airmass_max)

    _catch = warnings.catch_warnings()
    _catch.__enter__()
    warnings.filterwarnings(
        "ignore",
        message="Extrapolating twilight beyond a sun altitude of -11 degrees",
    )
    try:
        for i, alt in enumerate(alts):
            try:
                alt_val = float(alt)
                if not np.isfinite(alt_val):
                    alt_val = -12.0
                alt_val = min(max(alt_val, sun_alt_min_deg), sun_alt_max_deg)

                model.set_params(
                    airmass=np.atleast_1d(np.array(X_in, dtype=float)),
                    sun_alt=np.atleast_1d(np.array(alt_val, dtype=float)),
                    azs=np.atleast_1d(np.array(90.0, dtype=float)),
                    degrees=True,
                    filter_names=[band],
                )
                mags = model.return_mags()
                arr = mags.get(band) if isinstance(mags, dict) else None
                out[i] = float(np.asarray(arr).ravel()[0]) if arr is not None else np.nan
            except Exception:
                out[i] = np.nan
    finally:
        try:
            _catch.__exit__(None, None, None)
        except Exception:
            pass
    # Fill any NaNs with a constant fallback to avoid breaking plots
    if np.any(~np.isfinite(out)):
        fallback = DEFAULT_DARK_SKY_MAG.get(band, 21.0)
        out = np.where(np.isfinite(out), out, fallback)
    return out
