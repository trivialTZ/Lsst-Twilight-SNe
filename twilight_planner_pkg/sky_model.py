from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol

from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

# mypy: ignore-errors


from .astro_utils import airmass_from_alt_deg

@dataclass
class SkyModelConfig:
    """Settings controlling simple sky-brightness estimates."""

    dark_sky_mag: dict | None = None
    twilight_delta_mag: float = 2.5
    use_override: bool = False
    override_mag: Optional[float] = None

    def __post_init__(self) -> None:
        if self.dark_sky_mag is None:
            # Zenith dark-sky brightness from SMTN-002 (mag/arcsec^2)
            self.dark_sky_mag = {
                "u": 23.05,
                "g": 22.25,
                "r": 21.20,
                "i": 20.46,
                "z": 19.61,
                "y": 18.60,
            }


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

    base = cfg.dark_sky_mag[band]
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
    """Sky provider using :func:`sky_mag_arcsec2` with optional Sun altitude."""

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
        sun_alt_deg: float | None = None
        if mjd is not None and self.site is not None:
            t = Time(mjd, format="mjd", scale="utc")
            sun_alt_deg = float(
                get_sun(t).transform_to(AltAz(obstime=t, location=self.site)).alt.deg
            )
        return sky_mag_arcsec2(
            band,
            self.cfg,
            sun_alt_deg=sun_alt_deg,
            airmass=airmass,
        )


class RubinSkyProvider:
    """Sky provider wrapping ``rubin_sim.skybrightness``."""

    def __init__(self, site: Optional[str] = None):
        import rubin_sim.skybrightness as sb

        self.model = sb.SkyModel() if site is None else sb.SkyModel(site=site)

    def sky_mag(
        self,
        mjd: float | None,
        ra_deg: float | None,
        dec_deg: float | None,
        band: str,
        airmass: float,
    ) -> float:
        # Attempt to pass through geometry for a full sky model; fall back to
        # airmass-only if the API does not accept additional keys.
        params = {"filter": band, "airmass": airmass}
        if mjd is not None:
            params["mjd"] = mjd
        if ra_deg is not None and dec_deg is not None:
            params["ra"] = float(ra_deg)
            params["dec"] = float(dec_deg)
        try:
            out = self.model.return_mags(params)
            return float(out.get("mag_sky", out.get("sky", 22.0)))
        except Exception:
            return float(self.model.return_mags({"filter": band, "airmass": airmass})["mag_sky"])
