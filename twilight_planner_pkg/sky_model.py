from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time


@dataclass
class SkyModelConfig:
    """Settings controlling simple sky-brightness estimates."""

    dark_sky_mag: dict | None = None
    twilight_delta_mag: float = 2.5
    use_override: bool = False
    override_mag: Optional[float] = None

    def __post_init__(self):
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
    band: str, cfg: SkyModelConfig, sun_alt_deg: float | None = None
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

    if sun_alt_deg is None:
        return cfg.dark_sky_mag[band] - cfg.twilight_delta_mag

    if sun_alt_deg <= -18.0:
        return cfg.dark_sky_mag[band]

    alt = min(sun_alt_deg, 0.0)
    factors = {"g": 0.35, "r": 0.30, "i": 0.15, "z": 0.10, "y": 0.05}
    penalty = (alt + 18.0) * factors.get(band, 0.2)
    return cfg.dark_sky_mag[band] - penalty


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
        return sky_mag_arcsec2(band, self.cfg, sun_alt_deg)


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
        return float(
            self.model.return_mags({"filter": band, "airmass": airmass})["mag_sky"]
        )
