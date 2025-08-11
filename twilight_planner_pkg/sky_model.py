from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class SkyModelConfig:
    """Settings controlling simple sky-brightness estimates."""

    dark_sky_mag: dict | None = None
    twilight_delta_mag: float = 2.5
    use_override: bool = False
    override_mag: Optional[float] = None

    def __post_init__(self):
        if self.dark_sky_mag is None:
            self.dark_sky_mag = {"u":22.9, "g":22.2, "r":21.2, "i":20.5, "z":19.9, "y":18.9}


def sky_mag_arcsec2(band: str, cfg: SkyModelConfig) -> float:
    """Return sky brightness for a band in mag/arcsec^2."""

    if cfg.use_override and cfg.override_mag is not None:
        return cfg.override_mag
    return cfg.dark_sky_mag[band] - cfg.twilight_delta_mag


class SkyProvider(Protocol):
    """Protocol for sky-brightness providers."""

    def sky_mag(self, mjd: float | None, ra_deg: float | None, dec_deg: float | None, band: str, airmass: float) -> float:
        ...


class SimpleSkyProvider:
    """Sky provider using the simple twilight-delta model."""

    def __init__(self, cfg: SkyModelConfig):
        self.cfg = cfg

    def sky_mag(self, mjd: float | None, ra_deg: float | None, dec_deg: float | None, band: str, airmass: float) -> float:
        return sky_mag_arcsec2(band, self.cfg)


class RubinSkyProvider:
    """Sky provider wrapping ``rubin_sim.skybrightness``."""

    def __init__(self, site: Optional[str] = None):
        import rubin_sim.skybrightness as sb

        self.model = sb.SkyModel() if site is None else sb.SkyModel(site=site)

    def sky_mag(self, mjd: float | None, ra_deg: float | None, dec_deg: float | None, band: str, airmass: float) -> float:
        return float(self.model.return_mags({"filter": band, "airmass": airmass})["mag_sky"])
