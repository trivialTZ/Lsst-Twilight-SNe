from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math

from .astro_utils import airmass_from_alt_deg


@dataclass
class PhotomConfig:
    """Configuration parameters for Rubin photometric calculations."""

    pixel_scale_arcsec: float = 0.2
    zpt1s: Dict[str, float] | None = None
    k_m: Dict[str, float] | None = None
    fwhm_eff: Dict[str, float] | None = None
    read_noise_e: float = 9.0
    gain_e_per_adu: float = 1.0
    zpt_err_mag: float = 0.005
    npe_pixel_saturate: int = 90000
    sky_mag_override: Optional[float] = None

    def __post_init__(self):
        if self.zpt1s is None:
            self.zpt1s = {"u":26.52,"g":28.51,"r":28.36,"i":28.17,"z":27.78,"y":26.82}
        if self.k_m is None:
            self.k_m = {"u":0.50,"g":0.17,"r":0.10,"i":0.07,"z":0.06,"y":0.06}
        if self.fwhm_eff is None:
            self.fwhm_eff = {"u":0.90,"g":0.85,"r":0.83,"i":0.83,"z":0.85,"y":0.90}


@dataclass
class EpochPhotom:
    """Photometric scalars for a single exposure."""

    ZPTAVG: float
    ZPT_pe: float
    ZPTERR: float
    SKYSIG: float
    NEA_pix: float
    RDNOISE: float
    GAIN: float


def nea_pixels(fwhm_eff_arcsec: float, pixel_scale_arcsec: float) -> float:
    """Noise-equivalent area in pixels for a Gaussian PSF."""

    return 2.266 * (fwhm_eff_arcsec / pixel_scale_arcsec) ** 2


def central_pixel_fraction_gaussian(fwhm_arcsec: float, pix_arcsec: float) -> float:
    """Fraction of flux landing in the central pixel for a Gaussian PSF."""
    sigma = fwhm_arcsec / 2.355
    x = pix_arcsec / (2.0 * math.sqrt(2.0) * sigma)
    erf = math.erf(x)
    return max(1e-6, min(1.0, erf * erf))


def epoch_zeropoints(zpt1s_pe: float, t_exp_s: float, k_m: float, X: float, gain: float) -> tuple[float, float]:
    """Return zeropoints in electrons and ADU."""
    ZPT_pe = zpt1s_pe + 2.5 * math.log10(max(1e-3, t_exp_s)) - k_m * (X - 1.0)
    ZPTAVG = ZPT_pe - 2.5 * math.log10(max(1e-6, gain))
    return ZPT_pe, ZPTAVG


def sky_rms_adu_per_pix(m_sky_arcsec2: float, ZPT_pe: float, pixel_scale_arcsec: float, gain: float) -> float:
    """Sky-background RMS in ADU per pixel."""

    area = pixel_scale_arcsec ** 2
    counts_e = 10 ** (-0.4 * (m_sky_arcsec2 - ZPT_pe)) * area
    return math.sqrt(max(0.0, counts_e)) / gain


def central_pixel_electrons(m_source: float, ZPT_pe: float, frac: float) -> float:
    """Electrons landing in the brightest pixel of a point source."""

    total_e = 10 ** (-0.4 * (m_source - ZPT_pe))
    return frac * total_e


def compute_epoch_photom(
    band: str,
    t_exp_s: float,
    alt_deg: float,
    sky_mag_arcsec2: float,
    cfg: PhotomConfig,
    fwhm_eff_arcsec: float | None = None,
) -> EpochPhotom:
    """Compute photometric parameters for a single exposure."""

    X = airmass_from_alt_deg(alt_deg)
    ZPT_pe, ZPTAVG = epoch_zeropoints(cfg.zpt1s[band], t_exp_s, cfg.k_m[band], X, cfg.gain_e_per_adu)
    fwhm = fwhm_eff_arcsec or cfg.fwhm_eff[band]
    nea = nea_pixels(fwhm, cfg.pixel_scale_arcsec)
    SKYSIG = sky_rms_adu_per_pix(sky_mag_arcsec2, ZPT_pe, cfg.pixel_scale_arcsec, cfg.gain_e_per_adu)
    RDNOISE = cfg.read_noise_e / cfg.gain_e_per_adu
    return EpochPhotom(
        ZPTAVG=ZPTAVG,
        ZPT_pe=ZPT_pe,
        ZPTERR=cfg.zpt_err_mag,
        SKYSIG=SKYSIG,
        NEA_pix=nea,
        RDNOISE=RDNOISE,
        GAIN=cfg.gain_e_per_adu,
    )


def cap_exposure_for_saturation(
    band: str,
    t_exp_s: float,
    alt_deg: float,
    src_mag: float,
    sky_mag_arcsec2: float,
    cfg: PhotomConfig,
    fwhm_eff_arcsec: float | None = None,
    min_exp_s: float = 1.0,
) -> float:
    """Return t_exp_s if safe, else a scaled exposure avoiding saturation."""
    t = max(min_exp_s, t_exp_s)
    for _ in range(3):
        eph = compute_epoch_photom(band, t, alt_deg, sky_mag_arcsec2, cfg, fwhm_eff_arcsec)
        frac = central_pixel_fraction_gaussian(fwhm_eff_arcsec or cfg.fwhm_eff[band], cfg.pixel_scale_arcsec)
        ce = central_pixel_electrons(src_mag, eph.ZPT_pe, frac)
        if ce <= cfg.npe_pixel_saturate:
            break
        scale = max(0.1, cfg.npe_pixel_saturate / max(1.0, ce))
        t = max(min_exp_s, t * scale)
    return t
