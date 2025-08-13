from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .astro_utils import airmass_from_alt_deg


@dataclass
class PhotomConfig:
    """Configuration parameters for Rubin photometric calculations.

    Attributes
    ----------
    npe_pixel_saturate: int
        Hard pixel full-well capacity in electrons where exposure must be
        shortened to avoid saturation (~100 ke-).
    npe_pixel_warn_nonlinear: int
        Non-linearity warning threshold in electrons; exposures predicting a
        per-pixel charge above this level (~80 ke-) will be flagged but not
        forcibly reduced unless the hard cap is exceeded.
    read_noise_e: float
        Read noise per pixel in electrons (typical 5.4–6.2 e⁻ depending on
        sensor vendor; requirement ≤9 e⁻ per LCA-48-J).
    gain_e_per_adu: float
        Gain in electrons per ADU (measured ≈1.5–1.7 e⁻/ADU; using 1 is
        acceptable for SNR/m₅ per SMTN-002).
    """

    pixel_scale_arcsec: float = 0.2
    zpt1s: Dict[str, float] | None = None
    k_m: Dict[str, float] | None = None
    fwhm_eff: Dict[str, float] | None = None
    read_noise_e: float = 6.0
    gain_e_per_adu: float = 1.0
    zpt_err_mag: float = 0.005
    npe_pixel_saturate: int = 100_000
    npe_pixel_warn_nonlinear: int = 80_000
    sky_mag_override: Optional[float] = None

    def __post_init__(self):
        if self.zpt1s is None:
            self.zpt1s = {
                "u": 26.52,
                "g": 28.51,
                "r": 28.36,
                "i": 28.17,
                "z": 27.78,
                "y": 26.82,
            }
        if self.k_m is None:
            self.k_m = {
                "u": 0.50,
                "g": 0.17,
                "r": 0.10,
                "i": 0.07,
                "z": 0.06,
                "y": 0.06,
            }
        if self.fwhm_eff is None:
            self.fwhm_eff = {
                "u": 0.90,
                "g": 0.85,
                "r": 0.83,
                "i": 0.83,
                "z": 0.85,
                "y": 0.90,
            }


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


def epoch_zeropoints(
    zpt1s_pe: float, t_exp_s: float, k_m: float, X: float, gain: float
) -> tuple[float, float]:
    """Return zeropoints in electrons and ADU."""
    ZPT_pe = zpt1s_pe + 2.5 * math.log10(max(1e-3, t_exp_s)) - k_m * (X - 1.0)
    ZPTAVG = ZPT_pe - 2.5 * math.log10(max(1e-6, gain))
    return ZPT_pe, ZPTAVG


def sky_rms_adu_per_pix(
    m_sky_arcsec2: float, ZPT_pe: float, pixel_scale_arcsec: float, gain: float
) -> float:
    """Sky-background RMS in ADU per pixel."""

    area = pixel_scale_arcsec**2
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
    ZPT_pe, ZPTAVG = epoch_zeropoints(
        cfg.zpt1s[band], t_exp_s, cfg.k_m[band], X, cfg.gain_e_per_adu
    )
    fwhm = fwhm_eff_arcsec or cfg.fwhm_eff[band]
    nea = nea_pixels(fwhm, cfg.pixel_scale_arcsec)
    SKYSIG = sky_rms_adu_per_pix(
        sky_mag_arcsec2, ZPT_pe, cfg.pixel_scale_arcsec, cfg.gain_e_per_adu
    )
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
) -> tuple[float, set[str]]:
    """Return a safe exposure time capped to avoid source or sky saturation.

    Parameters
    ----------
    band: str
        Photometric band ("g", "r", ...).
    t_exp_s: float
        Proposed exposure time in seconds.
    alt_deg: float
        Altitude of the target in degrees.
    src_mag: float
        Apparent magnitude of the source.
    sky_mag_arcsec2: float
        Sky surface brightness in mag/arcsec^2.
    cfg: PhotomConfig
        Photometric configuration parameters.
    fwhm_eff_arcsec: float, optional
        Effective seeing FWHM in arcsec. Defaults to config value.
    min_exp_s: float, default 1.0
        Minimum allowable exposure time in seconds.

    Returns
    -------
    tuple
        ``(t, flags)`` where ``t`` is the exposure time after applying the
        saturation policy and ``flags`` is a set of strings.  ``"sat_guard"``
        indicates the exposure was shortened to respect the hard cap
        (``cfg.npe_pixel_saturate`` ≈ 100 ke-). ``"warn_nonlinear"`` marks
        exposures that fall in the non-linear 80–100 ke- range but do not
        require shortening.
    """
    flags: set[str] = set()
    t = max(min_exp_s, t_exp_s)
    max_e = 0.0
    for _ in range(10):
        eph = compute_epoch_photom(
            band, t, alt_deg, sky_mag_arcsec2, cfg, fwhm_eff_arcsec
        )
        frac = central_pixel_fraction_gaussian(
            fwhm_eff_arcsec or cfg.fwhm_eff[band], cfg.pixel_scale_arcsec
        )
        ce = central_pixel_electrons(src_mag, eph.ZPT_pe, frac)
        se = (eph.SKYSIG * eph.GAIN) ** 2  # sky electrons per pixel
        max_e = max(ce, se)
        if max_e <= cfg.npe_pixel_saturate:
            if max_e >= cfg.npe_pixel_warn_nonlinear:
                flags.add("warn_nonlinear")
            break
        flags.add("sat_guard")
        scale_src = (
            cfg.npe_pixel_saturate / max(1.0, ce)
            if ce > cfg.npe_pixel_saturate
            else 1.0
        )
        scale_sky = (
            cfg.npe_pixel_saturate / max(1.0, se)
            if se > cfg.npe_pixel_saturate
            else 1.0
        )
        scale = max(0.1, min(scale_src, scale_sky))
        t = max(min_exp_s, t * scale)
    else:
        if max_e >= cfg.npe_pixel_warn_nonlinear:
            flags.add("warn_nonlinear")
    return t, flags
