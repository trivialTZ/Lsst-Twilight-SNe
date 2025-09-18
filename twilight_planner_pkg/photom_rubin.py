from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

from .astro_utils import airmass_from_alt_deg

# mypy: ignore-errors


@dataclass
class PhotomConfig:
    """Configuration parameters for Rubin photometric calculations.

    Attributes
    ----------
    npe_pixel_saturate: int
        Hard pixel full-well capacity in electrons where exposure must be
        shortened to avoid saturation (~80 ke⁻ by default).
    npe_pixel_warn_nonlinear: int
        Non-linearity warning threshold in electrons; exposures predicting a
        per-pixel charge above this level (~0.8×full well) will be flagged but
        not forcibly reduced unless the hard cap is exceeded.
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
    gain_e_per_adu: float = 1.6
    zpt_err_mag: float = 0.005
    # Pixel full-well thresholds (electrons)
    npe_pixel_saturate: int = 80_000
    npe_pixel_warn_nonlinear: int = 64_000
    # Optional extra safety factor for non-linearity region (1.0 = disabled)
    nonlinear_headroom: float = 1.0
    sky_mag_override: Optional[float] = None
    # Toggle which central-pixel fraction approximation to use for saturation
    # guards. ``True`` selects the SNANA Taylor expansion (recommended for SIMLIB
    # parity); ``False`` keeps the exact Gaussian integral used historically.
    use_snana_fA: bool = True

    def __post_init__(self) -> None:
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
        # Soft configuration sanity checks: warn when values look unexpected.
        if abs(float(self.pixel_scale_arcsec) - 0.2) > 1e-3:
            warnings.warn(
                (
                    f"PhotomConfig.pixel_scale_arcsec={self.pixel_scale_arcsec} "
                    "(Rubin design is ~0.2 arcsec/pix); verify the input."
                ),
                UserWarning,
            )
        if not (1.4 <= float(self.gain_e_per_adu) <= 1.8):
            warnings.warn(
                (
                    f"PhotomConfig.gain_e_per_adu={self.gain_e_per_adu} "
                    "(expected within ~1.5-1.7 e/ADU); saturation diagnostics may be off."
                ),
                UserWarning,
            )


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
    FWHM_eff_arcsec: float
    PSF1_pix: float
    PSF2_pix: float
    PSFRATIO: float


def nea_pixels(fwhm_eff_arcsec: float, pixel_scale_arcsec: float) -> float:
    """Noise-equivalent area in pixels for a Gaussian PSF."""

    return 2.266 * (fwhm_eff_arcsec / pixel_scale_arcsec) ** 2


def central_pixel_fraction_exact(fwhm_arcsec: float, pix_arcsec: float) -> float:
    """Exact central-pixel fraction for a Gaussian PSF via erf^2 integral."""

    sigma = max(1e-6, fwhm_arcsec / 2.355)
    x = pix_arcsec / (2.0 * math.sqrt(2.0) * sigma)
    erf = math.erf(x)
    return max(1e-6, min(1.0, erf * erf))


def central_pixel_fraction_snana(
    fwhm_arcsec: float, pixel_scale_arcsec: float
) -> float:
    """SNANA 4.7.4 Taylor-approximation to the central pixel fraction ``f_A``."""

    sigma = max(1e-6, fwhm_arcsec / 2.355)
    P = float(pixel_scale_arcsec)
    term = (P * P) / (2.0 * math.pi * sigma * sigma)
    frac = term * (1.0 - (P * P) / (4.0 * math.pi * sigma * sigma))
    return max(0.0, min(1.0, frac))


# Backwards-compatibility alias (deprecated name used in older notebooks/tests).
central_pixel_fraction_gaussian = central_pixel_fraction_exact


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


def electrons_per_pixel_from_sb(
    mu_arcsec2: float, ZPT_pe: float, pixel_scale_arcsec: float
) -> float:
    """Electrons per pixel from a surface-brightness value in mag/arcsec^2.

    This mirrors the sky calculation but returns electrons directly rather than
    an RMS in ADU. ``ZPT_pe`` is the electron zeropoint for the epoch.
    """
    area = pixel_scale_arcsec**2
    return max(0.0, 10 ** (-0.4 * (mu_arcsec2 - ZPT_pe)) * area)


def observed_sb_from_rest(
    mu_rest_arcsec2: float, z: float, K_mag: float = 0.0
) -> float:
    """Convert rest-frame surface brightness to observed-frame.

    Applies Tolman dimming by (1+z)^4 and an optional K-correction ``K_mag``
    (mag). Returns the observed-frame surface brightness in mag/arcsec^2.
    """
    # 2.5 * log10( (1+z)^4 ) = 10 * log10(1+z)
    return float(mu_rest_arcsec2) + 10.0 * math.log10(1.0 + float(z)) + float(K_mag)


def compute_epoch_photom(
    band: str,
    t_exp_s: float,
    alt_deg: float,
    sky_mag_arcsec2: float,
    cfg: PhotomConfig,
    fwhm_eff_arcsec: float | None = None,
) -> EpochPhotom:
    """Compute photometric parameters for a single exposure.

    The returned ``SKYSIG`` and ``RDNOISE`` fields are emitted as ADU/pixel for
    SIMLIB rows: ``SKYSIG`` corresponds to ``sqrt(sky electrons) / gain`` (read
    noise excluded) while ``NOISE`` in the SIMLIB output is ``RDNOISE / gain``.
    """

    X = airmass_from_alt_deg(alt_deg)
    ZPT_pe, ZPTAVG = epoch_zeropoints(
        cfg.zpt1s[band], t_exp_s, cfg.k_m[band], X, cfg.gain_e_per_adu
    )
    fwhm = fwhm_eff_arcsec or cfg.fwhm_eff[band]
    nea = nea_pixels(fwhm, cfg.pixel_scale_arcsec)
    sigma_arcsec = fwhm / 2.355
    sigma_pix = sigma_arcsec / max(1e-6, cfg.pixel_scale_arcsec)
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
        FWHM_eff_arcsec=fwhm,
        PSF1_pix=sigma_pix,
        PSF2_pix=0.0,
        PSFRATIO=0.0,
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
    *,
    nexpose: int = 1,
    # Optional host terms; supply either observed SB or (rest SB + z)
    mu_host_obs_arcsec2: float | None = None,
    mu_host_rest_arcsec2: float | None = None,
    z_host: float | None = None,
    K_host: float = 0.0,
    # Optional compact host knot (treated as a point-like component)
    host_point_mag: float | None = None,
    host_point_frac: float | None = None,
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

    nexpose: int, default 1
        Number of equal-length exposures coadded into the epoch. SNANA compares
        the *average* per-exposure pixel charge against the saturation
        threshold. The guard therefore divides the summed electrons by this
        value when checking against ``cfg.npe_pixel_saturate``.

    Returns
    -------
    tuple
        ``(t, flags)`` where ``t`` is the exposure time after applying the
        saturation policy and ``flags`` is a set of strings. ``"sat_guard"``
        indicates the exposure was shortened to respect the hard cap
        (``cfg.npe_pixel_saturate`` ≈ 100 ke-). ``"warn_nonlinear"`` marks
        exposures that fall in the non-linear 80–100 ke- range but do not
        require shortening.
    """
    flags: set[str] = set()
    t = max(min_exp_s, t_exp_s)
    last_total_e_avg = 0.0
    n_expose = max(1, int(round(nexpose)))
    for _ in range(10):
        eph = compute_epoch_photom(
            band, t, alt_deg, sky_mag_arcsec2, cfg, fwhm_eff_arcsec
        )

        # Central-pixel electrons from the SN (point source)
        seeing = fwhm_eff_arcsec or cfg.fwhm_eff[band]
        frac_exact = central_pixel_fraction_exact(seeing, cfg.pixel_scale_arcsec)
        frac_snana = central_pixel_fraction_snana(seeing, cfg.pixel_scale_arcsec)
        frac = frac_snana if cfg.use_snana_fA else frac_exact
        e_src = central_pixel_electrons(src_mag, eph.ZPT_pe, frac)

        # Per-pixel electrons from the sky (use direct electrons, not RMS)
        e_sky = electrons_per_pixel_from_sb(
            sky_mag_arcsec2, eph.ZPT_pe, cfg.pixel_scale_arcsec
        )

        # Per-pixel electrons from the host (observed SB or rest SB + z)
        e_host = 0.0
        if mu_host_obs_arcsec2 is not None:
            e_host += electrons_per_pixel_from_sb(
                mu_host_obs_arcsec2, eph.ZPT_pe, cfg.pixel_scale_arcsec
            )
        elif mu_host_rest_arcsec2 is not None and z_host is not None:
            mu_obs = observed_sb_from_rest(mu_host_rest_arcsec2, z_host, K_host)
            e_host += electrons_per_pixel_from_sb(
                mu_obs, eph.ZPT_pe, cfg.pixel_scale_arcsec
            )

        # Optional compact host knot approximated as a point-like component
        if host_point_mag is not None:
            default_frac = frac_snana if cfg.use_snana_fA else frac_exact
            fpt = host_point_frac if host_point_frac is not None else default_frac
            e_host += central_pixel_electrons(host_point_mag, eph.ZPT_pe, fpt)

        total_e = e_src + e_host + e_sky
        total_e_avg = total_e / float(n_expose)
        last_total_e_avg = total_e_avg

        # Threshold checks (include optional headroom for the warn zone)
        warn_thr = cfg.npe_pixel_warn_nonlinear * max(1.0, cfg.nonlinear_headroom)
        if total_e_avg <= cfg.npe_pixel_saturate:
            if total_e_avg >= warn_thr:
                flags.add("warn_nonlinear")
            break

        # Over the hard cap: shorten exposure proportionally (linear scaling)
        flags.add("sat_guard")
        scale = cfg.npe_pixel_saturate / max(1.0, total_e_avg)
        scale = max(0.1, min(1.0, scale))
        t = max(min_exp_s, t * scale)
    else:
        if last_total_e_avg >= cfg.npe_pixel_warn_nonlinear:
            flags.add("warn_nonlinear")
    return t, flags
