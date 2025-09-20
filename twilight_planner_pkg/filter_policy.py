"""Filter selection policy using an m5/SNR model aligned with LSST practice."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, TYPE_CHECKING

from .sky_model import SkyModelConfig, SimpleSkyProvider, sky_mag_arcsec2

if TYPE_CHECKING:
    from .photom_rubin import PhotomConfig

MIN_SNR = 5.0

# Reference dark-sky m5 at X=1, seeing=0.7", t_vis=30s (approximate PSTN-054)
M5_REF = {"u": 23.87, "g": 24.64, "r": 24.21, "i": 23.79, "z": 23.18, "y": 22.37}

# Zenith dark sky (mag/arcsec^2) per band (SMTN-002)
DARK_SKY_MAG = {
    "u": 23.05,
    "g": 22.25,
    "r": 21.20,
    "i": 20.46,
    "z": 19.61,
    "y": 18.60,
}

# Read-noise correction asymptote ΔC_m^∞ (mag)
DELTA_CM_INF = {"u": 0.54, "g": 0.09, "r": 0.04, "i": 0.03, "z": 0.02, "y": 0.02}


def _m5_scale(
    band: str,
    *,
    m_sky_arcsec2: float,
    seeing_fwhm_arcsec: float,
    airmass: float,
    t_vis_s: float,
    k_m: Optional[float] = None,
) -> float:
    """Scaled m5 using LSST depth relations including read-noise correction."""
    base = M5_REF.get(band, 23.0)
    msky0 = DARK_SKY_MAG.get(band, 21.0)
    term_sky = 0.50 * (float(m_sky_arcsec2) - float(msky0))
    term_see = 2.5 * math.log10(max(1e-3, 0.7 / max(1e-3, seeing_fwhm_arcsec)))
    term_t = 1.25 * math.log10(max(1e-3, float(t_vis_s) / 30.0))
    term_air = -float(k_m or 0.0) * (float(airmass) - 1.0)
    delta_inf = DELTA_CM_INF.get(band, 0.0)
    if delta_inf > 0.0:
        # T_scale per SMTN-002 uses the band-specific dark-sky reference
        tau = max(1e-3, (float(t_vis_s) / 30.0) * 10 ** (-0.4 * (m_sky_arcsec2 - msky0)))
        term_delta = delta_inf - 1.25 * math.log10(
            1.0 + (10 ** (0.8 * delta_inf) - 1.0) / tau
        )
    else:
        term_delta = 0.0
    return float(base + term_sky + term_see + term_t + term_air + term_delta)


def heuristic_filters_from_sun_alt(sun_alt_deg: float) -> List[str]:
    """Utility ordering; not used as a feasibility fallback."""
    if sun_alt_deg > -12:
        return ["i", "r", "z", "y"]
    if sun_alt_deg > -15:
        return ["r", "i", "z", "y"]
    return ["g", "r", "i", "z", "y"]


def allowed_filters_for_window(
    target_mag_dict: Dict[str, float],
    sun_alt_deg: float,
    moon_alt_deg: float,
    moon_phase: float,
    moon_sep_deg: float,
    airmass: float,
    seeing_default: float,
    *,
    exposure_by_filter: Optional[Dict[str, float]] = None,
    phot_cfg: Optional["PhotomConfig"] = None,
    sky_provider=None,
    sky_cfg: Optional[SkyModelConfig] = None,
    mjd: Optional[float] = None,
    ra_deg: Optional[float] = None,
    dec_deg: Optional[float] = None,
) -> List[str]:
    """Select filters that are predicted to reach the target's magnitude.

    Parameters
    ----------
    target_mag_dict : dict of str to float
        Target magnitudes by filter. Missing filters fall back to the ``"r"``
        entry or ``21.5`` if absent.
    sun_alt_deg : float
        Altitude of the Sun in degrees.
    moon_alt_deg : float
        Altitude of the Moon in degrees.
    moon_phase : float
        Fractional lunar illumination in ``[0, 1]``.
    moon_sep_deg : float
        Separation between target and Moon in degrees.
    airmass : float
        Airmass along the line of sight.
    seeing : float
        Seeing FWHM in arcseconds.

    Returns
    -------
    list of str
        Filters deemed viable. When no band passes, an empty list is returned
        and the window policy applies separately.

    Notes
    -----
    This selector is a pruning step; final use is constrained by the
    user-configured ``sun_alt_policy`` at the window stage.
    """

    try:
        from .photom_rubin import PhotomConfig
    except Exception:  # pragma: no cover - fallback for unexpected import issues
        PhotomConfig = None  # type: ignore

    if phot_cfg is None and PhotomConfig is not None:
        phot_cfg = PhotomConfig()

    exp_map = exposure_by_filter or {}
    sky_cfg_local = sky_cfg or SkyModelConfig()
    allowed: List[str] = []
    snr_threshold = 2.5 * math.log10(max(1e-3, MIN_SNR / 5.0))

    use_provider = None
    if sky_provider is not None and not isinstance(sky_provider, SimpleSkyProvider):
        use_provider = sky_provider

    for filt in ["g", "r", "i", "z", "y"]:
        target_mag = target_mag_dict.get(filt, target_mag_dict.get("r", 21.5))
        t_vis = float(exp_map.get(filt, 30.0))
        if phot_cfg is not None and phot_cfg.fwhm_eff is not None:
            seeing_fwhm = float(phot_cfg.fwhm_eff.get(filt, seeing_default))
        else:
            seeing_fwhm = float(seeing_default)
        k_band = 0.0
        if phot_cfg is not None and phot_cfg.k_m is not None:
            k_band = float(phot_cfg.k_m.get(filt, 0.0))

        # Sky brightness per band
        if use_provider is not None:
            try:
                m_sky = float(
                    use_provider.sky_mag(mjd, ra_deg, dec_deg, filt, airmass)
                )
            except Exception:
                m_sky = sky_mag_arcsec2(
                    filt,
                    sky_cfg_local,
                    sun_alt_deg,
                    moon_alt_deg,
                    moon_phase,
                    moon_sep_deg,
                    airmass,
                    k_band=k_band,
                )
        else:
            m_sky = sky_mag_arcsec2(
                filt,
                sky_cfg_local,
                sun_alt_deg,
                moon_alt_deg,
                moon_phase,
                moon_sep_deg,
                airmass,
                k_band=k_band,
            )

        m5 = _m5_scale(
            filt,
            m_sky_arcsec2=m_sky,
            seeing_fwhm_arcsec=seeing_fwhm,
            airmass=airmass,
            t_vis_s=t_vis,
            k_m=k_band,
        )

        if float(m5) - float(target_mag) >= snr_threshold:
            allowed.append(filt)

    return allowed
