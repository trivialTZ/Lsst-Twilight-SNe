"""Filter selection policy driven by a simple m5/SNR model."""

from __future__ import annotations

from typing import Dict, List

from .constraints import moon_separation_factor

BASE_M5 = {"g": 24.2, "r": 23.7, "i": 23.3, "z": 22.8, "y": 22.2}
MIN_SNR = 5.0
MARGIN_MAG = 0.3


def predict_m5(
    filt: str,
    sun_alt_deg: float,
    moon_alt_deg: float,
    moon_phase: float,
    moon_sep_deg: float,
    airmass: float,
    seeing: float,
) -> float:
    """Estimate the 5σ limiting magnitude for a single exposure.

    Parameters
    ----------
    filt : str
        Photometric filter name.
    sun_alt_deg : float
        Altitude of the Sun in degrees (negative below horizon).
    moon_alt_deg : float
        Altitude of the Moon in degrees.
    moon_phase : float
        Fractional lunar illumination in ``[0, 1]``.
    moon_sep_deg : float
        Angular separation between target and Moon in degrees.
    airmass : float
        Airmass along the line of sight.
    seeing : float
        Full width at half maximum of the point-spread function in arcseconds.

    Returns
    -------
    float
        Heuristic estimate of the 5σ depth ``m5`` in magnitudes.

    Notes
    -----
    This calculator is intentionally lightweight and is not a substitute for a
    full Rubin Observatory sky brightness model. It merely captures relative
    trends with Sun and Moon conditions.
    """
    sun_term = max(0.0, sun_alt_deg + 12.0)
    factors = {"g": 0.35, "r": 0.30, "i": 0.15, "z": 0.10, "y": 0.05}
    sun_penalty = sun_term * factors.get(filt, 0.2)
    moon_penalty = (
        3.0
        * moon_phase
        * max(0.0, 1.0 - moon_sep_deg / 90.0)
        * moon_separation_factor(moon_alt_deg, moon_phase)
    )
    return BASE_M5.get(filt, 23.0) - sun_penalty - moon_penalty


def heuristic_filters_from_sun_alt(sun_alt_deg: float) -> List[str]:
    """Fallback filter ordering based solely on Sun altitude.

    Parameters
    ----------
    sun_alt_deg : float
        Altitude of the Sun in degrees.

    Returns
    -------
    list of str
        Filters sorted from most to least preferred under the heuristic
        assumption that redder filters cope better with twilight brightness.
    """
    # Redder filters cope better with bright twilight; allow g only when very dark.
    if sun_alt_deg > -12:
        return ["i", "r", "z", "y"]
    if sun_alt_deg > -15:
        return ["r", "i", "z", "y"]  # g generally too bright here unless m5 gate passes
    return ["g", "r", "i", "z", "y"]


def allowed_filters_for_window(
    target_mag_dict: Dict[str, float],
    sun_alt_deg: float,
    moon_alt_deg: float,
    moon_phase: float,
    moon_sep_deg: float,
    airmass: float,
    seeing: float,
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
        Filters deemed viable. When the m5 model rejects all filters, a
        heuristic set based on Sun altitude is returned as a fallback.

    Notes
    -----
    The selection is heuristic and meant for quick pruning; it does not
    guarantee the requested signal-to-noise ratio in practice.
    """
    allowed: List[str] = []
    for filt in ["g", "r", "i", "z", "y"]:
        m5 = predict_m5(
            filt, sun_alt_deg, moon_alt_deg, moon_phase, moon_sep_deg, airmass, seeing
        )
        target_mag = target_mag_dict.get(filt, target_mag_dict.get("r", 21.5))
        margin = MARGIN_MAG
        if filt == "g":
            if sun_alt_deg > -12:
                continue  # never allow g here
            if sun_alt_deg > -15:
                margin += 0.3  # require extra headroom for g in brighter twilight
        if m5 - target_mag >= margin:
            allowed.append(filt)
    if not allowed:
        allowed = heuristic_filters_from_sun_alt(sun_alt_deg)
    return allowed
