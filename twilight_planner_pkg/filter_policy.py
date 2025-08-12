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
    """Very small heuristic m5 estimator."""
    sun_term = max(0.0, sun_alt_deg + 12.0)
    factors = {"g": 0.35, "r": 0.30, "i": 0.15, "z": 0.10, "y": 0.05}
    sun_penalty = sun_term * factors.get(filt, 0.2)
    moon_penalty = 3.0 * moon_phase * max(0.0, 1.0 - moon_sep_deg / 90.0) * moon_separation_factor(
        moon_alt_deg, moon_phase
    )
    return BASE_M5.get(filt, 23.0) - sun_penalty - moon_penalty


def heuristic_filters_from_sun_alt(sun_alt_deg: float) -> List[str]:
    if sun_alt_deg > -8:
        return ["i", "z", "y"]
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
    seeing: float,
) -> List[str]:
    allowed: List[str] = []
    for filt in ["g", "r", "i", "z", "y"]:
        m5 = predict_m5(filt, sun_alt_deg, moon_alt_deg, moon_phase, moon_sep_deg, airmass, seeing)
        target_mag = target_mag_dict.get(filt, target_mag_dict.get("r", 21.5))
        if m5 - target_mag >= MARGIN_MAG:
            allowed.append(filt)
    if not allowed:
        allowed = heuristic_filters_from_sun_alt(sun_alt_deg)
    return allowed
