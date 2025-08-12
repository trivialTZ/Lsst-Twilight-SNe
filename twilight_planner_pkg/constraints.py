"""Observation constraint utilities."""
from __future__ import annotations

from typing import Dict
import numpy as np

BASE_MIN_SEP: Dict[str, float] = {"g": 30.0, "r": 25.0, "i": 20.0, "z": 15.0, "y": 10.0}


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def moon_separation_factor(moon_alt_deg: float, moon_phase_frac: float):
    """Return a weight (0..1) for Moon separation requirements."""
    alt = np.asarray(moon_alt_deg, dtype=float)
    phase = np.asarray(moon_phase_frac, dtype=float)
    out = np.ones_like(alt, dtype=float)
    mask_low = alt <= -10.0
    out[mask_low] = 0.0
    mask_mid = (alt > -10.0) & (alt < 0.0)
    if np.any(mask_mid):
        alt_w = (alt[mask_mid] + 10.0) / 10.0
        phase_w = _clamp(0.3 + 0.7 * phase[mask_mid], 0.3, 1.0)
        out[mask_mid] = alt_w * phase_w
    return out if out.size > 1 else float(out)


def effective_min_sep(
    filt: str,
    moon_alt_deg: float,
    moon_phase_frac: float,
    base_min_sep: Dict[str, float] | None = None,
) -> float:
    base = (base_min_sep or BASE_MIN_SEP).get(filt, (base_min_sep or BASE_MIN_SEP).get("r", 25.0))
    return base * moon_separation_factor(moon_alt_deg, moon_phase_frac)
