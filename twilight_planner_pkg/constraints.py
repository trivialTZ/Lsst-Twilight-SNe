"""Observation constraint utilities."""
from __future__ import annotations

from typing import Dict
import numpy as np

BASE_MIN_SEP: Dict[str, float] = {"g": 30.0, "r": 25.0, "i": 20.0, "z": 15.0, "y": 10.0}


def _clamp(val: float, lo: float, hi: float) -> float:
    """Clamp a value into the interval ``[lo, hi]``.

    Parameters
    ----------
    val : float
        Value to constrain.
    lo : float
        Lower bound of the interval.
    hi : float
        Upper bound of the interval.

    Returns
    -------
    float
        ``val`` limited to the inclusive range ``[lo, hi]``.

    Notes
    -----
    This helper performs no type checking and is intended for internal use.
    """
    return max(lo, min(hi, val))


def moon_separation_factor(moon_alt_deg: float, moon_phase_frac: float):
    """Weight Moon separation requirements by altitude and phase.

    Parameters
    ----------
    moon_alt_deg : float or array-like
        Altitude of the Moon in degrees. Negative values correspond to the
        Moon below the horizon. Arrays are accepted and will be broadcast with
        ``moon_phase_frac`` using :func:`numpy.asarray`.
    moon_phase_frac : float or array-like
        Fractional illumination of the Moon in ``[0, 1]`` where ``0`` is new
        Moon and ``1`` is full Moon. Must be broadcastable to the shape of
        ``moon_alt_deg``.

    Returns
    -------
    float or :class:`numpy.ndarray`
        Multiplicative weight between ``0`` and ``1``. The returned object has
        the same shape as the broadcast inputs; a scalar is returned when the
        inputs are scalars.

    Notes
    -----
    For altitudes below ``-10``° the requirement is fully relaxed. Between
    ``-10`` and ``0``° the weight varies linearly with altitude and is further
    modulated by the illuminated fraction of the Moon. Above the horizon the
    weight is ``1``.
    """
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
    """Scaled minimum Moon separation for a given filter.

    Parameters
    ----------
    filt : str
        Photometric filter name (e.g., ``"r"``).
    moon_alt_deg : float
        Altitude of the Moon in degrees.
    moon_phase_frac : float
        Moon illumination fraction in the range ``[0, 1]``.
    base_min_sep : dict of str to float, optional
        Mapping of filters to baseline separation requirements in degrees.
        When ``None`` (default) :data:`BASE_MIN_SEP` is used.

    Returns
    -------
    float
        Required separation in degrees after applying the altitude/phase
        weighting.

    Notes
    -----
    Filters not present in ``base_min_sep`` fall back to the ``"r"`` value.
    """
    base = (base_min_sep or BASE_MIN_SEP).get(
        filt, (base_min_sep or BASE_MIN_SEP).get("r", 25.0)
    )
    return base * moon_separation_factor(moon_alt_deg, moon_phase_frac)
