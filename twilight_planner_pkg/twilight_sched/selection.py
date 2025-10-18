"""Selection and policy helpers used by candidate preparation and scoring.

Responsibility
--------------
- Filter policy at mid-window sun altitude
- Low-z Ia identification helper
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from astropy.coordinates import AltAz, get_sun, EarthLocation
from astropy.time import Time

from ..astro_utils import allowed_filters_for_sun_alt
from ..config import PlannerConfig


def _policy_filters_mid(sun_alt_deg: float, cfg: PlannerConfig) -> list[str]:
    """Allowed filters at ``sun_alt_deg`` according to policy."""
    if not getattr(cfg, "sun_alt_policy", None):
        return list(cfg.filters)
    return allowed_filters_for_sun_alt(sun_alt_deg, cfg)


def _is_low_z_ia(sn_type, redshift, cfg: PlannerConfig) -> bool:
    """Return True if the target is Ia-like and below the configured z threshold.

    Ia-like detection is based on cfg.low_z_ia_markers: for the special token
    'ia' we do a substring match (e.g., 'Ia-91bg' counts), while other markers
    are compared by exact, case-insensitive match (e.g., '1', '101').
    """
    try:
        z = float(redshift) if redshift is not None else math.nan
    except Exception:
        z = math.nan
    z_thresh = getattr(cfg, "low_z_ia_z_threshold", PlannerConfig.low_z_ia_z_threshold)
    try:
        z_thresh_val = float(z_thresh)
    except Exception:
        z_thresh_val = math.nan
    if not (np.isfinite(z_thresh_val) and z < z_thresh_val):
        return False
    if sn_type is None:
        return False
    t = str(sn_type).strip().lower()
    if not t or t in {"nan", "none"}:
        return False
    markers = [
        str(m).strip().lower()
        for m in getattr(cfg, "low_z_ia_markers", ["ia"]) or ["ia"]
    ]
    for m in markers:
        if m == "ia":
            if "ia" in t:
                return True
        else:
            if t == m:
                return True
    return False

# -----------------------------
# Sun altitude minute-bin cache
# -----------------------------
_SUN_ALT_CACHE: dict[tuple[int, int], float] = {}


def _minute_bin(mjd: float, step_min: int) -> int:
    """Quantize MJD to an integer minute bucket at the given step size."""
    return int(math.floor(float(mjd) * 1440.0 / max(1, int(step_min))))


def get_sun_alt_deg_cached(
    ts_utc: "pd.Timestamp | datetime",
    site: EarthLocation,
    *,
    step_minutes: int = 1,
) -> float:
    """Return Sun altitude (deg) cached by minute-bucket.

    The cache key only depends on time (and the step size); the location is
    assumed constant across planner runs.
    """
    import pandas as pd

    if isinstance(ts_utc, pd.Timestamp):
        ts = ts_utc.tz_convert("UTC") if ts_utc.tzinfo else ts_utc.tz_localize("UTC")
    else:
        ts = pd.Timestamp(ts_utc).tz_localize("UTC")
    mjd = Time(ts).mjd
    key = (_minute_bin(mjd, step_minutes), int(step_minutes))
    cached = _SUN_ALT_CACHE.get(key)
    if cached is not None:
        return cached
    alt = float(
        get_sun(Time(ts)).transform_to(AltAz(obstime=Time(ts), location=site)).alt.deg
    )
    _SUN_ALT_CACHE[key] = alt
    return alt
