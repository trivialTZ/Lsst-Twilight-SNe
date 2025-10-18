"""Cost and timing helpers for in-window planning and selection.

Responsibility
--------------
- Visit unit time calculation (exposure + guard + slew/readout).
- Residual slew guard cost beyond minimum inter-exposure gap.
- Opportunity cost in seconds from observing now vs. best time in window.

Hot path
--------
- ``_visit_unit_time``
- ``_opportunity_cost_seconds``
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from ..astro_utils import great_circle_sep_deg, slew_time_seconds
from ..config import PlannerConfig
from ..photom_rubin import PhotomConfig
from ..sky_model import SkyModelConfig
from .caching import _cached_m5_at_time


def _visit_unit_time(
    exp_s: float,
    prev_coord: Optional[Tuple[float, float]],
    target_coord: Optional[Tuple[float, float]],
    cfg: PlannerConfig,
) -> float:
    """Return guard-aware single-visit duration in seconds."""

    if prev_coord and target_coord:
        sep = great_circle_sep_deg(*prev_coord, *target_coord)
    else:
        sep = 0.0
    slew = slew_time_seconds(
        float(sep),
        small_deg=cfg.slew_small_deg,
        small_time=cfg.slew_small_time_s,
        rate_deg_per_s=cfg.slew_rate_deg_per_s,
        settle_s=cfg.slew_settle_s,
    )
    guard = max(cfg.inter_exposure_min_s, cfg.readout_s + slew)
    return float(guard + exp_s)


def _residual_slew_cost(
    prev_coord: Optional[Tuple[float, float]],
    target_coord: Optional[Tuple[float, float]],
    cfg: PlannerConfig,
) -> float:
    """Return additional guard time required beyond the minimum for the slew."""

    if not prev_coord or not target_coord:
        return 0.0
    sep = great_circle_sep_deg(*prev_coord, *target_coord)
    slew = slew_time_seconds(
        float(sep),
        small_deg=cfg.slew_small_deg,
        small_time=cfg.slew_small_time_s,
        rate_deg_per_s=cfg.slew_rate_deg_per_s,
        settle_s=cfg.slew_settle_s,
    )
    return float(max(0.0, cfg.readout_s + slew - cfg.inter_exposure_min_s))


def _opportunity_cost_seconds(
    t: dict,
    filt: str,
    est_visit_s: float,
    now_mjd: float,
    cfg: PlannerConfig,
    phot_cfg: PhotomConfig,
    sky_provider,
    sky_cfg: SkyModelConfig,
    site: EarthLocation,
) -> float:
    """Extra seconds required if observing now vs best-in-window.

    Uses Δm5 between the target's best time and now. Caps extreme values via
    cfg.inbatch_cost_cap_s when present (default 600s).
    """

    def _m5_at_time(mjd: float | None) -> float | None:
        if mjd is None:
            return None
        try:
            mjd_val = float(mjd)
        except Exception:
            return None
        # Consider this a "best" tag if now matches best_mjd to ~0.04s
        best_mjd_val = t.get("best_time_mjd")
        try:
            best_mjd_f = float(best_mjd_val) if best_mjd_val is not None else None
        except Exception:
            best_mjd_f = None
        tag_best = bool(best_mjd_f is not None and abs(best_mjd_f - mjd_val) < 5e-7)
        return _cached_m5_at_time(
            target=t,
            filt=filt,
            cfg=cfg,
            phot_cfg=phot_cfg,
            sky_provider=sky_provider,
            sky_cfg=sky_cfg,
            mjd=mjd_val,
            minutes=1,
            site=site,
            tag_best=tag_best,
        )

    m5_now = _m5_at_time(now_mjd)
    best_ts = t.get("best_time_utc")
    if best_ts is None:
        best_mjd = None
    else:
        try:
            import pandas as pd

            if isinstance(best_ts, pd.Timestamp):
                best_mjd = Time(best_ts.tz_convert("UTC")).mjd
            else:
                best_mjd = Time(pd.Timestamp(best_ts).tz_localize("UTC")).mjd
        except Exception:
            best_mjd = None
    m5_best = _m5_at_time(best_mjd) if best_mjd is not None else m5_now
    if m5_now is None or m5_best is None:
        return 0.0
    delta_m = max(0.0, float(m5_best) - float(m5_now))
    opp = float(est_visit_s) * (10 ** (0.8 * delta_m) - 1.0)
    cap = float(getattr(cfg, "inbatch_cost_cap_s", 600.0))
    return float(min(max(0.0, opp), cap))


# --- Fast gate: is there enough remaining window time for any single visit? ---

def min_exposure_in_palette(cfg: PlannerConfig, palette: Optional[Iterable[str]] = None) -> float:
    """Return the minimum configured exposure time (seconds) among the given palette.

    Falls back to the global minimum across cfg.exposure_by_filter if palette is None.
    If a filter in the palette is not in exposure_by_filter, it is ignored.
    If nothing matches, returns +inf.
    """
    if palette is None:
        vals = list((cfg.exposure_by_filter or {}).values())
        return float(min(vals)) if vals else float("inf")
    exp_map = cfg.exposure_by_filter or {}
    vals = [float(exp_map[f]) for f in palette if f in exp_map]
    return float(min(vals)) if vals else float("inf")


def has_room_for_any_visit(
    window_sum: float,
    cap_s: float,
    cfg: PlannerConfig,
    palette: Optional[Iterable[str]] = None,
    *,
    include_readout_floor: bool = False,
) -> bool:
    """Cheap pre-check before attempting backfill/repeats.

    We skip building candidate batches when the leftover window time is not
    strictly greater than (inter_exposure_min_s + min exposure in palette).
    This mirrors the user's intention: if we cannot clear one more exposure
    plus the mandatory inter-exposure guard, we do nothing.

    Parameters
    ----------
    window_sum : float
        Seconds already consumed in the window.
    cap_s : float
        Window cap in seconds.
    cfg : PlannerConfig
        Global planner configuration.
    palette : Optional[Iterable[str]]
        Optional set/list of filters to consider for the min exposure. If None,
        use cfg.filters / cfg.exposure_by_filter.
    include_readout_floor : bool
        If True, includes cfg.readout_s as a conservative floor. Default False
        to match the "inter_exposure_min + exposure_by_filter" rule.

    Returns
    -------
    bool
        True if cap_s - window_sum > (inter_exposure_min_s + min_exposure_in_palette [+ readout_s])
    """
    remaining = float(cap_s) - float(window_sum)
    if remaining <= 0.0 or not math.isfinite(remaining):
        return False
    exp_min = min_exposure_in_palette(cfg, palette)
    if not math.isfinite(exp_min):
        # No exposure data; be conservative and allow planning to proceed.
        return True
    threshold = float(cfg.inter_exposure_min_s) + float(exp_min)
    if include_readout_floor:
        try:
            threshold += float(cfg.readout_s)
        except Exception:
            pass
    return remaining > threshold


def opportunity_cost_seconds_cached(
    t: dict,
    filt: str,
    est_visit_s: float,
    now_mjd: float,
    cfg: PlannerConfig,
    phot_cfg: PhotomConfig,
    sky_provider,
    sky_cfg: SkyModelConfig,
    site: EarthLocation,
    *,
    minutes: int = 1,
) -> float:
    """Fast path opportunity cost using minute-binned cached m5 lookups.

    This mirrors the math in ``_opportunity_cost_seconds`` but allows the
    caller to control the m5 minute bucket size via ``minutes`` and avoids
    recomputing celestial transforms when cached values are available.
    """

    # m5 at "now"
    try:
        now_mjd_val = float(now_mjd)
    except Exception:
        return 0.0
    m5_now = _cached_m5_at_time(
        target=t,
        filt=filt,
        cfg=cfg,
        phot_cfg=phot_cfg,
        sky_provider=sky_provider,
        sky_cfg=sky_cfg,
        mjd=now_mjd_val,
        minutes=max(1, int(minutes)),
        site=site,
        tag_best=False,
    )
    # m5 at best time (fall back to now if missing)
    best_ts = t.get("best_time_utc")
    best_mjd: float | None
    if best_ts is None:
        best_mjd = None
    else:
        try:
            import pandas as pd

            if isinstance(best_ts, pd.Timestamp):
                best_mjd = Time(best_ts.tz_convert("UTC")).mjd
            else:
                best_mjd = Time(pd.Timestamp(best_ts).tz_localize("UTC")).mjd
        except Exception:
            best_mjd = None
    m5_best = (
        _cached_m5_at_time(
            target=t,
            filt=filt,
            cfg=cfg,
            phot_cfg=phot_cfg,
            sky_provider=sky_provider,
            sky_cfg=sky_cfg,
            mjd=float(best_mjd),
            minutes=max(1, int(minutes)),
            site=site,
            tag_best=True,
        )
        if best_mjd is not None
        else m5_now
    )
    if m5_now is None or m5_best is None:
        return 0.0
    delta_m = max(0.0, float(m5_best) - float(m5_now))
    opp = float(est_visit_s) * (10 ** (0.8 * delta_m) - 1.0)
    cap = float(getattr(cfg, "inbatch_cost_cap_s", 600.0))
    return float(min(max(0.0, opp), cap))
