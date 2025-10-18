"""Pair construction and scoring for DP planning.

Responsibility
--------------
- Candidate coordinate extraction
- Saturation-capped exposure estimation per pair
- Approximate m5 estimation per pair (cached)
- Pair scoring including cadence and low-z Ia boosts
- Build per-filter PairItem lists for a twilight window

Hot path
--------
- ``_pair_score``
- ``_build_global_pairs_for_window``
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import os

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ..astro_utils import (
    compute_capped_exptime,
    great_circle_sep_deg,
    slew_time_seconds,
)
from ..config import PlannerConfig
from ..photom_rubin import PhotomConfig
from ..priority import PriorityTracker
from ..sky_model import SkyModelConfig
from .caching import _cached_m5_at_time, _finite_or_none
from .selection import _is_low_z_ia
from .types import PairItem


# -------------------- Parallel context (per worker process) --------------------
_PAIRS_CTX: dict = {}


def _init_pairs_worker(
    cfg: PlannerConfig,
    tracker: PriorityTracker,
    phot_cfg: PhotomConfig,
    sky_cfg: SkyModelConfig,
    sky_provider,
    mag_lookup: Dict[str, Dict[str, float]],
    epsilon_snr: float,
    prev_coord: Optional[Tuple[float, float]],
    now_mjd: float,
    cadence_params: dict,
):
    """Initializer runs once per worker to avoid re-pickling big objects."""
    global _PAIRS_CTX
    _PAIRS_CTX = {
        "cfg": cfg,
        "tracker": tracker,
        "phot_cfg": phot_cfg,
        "sky_cfg": sky_cfg,
        "sky_provider": sky_provider,
        "mag_lookup": mag_lookup,
        "epsilon_snr": float(epsilon_snr),
        "prev_coord": prev_coord,
        "now_mjd": float(now_mjd),
        "cadence_params": dict(cadence_params),
    }


def _score_pair_worker(task: Tuple[dict, str, float]):
    """Compute one (target, filter) PairItem. Return (filt, PairItem) or None."""
    t, filt, cad_tgt_eff = task
    C = _PAIRS_CTX
    cfg = C["cfg"]
    tracker = C["tracker"]
    phot_cfg = C["phot_cfg"]
    sky_cfg = C["sky_cfg"]
    sky_provider = C["sky_provider"]
    mag_lookup = C["mag_lookup"]
    epsilon_snr = C["epsilon_snr"]
    prev_coord = C["prev_coord"]
    now_mjd = C["now_mjd"]
    cadence_params = C["cadence_params"]
    cadence_params["cad_tgt_eff"] = cad_tgt_eff
    try:
        exp_s = _compute_capped_exptime_for_pair(
            t, filt, cfg, mag_lookup, cfg.exposure_by_filter.get(filt, 30.0), now_mjd
        )
        approx_time = _visit_unit_time(exp_s, prev_coord, _candidate_coord(t), cfg)
        score, snr_margin = _pair_score(
            t,
            filt,
            now_mjd,
            tracker,
            cfg,
            cadence_params,
            mag_lookup,
            phot_cfg,
            sky_cfg,
            sky_provider,
            epsilon_snr,
        )
        if score <= 0.0:
            return None
        item = PairItem(
            name=t["Name"],
            filt=filt,
            score=score,
            approx_time_s=approx_time,
            density=score / max(approx_time, 1.0),
            snr_margin=snr_margin,
            exp_s=exp_s,
            candidate=t,
        )
        return (filt, item)
    except Exception:
        return None


def _candidate_coord(target: dict) -> Optional[Tuple[float, float]]:
    """Return ``(ra_deg, dec_deg)`` if both coordinates are finite."""

    try:
        ra = float(target.get("RA_deg"))
        dec = float(target.get("Dec_deg"))
    except Exception:
        return None
    if not np.isfinite(ra) or not np.isfinite(dec):
        return None
    return (ra, dec)


def _compute_capped_exptime_for_pair(
    target: dict,
    filt: str,
    cfg: PlannerConfig,
    mag_lookup: Dict[str, Dict[str, float]],
    default_exp: float,
    now_mjd: float,
) -> float:
    """Return saturation-capped exposure for ``target`` in ``filt``."""

    name = target.get("Name")
    mag_map = mag_lookup.get(name, {}) if name is not None else {}
    orig_mag = getattr(cfg, "current_mag_by_filter", None)
    orig_alt = getattr(cfg, "current_alt_deg", None)
    orig_ra = getattr(cfg, "current_ra_deg", None)
    orig_dec = getattr(cfg, "current_dec_deg", None)
    orig_mjd = getattr(cfg, "current_mjd", None)
    orig_z = getattr(cfg, "current_redshift", None)
    try:
        cfg.current_mag_by_filter = mag_map
        cfg.current_alt_deg = float(target.get("max_alt_deg", np.nan))
        coord = _candidate_coord(target)
        cfg.current_ra_deg = coord[0] if coord else None
        cfg.current_dec_deg = coord[1] if coord else None
        best_mjd = target.get("best_time_mjd")
        cfg.current_mjd = best_mjd if isinstance(best_mjd, (float, int)) else now_mjd
        redshift = target.get("redshift")
        cfg.current_redshift = float(redshift) if redshift is not None else None
        exp_s, _flags = compute_capped_exptime(filt, cfg)
        return float(exp_s)
    except Exception:
        return float(default_exp)
    finally:
        cfg.current_mag_by_filter = orig_mag
        cfg.current_alt_deg = orig_alt
        cfg.current_ra_deg = orig_ra
        cfg.current_dec_deg = orig_dec
        cfg.current_mjd = orig_mjd
        cfg.current_redshift = orig_z


def _estimate_m5_for_pair(
    target: dict,
    filt: str,
    cfg: PlannerConfig,
    phot_cfg: PhotomConfig,
    sky_provider,
    sky_cfg: SkyModelConfig,
    now_mjd: float,
) -> Optional[float]:
    """Return approximate 5σ depth for ``target`` in ``filt``."""

    best_mjd_val = _finite_or_none(target.get("best_time_mjd"))
    tag_best = best_mjd_val is not None
    mjd_use = best_mjd_val if tag_best else float(now_mjd)

    site_loc = getattr(cfg, "_site_location_cache", None)
    if site_loc is None:
        site_loc = EarthLocation(
            lat=float(cfg.lat_deg) * u.deg,
            lon=float(cfg.lon_deg) * u.deg,
            height=float(cfg.height_m) * u.m,
        )
        setattr(cfg, "_site_location_cache", site_loc)

    m5_step = int(getattr(cfg, "pairs_m5_minutes", 1) or 1)
    return _cached_m5_at_time(
        target=target,
        filt=filt,
        cfg=cfg,
        phot_cfg=phot_cfg,
        sky_provider=sky_provider,
        sky_cfg=sky_cfg,
        mjd=float(mjd_use),
        minutes=max(1, m5_step),
        site=site_loc,
        tag_best=tag_best,
    )


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


def _pair_score(
    target: dict,
    filt: str,
    now_mjd: float,
    tracker: PriorityTracker,
    cfg: PlannerConfig,
    cadence_params: dict,
    mag_lookup: Dict[str, Dict[str, float]],
    phot_cfg: PhotomConfig,
    sky_cfg: SkyModelConfig,
    sky_provider,
    epsilon_snr: float,
) -> tuple[float, float]:
    """Return ``(score, snr_margin)`` for a candidate/filter pair."""

    name = target["Name"]
    strategy = getattr(cfg, "priority_strategy", "hybrid")
    need = tracker.peek_score(name, target.get("sn_type"), strategy, now_mjd)
    if need <= 0.0:
        return (0.0, 0.0)
    lowz_boost = 1.0
    if getattr(cfg, "redshift_boost_enable", True):
        lowz_boost *= tracker.redshift_boost(
            target.get("redshift"),
            getattr(cfg, "redshift_low_ref", 0.08),
            getattr(cfg, "redshift_boost_max", 1.7),
        )
    if _is_low_z_ia(target.get("sn_type"), target.get("redshift"), cfg):
        mult = getattr(cfg, "low_z_ia_priority_multiplier", 1.0)
        try:
            mult = float(mult)
        except Exception:
            mult = 1.0
        lowz_boost *= max(mult, 0.0)
    cad_bonus = 0.0
    if cadence_params["cad_on"]:
        cad_bonus = tracker.compute_filter_bonus(
            name,
            filt,
            now_mjd,
            cadence_params["cad_tgt_eff"],
            cadence_params["cad_sig"],
            cadence_params["cad_wt"],
            cadence_params["cad_first"],
            cfg.cosmo_weight_by_filter,
            cfg.color_target_pairs,
            cfg.color_window_days,
            cfg.color_alpha,
            cfg.first_epoch_color_boost,
            diversity_enable=cadence_params["diversity_enable"],
            diversity_target_per_filter=cadence_params["diversity_target"],
            diversity_window_days=cadence_params["diversity_window"],
            diversity_alpha=cadence_params["diversity_alpha"],
        )
    snr_margin = 0.0
    if epsilon_snr > 0.0:
        mag = mag_lookup.get(name, {}).get(filt)
        m5 = _estimate_m5_for_pair(
            target, filt, cfg, phot_cfg, sky_provider, sky_cfg, now_mjd
        )
        if mag is not None and m5 is not None:
            snr_margin = max(0.0, float(m5) - float(mag))
    score = float(need) * float(lowz_boost) * (1.0 + cad_bonus) * (
        1.0 + epsilon_snr * snr_margin
    )
    return (score, snr_margin)


def _build_global_pairs_for_window(
    candidates: list[dict],
    cfg: PlannerConfig,
    tracker: PriorityTracker,
    now_mjd: float,
    prev_coord: Optional[Tuple[float, float]],
    mag_lookup: Dict[str, Dict[str, float]],
    cadence_params: dict,
    phot_cfg: PhotomConfig,
    sky_cfg: SkyModelConfig,
    sky_provider,
) -> dict[str, list[PairItem]]:
    """Return per-filter ``PairItem`` lists sorted by score.
    Optionally parallelize over (candidate, filter) pairs using processes.
    """

    epsilon_snr = float(getattr(cfg, "dp_snr_margin_epsilon", 0.0) or 0.0)
    per_filter: dict[str, list[PairItem]] = defaultdict(list)
    topk_per_filter = getattr(cfg, "pairs_topk_per_filter", None)
    # Build tasks in the parent process (apply cadence gate here).
    tasks: list[tuple[dict, str, float]] = []
    for target in candidates:
        coord = _candidate_coord(target)
        cad_tgt_eff = cadence_params["cad_tgt"]
        if cadence_params["cad_on"] and _is_low_z_ia(
            target.get("sn_type"), target.get("redshift"), cfg
        ):
            override = getattr(cfg, "low_z_ia_cadence_days_target", None)
            if isinstance(override, (int, float)) and override > 0.0:
                cad_tgt_eff = float(override)
        cadence_params["cad_tgt_eff"] = cad_tgt_eff
        policy_allowed = target.get("policy_allowed", [])
        for filt in policy_allowed:
            if cadence_params["cad_on"] and not tracker.cadence_gate(
                target["Name"],
                filt,
                now_mjd,
                cadence_params["cad_tgt_eff"],
                cadence_params["cad_jit"],
            ):
                continue
            tasks.append((target, filt, cad_tgt_eff))

    use_parallel = bool(getattr(cfg, "pairs_parallel", True))
    threshold = int(getattr(cfg, "pairs_parallel_threshold", 5000))
    max_workers_cfg = getattr(cfg, "pairs_max_workers", 0)
    workers = (
        os.cpu_count() or 2
        if (not isinstance(max_workers_cfg, int) or max_workers_cfg <= 0)
        else max_workers_cfg
    )

    if use_parallel and len(tasks) > threshold and workers > 1:
        chunksize = max(1, len(tasks) // (workers * 4))
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_pairs_worker,
                initargs=(
                    cfg,
                    tracker,
                    phot_cfg,
                    sky_cfg,
                    sky_provider,
                    mag_lookup,
                    epsilon_snr,
                    prev_coord,
                    now_mjd,
                    cadence_params,
                ),
            ) as ex:
                for res in ex.map(_score_pair_worker, tasks, chunksize=chunksize):
                    if res is None:
                        continue
                    f, item = res
                    per_filter[f].append(item)
        except Exception as e:
            if bool(getattr(cfg, "debug_planner", False)):
                print(f"[pairs] Parallel disabled ({e}); falling back to serial.")
            _init_pairs_worker(
                cfg,
                tracker,
                phot_cfg,
                sky_cfg,
                sky_provider,
                mag_lookup,
                epsilon_snr,
                prev_coord,
                now_mjd,
                cadence_params,
            )
            for task in tasks:
                res = _score_pair_worker(task)
                if res is None:
                    continue
                f, item = res
                per_filter[f].append(item)
    else:
        _init_pairs_worker(
            cfg,
            tracker,
            phot_cfg,
            sky_cfg,
            sky_provider,
            mag_lookup,
            epsilon_snr,
            prev_coord,
            now_mjd,
            cadence_params,
        )
        for task in tasks:
            res = _score_pair_worker(task)
            if res is None:
                continue
            f, item = res
            per_filter[f].append(item)

    for filt, items in per_filter.items():
        items.sort(key=lambda it: it.score, reverse=True)
        if isinstance(topk_per_filter, int) and topk_per_filter > 0:
            per_filter[filt] = items[: topk_per_filter]
    return per_filter
