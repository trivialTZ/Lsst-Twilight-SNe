"""Utilities for tracking per-supernova detection history and priorities.

The planner keeps a small record for each supernova describing how many
detections have been taken, in which filters and the accumulated exposure time.
From this history a simple priority score is derived; a value of ``1`` means
the SN should be observed in the current strategy stage, while ``0`` indicates
that the goal for the stage has been met.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from .constraints import effective_min_sep


@dataclass
class _SNHistory:
    """Internal record for a single supernova.

    Attributes
    ----------
    detections : int
        Number of filter exposures recorded.
    exposure_s : float
        Total exposure time in seconds.
    filters : set of str
        Unique filters observed for the supernova.
    escalated : bool
        Flag indicating whether the target has progressed to the light-curve
        strategy stage.
    last_seen_mjd : float or None
        MJD of the most recent observation across all filters.
    """

    detections: int = 0
    exposure_s: float = 0.0
    filters: Set[str] = field(default_factory=set)
    escalated: bool = False
    # Last observation MJD per filter for cadence enforcement
    last_mjd_by_filter: Dict[str, float] = field(default_factory=dict)
    # Last observation regardless of filter for unique-first strategy
    last_seen_mjd: Optional[float] = None
    # Full visit log of (mjd, filter)
    visits: List[Tuple[float, str]] = field(default_factory=list)


@dataclass
class PriorityTracker:
    """Track detections and compute dynamic priority scores.

    ``detections`` counts the number of individual filter exposures recorded
    for the SN.  The :meth:`score` method returns ``1.0`` when a supernova still
    requires attention under the current strategy (Hybrid, light-curve, or
    unique-first) and ``0.0`` once the respective goal has been satisfied.
    :meth:`peek_score` provides the same evaluation without mutating the
    internal state.
    """

    hybrid_detections: int = 2
    hybrid_exposure_s: float = 300.0
    lc_detections: int = 5
    lc_exposure_s: float = 300.0
    unique_lookback_days: float = 999.0
    unique_first_resume_score: float = 0.0
    history: Dict[str, _SNHistory] = field(default_factory=dict)
    # Optional external visit schedule (e.g., WFD) keyed by name → band → sorted MJDs
    external_visits_by_name: Optional[Dict[str, Dict[str, List[float]]]] = None

    BLUE = {"g", "r"}
    RED = {"i", "z", "y"}

    def color_counts(
        self, name: str, now_mjd: float, window_days: float
    ) -> tuple[int, int]:
        """Return (#blue, #red) visits within ``[now - window_days, now]``."""
        hist = self.history.get(name)
        if not hist or not hist.visits:
            return (0, 0)
        start = now_mjd - window_days
        blue = 0
        red = 0
        for mjd, filt in hist.visits:
            if mjd < start or mjd > now_mjd:
                continue
            if filt in self.BLUE:
                blue += 1
            elif filt in self.RED:
                red += 1
        return (blue, red)

    def filter_counts(
        self, name: str, now_mjd: float, window_days: float
    ) -> Dict[str, int]:
        """Return per-filter visit counts within ``[now - window_days, now]``.

        Used by the band-diversity boost to encourage under-observed filters.
        """
        hist = self.history.get(name)
        counts: Dict[str, int] = {}
        if not hist or not hist.visits:
            return counts
        start = now_mjd - window_days
        for mjd, filt in hist.visits:
            if mjd < start or mjd > now_mjd:
                continue
            counts[filt] = counts.get(filt, 0) + 1
        return counts

    def _prev_next_mjd(
        self, name: str, filt: str, now_mjd: float
    ) -> tuple[Optional[float], Optional[float]]:
        """Return (previous, next) MJD for ``name``/``filt`` including external visits."""

        mjds: List[float] = []
        hist = self.history.get(name)
        if hist and hist.visits:
            for mjd, f in hist.visits:
                if f != filt:
                    continue
                try:
                    val = float(mjd)
                except Exception:
                    continue
                if math.isfinite(val):
                    mjds.append(val)
        if self.external_visits_by_name:
            try:
                ext = self.external_visits_by_name.get(name, {}).get(filt, [])
            except Exception:
                ext = []
            for mjd in ext or []:
                try:
                    val = float(mjd)
                except Exception:
                    continue
                if math.isfinite(val):
                    mjds.append(val)
        if not mjds:
            return (None, None)
        prev_candidates = [m for m in mjds if m <= now_mjd]
        next_candidates = [m for m in mjds if m >= now_mjd]
        prev = max(prev_candidates) if prev_candidates else None
        next_mjd = min(next_candidates) if next_candidates else None
        return (prev, next_mjd)

    def color_deficit(
        self, name: str, now_mjd: float, target_pairs: int, window_days: float
    ) -> tuple[int, int]:
        """Return deficits in (#blue, #red) pairs relative to ``target_pairs``."""
        blue, red = self.color_counts(name, now_mjd, window_days)
        return (target_pairs - blue, target_pairs - red)

    def _has_only_blue_or_red(self, name: str) -> tuple[bool, Optional[str]]:
        """Return flag and colour group if only blue or only red has been seen."""
        hist = self.history.get(name)
        if not hist or not hist.visits:
            return (False, None)
        colours = {
            ("blue" if f in self.BLUE else "red")
            for _, f in hist.visits
            if f in (self.BLUE | self.RED)
        }
        if colours == {"blue"}:
            return (True, "blue")
        if colours == {"red"}:
            return (True, "red")
        return (False, None)

    def cosmology_boost(
        self,
        name: str,
        filt: str,
        now_mjd: float,
        target_pairs: int,
        window_days: float,
        alpha: float,
    ) -> float:
        """Return a multiplicative boost >=1.0 favouring the missing colour group."""
        blue_def, red_def = self.color_deficit(name, now_mjd, target_pairs, window_days)
        deficit = blue_def if filt in self.BLUE else red_def
        return float(1.0 + alpha * max(deficit, 0))

    def compute_filter_bonus(
        self,
        name: str,
        filt: str,
        now_mjd: float,
        target_d: float,
        sigma_d: float,
        cadence_weight: float,
        first_epoch_weight: float,
        cosmo_weight_by_filter: Dict[str, float],
        target_pairs: int,
        window_days: float,
        alpha: float,
        first_epoch_color_boost: float,
        *,
        diversity_enable: bool = False,
        diversity_target_per_filter: int = 1,
        diversity_window_days: float = 5.0,
        diversity_alpha: float = 0.3,
    ) -> float:
        """Return combined cadence and band preference bonus for ``filt``.

        The function blends a cadence-centered Gaussian bonus with either
        (a) the original blue/red colour-group boost, or (b) a per-filter
        band-diversity boost (when ``diversity_enable=True``) that encourages
        under-observed individual bands within a recent time window.
        """

        base = self.cadence_bonus(
            name,
            filt,
            now_mjd,
            target_d,
            sigma_d,
            weight=cadence_weight,
            first_epoch_weight=first_epoch_weight,
        )
        cosmo_w = cosmo_weight_by_filter.get(filt, 1.0)

        if diversity_enable:
            # Per-filter diversity: favour a band if its recent count is below
            # the configured target.  Instead of pure multiplication (which can
            # vanish when the cadence term is zero), add a diversity component
            # scaled by the cadence weight so that an unseen filter still
            # receives a positive preference.
            counts = self.filter_counts(name, now_mjd, diversity_window_days)
            seen_in_band = int(counts.get(filt, 0))
            deficit = max(0, int(diversity_target_per_filter) - seen_in_band)
            cadence_component = base * cosmo_w
            diversity_component = 0.0
            if deficit > 0:
                diversity_component = (
                    cosmo_w
                    * max(0.0, cadence_weight)
                    * max(0.0, diversity_alpha)
                    * deficit
                )
            # Re-interpret first-epoch boost as "first epoch in this band" nudge.
            first_epoch_nudge = 1.0
            if seen_in_band == 0:
                first_epoch_nudge = max(1.0, first_epoch_color_boost)
            return float((cadence_component + diversity_component) * first_epoch_nudge)
        else:
            # Original colour-group boost: favour the missing color side.
            boost = self.cosmology_boost(
                name, filt, now_mjd, target_pairs, window_days, alpha
            )
            only_one, which = self._has_only_blue_or_red(name)
            nudge = 1.0
            if only_one:
                if which == "blue" and filt in self.RED:
                    nudge = max(1.0, first_epoch_color_boost)
                elif which == "red" and filt in self.BLUE:
                    nudge = max(1.0, first_epoch_color_boost)
            return float(base * cosmo_w * boost * nudge)

    def redshift_boost(
        self,
        z: Optional[float],
        z_ref: float = 0.08,
        max_boost: float = 1.7,
    ) -> float:
        """Return a multiplicative boost favouring low-redshift SNe.

        Parameters
        ----------
        z : float or None
            Redshift value for the supernova. If ``None`` or not finite,
            returns ``1.0`` (no boost).
        z_ref : float, default 0.08
            Reference redshift. Objects at ``z_ref`` or higher receive
            no boost; lower redshift objects are boosted linearly toward
            ``max_boost`` as ``z``→0.
        max_boost : float, default 1.7
            Maximum multiplicative boost applied at ``z=0``.

        Returns
        -------
        float
            Boost factor ≥ 1.0.
        """
        try:
            if z is None:
                return 1.0
            z_val = float(z)
        except Exception:
            return 1.0
        if not math.isfinite(z_val) or z_ref <= 0.0 or max_boost <= 1.0:
            return 1.0
        if z_val >= z_ref:
            return 1.0
        frac = max(0.0, (z_ref - z_val) / z_ref)
        return float(1.0 + (max_boost - 1.0) * frac)

    def record_detection(
        self,
        name: str,
        exposure_s: float,
        filters: List[str],
        mjd: float | None = None,
    ) -> None:
        """Record detections for ``name`` with given exposure, filters, and time.

        Parameters
        ----------
        name : str
            Supernova identifier.
        exposure_s : float
            Exposure time in seconds for the visit.
        filters : list of str
            Filters used during the visit.
        mjd : float, optional
            MJD start time of the visit. If provided, per-filter cadence
            timestamps are updated.

        Returns
        -------
        None

        Notes
        -----
        Updates the internal history for ``name`` in-place.
        """
        hist = self.history.setdefault(name, _SNHistory())
        hist.detections += len(filters)
        hist.exposure_s += exposure_s
        hist.filters.update(filters)
        if mjd is not None:
            for f in filters:
                hist.last_mjd_by_filter[f] = mjd
                hist.visits.append((mjd, f))
            hist.last_seen_mjd = mjd

    # alias for clarity
    update = record_detection

    def days_since(self, name: str, filt: str, now_mjd: float) -> Optional[float]:
        """Return days since ``filt`` was last observed for ``name``.

        Parameters
        ----------
        name : str
            Supernova identifier.
        filt : str
            Photometric filter.
        now_mjd : float
            Current MJD against which the elapsed time is measured.

        Returns
        -------
        float or None
            Days since last observation in ``filt`` or ``None`` if the SN has
            not been observed in that filter.
        """

        prev, _ = self._prev_next_mjd(name, filt, now_mjd)
        if prev is None:
            return None
        return now_mjd - prev

    def cadence_gate(
        self,
        name: str,
        filt: str,
        now_mjd: float,
        target_d: float,
        jitter_d: float,
    ) -> bool:
        """Determine if ``filt`` may be observed at ``now_mjd``.

        Returns ``True`` if both conditions hold:
        - No previous visit exists or the elapsed time since the last visit
          exceeds ``target_d - jitter_d``.
        - No future external visit is scheduled sooner than ``target_d - jitter_d``
          from ``now_mjd``.
        """

        prev, next_mjd = self._prev_next_mjd(name, filt, now_mjd)
        threshold = max(0.0, target_d - jitter_d)
        prev_ok = True
        if prev is not None:
            prev_ok = (now_mjd - prev) >= threshold
        next_ok = True
        if next_mjd is not None:
            next_ok = (next_mjd - now_mjd) >= threshold
        return bool(prev_ok and next_ok)

    def cadence_bonus(
        self,
        name: str,
        filt: str,
        now_mjd: float,
        target_d: float,
        sigma_d: float,
        weight: float = 0.25,
        first_epoch_weight: float = 0.0,
    ) -> float:
        """Gaussian bonus peaking at ``target_d`` days since last visit.

        The bonus is scaled by ``weight`` and defaults to ``first_epoch_weight``
        when ``filt`` has not yet been observed.
        """

        prev, next_mjd = self._prev_next_mjd(name, filt, now_mjd)
        if prev is None and next_mjd is None:
            return float(first_epoch_weight)
        prev_dist = math.inf if prev is None else now_mjd - prev
        next_dist = math.inf if next_mjd is None else next_mjd - now_mjd
        delta_eff = min(prev_dist, next_dist)
        if not math.isfinite(delta_eff):
            return float(first_epoch_weight)
        if sigma_d <= 0:
            return 0.0
        return float(weight * math.exp(-0.5 * ((delta_eff - target_d) / sigma_d) ** 2))

    def _score(
        self,
        hist: _SNHistory,
        sn_type: Optional[str],
        strategy: Literal["hybrid", "lc", "unique_first"],
        mutate: bool,
        now_mjd: Optional[float] = None,
    ) -> float:
        """Internal helper implementing the priority scoring rules.

        For ``unique_first``, return ``1.0`` until the first successful
        detection is recorded; thereafter return a negative score so the
        scheduler can drop the supernova before applying any global caps. If
        a current MJD is provided and the elapsed time since the last visit
        exceeds ``unique_lookback_days``, ``unique_first_resume_score`` is
        returned instead, allowing opt-in repeats.

        Parameters
        ----------
        hist : _SNHistory
            Detection history for the supernova.
        sn_type : str or None
            Classification string (e.g., ``'Ia'``) used to decide escalation.
        strategy : {'hybrid', 'lc', 'unique_first'}
            Current observing strategy.
        mutate : bool
            If ``True``, ``hist`` may be modified (e.g., ``escalated`` flag).

        Returns
        -------
        float
            Priority score in ``[0, 1]`` where ``1`` means further
            observations are required.

        Notes
        -----
        When ``mutate`` is ``True`` the ``escalated`` field of ``hist`` can be
        updated as side effect.
        """

        if strategy == "unique_first":
            # Unseen objects get a positive score so they can be scheduled.
            if hist.detections == 0:
                return 1.0
            # Optionally allow repeats after a lookback interval.
            if now_mjd is not None and self.unique_lookback_days is not None:
                last = hist.last_seen_mjd
                if last is not None and (now_mjd - last) > self.unique_lookback_days:
                    return float(self.unique_first_resume_score)
            # Otherwise return a negative score so the scheduler can drop it
            # before applying any caps.
            return -1.0

        escalated = hist.escalated or strategy == "lc"
        if strategy == "lc" and mutate:
            hist.escalated = True
        if not escalated:
            met_hybrid = (
                hist.detections >= self.hybrid_detections and len(hist.filters) >= 2
            ) or hist.exposure_s >= self.hybrid_exposure_s
            if not met_hybrid:
                return 1.0
            if sn_type and "ia" in sn_type.lower() or strategy == "lc":
                if mutate:
                    hist.escalated = True
                escalated = True
            else:
                return 0.0

        met_lc = hist.detections >= self.lc_detections or (
            hist.exposure_s >= self.lc_exposure_s and len(hist.filters) >= 2
        )
        return 0.0 if met_lc else 1.0

    def score(
        self,
        name: str,
        sn_type: Optional[str] = None,
        strategy: Literal["hybrid", "lc", "unique_first"] = "hybrid",
        now_mjd: Optional[float] = None,
    ) -> float:
        """Return the priority score for a supernova and update its state.

        Parameters
        ----------
        name : str
            Supernova identifier.
        sn_type : str, optional
            Classification string; used to determine escalation policy.
        strategy : {'hybrid', 'lc', 'unique_first'}, default 'hybrid'
            Observing strategy stage.
        now_mjd : float, optional
            Current MJD used for the ``unique_first`` lookback evaluation.

        Returns
        -------
        float
            Priority score in ``[0, 1]``.

        Notes
        -----
        Calling this method mutates the internal history for ``name``.
        """
        hist = self.history.setdefault(name, _SNHistory())
        return self._score(hist, sn_type, strategy, mutate=True, now_mjd=now_mjd)

    def peek_score(
        self,
        name: str,
        sn_type: Optional[str] = None,
        strategy: Literal["hybrid", "lc", "unique_first"] = "hybrid",
        now_mjd: Optional[float] = None,
    ) -> float:
        """Compute the priority score without mutating state.

        Parameters
        ----------
        name : str
            Supernova identifier.
        sn_type : str, optional
            Classification string; used to determine escalation policy.
        strategy : {'hybrid', 'lc', 'unique_first'}, default 'hybrid'
            Observing strategy stage.

        Returns
        -------
        float
            Priority score in ``[0, 1]``.

        Notes
        -----
        Uses a shallow copy of the SN history so the original record remains
        unchanged.
        """
        hist = self.history.setdefault(name, _SNHistory())
        # Work on a shallow copy so the caller does not see side effects
        tmp = _SNHistory(
            hist.detections,
            hist.exposure_s,
            set(hist.filters),
            hist.escalated,
            dict(hist.last_mjd_by_filter),
            hist.last_seen_mjd,
        )
        return self._score(tmp, sn_type, strategy, mutate=False, now_mjd=now_mjd)


def apply_moon_penalty(
    priority_score: float,
    filt: str,
    moon_sep_deg: float,
    moon_alt_deg: float,
    moon_phase_frac: float,
    weight: float = 1.0,
    base_min_sep: Dict[str, float] | None = None,
) -> float:
    """Reduce ``priority_score`` if the Moon is too close.

    Parameters
    ----------
    priority_score : float
        Baseline priority score before applying the Moon penalty.
    filt : str
        Photometric filter for the planned observation.
    moon_sep_deg : float
        Angular separation between target and Moon in degrees.
    moon_alt_deg : float
        Altitude of the Moon in degrees.
    moon_phase_frac : float
        Fractional lunar illumination in ``[0, 1]``.
    weight : float, default 1.0
        Scaling applied to the penalty.
    base_min_sep : dict of str to float, optional
        Baseline separation requirements; defaults to :data:`BASE_MIN_SEP`.

    Returns
    -------
    float
        Adjusted priority score, reduced when the Moon is too close.

    Notes
    -----
    The penalty is proportional to the shortfall in separation relative to the
    required minimum. If the requirement is non-positive, the score is returned
    unchanged.
    """

    req = effective_min_sep(filt, moon_alt_deg, moon_phase_frac, base_min_sep)
    if req <= 0:
        return priority_score
    penalty = max(0.0, (req - moon_sep_deg) / req)
    return priority_score - weight * penalty
