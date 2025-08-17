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
from typing import Dict, List, Optional, Set

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
    """

    detections: int = 0
    exposure_s: float = 0.0
    filters: Set[str] = field(default_factory=set)
    escalated: bool = False
    # Last observation MJD per filter for cadence enforcement
    last_mjd_by_filter: Dict[str, float] = field(default_factory=dict)


@dataclass
class PriorityTracker:
    """Track detections and compute dynamic priority scores.

    ``detections`` counts the number of individual filter exposures recorded
    for the SN.  The :meth:`score` method returns ``1.0`` when a supernova still
    requires attention under the current strategy (Hybrid or light-curve) and
    ``0.0`` once the respective goal has been satisfied.  :meth:`peek_score`
    provides the same evaluation without mutating the internal state.
    """

    hybrid_detections: int = 2
    hybrid_exposure_s: float = 300.0
    lc_detections: int = 5
    lc_exposure_s: float = 300.0
    history: Dict[str, _SNHistory] = field(default_factory=dict)

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

        hist = self.history.get(name)
        if not hist:
            return None
        last = hist.last_mjd_by_filter.get(filt)
        if last is None:
            return None
        return now_mjd - last

    def cadence_gate(
        self,
        name: str,
        filt: str,
        now_mjd: float,
        target_d: float,
        jitter_d: float,
    ) -> bool:
        """Determine if ``filt`` may be observed at ``now_mjd``.

        Returns ``True`` if the filter has never been observed or if the
        elapsed time since the last observation exceeds ``target_d - jitter_d``.
        """

        delta = self.days_since(name, filt, now_mjd)
        if delta is None:
            return True
        return delta >= max(0.0, target_d - jitter_d)

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

        delta = self.days_since(name, filt, now_mjd)
        if delta is None:
            return float(first_epoch_weight)
        if sigma_d <= 0:
            return 0.0
        return float(weight * math.exp(-0.5 * ((delta - target_d) / sigma_d) ** 2))

    def _score(
        self, hist: _SNHistory, sn_type: Optional[str], strategy: str, mutate: bool
    ) -> float:
        """Internal helper implementing the priority scoring rules.

        Parameters
        ----------
        hist : _SNHistory
            Detection history for the supernova.
        sn_type : str or None
            Classification string (e.g., ``'Ia'``) used to decide escalation.
        strategy : {'hybrid', 'lc'}
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
        self, name: str, sn_type: Optional[str] = None, strategy: str = "hybrid"
    ) -> float:
        """Return the priority score for a supernova and update its state.

        Parameters
        ----------
        name : str
            Supernova identifier.
        sn_type : str, optional
            Classification string; used to determine escalation policy.
        strategy : {'hybrid', 'lc'}, default 'hybrid'
            Observing strategy stage.

        Returns
        -------
        float
            Priority score in ``[0, 1]``.

        Notes
        -----
        Calling this method mutates the internal history for ``name``.
        """
        hist = self.history.setdefault(name, _SNHistory())
        return self._score(hist, sn_type, strategy, mutate=True)

    def peek_score(
        self, name: str, sn_type: Optional[str] = None, strategy: str = "hybrid"
    ) -> float:
        """Compute the priority score without mutating state.

        Parameters
        ----------
        name : str
            Supernova identifier.
        sn_type : str, optional
            Classification string; used to determine escalation policy.
        strategy : {'hybrid', 'lc'}, default 'hybrid'
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
            hist.detections, hist.exposure_s, set(hist.filters), hist.escalated
        )
        return self._score(tmp, sn_type, strategy, mutate=False)


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
