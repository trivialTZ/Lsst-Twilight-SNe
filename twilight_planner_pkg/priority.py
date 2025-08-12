from __future__ import annotations
"""Utilities for tracking per-supernova detection history and priorities.

The planner keeps a small record for each supernova describing how many
detections have been taken, in which filters and the accumulated exposure time.
From this history a simple priority score is derived; a value of ``1`` means
the SN should be observed in the current strategy stage, while ``0`` indicates
that the goal for the stage has been met.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, List


@dataclass
class _SNHistory:
    """Internal record for a single supernova."""

    detections: int = 0
    exposure_s: float = 0.0
    filters: Set[str] = field(default_factory=set)
    escalated: bool = False


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

    def record_detection(self, name: str, exposure_s: float, filters: List[str]) -> None:
        """Record detections for ``name`` with given exposure and filters."""
        hist = self.history.setdefault(name, _SNHistory())
        hist.detections += len(filters)
        hist.exposure_s += exposure_s
        hist.filters.update(filters)

    # alias for clarity
    update = record_detection

    def _score(self, hist: _SNHistory, sn_type: Optional[str], strategy: str, mutate: bool) -> float:
        """Internal helper implementing the scoring rules."""

        escalated = hist.escalated or strategy == "lc"
        if strategy == "lc" and mutate:
            hist.escalated = True
        if not escalated:
            met_hybrid = (
                (hist.detections >= self.hybrid_detections and len(hist.filters) >= 2)
                or hist.exposure_s >= self.hybrid_exposure_s
            )
            if not met_hybrid:
                return 1.0
            if sn_type and "ia" in sn_type.lower() or strategy == "lc":
                if mutate:
                    hist.escalated = True
                escalated = True
            else:
                return 0.0

        met_lc = (
            hist.detections >= self.lc_detections
            or (hist.exposure_s >= self.lc_exposure_s and len(hist.filters) >= 2)
        )
        return 0.0 if met_lc else 1.0

    def score(self, name: str, sn_type: Optional[str] = None, strategy: str = "hybrid") -> float:
        """Return the priority score for a supernova and update its state."""
        hist = self.history.setdefault(name, _SNHistory())
        return self._score(hist, sn_type, strategy, mutate=True)

    def peek_score(self, name: str, sn_type: Optional[str] = None, strategy: str = "hybrid") -> float:
        """Return the score without updating the internal history."""
        hist = self.history.setdefault(name, _SNHistory())
        # Work on a shallow copy so the caller does not see side effects
        tmp = _SNHistory(hist.detections, hist.exposure_s, set(hist.filters), hist.escalated)
        return self._score(tmp, sn_type, strategy, mutate=False)
