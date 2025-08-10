from __future__ import annotations
"""Utilities for tracking per-supernova detection history and priorities."""
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

    Parameters
    ----------
    hybrid_detections : int, optional
        Minimum detections (across ≥2 filters) for the Hybrid goal.
    hybrid_exposure_s : float, optional
        Total exposure seconds triggering the Hybrid goal.
    lc_detections : int, optional
        Detections required for the LSST-only light-curve goal.
    lc_exposure_s : float, optional
        Exposure seconds for the LSST-only goal (must also span ≥2 filters).
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

    def score(self, name: str, sn_type: Optional[str] = None, strategy: str = "hybrid") -> float:
        """Return the priority score for a supernova."""
        hist = self.history.setdefault(name, _SNHistory())

        if strategy == "lc":
            hist.escalated = True

        if not hist.escalated:
            met_hybrid = (
                (hist.detections >= self.hybrid_detections and len(hist.filters) >= 2)
                or hist.exposure_s >= self.hybrid_exposure_s
            )
            if not met_hybrid:
                return 1.0
            if sn_type and "ia" in sn_type.lower() or strategy == "lc":
                hist.escalated = True
            else:
                return 0.0

        met_lc = (
            hist.detections >= self.lc_detections
            or (hist.exposure_s >= self.lc_exposure_s and len(hist.filters) >= 2)
        )
        return 0.0 if met_lc else 1.0
