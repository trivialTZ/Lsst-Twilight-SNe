"""Shared lightweight types for the twilight scheduler.

Only minimal, import-light definitions should live here to avoid overhead in
hot paths and to keep cross-module coupling low.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PairItem:
    """Container for a candidate SN/filter pair used by the DP planner."""

    __slots__ = (
        "name",
        "filt",
        "score",
        "approx_time_s",
        "density",
        "snr_margin",
        "exp_s",
        "candidate",
    )

    name: str
    filt: str
    score: float
    approx_time_s: float
    density: float
    snr_margin: float
    exp_s: float
    candidate: dict

