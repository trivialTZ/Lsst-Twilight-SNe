from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class PlannerConfig:
    """Configuration container for twilight planning.

    All angles are in degrees unless otherwise noted. The planner is instrument-agnostic;
    supply appropriate parameters for your facility.
    """
    # Site
    lat_deg: float
    lon_deg: float
    height_m: float

    # Selection
    min_alt_deg: float = 20.0
    typical_days_by_type: Dict[str, int] = field(default_factory=dict)
    default_typical_days: int = 30

    # Slew/overheads
    slew_small_deg: float = 3.0
    slew_small_time_s: float = 2.0
    slew_rate_deg_per_s: float = 3.5
    slew_settle_s: float = 2.0
    readout_s: float = 2.0
    filter_change_s: float = 120.0

    # Filters
    filters: List[str] = field(default_factory=lambda: ["g","r","i","z"])
    exposure_by_filter: Dict[str, float] = field(default_factory=lambda: {"g":5.0, "r":5.0, "i":5.0, "z":5.0})
    carousel_capacity: int = 6

    # Twilight/caps
    twilight_step_min: int = 2
    evening_cap_s: float = 600.0
    morning_cap_s: float = 600.0
    max_sn_per_night: int = 10
    per_sn_cap_s: float = 120.0

    # Moon
    min_moon_sep_by_filter: Dict[str, float] = field(default_factory=lambda: {"g":30.0, "r":25.0, "i":20.0, "z":15.0})
    require_single_time_for_all_filters: bool = True

    # CSV columns (Optional: set to None for auto-detect)
    ra_col: Optional[str] = None
    dec_col: Optional[str] = None
    disc_col: Optional[str] = None
    name_col: Optional[str] = None
    type_col: Optional[str] = None
