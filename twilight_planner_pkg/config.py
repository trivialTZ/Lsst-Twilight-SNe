from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class PlannerConfig:
    """Configuration container for twilight planning.

    Attributes
    ----------
    lat_deg, lon_deg, height_m : float
        Observatory latitude, longitude, and elevation.
    min_alt_deg : float, optional
        Minimum altitude for targets in degrees.
    typical_days_by_type : dict[str, int], optional
        Mapping from supernova type to typical visibility window.
    default_typical_days : int, optional
        Fallback visibility window in days when type is unknown.
    slew_small_deg : float, optional
        Threshold for small slews in degrees.
    slew_small_time_s : float, optional
        Time in seconds for slews ``<=`` ``slew_small_deg``.
    slew_rate_deg_per_s : float, optional
        Slew rate for large moves.
    slew_settle_s : float, optional
        Additional settling time in seconds.
    readout_s : float, optional
        Detector readout time per exposure.
    filter_change_s : float, optional
        Overhead for a filter change in seconds.
    filters : list[str], optional
        Available filters in order of priority.
    exposure_by_filter : dict[str, float], optional
        Exposure time for each filter in seconds.
    carousel_capacity : int, optional
        Maximum number of filters the carousel can hold.
    twilight_step_min : int, optional
        Step size in minutes when sampling twilight windows.
    evening_cap_s, morning_cap_s : float, optional
        Time caps in seconds for evening and morning windows.
    max_sn_per_night : int, optional
        Maximum number of supernovae scheduled per night.
    per_sn_cap_s : float, optional
        Time cap per supernova in seconds.
    min_moon_sep_by_filter : dict[str, float], optional
        Minimum Moon separation per filter in degrees.
    require_single_time_for_all_filters : bool, optional
        Require one time satisfying Moon separation for all filters.
    priority_strategy : str, optional
        Strategy for dynamic prioritization (``"hybrid"`` or ``"lc"``).
    hybrid_detections : int, optional
        Detections in ≥2 filters marking completion of the Hybrid goal.
    hybrid_exposure_s : float, optional
        Total exposure seconds marking completion of the Hybrid goal.
    lc_detections : int, optional
        Detections required for the LSST-only goal.
    lc_exposure_s : float, optional
        Total exposure seconds for the LSST-only goal.
    lc_phase_range : tuple[float, float], optional
        Phase range (days from discovery) for LSST-only coverage.
    ra_col, dec_col, disc_col, name_col, type_col : str or None, optional
        Overrides for catalog column names. ``None`` enables auto-detection.
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

    # Priority
    priority_strategy: str = "hybrid"
    hybrid_detections: int = 2
    hybrid_exposure_s: float = 300.0
    lc_detections: int = 5
    lc_exposure_s: float = 300.0
    lc_phase_range: tuple[float, float] = (-7.0, 20.0)

    # CSV columns (Optional: set to None for auto-detect)
    ra_col: Optional[str] = None
    dec_col: Optional[str] = None
    disc_col: Optional[str] = None
    name_col: Optional[str] = None
    type_col: Optional[str] = None
