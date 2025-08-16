"""Configuration definitions for the twilight planner.

This module exposes :class:`PlannerConfig`, a dataclass collecting the many
parameters used by the scheduler.  The defaults are tailored for the LSST site
at Cerro Pach\u00f3n and can be customised for testing purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


@dataclass
class PlannerConfig:
    """Container of user‑tunable parameters for twilight planning.

    Only a subset of the fields are typically supplied by a user; the rest
    carry defaults that reflect LSST design values.  The dataclass is purposely
    lightweight so it can be serialised in tests and debugging sessions.
    Optional ``sun_alt_exposure_ladder`` entries allow shorter exposures in
    bright twilight, overriding ``exposure_by_filter`` within the specified Sun
    altitude ranges.  The scheduler populates transient fields such as
    ``current_mag_by_filter``, ``current_alt_deg``, and ``current_mjd`` for
    per-target exposure capping; users normally leave these as ``None``.  Minimum
    required time between consecutive exposures (idle 'guard' time is inserted if
    natural overhead < this value).
    """

    # -- Site ---------------------------------------------------------------
    lat_deg: float = -30.2446
    lon_deg: float = -70.7494
    height_m: float = 2647.0

    # -- Visibility --------------------------------------------------------
    min_alt_deg: float = 30.0
    twilight_sun_alt_min_deg: float = -18.0
    twilight_sun_alt_max_deg: float = 0.0

    # -- Filters and hardware ---------------------------------------------
    filters: List[str] = field(default_factory=lambda: ["g", "r", "i", "z", "y"])
    carousel_capacity: int = 5
    filter_change_s: float = 120.0
    readout_s: float = 2.0
    inter_exposure_min_s: float = 15.0
    # Legacy argument names supported via __post_init__
    filter_change_time_s: float | None = None
    readout_time_s: float | None = None
    exposure_by_filter: Dict[str, float] = field(
        default_factory=lambda: {
            "u": 30.0,
            "g": 15.0,
            "r": 15.0,
            "i": 15.0,
            "z": 15.0,
            "y": 15.0,
        }
    )
    max_filters_per_visit: int = 1
    start_filter: str | None = None
    sun_alt_policy: List[Tuple[float, float, List[str]]] = field(
        default_factory=lambda: [
            (-18.0, -15.0, ["y", "z", "i"]),
            (-15.0, -12.0, ["z", "i", "r"]),
            (-12.0, 0.0, ["i", "z", "y"]),
        ]
    )
    # Exposure overrides per Sun-altitude range; later entries take precedence
    sun_alt_exposure_ladder: List[Tuple[float, float, Dict[str, float]]] = field(
        default_factory=list
    )

    # -- Slew model --------------------------------------------------------
    slew_small_deg: float = 3.5
    slew_small_time_s: float = 4.0
    slew_rate_deg_per_s: float = 5.25
    slew_settle_s: float = 1.0

    # -- Moon --------------------------------------------------------------
    min_moon_sep_by_filter: Dict[str, float] = field(
        default_factory=lambda: {
            "u": 80.0,
            "g": 50.0,
            "r": 35.0,
            "i": 30.0,
            "z": 25.0,
            "y": 20.0,
        }
    )
    require_single_time_for_all_filters: bool = True

    # -- Time caps ---------------------------------------------------------
    per_sn_cap_s: float = 600.0
    morning_cap_s: float | Literal["auto"] = "auto"
    evening_cap_s: float | Literal["auto"] = "auto"
    morning_twilight: str | None = None
    evening_twilight: str | None = None
    twilight_step_min: int = 2
    max_sn_per_night: int = 20

    # -- Priority tracking -------------------------------------------------
    hybrid_detections: int = 2
    hybrid_exposure_s: float = 300.0
    lc_detections: int = 5
    lc_exposure_s: float = 300.0
    priority_strategy: str = "hybrid"

    # -- Photometry / sky --------------------------------------------------
    pixel_scale_arcsec: float = 0.2
    zpt1s: Dict[str, float] | None = None
    k_m: Dict[str, float] | None = None
    fwhm_eff: Dict[str, float] | None = None
    read_noise_e: float = 6.0
    gain_e_per_adu: float = 1.0
    zpt_err_mag: float = 0.01
    dark_sky_mag: Dict[str, float] | None = None
    twilight_delta_mag: float = 2.5

    # -- SIMLIB ------------------------------------------------------------
    simlib_out: str | None = None
    simlib_survey: str = "LSST"
    simlib_filters: str = "grizy"
    simlib_pixsize: float = 0.2
    simlib_npe_pixel_saturate: float = 100_000.0
    simlib_photflag_saturate: int = 4096
    simlib_psf_unit: str = "arcsec"

    # -- Miscellaneous -----------------------------------------------------
    typical_days_by_type: Dict[str, int] = field(
        default_factory=lambda: {
            "Ia": 70,
            "II-P": 100,
            "II-L": 70,
            "IIn": 120,
            "IIb": 70,
            "Ib": 60,
            "Ic": 60,
        }
    )
    default_typical_days: int = 60

    ra_col: Optional[str] = None
    dec_col: Optional[str] = None
    disc_col: Optional[str] = None
    name_col: Optional[str] = None
    type_col: Optional[str] = None

    # Hooks filled in by the scheduler for per-target context ----------------
    current_mag_by_filter: Optional[Dict[str, float]] = None
    current_alt_deg: Optional[float] = None
    current_mjd: Optional[float] = None  # populated by scheduler for exposure capping
    sky_provider: Optional[object] = None

    # Backwards-compatibility options
    allow_filter_changes_in_twilight: bool = False

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.filter_change_time_s is not None:
            self.filter_change_s = self.filter_change_time_s
        if self.readout_time_s is not None:
            self.readout_s = self.readout_time_s
        if self.start_filter is None and self.filters:
            self.start_filter = self.filters[0]
