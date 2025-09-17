"""Configuration definitions for the twilight planner.

This module exposes :class:`PlannerConfig`, a dataclass collecting the many
parameters used by the scheduler.  The defaults are tailored for the LSST site
at Cerro Pach\u00f3n and can be customised for testing purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
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

    # Whether to evaluate filter viability using Sun altitude at each target's
    # best_time_utc (more permissive early/late in the window) instead of the
    # window midpoint Sun altitude. Defaults to midpoint for backwards-compat.
    filter_policy_use_best_time_alt: bool = False

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

    # Optional manual overrides of twilight windows (local clock). If provided,
    # these take precedence over computed twilight. Example format: "HH:MM-HH:MM".
    # (They are pass-through strings; parsing/validation should happen where used.)
    morning_twilight: str | None = None
    evening_twilight: str | None = None

    twilight_step_min: int = 2
    max_sn_per_night: float | int = inf

    # -- Priority tracking -------------------------------------------------
    hybrid_detections: int = 2
    hybrid_exposure_s: float = 300.0
    lc_detections: int = 5
    lc_exposure_s: float = 300.0
    priority_strategy: str = "hybrid"
    unique_first_fill_with_color: bool = True
    unique_lookback_days: float = 999.0
    unique_first_drop_threshold: float = 0.0
    """Drop `unique_first` rows with score <= this threshold."""
    unique_first_resume_score: float = 0.0
    """Score used after lookback days if repeats are allowed again."""

    # -- Cadence ----------------------------------------------------------
    cadence_enable: bool = True
    """Enable cadence gating and bonus calculations."""

    cadence_per_filter: bool = True
    """If ``True``, track cadence separately for each filter."""

    cadence_days_target: float = 3.0
    """Target days between revisits in a given filter."""

    cadence_jitter_days: float = 0.25
    """Early revisit allowance below target days."""

    cadence_days_tolerance: float = 0.5
    """Tolerance window for cadence KPI calculations."""

    cadence_bonus_sigma_days: float = 0.5
    """Gaussian width (days) for the due-soon bonus."""

    cadence_bonus_weight: float = 0.25
    """Weight applied to the cadence bonus when ordering filters."""

    cadence_first_epoch_bonus_weight: float = 0.0
    """Bonus for a never-before-seen filter in cadence_bonus (0.0 = none)."""

    # -- Cosmology / colour tracking --------------------------------------
    cosmo_weight_by_filter: Dict[str, float] = field(
        default_factory=lambda: {"g": 1.25, "r": 1.10, "i": 1.0, "z": 0.85, "y": 0.60}
    )
    color_window_days: float = 5.0
    color_target_pairs: int = 2
    color_alpha: float = 0.3
    swap_cost_scale_color: float = 0.6
    swap_amortize_min: int = 6
    palette_rotation_days: int = 4
    palette_evening: List[str] = field(default_factory=lambda: ["i", "r", "z", "i"])
    palette_morning: List[str] = field(default_factory=lambda: ["r", "g", "i", "r"])
    max_swaps_per_window: int = 2
    first_epoch_color_boost: float = 1.5

    # -- Redshift prioritization -----------------------------------------
    redshift_boost_enable: bool = True
    """If True, apply a mild boost to low-redshift SNe in hybrid strategy."""
    redshift_low_ref: float = 0.1
    """Reference z below which boost ramps up linearly toward max."""
    redshift_boost_max: float = 1.2
    """Maximum multiplicative boost at z=0 (≥1)."""
    redshift_column: Optional[str] = None
    """Optional explicit column name for redshift in the input catalog."""

    # -- Low-z Ia special handling ----------------------------------------
    # Disabled by default to preserve baseline/test behaviour; set these to
    # enable stronger preference and tailored cadence/repeat policy for
    # Ia-like objects at very low redshift.
    low_z_ia_markers: List[str] = field(default_factory=lambda: ["ia", "1", "101"])
    """Case-insensitive markers identifying Type Ia (e.g., 'Ia', '1', '101')."""
    low_z_ia_z_threshold: float = 0.05
    """Redshift threshold defining "low-z" for special Ia handling."""
    low_z_ia_priority_multiplier: float = 1.0
    """Multiply base priority for low-z Ia (1.0 = disabled)."""
    low_z_ia_cadence_days_target: Optional[float] = None
    """Optional per-target cadence target (days) for low-z Ia; None = global."""
    low_z_ia_repeats_per_window: Optional[int] = None
    """Optional max repeats per twilight window for low-z Ia (default 1)."""

    # -- Backfill relax cadence -------------------------------------------
    backfill_relax_cadence: bool = False
    """If True, allow a last-resort backfill that ignores cadence gating
    when time would otherwise go unused in a window. Swap and other
    constraints still apply; only triggered after normal backfill/repeats.
    """

    # -- Catalog pre-filtering --------------------------------------------
    only_ia: bool = False
    """If True, restrict the input catalog to Ia-like types only.

    Matching is case-insensitive and uses a simple substring rule for 'Ia':
    any SN type whose normalized string contains 'ia' (e.g., 'SN Ia',
    'Ia-91bg', 'Iax', 'SNIa?') is kept. Other types like 'II', 'Ib', 'Ic',
    'IIn' are excluded.
    """

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

    # Optional per-target host galaxy context (used in saturation capping)
    # If available, provide either observed-frame surface brightness at the SN
    # location per filter, or rest-frame SB plus a redshift (and optional K-corr).
    current_host_mu_arcsec2_by_filter: Optional[Dict[str, float]] = None
    current_host_mu_rest_arcsec2_by_filter: Optional[Dict[str, float]] = None
    current_host_z: Optional[float] = None
    current_host_K_by_filter: Optional[Dict[str, float]] = None
    # Optional compact host knot approximated as a point-like component
    current_host_point_mag_by_filter: Optional[Dict[str, float]] = None
    current_host_point_frac: Optional[float] = None

    # Default host SB fallback when per-target host inputs are missing.
    # Based on literature: r-band typical µ_host ~ 21–23 mag/arcsec^2.
    # We default to 22.0 mag/arcsec^2 across filters as a conservative mid-point.
    use_default_host_sb: bool = True
    default_host_mu_arcsec2_by_filter: Dict[str, float] = field(
        default_factory=lambda: {"u": 22.0, "g": 22.0, "r": 22.0, "i": 22.0, "z": 22.0, "y": 22.0}
    )

    # Backwards-compatibility options
    allow_filter_changes_in_twilight: bool = False

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.filter_change_time_s is not None:
            self.filter_change_s = self.filter_change_time_s
        if self.readout_time_s is not None:
            self.readout_s = self.readout_time_s
        if self.start_filter is None and self.filters:
            self.start_filter = self.filters[0]
