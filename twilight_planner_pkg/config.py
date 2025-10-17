"""Configuration definitions for the twilight planner.

This module exposes :class:`PlannerConfig`, a dataclass collecting the many
parameters used by the scheduler.  The defaults are tailored for the LSST site
at Cerro Pach\u00f3n and can be customised for testing purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from math import inf
from typing import Any, Dict, List, Literal, Optional, Tuple, Callable


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
    filters_per_visit_cap: int = 1
    """Upper bound on distinct filters used within a single visit."""
    auto_color_pairing: bool = True
    """If True, automatically add an opposite-colour filter when the cap allows."""
    start_filter: str | None = None
    # Optional override of the first-filter priority order. If provided, this
    # list is used (in order) as the preference when choosing the first filter
    # for a target; any remaining bands fall back to a default red-to-blue
    # ordering. Leave as None to use the built-in default.
    first_filter_order: Optional[List[str]] = None
    # Optional per-filter weights applied when ranking first-filter candidates
    # via the cadence/diversity bonus. Values >1.0 favour a filter; values <1.0
    # down-weight it. When ``None`` the default weight of 1.0 is used for all
    # filters and the historical ordering applies.
    first_filter_bonus_weights: Optional[Dict[str, float]] = None
    # Optional user-supplied hook to select the first filter. When set to a
    # callable, it is invoked as
    #   fn(name, allowed_filters, cfg, context={...}) → filter | None
    # The context dictionary includes keys: 'tracker', 'sun_alt_deg',
    # 'moon_sep_ok', 'current_mag', and 'current_filter'. If the hook raises
    # or returns an invalid band, the scheduler falls back to the default
    # policy.
    pick_first_filter: Optional[Callable[..., Optional[str]]] = None
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
    # Base palettes (tie-breakers). First-filter cycling can override the head.
    palette_evening: List[str] = field(default_factory=lambda: ["z", "i", "r", "g"])
    palette_morning: List[str] = field(default_factory=lambda: ["g", "r", "i", "z"])
    max_swaps_per_window: int = 2
    first_epoch_color_boost: float = 1.5
    swap_boost: float = 0.95
    dp_time_mode: bool = False
    n_estimate_mode: Literal["guard_plus_exp", "per_filter"] = "guard_plus_exp"
    dp_hysteresis_theta: float = 0.02
    dp_max_swaps: Optional[int] = None
    min_batch_payoff_s: Optional[float] = None
    # In DP hard-batch execution, treat only the first segment as already
    # positioned at its filter (no initial cross-filter swap cost). Subsequent
    # DP segments will incur normal swap cost based on carousel state.
    # Default False preserves legacy behaviour where every DP segment suppressed
    # cross-filter cost.
    dp_free_first_swap_only: bool = False

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

    # -- Band diversity (per-filter) ----------------------------------------
    diversity_enable: bool = True
    """If True, use per-filter diversity bonus instead of blue/red color boost.

    Encourages under-observed individual bands (g, r, i, z) rather than only
    favoring the opposite color group. Cadence gating remains per filter.
    """
    diversity_target_per_filter: int = 1
    """Target number of visits per filter within ``diversity_window_days``.

    The diversity multiplier scales with the shortfall relative to this target.
    """
    diversity_window_days: float = 5.0
    """Time window (days) for counting per-filter visits in diversity mode."""
    diversity_alpha: float = 0.3
    """Strength of the diversity multiplier (≥0)."""

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
    gain_e_per_adu: float = 1.6
    zpt_err_mag: float = 0.01
    dark_sky_mag: Dict[str, float] | None = None
    twilight_delta_mag: float = 2.5

    # -- Window first-filter cycle ------------------------------------------
    first_filter_cycle_enable: bool = True
    """If True, enforce a day-by-day first-filter alternation per window.

    Morning windows alternate ``first_filter_cycle_morning``; evening
    windows alternate ``first_filter_cycle_evening``. The chosen first filter
    is placed at the head of the per-window filter batch order.
    """
    first_filter_cycle_morning: List[str] = field(default_factory=lambda: ["g", "r"])
    first_filter_cycle_evening: List[str] = field(default_factory=lambda: ["z", "i"])

    # -- Cosmology / peak-magnitude guardrails ----------------------------
    H0_km_s_Mpc: float = 70.0
    Omega_m: float = 0.3
    Omega_L: float = 0.7
    MB_absolute: float = -19.36
    SALT2_alpha: float = 0.14
    SALT2_beta: float = 3.1
    Kcorr_approx_mag: float = 0.0
    Kcorr_approx_mag_by_filter: Dict[str, float] = field(default_factory=dict)
    peak_extra_bright_margin_mag: float = 0.3

    # -- SIMLIB ------------------------------------------------------------
    simlib_out: str | None = None
    simlib_survey: str = "LSST"
    simlib_filters: str = "grizy"
    simlib_pixsize: float = 0.2
    simlib_npe_pixel_saturate: float = 80_000.0
    simlib_photflag_saturate: int = 2048
    simlib_psf_unit: str = "PIXEL"

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
    current_redshift: Optional[float] = None
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
    # Rest-frame values are dimmed via Tolman + a linear K-term when a
    # redshift is available. Observed-frame overrides remain configurable for
    # backwards compatibility.
    use_default_host_sb: bool = True
    default_host_mu_arcsec2_by_filter: Dict[str, float] = field(
        default_factory=lambda: {
            "u": 22.0,
            "g": 22.0,
            "r": 22.0,
            "i": 22.0,
            "z": 22.0,
            "y": 22.0,
        }
    )
    default_host_mu_rest_arcsec2_by_filter: Dict[str, float] = field(
        default_factory=lambda: {
            "u": 22.8,
            "g": 22.2,
            "r": 21.7,
            "i": 21.5,
            "z": 21.4,
            "y": 21.3,
        }
    )
    default_host_kcorr_slope_by_filter: Dict[str, float] = field(
        default_factory=lambda: {
            "u": 0.35,
            "g": 0.30,
            "r": 0.25,
            "i": 0.20,
            "z": 0.20,
            "y": 0.15,
        }
    )

    # Backwards-compatibility options
    allow_filter_changes_in_twilight: bool = False

    # -- Discovery magnitude fallback (for saturation capping) ---------------
    use_discovery_fallback: bool = True
    """If True and the input catalog contains a discovery magnitude column
    (e.g., ``discoverymag``), build per-band source magnitude estimates to
    drive saturation capping when explicit per-band mags are missing."""

    discovery_policy: str = "atlas_priors"
    """Fallback policy: ``"atlas_transform"`` converts ATLAS ``cyan``/``orange``
    to r-band (using :math:`g-r` assumption) and copies to other bands with a
    small safety margin; ``"copy"`` copies discovery mag into all bands with
    a small margin (more conservative)."""

    discovery_assumed_gr: float = 0.0
    """Assumed :math:`g-r` color used when converting ATLAS ``c``/``o`` to r."""

    discovery_margin_mag: float = 0.2
    """Safety margin (mag) subtracted when copying mags to other bands so the
    fallback treats the source as slightly brighter, avoiding saturation."""

    # Color prior ranges for discovery fallback (mag). Keys are color names.
    discovery_color_priors_min: Dict[str, float] = field(
        default_factory=lambda: {
            "u-g": 0.3,
            "g-r": -0.25,
            "r-i": -0.15,
            "i-z": -0.10,
            "z-y": -0.05,
        }
    )
    discovery_color_priors_max: Dict[str, float] = field(
        default_factory=lambda: {
            "u-g": 1.0,
            "g-r": 0.15,
            "r-i": 0.25,
            "i-z": 0.20,
            "z-y": 0.30,
        }
    )
    discovery_non_ia_widen_mag: float = 0.1
    """Widen color prior extreme by this amount for non‑Ia or unknown types.
    Applied on the chosen extreme only, in the direction that brightens the
    target band for saturation safety."""

    discovery_y_extra_margin_mag: float = 0.25
    """Additional safety margin (mag) applied when extrapolating to y band."""

    # Optional ATLAS c/o → r linear coefficients (alpha + beta*(g-r)).
    discovery_atlas_linear: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "c": {"alpha": 0.0, "beta": -0.47},
            "o": {"alpha": 0.0, "beta": 0.26},
        }
    )

    discovery_error_on_missing: bool = True
    """If True, raise an error when discovery fallback fails to provide a
    magnitude for any planned band and target in the input catalog. This ensures
    the saturation guard always has a source magnitude and never falls back to
    a baseline exposure silently."""

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.filter_change_time_s is not None:
            self.filter_change_s = self.filter_change_time_s
        if self.readout_time_s is not None:
            self.readout_s = self.readout_time_s
        if self.start_filter is None and self.filters:
            self.start_filter = self.filters[0]
        try:
            self.filters_per_visit_cap = int(self.filters_per_visit_cap)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError("filters_per_visit_cap must be coercible to int") from exc
        if self.filters_per_visit_cap < 1:
            raise ValueError("filters_per_visit_cap must be >= 1")

        def _norm_filter_name(value: Any) -> str | None:
            """Return a canonical, lowercase filter name or ``None`` if invalid."""

            if value is None:
                return None
            try:
                stripped = str(value).strip()
            except Exception:
                return None
            if not stripped:
                return None
            return stripped.lower()

        def _norm_filter_list(seq: Any) -> list[str]:
            result: list[str] = []
            if not seq:
                return result
            for item in seq:
                name = _norm_filter_name(item)
                if name is None:
                    continue
                result.append(name)
            return result

        def _norm_filter_dict(mapping: Any) -> dict[str, Any]:
            result: dict[str, Any] = {}
            if not mapping:
                return result
            for key, val in mapping.items():
                name = _norm_filter_name(key)
                if name is None:
                    continue
                result[name] = val
            return result

        # Normalise core filter configuration to lowercase to keep scheduler internals
        # case-insensitive while still permitting uppercase inputs in notebooks.
        self.filters = _norm_filter_list(self.filters)

        if self.start_filter is not None:
            start_norm = _norm_filter_name(self.start_filter)
            self.start_filter = start_norm
        if self.start_filter and self.start_filter not in self.filters:
            self.filters.insert(0, self.start_filter)

        self.exposure_by_filter = _norm_filter_dict(self.exposure_by_filter)
        self.min_moon_sep_by_filter = _norm_filter_dict(self.min_moon_sep_by_filter)
        self.cosmo_weight_by_filter = _norm_filter_dict(self.cosmo_weight_by_filter)
        self.default_host_mu_arcsec2_by_filter = _norm_filter_dict(
            self.default_host_mu_arcsec2_by_filter
        )
        self.default_host_mu_rest_arcsec2_by_filter = _norm_filter_dict(
            self.default_host_mu_rest_arcsec2_by_filter
        )
        self.default_host_kcorr_slope_by_filter = _norm_filter_dict(
            self.default_host_kcorr_slope_by_filter
        )
        self.Kcorr_approx_mag_by_filter = _norm_filter_dict(
            getattr(self, "Kcorr_approx_mag_by_filter", {})
        )
        self.palette_evening = _norm_filter_list(self.palette_evening)
        self.palette_morning = _norm_filter_list(self.palette_morning)
        # Normalise user-specified first-filter order if provided
        ffo = getattr(self, "first_filter_order", None)
        ffo_norm = _norm_filter_list(ffo) if ffo else []
        self.first_filter_order = ffo_norm or None
        # Normalise per-filter weight mapping (if provided)
        ffw = getattr(self, "first_filter_bonus_weights", None)
        ffw_norm = _norm_filter_dict(ffw) if ffw else {}
        cleaned_weights: dict[str, float] = {}
        for key, val in ffw_norm.items():
            try:
                cleaned_weights[key] = float(val)
            except Exception:
                continue
        self.first_filter_bonus_weights = cleaned_weights or None

        policy_norm: list[tuple[float, float, list[str]]] = []
        for low, high, flist in self.sun_alt_policy:
            policy_norm.append((low, high, _norm_filter_list(flist)))
        self.sun_alt_policy = policy_norm


_original_planner_config_init = PlannerConfig.__init__


def _planner_config_init_wrapper(self, *args, **kwargs):
    if "max_filters_per_visit" in kwargs and "filters_per_visit_cap" not in kwargs:
        warnings.warn(
            "max_filters_per_visit is deprecated; use filters_per_visit_cap",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs["filters_per_visit_cap"] = kwargs.pop("max_filters_per_visit")
    return _original_planner_config_init(self, *args, **kwargs)


PlannerConfig.__init__ = _planner_config_init_wrapper  # type: ignore[attr-defined]


@property
def _deprecated_max_filters_per_visit(self) -> int:
    warnings.warn(
        "max_filters_per_visit is deprecated; use filters_per_visit_cap",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.filters_per_visit_cap


@_deprecated_max_filters_per_visit.setter  # type: ignore[attr-defined]
def _deprecated_max_filters_per_visit_set(self, value: int) -> None:
    warnings.warn(
        "max_filters_per_visit is deprecated; use filters_per_visit_cap",
        DeprecationWarning,
        stacklevel=2,
    )
    self.filters_per_visit_cap = int(value)


PlannerConfig.max_filters_per_visit = _deprecated_max_filters_per_visit  # type: ignore[assignment]
