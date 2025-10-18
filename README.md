# LSST Twilight Supernova Planner

Tools for planning LSST twilight observations of supernovae and
experimenting with dynamic priority strategies.

## Repository layout

- `twilight_planner_pkg/` – core Python package with the scheduler and
  priority logic.
- `data/` – example input tables.
- `notebook/` – Jupyter notebooks demonstrating the planner.
- `twilight_outputs/` – sample planning results.

Run the CLI via the package module:

- `python -m twilight_planner_pkg.main` — command‑line planner entry point.

The package is built in a modular fashion: separate modules handle
configuration, astronomy utilities, per‑SN priority tracking, and the
scheduler itself.  This structure makes it easy to swap in new
strategies or extend the planner for different surveys.

Recent additions provide a Rubin‑style photometry model and a minimal
SNANA SIMLIB writer. Planned exposures are capped to avoid pixel
full‑well saturation by summing the central‑pixel electrons from the SN
point source, local host‑galaxy surface brightness (per pixel), and sky
background (per pixel). Optional inputs support rest‑frame host SB with
Tolman dimming and an optional compact host knot. Setting ``--simlib-out``
on the CLI writes a SIMLIB file alongside the usual planning CSVs.

When per-band source magnitudes are missing, the planner can fall back to a
catalog `discoverymag` to estimate source brightness for saturation guard
(enabled by default; see `PlannerConfig.use_discovery_fallback`). The fallback
now also leverages redshift information: a lightweight ΛCDM distance-modulus
calculator plus a conservative Ia peak-standardization is used to derive
`z→peak` magnitudes, and the discovery-based and redshift-based estimates are
merged by taking the brighter (smaller) magnitude per filter. Configuration
knobs such as `PlannerConfig.peak_extra_bright_margin_mag`,
`PlannerConfig.Kcorr_approx_mag`, and cosmology parameters allow tuning the
safety margin. If neither discovery nor redshift information yields a per-band
estimate for a target, the code raises a `ValueError`
(`PlannerConfig.discovery_error_on_missing=True`) to avoid silently skipping
saturation protection.

Rubin photometry defaults now adopt a gain of 1.6 e⁻/ADU in both
`PlannerConfig` and `PhotomConfig`, which keeps conversions between ADU and
electrons consistent with LSST camera specifications while feeding the
saturation guard.

See `twilight_planner_pkg/README.md` for detailed usage instructions and
module documentation.

## Twilight Strategy Highlights

The planner follows a Sun‑altitude policy inspired by twilight brightness.
Redder filters cope better with bright twilight. The default mapping in
`PlannerConfig.sun_alt_policy` is:

| Sun altitude (deg) | Allowed filters |
|--------------------|-----------------|
| -18 to -15         | y, z, i         |
| -15 to -12         | z, i, r         |
| -12 to 0           | i, z, y         |

This mapping is configurable, and feasibility is further pruned by the
per‑band m₅/SNR gate (see below). There are no hard‑coded bans in the gating
logic; the final set used in a window is the intersection of feasibility with
the Sun‑alt policy.

### Band‑diversity mode (per‑filter balance)

To avoid starving specific bands (e.g. too few g visits), the planner can
prioritise under‑observed individual bands instead of only “opposite colour”
pairs. Enable via:

- `PlannerConfig.diversity_enable=True`
- `PlannerConfig.diversity_target_per_filter` (e.g. 1)
- `PlannerConfig.diversity_window_days` (e.g. 5)
- `PlannerConfig.diversity_alpha` (boost strength)

This mode keeps per‑filter cadence gating intact; the bonus only ranks among
filters that already pass the gate.

### First‑filter cycle (per window)

You can explicitly alternate the first batch’s filter by window/day:

- Morning cycles through `["g", "r"]`
- Evening cycles through `["z", "i"]`

Turn on with `PlannerConfig.first_filter_cycle_enable=True`. The selected
filter is placed at the head of the per‑window batch order (other filters
follow in the configured palette order). This helps ensure all bands get time
in short twilight windows while still respecting swap limits and cadence.

LSST's filter carousel can host at most five filters per night. Cross-target
swaps incur a 120 s cost, amortized across same-filter batches and scaled down
when the new filter supplies a missing color. Palette rotation (separate
evening and morning cycles) and a per-window swap cap further discourage
unnecessary filter changes. Readout time is 2 s per exposure and slews follow a
hybrid model (``3.5° in 4 s`` plus ``5.25°/s``).

An inter‑exposure guard of 15 s is enforced. If the natural overhead
(max of slew vs readout plus any cross‑filter change) is shorter than 15 s,
idle time is inserted before the next exposure; this guard is accounted for in
window cap checks and reported in summaries.

### DP filter planning (global SN/filter pool)

Within each twilight window the planner now builds a **global pool of
(supernova, filter)** candidates that pass the Sun-alt, Moon, cadence, and
m₅/SNR feasibility gates.  Guard-aware visit durations combine the 15 s
inter-exposure minimum, slew + settle time, and saturation-capped exposure
(`compute_capped_exptime`).  A compact dynamic program allocates those visits
across filters while charging the full **120 s Rubin filter-change penalty**.
The DP works in “visit units”, choosing how many batches to execute in each
filter so that only swaps with sufficient science payoff survive.  Key tuning
knobs in `PlannerConfig`:

- `swap_boost` — slight <1 multiplier applied to post-swap batches (default 0.95).
- `dp_hysteresis_theta` — require a % improvement before accepting a swap plan.
- `n_estimate_mode` — fast visit-count mode (`"guard_plus_exp"`) or per-filter visit units.
- `dp_max_swaps` — optional hard cap for the DP (defaults to `max_swaps_per_window`).
- `min_batch_payoff_s` — minimum wall-clock payoff needed to justify a swap (defaults to `filter_change_s` if unset).
- `dp_time_mode` (experimental) — enable a time-budget DP instead of visit counts.

- `swap_cost_scale_color` — color-aware scaling of swap costs.
- `swap_amortize_min` — amortize swap cost for sufficiently long runs.
- `policy_sun_alt_minutes` — minute-binned cache resolution for Sun altitude.
- `pairs_topk_per_filter` — prune candidate (SN, filter) pairs per band to cap combinatorics.
- `debug_planner` — emit structured diagnostics for DP plan, pair counts, and execution results.

The resulting filter sequence and per-filter visit counts drive the existing
execution loop, which still routes within a batch by the usual score density
heuristic (score divided by guard-aware visit cost) and honors cadence gates.

Moon–target separations use Astropy's `get_body('moon')` in a shared AltAz
frame. If the Moon is below the horizon, the separation requirement is
automatically waived.

### Cadence constraint (per-filter)

Each filter maintains its own last-observed MJD. Repeats in the same band are
gated until `cadence_days_target - cadence_jitter_days` days have elapsed.
Filters nearing their due date receive a Gaussian "due-soon" bonus so the
scheduler prefers them. Different filters may still be taken within a single
visit because cadence is tracked per filter.

Airmass calculations adopt the "simple" formula of Kasten & Young (1989), and
the overhead values above follow Rubin Observatory technical notes.

### Filter feasibility (m5 / SNR gate)

- m5 scaling: LSST-style 5σ depth that responds to sky brightness, seeing,
  airmass, and the **per-filter exposure time** currently in force (including
  any `sun_alt_exposure_ladder` overrides).
- Read-noise correction `ΔC_m(τ)` is applied using the standard τ scaling so
  short (e.g. 5 s) visits lose depth appropriately.
- Sky brightness: when `rubin_sim.skybrightness` is available the exact Rubin
  model is used; otherwise a fallback combines a twilight term with a
  Krisciunas & Schaefer (1991) moon-scatter model.
- Gate: a band is eligible when `m5 ≥ m_target` (SNR ≥ 5). The final decision
  is the intersection with `sun_alt_policy`. If no band passes the m5/SNR gate,
  an empty set is returned—there is no heuristic fallback. Per‑filter Moon
  separations are checked afterwards using `min_moon_sep_by_filter`.

## Minimal example

```python
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps

cfg = PlannerConfig(filters=["i", "z"], start_filter="i")
plan_twilight_range_with_caps(
    "my_catalog.csv",
    "out",
    "2024-01-01",
    "2024-01-01",
    cfg,
    run_label="hybrid_strategy",
)
```

The repository includes `data/demo_three_targets.csv` illustrating a simple
three-target run.

If ``run_label`` is omitted, the planner uses ``"hybrid"`` (the default
priority strategy) and filenames like
``lsst_twilight_plan_hybrid_<start>_to_<end>.csv`` are produced.

## Outputs

The per-SN planning CSV now includes an `sn_end_utc` column giving the
end of each visit (start time plus total scheduled duration).

The night/window summary CSV now includes the following columns capturing
twilight timing and basic science metrics:

- `window_start_utc`, `window_end_utc`, `window_duration_s`, `window_mid_utc`
- `sun_alt_mid_deg`, `policy_filters_mid_csv`
- `window_utilization`, `cap_utilization`, `cap_source`
- `median_sky_mag_arcsec2`, `median_alt_deg`
- `cad_median_abs_err_by_filter_csv`, `cad_within_pct_by_filter_csv`
- `cad_median_abs_err_all_d`, `cad_within_pct_all`

`window_cap_s` records the effective limit on scheduled time in each twilight
window. It comes from `PlannerConfig.morning_cap_s` or `PlannerConfig.evening_cap_s`.
When these are set to "auto" (the default), the cap equals the true duration of
each window; otherwise a fixed number of seconds is used.

For the true, serialized on‑sky order within each twilight window, use the
sequence file:

- `lsst_twilight_sequence_true_<run_label>_<start>_to_<end>.csv` — non‑overlapping
  execution order with `order_in_window`, `sn_start_utc`, `sn_end_utc`, and
  `filters_used_csv`. One row per SN visit (multi‑filter visits are a single row).

## Installation

```bash
pip install -r requirements.txt
```
