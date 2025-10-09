# LSST Twilight Supernova Planner — Example Notebook

This README provides a high-level overview of how the twilight planner operates and how the notebook parameters influence its behaviour.

---

## Planner Overview

The planner schedules supernova (SN) observations during astronomical twilight by balancing:

- **Science constraints**: target altitude, Moon separation (scaled by Moon altitude/phase), twilight sky brightness, and typical post-discovery visibility windows by SN type.
- **Engineering constraints**: slew and settle times, readout, filter-change overheads, carousel capacity, and per-window time caps.
- **Strategy**: hybrid priority scheme ("quick color → escalate" or light-curve depth), batching by the first filter with a dynamic batch order (filters with more first‑filter demand go first; palette provides tie‑breaks), and greedy routing to minimize combined slew and filter-change cost.
- **Guard spacing**: a minimum inter-exposure spacing of 15 s is enforced. Readout overlaps with slewing, so the natural inter-visit gap is max(slew, readout) + cross-filter-change. If natural overheads are shorter, idle guard time is inserted before the next exposure. Guard time is accounted for prior to window cap checks and reported in per-row and per-window summaries.

### Cadence constraint (per-filter)

- Revisit spacing is enforced **per filter**, not per SN.
- A same-band revisit is blocked until `cadence_days_target - cadence_jitter_days` days have
  elapsed since that band was last observed.
- First-time observations in a band always pass the gate, enabling quick colors.
- A Gaussian “due-soon” bonus nudges bands whose last visit is near the target cadence
  without hard-blocking other filters.
- Nightly summaries report per-filter cadence compliance via
  `cad_median_abs_err_by_filter_csv` and `cad_within_pct_by_filter_csv`, with overall
  aggregates `cad_median_abs_err_all_d` and `cad_within_pct_all`.

### Outputs

- Per-SN plan CSV (one row per scheduled visit). Represents the **best-in-theory** schedule keyed to each target's `best_time_utc`; times may overlap across different SNe and do not reflect the serialized on-sky order.
- True sequence CSV `lsst_twilight_sequence_true_<run_label>_<start>_to_<end>.csv` — **true, non-overlapping execution order** within each twilight window. Visits are packed as soon as the previous one ends (ignoring `best_time_utc` slack); `preferred_best_utc` records the original preference. Columns include `order_in_window`, `sn_start_utc`, `sn_end_utc`, and `filters_used_csv`. One row per SN visit (multi-filter visits are a single row).
- Per-night summary CSV (morning/evening window statistics).
- Optional SNANA SIMLIB file for simulations.

---

## Main Steps & Logic

### 1. Initialization & Data Ingest
- **Prepare I/O and data**: create output directory, read the SN CSV, standardize column names (`standardize_columns`), and derive current estimated magnitudes (`extract_current_mags`) used for filter feasibility and saturation checks.
- **Photometry configuration**: build `PhotomConfig` (pixel scale, zeropoint for 1s, extinction coefficient, effective FWHM, read noise, gain, zeropoint error, pixel-saturation threshold). These drive sky noise, NEA, and saturation checks.
- **Observatory site**: set `EarthLocation` to Cerro Pachón (latitude −30.2446°, longitude −70.7494°, height 2663 m).
- **Sky brightness model**: try `RubinSkyProvider` (Sun-altitude aware); if unavailable, fall back to `SimpleSkyProvider` (dark-sky magnitude with a twilight brightening offset).
- **SIMLIB (optional)**: if `cfg.simlib_out` is set, write the SIMLIB header and prepare to append per-epoch entries (MJD, band, GAIN, RDNOISE, SKYSIG, NEA, zeropoint, etc.).
- **Carousel capacity check**: if `cfg.filters` exceeds `cfg.carousel_capacity` (here 5), drop one filter (typically `u` or the last) and warn. Remaining scheduling only considers loaded filters.

### 2. Night‑by‑Night Loop
For each UTC date in `START_DATE` … `END_DATE`:

1. **Find astronomical twilight windows**: compute morning/evening intervals where the Sun altitude satisfies `cfg.twilight_sun_alt_min_deg` < alt < `cfg.twilight_sun_alt_max_deg` (defaults −18° < alt < 0°) (`twilight_windows_astro`).
2. **Set a conservative Moon‑separation guard**: use the max of `MIN_MOON_SEP_BY_FILTER` over loaded filters as a baseline; automatically ignore the constraint when the Moon is below the horizon.
3. **Initialize window state**: for each window keep the current loaded filter (start with `START_FILTER`), filter-swap counters, and time caps (`MORNING_CAP_S` / `EVENING_CAP_S`).
4. **Filter eligible SNe by “lifetime”**: using discovery time and `TYPICAL_DAYS_BY_TYPE` (with a 1.2× safety factor via `parse_sn_type_to_window_days`), accept only SNe still in their typical observability window on this date.
5. **Best time per SN (Moon‑aware)**: for each candidate, sample the two twilight windows at `TWILIGHT_STEP_MIN` minutes (here 2 min); call `_best_time_with_moon` to get the maximum altitude time that also honors Moon separation (scaled by Moon altitude/phase). Keep the better of the two windows.
6. **Visible SN filter**: keep targets with a valid best time, max altitude ≥ `MIN_ALT_DEG` (here 20°), and a valid window index.
7. **Priorities & global selection**: score with `PriorityTracker` (hybrid or LC). Drop non‑positive scores under `unique_first`, sort by priority then max altitude, and cap per window using a stratified split of `MAX_SN_PER_NIGHT` (default infinity).

### 3. Scheduling Within Each Twilight Window
For each window index `idx_w` present that night:

1. **Sun‑altitude exposure ladder (optional)**: compute the window mid-time Sun altitude. If `cfg.sun_alt_exposure_ladder` is defined, temporarily override `cfg.exposure_by_filter` to shorten exposures at brighter twilight.
2. **Build candidates per target**:
   - Use `allowed_filters_for_window(...)` to get physically allowed filters combining current magnitude, Sun altitude, Moon geometry, airmass, and seeing.
   - Intersect with user-requested `FILTERS`.
   - Enforce Sun‑alt policy via `allowed_filters_for_sun_alt(sun_alt, cfg)` (strict: anything outside policy is dropped).
   - For each allowed filter, check Moon separation with `effective_min_sep` (dynamic with Moon altitude/phase).
   - Choose the first filter for this target via `pick_first_filter_for_target(...)`, which accounts for priority stage, Moon OK flags, current carousel state, and (if available) the target’s current magnitudes. Discard the target if none qualifies.
3. **Batch by first filter**: group targets by their chosen first filter. A rotating palette (`palette_evening`/`palette_morning` shifted by `palette_rotation_days`) sets the batch order; any remaining filters are appended by decreasing 5σ depth.
4. **Greedy routing with time‑packing under the window cap**:
   - Maintain `window_sum`, previous target `prev`, and current loaded filter state.
   - Within each batch, repeatedly pick the next target by minimizing slew time plus a cross-target filter-change penalty. The penalty is amortized over the batch size, scaled down if the new filter supplies a missing color, and skipped when the swap cap for the window is reached. (This “cost” is for ordering only; the booked time comes from the next step.)
   - For the chosen target:
     - Set `cfg.current_mag_by_filter`, `cfg.current_alt_deg`, `cfg.current_mjd` (context for saturation guard).
     - Build the visit’s filter list: `[first] +` other allowed (de-duplicated).
     - Call `choose_filters_with_cap(...)` to select a subset that fits `PER_SN_CAP_S` (here 120 s) and return a full timing breakdown: slew, readout, exposure, cross-target and internal filter-change times, plus any saturation/non-linearity flags (if capped exposures are used).
     - If adding this visit would exceed the window cap (`MORNING_CAP_S`/`EVENING_CAP_S`, each 600 s here), skip it; otherwise:
       - Accumulate the booked time; update total filter-change seconds for reporting.
       - For each filter used, compute sky brightness (Rubin or fallback model) and photometric products (`compute_epoch_photom`: zeropoint, `SKYSIG`, `NEA`, `RDNOISE`, `GAIN`).
       - Append a per-SN row with times, conditions, flags, priority, and time breakdown.
       - If SIMLIB is enabled, write `EPOCH` entries (MJD, band, `GAIN`, `RDNOISE`, `SKYSIG`, `NEA`, `ZP`, etc.).
       - Update state to the last used filter, bump internal filter-change counters, set `prev = target`, and record the detection/exposure to `PriorityTracker` (used for cross-night strategy).
5. **Window summary row**: after finishing a window, write a per-window summary: number of candidates vs. planned, total time vs. cap, filter-swap counts, average slew time, median airmass, requested vs. actually used filters, and color metrics (`color_pairs`, `color_diversity`, `%pair_different_filters`).
6. **Restore exposures if overridden**: if an exposure ladder was applied for this window, revert to the original `cfg.exposure_by_filter`.

### 4. Outputs
- **Per-SN CSV**: `lsst_twilight_plan_<run_label>_<start>_to_<end>.csv`
  One row per scheduled visit with date, window, best time, filter, `t_exp_s`, airmass/altitude, sky brightness, photometric terms (`ZPT`, `SKYSIG`, `NEA`, `RDNOISE`, `GAIN`), saturation/non-linear flags, priority, and time breakdown (slew/readout/exposure/filter changes/total). Represents the **best-in-theory** schedule keyed to each target's `best_time_utc`; times may overlap across different SNe and do not reflect the serialized on-sky order.
- **True sequence CSV**: `lsst_twilight_sequence_true_<run_label>_<start>_to_<end>.csv`
  **True, non-overlapping execution order** within each twilight window. Visits are packed as soon as the previous one ends (ignoring `best_time_utc` slack); the original preference is stored in `preferred_best_utc`. Columns include `order_in_window`, `sn_start_utc`, `sn_end_utc`, and `filters_used_csv`. One row per SN visit (multi-filter visits are a single row).
- **Per-night summary CSV**: `lsst_twilight_summary_<run_label>_<start>_to_<end>.csv`
  One row per window: candidate/planned counts, time usage vs. cap, swap counts, internal filter changes, color metrics, mean slew, median airmass, loaded filters, actually used filters.
- **SIMLIB (optional)**: if `SIMLIB_OUT` is set, a SNANA-compatible library with all `EPOCH`s is produced.

In all of the above, ``<run_label>`` defaults to ``hybrid`` matching the
default priority strategy.

---

## Band Diversity and First‑Filter Cycle

Short twilight windows can starve specific bands (typically g). The notebook’s configuration supports two complementary controls to keep bands balanced while preserving per‑filter cadence gating:

### Band‑diversity mode (per‑filter)
- Enable a per‑filter bonus that favors under‑observed bands within a recent window:
  - `diversity_enable=True`
  - `diversity_target_per_filter=1` (or 2)
  - `diversity_window_days=5.0`
  - `diversity_alpha=0.3` (strength of the multiplier)
- This does not bypass cadence gates; it only reorders eligible filters.
- In diversity mode, `first_epoch_color_boost` is interpreted as a “first epoch in this band” nudge.

### First‑filter cycle (per window/day)
- Alternate the first batch’s filter by window and day to ensure each band regularly leads a window when time is tight:
  - `first_filter_cycle_enable=True`
  - Morning: `first_filter_cycle_morning=['g','r']`
  - Evening: `first_filter_cycle_evening=['z','i']`
- The selected filter is placed at the head of the per‑window batch order; remaining filters follow the palette (e.g., `palette_morning=['g','r','i','z']`, `palette_evening=['z','i','r','g']`).

With these two levers active, the planner tends to produce a much more even distribution across g/r/i/z without relaxing cadence or over‑swapping filters.

### m5 feasibility notes
- Filter feasibility uses LSST-style `m5` scaling: it responds to band-specific
  sky brightness, seeing, airmass, and the **current exposure time** (including
  any `sun_alt_exposure_ladder` overrides). The read-noise correction
  `ΔC_m(τ)` is applied so short 5 s visits lose depth appropriately.
- Sky brightness is sourced from `rubin_sim.skybrightness` when available;
  otherwise a fallback combines a twilight brightening term with the
  Krisciunas & Schaefer (1991) moon-scatter model, so moon geometry directly
  affects `m5`.
- Per-filter Moon safety uses the notebook’s `min_moon_sep_by_filter`
  (e.g. `g:35°`) after feasibility: bands that pass the m5/SNR gate remain
  subject to these minimum separations within each window.


## Notebook Parameter Highlights

- **Date range (UTC):**
  `START_DATE="2024-01-01"`, `END_DATE="2024-01-03"` — a short window for quick iteration and validation.
- **Run label:**
  `RUN_LABEL=None` — optional tag inserted into output filenames (e.g., "hybrid_strategy").
- **Site:**
  `LAT_DEG=-30.2446`, `LON_DEG=-70.7494`, `HEIGHT_M=2647` — Rubin site; required for correct twilight and airmass calculations.
- **Visibility:**  
  `MIN_ALT_DEG=20.0` — avoids the worst airmass while keeping more sky accessible in twilight.
- **Filters & Hardware:**  
  `FILTERS=["g","r","i","z"]`, `CAROUSEL_CAPACITY=5` — Rubin carousel holds up to five; `u`/`y` are not loaded in this example.  
  `EXPOSURE_BY_FILTER=5s` — very short exposures to mitigate twilight brightness and saturation risk.  
  `FILTERS_PER_VISIT_CAP=1` — one filter per visit reduces swaps and keeps within `PER_SN_CAP_S`.  
  `START_FILTER="g"` — initial carousel state only; the plan adapts per target.
- **Sun-altitude policy:**

  ```
  (-18, -15): ["y", "z", "i"]
  (-15, -12): ["z", "i", "r"]
  (-12,   0): ["i", "z", "y"]
  ```

  Only filters allowed by policy and present in `FILTERS` are considered. There
  are no additional band-specific bans; feasibility now follows the m5/SNR gate
  described above.

- **Slew model:**  
  `SLEW_SMALL_DEG=3.5`, `SLEW_SMALL_TIME_S=4.0`, `SLEW_RATE_DEG_PER_S=5.25`, `SLEW_SETTLE_S=1.0` — a simple piecewise model with a constant small-slew time and a linear rate for larger moves.
- **Moon:**  
  `MIN_MOON_SEP_BY_FILTER={"g":50,"r":35,"i":30,"z":25, ...}` — tighter in blue to protect S/N; dynamically scaled by Moon altitude/phase; ignored when the Moon is set.  
  `REQUIRE_SINGLE_TIME_FOR_ALL_FILTERS=True` — enforces a single “best time” per visit, ensuring all filters (if >1) share the same epoch in a visit.
- **Time caps:**  
  `PER_SN_CAP_S=120` — bounds per-SN work (slew + readout + exposure + filter changes).  
  `MORNING_CAP_S=600`, `EVENING_CAP_S=600` — roughly 10 minutes each; ensures plans pack into tight twilight windows.  
  `TWILIGHT_STEP_MIN=2` — few-minute sampling captures altitude/Moon geometry changes without over-sampling.  
  `MAX_SN_PER_NIGHT=float('inf')` — unlimited candidates; set finite for stress testing. If finite, cap is split between windows.
- **Priority strategy:**  
  `PRIORITY_STRATEGY="hybrid"` with `HYBRID_DETECTIONS=2`, `HYBRID_EXPOSURE=300s`, `LC_DETECTIONS=5`, `LC_EXPOSURE=300s`.  
  Starts broad with quick detections; escalates to deeper coverage for promising SNe.
- **Photometry / Sky:**
  `PIXEL_SCALE_ARCSEC=0.2` (Rubin pixel scale), `READ_NOISE_E=6` (typical 5.4–6.2 e⁻; requirement ≤9 e⁻ per LCA‑48‑J), `GAIN_E_PER_ADU=1.6` (measured ≈1.5–1.7 e⁻/ADU), `ZPT_ERR_MAG=0.01`, saturation threshold ≈8×10⁴ e⁻ by default.
  Dark‑sky surface brightnesses {u:23.05, g:22.25, r:21.20, i:20.46, z:19.61, y:18.60} mag/arcsec² (SMTN‑002); prefer `rubin_sim.skybrightness` when available. `TWILIGHT_DELTA_MAG=2.5` is an approximate fallback. Airmass uses the Kasten–Young (1989) approximation.
- **SIMLIB:**  
  `SIMLIB_OUT=None` (disabled in the example). Set e.g. `"twilight.simlib"` to generate a SIMLIB.
- **Misc:**  
  `TYPICAL_DAYS_BY_TYPE` (e.g., Ia: 70, II-P: 100, …) and `DEFAULT_TYPICAL_DAYS=60` define the baseline lifetime window (inflated by 1.2× for safety).  
  `ALLOW_FILTER_CHANGES_IN_TWILIGHT=False` — the example already constrains visits via `FILTERS_PER_VISIT_CAP=1`; if your build of the planner honors this flag, it further discourages cross-target swaps in twilight.

---

## Practical Tips

- Short exposures and single-filter visits are deliberate: twilight is bright and short; you’ll get more distinct SNe with fewer swaps.
- The Sun-altitude policy is strict: even if a filter is loaded, it won’t be used when the Sun is too high for that band.
- If you want multi-filter color on the same visit, increase `FILTERS_PER_VISIT_CAP` (optionally leaving `auto_color_pairing=True`) and be prepared to raise `PER_SN_CAP_S` and the window caps—or add a Sun-alt exposure ladder so exposures shrink as the Sun rises.

---

## References
1. [Rubin Observatory key numbers](https://www.lsst.org/scientists/keynumbers) — pixel scale, readout/shutter timing, site coordinates.
2. [DMTN-065: Detailed Filter Changer Timing](https://dmtn-065.lsst.io) — 120 s filter-change breakdown.
3. [SMTN-002: Expected LSST Performance](https://smtn-002.lsst.io) — sky brightness, read-noise/gain assumptions.
4. [LCA-48-J](https://project.lsst.org/reviews/jdr-2019/sites/lsst.org.reviews.jdr-2019/files/LCA-48-J_APPROVED_%28CameraSpec%29.pdf) — camera read-noise requirement (≤9 e⁻).
5. Ivezić, Ž., et al. 2019, ApJ, 873, 111 — LSST overview.
6. Kasten, F., & Young, A. T. 1989, Appl. Opt., 28, 4735 — airmass formula.
