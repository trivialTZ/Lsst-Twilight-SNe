# LSST Twilight Planner

Modular planner for scheduling LSST twilight observations of supernovae.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m twilight_planner_pkg.main --csv your.csv --out results \
    --start 2024-01-01 --end 2024-01-07 \
    --lat -30.2446 --lon -70.7494 --height 2663 \
    --filters g,r,i,z --exp g:5,r:5,i:5,z:5 \
    --min_alt 20 --evening_cap 600 --morning_cap 600 \
    --per_sn_cap 120 --max_sn 10 --strategy hybrid \
    --hybrid-detections 2 --hybrid-exposure 300 \
    --simlib-out results/night.SIMLIB --simlib-survey LSST_TWILIGHT
```

Use `--strategy lc` to require the LSST-only light-curve goal for all SNe.
Thresholds can be tuned with `--lc-detections` and `--lc-exposure`.
Example forcing LSST-only coverage:

```bash
python -m twilight_planner_pkg.main --csv your.csv --out results \
    --start 2024-01-01 --end 2024-01-07 \
    --lat -30.2446 --lon -70.7494 --height 2663 \
    --strategy lc --lc-detections 5 --lc-exposure 300
```

### Priority modes

The planner supports three observation philosophies:

1. **Discovery-optimized** – favour breadth with minimal repeat exposures
   (e.g. low `--hybrid-exposure` limits).
2. **Hybrid (default)** – gather a couple detections across colours and then
   down-weight non-Ia SNe.
3. **LSST-only light curves** – pursue full light curves for every object via
   `--strategy lc`.

### Notebook example

```python
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps

cfg = PlannerConfig(lat_deg=-30.2446, lon_deg=-70.7494, height_m=2663)
plan_twilight_range_with_caps('/path/to/your.csv', '/tmp/out',
                              '2024-01-01', '2024-01-07', cfg)
```

`PlannerConfig` includes dynamic-priority options:

* `priority_strategy` – either `"hybrid"` (default) or `"lc"` to always pursue
  a full light curve.
* `hybrid_detections` / `hybrid_exposure_s` – thresholds for the Hybrid goal
  (defaults: 2 detections across ≥2 filters or 300 s total).
* `lc_detections` / `lc_exposure_s` – thresholds for the LSST-only light-curve
  goal (defaults: 5 detections or 300 s spanning ≥2 filters).
* Default workflow: once the Hybrid goal is met, the cached type from the CSV
  is checked—Type Ia SNe escalate to the LSST-only goal, others drop to zero
  priority.

## How It Works

### Inputs & Preprocessing

* **Inputs** – CSV path, output directory, UTC start/end dates, and a
  `PlannerConfig` describing site details, slews/overheads, filters, caps, and
  column overrides.
* **Column handling** – RA, Dec, discovery date (ISO or MJD), name, and type
  columns are auto-detected if not explicitly supplied.
* **Units** – RA/Dec values are parsed to degrees with a series-level inference
  report that warns about ambiguous ranges or mixed types.
* **Derived columns** – `RA_deg`, `Dec_deg`, `discovery_datetime` (UTC),
  `Name`, and `SN_type_raw` are added for later stages.

### Eligibility

* A target must reach `max_alt_deg` above `min_alt_deg` (default 20°).
* Supernovae lacking a valid `discovery_datetime` are skipped.
* An object remains eligible for `ceil(1.2 × days)` after discovery, where the
  base `days` comes from `typical_days_by_type` or a default value.

### Best-Time Selection

* Twilight windows are spans when the Sun altitude is between −18° and 0°.
* Windows are sampled every `twilight_step_min` minutes to test altitude and
  Moon separation constraints.
* The time with the highest altitude satisfying `alt ≥ min_alt_deg` and Moon
  rules (Moon below horizon or sufficiently separated) is chosen.

### Selection & Scheduling

* Rank visible supernovae by their best altitude and keep the top
  `max_sn_per_night` globally.
* Split targets by window (index 0 morning, 1 evening) and order within each
  group by a greedy nearest-neighbor search on great-circle distance.
* Enforce per-window caps (`morning_cap_s`, `evening_cap_s`) while scheduling.
* Each target must fit within `per_sn_cap_s`; filters are trimmed greedily if
  needed.

### Slew & Overhead Model

* **Slew time** – if separation ≤ `slew_small_deg`, cost is `slew_small_time_s`;
  otherwise add rate-based time plus `slew_settle_s`.
* **Exposure time** – sum of `exposure_by_filter` values for used filters.
* **Additional costs** – `readout_s` per filter and `filter_change_s` per change.
* **Per-SN total** – slew + exposures + readout + filter changes.

### Filters & Exposure Assignment

* Filters are requested in priority order. If adding another filter would exceed
  `per_sn_cap_s`, the planner stops and uses the current set (ensuring at least
  one filter when possible).
* If the requested filter count exceeds `carousel_capacity`, a warning is
  emitted but planning continues.

### Outputs

* **Per-SN plan** – CSV and DataFrame listing date, window, chosen time,
  altitude, filters, exposure settings, and detailed time budget components.
* **Night summary** – CSV and DataFrame with counts of candidates vs. planned
  targets and cumulative time per window.
* **SIMLIB** – if ``--simlib-out`` is provided, a SNANA SIMLIB describing the
  simulated observations.

  
### Photometry, Saturation & SIMLIB

* Exposure times are automatically capped to keep the brightest pixel below
  `NPE_PIXEL_SATURATE` (default 1.2×10⁵ e⁻). A Gaussian PSF central-pixel
  fraction plus the relation
  `m_sat(t) = m_sat(15 s) + 2.5 log10(t/15)` shifts the r-band bright limit from
  ≈15.8 mag at 15 s to ≈12.9 mag at 1 s, so short twilight snaps avoid
  saturation while reducing overheads.
* Zeropoints use 1‑s instrumental values from SMTN‑002 and apply extinction
  `k_m (X − 1)`; sky noise comes from a twilight sky model or
  `rubin_sim.skybrightness` when available.
* When `--simlib-out` is supplied, the planner writes SNANA `S:` rows including
  MJD, gain, read noise, sky sigma, PSF FWHM, `ZPTAVG`, and `MAG=-99` using the
  above photometric scalars.

## Module overview

* `config.py` – `PlannerConfig` dataclass housing site parameters and
  threshold settings.
* `priority.py` – tracks per-SN detections, exposure time, filter coverage, and
  escalates Type Ia objects to an LSST-only light-curve goal.
* `scheduler.py` – core planner that queries visibility, applies priorities,
  and produces nightly schedules.
* `astro_utils.py` – astronomy helpers for twilight windows, moon separation,
  and slew calculations.
* `io_utils.py` – CSV column detection, RA/Dec normalization, and discovery
  date parsing.
* `photom_rubin.py` – Rubin photometric kernel and saturation guard.
* `simlib_writer.py` – minimal SNANA SIMLIB writer.
* `sky_model.py` – simple and Rubin-based sky-brightness providers.
* `main.py` – lightweight CLI entry point wrapping the scheduler.

## Documentation

All modules include NumPy-style docstrings detailing parameters, return values,
and algorithmic behavior for reference.

