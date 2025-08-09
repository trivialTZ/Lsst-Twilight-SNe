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
    --per_sn_cap 120 --max_sn 10
```

### Notebook example

```python
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps

cfg = PlannerConfig(lat_deg=-30.2446, lon_deg=-70.7494, height_m=2663)
plan_twilight_range_with_caps('/path/to/your.csv', '/tmp/out',
                              '2024-01-01', '2024-01-07', cfg)
```

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
  altitude, filters, and detailed time budget components.
* **Night summary** – CSV and DataFrame with counts of candidates vs. planned
  targets and cumulative time per window.

## Layout

* `config.py` — `PlannerConfig` dataclass for all knobs.
* `io_utils.py` — CSV column detection, RA/Dec normalization, discovery date
  parsing.
* `astro_utils.py` — astronomy helpers (twilight windows, moon separation
  checks, slews, etc.).
* `scheduler.py` — core scheduler `plan_twilight_range_with_caps(...)`
  orchestrating the plan.
* `main.py` — lightweight CLI entry point.

## Documentation

All modules include NumPy-style docstrings detailing parameters, return values,
and algorithmic behavior for reference.

