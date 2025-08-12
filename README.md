# LSST Twilight Supernova Planner

Tools for planning LSST twilight observations of supernovae and
experimenting with dynamic priority strategies.

## Repository layout

- `twilight_planner_pkg/` – core Python package with the scheduler and
  priority logic.
- `data/` – example input tables.
- `notebook/` – Jupyter notebooks demonstrating the planner.
- `twilight_outputs/` – sample planning results.
- `main.py` – small wrapper that invokes the package CLI.

The package is built in a modular fashion: separate modules handle
configuration, astronomy utilities, per‑SN priority tracking, and the
scheduler itself.  This structure makes it easy to swap in new
strategies or extend the planner for different surveys.

Recent additions provide a Rubin-style photometry model and a minimal
SNANA SIMLIB writer.  Planned exposures are now capped to avoid pixel
saturation, and setting ``--simlib-out`` on the CLI writes a SIMLIB file
alongside the usual planning CSVs.

See `twilight_planner_pkg/README.md` for detailed usage instructions and
module documentation.

## Twilight strategy highlights

The planner follows a Sun-altitude policy inspired by twilight brightness:

| Sun altitude (deg) | Allowed filters |
|--------------------|-----------------|
| -18 to -15         | y, z, i         |
| -15 to -12         | z, i, r         |
| -12 to 0           | i, z, y         |

This mapping is configurable via `PlannerConfig.sun_alt_policy`.

LSST's filter carousel can host at most five filters per night.  Cross-target
swaps incur a 120 s cost and additional in-visit filter changes add the same
overhead.  Readout time is 2 s per exposure and slews follow a hybrid model
(`3.5° in 4 s` plus `5.25°/s`).

Moon–target separations use Astropy's `get_body('moon')` in a shared AltAz
frame.  If the Moon is below the horizon, the separation requirement is
automatically waived.

Airmass calculations adopt the "simple" formula of Kasten & Young (1989), and
the overhead values above follow Rubin Observatory technical notes.

## Minimal example

```python
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps

cfg = PlannerConfig(filters=["i", "z"], start_filter="i")
plan_twilight_range_with_caps("my_catalog.csv", "out", "2024-01-01", "2024-01-01", cfg)
```

The repository includes `data/demo_three_targets.csv` illustrating a simple
three-target run.

## Installation

```bash
pip install -r requirements.txt
```
