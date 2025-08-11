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

## Installation

```bash
pip install -r requirements.txt
```
