# Modular LSST Twilight Planner

This is a split, maintainable version of your original `twilight_planner.py`.

## Layout
- `config.py` — `PlannerConfig` dataclass for all knobs.
- `io_utils.py` — CSV column detection, RA/Dec normalization, discovery date parsing.
- `astro_utils.py` — astronomy helpers (twilight windows, moon separation checks, slews, etc.).
- `scheduler.py` — core scheduler `plan_twilight_range_with_caps(...)` orchestrating the plan.
- `main.py` — lightweight CLI entry point.

## Quick start (example)
```bash
python -m twilight_planner_pkg.main   --csv /mnt/data/ATLAS_2021_to25_cleaned.csv   --out /mnt/data/out   --start 2024-01-01 --end 2024-01-07   --lat -30.2446 --lon -70.7494 --height 2663   --filters g,r,i,z   --exp g:5,r:5,i:5,z:5   --min_alt 20 --evening_cap 600 --morning_cap 600 --per_sn_cap 120 --max_sn 10
```

You can also import it in notebooks:
```python
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps

cfg = PlannerConfig(lat_deg=-30.2446, lon_deg=-70.7494, height_m=2663)
plan_twilight_range_with_caps(
    '/path/to/your.csv', '/tmp/out', '2024-01-01', '2024-01-07', cfg
)
```
