from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def _make_subset_csv(src: Path, out: Path, n: int = 2) -> Path:
    df = pd.read_csv(src).head(n)
    df.to_csv(out, index=False)
    return out


def test_simlib_output(tmp_path, monkeypatch):
    data_path = Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    subset_csv = _make_subset_csv(data_path, tmp_path / "subset.csv", n=2)

    from twilight_planner_pkg import scheduler

    def mock_twilight_windows_astro(date_utc, loc):
        start_morning = date_utc.replace(hour=5, minute=0, second=0)
        end_morning = date_utc.replace(hour=5, minute=30, second=0)
        return [(start_morning, end_morning)]

    def mock_best_time_with_moon(sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg):
        start, _ = window
        return 50.0, start + timedelta(minutes=5)

    monkeypatch.setattr(scheduler, "twilight_windows_astro", mock_twilight_windows_astro)
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(scheduler, "RubinSkyProvider", lambda: scheduler.SimpleSkyProvider(scheduler.SkyModelConfig()))

    simlib_path = tmp_path / "plan.SIMLIB"

    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["r"],
        exposure_by_filter={"r": 3.0},
        readout_s=1.0,
        filter_change_s=0.0,
        evening_cap_s=30.0,
        morning_cap_s=30.0,
        max_sn_per_night=1,
        per_sn_cap_s=10.0,
        min_moon_sep_by_filter={"r": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        allow_filter_changes_in_twilight=True,
        simlib_out=str(simlib_path),
    )

    start_date = end_date = "2025-07-30"

    pernight_df, _ = plan_twilight_range_with_caps(
        csv_path=str(subset_csv),
        outdir=str(tmp_path),
        start_date=start_date,
        end_date=end_date,
        cfg=cfg,
        verbose=False,
    )

    assert simlib_path.exists()
    lines = simlib_path.read_text().strip().splitlines()
    s_lines = [l for l in lines if l.startswith("S:")]
    assert len(s_lines) == len(pernight_df)
    assert any("SURVEY:" in l for l in lines[:5])
