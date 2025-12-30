from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from astropy.time import Time

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps
from twilight_planner_pkg.simlib_reader import simlib_visits_by_name


def test_scheduler_respects_future_wfd_gate(tmp_path, monkeypatch):
    toy_simlib = Path(__file__).resolve().parents[2] / "twilight_planner_pkg" / "tests" / "data" / "wfd_toy.SIMLIB"
    visits = simlib_visits_by_name(toy_simlib)

    start_mjd = 60001.0
    start_dt = Time(start_mjd, format="mjd").to_datetime(timezone.utc)
    end_dt = start_dt + timedelta(minutes=30)

    def mock_twilight_windows_for_local_night(date_local, loc=None, min_sun_alt_deg=-18.0, max_sun_alt_deg=0.0):
        return [
            {
                "start": start_dt,
                "end": end_dt,
                "label": "morning",
                "night_date": date_local,
            }
        ]

    def mock_best_time_with_moon(sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg):
        start, _ = window
        return 50.0, start + timedelta(minutes=5), 0.0, 0.0, 180.0

    from twilight_planner_pkg import scheduler

    monkeypatch.setattr(
        scheduler,
        "twilight_windows_for_local_night",
        mock_twilight_windows_for_local_night,
    )
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        scheduler,
        "RubinSkyProvider",
        lambda: scheduler.SimpleSkyProvider(scheduler.SkyModelConfig()),
    )

    csv_path = tmp_path / "catalog.csv"
    df = pd.DataFrame(
        [
            {
                "Name": "SN2023abc",
                "RA": 10.0,
                "DEC": -5.0,
                "discoverydate": (start_dt - timedelta(days=5)).date().isoformat(),
            }
        ]
    )
    df.to_csv(csv_path, index=False)

    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["i"],
        exposure_by_filter={"i": 5.0},
        readout_s=1.0,
        filter_change_s=0.0,
        evening_cap_s=100.0,
        morning_cap_s=100.0,
        max_sn_per_night=1,
        per_sn_cap_s=20.0,
        min_moon_sep_by_filter={"i": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        allow_filter_changes_in_twilight=True,
    )

    # With future WFD visit at start_mjd for filter i, cadence gate should block scheduling.
    pernight_df, _ = plan_twilight_range_with_caps(
        csv_path=str(csv_path),
        outdir=str(tmp_path / "with_wfd"),
        start_date=start_dt.date().isoformat(),
        end_date=start_dt.date().isoformat(),
        cfg=cfg,
        verbose=False,
        wfd_visits_by_name=visits,
    )
    assert pernight_df.empty

    # Without the WFD cadence map, the same target should be scheduled.
    pernight_df2, _ = plan_twilight_range_with_caps(
        csv_path=str(csv_path),
        outdir=str(tmp_path / "no_wfd"),
        start_date=start_dt.date().isoformat(),
        end_date=start_dt.date().isoformat(),
        cfg=cfg,
        verbose=False,
    )
    assert len(pernight_df2) == 1
