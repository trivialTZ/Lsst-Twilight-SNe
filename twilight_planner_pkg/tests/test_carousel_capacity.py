import pathlib
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def test_carousel_capacity_enforced(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "ra": [0.0],
            "dec": [0.0],
            "discoverydate": ["2023-12-01T00:00:00Z"],
            "name": ["SN1"],
            "type": ["Ia"],
        }
    )
    csv = tmp_path / "cat.csv"
    df.to_csv(csv, index=False)
    cfg = PlannerConfig(
        filters=["u", "g", "r", "i", "z", "y"],
        carousel_capacity=5,
        morning_cap_s=100.0,
        evening_cap_s=100.0,
    )

    from twilight_planner_pkg import scheduler

    def mock_twilight_windows_for_local_night(date_local, loc):
        start = datetime(
            date_local.year,
            date_local.month,
            date_local.day,
            5,
            0,
            0,
            tzinfo=timezone.utc,
        )
        end = start + timedelta(minutes=30)
        return [
            {"start": start, "end": end, "label": "morning", "night_date": date_local}
        ]

    def mock_best_time_with_moon(
        sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg
    ):
        start, _ = window
        return 50.0, start + timedelta(minutes=5), 0.0, 0.0, 180.0

    def mock_sep(ra1, dec1, ra2, dec2):
        return 0.0

    monkeypatch.setattr(
        scheduler,
        "twilight_windows_for_local_night",
        mock_twilight_windows_for_local_night,
    )
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", mock_sep)

    _, nights = plan_twilight_range_with_caps(
        str(csv), tmp_path, "2024-01-01", "2024-01-01", cfg, verbose=False
    )
    assert not nights["loaded_filters"].str.contains("u").any()
    assert all(
        len(row.split(",")) <= cfg.carousel_capacity for row in nights["loaded_filters"]
    )
