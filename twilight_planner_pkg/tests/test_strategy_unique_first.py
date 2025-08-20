from datetime import datetime, timedelta, timezone

import pandas as pd

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.priority import PriorityTracker
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def test_unique_first_scoring() -> None:
    """Unseen SNe score high; repeats are suppressed."""

    tracker = PriorityTracker()
    assert tracker.score("SN1", strategy="unique_first") == 1.0
    tracker.record_detection("SN1", 10.0, ["g"])
    assert tracker.score("SN1", strategy="unique_first") == -1.0


def test_scheduler_unique_first_no_repeats(tmp_path, monkeypatch) -> None:
    """Scheduler avoids repeats and reports summary metrics."""

    df = pd.DataFrame(
        {
            "ra": [0.0, 1.0],
            "dec": [0.0, 1.0],
            "discoverydate": ["2023-12-01T00:00:00Z", "2023-12-01T00:00:00Z"],
            "name": ["SN1", "SN2"],
            "type": ["Ia", "Ib"],
        }
    )
    csv = tmp_path / "cat.csv"
    df.to_csv(csv, index=False)

    from twilight_planner_pkg import scheduler

    def mock_twilight_windows_for_local_night(
        date_local, loc, min_sun_alt_deg=-18.0, max_sun_alt_deg=0.0
    ):
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

    monkeypatch.setattr(
        scheduler,
        "twilight_windows_for_local_night",
        mock_twilight_windows_for_local_night,
    )
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)

    cfg = PlannerConfig(
        filters=["i"],
        morning_cap_s=1000.0,
        evening_cap_s=1000.0,
        priority_strategy="unique_first",
    )
    pernight, nights = plan_twilight_range_with_caps(
        str(csv), tmp_path, "2024-01-01", "2024-01-01", cfg, verbose=False
    )
    row = nights.iloc[0]
    assert row["n_planned"] == 2
    assert row["unique_targets_observed"] == 2
    assert row["repeat_fraction"] == 0.0


def test_scheduler_unique_first_filters_after_detection(tmp_path, monkeypatch) -> None:
    """Second night drops targets already seen once."""

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

    from twilight_planner_pkg import scheduler

    def mock_twilight_windows_for_local_night(
        date_local, loc, min_sun_alt_deg=-18.0, max_sun_alt_deg=0.0
    ):
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

    monkeypatch.setattr(
        scheduler,
        "twilight_windows_for_local_night",
        mock_twilight_windows_for_local_night,
    )
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)

    cfg = PlannerConfig(
        filters=["i"],
        morning_cap_s=1000.0,
        evening_cap_s=1000.0,
        priority_strategy="unique_first",
    )
    pernight, nights = plan_twilight_range_with_caps(
        str(csv), tmp_path, "2024-01-01", "2024-01-02", cfg, verbose=False
    )
    assert len(nights) == 1
    assert nights.iloc[0]["n_planned"] == 1
