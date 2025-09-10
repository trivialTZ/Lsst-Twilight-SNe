import pathlib
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

# Ensure package root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def test_nightly_summary_fields(tmp_path, monkeypatch):
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
        return 50.0, start + pd.Timedelta(minutes=5), 0.0, 0.0, 180.0

    monkeypatch.setattr(
        scheduler,
        "twilight_windows_for_local_night",
        mock_twilight_windows_for_local_night,
    )
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)
    # Ensure deterministic first-filter choice and avoid extra swaps/repeats
    monkeypatch.setattr(scheduler, "pick_first_filter_for_target", lambda *a, **k: "z")
    monkeypatch.setattr(scheduler, "allowed_filters_for_window", lambda *a, **k: ["z"]) 

    # Set start_filter to match the likely first-choice filter ('z') so
    # the initial target does not count as a swap.
    cfg = PlannerConfig(
        filters=["i", "z"],
        start_filter="z",
        max_sn_per_night=1,
        morning_cap_s=1000.0,
        evening_cap_s=1000.0,
    )
    pernight, nights = plan_twilight_range_with_caps(
        str(csv), tmp_path, "2024-01-01", "2024-01-01", cfg, verbose=False
    )
    row = nights.iloc[0]
    # Ensure we never emit unlabeled windows (e.g., "W0")
    assert set(pernight["twilight_window"].unique()) <= {"morning", "evening"}
    assert row["swap_count"] == 0
    assert row["internal_filter_changes"] == 0
    assert row["filter_change_s_total"] == 0.0
    mean_expected = pernight["slew_s"].mean()
    assert row["mean_slew_s"] == pytest.approx(mean_expected)
    am_expected = pernight["airmass"].median()
    assert row["median_airmass"] == pytest.approx(am_expected, rel=1e-3)
    assert row["filters_used_csv"] == ",".join(sorted(pernight["filter"].unique()))
    assert row["n_planned"] == 1
    assert "moon_sep" in pernight.columns
    start = datetime(2024, 1, 1, 5, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)
    assert row["window_start_utc"] == pd.Timestamp(start).tz_convert("UTC").isoformat()
    assert row["window_end_utc"] == pd.Timestamp(end).tz_convert("UTC").isoformat()
    assert row["window_duration_s"] == 1800
    assert row["cap_source"] == "morning_cap_s"
    assert row["median_alt_deg"] == pytest.approx(pernight["alt_deg"].median())
    assert row["unique_targets_observed"] == row["n_planned"]
    assert row["repeat_fraction"] == 0.0
