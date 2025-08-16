"""Tests for verbose twilight window printing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def test_verbose_output_includes_local_times(tmp_path, monkeypatch, capsys):
    """Ensure UTC and local times appear in verbose summaries."""

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
        evening_start = datetime(2024, 1, 2, 0, 16, tzinfo=timezone.utc)
        evening_end = evening_start + timedelta(minutes=30)
        morning_start = datetime(2024, 1, 2, 9, 2, tzinfo=timezone.utc)
        morning_end = morning_start + timedelta(minutes=30)
        return [
            {
                "start": evening_start,
                "end": evening_end,
                "label": "evening",
                "night_date": date_local,
            },
            {
                "start": morning_start,
                "end": morning_end,
                "label": "morning",
                "night_date": date_local,
            },
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
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", lambda *args, **kwargs: 0.0)

    cfg = PlannerConfig(
        filters=["g"],
        exposure_by_filter={"g": 10.0},
        readout_s=1.0,
        filter_change_s=1.0,
        evening_cap_s=1000.0,
        morning_cap_s=1000.0,
        min_moon_sep_by_filter={"g": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        allow_filter_changes_in_twilight=True,
    )

    plan_twilight_range_with_caps(
        str(csv), tmp_path, "2024-01-01", "2024-01-01", cfg, verbose=True
    )
    out = capsys.readouterr().out

    assert (
        "  evening_twilight: 2024-01-02T00:16:00+00:00 \u2192 2024-01-02T00:46:00+00:00 (local 19:33 \u2192 20:03 UTC-04:43)"
        in out
    )
    assert (
        "  morning_twilight: 2024-01-02T09:02:00+00:00 \u2192 2024-01-02T09:32:00+00:00 (local 04:19 \u2192 04:49 UTC-04:43)"
        in out
    )
