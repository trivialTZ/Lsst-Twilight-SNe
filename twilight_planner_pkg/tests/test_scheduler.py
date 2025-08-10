"""Tests for the twilight scheduler."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def _make_subset_csv(src: Path, out: Path, n: int = 3) -> Path:
    """Create a small CSV subset for testing."""
    df = pd.read_csv(src).head(n)
    df.to_csv(out, index=False)
    return out


def test_plan_twilight_range_basic(tmp_path, monkeypatch):
    """Plan a single night and validate basic constraints."""

    # ------------------------------------------------------------------
    # Prepare small deterministic input catalogue
    data_path = Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    subset_csv = _make_subset_csv(data_path, tmp_path / "subset.csv", n=3)

    # ------------------------------------------------------------------
    # Mock astronomy-heavy helpers for deterministic behaviour
    from twilight_planner_pkg import scheduler

    def mock_twilight_windows_astro(date_utc, loc):
        start_morning = date_utc.replace(hour=5, minute=0, second=0)
        end_morning = date_utc.replace(hour=5, minute=30, second=0)
        start_evening = date_utc.replace(hour=18, minute=0, second=0)
        end_evening = date_utc.replace(hour=18, minute=30, second=0)
        return [(start_morning, end_morning), (start_evening, end_evening)]

    def mock_best_time_with_moon(sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg):
        start, _ = window
        return 50.0, start + timedelta(minutes=5)

    def mock_sep(ra1, dec1, ra2, dec2):
        return 0.0

    monkeypatch.setattr(scheduler, "twilight_windows_astro", mock_twilight_windows_astro)
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", mock_sep)

    # ------------------------------------------------------------------
    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["g", "r"],
        exposure_by_filter={"g": 10.0, "r": 10.0},
        readout_s=1.0,
        filter_change_s=1.0,
        evening_cap_s=50.0,
        morning_cap_s=50.0,
        max_sn_per_night=2,
        per_sn_cap_s=25.0,
        min_moon_sep_by_filter={"g": 0.0, "r": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
    )

    start_date = end_date = "2025-07-30"

    pernight_df, nights_df = plan_twilight_range_with_caps(
        csv_path=str(subset_csv),
        outdir=str(tmp_path),
        start_date=start_date,
        end_date=end_date,
        cfg=cfg,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # pernight_df checks
    expected_cols = {
        "date",
        "twilight_window",
        "SN",
        "RA_deg",
        "Dec_deg",
        "best_twilight_time_utc",
        "best_alt_deg",
        "priority_score",
        "filters",
        "exposure_s",
        "readout_s",
        "filter_changes_s",
        "slew_s",
        "total_time_s",
    }
    assert expected_cols.issubset(pernight_df.columns)

    # No more than max_sn_per_night scheduled
    assert len(pernight_df) <= cfg.max_sn_per_night

    # Per-SN timing should be non-negative and within per_sn_cap_s
    assert (pernight_df["total_time_s"] >= 0).all()
    assert (pernight_df["total_time_s"] <= cfg.per_sn_cap_s).all()
    for col in ["exposure_s", "readout_s", "filter_changes_s", "slew_s"]:
        assert (pernight_df[col] >= 0).all()

    # ------------------------------------------------------------------
    # nights_df checks
    expected_night_cols = {"date", "twilight_window", "n_candidates", "n_planned", "sum_time_s", "window_cap_s"}
    assert expected_night_cols.issubset(nights_df.columns)
    assert (nights_df["sum_time_s"] >= 0).all()
    assert (nights_df["sum_time_s"] <= nights_df["window_cap_s"]).all()

