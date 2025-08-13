"""Tests for the twilight scheduler."""

from __future__ import annotations

from datetime import timedelta
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
    data_path = (
        Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    )
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

    def mock_best_time_with_moon(
        sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg
    ):
        start, _ = window
        return 50.0, start + timedelta(minutes=5), 0.0, 0.0, 180.0

    def mock_sep(ra1, dec1, ra2, dec2):
        return 0.0

    monkeypatch.setattr(
        scheduler, "twilight_windows_astro", mock_twilight_windows_astro
    )
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
        allow_filter_changes_in_twilight=True,
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
        "filter",
        "t_exp_s",
        "airmass",
        "alt_deg",
        "sky_mag_arcsec2",
        "ZPT",
        "SKYSIG",
        "NEA_pix",
        "RDNOISE",
        "GAIN",
        "saturation_guard_applied",
        "warn_nonlinear",
        "priority_score",
    }
    assert expected_cols.issubset(pernight_df.columns)

    assert (pernight_df["t_exp_s"] >= 0).all()

    # ------------------------------------------------------------------
    # nights_df checks
    expected_night_cols = {
        "date",
        "twilight_window",
        "n_candidates",
        "n_planned",
        "sum_time_s",
        "window_cap_s",
    }
    assert expected_night_cols.issubset(nights_df.columns)
    assert (nights_df["sum_time_s"] >= 0).all()
    assert (nights_df["sum_time_s"] <= nights_df["window_cap_s"]).all()


def test_sun_alt_policy_enforced(tmp_path, monkeypatch):
    """Filters outside the sun_alt_policy are never scheduled."""

    data_path = (
        Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    )
    subset_csv = _make_subset_csv(data_path, tmp_path / "subset.csv", n=1)

    from types import SimpleNamespace

    import astropy.units as u

    from twilight_planner_pkg import scheduler

    def mock_twilight_windows_astro(date_utc, loc):
        start_morning = date_utc.replace(hour=5, minute=0, second=0)
        end_morning = date_utc.replace(hour=5, minute=30, second=0)
        return [(start_morning, end_morning)]

    def mock_best_time_with_moon(
        sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg
    ):
        start, _ = window
        return 50.0, start + timedelta(minutes=5), 0.0, 0.0, 180.0

    def mock_sep(ra1, dec1, ra2, dec2):
        return 0.0

    def mock_allowed_filters_for_window(*args, **kwargs):
        return ["r", "i"]

    def mock_get_sun(time):
        return SimpleNamespace(
            transform_to=lambda frame: SimpleNamespace(alt=-10 * u.deg)
        )

    monkeypatch.setattr(
        scheduler, "twilight_windows_astro", mock_twilight_windows_astro
    )
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", mock_sep)
    monkeypatch.setattr(
        scheduler, "allowed_filters_for_window", mock_allowed_filters_for_window
    )
    monkeypatch.setattr(scheduler, "get_sun", mock_get_sun)

    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["r", "i"],
        exposure_by_filter={"r": 10.0, "i": 10.0},
        readout_s=1.0,
        filter_change_s=1.0,
        morning_cap_s=50.0,
        evening_cap_s=50.0,
        max_sn_per_night=1,
        per_sn_cap_s=25.0,
        min_moon_sep_by_filter={"r": 0.0, "i": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        allow_filter_changes_in_twilight=True,
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

    assert set(pernight_df["filter"].unique()) == {"i"}


def test_exposure_ladder_applied(tmp_path, monkeypatch):
    """Exposure-time ladder overrides baseline exposures."""

    data_path = (
        Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    )
    subset_csv = _make_subset_csv(data_path, tmp_path / "subset.csv", n=1)

    from types import SimpleNamespace

    import astropy.units as u

    from twilight_planner_pkg import astro_utils, scheduler

    def mock_twilight_windows_astro(date_utc, loc):
        start = date_utc.replace(hour=5, minute=0, second=0)
        end = date_utc.replace(hour=5, minute=30, second=0)
        return [(start, end)]

    def mock_best_time_with_moon(
        sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg
    ):
        start, _ = window
        return 50.0, start + timedelta(minutes=5), 0.0, 0.0, 180.0

    def mock_sep(ra1, dec1, ra2, dec2):
        return 0.0

    def mock_allowed_filters_for_window(*args, **kwargs):
        return ["r"]

    def mock_get_sun(time):
        return SimpleNamespace(
            transform_to=lambda frame: SimpleNamespace(alt=-10 * u.deg)
        )

    def mock_compute_capped_exptime(band, cfg):
        return cfg.exposure_by_filter[band], set()

    monkeypatch.setattr(
        scheduler, "twilight_windows_astro", mock_twilight_windows_astro
    )
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", mock_sep)
    monkeypatch.setattr(
        scheduler, "allowed_filters_for_window", mock_allowed_filters_for_window
    )
    monkeypatch.setattr(scheduler, "get_sun", mock_get_sun)
    monkeypatch.setattr(
        astro_utils, "compute_capped_exptime", mock_compute_capped_exptime
    )

    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["r"],
        exposure_by_filter={"r": 15.0},
        readout_s=0.0,
        filter_change_s=0.0,
        morning_cap_s=50.0,
        evening_cap_s=50.0,
        max_sn_per_night=1,
        per_sn_cap_s=25.0,
        min_moon_sep_by_filter={"r": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        allow_filter_changes_in_twilight=True,
        sun_alt_exposure_ladder=[(-12.0, -8.0, {"r": 5.0})],
        sun_alt_policy=[],
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

    assert pernight_df["t_exp_s"].tolist() == [5.0]
