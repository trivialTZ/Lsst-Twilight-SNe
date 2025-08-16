from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import twilight_planner_pkg.astro_utils as astro
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def _make_subset_csv(src: Path, out: Path, n: int = 2) -> Path:
    df = pd.read_csv(src).head(n)
    df.to_csv(out, index=False)
    return out


def _patch_scheduler(monkeypatch, allowed_filters):
    from twilight_planner_pkg import scheduler

    def mock_twilight_windows_for_local_night(
        date_local, loc, min_sun_alt_deg=-18.0, max_sun_alt_deg=0.0
    ):
        start = datetime(
            date_local.year,
            date_local.month,
            date_local.day,
            18,
            0,
            0,
            tzinfo=timezone.utc,
        )
        end = start + timedelta(minutes=30)
        return [
            {"start": start, "end": end, "label": "evening", "night_date": date_local}
        ]

    def mock_best_time_with_moon(
        sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg
    ):
        start, _ = window
        return 50.0, start + timedelta(minutes=5), 0.0, 0.0, 180.0

    def mock_pick_first(name, sn_type, tracker, allowed, cfg, **kwargs):
        return allowed[0] if allowed else None

    monkeypatch.setattr(
        scheduler,
        "twilight_windows_for_local_night",
        mock_twilight_windows_for_local_night,
    )
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", lambda *a, **k: 0.0)
    monkeypatch.setattr(
        scheduler, "allowed_filters_for_window", lambda *a, **k: allowed_filters
    )
    monkeypatch.setattr(
        scheduler, "allowed_filters_for_sun_alt", lambda alt, cfg: cfg.filters
    )
    monkeypatch.setattr(scheduler, "effective_min_sep", lambda *a, **k: 0.0)
    monkeypatch.setattr(scheduler, "pick_first_filter_for_target", mock_pick_first)
    monkeypatch.setattr(
        scheduler,
        "RubinSkyProvider",
        lambda *a, **k: (_ for _ in ()).throw(Exception("no rubin_sim")),
    )


def test_guard_between_visits(tmp_path, monkeypatch):
    data_path = (
        Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    )
    subset_csv = _make_subset_csv(data_path, tmp_path / "subset.csv", n=2)

    _patch_scheduler(monkeypatch, ["g"])
    monkeypatch.setattr(astro, "slew_time_seconds", lambda *a, **k: 2.0)

    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["g"],
        exposure_by_filter={"g": 5.0},
        readout_s=1.0,
        filter_change_s=0.0,
        evening_cap_s=1000.0,
        morning_cap_s=0.0,
        max_sn_per_night=2,
        per_sn_cap_s=50.0,
        min_moon_sep_by_filter={"g": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        start_filter="g",
        inter_exposure_min_s=15.0,
    )

    pernight_df, nights_df = plan_twilight_range_with_caps(
        csv_path=str(subset_csv),
        outdir=str(tmp_path),
        start_date="2025-07-30",
        end_date="2025-07-30",
        cfg=cfg,
        verbose=False,
    )

    assert len(pernight_df) == 2
    assert pernight_df.iloc[0]["guard_s"] == 0.0
    assert bool(pernight_df.iloc[1]["inter_exposure_guard_enforced"]) is True
    assert pernight_df.iloc[1]["guard_s"] == pytest.approx(13.0, abs=0.1)
    assert nights_df.iloc[0]["inter_exposure_guard_s"] == pytest.approx(13.0, abs=0.1)
    assert nights_df.iloc[0]["inter_exposure_guard_count"] == 1
    assert nights_df.iloc[0]["sum_time_s"] <= nights_df.iloc[0]["window_cap_s"]


def test_no_guard_when_slew_large(tmp_path, monkeypatch):
    data_path = (
        Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    )
    subset_csv = _make_subset_csv(data_path, tmp_path / "subset.csv", n=2)

    _patch_scheduler(monkeypatch, ["g"])
    monkeypatch.setattr(astro, "slew_time_seconds", lambda *a, **k: 25.0)

    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["g"],
        exposure_by_filter={"g": 5.0},
        readout_s=1.0,
        filter_change_s=0.0,
        evening_cap_s=1000.0,
        morning_cap_s=0.0,
        max_sn_per_night=2,
        per_sn_cap_s=50.0,
        min_moon_sep_by_filter={"g": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        start_filter="g",
        inter_exposure_min_s=15.0,
    )

    pernight_df, nights_df = plan_twilight_range_with_caps(
        csv_path=str(subset_csv),
        outdir=str(tmp_path),
        start_date="2025-07-30",
        end_date="2025-07-30",
        cfg=cfg,
        verbose=False,
    )

    assert (pernight_df["guard_s"] == 0).all()
    assert not pernight_df["inter_exposure_guard_enforced"].any()
    assert nights_df.iloc[0]["inter_exposure_guard_s"] == 0.0
    assert nights_df.iloc[0]["inter_exposure_guard_count"] == 0


def test_multi_filter_visit_no_internal_guard(tmp_path, monkeypatch):
    data_path = (
        Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    )
    subset_csv = _make_subset_csv(data_path, tmp_path / "subset.csv", n=1)

    _patch_scheduler(monkeypatch, ["g", "r"])
    monkeypatch.setattr(astro, "slew_time_seconds", lambda *a, **k: 2.0)

    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["g", "r"],
        exposure_by_filter={"g": 5.0, "r": 5.0},
        readout_s=2.0,
        filter_change_s=120.0,
        evening_cap_s=1000.0,
        morning_cap_s=0.0,
        max_sn_per_night=1,
        per_sn_cap_s=300.0,
        min_moon_sep_by_filter={"g": 0.0, "r": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        start_filter="g",
        inter_exposure_min_s=15.0,
        max_filters_per_visit=2,
    )

    pernight_df, nights_df = plan_twilight_range_with_caps(
        csv_path=str(subset_csv),
        outdir=str(tmp_path),
        start_date="2025-07-30",
        end_date="2025-07-30",
        cfg=cfg,
        verbose=False,
    )

    assert len(pernight_df) == 2
    assert (pernight_df["guard_s"] == 0).all()
    assert not pernight_df["inter_exposure_guard_enforced"].any()
    assert nights_df.iloc[0]["inter_exposure_guard_s"] == 0.0
    assert nights_df.iloc[0]["inter_exposure_guard_count"] == 0


def test_overlap_readout_with_slew(tmp_path, monkeypatch):
    data_path = (
        Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    )
    subset_csv = _make_subset_csv(data_path, tmp_path / "subset.csv", n=2)

    _patch_scheduler(monkeypatch, ["g"])
    # Slew is short but readout is long so readout dominates the natural gap.
    monkeypatch.setattr(astro, "slew_time_seconds", lambda *a, **k: 2.0)

    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["g"],
        exposure_by_filter={"g": 5.0},
        readout_s=6.0,
        filter_change_s=0.0,
        evening_cap_s=1000.0,
        morning_cap_s=0.0,
        max_sn_per_night=2,
        per_sn_cap_s=50.0,
        min_moon_sep_by_filter={"g": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        start_filter="g",
        inter_exposure_min_s=15.0,
    )

    pernight_df, nights_df = plan_twilight_range_with_caps(
        csv_path=str(subset_csv),
        outdir=str(tmp_path),
        start_date="2025-07-30",
        end_date="2025-07-30",
        cfg=cfg,
        verbose=False,
    )

    assert len(pernight_df) == 2
    assert pernight_df.iloc[0]["guard_s"] == 0.0
    assert pernight_df.iloc[1]["guard_s"] == pytest.approx(9.0, abs=0.1)
    assert pernight_df.iloc[1]["elapsed_overhead_s"] == pytest.approx(6.0, abs=0.1)
    assert pernight_df.iloc[1]["total_time_s"] == pytest.approx(20.0, abs=0.1)
    assert nights_df.iloc[0]["inter_exposure_guard_s"] == pytest.approx(9.0, abs=0.1)
