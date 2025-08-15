import pathlib
import sys
import warnings
from datetime import datetime, timedelta, timezone

# isort:skip_file

import astropy.units as u
import pandas as pd
from astropy.coordinates import EarthLocation, SkyCoord

# Ensure package root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.astro_utils import (
    _best_time_with_moon,
    choose_filters_with_cap,
    pick_first_filter_for_target,
)
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.priority import PriorityTracker
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def test_best_time_with_moon_no_warning():
    sc = SkyCoord(0 * u.deg, 0 * u.deg)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = (now, now + timedelta(hours=1))
    loc = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")
        alt, t, *_ = _best_time_with_moon(sc, window, loc, 10, 0.0, 0.0)
    assert w == []
    assert isinstance(alt, float)


def test_filter_capacity_drop(tmp_path, capsys):
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
    plan_twilight_range_with_caps(
        str(csv), tmp_path, "2024-01-01", "2024-01-01", cfg, verbose=True
    )
    captured = capsys.readouterr()
    assert "dropping u" in captured.out.lower()
    assert "u" not in cfg.filters


def test_cross_vs_internal_changes():
    cfg = PlannerConfig()
    used1, t1 = choose_filters_with_cap(["z"], 0.0, 1000.0, cfg)
    used2, t2 = choose_filters_with_cap(
        ["i"], 0.0, 1000.0, cfg, current_filter=used1[-1]
    )
    used3, t3 = choose_filters_with_cap(
        ["z"], 0.0, 1000.0, cfg, current_filter=used2[-1]
    )
    cross = sum(int(x["cross_filter_change_s"] > 0) for x in [t1, t2, t3])
    assert cross == 2


def test_pick_first_filter_priority():
    cfg = PlannerConfig()
    tracker = PriorityTracker()
    tracker.record_detection("SN_A", 15.0, ["g"])
    f1 = pick_first_filter_for_target("SN_A", "Ia", tracker, ["g", "r"], cfg)
    assert f1 == "r"  # missing second filter

    tracker.record_detection("SN_B", 15.0, ["g", "r"])
    tracker.history["SN_B"].escalated = True
    f2 = pick_first_filter_for_target(
        "SN_B", "Ia", tracker, ["g", "r"], cfg, current_filter="g"
    )
    assert f2 == "r"  # red preference in LC stage


def test_per_sn_cap_allows_one():
    cfg = PlannerConfig()
    used, timing = choose_filters_with_cap(["g", "r"], 0.0, 1.0, cfg)
    assert used == ["g"]
    assert timing["total_s"] > 1.0


def test_window_cap_skips_last(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "ra": [0.0, 0.0],
            "dec": [0.0, 0.0],
            "discoverydate": ["2023-12-01T00:00:00Z", "2023-12-01T00:00:00Z"],
            "name": ["SN1", "SN2"],
            "type": ["Ia", "Ia"],
        }
    )
    csv = tmp_path / "cat.csv"
    df.to_csv(csv, index=False)
    cfg = PlannerConfig(filters=["i"], morning_cap_s=20.0, evening_cap_s=20.0)

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

    pernight, nights = plan_twilight_range_with_caps(
        str(csv), tmp_path, "2024-01-01", "2024-01-01", cfg, verbose=False
    )
    assert not nights.empty and nights["n_planned"].max() <= 1


def test_exposure_capping_reduces_total():
    cfg = PlannerConfig()
    cfg.current_mag_by_filter = {"r": 14.0}
    cfg.current_alt_deg = 60.0
    used_cap, timing_cap = choose_filters_with_cap(
        ["r"], 0.0, 1000.0, cfg, use_capped_exposure=True
    )
    used_base, timing_base = choose_filters_with_cap(
        ["r"], 0.0, 1000.0, cfg, use_capped_exposure=False
    )
    assert timing_cap["exp_times"]["r"] < cfg.exposure_by_filter["r"]
    assert timing_cap["total_s"] < timing_base["total_s"]
