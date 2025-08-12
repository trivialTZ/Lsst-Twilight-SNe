import pathlib
import sys
from datetime import datetime, timezone, timedelta
import warnings

import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation

# Ensure package root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.astro_utils import (
    _best_time_with_moon,
    choose_filters_with_cap,
    pick_first_filter_for_target,
)
from twilight_planner_pkg.priority import PriorityTracker
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def test_best_time_with_moon_no_warning():
    sc = SkyCoord(0 * u.deg, 0 * u.deg)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = (now, now + timedelta(hours=1))
    loc = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")
        alt, t = _best_time_with_moon(sc, window, loc, 10, 0.0, 0.0)
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
    plan_twilight_range_with_caps(str(csv), tmp_path, "2024-01-01", "2024-01-01", cfg, verbose=True)
    captured = capsys.readouterr()
    assert "dropping u" in captured.out.lower()
    assert "u" not in cfg.filters


def test_cross_vs_internal_changes():
    cfg = PlannerConfig()
    used1, t1 = choose_filters_with_cap(["z"], 0.0, 1000.0, cfg)
    used2, t2 = choose_filters_with_cap(["i"], 0.0, 1000.0, cfg, current_filter=used1[-1])
    used3, t3 = choose_filters_with_cap(["z"], 0.0, 1000.0, cfg, current_filter=used2[-1])
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
    f2 = pick_first_filter_for_target("SN_B", "Ia", tracker, ["g", "r"], cfg, current_filter="g")
    assert f2 == "r"  # red preference in LC stage


def test_per_sn_cap_allows_one():
    cfg = PlannerConfig()
    used, timing = choose_filters_with_cap(["g", "r"], 0.0, 1.0, cfg)
    assert used == ["g"]
    assert timing["total_s"] > 1.0


def test_window_cap_skips_last(tmp_path):
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
    cfg = PlannerConfig(filters=["g"], morning_cap_s=20.0, evening_cap_s=20.0)
    pernight, nights = plan_twilight_range_with_caps(
        str(csv), tmp_path, "2024-01-01", "2024-01-01", cfg, verbose=False
    )
    assert nights["n_planned"].max() <= 1
