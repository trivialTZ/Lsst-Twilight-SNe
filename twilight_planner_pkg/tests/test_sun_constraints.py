from datetime import datetime, timedelta, timezone

import astropy.units as u
import pandas as pd
from astropy.coordinates import EarthLocation, get_sun
from astropy.time import Time

from twilight_planner_pkg import scheduler
from twilight_planner_pkg.astro_utils import (
    _best_time_with_moon,
    allowed_filters_for_sun_alt,
    sun_target_geometry,
)
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.photom_rubin import PhotomConfig
from twilight_planner_pkg.priority import PriorityTracker
from twilight_planner_pkg.sky_model import SkyModelConfig


SITE = EarthLocation(
    lat=-30.2446 * u.deg,
    lon=-70.7494 * u.deg,
    height=2647.0 * u.m,
)


def test_default_sun_separation_is_conservative():
    cfg = PlannerConfig()
    assert cfg.min_sun_sep_deg == 90.0


def test_sun_separation_must_be_valid():
    for invalid in (-0.1, 180.1):
        try:
            PlannerConfig(min_sun_sep_deg=invalid)
        except ValueError:
            pass
        else:  # pragma: no cover - assertion gives a clearer failure than pytest here
            raise AssertionError(f"min_sun_sep_deg={invalid} should be rejected")


def test_sun_target_geometry_uses_shared_topocentric_frame():
    obstime = Time("2025-03-20T23:00:00", scale="utc")
    sun = get_sun(obstime)
    sun_alt_deg, sun_sep_deg, sun_az_rel_deg = sun_target_geometry(sun, obstime, SITE)

    assert -90.0 <= sun_alt_deg <= 90.0
    assert sun_sep_deg < 1.0e-8
    assert sun_az_rel_deg < 1.0e-8


def test_best_time_rejects_target_inside_sun_avoidance_radius():
    start = datetime(2025, 3, 20, 22, 0, tzinfo=timezone.utc)
    window = (start, start + timedelta(minutes=10))
    sun = get_sun(Time(start))

    alt_deg, best_time, *_ = _best_time_with_moon(
        sun,
        window,
        SITE,
        step_min=2,
        min_alt_deg=-90.0,
        min_moon_sep_deg=0.0,
        min_sun_sep_deg=10.0,
    )

    assert best_time is None
    assert alt_deg == float("-inf")


def test_sun_policy_returns_no_filters_outside_configured_ranges():
    cfg = PlannerConfig(
        filters=["g", "i"],
        sun_alt_policy=[(-18.0, 0.0, ["i"])],
    )
    assert allowed_filters_for_sun_alt(1.0, cfg) == []
    assert allowed_filters_for_sun_alt(-19.0, cfg) == []


def test_empty_sun_policy_intersection_rejects_candidate(monkeypatch):
    cfg = PlannerConfig(
        filters=["g", "i"],
        min_sun_sep_deg=0.0,
        sun_alt_policy=[(-18.0, 0.0, ["i"])],
        min_moon_sep_by_filter={"g": 0.0, "i": 0.0},
    )
    best_time = pd.Timestamp("2025-03-20T23:00:00Z")
    group = pd.DataFrame(
        [
            {
                "Name": "SN-test",
                "RA_deg": 120.0,
                "Dec_deg": -30.0,
                "best_time_utc": best_time,
                "max_alt_deg": 60.0,
                "priority_score": 1.0,
                "SN_type_raw": "Ia",
                "redshift": 0.05,
                "_moon_alt": -20.0,
                "_moon_phase": 0.5,
                "_moon_sep": 180.0,
                "_sun_alt": -10.0,
                "_sun_sep": 120.0,
                "_sun_az_rel": 180.0,
            }
        ]
    )
    window = {
        "start": best_time.to_pydatetime(),
        "end": (best_time + pd.Timedelta(minutes=20)).to_pydatetime(),
        "label": "evening",
    }

    monkeypatch.setattr(scheduler, "allowed_filters_for_window", lambda *a, **k: ["g"])
    monkeypatch.setattr(scheduler, "allowed_filters_for_sun_alt", lambda *a, **k: ["i"])
    monkeypatch.setattr(scheduler, "get_sun_alt_deg_cached", lambda *a, **k: -10.0)

    candidates, _ = scheduler._prepare_window_candidates(
        group,
        window,
        0,
        cfg,
        SITE,
        PriorityTracker(),
        {"SN-test": {"g": 20.0, "i": 20.0}},
        {0: "g"},
        False,
        3.0,
        0.5,
        0.25,
        0.0,
        PhotomConfig(),
        None,
        SkyModelConfig(),
    )

    assert candidates == []


def test_serialized_visit_rechecks_sun_separation(tmp_path, monkeypatch):
    catalog = pd.DataFrame(
        {
            "ra": [0.0],
            "dec": [0.0],
            "discoverydate": ["2025-01-01T00:00:00Z"],
            "name": ["SN-sun-edge"],
            "type": ["Ia"],
        }
    )
    csv_path = tmp_path / "catalog.csv"
    catalog.to_csv(csv_path, index=False)

    def mock_windows(
        date_local,
        loc,
        min_sun_alt_deg=-18.0,
        max_sun_alt_deg=0.0,
    ):
        start = datetime(
            date_local.year,
            date_local.month,
            date_local.day,
            5,
            tzinfo=timezone.utc,
        )
        return [
            {
                "start": start,
                "end": start + timedelta(minutes=30),
                "label": "morning",
                "night_date": date_local,
            }
        ]

    def mock_best_time(
        sc,
        window,
        loc,
        step_min,
        min_alt_deg,
        min_moon_sep_deg,
        **kwargs,
    ):
        return 50.0, window[0] + timedelta(minutes=5), -20.0, 0.0, 180.0

    geometry_calls = 0

    def moving_sun_geometry(*args, **kwargs):
        nonlocal geometry_calls
        geometry_calls += 1
        # Preferred time and serialized start pass, but the end of the visit
        # enters the hard radius and must be rejected.
        return (
            (-10.0, 120.0, 180.0)
            if geometry_calls <= 2
            else (-10.0, 30.0, 0.0)
        )

    monkeypatch.setattr(scheduler, "twilight_windows_for_local_night", mock_windows)
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time)
    monkeypatch.setattr(scheduler, "sun_target_geometry", moving_sun_geometry)
    monkeypatch.setattr(scheduler, "allowed_filters_for_window", lambda *a, **k: ["i"])
    monkeypatch.setattr(scheduler, "allowed_filters_for_sun_alt", lambda *a, **k: ["i"])
    monkeypatch.setattr(scheduler, "get_sun_alt_deg_cached", lambda *a, **k: -10.0)
    monkeypatch.setattr(scheduler, "pick_first_filter_for_target", lambda *a, **k: "i")
    monkeypatch.setattr(scheduler, "_cached_m5_at_time", lambda *a, **k: 30.0)

    cfg = PlannerConfig(
        filters=["i"],
        start_filter="i",
        min_sun_sep_deg=60.0,
        min_moon_sep_by_filter={"i": 0.0},
        morning_cap_s=1000.0,
        evening_cap_s=0.0,
        max_sn_per_night=1,
    )
    pernight, nights = scheduler.plan_twilight_range_with_caps(
        str(csv_path),
        str(tmp_path),
        "2025-01-02",
        "2025-01-02",
        cfg,
        verbose=False,
    )

    assert geometry_calls >= 3
    assert pernight.empty
    assert nights.iloc[0]["n_planned"] == 0
