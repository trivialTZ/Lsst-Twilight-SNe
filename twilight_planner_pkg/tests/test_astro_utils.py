import pathlib
import sys
from datetime import datetime, timezone

# Ensure package root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import astropy.units as u
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time

from twilight_planner_pkg.astro_utils import (
    _best_time_with_moon,
    choose_filters_with_cap,
    great_circle_sep_deg,
    parse_sn_type_to_window_days,
    per_sn_time_seconds,
    slew_time_seconds,
    twilight_windows_astro,
)
from twilight_planner_pkg.config import PlannerConfig

# Observatory location (CTIO)
LOC = EarthLocation(lat=-30.1652778 * u.deg, lon=-70.815 * u.deg, height=2215 * u.m)


def test_twilight_windows_astro():
    date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    cfg = PlannerConfig()
    windows = twilight_windows_astro(
        date,
        LOC,
        min_sun_alt_deg=cfg.twilight_sun_alt_min_deg,
        max_sun_alt_deg=cfg.twilight_sun_alt_max_deg,
    )
    assert windows == sorted(windows, key=lambda w: w["start"])
    assert all(w["start"] < w["end"] for w in windows)
    assert windows
    for w in windows:
        mid = w["start"] + (w["end"] - w["start"]) / 2
        alt = (
            get_sun(Time(mid))
            .transform_to(AltAz(obstime=Time(mid), location=LOC))
            .alt.deg
        )
        assert cfg.twilight_sun_alt_min_deg < alt < cfg.twilight_sun_alt_max_deg


def test_twilight_windows_astro_respects_bounds():
    date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    cfg = PlannerConfig(twilight_sun_alt_min_deg=-15.0, twilight_sun_alt_max_deg=-5.0)
    windows = twilight_windows_astro(
        date,
        LOC,
        min_sun_alt_deg=cfg.twilight_sun_alt_min_deg,
        max_sun_alt_deg=cfg.twilight_sun_alt_max_deg,
    )
    assert windows
    for w in windows:
        mid = w["start"] + (w["end"] - w["start"]) / 2
        alt = (
            get_sun(Time(mid))
            .transform_to(AltAz(obstime=Time(mid), location=LOC))
            .alt.deg
        )
        assert cfg.twilight_sun_alt_min_deg < alt < cfg.twilight_sun_alt_max_deg


def test_great_circle_sep_deg():
    assert great_circle_sep_deg(0, 0, 90, 0) == pytest.approx(90.0)
    assert great_circle_sep_deg(10, 10, 10, 10) == pytest.approx(0.0)


def test_slew_time_seconds():
    params = dict(small_deg=5, small_time=1, rate_deg_per_s=2, settle_s=0.5)
    assert slew_time_seconds(0, **params) == 0.0
    assert slew_time_seconds(3, **params) == pytest.approx(1.5)
    assert slew_time_seconds(10, **params) == pytest.approx(4.0)


def test_per_sn_time_and_filter_cap():
    cfg = PlannerConfig(
        lat_deg=0,
        lon_deg=0,
        height_m=0,
        slew_small_deg=5,
        slew_small_time_s=1,
        slew_rate_deg_per_s=1,
        slew_settle_s=1,
        readout_s=1,
        filter_change_s=5,
        exposure_by_filter={"g": 10, "r": 10},
        filters=["g", "r"],
        allow_filter_changes_in_twilight=True,
    )
    total, slew, exptime, readout, fchanges = per_sn_time_seconds(
        ["g", "r"], sep_deg=2, cfg=cfg
    )
    assert (total, slew, exptime, readout, fchanges) == pytest.approx((29, 2, 20, 2, 5))

    used, timing = choose_filters_with_cap(
        ["g", "r"], sep_deg=2, cap_s=40, cfg=cfg, max_filters_per_visit=2
    )
    assert used == ["g", "r"]
    assert timing["total_s"] == pytest.approx(29)

    used, timing = choose_filters_with_cap(
        ["g", "r"], sep_deg=2, cap_s=20, cfg=cfg, max_filters_per_visit=2
    )
    assert used == ["g"]
    assert timing["total_s"] == pytest.approx(13)

    used, timing = choose_filters_with_cap(
        ["g", "r"], sep_deg=2, cap_s=5, cfg=cfg, max_filters_per_visit=2
    )
    assert used == ["g"]
    assert timing["total_s"] == pytest.approx(13)


def test_parse_sn_type_to_window_days():
    cfg = PlannerConfig(
        lat_deg=0,
        lon_deg=0,
        height_m=0,
        typical_days_by_type={"Ia": 25},
        default_typical_days=10,
    )
    assert parse_sn_type_to_window_days("Type Ia", cfg) == 30
    assert parse_sn_type_to_window_days("Ib", cfg) == 12
    assert parse_sn_type_to_window_days("", cfg) == 12
    assert parse_sn_type_to_window_days(None, cfg) == 12


def test_best_time_with_moon():
    window = (
        datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 15, 1, 0, tzinfo=timezone.utc),
    )
    target_far = SkyCoord(74.817 * u.deg, -9.76 * u.deg)
    alt, t, *_ = _best_time_with_moon(
        target_far, window, LOC, step_min=30, min_alt_deg=30, min_moon_sep_deg=30
    )
    assert t is not None and window[0] <= t <= window[1]
    ts = Time(t)
    altaz = AltAz(obstime=ts, location=LOC)
    alt_check = target_far.transform_to(altaz).alt.deg
    moon_coord = get_body("moon", ts, location=LOC).transform_to(altaz)
    sep = target_far.transform_to(altaz).separation(moon_coord).deg
    assert alt_check >= 30
    assert sep >= 30

    target_near = SkyCoord(344.81747793145 * u.deg, -9.76040058827 * u.deg)
    alt, t, *_ = _best_time_with_moon(
        target_near, window, LOC, step_min=30, min_alt_deg=20, min_moon_sep_deg=30
    )
    assert t is None
