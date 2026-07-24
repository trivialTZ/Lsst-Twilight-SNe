from __future__ import annotations

from datetime import datetime, timezone

import astropy.units as u
import pytest
from astropy.coordinates import EarthLocation
from rubin_scheduler.scheduler.model_observatory.jerk import jerk_time

from twilight_planner_pkg.astro_utils import (
    choose_filters_with_cap,
    dome_motion_seconds,
)
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import _route_motion_seconds_for_batch

SITE = EarthLocation(
    lat=-30.2446 * u.deg,
    lon=-70.7494 * u.deg,
    height=2663 * u.m,
)


def test_dome_motion_uses_rubin_jerk_profile_and_shortest_azimuth() -> None:
    cfg = PlannerConfig(dome_motion_enabled=True)

    motion = dome_motion_seconds(
        start_alt_deg=40.0,
        start_az_deg=350.0,
        end_alt_deg=50.0,
        end_az_deg=10.0,
        cfg=cfg,
    )

    expected_az = float(jerk_time(20.0, 1.5, 0.75, 3.0)[0]) + 1.0
    expected_el = float(jerk_time(10.0, 1.75, 0.875, 3.5)[0])
    assert motion["enabled"] is True
    assert motion["azimuth_delta_deg"] == pytest.approx(20.0)
    assert motion["elevation_delta_deg"] == pytest.approx(10.0)
    assert motion["azimuth_time_s"] == pytest.approx(expected_az)
    assert motion["elevation_time_s"] == pytest.approx(expected_el)
    assert motion["total_time_s"] == pytest.approx(max(expected_az, expected_el))


def test_disabled_dome_has_zero_motion_cost() -> None:
    cfg = PlannerConfig(dome_motion_enabled=False)

    motion = dome_motion_seconds(40.0, 0.0, 60.0, 180.0, cfg)

    assert motion["enabled"] is False
    assert motion["total_time_s"] == 0.0


def test_visit_timing_charges_slower_of_telescope_and_dome() -> None:
    cfg = PlannerConfig(
        filters=["r"],
        exposure_by_filter={"r": 5.0},
        readout_s=2.0,
    )

    used, timing = choose_filters_with_cap(
        ["r"],
        sep_deg=0.0,
        cap_s=1000.0,
        cfg=cfg,
        use_capped_exposure=False,
        dome_slew_s=30.0,
    )

    assert used == ["r"]
    assert timing["telescope_slew_s"] == 0.0
    assert timing["dome_slew_s"] == 30.0
    assert timing["slew_s"] == 30.0
    assert timing["dome_limited"] is True
    assert timing["total_s"] == 37.0


def test_route_ranking_uses_effective_dome_motion() -> None:
    cfg = PlannerConfig(dome_motion_enabled=True)
    previous = {"RA_deg": 10.0, "Dec_deg": -20.0}
    targets = [
        {"RA_deg": 35.0, "Dec_deg": -10.0},
        {"RA_deg": 190.0, "Dec_deg": -55.0},
    ]

    motion_times = _route_motion_seconds_for_batch(
        previous,
        targets,
        current_time_utc=datetime(2025, 3, 20, 0, 0, 0, tzinfo=timezone.utc),
        cfg=cfg,
        site=SITE,
    )

    assert len(motion_times) == 2
    assert all(value > 0.0 for value in motion_times)
    assert max(motion_times) > 30.0


def test_route_ranking_has_no_initial_motion() -> None:
    cfg = PlannerConfig(dome_motion_enabled=True)

    motion_times = _route_motion_seconds_for_batch(
        None,
        [{"RA_deg": 35.0, "Dec_deg": -10.0}],
        current_time_utc=datetime(2025, 3, 20, 0, 0, 0, tzinfo=timezone.utc),
        cfg=cfg,
        site=SITE,
    )

    assert motion_times == [0.0]


@pytest.mark.parametrize(
    "field_name",
    [
        "dome_azimuth_max_speed_deg_per_s",
        "dome_azimuth_accel_deg_per_s2",
        "dome_azimuth_jerk_deg_per_s3",
        "dome_elevation_max_speed_deg_per_s",
        "dome_elevation_accel_deg_per_s2",
        "dome_elevation_jerk_deg_per_s3",
    ],
)
def test_invalid_dome_motion_parameters_are_rejected(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        PlannerConfig(**{field_name: 0.0})
