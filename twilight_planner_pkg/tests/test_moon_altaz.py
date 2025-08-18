import pathlib
import sys
import warnings
from datetime import datetime, timedelta, timezone

import astropy.units as u
import pytest
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.utils.exceptions import AstropyWarning

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from twilight_planner_pkg.astro_utils import _best_time_with_moon
from twilight_planner_pkg.constraints import moon_separation_factor


@pytest.mark.parametrize(
    "when, sc",
    [
        (datetime(2024, 1, 1, 0, tzinfo=timezone.utc), SkyCoord(0 * u.deg, 0 * u.deg)),
        (
            datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
            SkyCoord(45 * u.deg, 10 * u.deg),
        ),
    ],
)
def test_best_time_with_moon_no_warnings(when, sc):
    window = (when, when + timedelta(hours=1))
    loc = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    with warnings.catch_warnings(record=True) as w:
        _best_time_with_moon(sc, window, loc, 10, -10.0, 5.0)
    assert not any(issubclass(wi.category, AstropyWarning) for wi in w)


def test_moon_separation_factor_allows_low_altitude_moon():
    f_deep = moon_separation_factor(-15.0, 0.8)
    f_mid = moon_separation_factor(-5.0, 0.8)
    assert f_deep == 0.0
    assert 0.0 < f_mid < 1.0


def test_best_time_with_moon_zero_length_window():
    sc = SkyCoord(0 * u.deg, 0 * u.deg)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    window = (now, now)
    loc = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    alt, t, m_alt, m_phase, m_sep = _best_time_with_moon(
        sc, window, loc, 10, -10.0, 5.0
    )
    assert alt == float("-inf")
    assert t is None
    assert np.isnan(m_alt) and np.isnan(m_phase) and np.isnan(m_sep)
