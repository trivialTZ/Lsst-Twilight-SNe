import warnings
from datetime import datetime, timezone, timedelta
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.astro_utils import _best_time_with_moon


def test_moon_separation_waived_when_down():
    sc = SkyCoord(0 * u.deg, 0 * u.deg)
    now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    window = (now, now + timedelta(hours=1))
    loc = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    with warnings.catch_warnings(record=True) as w:
        alt, t = _best_time_with_moon(sc, window, loc, 10, -10.0, 180.0)
    assert w == []
    assert alt != float("-inf")
    assert t is not None
