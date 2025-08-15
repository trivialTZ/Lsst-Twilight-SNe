import pathlib
import sys
from datetime import date

import astropy.units as u
from astropy.coordinates import EarthLocation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.astro_utils import twilight_windows_for_local_night

LOC = EarthLocation(lat=-30.1652778 * u.deg, lon=-70.815 * u.deg, height=2215 * u.m)


def test_local_night_bundling_returns_evening_and_morning():
    # Pick an arbitrary date; both evening and morning should belong to this local night
    d = date(2024, 1, 1)
    wins = twilight_windows_for_local_night(d, LOC)
    labels = [w["label"] for w in wins]
    assert set(labels) <= {"evening", "morning"}
    assert all(w["night_date"] == d for w in wins)
    # If both exist, they must be unique
    assert labels.count("evening") <= 1
    assert labels.count("morning") <= 1
