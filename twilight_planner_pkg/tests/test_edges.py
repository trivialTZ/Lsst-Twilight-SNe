import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.astro_utils import great_circle_sep_deg, ra_delta_shortest_deg
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.constraints import effective_min_sep
from twilight_planner_pkg.filter_policy import allowed_filters_for_window
from twilight_planner_pkg.io_utils import standardize_columns


def test_sun_alt_monotonic():
    mags = {}
    sets = []
    for alt in [-18.1, -18.0, -17.9, -0.1, 0.0, 0.1]:
        allowed = allowed_filters_for_window(mags, alt, -20.0, 0.0, 180.0, 1.0, 0.7)
        sets.append(set(allowed))
    for earlier, later in zip(sets, sets[1:]):
        assert later.issubset(earlier)


def test_moon_factor_monotonic():
    base = {"r": 25.0}
    c1 = effective_min_sep("r", -15.0, 0.8, base)
    c2 = effective_min_sep("r", -5.0, 0.8, base)
    c3 = effective_min_sep("r", 5.0, 0.8, base)
    assert c1 < c2 < c3


def test_ra_wrap_shortest():
    assert ra_delta_shortest_deg(359, 1) == pytest.approx(2)
    assert ra_delta_shortest_deg(1, 359) == pytest.approx(2)
    sep = great_circle_sep_deg(359, 0, 1, 0)
    assert sep < 5


def test_no_dec_clamp():
    df = pd.DataFrame(
        {"ra": [0.0], "dec": [93.0], "name": ["SNX"], "discoverydate": [60000.0]}
    )
    with pytest.raises(ValueError):
        standardize_columns(df, PlannerConfig())


def test_allowed_filters_extremes():
    mags = {}
    bright = allowed_filters_for_window(mags, -5.0, 5.0, 1.0, 20.0, 1.0, 0.7)
    assert set(bright).issubset({"i", "z", "y"})
    dark = allowed_filters_for_window(mags, -15.0, -20.0, 0.0, 120.0, 1.0, 0.7)
    assert {"g", "r", "i"}.issubset(set(dark))
