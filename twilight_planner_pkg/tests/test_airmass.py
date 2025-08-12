import numpy as np
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.astro_utils import airmass_from_alt_deg

def test_airmass_kasten_young():
    assert abs(airmass_from_alt_deg(30) - 1.9943) < 1e-3
    assert abs(airmass_from_alt_deg(60) - 1.1540) < 1e-3
    assert abs(airmass_from_alt_deg(80) - 1.0151) < 1e-3

def test_airmass_limits():
    assert airmass_from_alt_deg(90) == 1.0
    assert np.isinf(airmass_from_alt_deg(-5))
