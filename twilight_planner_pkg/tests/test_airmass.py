import math
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.astro_utils import airmass_from_alt_deg

def test_airmass_kasten_young():
    assert math.isclose(airmass_from_alt_deg(30), 1.994, abs_tol=0.01)
    assert math.isclose(airmass_from_alt_deg(60), 1.155, abs_tol=0.005)


def test_airmass_limits():
    assert airmass_from_alt_deg(90) == 1.0
    assert airmass_from_alt_deg(95) == 1.0
    assert math.isinf(airmass_from_alt_deg(0))
    assert math.isinf(airmass_from_alt_deg(-5))
