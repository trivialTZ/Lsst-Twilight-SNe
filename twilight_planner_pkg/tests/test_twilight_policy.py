import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.astro_utils import allowed_filters_for_sun_alt


def test_twilight_policy_sets():
    cfg = PlannerConfig()
    assert allowed_filters_for_sun_alt(-16, cfg) == ["y", "z", "i"]
    assert allowed_filters_for_sun_alt(-13, cfg) == ["z", "i", "r"]
    assert allowed_filters_for_sun_alt(-5, cfg) == ["i", "z", "y"]
