import pathlib, sys

# Ensure package root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.config import PlannerConfig


def test_default_config_values():
    cfg = PlannerConfig(10.0, 20.0, 30.0)
    assert cfg.filters == ["g", "r", "i", "z", "y"]
    assert cfg.exposure_by_filter["g"] == 15.0
    assert cfg.evening_cap_s == 1800.0


def test_custom_overrides():
    custom_filters = ["g", "r"]
    custom_typical = {"Ia": 40}
    cfg = PlannerConfig(10.0, 20.0, 30.0, filters=custom_filters,
                        typical_days_by_type=custom_typical)
    assert cfg.filters == custom_filters
    assert cfg.typical_days_by_type == custom_typical
