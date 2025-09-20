import pathlib
import sys

# Ensure package root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.config import PlannerConfig


def test_default_config_values():
    cfg = PlannerConfig(10.0, 20.0, 30.0)
    assert cfg.filters == ["g", "r", "i", "z", "y"]
    assert cfg.exposure_by_filter["g"] == 15.0
    assert cfg.evening_cap_s == "auto"
    assert cfg.morning_cap_s == "auto"
    assert cfg.max_swaps_per_window == 2
    assert cfg.first_epoch_color_boost == 1.5


def test_custom_overrides():
    custom_filters = ["g", "r"]
    custom_typical = {"Ia": 40}
    cfg = PlannerConfig(
        10.0, 20.0, 30.0, filters=custom_filters, typical_days_by_type=custom_typical
    )
    assert cfg.filters == custom_filters
    assert cfg.typical_days_by_type == custom_typical


def test_filter_name_normalisation():
    cfg = PlannerConfig(
        10.0,
        20.0,
        30.0,
        filters=["g", "Y"],
        start_filter="Y",
        exposure_by_filter={"g": 15.0, "Y": 20.0},
        min_moon_sep_by_filter={"g": 50.0, "Y": 25.0},
        sun_alt_policy=[(-18.0, -12.0, ["Y", "g"])],
        cosmo_weight_by_filter={"g": 1.1, "Y": 0.7},
    )

    assert cfg.filters == ["g", "y"]
    assert cfg.start_filter == "y"
    assert cfg.exposure_by_filter["y"] == 20.0
    assert cfg.min_moon_sep_by_filter["y"] == 25.0
    assert cfg.sun_alt_policy[0][2] == ["y", "g"]
    assert cfg.cosmo_weight_by_filter["y"] == 0.7
