import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.astro_utils import choose_filters_with_cap

def test_cross_target_change_accounted():
    cfg = PlannerConfig()
    _, timing = choose_filters_with_cap(["z"], 0.0, 1000.0, cfg, current_filter="i")
    assert timing["cross_filter_change_s"] == cfg.filter_change_time_s
    assert timing["total_s"] == timing["slew_s"] + timing["exposure_s"] + timing["readout_s"] + cfg.filter_change_time_s

def test_no_cross_change_when_same_filter():
    cfg = PlannerConfig()
    _, timing = choose_filters_with_cap(["z"], 0.0, 1000.0, cfg, current_filter="z")
    assert timing["cross_filter_change_s"] == 0.0
