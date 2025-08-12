import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.astro_utils import slew_time_seconds, great_circle_sep_deg

def test_filter_aware_cost_prefers_same_filter():
    cfg = PlannerConfig()
    prev = {"RA_deg": 0.0, "Dec_deg": 0.0, "first_filter": "z"}
    target_diff = {"RA_deg": 1.0, "Dec_deg": 0.0, "first_filter": "i"}
    target_same = {"RA_deg": 1.5, "Dec_deg": 0.0, "first_filter": "z"}
    sep_diff = great_circle_sep_deg(prev["RA_deg"], prev["Dec_deg"], target_diff["RA_deg"], target_diff["Dec_deg"])
    sep_same = great_circle_sep_deg(prev["RA_deg"], prev["Dec_deg"], target_same["RA_deg"], target_same["Dec_deg"])
    cost_diff = slew_time_seconds(sep_diff,
                                  small_deg=cfg.slew_small_deg,
                                  small_time=cfg.slew_small_time_s,
                                  rate_deg_per_s=cfg.slew_rate_deg_per_s,
                                  settle_s=cfg.slew_settle_s) + cfg.filter_change_time_s
    cost_same = slew_time_seconds(sep_same,
                                  small_deg=cfg.slew_small_deg,
                                  small_time=cfg.slew_small_time_s,
                                  rate_deg_per_s=cfg.slew_rate_deg_per_s,
                                  settle_s=cfg.slew_settle_s)
    assert cost_same < cost_diff
