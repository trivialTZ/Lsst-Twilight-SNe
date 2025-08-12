import pathlib, sys
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.astro_utils import (
    slew_time_seconds,
    great_circle_sep_deg,
    choose_filters_with_cap,
)

def test_filter_aware_cost_prefers_same_filter():
    cfg = PlannerConfig()
    prev = {"RA_deg": 0.0, "Dec_deg": 0.0, "first_filter": "z"}
    target_diff = {"RA_deg": 1.0, "Dec_deg": 0.0, "first_filter": "i"}
    target_same = {"RA_deg": 1.5, "Dec_deg": 0.0, "first_filter": "z"}
    sep_diff = great_circle_sep_deg(prev["RA_deg"], prev["Dec_deg"], target_diff["RA_deg"], target_diff["Dec_deg"])
    sep_same = great_circle_sep_deg(prev["RA_deg"], prev["Dec_deg"], target_same["RA_deg"], target_same["Dec_deg"])
    cost_diff = slew_time_seconds(
        sep_diff,
        small_deg=cfg.slew_small_deg,
        small_time=cfg.slew_small_time_s,
        rate_deg_per_s=cfg.slew_rate_deg_per_s,
        settle_s=cfg.slew_settle_s,
    ) + cfg.filter_change_s
    cost_same = slew_time_seconds(sep_same,
                                  small_deg=cfg.slew_small_deg,
                                  small_time=cfg.slew_small_time_s,
                                  rate_deg_per_s=cfg.slew_rate_deg_per_s,
                                  settle_s=cfg.slew_settle_s)
    assert cost_same < cost_diff


def test_routing_groups_same_filters_and_counts_swaps():
    cfg = PlannerConfig(filters=["g", "i"], start_filter="i")
    targets = [
        {"Name": "SN1", "RA_deg": 0.0, "Dec_deg": 0.0, "first_filter": "i"},
        {"Name": "SN2", "RA_deg": 10.0, "Dec_deg": 0.0, "first_filter": "i"},
        {"Name": "SN3", "RA_deg": 0.5, "Dec_deg": 0.0, "first_filter": "g"},
    ]
    # naive nearest-neighbour ignoring filter swaps
    remaining = targets[1:].copy()
    current = targets[0]
    order_naive = [current["Name"]]
    while remaining:
        dists = [
            great_circle_sep_deg(current["RA_deg"], current["Dec_deg"], t["RA_deg"], t["Dec_deg"])
            for t in remaining
        ]
        j = int(np.argmin(dists))
        current = remaining.pop(j)
        order_naive.append(current["Name"])

    # filter-aware route
    remaining = targets[1:].copy()
    current = targets[0]
    order = [current["Name"]]
    state = cfg.start_filter
    swap_count = 0
    filter_change_s = 0.0
    while remaining:
        costs = []
        for t in remaining:
            sep = great_circle_sep_deg(current["RA_deg"], current["Dec_deg"], t["RA_deg"], t["Dec_deg"])
            cost = slew_time_seconds(
                sep,
                small_deg=cfg.slew_small_deg,
                small_time=cfg.slew_small_time_s,
                rate_deg_per_s=cfg.slew_rate_deg_per_s,
                settle_s=cfg.slew_settle_s,
            )
            if state is not None and state != t["first_filter"]:
                cost += cfg.filter_change_s
            costs.append(cost)
        j = int(np.argmin(costs))
        t = remaining.pop(j)
        sep = great_circle_sep_deg(current["RA_deg"], current["Dec_deg"], t["RA_deg"], t["Dec_deg"])
        used, timing = choose_filters_with_cap([t["first_filter"]], sep, 1000.0, cfg, current_filter=state)
        if state is not None and used[0] != state:
            swap_count += 1
        state = used[-1]
        filter_change_s += timing["filter_changes_s"]
        order.append(t["Name"])
        current = t

    assert order_naive == ["SN1", "SN3", "SN2"]
    assert order == ["SN1", "SN2", "SN3"]
    assert swap_count == 1
    assert filter_change_s == pytest.approx(cfg.filter_change_s)
