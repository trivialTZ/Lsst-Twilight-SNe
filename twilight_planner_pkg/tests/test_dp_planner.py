import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import (
    PairItem,
    _plan_batches_by_dp,
    _prefix_scores,
)


def _make_pair(name, filt, score, approx_time_s):
    return PairItem(
        name=name,
        filt=filt,
        score=score,
        approx_time_s=approx_time_s,
        density=score / max(approx_time_s, 1e-3),
        snr_margin=0.0,
        exp_s=approx_time_s / 2.0,
        candidate={"Name": name, "policy_allowed": [filt], "allowed": [filt]},
    )


def test_dp_prefers_single_filter_when_swap_payoff_low():
    cfg = PlannerConfig()
    cfg.filter_change_s = 120.0
    cfg.swap_boost = 1.0
    cfg.dp_hysteresis_theta = 0.0
    per_filter = {
        "g": [
            _make_pair("g1", "g", 10.0, 30.0),
            _make_pair("g2", "g", 9.5, 30.0),
            _make_pair("g3", "g", 9.0, 30.0),
        ],
        "r": [
            _make_pair("r1", "r", 8.0, 30.0),
            _make_pair("r2", "r", 7.5, 30.0),
            _make_pair("r3", "r", 7.0, 30.0),
        ],
    }
    prefix = _prefix_scores(per_filter)
    seq, counts, total = _plan_batches_by_dp(per_filter, prefix, 600.0, cfg, None)
    assert seq == ["g"]
    assert counts == [3]
    assert total > 0


def test_dp_accepts_swap_when_second_filter_dominates():
    cfg = PlannerConfig()
    cfg.filter_change_s = 90.0
    cfg.swap_boost = 1.0
    cfg.dp_hysteresis_theta = 0.0
    per_filter = {
        "g": [
            _make_pair("g1", "g", 6.0, 30.0),
            _make_pair("g2", "g", 5.5, 30.0),
        ],
        "r": [
            _make_pair("r1", "r", 15.0, 30.0),
            _make_pair("r2", "r", 14.5, 30.0),
        ],
    }
    prefix = _prefix_scores(per_filter)
    seq, counts, total = _plan_batches_by_dp(per_filter, prefix, 600.0, cfg, None)
    assert seq == ["g", "r"]
    assert counts[0] >= 0
    assert counts[1] > 0
    assert total > 0
