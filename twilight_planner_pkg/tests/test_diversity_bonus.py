import pathlib
import sys

# Ensure package root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.priority import PriorityTracker


def test_diversity_bonus_prefers_under_observed_band():
    tracker = PriorityTracker()

    # Record a single 'g' visit for SN_X at MJD=100.0
    tracker.record_detection("SN_X", 5.0, ["g"], mjd=100.0)

    now = 105.0  # within 5-day window

    # Compute bonuses for g vs r using diversity mode with target 1 per band
    # Since g already has 1 visit and r has 0, the r bonus should be larger.
    kw = dict(
        name="SN_X",
        now_mjd=now,
        target_d=3.0,
        sigma_d=0.5,
        cadence_weight=0.25,
        first_epoch_weight=0.0,
        cosmo_weight_by_filter={"g": 1.0, "r": 1.0, "i": 1.0, "z": 1.0},
        target_pairs=2,
        window_days=5.0,
        alpha=0.3,
        first_epoch_color_boost=1.2,
        diversity_enable=True,
        diversity_target_per_filter=1,
        diversity_window_days=5.0,
        diversity_alpha=0.5,
    )

    b_g = tracker.compute_filter_bonus(filt="g", **kw)
    b_r = tracker.compute_filter_bonus(filt="r", **kw)

    assert b_r > b_g

