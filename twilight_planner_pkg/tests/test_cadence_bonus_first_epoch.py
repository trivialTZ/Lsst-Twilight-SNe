from twilight_planner_pkg.priority import PriorityTracker


def test_first_epoch_bonus_default_zero():
    t = PriorityTracker()
    b0 = t.cadence_bonus(
        "S",
        "r",
        now_mjd=1.0,
        target_d=3.0,
        sigma_d=0.5,
        weight=0.25,
        first_epoch_weight=0.0,
    )
    assert b0 == 0.0
    b1 = t.cadence_bonus(
        "S",
        "r",
        now_mjd=1.0,
        target_d=3.0,
        sigma_d=0.5,
        weight=0.25,
        first_epoch_weight=0.05,
    )
    assert 0.0 < b1 <= 0.25
