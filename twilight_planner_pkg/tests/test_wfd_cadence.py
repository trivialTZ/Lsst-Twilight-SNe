from __future__ import annotations

import math

from twilight_planner_pkg.priority import PriorityTracker


def test_future_far_does_not_change_bonus():
    base = PriorityTracker()
    base.record_detection("SN1", 30.0, ["g"], mjd=1.0)
    now = 2.0
    bonus_base = base.cadence_bonus(
        "SN1",
        "g",
        now_mjd=now,
        target_d=3.0,
        sigma_d=0.5,
        weight=1.0,
        first_epoch_weight=0.0,
    )

    with_future = PriorityTracker()
    with_future.external_visits_by_name = {"SN1": {"g": [20.0]}}
    with_future.record_detection("SN1", 30.0, ["g"], mjd=1.0)
    bonus_future = with_future.cadence_bonus(
        "SN1",
        "g",
        now_mjd=now,
        target_d=3.0,
        sigma_d=0.5,
        weight=1.0,
        first_epoch_weight=0.0,
    )

    assert math.isclose(bonus_base, bonus_future, rel_tol=1e-9, abs_tol=0.0)


def test_future_near_blocks_gate():
    tracker = PriorityTracker()
    tracker.external_visits_by_name = {"SN2": {"r": [5.5]}}
    now = 5.0
    assert not tracker.cadence_gate("SN2", "r", now_mjd=now, target_d=3.0, jitter_d=0.25)


def test_prev_near_blocks_gate():
    tracker = PriorityTracker()
    tracker.external_visits_by_name = {"SN3": {"i": [10.0]}}
    tracker.record_detection("SN3", 30.0, ["i"], mjd=1.0)
    now = 1.5
    assert not tracker.cadence_gate("SN3", "i", now_mjd=now, target_d=3.0, jitter_d=0.25)
