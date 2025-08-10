import pathlib, sys

# Ensure package root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.priority import PriorityTracker


def tracker():
    return PriorityTracker(hybrid_detections=2, hybrid_exposure_s=300,
                           lc_detections=5, lc_exposure_s=300)


def test_non_ia_drops_after_hybrid():
    t = tracker()
    t.record_detection("SN1", 150, ["g"])
    assert t.score("SN1", sn_type="II") == 1.0
    t.record_detection("SN1", 150, ["r"])
    assert t.score("SN1", sn_type="II") == 0.0


def test_hybrid_exposure_triggers_escalation():
    t = tracker()
    t.record_detection("SN2", 150, ["g"])
    t.record_detection("SN2", 150, ["g"])
    assert t.score("SN2", sn_type="II") == 0.0
    t.record_detection("SN3", 150, ["g"])
    t.record_detection("SN3", 150, ["g"])
    assert t.score("SN3", sn_type="Ia") == 1.0
    assert t.history["SN3"].escalated is True


def test_ia_requires_lc_goal():
    t = tracker()
    t.record_detection("SN4", 100, ["g"])
    t.record_detection("SN4", 100, ["r"])
    assert t.score("SN4", sn_type="Ia") == 1.0
    t.record_detection("SN4", 50, ["g"])
    t.record_detection("SN4", 50, ["r"])
    t.record_detection("SN4", 50, ["g"])
    assert t.score("SN4", sn_type="Ia") == 0.0


def test_update_aliases_record_detection():
    assert PriorityTracker.update is PriorityTracker.record_detection


def test_score_strategy_lc_escalates_without_detections():
    t = tracker()
    assert t.score("SN5", strategy="lc") == 1.0
    assert t.history["SN5"].escalated is True


def test_score_zero_after_lc_completion_regardless_type():
    t = tracker()
    for _ in range(5):
        t.record_detection("SN6", 60, ["g"])
    assert t.score("SN6", strategy="lc") == 0.0
    for typ in (None, "Ia", "II"):
        assert t.score("SN6", sn_type=typ) == 0.0


def test_repeated_filter_usage_does_not_meet_hybrid():
    t = tracker()
    t.record_detection("SN7", 150, ["g", "g"])
    assert t.score("SN7", sn_type="II") == 1.0
