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

