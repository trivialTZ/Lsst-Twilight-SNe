import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.priority import PriorityTracker


def test_hybrid_to_lc_path():
    t = PriorityTracker(hybrid_detections=2, lc_detections=3)
    assert t.score("SN1", sn_type="Ia") == 1.0
    t.record_detection("SN1", 100, ["g"])
    t.record_detection("SN1", 100, ["r"])
    assert t.score("SN1", sn_type="Ia") == 1.0  # escalated to LC
    for _ in range(3):
        t.record_detection("SN1", 60, ["g"])
    assert t.score("SN1", sn_type="Ia") == 0.0
