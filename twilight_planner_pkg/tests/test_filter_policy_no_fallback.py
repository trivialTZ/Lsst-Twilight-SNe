import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.filter_policy import allowed_filters_for_window


def test_no_heuristic_fallback_when_none_pass():
    # Extremely bright twilight: Sun above horizon; very high airmass
    mags = {"r": 25.0}  # faint target to force rejection
    allowed = allowed_filters_for_window(
        mags, 0.0, 10.0, 0.9, 5.0, 2.5, 1.5
    )
    # Must not inject heuristic bands; should be empty when m5 fails
    assert allowed == []

