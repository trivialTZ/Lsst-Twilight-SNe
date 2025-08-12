import pathlib, sys
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.reports import summarize_night


def make_plan_df():
    return pd.DataFrame(
        {
            "Name": ["SN1", "SN1", "SN2"],
            "filter": ["g", "r", "r"],
            "exposure_s": [30.0, 30.0, 30.0],
            "readout_s": [2.0, 2.0, 2.0],
            "filter_changes_s": [1.0, 1.0, 1.0],
            "slew_s": [1.0, 1.0, 1.0],
            "airmass": [1.1, 1.2, 1.3],
            "moon_sep": [40.0, 50.0, 60.0],
        }
    )


def test_summarize_night_basic(tmp_path, monkeypatch):
    plan = make_plan_df()
    out_dir = tmp_path / "twilight_outputs"
    out_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    m = summarize_night(plan, twilight_window_s=600)
    assert 0.0 <= m["science_efficiency"] <= 1.0
    assert m["color_completeness_frac"] >= 0.5
    csv_path = tmp_path / "twilight_outputs" / "nightly_metrics.csv"
    assert csv_path.exists()
