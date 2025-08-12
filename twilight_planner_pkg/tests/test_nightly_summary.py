import pathlib
import sys

import pandas as pd
import pytest

# Ensure package root importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def test_nightly_summary_fields(tmp_path):
    df = pd.DataFrame(
        {
            "ra": [0.0],
            "dec": [0.0],
            "discoverydate": ["2023-12-01T00:00:00Z"],
            "name": ["SN1"],
            "type": ["Ia"],
        }
    )
    csv = tmp_path / "cat.csv"
    df.to_csv(csv, index=False)
    cfg = PlannerConfig(filters=["i", "z"], morning_cap_s=1000.0, evening_cap_s=1000.0)
    pernight, nights = plan_twilight_range_with_caps(
        str(csv), tmp_path, "2024-01-01", "2024-01-01", cfg, verbose=False
    )
    row = nights.iloc[0]
    assert row["swap_count"] == 0
    assert row["internal_filter_changes"] == 0
    assert row["filter_change_s_total"] == 0.0
    mean_expected = pernight["slew_s"].mean()
    assert row["mean_slew_s"] == pytest.approx(mean_expected)
    am_expected = pernight["airmass"].median()
    assert row["median_airmass"] == pytest.approx(am_expected, rel=1e-3)
    assert row["filters_used_csv"] == ",".join(sorted(pernight["filter"].unique()))
    assert row["n_planned"] == 1
