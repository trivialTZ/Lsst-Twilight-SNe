import pandas as pd
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def test_carousel_capacity_enforced(tmp_path):
    df = pd.DataFrame({
        "ra": [0.0],
        "dec": [0.0],
        "discoverydate": ["2023-12-01T00:00:00Z"],
        "name": ["SN1"],
        "type": ["Ia"],
    })
    csv = tmp_path / "cat.csv"
    df.to_csv(csv, index=False)
    cfg = PlannerConfig(filters=["u", "g", "r", "i", "z", "y"], carousel_capacity=5,
                        morning_cap_s=100.0, evening_cap_s=100.0)
    _, nights = plan_twilight_range_with_caps(str(csv), tmp_path, "2024-01-01", "2024-01-01", cfg, verbose=False)
    assert nights["loaded_filters"].str.contains("u").sum() == 0
    assert all(len(row.split(",")) <= cfg.carousel_capacity for row in nights["loaded_filters"])
