import pandas as pd

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.io_utils import (
    build_mag_lookup_with_fallback,
    standardize_columns,
)


def _mk_df(rows):
    return pd.DataFrame(rows)


def test_orange_with_priors_minimizes_r_and_extrapolates():
    cfg = PlannerConfig(
        filters=["g", "r", "i", "z", "y"],
        discovery_policy="atlas_priors",
        discovery_y_extra_margin_mag=0.25,
    )
    df = _mk_df(
        [
            {
                "ra": 10.0,
                "dec": -10.0,
                "Name": "SN1",
                "discoverymag": 17.0,
                "discmagfilter": "orange",
                "SN_type_raw": "Ia",
            }
        ]
    )
    std = standardize_columns(df, cfg)
    mags = build_mag_lookup_with_fallback(std, cfg)
    m = mags["SN1"]
    # r expected slightly brighter than discovery due to beta_o>0 and using g-r min
    assert m["r"] < 17.0
    # y should include extra margin relative to z
    assert m["y"] < m["z"]


def test_g_discovery_chooses_color_extreme_that_brightens_r():
    cfg = PlannerConfig(filters=["r"], discovery_policy="atlas_priors")
    df = _mk_df([
        {"ra": 1.0, "dec": 1.0, "Name": "SN2", "discoverymag": 18.0, "discmagfilter": "g"}
    ])
    std = standardize_columns(df, cfg)
    mags = build_mag_lookup_with_fallback(std, cfg)
    r_prior = mags["SN2"]["r"]
    # With r = g - (g-r), choosing the upper bound of (g-r) lowers r (brighter)
    # g-r upper bound is 0.15 → r = 18 - 0.15 = 17.85
    assert abs(r_prior - 17.85) < 1e-6


def test_non_ia_widen_produces_brighter_target():
    base_cfg = PlannerConfig(filters=["r"], discovery_policy="atlas_priors", discovery_non_ia_widen_mag=0.0)
    wide_cfg = PlannerConfig(filters=["r"], discovery_policy="atlas_priors", discovery_non_ia_widen_mag=0.2)
    df = _mk_df([
        {"ra": 2.0, "dec": 2.0, "Name": "SN3", "discoverymag": 17.5, "discmagfilter": "orange", "SN_type_raw": "II"}
    ])
    std = standardize_columns(df, base_cfg)
    m0 = build_mag_lookup_with_fallback(std, base_cfg)["SN3"]["r"]
    std2 = standardize_columns(df, wide_cfg)
    m1 = build_mag_lookup_with_fallback(std2, wide_cfg)["SN3"]["r"]
    # Wider prior should choose a more extreme color → brighter r (numerically smaller)
    assert m1 < m0

