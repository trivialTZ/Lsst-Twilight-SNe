import pandas as pd

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.io_utils import (
    build_mag_lookup_with_fallback,
    standardize_columns,
)


def _mk_df(rows):
    return pd.DataFrame(rows)


def test_atlas_orange_to_r_and_copy_others():
    cfg = PlannerConfig(filters=["r", "i"], discovery_policy="atlas_transform")  # planned bands
    df = _mk_df(
        [
            {
                "ra": 10.0,
                "dec": -10.0,
                "Name": "SN1",
                "discoverymag": 17.0,
                "discmagfilter": "orange",
            }
        ]
    )
    std = standardize_columns(df, cfg)
    mags = build_mag_lookup_with_fallback(std, cfg)
    assert "SN1" in mags
    rmag = mags["SN1"]["r"]
    imag = mags["SN1"]["i"]
    # With assumed g-r=0, r ~= discoverymag
    assert abs(rmag - 17.0) < 1e-6
    # Other bands are copied with margin (default 0.2)
    assert abs(imag - (rmag - 0.2)) < 1e-6


def test_unknown_filter_falls_back_to_copy_policy_like():
    cfg = PlannerConfig(filters=["r", "i"], discovery_policy="atlas_transform")  # planned bands
    df = _mk_df(
        [
            {
                "ra": 11.0,
                "dec": -9.0,
                "Name": "SN2",
                "discoverymag": 16.0,
                "discmagfilter": "unknown",
            }
        ]
    )
    std = standardize_columns(df, cfg)
    mags = build_mag_lookup_with_fallback(std, cfg)
    rmag = mags["SN2"]["r"]
    imag = mags["SN2"]["i"]
    # r is dm - margin, i is r - margin (conservative)
    assert abs(rmag - 15.8) < 1e-6
    assert abs(imag - 15.6) < 1e-6


def test_explicit_r_filter_keeps_value():
    cfg = PlannerConfig(filters=["r", "i"], discovery_policy="atlas_transform")  # planned bands
    df = _mk_df(
        [
            {
                "ra": 12.0,
                "dec": -8.0,
                "Name": "SN3",
                "discoverymag": 18.0,
                "discmagfilter": "r",
            }
        ]
    )
    std = standardize_columns(df, cfg)
    mags = build_mag_lookup_with_fallback(std, cfg)
    assert abs(mags["SN3"]["r"] - 18.0) < 1e-6
    assert abs(mags["SN3"]["i"] - 17.8) < 1e-6
