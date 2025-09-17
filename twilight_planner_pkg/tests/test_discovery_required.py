import pytest
import pandas as pd

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.io_utils import build_mag_lookup_with_fallback, standardize_columns


def _mk_df(rows):
    return pd.DataFrame(rows)


def test_missing_discoverymag_raises_error():
    cfg = PlannerConfig(filters=["r", "i"], discovery_policy="atlas_priors", discovery_error_on_missing=True)
    df = _mk_df([
        {"ra": 1.0, "dec": 1.0, "Name": "SNX", "discmagfilter": "orange"},  # no discoverymag
    ])
    std = standardize_columns(df, cfg)
    with pytest.raises(ValueError):
        build_mag_lookup_with_fallback(std, cfg)


def test_non_numeric_discoverymag_raises_error():
    cfg = PlannerConfig(filters=["r", "i"], discovery_policy="atlas_priors", discovery_error_on_missing=True)
    df = _mk_df([
        {"ra": 1.0, "dec": 1.0, "Name": "SNX", "discmagfilter": "cyan", "discoverymag": "bad"},
    ])
    std = standardize_columns(df, cfg)
    with pytest.raises(ValueError):
        build_mag_lookup_with_fallback(std, cfg)


def test_unknown_filter_still_produces_values():
    cfg = PlannerConfig(filters=["r", "i", "z", "y"], discovery_policy="atlas_priors", discovery_error_on_missing=True)
    df = _mk_df([
        {"ra": 2.0, "dec": 2.0, "Name": "SNY", "discmagfilter": "unknown", "discoverymag": 18.5},
    ])
    std = standardize_columns(df, cfg)
    mags = build_mag_lookup_with_fallback(std, cfg)
    assert set(mags["SNY"].keys()) == {"r", "i", "z", "y"}
