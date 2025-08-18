import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

# Ensure package root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.io_utils import (
    _infer_units,
    _parse_dec_value,
    _parse_discovery_to_datetime,
    _parse_ra_value,
    normalize_ra_dec_to_degrees,
    resolve_columns,
    standardize_columns,
    unit_report_from_df,
)


def cfg():
    return PlannerConfig(lat_deg=0, lon_deg=0, height_m=0)


def test_resolve_and_standardize_with_csv():
    data_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "data"
        / "ATLAS_2021_to25_cleaned.csv"
    )
    df = pd.read_csv(data_path)
    ra_col, dec_col, disc_col, name_col, type_col = resolve_columns(df, cfg())
    assert (ra_col, dec_col, disc_col, name_col, type_col) == (
        "ra",
        "declination",
        "discoverydate",
        "name",
        "type",
    )
    std = standardize_columns(df, cfg())
    assert {"RA_deg", "Dec_deg", "discovery_datetime"}.issubset(std.columns)
    assert str(std["discovery_datetime"].dtype) == "datetime64[ns, UTC]"


def test_parse_ra_dec_value_variants():
    assert np.isclose(_parse_ra_value(180.0), 180.0)
    assert np.isclose(_parse_ra_value(12.0), 180.0)
    assert np.isclose(_parse_ra_value(np.pi), 180.0)
    assert np.isclose(_parse_ra_value("12h00m00s"), 180.0)

    assert np.isclose(_parse_dec_value(45.0), 45.0)
    assert np.isclose(_parse_dec_value(np.pi / 4), 45.0)
    assert np.isclose(_parse_dec_value("+45d00m00s"), 45.0)


def test_normalize_ra_dec_to_degrees():
    df_deg = pd.DataFrame({"ra": [100.0, 200.0], "dec": [-5.0, 5.0]})
    out_deg = normalize_ra_dec_to_degrees(df_deg, "ra", "dec")
    assert np.allclose(out_deg["RA_deg"], [100.0, 200.0])
    assert np.allclose(out_deg["Dec_deg"], [-5.0, 5.0])

    df_hr = pd.DataFrame({"ra": [1.0, 2.0], "dec": [-5.0, 5.0]})
    out_hr = normalize_ra_dec_to_degrees(df_hr, "ra", "dec")
    assert np.allclose(out_hr["RA_deg"], [15.0, 30.0])
    assert np.allclose(out_hr["Dec_deg"], [-5.0, 5.0])

    df_rad = pd.DataFrame(
        {"ra": ["0.523 rad", "1.047 rad"], "dec": ["-0.523 rad", "0.523 rad"]}
    )
    out_rad = normalize_ra_dec_to_degrees(df_rad, "ra", "dec")
    assert np.allclose(out_rad["RA_deg"], [30.0, 60.0], atol=0.05)
    assert np.allclose(out_rad["Dec_deg"], [-30.0, 30.0], atol=0.05)

    df_str = pd.DataFrame({"ra": ["1h0m0s", "2h0m0s"], "dec": ["-5d0m0s", "5d0m0s"]})
    out_str = normalize_ra_dec_to_degrees(df_str, "ra", "dec")
    assert np.allclose(out_str["RA_deg"], [15.0, 30.0])
    assert np.allclose(out_str["Dec_deg"], [-5.0, 5.0])


def test_infer_units_and_unit_report():
    ra_series = pd.Series([1.0, 2.0])
    dec_series = pd.Series([np.pi / 2])
    ra_unit, dec_unit, notes = _infer_units(ra_series, dec_series)
    assert ra_unit == "hour"
    assert dec_unit == "deg"
    assert any("1.57 rad" in n for n in notes)

    df = pd.DataFrame({"ra": [30.0], "dec": [0.0], "name": ["A"]})
    report = unit_report_from_df(df, cfg())
    assert report["ra"]["unit_inferred"] == "deg"
    assert any("between 24h and 50°" in n for n in report["notes"])


def test_parse_discovery_to_datetime():
    iso_series = pd.Series(["2025-01-01T00:00:00Z", "2025-01-02T12:34:56Z"])
    iso_parsed = _parse_discovery_to_datetime(iso_series)
    expected_iso = pd.to_datetime(iso_series, utc=True)
    pd.testing.assert_series_equal(iso_parsed, expected_iso)

    mjd_series = pd.Series([60000.0, 60001.0])
    orig_to_datetime = Time.to_datetime
    import datetime as datetime_module

    def _patched(self, timezone=None):
        if isinstance(timezone, str) and timezone.lower() == "utc":
            timezone = datetime_module.timezone.utc
        return orig_to_datetime(self, timezone=timezone)

    Time.to_datetime = _patched
    try:
        mjd_parsed = _parse_discovery_to_datetime(mjd_series)
        expected_mjd = pd.Series(
            Time([60000.0, 60001.0], format="mjd").to_datetime(
                timezone=datetime_module.timezone.utc
            )
        )
    finally:
        Time.to_datetime = orig_to_datetime
    mjd_parsed = pd.to_datetime(mjd_parsed, utc=True)
    pd.testing.assert_series_equal(mjd_parsed, expected_mjd)


def test_standardize_columns_normalizes_and_parses():
    df = pd.DataFrame(
        {
            "ra": [400.0, -10.0],
            "dec": [90.0000003, -89.5],
            "discoverydate": [60000.0, 2459123.5],
            "name": ["SN1", "SN2"],
            "type": ["Ia", "II-P"],
        }
    )
    cfg_inst = cfg()
    orig_to_datetime = Time.to_datetime
    import datetime as dt_mod

    def _patched(self, timezone=None):
        if isinstance(timezone, str) and timezone.lower() == "utc":
            timezone = dt_mod.timezone.utc
        return orig_to_datetime(self, timezone=timezone)

    Time.to_datetime = _patched
    try:
        out = standardize_columns(df, cfg_inst)
    finally:
        Time.to_datetime = orig_to_datetime

    assert np.allclose(out["RA_deg"], [40.0, 350.0])
    assert np.allclose(out["Dec_deg"], [90.0, -89.5])
    expo = pd.to_datetime(Time(60000.0, format="mjd").to_datetime(dt_mod.timezone.utc))
    exp1 = pd.to_datetime(Time(2459123.5, format="jd").to_datetime(dt_mod.timezone.utc))
    assert out["discovery_datetime"].iloc[0] == expo
    assert out["discovery_datetime"].iloc[1] == exp1
    assert list(out["typical_lifetime_days"]) == [84, 120]


def test_standardize_columns_invalid_dec_raises():
    df = pd.DataFrame(
        {
            "ra": [10.0, 20.0],
            "dec": [91.0, -100.0],
            "name": ["SN1", "SN2"],
            "type": ["Ia", "II-P"],
        }
    )
    cfg_inst = cfg()
    with pytest.raises(ValueError):
        standardize_columns(df, cfg_inst)
