import numpy as np
import pandas as pd

from twilight_planner_pkg.io_utils import (
    normalize_filter_name,
    map_to_lsst,
    read_discovery_csv,
)
from twilight_planner_pkg.config import DISCOVERY_LINEAR_COEFFS


def test_normalize_filter_name_variants():
    assert normalize_filter_name("w-P1") == "PS1_w"
    assert normalize_filter_name("L-GOTO") == "GOTO_L"
    assert normalize_filter_name("BG-q-BlackGem") == "BG_q"
    assert normalize_filter_name("Clear-") == "Clear"
    assert normalize_filter_name("V-crts-CRTS") == "CRTS_V"
    assert normalize_filter_name("g-ZTF") == "g"
    assert normalize_filter_name("r-ZTF") == "r"


def test_map_to_lsst_coefficients_application():
    gr = 0.5
    ri = 0.2
    m = 19.0

    # PS1_w → r with a+b*(g-r)+c*(r-i)
    d = DISCOVERY_LINEAR_COEFFS["PS1_w"]
    band, mt = map_to_lsst("PS1_w", m, {"g-r": gr, "r-i": ri})
    expect = m + d["a"] + d["b"] * gr + d.get("c", 0.0) * ri
    assert band == d["target"]
    assert abs(mt - expect) < 1e-6

    # GOTO_L → g with a+b*(g-r)
    d2 = DISCOVERY_LINEAR_COEFFS["GOTO_L"]
    band2, mt2 = map_to_lsst("GOTO_L", m, {"g-r": gr, "r-i": ri})
    expect2 = m + d2["a"] + d2["b"] * gr
    assert band2 == d2["target"]
    assert abs(mt2 - expect2) < 1e-6

    # BG_q → r with a+b*(g-r)
    d3 = DISCOVERY_LINEAR_COEFFS["BG_q"]
    band3, mt3 = map_to_lsst("BG_q", m, {"g-r": gr, "r-i": ri})
    expect3 = m + d3["a"] + d3["b"] * gr
    assert band3 == d3["target"]
    assert abs(mt3 - expect3) < 1e-6


def test_read_discovery_csv_mixed_filters(tmp_path):
    # Build a minimal mixed-filter CSV
    rows = [
        {"mjd": 60000.0, "mag": 19.5, "magerr": 0.1, "filter": "w-P1", "ra": 10.0, "dec": -10.0, "survey": "PS1"},
        {"mjd": 60000.1, "mag": 18.9, "magerr": 0.1, "filter": "L-GOTO", "ra": 10.1, "dec": -10.1, "survey": "GOTO"},
        {"mjd": 60000.2, "mag": 19.0, "magerr": 0.1, "filter": "BG-q-BlackGem", "ra": 10.2, "dec": -10.2, "survey": "BG"},
        {"mjd": 60000.3, "mag": 18.0, "magerr": 0.1, "filter": "Clear-", "ra": 10.3, "dec": -10.3, "survey": "Misc"},
        {"mjd": 60000.4, "mag": 17.5, "magerr": 0.1, "filter": "V-crts-CRTS", "ra": 10.4, "dec": -10.4, "survey": "CRTS"},
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / "mixed.csv"
    df.to_csv(p, index=False)

    out = read_discovery_csv(str(p))
    # Expected columns present
    for c in ["mjd", "ra", "dec", "lsst_band", "mag_lsst", "magerr"]:
        assert c in out.columns
    # Bands restricted to g/r
    assert set(out["lsst_band"]).issubset({"g", "r"})
    # All transformed magnitudes populated (finite)
    assert np.isfinite(out["mag_lsst"]).all()
