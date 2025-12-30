from __future__ import annotations

from pathlib import Path

from twilight_planner_pkg.simlib_reader import (
    SimlibDocument,
    parse_simlib,
    simlib_to_catalog_df,
    simlib_visits_by_name,
)


def _toy_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "wfd_toy.SIMLIB"


def test_parses_match_tns_name():
    df = simlib_to_catalog_df(_toy_path())
    # MATCH_TNS line should win over generic LIBID-derived name.
    row = df.loc[df["wfd_libid"] == 10].iloc[0]
    assert row["Name"] == "SN2023abc"
    assert row["RA_deg"] == 10.0
    assert row["Dec_deg"] == -5.0
    assert abs(row["redshift"] - 0.12345) < 1e-6
    # LIBID without MATCH_TNS falls back to deterministic placeholder.
    fallback = df.loc[df["wfd_libid"] == 11].iloc[0]
    assert fallback["Name"] == "WFD_LIBID_11"
    assert fallback["source"] == "WFD"


def test_extracts_visits_by_name_and_band():
    visits = simlib_visits_by_name(_toy_path())
    assert set(visits.keys()) == {"SN2023abc", "WFD_LIBID_11"}
    # y-band should be normalized from uppercase Y.
    assert visits["SN2023abc"] == {
        "g": [60000.0],
        "i": [60001.0],
        "y": [60002.5],
    }
    assert visits["WFD_LIBID_11"]["r"] == [60003.0, 60004.0]


def test_parse_round_trips_document_structure():
    doc = parse_simlib(_toy_path())
    assert isinstance(doc, SimlibDocument)
    assert doc.documentation  # DOCUMENTATION block captured
    assert any("BEGIN LIBGEN" in h for h in doc.global_header)
    assert len(doc.entries) == 2
    assert doc.entries[0].peakmjd == 60000.0
