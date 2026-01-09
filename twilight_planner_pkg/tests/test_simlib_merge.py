from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pandas as pd
from astropy.time import Time

from twilight_planner_pkg.simlib_merge import merge_simlib_with_twilight
from twilight_planner_pkg.simlib_reader import entry_name, parse_simlib


def _toy_simlib() -> Path:
    return Path(__file__).resolve().parents[2] / "twilight_planner_pkg" / "tests" / "data" / "wfd_toy.SIMLIB"


def _plan_row(name: str, filt: str, mjd: float) -> dict:
    ts = Time(mjd, format="mjd").to_datetime(timezone.utc)
    return {
        "SN": name,
        "RA_deg": 10.0,
        "Dec_deg": -5.0,
        "filter": filt,
        "visit_start_utc": ts.isoformat(),
        "ZPT": 30.0,
        "SKYSIG": 1.0,
        "PSF1_pix": 1.0,
        "PSF2_pix": 0.0,
        "PSFRATIO": 0.0,
        "GAIN": 1.6,
        "RDNOISE": 5.0,
    }


def test_preserves_wfd_libids_and_no_duplicates(tmp_path):
    base_doc = parse_simlib(_toy_simlib())
    original = {e.libid: len(e.epochs) for e in base_doc.entries}

    tw_df = pd.DataFrame([_plan_row("SN2023abc", "i", 60005.0)])
    out_path = merge_simlib_with_twilight(_toy_simlib(), tw_df, tmp_path / "merged.SIMLIB")
    merged = parse_simlib(out_path)

    libids = {e.libid for e in merged.entries}
    assert libids == set(original.keys())
    lib10 = next(e for e in merged.entries if e.libid == 10)
    assert len(lib10.epochs) == original[10] + 1
    assert "SOURCE=WFD+TWILIGHT" in (lib10.comment or "")


def test_twilight_only_targets_get_new_libids(tmp_path):
    tw_df = pd.DataFrame([_plan_row("NewSN", "r", 60010.0)])
    out_path = merge_simlib_with_twilight(_toy_simlib(), tw_df, tmp_path / "merged_new.SIMLIB")
    merged = parse_simlib(out_path)

    libids = sorted(e.libid for e in merged.entries)
    assert max(libids) > max(10, 11)
    new_entry = max(merged.entries, key=lambda e: e.libid)
    assert entry_name(new_entry) == "NewSN"
    assert "SOURCE=TWILIGHT" in (new_entry.comment or "")


def test_header_includes_wfd_documentation(tmp_path):
    tw_df = pd.DataFrame([_plan_row("SN2023abc", "i", 60005.0)])
    out_path = merge_simlib_with_twilight(_toy_simlib(), tw_df, tmp_path / "merged_doc.SIMLIB")
    text = Path(out_path).read_text()

    assert "toy WFD SIMLIB for parser tests" in text
    assert "Merged twilight planner output" in text
    # Keep a single DOCUMENTATION block in the merged output.
    assert text.count("DOCUMENTATION:") == 1
    assert text.count("DOCUMENTATION_END:") == 1
