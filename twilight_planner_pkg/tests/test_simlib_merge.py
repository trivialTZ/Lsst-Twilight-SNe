from __future__ import annotations

from datetime import timezone
import math
from pathlib import Path

import pandas as pd
import pytest
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
        "ZPTERR": 0.01,
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
    twilight = next(epoch for epoch in lib10.epochs if epoch.mjd == 60005.0)
    assert twilight.zperr == 0.01


def test_merge_accepts_legacy_lowercase_zperr(tmp_path):
    row = _plan_row("SN2023abc", "i", 60005.0)
    row["zperr"] = row.pop("ZPTERR")
    out_path = merge_simlib_with_twilight(
        _toy_simlib(), pd.DataFrame([row]), tmp_path / "merged_legacy_zperr.SIMLIB"
    )
    merged = parse_simlib(out_path)
    lib10 = next(entry for entry in merged.entries if entry.libid == 10)
    twilight = next(epoch for epoch in lib10.epochs if epoch.mjd == 60005.0)
    assert twilight.zperr == 0.01


def test_merge_uses_first_finite_zperr_value(tmp_path):
    row = _plan_row("SN2023abc", "i", 60005.0)
    row["zpt_err_mag"] = float("nan")
    row["ZPTERR"] = 0.012
    out_path = merge_simlib_with_twilight(
        _toy_simlib(), pd.DataFrame([row]), tmp_path / "merged_finite_zperr.SIMLIB"
    )
    merged = parse_simlib(out_path)
    lib10 = next(entry for entry in merged.entries if entry.libid == 10)
    twilight = next(epoch for epoch in lib10.epochs if epoch.mjd == 60005.0)
    assert twilight.zperr == 0.012


def test_merge_requires_explicit_default_when_zperr_missing(tmp_path):
    row = _plan_row("SN2023abc", "i", 60005.0)
    row.pop("ZPTERR")
    with pytest.raises(ValueError, match="zeropoint uncertainty"):
        merge_simlib_with_twilight(
            _toy_simlib(), pd.DataFrame([row]), tmp_path / "missing_zperr.SIMLIB"
        )

    out_path = merge_simlib_with_twilight(
        _toy_simlib(),
        pd.DataFrame([row]),
        tmp_path / "default_zperr.SIMLIB",
        default_zpt_err_mag=0.01,
    )
    merged = parse_simlib(out_path)
    lib10 = next(entry for entry in merged.entries if entry.libid == 10)
    twilight = next(epoch for epoch in lib10.epochs if epoch.mjd == 60005.0)
    assert twilight.zperr == 0.01


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


def test_merge_converts_twilight_rows_to_wfd_global_units(tmp_path):
    source = tmp_path / "wfd_with_units.SIMLIB"
    source.write_text(
        _toy_simlib()
        .read_text()
        .replace(
            "PIXSIZE: 0.200",
            "PIXSIZE: 0.200\n"
            "PSF_UNIT: ARCSEC_FWHM\n"
            "SKYSIG_UNIT: ADU_PER_SQARCSEC",
        )
    )
    tw_df = pd.DataFrame([_plan_row("SN2023abc", "i", 60005.0)])
    out_path = merge_simlib_with_twilight(
        source, tw_df, tmp_path / "merged_units.SIMLIB"
    )
    text = out_path.read_text()
    assert "PSF_UNIT: ARCSEC_FWHM" in text
    assert "SKYSIG_UNIT: ADU_PER_SQARCSEC" in text

    merged = parse_simlib(out_path)
    lib10 = next(entry for entry in merged.entries if entry.libid == 10)

    # Existing WFD rows make a source-unit -> native -> source-unit round trip.
    original_wfd = next(epoch for epoch in lib10.epochs if epoch.mjd == 60000.0)
    assert original_wfd.psf1 == 1.0
    assert original_wfd.skysig == 10.0

    # The planner row starts in sigma-pixel and ADU/pixel units.  The merged
    # WFD header requires FWHM arcsec and ADU/sqrt(arcsec^2), respectively.
    twilight = next(epoch for epoch in lib10.epochs if epoch.mjd == 60005.0)
    assert twilight.noise == 5.0
    assert math.isclose(twilight.psf1, 0.471, abs_tol=5e-4)
    assert twilight.skysig == 5.0


def test_merge_adds_explicit_native_units_when_source_uses_defaults(tmp_path):
    tw_df = pd.DataFrame([_plan_row("NewSN", "r", 60010.0)])
    out_path = merge_simlib_with_twilight(
        _toy_simlib(), tw_df, tmp_path / "merged_default_units.SIMLIB"
    )
    text = out_path.read_text()
    assert "PSF_UNIT: PIXEL_SIGMA" in text
    assert "SKYSIG_UNIT: ADU_PER_SQPIXEL" in text
