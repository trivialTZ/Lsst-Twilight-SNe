from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from Motivation.tw_convert_ra_dec import fix_ra_dec_from_simlib


def _write_simlib(path: Path) -> Path:
    path.write_text(
        "DOCUMENTATION:\nDOCUMENTATION_END:\n"
        "SURVEY: TEST\nFILTERS: griz\nNLIBID: 1\nBEGIN LIBGEN\n"
        "LIBID: 7\nNOBS: 1 RA: 123.4 DEC: -45.6\n"
        "REDSHIFT: 0.03000 PEAKMJD: 60001.0\n"
        "S: 60000.0 1 r 1.0 1.0 1.0 0.8 0.0 0.0 30.0 0.01\n"
        "END_LIBID:\nEND_OF_SIMLIB:\n"
    )
    return path


def test_require_all_mapped_accepts_complete_mapping(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"SIM_LIBID": [7], "RA": [9.0], "DEC": [-9.0], "z": [0.03]}
    )
    fixed, stats = fix_ra_dec_from_simlib(
        frame,
        _write_simlib(tmp_path / "one.SIMLIB"),
        require_all_mapped=True,
        redshift_col="z",
        require_redshift_match=True,
    )
    assert fixed.loc[0, "RA"] == 123.4
    assert fixed.loc[0, "DEC"] == -45.6
    assert stats["n_rows_mapped"] == 1
    assert stats["n_rows_unmapped"] == 0
    assert stats["n_redshift_compared"] == 1
    assert stats["n_redshift_mismatch"] == 0


def test_require_all_mapped_rejects_stale_simlib(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"SIM_LIBID": [8], "RA": [9.0], "DEC": [-9.0], "z": [0.03]}
    )
    with pytest.raises(ValueError, match="mapped 0/1 rows"):
        fix_ra_dec_from_simlib(
            frame,
            _write_simlib(tmp_path / "one.SIMLIB"),
            require_all_mapped=True,
        )


def test_require_redshift_match_rejects_reassigned_libid(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"SIM_LIBID": [7], "RA": [9.0], "DEC": [-9.0], "z": [0.08]}
    )
    with pytest.raises(ValueError, match="found 1 mismatches"):
        fix_ra_dec_from_simlib(
            frame,
            _write_simlib(tmp_path / "one.SIMLIB"),
            require_all_mapped=True,
            redshift_col="z",
            require_redshift_match=True,
        )
