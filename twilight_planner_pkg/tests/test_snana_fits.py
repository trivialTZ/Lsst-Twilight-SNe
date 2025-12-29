from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.snana_fits import (
    annotate_phot_with_snid,
    build_combined_dataframe,
    infer_pointer_base,
)


def _make_fake_head_phot():
    marker = -777.0

    phot = np.zeros(
        7,
        dtype=[
            ("MJD", "f8"),
            ("BAND", "S8"),
            ("FLUXCAL", "f4"),
        ],
    )
    phot["BAND"] = b"LSST-r"
    phot["MJD"] = [60000.0, 60001.0, marker, 60010.0, 60011.0, 60012.0, marker]
    phot["FLUXCAL"] = [10, 11, 0, 20, 21, 22, 0]

    head = np.zeros(
        2,
        dtype=[
            ("SNID", "i8"),
            ("NOBS", "i8"),
            ("PTROBS_MIN", "i8"),
            ("PTROBS_MAX", "i8"),
            ("RA", "f8"),
        ],
    )
    head[0] = (101, 2, 1, 2, 0.1)
    head[1] = (202, 3, 4, 6, 0.2)

    return head, phot


def test_infer_pointer_base_one_based():
    head, phot = _make_fake_head_phot()
    assert infer_pointer_base(head, phot) == 1


def test_annotate_phot_with_snid_counts_and_markers():
    head, phot = _make_fake_head_phot()
    snid_by_row = annotate_phot_with_snid(head, phot, pointer_base="auto", validate=True)

    assert list(snid_by_row) == [101, 101, -1, 202, 202, 202, -1]


def test_build_combined_dataframe_drops_marker_rows():
    head, phot = _make_fake_head_phot()
    df = build_combined_dataframe(
        head,
        phot,
        head_columns=["RA"],
        phot_columns=["MJD", "BAND", "FLUXCAL"],
        pointer_base="auto",
        validate=True,
    )

    assert len(df) == 5
    assert set(df["SNID"].unique()) == {101, 202}
    assert (df["MJD"].astype(float) != -777.0).all()
    assert "RA" in df.columns
