from __future__ import annotations

from pathlib import Path

import pandas as pd

from twilight_planner_pkg.simlib_split import split_twilight_plan_by_source


def _count_prefix(lines: list[str], prefix: str) -> int:
    return sum(1 for line in lines if line.startswith(prefix))


def test_split_twilight_plan_by_source_writes_two_simlibs(tmp_path: Path) -> None:
    plan_df = pd.DataFrame(
        [
            {
                "SN": "TNS2023abc",
                "visit_start_utc": "2023-03-01T01:02:03Z",
                "filter": "r",
                "RA_deg": 10.0,
                "Dec_deg": -10.0,
                "GAIN": 1.6,
                "RDNOISE": 3.0,
                "SKYSIG": 12.0,
                "PSF1_pix": 0.8,
                "PSF2_pix": 0.8,
                "PSFRATIO": 1.0,
                "ZPT": 27.0,
            },
            {
                "SN": "WFD_LIBID_42",
                "visit_start_utc": "2023-03-01T01:02:04Z",
                "filter": "i",
                "RA_deg": 20.0,
                "Dec_deg": -20.0,
                "GAIN": 1.6,
                "RDNOISE": 3.0,
                "SKYSIG": 11.0,
                "PSF1_pix": 0.8,
                "PSF2_pix": 0.8,
                "PSFRATIO": 1.0,
                "ZPT": 27.0,
            },
        ]
    )
    catalog_df = pd.DataFrame(
        [
            {"Name": "TNS2023abc", "source": "TNS"},
            {"Name": "WFD_LIBID_42", "source": "WFD", "wfd_libid": 42},
        ]
    )

    out_tns = tmp_path / "tns.simlib"
    out_wfd = tmp_path / "wfd.simlib"
    split_twilight_plan_by_source(
        plan_df,
        catalog_df,
        out_tns_path=out_tns,
        out_wfd_path=out_wfd,
        preserve_wfd_libids=True,
        default_zpt_err_mag=0.01,
    )

    assert out_tns.exists()
    assert out_wfd.exists()

    tns_lines = out_tns.read_text().splitlines()
    wfd_lines = out_wfd.read_text().splitlines()

    assert _count_prefix(tns_lines, "LIBID:") == 1
    assert any(line.startswith("LIBID:") and "TNS2023abc" in line for line in tns_lines)
    assert _count_prefix(tns_lines, "S:") == 1

    assert _count_prefix(wfd_lines, "LIBID:") == 1
    libid_line = next(line for line in wfd_lines if line.startswith("LIBID:"))
    assert libid_line.split()[1] == "42"
    assert "WFD_LIBID_42" in libid_line
    assert _count_prefix(wfd_lines, "S:") == 1

