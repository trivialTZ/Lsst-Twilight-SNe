from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps
from twilight_planner_pkg.simlib_writer import SimlibHeader, SimlibWriter


def _make_subset_csv(src: Path, out: Path, n: int = 2) -> Path:
    df = pd.read_csv(src).head(n)
    df.to_csv(out, index=False)
    return out


def test_simlib_output(tmp_path, monkeypatch):
    data_path = (
        Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    )
    subset_csv = _make_subset_csv(data_path, tmp_path / "subset.csv", n=2)

    from twilight_planner_pkg import scheduler

    def mock_twilight_windows_for_local_night(
        date_local, loc, min_sun_alt_deg=-18.0, max_sun_alt_deg=0.0
    ):
        start_morning = datetime(
            date_local.year,
            date_local.month,
            date_local.day,
            5,
            0,
            0,
            tzinfo=timezone.utc,
        )
        end_morning = start_morning + timedelta(minutes=30)
        return [
            {
                "start": start_morning,
                "end": end_morning,
                "label": "morning",
                "night_date": date_local,
            }
        ]

    def mock_best_time_with_moon(
        sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg
    ):
        start, _ = window
        return 50.0, start + timedelta(minutes=5), 0.0, 0.0, 180.0

    monkeypatch.setattr(
        scheduler,
        "twilight_windows_for_local_night",
        mock_twilight_windows_for_local_night,
    )
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        scheduler,
        "RubinSkyProvider",
        lambda: scheduler.SimpleSkyProvider(scheduler.SkyModelConfig()),
    )

    simlib_path = tmp_path / "plan.SIMLIB"

    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["r"],
        exposure_by_filter={"r": 3.0},
        readout_s=1.0,
        filter_change_s=0.0,
        evening_cap_s=30.0,
        morning_cap_s=30.0,
        max_sn_per_night=1,
        per_sn_cap_s=10.0,
        min_moon_sep_by_filter={"r": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        allow_filter_changes_in_twilight=True,
        simlib_out=str(simlib_path),
    )

    start_date = end_date = "2025-07-30"

    pernight_df, _ = plan_twilight_range_with_caps(
        csv_path=str(subset_csv),
        outdir=str(tmp_path),
        start_date=start_date,
        end_date=end_date,
        cfg=cfg,
        verbose=False,
    )

    assert simlib_path.exists()
    lines = simlib_path.read_text().strip().splitlines()
    s_lines = [line for line in lines if line.startswith("S:")]
    assert len(s_lines) == len(pernight_df)

    # BEGIN LIBGEN should appear after the global header
    assert any(line.strip() == "BEGIN LIBGEN" for line in lines[:50])
    header_line = "#     MJD        ID+NEXPOSE FLT GAIN NOISE SKYSIG PSF1 PSF2 PSFRATIO ZPTAVG ZPTERR"
    libid_indices = [i for i, line in enumerate(lines) if line.startswith("LIBID:")]
    assert lines.count(header_line) == len(libid_indices)

    last_end_line = None
    for idx in libid_indices:
        # LIBID line may include a trailing comment, e.g.,
        # "LIBID:      1     # SN1".  Extract the numeric token.
        libid = lines[idx].split()[1]
        nobs_line = lines[idx + 1]
        assert nobs_line.startswith("NOBS:")
        nobs_val = int(nobs_line.split()[1])

        j = idx + 2
        while j < len(lines) and lines[j] != header_line:
            j += 1
        assert j < len(lines)

        k = j + 1
        s_count = 0
        while k < len(lines) and lines[k].startswith("S:"):
            s_count += 1
            k += 1

        assert lines[k].strip() == "END_LIBID:"
        assert nobs_val == s_count
        last_end_line = "END_LIBID:"

    # Final terminator present and last END_LIBID appears just before it
    assert lines[-1].strip() == "END_OF_SIMLIB:"
    # There may be a blank separator line between END_LIBID and END_OF_SIMLIB
    assert any(line.strip() == last_end_line for line in lines)
    assert any("SURVEY:" in line for line in lines[:30])


def test_writer_groups_epochs(tmp_path):
    path = tmp_path / "out.SIMLIB"
    with path.open("w") as fp:
        writer = SimlibWriter(fp, SimlibHeader())
        writer.write_header()
        writer.start_libid(1, 1.0, 2.0, 1, comment="SN1")
        writer.add_epoch(1.0, "r", 1.0, 1.0, 1.0, 0.8, 0.0, 0.0, 25.0, 0.1)
        writer.end_libid()
        writer.start_libid(2, 1.0, 2.0, 1, comment="SN1")
        writer.add_epoch(2.0, "r", 1.0, 1.0, 1.0, 0.8, 0.0, 0.0, 25.0, 0.1)
        writer.end_libid()
        writer.close()

    lines = path.read_text().splitlines()
    libid_lines = [line for line in lines if line.startswith("LIBID:")]
    assert len(libid_lines) == 1
    # Ensure the LIBID number is 1 even with padded spaces.
    assert libid_lines[0].split()[1] == "1"
    assert "# SN1" in libid_lines[0]
    nobs_line = next(line for line in lines if line.startswith("NOBS:"))
    assert "NOBS: 2" in nobs_line
    s_lines = [line for line in lines if line.startswith("S:")]
    assert len(s_lines) == 2
    parts0 = s_lines[0].split()
    parts1 = s_lines[1].split()
    assert parts0[2] == "1"
    assert parts0[3] == "r"
    assert parts1[2] == "2"
    assert parts1[3] == "r"


def test_writer_no_blank_after_redshift(tmp_path):
    path = tmp_path / "out_redshift.SIMLIB"
    with path.open("w") as fp:
        writer = SimlibWriter(fp, SimlibHeader())
        writer.write_header()
        writer.start_libid(1, 1.0, 2.0, 1, comment="SN1", redshift=0.12345, peakmjd=60000.0)
        writer.add_epoch(1.0, "r", 1.0, 1.0, 1.0, 0.8, 0.0, 0.0, 25.0, 0.1)
        writer.end_libid()
        writer.close()

    lines = path.read_text().splitlines()
    rz_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("REDSHIFT:"))
    assert lines[rz_idx + 1].startswith("#     MJD")
    assert any(line.strip() == "END_LIBID:" for line in lines)


def test_writer_emits_id_plus_nexpose_token(tmp_path):
    path = tmp_path / "out_coadd.SIMLIB"
    with path.open("w") as fp:
        writer = SimlibWriter(fp, SimlibHeader())
        writer.write_header()
        writer.start_libid(1, 1.0, 2.0, 1, comment="SN1")
        writer.add_epoch(1.0, "r", 1.0, 1.0, 1.0, 0.8, 0.0, 0.0, 25.0, 0.1, nexpose=6)
        writer.end_libid()
        writer.close()

    line = next(line for line in path.read_text().splitlines() if line.startswith("S:"))
    tokens = line.split()
    assert tokens[2] == "1*6"
    assert tokens[3] == "r"
    # Ensure no stray token was created between ID and band
    assert "*" in tokens[2]


def test_header_uppercases_y_band():
    header = SimlibHeader(FILTERS="grizy")
    assert header.FILTERS == "grizY"
