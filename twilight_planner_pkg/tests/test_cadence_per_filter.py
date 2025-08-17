from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.priority import PriorityTracker
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def _subset_csv(src: Path, out: Path) -> Path:
    df = pd.read_csv(src).head(1)
    df.to_csv(out, index=False)
    return out


def _mock_windows(date_local, loc=None, min_sun_alt_deg=-18.0, max_sun_alt_deg=0.0):
    start = datetime(
        date_local.year, date_local.month, date_local.day, 5, 0, 0, tzinfo=timezone.utc
    )
    end = start + timedelta(minutes=30)
    return [
        {"start": start, "end": end, "label": "morning", "night_date": date_local},
    ]


def _mock_best_time(sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg):
    start, _ = window
    return 50.0, start + timedelta(minutes=5), 0.0, 0.0, 180.0


def test_first_epoch_multi_band_allowed():
    t = PriorityTracker()
    t.record_detection("SN1", 10.0, ["r", "g"], mjd=1.0)
    assert set(t.history["SN1"].last_mjd_by_filter.keys()) == {"r", "g"}


def test_cadence_blocks_same_filter_allows_other():
    t = PriorityTracker()
    t.record_detection("SN1", 10.0, ["r"], mjd=1.0)
    assert not t.cadence_gate("SN1", "r", 1.5, 3.0, 0.25)
    assert t.cadence_gate("SN1", "g", 1.5, 3.0, 0.25)


def test_simlib_uses_visit_mjd(tmp_path, monkeypatch):
    data_path = (
        Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    )
    csv_path = _subset_csv(data_path, tmp_path / "subset.csv")

    from twilight_planner_pkg import scheduler

    monkeypatch.setattr(scheduler, "twilight_windows_for_local_night", _mock_windows)
    monkeypatch.setattr(scheduler, "_best_time_with_moon", _mock_best_time)
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", lambda *a, **k: 0.0)
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
        exposure_by_filter={"r": 5.0},
        readout_s=1.0,
        filter_change_s=0.0,
        morning_cap_s=50.0,
        evening_cap_s=50.0,
        max_sn_per_night=1,
        per_sn_cap_s=20.0,
        min_moon_sep_by_filter={"r": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        allow_filter_changes_in_twilight=True,
        simlib_out=str(simlib_path),
    )

    pernight_df, _ = plan_twilight_range_with_caps(
        csv_path=str(csv_path),
        outdir=str(tmp_path),
        start_date="2025-07-30",
        end_date="2025-07-30",
        cfg=cfg,
        verbose=False,
    )

    mjd_seq = float(
        pernight_df["visit_start_utc"]
        .apply(lambda x: pd.Timestamp(x, tz="UTC"))
        .apply(lambda t: t.to_julian_date() - 2400000.5)
        .iloc[0]
    )
    lines = simlib_path.read_text().strip().splitlines()
    s_line = next(line for line in lines if line.startswith("S:"))
    mjd_simlib = float(s_line.split()[1])
    assert abs(mjd_seq - mjd_simlib) < 1e-4
