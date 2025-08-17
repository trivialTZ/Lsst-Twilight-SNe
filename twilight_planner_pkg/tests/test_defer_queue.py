import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps


def _make_subset_csv(src: Path, out: Path, n: int = 3) -> tuple[Path, list[str]]:
    df = pd.read_csv(src).head(n)
    df.to_csv(out, index=False)
    return out, df["name"].tolist()


def test_defer_queue(tmp_path, monkeypatch):
    data_path = (
        Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    )
    subset_csv, names = _make_subset_csv(data_path, tmp_path / "subset.csv", n=3)
    name_a, name_b, name_c = names

    from twilight_planner_pkg import scheduler

    def mock_windows(date_local, loc, min_sun_alt_deg=-18.0, max_sun_alt_deg=0.0):
        start = datetime(
            date_local.year,
            date_local.month,
            date_local.day,
            5,
            0,
            0,
            tzinfo=timezone.utc,
        )
        end = start + timedelta(minutes=10)
        return [
            {"start": start, "end": end, "label": "morning", "night_date": date_local}
        ]

    def mock_best_time_with_moon(
        sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg
    ):
        start, _ = window
        return 50.0, start + timedelta(minutes=5), 30.0, 0.0, 180.0

    monkeypatch.setattr(scheduler, "twilight_windows_for_local_night", mock_windows)
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", lambda a, b, c, d: 0.0)
    monkeypatch.setattr(scheduler, "allowed_filters_for_window", lambda *a, **k: ["g"])
    monkeypatch.setattr(
        scheduler, "allowed_filters_for_sun_alt", lambda alt, cfg: ["g"]
    )

    start_dt = datetime(2025, 7, 30, 5, 0, 0, tzinfo=timezone.utc)
    start_mjd = Time(start_dt).mjd
    thresholds = {name_b: start_mjd + 1.0 / 86400.0, name_c: start_mjd + 5.0 / 86400.0}

    def mock_gate(self, name, filt, now_mjd, target, jitter):
        return now_mjd >= thresholds.get(name, start_mjd)

    monkeypatch.setattr(
        scheduler.PriorityTracker, "cadence_gate", mock_gate, raising=False
    )
    monkeypatch.setattr(
        scheduler.PriorityTracker, "cadence_bonus", lambda *a, **k: 0.0, raising=False
    )

    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["g"],
        exposure_by_filter={"g": 1.0},
        readout_s=1.0,
        filter_change_s=0.0,
        evening_cap_s=0.0,
        morning_cap_s=600.0,
        max_sn_per_night=3,
        per_sn_cap_s=100.0,
        min_moon_sep_by_filter={"g": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        allow_filter_changes_in_twilight=True,
    )

    pernight_df, _ = plan_twilight_range_with_caps(
        csv_path=str(subset_csv),
        outdir=str(tmp_path),
        start_date="2025-07-30",
        end_date="2025-07-30",
        cfg=cfg,
        verbose=False,
    )

    assert (pernight_df["SN"] == name_b).any()
