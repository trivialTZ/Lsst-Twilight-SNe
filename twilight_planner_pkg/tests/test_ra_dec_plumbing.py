from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg import scheduler


class CapturingSkyProvider:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def sky_mag(self, mjd, ra_deg, dec_deg, band, airmass):
        # Record the exact arguments to verify RA/Dec plumbing
        self.calls.append((mjd, ra_deg, dec_deg, band, airmass))
        # Return a benign constant sky brightness
        return 21.0


def _subset_csv(src: Path, out: Path, n: int = 2) -> Path:
    df = pd.read_csv(src).head(n)
    df.to_csv(out, index=False)
    return out


def test_scheduler_passes_ra_dec_to_sky_provider(tmp_path, monkeypatch):
    # Deterministic, short windows to exercise code paths quickly
    def mock_twilight_windows_for_local_night(date_local, loc, min_sun_alt_deg=-18.0, max_sun_alt_deg=0.0):
        start = datetime(date_local.year, date_local.month, date_local.day, 8, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=20)
        return [
            {"start": start, "end": end, "label": "morning", "night_date": date_local},
        ]

    def mock_best_time_with_moon(sc, window, loc, step_min, min_alt_deg, min_moon_sep_deg):
        start, _ = window
        # Return (airmass, best_time, sun_alt, moon_alt, moon_sep)
        return 1.2, start + timedelta(minutes=2), -12.0, -10.0, 180.0

    def mock_sep(*args, **kwargs):
        return 0.0

    monkeypatch.setattr(scheduler, "twilight_windows_for_local_night", mock_twilight_windows_for_local_night)
    monkeypatch.setattr(scheduler, "_best_time_with_moon", mock_best_time_with_moon)
    monkeypatch.setattr(scheduler, "great_circle_sep_deg", mock_sep)

    # Capture provider instance used by the scheduler
    cap = CapturingSkyProvider()
    monkeypatch.setattr(scheduler, "RubinSkyProvider", lambda *a, **k: cap)

    # Small input catalogue
    data_path = Path(__file__).resolve().parents[2] / "data" / "ATLAS_2021_to25_cleaned.csv"
    csv_path = _subset_csv(data_path, tmp_path / "subset.csv", n=2)

    cfg = PlannerConfig(
        lat_deg=0.0,
        lon_deg=0.0,
        height_m=0.0,
        filters=["g"],
        exposure_by_filter={"g": 10.0},
        readout_s=1.0,
        filter_change_s=1.0,
        evening_cap_s=0.0,
        morning_cap_s=60.0,
        max_sn_per_night=1,
        per_sn_cap_s=30.0,
        min_moon_sep_by_filter={"g": 0.0},
        require_single_time_for_all_filters=False,
        min_alt_deg=0.0,
        twilight_step_min=1,
        allow_filter_changes_in_twilight=True,
    )

    # Run a minimal planning pass
    scheduler.plan_twilight_range_with_caps(
        csv_path=str(csv_path),
        outdir=str(tmp_path),
        start_date="2025-07-30",
        end_date="2025-07-30",
        cfg=cfg,
        verbose=False,
    )

    # Provider should have been called at least once
    assert len(cap.calls) > 0

    # At least one call must include RA/Dec and MJD values
    assert any((isinstance(mjd, (int, float)) and mjd is not None and isinstance(ra, (int, float)) and isinstance(dc, (int, float))) for (mjd, ra, dc, _b, _x) in cap.calls)

