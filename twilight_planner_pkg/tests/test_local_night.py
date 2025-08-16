import types
from datetime import date, datetime, timezone

import astropy.units as u
import pandas as pd
from astropy.coordinates import EarthLocation

from twilight_planner_pkg import astro_utils
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps

SITE = EarthLocation(lat=-30.2446 * u.deg, lon=-70.7494 * u.deg, height=2663 * u.m)


def test_twilight_windows_for_local_night_accepts_timestamp():
    """Passing a pandas.Timestamp to twilight_windows_for_local_night should work."""
    ts = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    wins = astro_utils.twilight_windows_for_local_night(ts, SITE)
    labels = [w.get("label") for w in wins]
    assert any(lbl == "evening" for lbl in labels), labels
    assert any(lbl == "morning" for lbl in labels), labels
    assert len(wins) == 2


def test_scheduler_passes_date_to_local_night(monkeypatch, tmp_path):
    """Ensure the scheduler passes a datetime.date into twilight_windows_for_local_night."""
    seen_types = []

    def fake_twilight_windows_for_local_night(
        date_local, loc, min_sun_alt_deg=-18.0, max_sun_alt_deg=0.0
    ):
        seen_types.append(type(date_local))
        d = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        return [
            {
                "start": d.replace(hour=23, minute=45),
                "end": d.replace(day=2, hour=1, minute=24),
                "label": "evening",
            },
            {
                "start": d.replace(hour=8, minute=9),
                "end": d.replace(hour=9, minute=49),
                "label": "morning",
            },
        ]

    import twilight_planner_pkg.scheduler as sched

    monkeypatch.setattr(
        sched, "twilight_windows_for_local_night", fake_twilight_windows_for_local_night
    )

    def fake_read_csv(_):
        return pd.DataFrame(
            {
                "Name": ["SNTEST"],
                "RA": [10.0],
                "DEC": [-20.0],
                "DiscoveryDate": [pd.Timestamp("2024-12-15T00:00:00Z")],
                "Type": ["Ia"],
                "mag_g": [19.5],
                "mag_r": [19.0],
                "mag_i": [18.8],
                "mag_z": [18.6],
                "mag_y": [18.4],
            }
        )

    monkeypatch.setattr(sched.pd, "read_csv", fake_read_csv)

    def fake_standardize_columns(df, cfg):
        out = df.copy()
        out["RA_deg"] = out.pop("RA")
        out["Dec_deg"] = out.pop("DEC")
        out["discovery_datetime"] = pd.to_datetime(out.pop("DiscoveryDate"), utc=True)
        out["SN_type_raw"] = out.pop("Type")
        return out

    monkeypatch.setattr(sched, "standardize_columns", fake_standardize_columns)

    Cfg = types.SimpleNamespace
    cfg = Cfg(
        pixel_scale_arcsec=0.2,
        zpt1s=None,
        k_m=None,
        fwhm_eff=None,
        read_noise_e=5.0,
        gain_e_per_adu=1.0,
        zpt_err_mag=0.01,
        simlib_npe_pixel_saturate=90000,
        simlib_out=None,
        simlib_survey="TEST",
        simlib_filters="grizy",
        simlib_pixsize=0.2,
        simlib_photflag_saturate=4096,
        simlib_psf_unit="NORM",
        lat_deg=-30.2446,
        lon_deg=-70.7494,
        height_m=2663,
        dark_sky_mag={"g": 22.0, "r": 21.5, "i": 21.0},
        twilight_delta_mag=-3.0,
        sky_provider=None,
        carousel_capacity=5,
        filters=["g", "r", "i"],
        start_filter="i",
        morning_cap_s=600,
        evening_cap_s=600,
        min_moon_sep_by_filter={"g": 40, "r": 30, "i": 25},
        min_alt_deg=20.0,
        twilight_step_min=2,
        priority_strategy="hybrid",
        hybrid_detections=1,
        hybrid_exposure_s=90,
        lc_detections=5,
        lc_exposure_s=300,
        exposure_by_filter={"g": 5, "r": 5, "i": 5},
        per_sn_cap_s=60.0,
        max_filters_per_visit=1,
        max_sn_per_night=10,
        readout_s=0.0,
        slew_small_deg=1.0,
        slew_small_time_s=2.0,
        slew_rate_deg_per_s=2.0,
        slew_settle_s=2.0,
        filter_change_s=120.0,
        sun_alt_exposure_ladder=[],
        typical_days_by_type={},
        default_typical_days=100,
        sun_alt_policy=[],
        twilight_sun_alt_min_deg=-18.0,
        twilight_sun_alt_max_deg=0.0,
    )

    monkeypatch.setattr(sched, "RubinSkyProvider", lambda *a, **k: None)
    monkeypatch.setattr(sched, "SimpleSkyProvider", lambda *a, **k: None)

    outdir = tmp_path / "out"
    plan_twilight_range_with_caps(
        csv_path="dummy.csv",
        outdir=str(outdir),
        start_date="2025-01-01",
        end_date="2025-01-01",
        cfg=cfg,
        verbose=False,
    )

    assert seen_types and seen_types[0] is date


def test_coarse_moon_gate_uses_min(monkeypatch, tmp_path):
    """The coarse best-time sampler should use min separation across filters."""
    seen_req = []

    import twilight_planner_pkg.scheduler as sched

    def fake_best_time_with_moon(sc, window, site, step_min, min_alt_deg, req_sep):
        seen_req.append(req_sep)
        now = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)
        return 45.0, now, -10.0, 0.5, 60.0

    monkeypatch.setattr(sched, "_best_time_with_moon", fake_best_time_with_moon)

    Cfg = types.SimpleNamespace
    cfg = Cfg(
        pixel_scale_arcsec=0.2,
        zpt1s=None,
        k_m=None,
        fwhm_eff=None,
        read_noise_e=5.0,
        gain_e_per_adu=1.0,
        zpt_err_mag=0.01,
        simlib_npe_pixel_saturate=90000,
        simlib_out=None,
        simlib_survey="TEST",
        simlib_filters="grizy",
        simlib_pixsize=0.2,
        simlib_photflag_saturate=4096,
        simlib_psf_unit="NORM",
        lat_deg=-30.2446,
        lon_deg=-70.7494,
        height_m=2663,
        dark_sky_mag={"g": 22.0, "i": 21.0},
        twilight_delta_mag=0.0,
        sky_provider=None,
        carousel_capacity=5,
        filters=["g", "i"],
        start_filter="i",
        morning_cap_s=0,
        evening_cap_s=0,
        min_moon_sep_by_filter={"g": 40.0, "i": 20.0},
        min_alt_deg=20.0,
        twilight_step_min=2,
        priority_strategy="hybrid",
        hybrid_detections=1,
        hybrid_exposure_s=90,
        lc_detections=5,
        lc_exposure_s=300,
        exposure_by_filter={"g": 5, "i": 5},
        per_sn_cap_s=60.0,
        max_filters_per_visit=1,
        max_sn_per_night=10,
        readout_s=0.0,
        slew_small_deg=1.0,
        slew_small_time_s=2.0,
        slew_rate_deg_per_s=2.0,
        slew_settle_s=2.0,
        filter_change_s=120.0,
        sun_alt_exposure_ladder=[],
        typical_days_by_type={},
        default_typical_days=100,
        sun_alt_policy=[],
        twilight_sun_alt_min_deg=-18.0,
        twilight_sun_alt_max_deg=0.0,
    )

    def fake_local_night(date_local, loc, min_sun_alt_deg=-18.0, max_sun_alt_deg=0.0):
        d = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        return [
            {
                "start": d.replace(hour=23, minute=45),
                "end": d.replace(day=2, hour=1, minute=24),
                "label": "evening",
            },
            {
                "start": d.replace(hour=8, minute=9),
                "end": d.replace(hour=9, minute=49),
                "label": "morning",
            },
        ]

    monkeypatch.setattr(sched, "twilight_windows_for_local_night", fake_local_night)

    def fake_read_csv(_):
        return pd.DataFrame(
            {
                "Name": ["SNTEST"],
                "RA": [10.0],
                "DEC": [-20.0],
                "DiscoveryDate": [pd.Timestamp("2024-12-20T00:00:00Z")],
                "Type": ["Ia"],
            }
        )

    monkeypatch.setattr(sched.pd, "read_csv", fake_read_csv)

    def fake_standardize_columns(df, cfg2):
        out = df.copy()
        out["RA_deg"] = out.pop("RA")
        out["Dec_deg"] = out.pop("DEC")
        out["discovery_datetime"] = pd.to_datetime(out.pop("DiscoveryDate"), utc=True)
        out["SN_type_raw"] = out.pop("Type")
        return out

    monkeypatch.setattr(sched, "standardize_columns", fake_standardize_columns)

    outdir = tmp_path / "out2"
    plan_twilight_range_with_caps(
        csv_path="dummy.csv",
        outdir=str(outdir),
        start_date="2025-01-01",
        end_date="2025-01-01",
        cfg=cfg,
        verbose=False,
    )

    assert seen_req and seen_req[0] == 20.0
