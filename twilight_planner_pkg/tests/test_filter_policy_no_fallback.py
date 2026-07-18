import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import math

from twilight_planner_pkg.filter_policy import _m5_scale, allowed_filters_for_window
from twilight_planner_pkg.photom_rubin import PhotomConfig, point_source_limiting_mag


def test_no_heuristic_fallback_when_none_pass():
    # Extremely bright twilight: Sun above horizon; very high airmass
    mags = {"r": 25.0}  # faint target to force rejection
    allowed = allowed_filters_for_window(
        mags, 0.0, 10.0, 0.9, 5.0, 2.5, 1.5
    )
    # Must not inject heuristic bands; should be empty when m5 fails
    assert allowed == []


def test_scheduler_m5_is_the_simlib_physical_noise_model():
    cfg = PhotomConfig(read_noise_e=7.0, gain_e_per_adu=1.7)
    kwargs = {
        "band": "i",
        "m_sky_arcsec2": 19.8,
        "seeing_fwhm_arcsec": 0.91,
        "airmass": 1.37,
        "t_vis_s": 5.0,
        "k_m": 0.08,
    }
    scheduler_m5 = _m5_scale(**kwargs, phot_cfg=cfg)
    physical_m5 = point_source_limiting_mag(
        kwargs["band"],
        t_exp_s=kwargs["t_vis_s"],
        airmass=kwargs["airmass"],
        sky_mag_arcsec2=kwargs["m_sky_arcsec2"],
        cfg=cfg,
        fwhm_eff_arcsec=kwargs["seeing_fwhm_arcsec"],
        k_m=kwargs["k_m"],
        snr=5.0,
    )
    assert math.isclose(scheduler_m5, physical_m5, rel_tol=0.0, abs_tol=1e-12)
