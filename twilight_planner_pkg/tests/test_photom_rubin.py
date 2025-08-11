from twilight_planner_pkg.photom_rubin import PhotomConfig, compute_epoch_photom, cap_exposure_for_saturation


def test_compute_epoch_photom_basics():
    cfg = PhotomConfig(gain_e_per_adu=1.6, npe_pixel_saturate=120000)
    eph = compute_epoch_photom('r', 1.0, 60.0, 21.0, cfg)
    assert abs(eph.ZPTAVG - 27.83) < 0.1
    assert eph.SKYSIG > 0
    assert eph.NEA_pix > 0


def test_cap_exposure_for_saturation():
    cfg = PhotomConfig(gain_e_per_adu=1.6, npe_pixel_saturate=120000)
    t_new = cap_exposure_for_saturation('r', 5.0, 60.0, 14.0, 21.0, cfg)
    assert 0 < t_new < 5.0
