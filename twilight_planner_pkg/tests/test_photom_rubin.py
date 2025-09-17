from twilight_planner_pkg.photom_rubin import (
    PhotomConfig,
    cap_exposure_for_saturation,
    compute_epoch_photom,
    electrons_per_pixel_from_sb,
    observed_sb_from_rest,
)


def test_compute_epoch_photom_basics():
    cfg = PhotomConfig(gain_e_per_adu=1.6, npe_pixel_saturate=120000)
    eph = compute_epoch_photom("r", 1.0, 60.0, 21.0, cfg)
    assert abs(eph.ZPTAVG - 27.83) < 0.1
    assert eph.SKYSIG > 0
    assert eph.NEA_pix > 0


def test_cap_exposure_warns_and_caps_source():
    cfg = PhotomConfig()
    t_new, flags = cap_exposure_for_saturation("r", 5.0, 60.0, 14.0, 21.0, cfg)
    assert 0 < t_new < 5.0
    assert "sat_guard" in flags
    assert "warn_nonlinear" in flags


def test_cap_exposure_warns_and_caps_sky():
    cfg = PhotomConfig()
    t_new, flags = cap_exposure_for_saturation("r", 15.0, 60.0, 25.0, 14.0, cfg)
    assert 0 < t_new < 15.0
    assert "sat_guard" in flags
    assert "warn_nonlinear" in flags


def test_cap_exposure_no_warn_no_cap():
    cfg = PhotomConfig()
    t_new, flags = cap_exposure_for_saturation("r", 5.0, 60.0, 20.0, 21.0, cfg)
    assert t_new == 5.0
    assert not flags


def test_electrons_per_pixel_from_sb_scaling():
    # Basic monotonicity: brighter surface brightness -> more electrons
    cfg = PhotomConfig()
    eph = compute_epoch_photom("r", 5.0, 60.0, 21.0, cfg)
    e1 = electrons_per_pixel_from_sb(22.0, eph.ZPT_pe, cfg.pixel_scale_arcsec)
    e2 = electrons_per_pixel_from_sb(20.0, eph.ZPT_pe, cfg.pixel_scale_arcsec)
    assert e2 > e1 > 0


def test_observed_sb_from_rest_tolman_dimming():
    # Check Tolman dimming adds ~10*log10(1+z) mag to rest SB
    mu_rest = 21.0
    z = 0.5
    mu_obs = observed_sb_from_rest(mu_rest, z)
    assert mu_obs > mu_rest


def test_cap_exposure_more_aggressive_with_host_sb():
    # With a bright SN, adding a bright host SB should shorten exposure further
    cfg = PhotomConfig()
    # Start from a case that already triggers capping
    t1, flags1 = cap_exposure_for_saturation("r", 5.0, 60.0, 14.0, 21.0, cfg)
    t2, flags2 = cap_exposure_for_saturation(
        "r", 5.0, 60.0, 14.0, 21.0, cfg, mu_host_obs_arcsec2=20.0
    )
    assert 0 < t2 <= t1 < 5.0
    assert "sat_guard" in flags2


def test_tolman_dimming_relaxes_host_contribution():
    # Construct a scenario where host SB dominates when not dimmed
    # Use a small artificial saturation threshold to accentuate the effect
    cfg = PhotomConfig(npe_pixel_saturate=2000, npe_pixel_warn_nonlinear=1500)
    band = "r"
    t_exp = 30.0
    alt = 60.0
    src_mag = 25.0  # negligible point source
    sky_mu = 21.0   # moderate sky
    mu_rest = 20.0

    # Case A: treat rest SB as observed (i.e., no dimming)
    t_obs, _ = cap_exposure_for_saturation(
        band, t_exp, alt, src_mag, sky_mu, cfg, mu_host_obs_arcsec2=mu_rest
    )

    # Case B: apply Tolman dimming at z=0.5 -> fainter observed SB -> less capping
    t_dim, _ = cap_exposure_for_saturation(
        band,
        t_exp,
        alt,
        src_mag,
        sky_mu,
        cfg,
        mu_host_rest_arcsec2=mu_rest,
        z_host=0.5,
    )
    assert t_dim >= t_obs
