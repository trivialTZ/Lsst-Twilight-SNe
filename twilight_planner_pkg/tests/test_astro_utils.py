from twilight_planner_pkg.astro_utils import compute_capped_exptime, default_host_mu_obs
from twilight_planner_pkg.config import PlannerConfig


def test_default_host_mu_applied_when_missing():
    # Configure a scenario where host+sky can trigger capping if a small
    # saturation threshold is used, but source alone does not.
    cfg = PlannerConfig(
        filters=["r"],
        exposure_by_filter={"r": 30.0},
        simlib_npe_pixel_saturate=1200.0,
    )
    cfg.current_mag_by_filter = {"r": 25.0}
    cfg.current_alt_deg = 60.0
    cfg.current_mjd = None
    cfg.sky_provider = None  # fall back to 21 mag/arcsec^2 sky

    # With default host SB ON (22 mag/arcsec^2), expect stronger capping
    cfg.use_default_host_sb = True
    t_def, flags_def = compute_capped_exptime("r", cfg)

    # With default host SB OFF, the exposure should be equal or longer
    cfg.use_default_host_sb = False
    t_no, flags_no = compute_capped_exptime("r", cfg)

    assert t_def <= t_no
    # At least one of them should either warn or guard under this synthetic threshold
    assert (
        ("sat_guard" in flags_def)
        or ("warn_nonlinear" in flags_def)
        or ("sat_guard" in flags_no)
        or ("warn_nonlinear" in flags_no)
    )


def test_default_host_mu_obs_scales_with_redshift():
    cfg = PlannerConfig()
    cfg.use_default_host_sb = True
    cfg.current_host_z = 0.5
    mu_high = default_host_mu_obs("i", cfg)
    cfg.current_host_z = 0.1
    mu_low = default_host_mu_obs("i", cfg)
    assert mu_high is not None and mu_low is not None
    assert mu_high > mu_low
