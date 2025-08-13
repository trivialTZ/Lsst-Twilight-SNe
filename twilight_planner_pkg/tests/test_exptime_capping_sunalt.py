import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from twilight_planner_pkg.astro_utils import compute_capped_exptime
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.sky_model import SimpleSkyProvider, SkyModelConfig


def test_capping_is_stronger_when_sun_is_higher():
    """Exposure capping should be more aggressive at brighter twilight."""

    site = EarthLocation(lat=-30.2446 * u.deg, lon=-70.7494 * u.deg, height=2647 * u.m)
    sky_cfg = SkyModelConfig()
    provider = SimpleSkyProvider(sky_cfg, site=site)

    cfg = PlannerConfig(
        filters=["r"],
        exposure_by_filter={"r": 15.0},
        readout_s=0.0,
        filter_change_s=0.0,
    )
    cfg.sky_provider = provider
    cfg.current_mag_by_filter = {"r": 20.0}
    cfg.current_alt_deg = 50.0

    mjd_dark = Time("2025-01-05T08:00:00", scale="utc").mjd
    mjd_bright = Time("2025-01-05T08:30:00", scale="utc").mjd

    cfg.current_mjd = mjd_dark
    t_dark, _ = compute_capped_exptime("r", cfg)

    cfg.current_mjd = mjd_bright
    t_bright, _ = compute_capped_exptime("r", cfg)

    assert t_bright <= t_dark
