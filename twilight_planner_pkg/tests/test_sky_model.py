import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

from twilight_planner_pkg.sky_model import (
    SimpleSkyProvider,
    SkyModelConfig,
    sky_mag_arcsec2,
)


def test_sky_brightness_brightens_with_sun_alt():
    cfg = SkyModelConfig()
    dark = sky_mag_arcsec2("g", cfg, sun_alt_deg=-18.0)
    mid = sky_mag_arcsec2("g", cfg, sun_alt_deg=-9.0)
    assert mid < dark


def test_fallback_without_sun_alt():
    cfg = SkyModelConfig()
    expected = cfg.dark_sky_mag["g"] - cfg.twilight_delta_mag
    assert sky_mag_arcsec2("g", cfg, sun_alt_deg=None) == expected


def test_simple_provider_uses_sun_altitude():
    cfg = SkyModelConfig()
    site = EarthLocation(
        lat=-30.2446 * u.deg, lon=-70.7494 * u.deg, height=2647.0 * u.m
    )
    t = Time("2023-09-01T10:00:00", scale="utc")
    sun_alt = get_sun(t).transform_to(AltAz(obstime=t, location=site)).alt.deg
    provider = SimpleSkyProvider(cfg, site=site)
    mag = provider.sky_mag(t.mjd, None, None, "g", 1.0)
    expected = sky_mag_arcsec2("g", cfg, sun_alt_deg=float(sun_alt))
    assert np.isclose(mag, expected)
