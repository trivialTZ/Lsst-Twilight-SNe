import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

from twilight_planner_pkg.sky_model import (
    RUBIN_TWILIGHT_PARAMS,
    SimpleSkyProvider,
    SkyModelConfig,
    sky_mag_arcsec2,
)


def test_sky_brightness_brightens_with_sun_alt():
    cfg = SkyModelConfig()
    dark = sky_mag_arcsec2("g", cfg, sun_alt_deg=-18.0)
    mid = sky_mag_arcsec2("g", cfg, sun_alt_deg=-9.0)
    assert mid < dark


def test_minus12_flux_normalization_matches_rubin_twilight_equation():
    cfg = SkyModelConfig()
    for band, (ratio_at_minus12, *_rest) in RUBIN_TWILIGHT_PARAMS.items():
        base = cfg.dark_sky_mag[band]
        expected = base - 2.5 * np.log10(1.0 + ratio_at_minus12)
        actual = sky_mag_arcsec2(
            band,
            cfg,
            sun_alt_deg=-12.0,
            airmass=1.0,
            sun_az_rel_deg=180.0,
        )
        assert np.isclose(actual, expected, atol=1.0e-6)


def test_bright_twilight_uses_empirical_flux_slope():
    cfg = SkyModelConfig()
    at_minus_six = sky_mag_arcsec2(
        "g",
        cfg,
        sun_alt_deg=-6.0,
        airmass=1.0,
        sun_az_rel_deg=180.0,
    )
    at_minus_five = sky_mag_arcsec2(
        "g",
        cfg,
        sun_alt_deg=-5.0,
        airmass=1.0,
        sun_az_rel_deg=180.0,
    )

    # Rubin's g-band slope is 23.005/radian, or about 1.00 mag per
    # degree once twilight dominates the dark-sky component.
    assert np.isclose(at_minus_six - at_minus_five, 1.00, atol=0.02)


def test_twilight_sky_is_brighter_toward_sun_at_same_airmass():
    cfg = SkyModelConfig()
    toward = sky_mag_arcsec2(
        "g",
        cfg,
        sun_alt_deg=-9.0,
        airmass=2.0,
        sun_az_rel_deg=0.0,
    )
    away = sky_mag_arcsec2(
        "g",
        cfg,
        sun_alt_deg=-9.0,
        airmass=2.0,
        sun_az_rel_deg=180.0,
    )
    assert toward < away


def test_sun_direction_effect_fades_in_deep_twilight():
    cfg = SkyModelConfig()
    toward = sky_mag_arcsec2(
        "g",
        cfg,
        sun_alt_deg=-30.0,
        airmass=2.0,
        sun_az_rel_deg=0.0,
    )
    away = sky_mag_arcsec2(
        "g",
        cfg,
        sun_alt_deg=-30.0,
        airmass=2.0,
        sun_az_rel_deg=180.0,
    )
    assert np.isclose(toward, away, atol=1.0e-6)


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
