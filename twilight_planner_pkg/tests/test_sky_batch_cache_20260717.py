from __future__ import annotations

import numpy as np

from twilight_planner_pkg.astro_utils import compute_capped_exptime
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.sky_model import RubinSkyProvider, SkyModelConfig
from twilight_planner_pkg.twilight_sched.caching import (
    _cached_sky_mag,
    bump_window_token,
    precache_sky_mags,
)
from twilight_planner_pkg.twilight_sched.scoring_pairs import (
    _precache_saturation_sky_for_tasks,
)


class _FakeRubinModel:
    def set_ra_dec_mjd(self, *, ra, dec, mjd, degrees, filter_names):
        assert degrees is True
        self.ra = np.atleast_1d(np.asarray(ra, dtype=float))
        self.dec = np.atleast_1d(np.asarray(dec, dtype=float))
        self.mjd = float(mjd)
        self.filter_names = list(filter_names)

    def return_mags(self):
        return {
            band: self.ra * 0.01 + self.dec * 0.001 + index
            for index, band in enumerate(self.filter_names)
        }


class _CountingBatchProvider:
    def __init__(self):
        self.batch_calls = 0
        self.scalar_calls = 0

    def sky_mags(self, mjd, ra_deg, dec_deg, bands, airmass=None):
        self.batch_calls += 1
        ra = np.asarray(ra_deg, dtype=float)
        return {
            band: ra * 0.01 + float(mjd) * 1e-5 + index
            for index, band in enumerate(bands)
        }

    def sky_mag(self, mjd, ra_deg, dec_deg, band, airmass):
        self.scalar_calls += 1
        return 20.0


def test_rubin_vectorized_geometry_returns_target_by_band_arrays() -> None:
    provider = object.__new__(RubinSkyProvider)
    provider.model = _FakeRubinModel()
    provider.strict = True

    result = provider.sky_mags(
        60000.25,
        [10.0, 20.0],
        [-30.0, -25.0],
        ["g", "r"],
    )

    np.testing.assert_allclose(result["g"], [0.07, 0.175], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(result["r"], [1.07, 1.175], rtol=0.0, atol=1e-15)


def test_policy_batch_populates_exact_scalar_cache_keys() -> None:
    provider = _CountingBatchProvider()
    bump_window_token()
    stored = precache_sky_mags(
        sky_provider=provider,
        mjd=60000.25,
        samples=[
            (10.0, -30.0, 1.2, ("g", "r")),
            (20.0, -25.0, 1.4, ("g", "r")),
        ],
        minutes=1,
    )

    assert stored == 4
    value = _cached_sky_mag(
        sky_provider=provider,
        sky_cfg=SkyModelConfig(),
        ra_deg=20.0,
        dec_deg=-25.0,
        band="r",
        airmass=1.4,
        mjd=60000.25,
        minutes=1,
    )
    assert value == provider.sky_mags(60000.25, [20.0], [-25.0], ["g", "r"])[
        "r"
    ][0]
    assert provider.batch_calls == 2
    assert provider.scalar_calls == 0


def test_pair_batch_is_reused_by_saturation_capping() -> None:
    provider = _CountingBatchProvider()
    cfg = PlannerConfig(
        filters=["r"],
        exposure_by_filter={"r": 5.0},
        sky_provider=provider,
        sky_batch_cache_enable=True,
    )
    target = {
        "Name": "SN-test",
        "RA_deg": 20.0,
        "Dec_deg": -25.0,
        "max_alt_deg": 60.0,
        "best_time_mjd": 60000.25,
    }
    stored = _precache_saturation_sky_for_tasks(
        [(target, "r", 3.0)], cfg, provider, 60000.0
    )
    assert stored == 1

    cfg.current_mag_by_filter = {"r": 18.0}
    cfg.current_alt_deg = 60.0
    cfg.current_ra_deg = 20.0
    cfg.current_dec_deg = -25.0
    cfg.current_mjd = 60000.25
    compute_capped_exptime("r", cfg)

    assert provider.batch_calls == 1
    assert provider.scalar_calls == 0
