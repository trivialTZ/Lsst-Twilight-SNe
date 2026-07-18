from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from Motivation.tw_pv_sensitivity import (
    DEFAULT_PV_PARAMETER_VALUES,
    PV_FISHER_PARAMETERS,
    _build_pv_covariance_and_derivatives,
    _gaussian_covariance_trace_fisher,
    _validate_pv_parameter_values,
    build_velocity_power_spectrum,
    compute_fs8_sigma_by_ranges,
    make_flip_ready,
)
from Motivation.tw_binning import sigma_mu_per_sn


def test_covariance_trace_fisher_matches_direct_formula() -> None:
    rng = np.random.default_rng(4102)
    factor = rng.normal(size=(6, 6))
    covariance = factor @ factor.T + 3.0 * np.eye(6)
    derivatives: dict[str, np.ndarray] = {}
    for name in PV_FISHER_PARAMETERS:
        raw = rng.normal(size=(6, 6))
        derivatives[name] = 0.5 * (raw + raw.T)

    result = _gaussian_covariance_trace_fisher(
        covariance,
        derivatives,
        parameter_scales={"fs8": 1.0, "sigv": 300.0, "sigma_M2": 0.02},
        max_condition_number=1.0e14,
        rank_rtol=1.0e-14,
    )

    inverse = np.linalg.inv(covariance)
    expected = np.array(
        [
            [
                0.5 * np.trace(inverse @ derivative_a @ inverse @ derivative_b)
                for derivative_b in derivatives.values()
            ]
            for derivative_a in derivatives.values()
        ]
    )
    np.testing.assert_allclose(result["fisher_matrix"], expected, rtol=1e-11)
    np.testing.assert_allclose(
        result["parameter_covariance"],
        np.linalg.inv(expected),
        rtol=1e-10,
    )
    marginalized = np.sqrt(np.diag(result["parameter_covariance"]))
    assert np.all(marginalized >= result["conditional_sigma"])


class _FakeDataFisher:
    def __init__(self, conversion_squared: np.ndarray, dmu_error: np.ndarray):
        self.conversion_squared = np.asarray(conversion_squared, dtype=float)
        self.dmu_error = np.asarray(dmu_error, dtype=float)

    def give_data_and_variance(self, parameter_values: dict[str, float]) -> np.ndarray:
        return self.conversion_squared * (
            self.dmu_error**2 + float(parameter_values["sigma_M"]) ** 2
        )


class _FakeCovariance:
    model_name = "carreres23"
    model_kind = "velocity"

    def __init__(self, signal: np.ndarray):
        self.covariance_dict = {"vv": np.asarray([signal], dtype=float)}


def test_zero_fiducial_added_variance_has_nonzero_derivative() -> None:
    conversion_squared = np.array([2.0e6, 3.0e6, 5.0e6])
    dmu_error = np.array([0.10, 0.13, 0.17])
    signal = np.array(
        [
            [8.0e4, 2.0e4, 1.0e4],
            [2.0e4, 9.0e4, 1.5e4],
            [1.0e4, 1.5e4, 7.0e4],
        ]
    )
    parameters = dict(DEFAULT_PV_PARAMETER_VALUES)

    covariance, derivatives, scales = _build_pv_covariance_and_derivatives(
        _FakeDataFisher(conversion_squared, dmu_error),
        _FakeCovariance(signal),
        parameters,
    )

    expected = (
        signal
        + parameters["sigv"] ** 2 * np.eye(3)
        + np.diag(conversion_squared * dmu_error**2)
    )
    np.testing.assert_allclose(covariance, expected)
    np.testing.assert_allclose(derivatives["sigma_M2"], np.diag(conversion_squared))
    assert np.any(derivatives["sigma_M2"] != 0)
    assert tuple(derivatives) == PV_FISHER_PARAMETERS
    assert scales["sigma_M2"] > 0


def test_flip_ready_preserves_x0_basis_covariances() -> None:
    fitres = pd.DataFrame(
        {
            "z": [0.035, 0.045],
            "RA": [15.0, 120.0],
            "DEC": [-20.0, 30.0],
            "mBERR": [0.08, 0.09],
            "x0": [1.0e-3, 8.0e-4],
            "x0ERR": [1.0e-5, 1.2e-5],
            "x1ERR": [0.5, 0.6],
            "cERR": [0.025, 0.03],
            "c": [0.0, 0.05],
            "COV_x1_c": [1.0e-4, -2.0e-4],
            "COV_x1_x0": [2.0e-7, -1.0e-7],
            "COV_c_x0": [1.0e-8, 2.0e-8],
        }
    )

    ready = make_flip_ready(fitres)

    for column in ("x0", "x0ERR", "COV_x1_x0", "COV_c_x0"):
        assert column in ready
    propagated = sigma_mu_per_sn(ready, sigma_vpec_kms=0)
    assert np.all(np.isfinite(propagated))
    assert np.all(propagated > 0)


def test_old_or_incomplete_parameter_requests_fail_loudly() -> None:
    with pytest.raises(ValueError, match="derivative is zero"):
        _validate_pv_parameter_values({"fs8": 1.0, "sigv": 300.0, "sigma_M": 0.0})
    with pytest.raises(ValueError, match="must be exactly"):
        _validate_pv_parameter_values({"fs8": 1.0})


def test_missing_added_variance_derivative_fails_loudly() -> None:
    class NoScatterDerivative(_FakeDataFisher):
        def give_data_and_variance(
            self, parameter_values: dict[str, float]
        ) -> np.ndarray:
            return self.conversion_squared * self.dmu_error**2

    signal = np.eye(3)
    with pytest.raises(RuntimeError, match="derivative is absent"):
        _build_pv_covariance_and_derivatives(
            NoScatterDerivative(np.ones(3), np.full(3, 0.1)),
            _FakeCovariance(signal),
            DEFAULT_PV_PARAMETER_VALUES,
        )


def test_rank_deficiency_needs_an_explicit_prior() -> None:
    covariance = np.eye(4)
    off_diagonal = np.array(
        [
            [0.0, 1.0, 0.2, 0.0],
            [1.0, 0.0, 0.0, 0.1],
            [0.2, 0.0, 0.0, 0.7],
            [0.0, 0.1, 0.7, 0.0],
        ]
    )
    derivatives = {
        "fs8": off_diagonal,
        "sigv": np.eye(4),
        "sigma_M2": np.eye(4),
    }
    scales = {"fs8": 1.0, "sigv": 1.0, "sigma_M2": 1.0}

    with pytest.raises(np.linalg.LinAlgError, match="rank deficient"):
        _gaussian_covariance_trace_fisher(
            covariance,
            derivatives,
            parameter_scales=scales,
        )

    result = _gaussian_covariance_trace_fisher(
        covariance,
        derivatives,
        parameter_scales=scales,
        parameter_priors={"sigma_M2": 0.5},
    )
    assert result["rank"] == 3
    assert result["parameter_priors"] == {"sigma_M2": 0.5}


def test_default_power_spectrum_uses_kmax_one(monkeypatch: pytest.MonkeyPatch) -> None:
    call: dict[str, float] = {}

    def fake_compute_power_spectra(
        engine: str,
        cosmology: dict[str, float],
        redshift: float,
        kmin: float,
        kmax: float,
        number_points: int,
        **kwargs: object,
    ) -> tuple[np.ndarray, ...]:
        call.update(kmin=kmin, kmax=kmax, number_points=float(number_points))
        k = np.geomspace(kmin, kmax, 8)
        zeros = np.zeros_like(k)
        return k, zeros, zeros, np.ones_like(k), zeros

    fake_flip = SimpleNamespace(
        power_spectra=SimpleNamespace(compute_power_spectra=fake_compute_power_spectra),
        utils=SimpleNamespace(Du=lambda k, sigma_u: np.ones_like(k)),
    )
    monkeypatch.setitem(sys.modules, "flip", fake_flip)

    spectrum = build_velocity_power_spectrum()

    assert call["kmax"] == pytest.approx(1.0)
    assert spectrum["_metadata"]["k_max_hmpc"] == pytest.approx(1.0)
    assert spectrum["_metadata"]["sigmau_hinv_mpc"] == pytest.approx(21.0)


def test_range_forecast_is_three_parameter_marginalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFisherVelFromHDres:
        def __init__(self, data: dict[str, list[float]]):
            self.data = {key: np.asarray(value) for key, value in data.items()}
            self.conversion_squared = np.linspace(1.5e6, 8.0e6, len(self.data["zobs"]))

        def give_data_and_variance(
            self, parameter_values: dict[str, float]
        ) -> np.ndarray:
            return self.conversion_squared * (
                self.data["dmu_error"] ** 2 + float(parameter_values["sigma_M"]) ** 2
            )

        def compute_covariance(
            self,
            model: str,
            power_spectrum_dict: dict[str, object],
            **kwargs: object,
        ) -> _FakeCovariance:
            assert model == "carreres23"
            index = np.arange(len(self.data["zobs"]))
            signal = 1.2e5 * np.exp(-np.abs(index[:, None] - index[None, :]) / 3.0)
            return _FakeCovariance(signal)

    fake_flip = SimpleNamespace(
        data_vector=SimpleNamespace(FisherVelFromHDres=FakeFisherVelFromHDres)
    )
    monkeypatch.setitem(sys.modules, "flip", fake_flip)

    size = 12
    redshift = np.linspace(0.021, 0.059, size)
    flow = pd.DataFrame(
        {
            "z": redshift,
            "zobs": redshift,
            "ra": np.linspace(0.1, 1.2, size),
            "dec": np.linspace(-0.4, 0.3, size),
            "rcom_zobs": np.linspace(60.0, 170.0, size),
            "hubble_norm": np.linspace(101.0, 104.0, size),
            "mBERR": np.linspace(0.06, 0.11, size),
            "x1ERR": np.linspace(0.35, 0.75, size),
            "cERR": np.linspace(0.02, 0.04, size),
            "c": np.linspace(-0.08, 0.12, size),
            "COV_mB_x1": np.zeros(size),
            "COV_mB_c": np.zeros(size),
            "COV_x1_c": np.zeros(size),
        }
    )
    k = np.geomspace(1.0e-5, 1.0, 12)
    spectrum = {
        "vv": [[k, np.ones_like(k)]],
        "_metadata": {
            "k_min_hmpc": 1.0e-5,
            "k_max_hmpc": 1.0,
            "n_k": len(k),
            "sigmau_hinv_mpc": 21.0,
        },
    }

    result = compute_fs8_sigma_by_ranges(flow, [(0.02, 0.06)], spectrum)
    row = result.iloc[0]

    assert row["fisher_dimension"] == 3
    assert row["fisher_rank"] == 3
    assert bool(row["is_nuisance_marginalized"])
    assert json.loads(row["fisher_parameters"]) == list(PV_FISHER_PARAMETERS)
    assert json.loads(row["marginalized_parameters"]) == ["sigv", "sigma_M2"]
    assert json.loads(row["parameter_priors_json"]) == {}
    assert not bool(row["has_parameter_priors"])
    assert row["k_max_hmpc"] == pytest.approx(1.0)
    assert row["sigma_M2_fiducial_mag2"] == pytest.approx(0.0)
    assert row["sigma_fs8"] >= row["sigma_fs8_conditional"]
    assert row["marginalization_degradation"] >= 1.0


def test_legacy_inverse_setting_is_not_silently_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "flip", SimpleNamespace())
    k = np.array([1.0e-5, 1.0])
    with pytest.raises(ValueError, match="former FLIP 'inverse'"):
        compute_fs8_sigma_by_ranges(
            pd.DataFrame(),
            [(0.02, 0.06)],
            {"vv": [[k, np.ones_like(k)]]},
            fisher_properties={"inversion_method": "inverse"},
        )
