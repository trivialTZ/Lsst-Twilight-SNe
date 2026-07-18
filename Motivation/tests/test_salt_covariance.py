"""Regression tests for SALT covariance conversion and standardization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Astropy releases in the analysis environment still import NumPy's removed
# ``in1d`` alias while the production runner supplies the same compatibility
# shim.  Keep these focused tests independent of that environment mismatch.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from Motivation.tw_binning import (
    _sigma_int_from_color,
    sigma_mu_per_sn,
    write_binned_catalogs_v2,
)
from Motivation.tw_io import salt_covariance_mB_x1_c


def test_x0_covariance_is_converted_with_the_magnitude_derivative() -> None:
    df = pd.DataFrame(
        {
            "x0": [2.0],
            "x0ERR": [0.1],
            "x1ERR": [0.2],
            "cERR": [0.03],
            "COV_x1_x0": [0.004],
            "COV_c_x0": [-0.002],
            "COV_x1_c": [0.001],
        }
    )

    cov = salt_covariance_mB_x1_c(df)
    dm_b_dx0 = -2.5 / (np.log(10.0) * 2.0)

    assert cov.loc[0, "VAR_mB"] == pytest.approx(dm_b_dx0**2 * 0.1**2)
    assert cov.loc[0, "COV_mB_x1"] == pytest.approx(dm_b_dx0 * 0.004)
    assert cov.loc[0, "COV_mB_c"] == pytest.approx(dm_b_dx0 * -0.002)
    assert cov.loc[0, "COV_mB_x1"] < 0.0
    assert cov.loc[0, "COV_mB_c"] > 0.0


def test_direct_mb_covariance_wins_and_x0_fills_only_missing_rows() -> None:
    df = pd.DataFrame(
        {
            "x0": [1.0, 2.0],
            "mBERR": [0.11, 0.12],
            "x1ERR": [0.2, 0.2],
            "cERR": [0.03, 0.03],
            "COV_mB_x1": [0.007, np.nan],
            "COV_mB_c": [-0.008, np.nan],
            "COV_x0_x1": [0.5, 0.004],
            "COV_x0_c": [0.5, -0.002],
            "COV_x1_c": [0.001, 0.001],
        }
    )

    cov = salt_covariance_mB_x1_c(df)
    second_derivative = -2.5 / (np.log(10.0) * 2.0)

    assert cov.loc[0, "COV_mB_x1"] == pytest.approx(0.007)
    assert cov.loc[0, "COV_mB_c"] == pytest.approx(-0.008)
    assert cov.loc[1, "COV_mB_x1"] == pytest.approx(second_derivative * 0.004)
    assert cov.loc[1, "COV_mB_c"] == pytest.approx(second_derivative * -0.002)


def test_distance_variance_uses_gradient_one_alpha_minus_beta() -> None:
    df = pd.DataFrame(
        {
            "z": [0.0],
            "c": [0.0],
            "COV_mB_mB": [0.0100],
            "COV_x1_x1": [0.0400],
            "COV_c_c": [0.0009],
            "COV_mB_x1": [0.0010],
            "COV_mB_c": [-0.0005],
            "COV_x1_c": [0.0002],
        }
    )
    alpha = 0.14
    beta = 3.1
    gradient = np.array([1.0, alpha, -beta])
    matrix = np.array(
        [
            [0.0100, 0.0010, -0.0005],
            [0.0010, 0.0400, 0.0002],
            [-0.0005, 0.0002, 0.0009],
        ]
    )

    sigma = sigma_mu_per_sn(
        df,
        alpha=alpha,
        beta=beta,
        sigma_int=0.0,
        sigma_vpec_kms=0.0,
    )

    assert sigma.iloc[0] ** 2 == pytest.approx(gradient @ matrix @ gradient)


def test_fixed_coefficients_ignore_simulation_truth_by_default() -> None:
    df = pd.DataFrame(
        {
            "z": [0.0],
            "c": [0.0],
            "SIM_alpha": [1.5],
            "SIM_beta": [8.0],
            "mBERR": [0.10],
            "x1ERR": [0.20],
            "cERR": [0.03],
            "COV_mB_x1": [0.0],
            "COV_mB_c": [0.0],
            "COV_x1_c": [0.0],
        }
    )

    default_sigma = sigma_mu_per_sn(df, sigma_int=0.0, sigma_vpec_kms=0.0).iloc[0]
    override_sigma = sigma_mu_per_sn(
        df,
        alpha=0.20,
        beta=2.5,
        sigma_int=0.0,
        sigma_vpec_kms=0.0,
    ).iloc[0]

    expected_default = np.sqrt(0.10**2 + (0.14 * 0.20) ** 2 + (3.1 * 0.03) ** 2)
    expected_override = np.sqrt(0.10**2 + (0.20 * 0.20) ** 2 + (2.5 * 0.03) ** 2)
    assert default_sigma == pytest.approx(expected_default)
    assert override_sigma == pytest.approx(expected_override)


def test_bbc_color_scatter_is_added_once() -> None:
    color = 0.0
    df = pd.DataFrame(
        {
            "z": [0.0],
            "c": [color],
            "COV_mB_mB": [0.0],
            "COV_x1_x1": [0.0],
            "COV_c_c": [0.0],
            "COV_mB_x1": [0.0],
            "COV_mB_c": [0.0],
            "COV_x1_c": [0.0],
        }
    )

    sigma = sigma_mu_per_sn(df, sigma_vpec_kms=0.0).iloc[0]
    expected = _sigma_int_from_color(np.array([color]))[0]

    assert sigma == pytest.approx(expected)


def test_missing_covariance_fails_instead_of_silently_assuming_zero() -> None:
    df = pd.DataFrame(
        {
            "mBERR": [0.1],
            "x1ERR": [0.2],
            "cERR": [0.03],
        }
    )

    with pytest.raises(ValueError, match="COV_mB_x1"):
        salt_covariance_mB_x1_c(df)


def test_unbinned_exports_record_fixed_coefficients_and_recompute_sigma(
    tmp_path,
) -> None:
    salt_row = {
        "z": 0.05,
        "c": 0.0,
        "SIM_alpha": 1.5,
        "SIM_beta": 8.0,
        "mBERR": 0.10,
        "x1ERR": 0.20,
        "cERR": 0.03,
        "COV_mB_x1": 0.0,
        "COV_mB_c": 0.0,
        "COV_x1_c": 0.0,
    }
    fit_qc = pd.DataFrame([salt_row])
    promoted = pd.DataFrame([{**salt_row, "sigma_mu_sn": 999.0}])
    head = pd.DataFrame({"z": [0.05, 0.05]})

    write_binned_catalogs_v2(
        tmp_path,
        head,
        fit_qc,
        base_label="fixed_coefficients",
        dz=0.1,
        z_max=0.1,
        sigma_int=0.0,
        sigma_vpec_kms=0.0,
        fit_promoted_y1=promoted,
    )
    unbinned = pd.read_csv(tmp_path / "y1_sn_unbinned_fixed_coefficients.csv")

    assert np.allclose(unbinned["alpha_used"], 0.14)
    assert np.allclose(unbinned["beta_used"], 3.1)
    assert not np.isclose(unbinned["sigma_mu"], 999.0).any()
