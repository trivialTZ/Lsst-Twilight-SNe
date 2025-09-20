"""Fisher-matrix utilities for twilight cosmology forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tw_cosmo import C_KMS, CosmoParams, FisherSetup, jacobian_mu


def fisher_from_binned(df_bin: pd.DataFrame, setup: FisherSetup) -> np.ndarray:
    """Compute Fisher matrix for a binned catalog and marginalize over M.

    This is robust to empty bins (``N<=0``) and missing/invalid
    uncertainties by assigning zero weight to such bins instead of
    propagating NaNs into the Fisher matrix.
    """
    z = df_bin["z"].to_numpy(float)
    N = df_bin["N"].to_numpy(float)
    s = df_bin["sigma_mu"].to_numpy(float)
    # Safe weights: only bins with N>0 and finite, positive sigma contribute
    finite = np.isfinite(N) & np.isfinite(s) & (s > 0) & (N > 0)
    w = np.zeros_like(N, dtype=float)
    w[finite] = N[finite] / (s[finite] ** 2)
    J = jacobian_mu(z, setup.fid, setup)  # last column dμ/dM
    W = np.diag(w)
    F_full = J.T @ W @ J
    n = F_full.shape[0]
    idx_M = n - 1
    idx_theta = list(range(n - 1))
    F_tt = F_full[np.ix_(idx_theta, idx_theta)]
    F_tM = F_full[np.ix_(idx_theta, [idx_M])]
    F_Mt = F_full[np.ix_([idx_M], idx_theta)]
    F_MM = F_full[idx_M, idx_M]
    F_marg = F_tt - F_tM @ np.linalg.inv(np.array([[F_MM]])) @ F_Mt
    return F_marg


def combine_wfd_twilight(
    df_wfd: pd.DataFrame, df_twilight: pd.DataFrame
) -> pd.DataFrame:
    """Concatenate WFD and Twilight catalogs."""
    return pd.concat([df_wfd, df_twilight], ignore_index=True)


def _sigma_mu_from_salt2_row(
    mBERR: float,
    x1ERR: float,
    cERR: float,
    COV_mB_x1: float,
    COV_mB_c: float,
    COV_x1_c: float,
    z: float,
    alpha: float,
    beta: float,
    sigma_int: float,
    sigma_vpec_kms: float,
) -> float:
    var = (
        mBERR**2
        + (alpha**2) * (x1ERR**2)
        + (beta**2) * (cERR**2)
        + 2.0 * alpha * COV_mB_x1
        - 2.0 * beta * COV_mB_c
        - 2.0 * alpha * beta * COV_x1_c
        + sigma_int**2
    )
    z_eff = max(z, 1e-4)
    var += (5.0 / np.log(10.0)) ** 2 * (sigma_vpec_kms / (C_KMS * z_eff)) ** 2
    return float(np.sqrt(var))


def build_catalog_from_fitres(
    fitres_csv_path: str,
    *,
    alpha: float = 0.14,
    beta: float = 3.1,
    sigma_int: float = 0.10,
    sigma_vpec_kms: float = 300.0,
    z_col: str = "zHD",
) -> pd.DataFrame:
    """Build ``(z, sigma_mu)`` catalog from FITRES rows."""
    df = pd.read_csv(fitres_csv_path)
    needed = ["mBERR", "x1ERR", "cERR", "COV_x1_c", "COV_mB_c", "COV_mB_x1", z_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"FITRES missing columns: {missing}")
    sigmas = []
    zs = df[z_col].values
    for i in range(len(df)):
        sigmas.append(
            _sigma_mu_from_salt2_row(
                mBERR=df.loc[i, "mBERR"],
                x1ERR=df.loc[i, "x1ERR"],
                cERR=df.loc[i, "cERR"],
                COV_mB_x1=df.loc[i, "COV_mB_x1"],
                COV_mB_c=df.loc[i, "COV_mB_c"],
                COV_x1_c=df.loc[i, "COV_x1_c"],
                z=zs[i],
                alpha=alpha,
                beta=beta,
                sigma_int=sigma_int,
                sigma_vpec_kms=sigma_vpec_kms,
            )
        )
    return pd.DataFrame({"z": zs, "sigma_mu": sigmas})


def run_binned_forecast(
    df_wfd_bin: pd.DataFrame,
    df_tw_bin: pd.DataFrame,
    *,
    model: str = "lcdm",
) -> dict:
    """Return forecast errors and covariances for WFD and WFD+Twilight."""
    vary = ("Om",) if model == "lcdm" else ("Om", "w0", "wa")
    setup = FisherSetup(vary_params=vary, fid=CosmoParams())
    F_wfd = fisher_from_binned(df_wfd_bin, setup)
    F_tw = fisher_from_binned(df_tw_bin, setup)
    labels = list(vary)
    C_wfd = np.linalg.inv(F_wfd)
    C_tw = np.linalg.inv(F_tw)
    errs_wfd = {p: float(np.sqrt(C_wfd[i, i])) for i, p in enumerate(labels)}
    errs_tw = {p: float(np.sqrt(C_tw[i, i])) for i, p in enumerate(labels)}
    return {
        "WFD": {"labels": labels, "cov": C_wfd, "errs": errs_wfd},
        "WFD+Twilight": {"labels": labels, "cov": C_tw, "errs": errs_tw},
    }


def print_forecast_summary(res: dict, *, verbose: bool = True) -> None:
    """Pretty-print 1σ parameter uncertainties."""
    if not verbose:
        return
    labs = res["WFD"]["labels"]
    sig_wfd = {p: float(np.sqrt(res["WFD"]["cov"][i, i])) for i, p in enumerate(labs)}
    sig_tw = {p: float(np.sqrt(res["WFD+Twilight"]["cov"][i, i])) for i, p in enumerate(labs)}
    print("1σ parameter uncertainties")
    for p in labs:
        s1 = sig_wfd[p]
        s2 = sig_tw[p]
        ratio = s1 / s2 if (np.isfinite(s2) and s2 > 0) else float("nan")
        print(f"  {p:>3s}: WFD={s1:.4g}  WFD+Twilight={s2:.4g}  improvement x{ratio:.2f}")
