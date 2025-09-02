"""Binning utilities for twilight cosmology."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tw_constants import DZ_DEFAULT, Z_MAX_DEFAULT, Z_TW_MAX_DEFAULT, Z_TW_MIN_DEFAULT


def nz_hist(z: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    """Histogram counts per redshift bin."""
    h, _ = np.histogram(z, bins=z_edges)
    return h


def sigma_mu_per_sn(
    df: pd.DataFrame,
    *,
    alpha: float = 0.14,
    beta: float = 3.1,
    sigma_int: float = 0.08,
    sigma_vpec_kms: float = 300.0,
) -> pd.Series:
    """Compute per-SN σ_μ robustly using SALT2 covariances.

    Parameters
    ----------
    df : pandas.DataFrame
        FITRES-like table with SALT2 columns.
    alpha, beta : float, optional
        Stretch/color coefficients.
    sigma_int : float, optional
        Intrinsic scatter added in quadrature.
    sigma_vpec_kms : float, optional
        Peculiar velocity dispersion in km/s.

    Returns
    -------
    pandas.Series
        Per-SN distance-modulus uncertainty.
    """
    alpha = pd.to_numeric(
        df.get("SIM_alpha", df.get("alpha", alpha)), errors="coerce"
    ).fillna(alpha)
    beta = pd.to_numeric(
        df.get("SIM_beta", df.get("beta", beta)), errors="coerce"
    ).fillna(beta)
    mBERR = pd.to_numeric(df.get("mBERR", np.nan), errors="coerce").fillna(0.12)
    x1ERR = pd.to_numeric(df.get("x1ERR", np.nan), errors="coerce").fillna(0.9)
    cERR = pd.to_numeric(df.get("cERR", np.nan), errors="coerce").fillna(0.04)
    cov = pd.to_numeric(
        df.get("COV_x1_c", df.get("COV_x1c", 0.0)), errors="coerce"
    ).fillna(0.0)
    z = pd.to_numeric(df.get("z", np.nan), errors="coerce").astype(float)
    mu2 = (
        (mBERR**2)
        + (alpha * x1ERR) ** 2
        + (beta * cERR) ** 2
        - 2.0 * alpha * beta * cov
    )
    sig_lens = 0.055 * z
    sig_vpec = (5.0 / np.log(10.0)) * (
        sigma_vpec_kms / (299792.458 * np.maximum(z, 1e-3))
    )
    mu2 = mu2 + (sigma_int**2) + (sig_lens**2) + (sig_vpec**2)
    return np.sqrt(np.maximum(mu2, 0.0))


def write_binned_catalogs(
    head_y1: pd.DataFrame,
    fit_qc_y1: pd.DataFrame,
    *,
    dz: float = DZ_DEFAULT,
    z_max: float = Z_MAX_DEFAULT,
    derived_dir: Path,
    base_label: str = "ep_lsst",
) -> tuple[Path, Path]:
    """Build and write binned catalogs for WFD and WFD+Twilight.

    Returns
    -------
    tuple of Paths
        Paths to the base and twilight binned CSVs.
    """
    z_edges = np.arange(0.0, z_max + dz + 1e-12, dz)
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    N_det = nz_hist(head_y1["z"].to_numpy(float), z_edges)
    N_cos = nz_hist(fit_qc_y1["z"].to_numpy(float), z_edges)
    band = (z_edges[:-1] >= Z_TW_MIN_DEFAULT) & (z_edges[1:] <= Z_TW_MAX_DEFAULT)
    N_cos_tw = N_cos.copy()
    N_cos_tw[band] = np.maximum(N_cos_tw[band], N_det[band])
    fit_qc_y1 = fit_qc_y1.copy()
    fit_qc_y1["sigma_mu_sn"] = sigma_mu_per_sn(fit_qc_y1)
    sigma_bin = np.full_like(z_mid, np.nan, dtype=float)
    for k in range(len(z_mid)):
        m = (fit_qc_y1["z"] >= z_edges[k]) & (fit_qc_y1["z"] < z_edges[k + 1])
        if m.any():
            sigma_bin[k] = float(fit_qc_y1.loc[m, "sigma_mu_sn"].median())
    if np.isnan(sigma_bin).any():
        s = pd.Series(sigma_bin).fillna(method="ffill").fillna(method="bfill")
        global_med = (
            float(np.nanmedian(fit_qc_y1["sigma_mu_sn"]))
            if "sigma_mu_sn" in fit_qc_y1
            else 0.12
        )
        sigma_bin = s.fillna(global_med).to_numpy()
    df_base = pd.DataFrame({"z": z_mid, "N": N_cos, "sigma_mu": sigma_bin})
    df_tw = pd.DataFrame({"z": z_mid, "N": N_cos_tw, "sigma_mu": sigma_bin})
    base_path = derived_dir / f"y1_cat_bin_base_{base_label}.csv"
    tw_path = derived_dir / f"y1_cat_bin_tw_{base_label}.csv"
    df_base.to_csv(base_path, index=False)
    df_tw.to_csv(tw_path, index=False)
    # compat filenames
    df_base.to_csv(derived_dir / "y1_cat_bin_base_fix.csv", index=False)
    df_tw.to_csv(derived_dir / "y1_cat_bin_tw_fix.csv", index=False)
    return base_path, tw_path


def load_binned_catalogs(
    derived_dir: Path,
    *,
    z_max: float | None = None,
    base_glob: str = "y1_cat_bin_base_*.csv",
    tw_glob: str = "y1_cat_bin_tw_*.csv",
    tw_guess_glob: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load binned catalogs and clip to ``z_max`` if provided."""

    def _latest(glob: str) -> Path | None:
        paths = sorted(derived_dir.glob(glob), key=lambda p: p.stat().st_mtime)
        return paths[-1] if paths else None

    base = _latest(base_glob)
    tw = _latest(tw_glob)
    if tw_guess_glob is None:
        tw_guess_glob = "y1_cat_bin_tw_guess_*.csv"
    tw_guess = _latest(tw_guess_glob)
    out: dict[str, pd.DataFrame] = {}
    if base is not None:
        df = pd.read_csv(base)
        if z_max is not None:
            df = df.query("0 < z <= @z_max").copy()
        out["WFD"] = df
    if tw is not None:
        df = pd.read_csv(tw)
        if z_max is not None:
            df = df.query("0 < z <= @z_max").copy()
        out["WFD+Twilight"] = df
    if tw_guess is not None:
        df = pd.read_csv(tw_guess)
        if z_max is not None:
            df = df.query("0 < z <= @z_max").copy()
        out["WFD+Twilight (Guess)"] = df
    return out
