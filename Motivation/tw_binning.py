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


def _num_series(df: pd.DataFrame, cols: list[str] | tuple[str, ...] | str, default: float) -> pd.Series:
    """Return numeric Series for the first present column in `cols`, else a constant Series.

    Ensures a Series is always returned (never a scalar), avoiding `.fillna` errors
    when a column is absent.
    """
    if isinstance(cols, (str, bytes)):
        cols = [str(cols)]
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            return s.fillna(default)
    # None of the requested columns exists: return constant default Series
    return pd.Series(default, index=df.index, dtype=float)


def sigma_mu_per_sn(
    df: pd.DataFrame,
    *,
    alpha: float = 0.14,
    beta: float = 3.1,
    sigma_int: float = 0.08,
    sigma_vpec_kms: float = 300.0,
) -> pd.Series:
    """Compute per-SN σ_μ using full SALT2 covariances + lensing + vpec.

    Parameters
    ----------
    df : pandas.DataFrame
        FITRES-like table with SALT2 columns.
    alpha, beta : float, optional
        Stretch/color coefficients (fallback if not present in df).
    sigma_int : float, optional
        Intrinsic scatter added in quadrature.
    sigma_vpec_kms : float, optional
        Peculiar velocity dispersion in km/s.

    Returns
    -------
    pandas.Series
        Per-SN distance-modulus uncertainty.
    """
    alpha = _num_series(df, ["SIM_alpha", "alpha"], alpha)
    beta = _num_series(df, ["SIM_beta", "beta"], beta)
    mBERR = _num_series(df, "mBERR", 0.12)
    x1ERR = _num_series(df, "x1ERR", 0.9)
    cERR = _num_series(df, "cERR", 0.04)
    cov_x1_c = _num_series(df, ["COV_x1_c", "COV_x1c"], 0.0)
    cov_mB_x1 = _num_series(df, "COV_mB_x1", 0.0)
    cov_mB_c = _num_series(df, "COV_mB_c", 0.0)
    z = _num_series(df, "z", np.nan).astype(float)

    # SALT2 error propagation with full covariances
    mu2 = (
        (mBERR ** 2)
        + (alpha * x1ERR) ** 2
        + (beta * cERR) ** 2
        + 2.0 * alpha * cov_mB_x1
        - 2.0 * beta * cov_mB_c
        - 2.0 * alpha * beta * cov_x1_c
    )

    # Lensing and peculiar-velocity terms
    sig_lens = 0.055 * z
    sig_vpec = (5.0 / np.log(10.0)) * (
        sigma_vpec_kms / (299792.458 * np.maximum(z, 1e-3))
    )

    mu2 = mu2 + (sigma_int ** 2) + (sig_lens ** 2) + (sig_vpec ** 2)
    return np.sqrt(np.maximum(mu2, 0.0))


def write_binned_catalogs(
    head_y1: pd.DataFrame,
    fit_qc_y1: pd.DataFrame,
    *,
    dz: float = DZ_DEFAULT,
    z_max: float = Z_MAX_DEFAULT,
    derived_dir: Path,
    base_label: str = "ep_lsst",
    sigma_agg: str = "ivar",
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
        if not m.any():
            continue
        if sigma_agg.lower() == "ivar":
            sig = pd.to_numeric(fit_qc_y1.loc[m, "sigma_mu_sn"], errors="coerce").astype(float)
            sig = sig.replace([np.inf, -np.inf], np.nan).dropna()
            if len(sig) == 0:
                continue
            invvar_sum = np.sum(1.0 / (sig ** 2))
            n_bin = float(len(sig))
            if invvar_sum > 0:
                sigma_bin[k] = float(np.sqrt(n_bin / invvar_sum))
        else:
            # Fallback: robust median within the bin (original behavior)
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
