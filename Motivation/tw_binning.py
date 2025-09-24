"""Binning utilities for twilight cosmology."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tw_constants import DZ_DEFAULT, Z_MAX_DEFAULT, Z_TW_MAX_DEFAULT, Z_TW_MIN_DEFAULT


# σ_int(c) grid derived from Motivation/data/FITOPT000_MUOPT000.FITRES
# using robust MAD scatter in equal-count color windows following the
# BS21 analysis recipe (see Motivation/data/sigma_int_color.py).
_SIGMA_INT_COLOR_CENTERS = np.array(
    [
        -0.30,
        -0.29,
        -0.28,
        -0.27,
        -0.26,
        -0.25,
        -0.24,
        -0.23,
        -0.22,
        -0.21,
        -0.20,
        -0.19,
        -0.18,
        -0.17,
        -0.16,
        -0.15,
        -0.14,
        -0.13,
        -0.12,
        -0.11,
        -0.10,
        -0.09,
        -0.08,
        -0.07,
        -0.06,
        -0.05,
        -0.04,
        -0.03,
        -0.02,
        -0.01,
        0.0,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
        0.16,
        0.17,
        0.18,
        0.19,
        0.20,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.26,
        0.27,
        0.28,
        0.29,
        0.30,
    ],
    dtype=float,
)

_SIGMA_INT_COLOR_VALUES = np.array(
    [
        0.16642185,
        0.16642185,
        0.16642185,
        0.16642185,
        0.16642185,
        0.16642185,
        0.16642185,
        0.1647987,
        0.16167194,
        0.1590642,
        0.15662178,
        0.15357066,
        0.1500981,
        0.14577584,
        0.13981851,
        0.12880488,
        0.11359442,
        0.10363266,
        0.09524205,
        0.09262461,
        0.08891656,
        0.08796724,
        0.09010543,
        0.08494435,
        0.08729228,
        0.09406961,
        0.09241226,
        0.1030234,
        0.09525565,
        0.10110439,
        0.11705488,
        0.12325214,
        0.13039753,
        0.13903228,
        0.1360722,
        0.13626762,
        0.13844767,
        0.1369876,
        0.13686772,
        0.15317435,
        0.15935812,
        0.17103491,
        0.17902357,
        0.19332936,
        0.20446487,
        0.21450821,
        0.22303603,
        0.2264977,
        0.23251645,
        0.23990584,
        0.24604105,
        0.2545365,
        0.26145664,
        0.26573578,
        0.26959421,
        0.27126735,
        0.27219691,
        0.27313039,
        0.27375337,
        0.27375337,
        0.27375337,
        0.27375337,
    ],
    dtype=float,
)

_SIGMA_INT_COLOR_DEFAULT = float(np.nanmedian(_SIGMA_INT_COLOR_VALUES))


def _sigma_int_from_color(c: np.ndarray) -> np.ndarray:
    """Interpolate intrinsic scatter as a function of SALT2 color."""

    c = np.asarray(c, dtype=float)
    sigma = np.full(c.shape, _SIGMA_INT_COLOR_DEFAULT, dtype=float)
    mask = np.isfinite(c)
    if not np.any(mask):
        return sigma
    clipped = np.clip(
        c[mask], _SIGMA_INT_COLOR_CENTERS[0], _SIGMA_INT_COLOR_CENTERS[-1]
    )
    sigma[mask] = np.interp(
        clipped,
        _SIGMA_INT_COLOR_CENTERS,
        _SIGMA_INT_COLOR_VALUES,
    )
    return sigma


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
    sigma_int: float | str = "color_binned",
    sigma_vpec_kms: float = 300.0,
) -> pd.Series:
    """Compute per-SN σ_μ using full SALT2 covariances + lensing + vpec.

    Parameters
    ----------
    df : pandas.DataFrame
        FITRES-like table with SALT2 columns.
    alpha, beta : float, optional
        Stretch/color coefficients (fallback if not present in df).
    sigma_int : float | str, optional
        Intrinsic scatter added in quadrature. If set to ``"color_binned"``
        (the default) the intrinsic term is interpolated from the
        FITOPT000_MUOPT000.FITRES residuals as a function of SALT2 color.
        Supply a float to recover the previous constant-scatter behavior.
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
    color = _num_series(df, "c", np.nan).astype(float)

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

    if isinstance(sigma_int, str):
        if sigma_int.lower() != "color_binned":
            raise ValueError(
                "sigma_int string must be 'color_binned' or a numeric value"
            )
        sigma_int_series = pd.Series(
            _sigma_int_from_color(color.to_numpy(dtype=float, na_value=np.nan)),
            index=df.index,
        )
    else:
        sigma_int_series = pd.Series(float(sigma_int), index=df.index)

    mu2 = mu2 + sigma_int_series.pow(2) + (sig_lens ** 2) + (sig_vpec ** 2)
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
