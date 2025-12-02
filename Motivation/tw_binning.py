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


def sample_sigma_mu_promoted(
    df_det_bin: pd.DataFrame | None,
    df_fit_promoted_bin: pd.DataFrame | None,
    n_needed: int,
    *,
    alpha: float = 0.14,
    beta: float = 3.1,
    sigma_int: float | str = "color_binned",
    sigma_vpec_kms: float = 300.0,
    f_tw_scale: float = 1.0,
    z_fallback: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (sigma_mu, z) for `n_needed` promoted SNe in this bin.

    Preference order:
      (1) use SALT2 fit rows if provided (`df_fit_promoted_bin`),
      (2) else, if detection rows include SALT2 error columns, compute via
          :func:`sigma_mu_per_sn`,
      (3) else, fallback to scaled WFD-like sigma using ``f_tw_scale`` and
          ``z_fallback``.
    """

    if rng is None:
        rng = np.random.default_rng(12345)

    # Case 1: explicit promoted-fit rows
    if df_fit_promoted_bin is not None and len(df_fit_promoted_bin) > 0:
        s = sigma_mu_per_sn(
            df_fit_promoted_bin,
            alpha=alpha,
            beta=beta,
            sigma_int=sigma_int,
            sigma_vpec_kms=sigma_vpec_kms,
        ).to_numpy(dtype=float)
        z = pd.to_numeric(
            df_fit_promoted_bin.get("z", np.nan), errors="coerce"
        ).to_numpy(dtype=float)
        ok = np.isfinite(s) & (s > 0.0) & np.isfinite(z)
        s, z = s[ok], z[ok]
        if len(s) > 0:
            idx = rng.integers(0, len(s), size=n_needed)
            return s[idx], z[idx]

    # Case 2: detection dataframe has SALT2-like columns
    if df_det_bin is not None and len(df_det_bin) > 0:
        cols = {
            "mBERR",
            "x1ERR",
            "cERR",
            "COV_x1_c",
            "COV_mB_x1",
            "COV_mB_c",
            "z",
        }
        if cols.issubset(set(df_det_bin.columns)):
            s = sigma_mu_per_sn(
                df_det_bin,
                alpha=alpha,
                beta=beta,
                sigma_int=sigma_int,
                sigma_vpec_kms=sigma_vpec_kms,
            ).to_numpy(dtype=float)
            z = pd.to_numeric(df_det_bin.get("z", np.nan), errors="coerce").to_numpy(
                dtype=float
            )
            ok = np.isfinite(s) & (s > 0.0) & np.isfinite(z)
            s, z = s[ok], z[ok]
            if len(s) > 0:
                idx = rng.integers(0, len(s), size=n_needed)
                return s[idx], z[idx]

    # Case 3: fallback (filled by caller if NaNs remain). No noisy prints.
    if z_fallback is None:
        z_fallback = np.nan
    return (np.full(n_needed, np.nan) * f_tw_scale), np.full(n_needed, z_fallback)


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

    Notes
    -----
    Also writes an unbinned per-SN CSV named
    ``y1_sn_unbinned_{base_label}.csv`` in ``derived_dir`` with columns
    including ``z``, ``c``, ``alpha_used``, ``beta_used``, ``sigma_int``,
    ``sigma_lens``, ``sigma_vpec``, and per-SN ``sigma_mu``.
    """
    z_edges = np.arange(0.0, z_max + dz + 1e-12, dz)
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    N_det = nz_hist(head_y1["z"].to_numpy(float), z_edges)
    N_cos = nz_hist(fit_qc_y1["z"].to_numpy(float), z_edges)
    band = (z_edges[:-1] >= Z_TW_MIN_DEFAULT) & (z_edges[1:] <= Z_TW_MAX_DEFAULT)
    N_cos_tw = N_cos.copy()
    N_cos_tw[band] = np.maximum(N_cos_tw[band], N_det[band])
    fit_qc_y1 = fit_qc_y1.copy()
    # Per-SN uncertainty
    fit_qc_y1["sigma_mu_sn"] = sigma_mu_per_sn(fit_qc_y1)

    # Build unbinned, per-SN dataframe for diagnostics/exports
    # Components consistent with sigma_mu_per_sn defaults
    alpha_s = _num_series(fit_qc_y1, ["SIM_alpha", "alpha"], 0.14)
    beta_s = _num_series(fit_qc_y1, ["SIM_beta", "beta"], 3.1)
    z_s = _num_series(fit_qc_y1, "z", np.nan).astype(float)
    c_s = _num_series(fit_qc_y1, "c", np.nan).astype(float)
    sigma_int_s = pd.Series(
        _sigma_int_from_color(c_s.to_numpy(dtype=float, na_value=np.nan)),
        index=fit_qc_y1.index,
    )
    sig_lens_s = 0.055 * z_s
    sig_vpec_s = (5.0 / np.log(10.0)) * (
        300.0 / (299792.458 * np.maximum(z_s, 1e-3))
    )

    cols_id: dict[str, pd.Series] = {}
    for cand_id in ("CID", "SNID", "ID"):
        if cand_id in fit_qc_y1.columns:
            cols_id[cand_id] = fit_qc_y1[cand_id]
            break

    df_unbinned_cols: dict[str, pd.Series | np.ndarray | float | str] = {
        **cols_id,
        "source": pd.Series("WFD", index=fit_qc_y1.index),
        "z": z_s,
        "c": c_s,
        "alpha_used": alpha_s,
        "beta_used": beta_s,
        "sigma_int": sigma_int_s,
        "sigma_lens": sig_lens_s,
        "sigma_vpec": sig_vpec_s,
        "sigma_mu": pd.to_numeric(
            fit_qc_y1["sigma_mu_sn"], errors="coerce"
        ).astype(float),
    }
    df_unbinned = pd.DataFrame(df_unbinned_cols, index=fit_qc_y1.index)
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

    # Write unbinned per-SN export for transparency/diagnostics
    unbinned_path = derived_dir / f"y1_sn_unbinned_{base_label}.csv"
    df_unbinned.to_csv(unbinned_path, index=False)
    # compat filenames
    df_base.to_csv(derived_dir / "y1_cat_bin_base_fix.csv", index=False)
    df_tw.to_csv(derived_dir / "y1_cat_bin_tw_fix.csv", index=False)
    return base_path, tw_path


def write_binned_catalogs_v2(
    derived_dir: Path,
    head_y1: pd.DataFrame,
    fit_qc_y1: pd.DataFrame,
    *,
    base_label: str,
    dz: float = DZ_DEFAULT,
    z_max: float = Z_MAX_DEFAULT,
    sigma_agg: str = "ivar",
    alpha: float = 0.14,
    beta: float = 3.1,
    sigma_int: float | str = "color_binned",
    sigma_vpec_kms: float = 300.0,
    f_tw_scale: float = 1.15,
    fit_promoted_y1: pd.DataFrame | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Path, Path]:
    """Improved writer that assigns per-SN σ_μ to Twilight-promoted objects.

    The returned CSVs include an IVAR-weighted ``z_eff`` column and updated
    ``sigma_mu`` based on per-SN uncertainties for both WFD and promoted SNe.

    Notes
    -----
    Also writes an unbinned per-SN CSV named
    ``y1_sn_unbinned_{base_label}.csv`` in ``derived_dir`` for available
    sources (WFD and, if provided, Twilight-promoted SNe) with columns
    including ``z``, ``c``, ``alpha_used``, ``beta_used``, ``sigma_int``,
    ``sigma_lens``, ``sigma_vpec``, and per-SN ``sigma_mu``.
    """

    z_edges = np.arange(0.0, z_max + dz + 1e-12, dz)
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])

    N_det = nz_hist(
        pd.to_numeric(head_y1["z"], errors="coerce").to_numpy(dtype=float), z_edges
    )
    N_cos = nz_hist(
        pd.to_numeric(fit_qc_y1["z"], errors="coerce").to_numpy(dtype=float), z_edges
    )

    band = (z_edges[:-1] >= Z_TW_MIN_DEFAULT) & (z_edges[1:] <= Z_TW_MAX_DEFAULT)
    N_cos_tw = N_cos.copy()
    N_cos_tw[band] = np.maximum(N_cos_tw[band], N_det[band])

    # Per-SN σμ for WFD cosmology SNe
    fit_qc_y1 = fit_qc_y1.copy()
    fit_qc_y1["sigma_mu_sn"] = sigma_mu_per_sn(
        fit_qc_y1,
        alpha=alpha,
        beta=beta,
        sigma_int=sigma_int,
        sigma_vpec_kms=sigma_vpec_kms,
    )

    # Optional per-SN σμ for explicit promoted-fit rows
    if fit_promoted_y1 is not None and "sigma_mu_sn" not in fit_promoted_y1:
        fit_promoted_y1 = fit_promoted_y1.copy()
        fit_promoted_y1["sigma_mu_sn"] = sigma_mu_per_sn(
            fit_promoted_y1,
            alpha=alpha,
            beta=beta,
            sigma_int=sigma_int,
            sigma_vpec_kms=sigma_vpec_kms,
        )

    # Build unbinned, per-SN DataFrame(s)
    def _per_sn_df(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        a_s = _num_series(df, ["SIM_alpha", "alpha"], alpha)
        b_s = _num_series(df, ["SIM_beta", "beta"], beta)
        z_s = _num_series(df, "z", np.nan).astype(float)
        c_s = _num_series(df, "c", np.nan).astype(float)
        if isinstance(sigma_int, str):
            if sigma_int.lower() != "color_binned":
                raise ValueError(
                    "sigma_int string must be 'color_binned' or a numeric value"
                )
            si_s = pd.Series(
                _sigma_int_from_color(c_s.to_numpy(dtype=float, na_value=np.nan)),
                index=df.index,
            )
        else:
            si_s = pd.Series(float(sigma_int), index=df.index)
        sig_lens_s = 0.055 * z_s
        sig_vpec_s = (5.0 / np.log(10.0)) * (
            sigma_vpec_kms / (299792.458 * np.maximum(z_s, 1e-3))
        )

        cols_id: dict[str, pd.Series] = {}
        for cand_id in ("CID", "SNID", "ID"):
            if cand_id in df.columns:
                cols_id[cand_id] = df[cand_id]
                break

        return pd.DataFrame(
            {
                **cols_id,
                "source": pd.Series(source_label, index=df.index),
                "z": z_s,
                "c": c_s,
                "alpha_used": a_s,
                "beta_used": b_s,
                "sigma_int": si_s,
                "sigma_lens": sig_lens_s,
                "sigma_vpec": sig_vpec_s,
                "sigma_mu": pd.to_numeric(df["sigma_mu_sn"], errors="coerce").astype(
                    float
                ),
            },
            index=df.index,
        )

    df_unbinned_list: list[pd.DataFrame] = []
    df_unbinned_list.append(_per_sn_df(fit_qc_y1, "WFD"))
    if fit_promoted_y1 is not None:
        df_unbinned_list.append(_per_sn_df(fit_promoted_y1, "TwilightPromoted"))
    df_unbinned = (
        pd.concat(df_unbinned_list, ignore_index=True)
        if any(len(x) for x in df_unbinned_list)
        else pd.DataFrame()
    )

    sigma_bin_base = np.full_like(z_mid, np.nan, dtype=float)
    z_eff_base = np.full_like(z_mid, np.nan, dtype=float)
    sigma_bin_tw = np.full_like(z_mid, np.nan, dtype=float)
    z_eff_tw = np.full_like(z_mid, np.nan, dtype=float)

    if rng is None:
        rng = np.random.default_rng(1234)

    for k in range(len(z_mid)):
        z_lo, z_hi = z_edges[k], z_edges[k + 1]

        # WFD in this bin
        m_wfd = (fit_qc_y1["z"] >= z_lo) & (fit_qc_y1["z"] < z_hi)
        sig_wfd = pd.to_numeric(
            fit_qc_y1.loc[m_wfd, "sigma_mu_sn"], errors="coerce"
        ).astype(float)
        z_wfd = pd.to_numeric(
            fit_qc_y1.loc[m_wfd, "z"], errors="coerce"
        ).astype(float)
        ok_wfd = np.isfinite(sig_wfd) & (sig_wfd > 0.0) & np.isfinite(z_wfd)
        sig_wfd = sig_wfd[ok_wfd].to_numpy()
        z_wfd = z_wfd[ok_wfd].to_numpy()

        if sig_wfd.size > 0:
            iv = 1.0 / (sig_wfd ** 2)
            ssum = iv.sum()
            n = sig_wfd.size
            if ssum > 0.0:
                sigma_bin_base[k] = float(np.sqrt(n / ssum))
                z_eff_base[k] = float((z_wfd * iv).sum() / ssum)

        # Promoted counts needed
        n_prom = int(max(0, N_cos_tw[k] - sig_wfd.size))
        sig_all = sig_wfd.copy()
        z_all = z_wfd.copy()

        if n_prom > 0:
            df_det_bin = head_y1[(head_y1["z"] >= z_lo) & (head_y1["z"] < z_hi)]
            df_fit_prom_bin = None
            if fit_promoted_y1 is not None:
                df_fit_prom_bin = fit_promoted_y1[
                    (fit_promoted_y1["z"] >= z_lo)
                    & (fit_promoted_y1["z"] < z_hi)
                ]
            s_prom, z_prom = sample_sigma_mu_promoted(
                df_det_bin=df_det_bin,
                df_fit_promoted_bin=df_fit_prom_bin,
                n_needed=n_prom,
                alpha=alpha,
                beta=beta,
                sigma_int=sigma_int,
                sigma_vpec_kms=sigma_vpec_kms,
                f_tw_scale=f_tw_scale,
                z_fallback=z_mid[k],
                rng=rng,
            )

            if (not np.isfinite(s_prom).all()) or s_prom.size == 0:
                med = float(np.nanmedian(sig_wfd)) if sig_wfd.size > 0 else 0.15
                s_prom = np.full(n_prom, f_tw_scale * med, dtype=float)
                z_prom = np.full(n_prom, z_mid[k], dtype=float)

            sig_all = np.concatenate([sig_all, s_prom])
            z_all = np.concatenate([z_all, z_prom])

        if sig_all.size > 0:
            iv = 1.0 / (sig_all ** 2)
            ssum = iv.sum()
            n = sig_all.size
            if ssum > 0.0:
                sigma_bin_tw[k] = float(np.sqrt(n / ssum))
                z_eff_tw[k] = float((z_all * iv).sum() / ssum)

    def _fill(values: np.ndarray, fill: float | np.ndarray | pd.Series) -> np.ndarray:
        """Forward/back fill finite values, then fill remaining NaNs.

        ``fill`` may be a scalar or a vector/Series matching the length of
        ``values`` (e.g., using ``z_mid`` to fill ``z_eff_*``).
        """
        s = pd.Series(values).replace([np.inf, -np.inf], np.nan).astype(float)
        s = s.fillna(method="ffill").fillna(method="bfill")
        if np.isscalar(fill):
            return s.fillna(fill).to_numpy(dtype=float)
        # Vector-like: align by index
        fill_series = pd.Series(fill, index=s.index)
        return s.fillna(fill_series).to_numpy(dtype=float)

    global_med = (
        float(np.nanmedian(fit_qc_y1["sigma_mu_sn"]))
        if "sigma_mu_sn" in fit_qc_y1
        else 0.12
    )

    sigma_bin_base = _fill(sigma_bin_base, global_med)
    sigma_bin_tw = _fill(sigma_bin_tw, global_med)
    z_eff_base = _fill(z_eff_base, z_mid)
    z_eff_tw = _fill(z_eff_tw, z_mid)

    df_base = pd.DataFrame(
        {"z": z_mid, "z_eff": z_eff_base, "N": N_cos, "sigma_mu": sigma_bin_base}
    )
    df_tw = pd.DataFrame(
        {"z": z_mid, "z_eff": z_eff_tw, "N": N_cos_tw, "sigma_mu": sigma_bin_tw}
    )

    base_path = derived_dir / f"y1_cat_bin_base_{base_label}.csv"
    tw_path = derived_dir / f"y1_cat_bin_tw_{base_label}.csv"

    df_base.to_csv(base_path, index=False)
    df_tw.to_csv(tw_path, index=False)

    # Compatibility filenames
    df_base.to_csv(derived_dir / "y1_cat_bin_base_fix.csv", index=False)
    df_tw.to_csv(derived_dir / "y1_cat_bin_tw_fix.csv", index=False)

    # Write unbinned per-SN export if available
    if len(df_unbinned) > 0:
        unbinned_path = derived_dir / f"y1_sn_unbinned_{base_label}.csv"
        df_unbinned.to_csv(unbinned_path, index=False)

    return base_path, tw_path


def load_binned_catalogs(
    derived_dir: Path,
    *,
    z_max: float | None = None,
    base_glob: str = "y1_cat_bin_base_*.csv",
    tw_glob: str = "y1_cat_bin_tw_*.csv",
    tw_guess_glob: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Load binned catalogs and clip to ``z_max`` if provided.
        deprecate
    """


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


def write_binned_catalogs_v3(
    derived_dir: Path,
    fit_qc: pd.DataFrame,
    *,
    label: str,
    dz: float = DZ_DEFAULT,
    z_max: float = Z_MAX_DEFAULT,
    sigma_agg: str = "ivar",
    alpha: float = 0.14,
    beta: float = 3.1,
    sigma_int: float | str = "color_binned",
    sigma_vpec_kms: float = 300.0,
) -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Bin by redshift and write a single CSV with N(z), σμ(z), and z_eff.
    This v3 depends only on fit_qc (no Twilight or promoted inputs).

    Parameters
    ----------
    derived_dir : pathlib.Path
        Output directory for the CSV.
    fit_qc : pandas.DataFrame
        FITRES-like table for cosmology-quality SNe with SALT2 columns.
    label : str
        Used in filename: cat_bin_{label}.csv
    dz, z_max : float
        Bin width and max redshift edge (bins start at 0).
    sigma_agg : {"ivar", "median"}
        Per-bin σμ aggregation. "ivar" = inverse-variance; "median" = robust median.
    alpha, beta : float
        Stretch/color coefficients for σμ pipeline (fallback if cols absent).
    sigma_int : float | str
        Intrinsic scatter; if "color_binned", interpolate from color-grid.
    sigma_vpec_kms : float
        Peculiar-velocity dispersion (km/s).

    Returns
    -------
    (out_path, df_out, z_edges, z_mid)
        out_path : Path to cat_bin_{label}.csv
        df_out   : DataFrame with columns ["z","z_eff","N","sigma_mu","w"]
        z_edges  : ndarray of bin edges
        z_mid    : ndarray of bin centers

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "z":[0.05,0.06,0.12],
    ...     "mBERR":[0.12]*3, "x1ERR":[1.0]*3, "cERR":[0.04]*3,
    ...     "COV_x1_c":[0.0]*3, "COV_mB_x1":[0.0]*3, "COV_mB_c":[0.0]*3,
    ...     "c":[0.0, 0.1, -0.05],
    ... })
    >>> out_path, df_out, z_edges, z_mid = write_binned_catalogs_v3(
    ...     Path("."), df, label="test", dz=0.05, z_max=0.2
    ... )
    """
    # ---- per-SN σμ ----
    fit_qc = fit_qc.copy()
    fit_qc["sigma_mu_sn"] = sigma_mu_per_sn(
        fit_qc, alpha=alpha, beta=beta,
        sigma_int=sigma_int, sigma_vpec_kms=sigma_vpec_kms
    )

    # ---- binning grid ----
    z_edges = np.arange(0.0, z_max + dz + 1e-12, dz)
    z_mid   = 0.5 * (z_edges[:-1] + z_edges[1:])

    # numeric z once, reuse everywhere
    z_num = pd.to_numeric(fit_qc.get("z", np.nan), errors="coerce").astype(float)
    z_num = z_num.replace([np.inf, -np.inf], np.nan)

    # histogram counts (finite z only)
    N = nz_hist(z_num[np.isfinite(z_num)].to_numpy(), z_edges)

    # ---- per-bin aggregation ----
    sigma_bin = np.full_like(z_mid, np.nan, dtype=float)
    z_eff     = np.full_like(z_mid, np.nan, dtype=float)

    agg = str(sigma_agg).lower().strip()
    if agg not in {"ivar", "median"}:
        raise ValueError("sigma_agg must be either 'ivar' or 'median'")

    # pre-cache sigma_mu_sn as numeric
    sig_sn = pd.to_numeric(fit_qc["sigma_mu_sn"], errors="coerce").astype(float)
    sig_sn = sig_sn.replace([np.inf, -np.inf], np.nan)

    for k in range(len(z_mid)):
        z_lo, z_hi = z_edges[k], z_edges[k + 1]
        m = (z_num >= z_lo) & (z_num < z_hi)

        if not bool(m.any()):
            continue

        s = sig_sn[m]
        s = s[(s > 0) & np.isfinite(s)]

        if s.empty:
            continue

        if agg == "ivar":
            inv  = 1.0 / (s.values ** 2)
            ssum = float(inv.sum())
            n    = float(s.size)
            if ssum > 0.0:
                sigma_bin[k] = float(np.sqrt(n / ssum))
                z_bin = z_num[m].to_numpy(dtype=float)
                # align z to s by selecting same finite-positive mask
                z_bin = z_bin[(sig_sn[m] > 0) & np.isfinite(sig_sn[m])]
                z_eff[k] = float((z_bin * inv).sum() / ssum)
        else:
            sigma_bin[k] = float(np.nanmedian(s.values))

    # ---- fill missing ----
# --- 用下面这个实现替换 v3 里的 _fill() 辅助函数 ---
    def _fill(values: np.ndarray, fill: float | np.ndarray | pd.Series) -> np.ndarray:
        s = pd.Series(values).replace([np.inf, -np.inf], np.nan).astype(float)
        s = s.ffill().bfill()
        # 标量：直接用
        if np.isscalar(fill):
            return s.fillna(float(fill)).to_numpy(dtype=float)
        # 非标量：转成 Series 并与 s 对齐后按位置填
        if isinstance(fill, pd.Series):
            fill_s = fill.astype(float).reindex(s.index)
        else:
            fill_arr = np.asarray(fill, dtype=float)
            if fill_arr.shape != s.shape:
                # 形状不匹配就退回到“用 fill 的全体中位数”作为标量兜底
                fallback = float(np.nanmedian(fill_arr)) if fill_arr.size else np.nan
                return s.fillna(fallback).to_numpy(dtype=float)
            fill_s = pd.Series(fill_arr, index=s.index)
        out = s.where(s.notna(), fill_s)   # 只在 NaN 的位置用 fill_s
        return out.to_numpy(dtype=float)


    global_med = float(np.nanmedian(sig_sn.values)) if sig_sn.size else 0.12
    sigma_bin  = _fill(sigma_bin, global_med)
    z_eff      = _fill(z_eff, z_mid)

    # ---- output ----
    df_out = pd.DataFrame({
        "z": z_mid,
        "z_eff": z_eff,
        "N": N.astype(int),
        "sigma_mu": sigma_bin,
    })

    # Cosmology weight per redshift bin: w = N / sigma_mu^2 (zero if sigma_mu <= 0)
    # Use numeric arrays to avoid pandas dtype pitfalls
    _sig = df_out["sigma_mu"].to_numpy(dtype=float)
    _N   = df_out["N"].to_numpy(dtype=float)
    df_out["w"] = np.where((_sig > 0.0) & np.isfinite(_sig), _N / (_sig ** 2), 0.0)

    derived_dir.mkdir(parents=True, exist_ok=True)
    out_path    = derived_dir / f"cat_bin_{label}.csv"
    compat_path = derived_dir / f"y1_cat_bin_base_{label}.csv"

    df_out.to_csv(out_path, index=False)
    df_out.to_csv(compat_path, index=False)

    # return both the path AND the in-memory results + bins
    return out_path, df_out, z_edges, z_mid
