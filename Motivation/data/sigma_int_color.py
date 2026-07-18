"""Derive color-dependent intrinsic scatter, ``sigma_int(c)``, from FITRES output.

This script now follows the SNANA/SALT2mu intrinsic-scatter definition:

* If ``biasCor_muCOVADD`` exists (Dillon's binned intrinsic term), use
  ``sigma_int,SN = sqrt(max(biasCor_muCOVADD, 0))``.
* Otherwise infer an effective intrinsic term from distance errors as
  ``sigma_int,SN = sqrt(max(MUERR^2 - sigma_meas^2, 0))`` where
  ``sigma_meas^2 = MUERR_RAW^2 + MUERR_VPEC^2``.

For diagnostics, this script also computes ``sigma_HR(c)`` (robust residual
scatter) and ``sigma_meas(c)`` in the same color windows.

Running this file prints arrays suitable for hard-coding in ``tw_binning.py``
and writes:
* ``sigma_int_color.npz`` with both intrinsic and total-scatter diagnostics
* ``sigma_int_color_curve.png`` showing ``sigma_int(c)``
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d


HERE = Path(__file__).resolve().parent
FITRES_DEFAULT = HERE / "FITOPT000_MUOPT000.FITRES"
WINDOW_SIZE = 60  # number of SNe per window (equal-count)
SIGMA_MAD_FACTOR = 1.4826  # convert MAD to σ for a Gaussian
C_MIN = -0.3
C_MAX = 0.3
OUTPUT_GRID_SIZE = 61  # number of points printed/saved for downstream use
SIGMA_VPEC_KMS = 300.0


def _iter_fitres_rows(path: Path) -> Iterator[list[str]]:
    """Yield value rows from a FITRES file as lists of strings."""

    columns: list[str] | None = None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for raw in fh:
            if raw.startswith("VARNAMES:"):
                columns = raw.split()[1:]
                continue
            if columns is None or raw.startswith("#") or not raw.strip():
                continue
            parts = raw.split()
            if parts and parts[0].endswith(":"):
                parts = parts[1:]
            if len(parts) != len(columns):
                continue
            yield parts


def load_fitres(path: Path) -> pd.DataFrame:
    """Load the FITRES table with numeric columns of interest."""

    rows = list(_iter_fitres_rows(path))
    if not rows:
        raise FileNotFoundError(f"No data rows found in {path}")

    df = pd.DataFrame(rows)
    # Column names were read from the header; apply them now
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        header_line = next((line for line in fh if line.startswith("VARNAMES:")), None)
    if header_line is None:
        raise ValueError("VARNAMES header missing in FITRES file")
    columns = header_line.split()[1:]
    df.columns = columns

    numeric_cols = [
        "c",
        "x1",
        "MU",
        "MUMODEL",
        "M0DIF",
        "MURES",
        "MUERR",
        "MUERR_RAW",
        "MUERR_VPEC",
        "MUERR_RENORM",
        "biasCor_mu",
        "biasCor_muCOVADD",
        "biasCor_muCOVSCALE",
        "ERRFLAG_FIT",
        "CUTFLAG_SNANA",
        "FITPROB",
        "ISDATA_REAL",
        "z",
        "zHD",
        "zCMB",
        "zHEL",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def robust_sigma(values: np.ndarray) -> float:
    """Return the MAD-based scatter estimator for the provided residuals."""

    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0.0:
        return float(np.std(values, ddof=1))
    return float(SIGMA_MAD_FACTOR * mad)


def _measurement_sigma_mu(df: pd.DataFrame) -> np.ndarray:
    """Return per-SN non-intrinsic sigma(mu) used to define excess scatter.

    Preference order:
    1) Reconstruct from MUERR_RAW + MUERR_VPEC when available.
    2) Fallback to MUERR directly.

    SNANA semantics (SALT2mu):
    * MUERR_RAW excludes vpec and intrinsic/covadd, and already includes lensing.
    * MUERR_VPEC is the vpec-only contribution.
    Therefore do not add lensing again when using MUERR_RAW.
    """

    if "MUERR_RAW" in df and df["MUERR_RAW"].notna().any():
        muerr_raw = pd.to_numeric(df["MUERR_RAW"], errors="coerce").to_numpy(dtype=float)

        z_series: pd.Series | None = None
        for z_col in ("zHD", "z", "zCMB", "zHEL"):
            if z_col in df.columns:
                z_series = pd.to_numeric(df[z_col], errors="coerce")
                break
        if z_series is None:
            z = np.full(len(df), np.nan, dtype=float)
        else:
            z = z_series.to_numpy(dtype=float)
        z_safe = np.clip(np.nan_to_num(z, nan=0.0), 0.0, None)

        if "MUERR_VPEC" in df and df["MUERR_VPEC"].notna().any():
            sig_vpec = pd.to_numeric(df["MUERR_VPEC"], errors="coerce").to_numpy(dtype=float)
        else:
            z_for_vpec = np.clip(z_safe, 1e-3, None)
            sig_vpec = (5.0 / np.log(10.0)) * (SIGMA_VPEC_KMS / (299792.458 * z_for_vpec))

        sigma = np.sqrt(np.maximum(muerr_raw**2 + sig_vpec**2, 0.0))
        return sigma

    if "MUERR" in df and df["MUERR"].notna().any():
        return pd.to_numeric(df["MUERR"], errors="coerce").to_numpy(dtype=float)

    raise ValueError(
        "FITRES must include MUERR or MUERR_RAW to estimate measurement sigma(mu)."
    )


def _snana_sigma_int_per_sn(df: pd.DataFrame) -> np.ndarray:
    """Return per-SN intrinsic sigma(mu) following SALT2mu conventions.

    Preference order:
    1) ``biasCor_muCOVADD`` if available: sigma_int = sqrt(max(muCOVadd, 0)).
       In SALT2mu this is the additive variance term used for binned intrinsic
       scatter (Dillon mode).
    2) Fallback effective intrinsic term from total-vs-measurement variance:
       sigma_int = sqrt(max(MUERR^2 - sigma_meas^2, 0)).
    """

    if "biasCor_muCOVADD" in df and df["biasCor_muCOVADD"].notna().any():
        covadd = pd.to_numeric(df["biasCor_muCOVADD"], errors="coerce").to_numpy(dtype=float)
        return np.sqrt(np.maximum(covadd, 0.0))

    if "MUERR" in df and df["MUERR"].notna().any():
        muerr = pd.to_numeric(df["MUERR"], errors="coerce").to_numpy(dtype=float)
        sigma_meas = _measurement_sigma_mu(df)
        sigma_int2 = muerr**2 - sigma_meas**2
        return np.sqrt(np.maximum(sigma_int2, 0.0))

    return np.zeros(len(df), dtype=float)


def sigma_int_vs_color(
    df: pd.DataFrame,
    *,
    centers: np.ndarray | None = None,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute SNANA-defined ``sigma_int(c)`` on a fixed color range.

    The profile is evaluated at fixed color centers spanning ``[C_MIN, C_MAX]``
    (default: 61 points). At each center, an equal-count local window
    (``window_size`` SNe) is selected from the sorted sample.

    Returns
    -------
    centers : ndarray
        Fixed c centers where the profile is evaluated.
    sigma_int : ndarray
        SNANA-defined intrinsic scatter estimate at each center.
    sigma_hr : ndarray
        Diagnostic total robust Hubble-residual scatter at each center.
    sigma_meas : ndarray
        Diagnostic representative (median) measurement sigma(mu) per center window.
    """

    if "MURES" in df.columns and df["MURES"].notna().any():
        resid = df["MURES"].to_numpy(dtype=float)
    else:
        bias = df["biasCor_mu"].fillna(0.0) if "biasCor_mu" in df else 0.0
        resid = (
            df["MU"].to_numpy(dtype=float)
            - np.asarray(bias, dtype=float)
            - df["MUMODEL"].to_numpy(dtype=float)
            - df["M0DIF"].to_numpy(dtype=float)
        )

    mask = np.ones(len(df), dtype=bool)
    if "ISDATA_REAL" in df:
        mask &= df["ISDATA_REAL"].fillna(0).astype(int).eq(0).to_numpy()
    if "ERRFLAG_FIT" in df:
        mask &= df["ERRFLAG_FIT"].fillna(0).astype(int).eq(0).to_numpy()
    if "CUTFLAG_SNANA" in df:
        cutflag = df["CUTFLAG_SNANA"].fillna(0).astype(int)
        if (cutflag == 0).any():
            mask &= cutflag.eq(0).to_numpy()
    if "FITPROB" in df:
        mask &= (df["FITPROB"] > 0.05).fillna(False).to_numpy()
    if "x1" in df:
        mask &= ((df["x1"] > -3) & (df["x1"] < 3)).fillna(False).to_numpy()
    if "c" in df:
        mask &= ((df["c"] >= C_MIN) & (df["c"] <= C_MAX)).fillna(False).to_numpy()

    sigma_meas_all = _measurement_sigma_mu(df)
    sigma_int_all = _snana_sigma_int_per_sn(df)

    c = df.loc[mask, "c"].to_numpy(dtype=float)
    r = resid[mask]
    s_meas = sigma_meas_all[mask]
    s_int = sigma_int_all[mask]

    finite = np.isfinite(c) & np.isfinite(r) & np.isfinite(s_meas) & np.isfinite(s_int)
    c = c[finite]
    r = r[finite]
    s_meas = s_meas[finite]
    s_int = s_int[finite]

    order = np.argsort(c)
    c = c[order]
    r = r[order]
    s_meas = s_meas[order]
    s_int = s_int[order]

    if len(c) < window_size:
        raise ValueError(
            f"Not enough SNe ({len(c)}) for window size {window_size}; "
            "reduce WINDOW_SIZE or relax cuts."
        )

    if centers is None:
        eval_centers = np.linspace(C_MIN, C_MAX, OUTPUT_GRID_SIZE)
    else:
        eval_centers = np.asarray(centers, dtype=float)
    eval_centers = np.clip(eval_centers, C_MIN, C_MAX)

    centers_out: list[float] = []
    sigma_int_vals: list[float] = []
    sigma_hr_vals: list[float] = []
    sigma_meas_vals: list[float] = []

    half = window_size // 2
    max_start = len(c) - window_size
    for c0 in eval_centers:
        # Keep an equal-count window locally centered around the query c.
        pivot = int(np.searchsorted(c, c0, side="left"))
        start = pivot - half
        start = max(0, min(start, max_start))
        end = start + window_size

        slice_c = c[start:end]
        slice_r = r[start:end]
        slice_meas = s_meas[start:end]
        slice_int = s_int[start:end]
        sigma_hr = robust_sigma(slice_r)
        sigma_meas2 = float(np.nanmedian(slice_meas**2))
        sigma_int2 = float(np.nanmedian(slice_int**2))
        sigma_int = float(np.sqrt(max(sigma_int2, 0.0)))

        centers_out.append(float(c0))
        sigma_hr_vals.append(sigma_hr)
        sigma_meas_vals.append(float(np.sqrt(max(sigma_meas2, 0.0))))
        sigma_int_vals.append(sigma_int)

    centers_arr = np.asarray(centers_out, dtype=float)
    sigma_int_arr = np.asarray(sigma_int_vals, dtype=float)
    sigma_hr_arr = np.asarray(sigma_hr_vals, dtype=float)
    sigma_meas_arr = np.asarray(sigma_meas_vals, dtype=float)

    valid = np.isfinite(sigma_int_arr) & np.isfinite(sigma_hr_arr) & np.isfinite(sigma_meas_arr)
    if valid.sum() < 4:
        raise ValueError("Insufficient valid windows for interpolation")

    return (
        centers_arr[valid],
        sigma_int_arr[valid],
        sigma_hr_arr[valid],
        sigma_meas_arr[valid],
    )


def smooth_curve(
    centers: np.ndarray,
    sigmas: np.ndarray,
    *,
    grid_size: int = OUTPUT_GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth the window scatter with a PCHIP interpolator."""

    sorter = np.argsort(centers)
    centers = centers[sorter]
    sigmas = sigmas[sorter]

    kernel = max(3, int(round(0.12 * len(sigmas))))
    if kernel % 2 == 0:
        kernel += 1
    if kernel > 1:
        sigmas = uniform_filter1d(sigmas, size=kernel, mode="nearest")

    interpolator = PchipInterpolator(centers, sigmas, extrapolate=False)
    grid = np.linspace(C_MIN, C_MAX, grid_size)
    clamped = np.clip(grid, centers.min(), centers.max())
    smoothed = interpolator(clamped)
    return grid, smoothed


def plot_sigma_int(
    centers: np.ndarray,
    sigma_int: np.ndarray,
    sigma_hr: np.ndarray,
    grid_c: np.ndarray,
    grid_sigma_int: np.ndarray,
    out_path: Path,
) -> None:
    """Create the diagnostic plot for intrinsic and total residual scatter."""

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(
        grid_c,
        grid_sigma_int,
        color="#5b8fd6",
        linewidth=2.5,
        label=r"$\sigma_{\rm int}(c)$",
    )
    ax.scatter(centers, sigma_int, color="#d98d65", alpha=0.65, s=20, label=r"Window $\sigma_{\rm int}$")
    ax.scatter(
        centers,
        sigma_hr,
        color="#7a7a7a",
        alpha=0.35,
        s=15,
        marker="x",
        label=r"Window $\sigma_{\rm HR}$",
    )
    ax.set_xlabel("$c$")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(C_MIN, C_MAX)
    ax.set_ylim(0.0, 0.35)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25, linestyle=":", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def dump_arrays(name: str, values: Iterable[float]) -> None:
    print(f"{name} = [")
    for val in values:
        print(f"    {val:.8f},")
    print("]")


def main(path: Path = FITRES_DEFAULT) -> None:
    df = load_fitres(path)
    centers, sigma_int, sigma_hr, sigma_meas = sigma_int_vs_color(df)
    grid_c, grid_sigma_int = smooth_curve(centers, sigma_int)
    _, grid_sigma_hr = smooth_curve(centers, sigma_hr)

    dump_arrays("_SIGMA_INT_COLOR_CENTERS", grid_c)
    dump_arrays("_SIGMA_INT_COLOR_VALUES", grid_sigma_int)

    np.savez(
        HERE / "sigma_int_color.npz",
        window_centers=centers,
        window_sigmas=sigma_int,  # backward-compatible key: now intrinsic by definition
        window_sigma_int=sigma_int,
        window_hr_sigmas=sigma_hr,
        window_meas_sigmas=sigma_meas,
        grid_centers=grid_c,
        grid_sigmas=grid_sigma_int,  # backward-compatible key: now intrinsic by definition
        grid_sigma_int=grid_sigma_int,
        grid_hr_sigmas=grid_sigma_hr,
    )
    plot_sigma_int(
        centers,
        sigma_int,
        sigma_hr,
        grid_c,
        grid_sigma_int,
        HERE / "sigma_int_color_curve.png",
    )


if __name__ == "__main__":
    main()
