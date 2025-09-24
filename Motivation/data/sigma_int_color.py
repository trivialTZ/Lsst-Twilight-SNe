"""Derive the color-dependent distance-modulus scatter from FITRES output.

This replicates the approach used for the BS21 simulations:

* Use the cosmology residual (``MURES``) for each SN
* Apply the same QA cuts as the cosmology analysis
* Measure a robust scatter inside equal-count sliding windows in SALT2 color
* Smooth those window measurements with a shape-preserving interpolator

Running this file prints the centers/values suitable for hard-coding in
``tw_binning.py`` (or loading from the saved ``sigma_int_color.npz``) and writes
out ``sigma_int_color_curve.png`` showing the familiar blue curve.
"""

from __future__ import annotations

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
WINDOW_STEP = 15  # slide by this many SNe between windows
SIGMA_MAD_FACTOR = 1.4826  # convert MAD to σ for a Gaussian
C_MIN = -0.3
C_MAX = 0.3
OUTPUT_GRID_SIZE = 61  # number of points printed/saved for downstream use


def _iter_fitres_rows(path: Path) -> Iterator[list[str]]:
    """Yield value rows from a FITRES file as lists of strings."""

    columns: list[str] | None = None
    with path.open() as fh:
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
    header_line = next(
        (line for line in path.read_text().splitlines() if line.startswith("VARNAMES:")),
        None,
    )
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
        "MUERR_RENORM",
        "biasCor_mu",
        "ERRFLAG_FIT",
        "CUTFLAG_SNANA",
        "FITPROB",
        "ISDATA_REAL",
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


def sigma_vs_color(
    df: pd.DataFrame,
    *,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute robust σ_HR(c) using equal-count sliding windows."""

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

    c = df.loc[mask, "c"].to_numpy(dtype=float)
    r = resid[mask]

    finite = np.isfinite(c) & np.isfinite(r)
    c = c[finite]
    r = r[finite]

    order = np.argsort(c)
    c = c[order]
    r = r[order]

    if len(c) < window_size:
        raise ValueError(
            f"Not enough SNe ({len(c)}) for window size {window_size}; "
            "reduce WINDOW_SIZE or relax cuts."
        )

    centers: list[float] = []
    sigmas: list[float] = []
    end_limit = len(c) - window_size
    for start in range(0, end_limit + 1, step):
        end = start + window_size
        slice_c = c[start:end]
        slice_r = r[start:end]
        centers.append(float(np.median(slice_c)))
        sigmas.append(robust_sigma(slice_r))

    if centers:
        last_start = len(c) - window_size
        if last_start > 0 and last_start % step != 0:
            slice_c = c[last_start:]
            slice_r = r[last_start:]
            last_center = float(np.median(slice_c))
            if abs(last_center - centers[-1]) > 1e-6:
                centers.append(last_center)
                sigmas.append(robust_sigma(slice_r))

    centers_arr = np.asarray(centers, dtype=float)
    sigmas_arr = np.asarray(sigmas, dtype=float)

    valid = np.isfinite(sigmas_arr)
    if valid.sum() < 4:
        raise ValueError("Insufficient valid windows for interpolation")

    return centers_arr[valid], sigmas_arr[valid]


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
    sigmas: np.ndarray,
    grid_c: np.ndarray,
    grid_sigma: np.ndarray,
    out_path: Path,
) -> None:
    """Create the diagnostic plot with the blue σ_HR(c) curve."""

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(grid_c, grid_sigma, color="#5b8fd6", linewidth=2.5, label=r"$\sigma_{HR}(c)$")
    ax.scatter(centers, sigmas, color="#d98d65", alpha=0.6, s=22, label="Window σ")
    ax.set_xlabel("$c$")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(C_MIN, C_MAX)
    ax.set_ylim(0.05, 0.5)
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
    centers, sigmas = sigma_vs_color(df)
    grid_c, grid_sigma = smooth_curve(centers, sigmas)

    dump_arrays("_SIGMA_INT_COLOR_CENTERS", grid_c)
    dump_arrays("_SIGMA_INT_COLOR_VALUES", grid_sigma)

    np.savez(
        HERE / "sigma_int_color.npz",
        window_centers=centers,
        window_sigmas=sigmas,
        grid_centers=grid_c,
        grid_sigmas=grid_sigma,
    )
    plot_sigma_int(centers, sigmas, grid_c, grid_sigma, HERE / "sigma_int_color_curve.png")


if __name__ == "__main__":
    main()
