"""Peculiar-velocity robustness helpers for twilight cosmology forecasts."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Mapping

# Keep dense covariance algebra and FLIP workers from multiplying each
# other's thread pools.  These must be set before importing NumPy/SciPy.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .tw_binning import sigma_mu_per_sn
from .tw_convert_ra_dec import fix_ra_dec_from_simlib
from .tw_io import read_fitres_any, standardize_fitres
from .tw_qc import ross_qc_with_report

COSMO_DICT = {
    "h": 0.6766,
    "sigma8": 0.8102,
    "n_s": 0.9665,
    "omega_b": 0.02242,
    "omega_cdm": 0.11933,
}

DEFAULT_Z_RANGES: tuple[tuple[float, float], ...] = (
    (0.02, 0.06),
    (0.06, 0.10),
    (0.02, 0.14),
)

FlowInput = pd.DataFrame | Mapping[tuple[float, float], pd.DataFrame]

PV_FISHER_PARAMETERS: tuple[str, ...] = ("fs8", "sigv", "sigma_M2")
DEFAULT_PV_PARAMETER_VALUES = {
    "fs8": 1.0,
    "sigv": 300.0,
    # Added magnitude variance, not an added rms.  The zero fiducial therefore
    # adds no scatter on top of dmu_error while retaining a non-zero derivative.
    "sigma_M2": 0.0,
}

PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "savefig.dpi": 300,
    "figure.dpi": 120,
}


def repo_root() -> Path:
    """Return the Lsst-Twilight-SNe repository root."""

    return Path(__file__).resolve().parent.parent


def densest_k_year_window(
    df: pd.DataFrame,
    time_col: str = "PKMJD",
    k_years: float = 1.0,
) -> tuple[float, float]:
    """Return the densest contiguous ``k_years`` window in ``time_col``."""

    if time_col not in df.columns:
        return -np.inf, np.inf
    times = pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy()
    if times.size == 0:
        return -np.inf, np.inf
    times = np.sort(times)
    width = 365.25 * float(k_years)

    start = 0
    best_count = 0
    best_range = (float(times[0]), float(times[0] + width))
    for stop in range(times.size):
        while times[stop] - times[start] >= width:
            start += 1
        count = stop - start + 1
        if count > best_count:
            best_count = count
            best_range = (float(times[start]), float(times[start] + width))
    return best_range


def slice_by_window(
    df: pd.DataFrame,
    t0: float,
    t1: float,
    time_col: str = "PKMJD",
) -> pd.DataFrame:
    """Return rows with ``time_col`` in the half-open interval ``[t0, t1)``."""

    if time_col not in df.columns:
        return df.copy()
    times = pd.to_numeric(df[time_col], errors="coerce")
    return df.loc[(times >= t0) & (times < t1)].copy()


def first_existing_path(*paths: Path) -> Path:
    """Return the first existing, non-empty path."""

    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        "None of the candidate paths exist with non-zero size: "
        + ", ".join(str(path) for path in paths)
    )


def read_fitres_lowz_streaming(
    path: str | Path,
    *,
    z_min: float = 0.0,
    z_max: float = 0.14,
) -> pd.DataFrame:
    """Read only low-redshift, forecast-relevant columns from a large FITRES.

    The independent multi-year WFD benchmark is more than a gigabyte when
    decompressed.  The general FITRES reader materializes every row and column,
    although the PV forecast only needs the low-z subset and a small set of
    fit, QC, position, and covariance columns.  This streaming reader preserves
    the same FITRES values while keeping memory use bounded.
    """

    path = Path(path)
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    requested = {
        "CID",
        "CIDint",
        "zHD",
        "zCMB",
        "z",
        "PKMJD",
        "PKMJDERR",
        "x1",
        "x1ERR",
        "c",
        "cERR",
        "mBERR",
        "x0",
        "x0ERR",
        "SNRMAX1",
        "SNRMAX2",
        "SNRMAX3",
        "SNRSUM",
        "FITPROB",
        "SIM_alpha",
        "SIM_beta",
        "COV_x1_c",
        "COV_x0_x1",
        "COV_x1_x0",
        "COV_x0_c",
        "COV_c_x0",
        "COV_mB_c",
        "COV_mB_x1",
        "RA",
        "DEC",
    }
    names: list[str] | None = None
    selected_names: list[str] = []
    selected_indices: list[int] = []
    z_index: int | None = None
    rows: list[list[str]] = []

    with opener(path, "rt", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            if names is None:
                if stripped.upper().startswith("VARNAMES:"):
                    names = stripped.split()[1:]
                    selected_names = [name for name in names if name in requested]
                    selected_indices = [names.index(name) for name in selected_names]
                    for candidate in ("zHD", "zCMB", "z"):
                        if candidate in names:
                            z_index = names.index(candidate)
                            break
                    if z_index is None:
                        raise RuntimeError(f"No redshift column found in {path}")
                continue
            if stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if tokens and tokens[0].endswith(":"):
                tokens = tokens[1:]
            if len(tokens) < len(names):
                continue
            try:
                redshift = float(tokens[z_index])
            except (TypeError, ValueError):
                continue
            if z_min < redshift < z_max:
                rows.append([tokens[index] for index in selected_indices])

    if names is None:
        raise RuntimeError(f"VARNAMES not found in {path}")
    frame = pd.DataFrame(rows, columns=selected_names)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return standardize_fitres(frame)


def filter_valid_radec(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with finite RA/DEC inside the physical sky range."""

    if "RA" not in df.columns or "DEC" not in df.columns:
        return df.copy()
    out = df.copy()
    out["RA"] = pd.to_numeric(out["RA"], errors="coerce")
    out["DEC"] = pd.to_numeric(out["DEC"], errors="coerce")
    return out.loc[
        out["RA"].between(0, 360) & out["DEC"].between(-90, 90)
    ].copy()


def make_flip_ready(
    data_combine: pd.DataFrame,
    *,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    cosmo=Planck18,
) -> pd.DataFrame:
    """Prepare a FLIP-ready table from a FITRES-like DataFrame."""

    df = data_combine.copy()
    if ra_col not in df.columns or dec_col not in df.columns:
        raise KeyError(f"Input must contain '{ra_col}' and '{dec_col}'.")

    ra_deg = pd.to_numeric(df[ra_col], errors="coerce")
    dec_deg = pd.to_numeric(df[dec_col], errors="coerce")
    zarr = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)

    df["ra"] = np.deg2rad(np.mod(ra_deg, 360.0))
    df["dec"] = np.deg2rad(dec_deg)
    df["rcom_zobs"] = cosmo.comoving_transverse_distance(zarr).value * cosmo.h
    df["hubble_norm"] = 100.0 * cosmo.efunc(zarr)
    df["zobs"] = pd.to_numeric(df["z"], errors="coerce")

    keep = [
        "ra",
        "dec",
        "rcom_zobs",
        "hubble_norm",
        "zobs",
        "SIM_alpha",
        "SIM_beta",
        "mBERR",
        "x0",
        "x0ERR",
        "x1ERR",
        "cERR",
        "COV_x1_c",
        "COV_x0_x1",
        "COV_x1_x0",
        "COV_x0_c",
        "COV_c_x0",
        "COV_mB_c",
        "COV_mB_x1",
        "z",
        "c",
    ]
    return df.loc[:, [col for col in keep if col in df.columns]].copy()


def _validate_pv_parameter_values(
    parameter_dict: Mapping[str, float] | None
) -> dict[str, float]:
    """Validate the three parameters represented by the local PV Fisher.

    ``sigma_M`` is deliberately not accepted.  FLIP parameterizes it as an rms,
    whose covariance derivative vanishes at the zero-scatter fiducial.  The
    local Fisher instead uses ``sigma_M2``: an additive magnitude variance in
    mag².  This is a formal local nuisance coordinate and may take either sign
    in the Gaussian Fisher approximation, although its fiducial value must be
    non-negative.
    """

    values_in = (
        DEFAULT_PV_PARAMETER_VALUES if parameter_dict is None else parameter_dict
    )
    requested = set(values_in)
    expected = set(PV_FISHER_PARAMETERS)
    if "sigma_M" in requested:
        raise ValueError(
            "The nuisance parameter 'sigma_M' is not supported because its "
            "covariance derivative is zero at sigma_M=0. Use the additive "
            "variance parameter 'sigma_M2' (mag^2) instead."
        )
    missing = sorted(expected - requested)
    extra = sorted(requested - expected)
    if missing or extra:
        raise ValueError(
            "PV Fisher parameters must be exactly "
            f"{list(PV_FISHER_PARAMETERS)}; missing={missing}, extra={extra}. "
            "All reported sigma_fs8 values are required to marginalize over "
            "both sigv and sigma_M2."
        )

    values: dict[str, float] = {}
    for name in PV_FISHER_PARAMETERS:
        value = float(values_in[name])
        if not np.isfinite(value):
            raise ValueError(f"Fiducial parameter {name!r} must be finite.")
        values[name] = value
    if values["fs8"] <= 0:
        raise ValueError("The fs8 fiducial must be positive.")
    if values["sigv"] <= 0:
        raise ValueError("The sigv fiducial must be positive (km/s).")
    if values["sigma_M2"] < 0:
        raise ValueError("The sigma_M2 fiducial must be non-negative (mag^2).")
    return values


def _validate_parameter_priors(
    parameter_priors: Mapping[str, float] | None,
) -> dict[str, float]:
    """Return finite Gaussian prior widths in the parameters' physical units."""

    if parameter_priors is None:
        return {}
    unknown = sorted(set(parameter_priors) - set(PV_FISHER_PARAMETERS))
    if unknown:
        raise ValueError(f"Unknown PV Fisher prior parameter(s): {unknown}.")
    priors: dict[str, float] = {}
    for name, width_in in parameter_priors.items():
        width = float(width_in)
        if not np.isfinite(width) or width <= 0:
            raise ValueError(f"Prior width for {name!r} must be finite and positive.")
        priors[name] = width
    return priors


def _validate_fisher_properties(
    fisher_properties: Mapping[str, object] | None
) -> dict[str, float]:
    """Validate numerical settings for the local covariance-trace Fisher."""

    properties = {} if fisher_properties is None else dict(fisher_properties)
    allowed = {"inversion_method", "max_condition_number", "rank_rtol"}
    unknown = sorted(set(properties) - allowed)
    if unknown:
        raise ValueError(f"Unknown local PV Fisher properties: {unknown}.")
    inversion_method = properties.pop("inversion_method", "cholesky")
    if inversion_method != "cholesky":
        raise ValueError(
            "The nuisance-marginalized PV Fisher supports only the numerically "
            "stable 'cholesky' inversion method. The former FLIP 'inverse' "
            "setting no longer applies."
        )
    max_condition_number = float(properties.pop("max_condition_number", 1.0e12))
    rank_rtol = float(properties.pop("rank_rtol", 1.0e-12))
    if not np.isfinite(max_condition_number) or max_condition_number <= 1:
        raise ValueError("max_condition_number must be finite and greater than one.")
    if not np.isfinite(rank_rtol) or not 0 < rank_rtol < 1:
        raise ValueError("rank_rtol must lie strictly between zero and one.")
    return {
        "max_condition_number": max_condition_number,
        "rank_rtol": rank_rtol,
    }


def _gaussian_covariance_trace_fisher(
    covariance: np.ndarray,
    covariance_derivatives: Mapping[str, np.ndarray],
    *,
    parameter_scales: Mapping[str, float],
    parameter_priors: Mapping[str, float] | None = None,
    max_condition_number: float = 1.0e12,
    rank_rtol: float = 1.0e-12,
) -> dict[str, object]:
    r"""Evaluate and invert a Gaussian covariance Fisher matrix.

    For a zero-mean Gaussian data vector this computes

    .. math:: F_{ab}=\tfrac12\mathrm{Tr}[C^{-1}C_{,a}C^{-1}C_{,b}].

    Derivatives are whitened with a Cholesky factor instead of explicitly
    forming ``C^{-1}``.  ``parameter_scales`` only changes numerical
    coordinates; reported covariances are transformed back to physical units.
    Gaussian priors are optional and absent by default.
    """

    names = tuple(covariance_derivatives)
    if not names:
        raise ValueError("At least one covariance derivative is required.")
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("The fiducial covariance must be a square matrix.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("The fiducial covariance contains non-finite values.")
    covariance = 0.5 * (covariance + covariance.T)
    try:
        covariance_cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "The fiducial PV covariance is not positive definite."
        ) from exc

    scales = np.empty(len(names), dtype=float)
    whitened_derivatives: list[np.ndarray] = []
    for index, name in enumerate(names):
        if name not in parameter_scales:
            raise ValueError(f"No numerical parameter scale supplied for {name!r}.")
        scale = float(parameter_scales[name])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Parameter scale for {name!r} must be finite and positive."
            )
        scales[index] = scale

        derivative = np.asarray(covariance_derivatives[name], dtype=float)
        if derivative.shape != covariance.shape:
            raise ValueError(
                f"Covariance derivative {name!r} has shape {derivative.shape}; "
                f"expected {covariance.shape}."
            )
        if not np.all(np.isfinite(derivative)):
            raise ValueError(
                f"Covariance derivative {name!r} contains non-finite values."
            )
        derivative = 0.5 * (derivative + derivative.T)
        # If p = scale*q, then C_,q = scale*C_,p.  Whiten as
        # L^-1 C_,q L^-T so the trace is a stable Frobenius product.
        left_whitened = solve_triangular(
            covariance_cholesky,
            scale * derivative,
            lower=True,
            check_finite=False,
        )
        whitened = solve_triangular(
            covariance_cholesky,
            left_whitened.T,
            lower=True,
            check_finite=False,
        ).T
        whitened_derivatives.append(0.5 * (whitened + whitened.T))

    fisher_scaled = np.empty((len(names), len(names)), dtype=float)
    for i, derivative_i in enumerate(whitened_derivatives):
        for j in range(i, len(names)):
            value = 0.5 * float(np.sum(derivative_i * whitened_derivatives[j]))
            fisher_scaled[i, j] = value
            fisher_scaled[j, i] = value

    priors = _validate_parameter_priors(parameter_priors)
    for index, name in enumerate(names):
        if name in priors:
            fisher_scaled[index, index] += (scales[index] / priors[name]) ** 2

    fisher_scaled = 0.5 * (fisher_scaled + fisher_scaled.T)
    eigenvalues = np.linalg.eigvalsh(fisher_scaled)
    largest = float(np.max(eigenvalues))
    tolerance = rank_rtol * largest if largest > 0 else np.inf
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    if rank != len(names) or float(np.min(eigenvalues)) <= 0:
        raise np.linalg.LinAlgError(
            "The nuisance-marginalized PV Fisher matrix is rank deficient "
            f"(rank={rank}/{len(names)}). The requested parameters {list(names)} "
            "are not independently represented by this sample. Supply a "
            "scientifically justified explicit prior or change the sample; a "
            "conditional fs8 error will not be reported."
        )
    condition_number = largest / float(np.min(eigenvalues))
    if condition_number > max_condition_number:
        raise np.linalg.LinAlgError(
            "The nuisance-marginalized PV Fisher matrix is too ill-conditioned "
            f"after parameter scaling (condition={condition_number:.3e}, "
            f"limit={max_condition_number:.3e}). No conditional fallback is used."
        )

    fisher_cholesky = cho_factor(fisher_scaled, lower=True, check_finite=False)
    covariance_scaled = cho_solve(
        fisher_cholesky,
        np.eye(len(names)),
        check_finite=False,
    )
    covariance_parameters = covariance_scaled * np.outer(scales, scales)
    fisher_parameters = fisher_scaled / np.outer(scales, scales)
    conditional_sigma = scales / np.sqrt(np.diag(fisher_scaled))

    return {
        "parameter_names": names,
        "fisher_matrix": fisher_parameters,
        "parameter_covariance": covariance_parameters,
        "conditional_sigma": conditional_sigma,
        "parameter_scales": dict(zip(names, scales)),
        "parameter_priors": priors,
        "rank": rank,
        "condition_number_scaled": condition_number,
    }


def _extract_carreres23_signal_covariance(
    covariance: object, sample_size: int
) -> np.ndarray:
    """Extract the unit-amplitude Carreres23 velocity signal covariance."""

    if getattr(covariance, "model_name", None) != "carreres23":
        raise RuntimeError("Expected a FLIP carreres23 covariance object.")
    if getattr(covariance, "model_kind", None) != "velocity":
        raise RuntimeError("Expected a velocity-only FLIP covariance object.")
    covariance_dict = getattr(covariance, "covariance_dict", None)
    if not isinstance(covariance_dict, Mapping) or "vv" not in covariance_dict:
        raise RuntimeError("FLIP covariance does not contain a 'vv' signal component.")
    components = np.asarray(covariance_dict["vv"], dtype=float)
    if components.ndim == 2:
        signal = components
    elif components.ndim == 3 and components.shape[0] == 1:
        signal = components[0]
    else:
        raise RuntimeError(
            "Carreres23 must provide exactly one full velocity covariance "
            f"component; received shape {components.shape}."
        )
    if signal.shape != (sample_size, sample_size):
        raise RuntimeError(
            f"FLIP signal covariance has shape {signal.shape}; expected "
            f"({sample_size}, {sample_size})."
        )
    if not np.all(np.isfinite(signal)):
        raise RuntimeError("FLIP signal covariance contains non-finite values.")
    return 0.5 * (signal + signal.T)


def _build_pv_covariance_and_derivatives(
    data_fisher: object,
    covariance: object,
    parameter_values: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Build the fiducial PV covariance and all represented derivatives."""

    legacy_values = {
        "fs8": float(parameter_values["fs8"]),
        "sigv": float(parameter_values["sigv"]),
        "sigma_M": 0.0,
    }
    try:
        variance_base = np.asarray(
            data_fisher.give_data_and_variance(legacy_values), dtype=float
        )
        variance_plus_one_mag2 = np.asarray(
            data_fisher.give_data_and_variance({**legacy_values, "sigma_M": 1.0}),
            dtype=float,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "FisherVelFromHDres did not expose the expected distance-modulus "
            "variance transformation."
        ) from exc
    if variance_base.ndim != 1 or variance_plus_one_mag2.shape != variance_base.shape:
        raise RuntimeError(
            "Expected one-dimensional per-object velocity variances from FLIP."
        )
    if not np.all(np.isfinite(variance_base)) or np.any(variance_base < 0):
        raise RuntimeError("The baseline FLIP velocity variance is invalid.")

    conversion_squared = variance_plus_one_mag2 - variance_base
    if not np.all(np.isfinite(conversion_squared)) or np.any(conversion_squared <= 0):
        raise RuntimeError(
            "The sigma_M2 covariance derivative is absent or non-positive in "
            "FisherVelFromHDres; nuisance marginalization cannot proceed."
        )

    sample_size = variance_base.size
    signal = _extract_carreres23_signal_covariance(covariance, sample_size)
    identity = np.eye(sample_size)
    fs8 = float(parameter_values["fs8"])
    sigv = float(parameter_values["sigv"])
    sigma_m2 = float(parameter_values["sigma_M2"])
    total_covariance = (
        fs8**2 * signal
        + sigv**2 * identity
        + np.diag(variance_base + sigma_m2 * conversion_squared)
    )
    derivatives = {
        "fs8": 2.0 * fs8 * signal,
        "sigv": 2.0 * sigv * identity,
        "sigma_M2": np.diag(conversion_squared),
    }
    if tuple(derivatives) != PV_FISHER_PARAMETERS:
        raise RuntimeError(
            "Internal error: not all requested PV Fisher derivatives were constructed."
        )

    inferred_mag_variance = np.divide(
        variance_base,
        conversion_squared,
        out=np.full_like(variance_base, np.nan),
        where=conversion_squared > 0,
    )
    finite_mag_variance = inferred_mag_variance[
        np.isfinite(inferred_mag_variance) & (inferred_mag_variance > 0)
    ]
    sigma_m2_scale = (
        float(np.median(finite_mag_variance)) if finite_mag_variance.size else 0.01
    )
    parameter_scales = {
        "fs8": max(abs(fs8), 1.0e-3),
        "sigv": max(abs(sigv), 1.0),
        "sigma_M2": max(sigma_m2_scale, 1.0e-6),
    }
    return total_covariance, derivatives, parameter_scales


def _power_spectrum_metadata(
    power_spectrum_dict: Mapping[str, object]
) -> dict[str, object]:
    """Return auditable k-grid and damping metadata for a FLIP spectrum."""

    try:
        k_values = np.asarray(power_spectrum_dict["vv"][0][0], dtype=float)  # type: ignore[index]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "power_spectrum_dict must contain {'vv': [[k, Pvv]]}."
        ) from exc
    if k_values.ndim != 1 or k_values.size < 2 or not np.all(np.isfinite(k_values)):
        raise ValueError("The velocity power-spectrum k grid is invalid.")
    metadata = dict(power_spectrum_dict.get("_metadata", {}))  # type: ignore[arg-type]
    metadata.setdefault("k_min_hmpc", float(np.min(k_values)))
    metadata.setdefault("k_max_hmpc", float(np.max(k_values)))
    metadata.setdefault("n_k", int(k_values.size))
    return metadata


def build_velocity_power_spectrum(
    *,
    kmin: float = 1e-5,
    kmax: float = 1.0,
    sigmau_fiducial: float = 21.0,
) -> dict:
    """Build the damped FLIP velocity spectrum through ``kmax=1 h/Mpc``."""

    import flip

    kh, _, _, ptt, _ = flip.power_spectra.compute_power_spectra(
        "class_engine",
        COSMO_DICT,
        0,
        kmin,
        kmax,
        1500,
        normalization_power_spectrum="growth_rate",
    )
    return {
        "vv": [[kh, ptt * flip.utils.Du(kh, float(sigmau_fiducial)) ** 2]],
        "_metadata": {
            "k_min_hmpc": float(kmin),
            "k_max_hmpc": float(kmax),
            "n_k": int(len(kh)),
            "sigmau_hinv_mpc": float(sigmau_fiducial),
            "normalization": "growth_rate",
        },
    }


def compute_fs8_sigma_by_ranges(
    flow_df: FlowInput,
    z_ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    power_spectrum_dict: dict,
    *,
    size_batch: int = 10_000,
    number_worker: int = 2,
    fisher_properties: Mapping[str, object] | None = None,
    parameter_dict: Mapping[str, float] | None = None,
    parameter_priors: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Return nuisance-marginalized ``sigma(fs8)`` by redshift range.

    FLIP supplies the Carreres23 unit-amplitude signal covariance, but this
    function deliberately does not use FLIP's Fisher derivative dispatcher:
    the pinned version only exposes the ``fs8`` derivative.  Instead, the full
    covariance and three derivatives are constructed explicitly for
    ``fs8``, the uncorrelated velocity dispersion ``sigv`` (km/s), and an
    added magnitude variance ``sigma_M2`` (mag²).  The reported uncertainty is
    obtained from the inverse of the full 3x3 Fisher matrix.

    The baseline ``dmu_error`` already contains the adopted color-dependent
    intrinsic scatter.  Therefore the default ``sigma_M2=0`` adds no second
    intrinsic-scatter term.  ``sigma_M2`` is nevertheless marginalized because
    its covariance derivative is well-defined at zero.  No priors are applied
    unless ``parameter_priors`` is explicitly supplied; all priors and
    conditioning diagnostics are recorded in the returned table.
    """

    import flip

    numerical = _validate_fisher_properties(fisher_properties)
    parameter_values = _validate_pv_parameter_values(parameter_dict)
    priors = _validate_parameter_priors(parameter_priors)
    spectrum_metadata = _power_spectrum_metadata(power_spectrum_dict)

    common_metadata: dict[str, object] = {
        "fisher_method": "gaussian_covariance_trace_cholesky",
        "fisher_parameters": json.dumps(list(PV_FISHER_PARAMETERS)),
        "marginalized_parameters": json.dumps(list(PV_FISHER_PARAMETERS[1:])),
        "is_nuisance_marginalized": True,
        "fisher_dimension": len(PV_FISHER_PARAMETERS),
        "fs8_fiducial": parameter_values["fs8"],
        "sigv_fiducial_kms": parameter_values["sigv"],
        "sigma_M2_fiducial_mag2": parameter_values["sigma_M2"],
        "parameter_priors_json": json.dumps(priors, sort_keys=True),
        "has_parameter_priors": bool(priors),
        "prior_sigma_fs8": priors.get("fs8", np.nan),
        "prior_sigma_sigv_kms": priors.get("sigv", np.nan),
        "prior_sigma_sigma_M2_mag2": priors.get("sigma_M2", np.nan),
        "max_condition_number": numerical["max_condition_number"],
        "rank_rtol": numerical["rank_rtol"],
        "k_min_hmpc": spectrum_metadata.get("k_min_hmpc", np.nan),
        "k_max_hmpc": spectrum_metadata.get("k_max_hmpc", np.nan),
        "n_k": spectrum_metadata.get("n_k", np.nan),
        "sigmau_hinv_mpc": spectrum_metadata.get("sigmau_hinv_mpc", np.nan),
        "intrinsic_scatter_handling": (
            "embedded_in_dmu_error; sigma_M2 is zero-fiducial added variance"
        ),
    }

    prepared_cache: dict[int, pd.DataFrame] = {}
    per_range_frames = (
        {
            (float(z_min), float(z_max)): frame
            for (z_min, z_max), frame in flow_df.items()
        }
        if isinstance(flow_df, Mapping)
        else None
    )
    out: list[dict[str, object]] = []
    for zmin, zmax in z_ranges:
        if per_range_frames is None:
            df_source = flow_df
        else:
            key = (float(zmin), float(zmax))
            if key not in per_range_frames:
                raise KeyError(f"Missing flow sample for z-range {key}.")
            df_source = per_range_frames[key]

        cache_key = id(df_source)
        if cache_key not in prepared_cache:
            df_all = df_source.copy()
            if "z" not in df_all.columns:
                df_all["z"] = df_all["zobs"]
            df_all["dmu_error_base"] = sigma_mu_per_sn(df_all, sigma_vpec_kms=0).astype(
                float
            )
            prepared_cache[cache_key] = df_all
        df_all = prepared_cache[cache_key]
        zobs_all = pd.to_numeric(df_all["zobs"], errors="coerce")
        df_sub = df_all.loc[(zobs_all >= zmin) & (zobs_all < zmax)].copy()
        if df_sub.empty:
            out.append(
                {
                    "z_min": float(zmin),
                    "z_max": float(zmax),
                    "zeff": np.nan,
                    "sigma_fs8": np.nan,
                    "sigma_fs8_conditional": np.nan,
                    "marginalization_degradation": np.nan,
                    "N": 0,
                    "fisher_rank": 0,
                    "fisher_condition_number_scaled": np.nan,
                    "parameter_scales_json": "{}",
                    **common_metadata,
                }
            )
            continue

        sig = pd.to_numeric(df_sub["dmu_error_base"], errors="coerce").to_numpy(float)
        zobs = pd.to_numeric(df_sub["zobs"], errors="coerce").to_numpy(float)
        required_columns = ("ra", "dec", "rcom_zobs", "hubble_norm")
        missing_columns = [
            column for column in required_columns if column not in df_sub
        ]
        if missing_columns:
            raise KeyError(
                f"PV sample for {zmin} <= z < {zmax} is missing {missing_columns}."
            )
        finite_coordinates = np.ones(len(df_sub), dtype=bool)
        for column in required_columns:
            finite_coordinates &= np.isfinite(
                pd.to_numeric(df_sub[column], errors="coerce").to_numpy(float)
            )
        good = np.isfinite(sig) & np.isfinite(zobs) & (sig > 0) & finite_coordinates
        if not np.all(good):
            raise ValueError(
                f"PV sample for {zmin} <= z < {zmax} contains "
                f"{int(np.count_nonzero(~good))} row(s) with invalid coordinates, "
                "redshift, or distance error."
            )
        weight = 1.0 / sig**2
        zeff = float(np.sum(weight * zobs) / np.sum(weight))

        df_sub["dmu_error"] = df_sub["dmu_error_base"]
        data_fisher = flip.data_vector.FisherVelFromHDres(df_sub.to_dict(orient="list"))
        covariance = data_fisher.compute_covariance(
            "carreres23",
            power_spectrum_dict,
            size_batch=size_batch,
            number_worker=number_worker,
        )
        total_covariance, derivatives, parameter_scales = (
            _build_pv_covariance_and_derivatives(
                data_fisher,
                covariance,
                parameter_values,
            )
        )
        try:
            fisher_result = _gaussian_covariance_trace_fisher(
                total_covariance,
                derivatives,
                parameter_scales=parameter_scales,
                parameter_priors=priors,
                max_condition_number=numerical["max_condition_number"],
                rank_rtol=numerical["rank_rtol"],
            )
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"PV Fisher failed for {zmin} <= z < {zmax}: {exc}"
            ) from exc

        parameter_names = tuple(fisher_result["parameter_names"])
        if parameter_names != PV_FISHER_PARAMETERS:
            raise RuntimeError(
                "PV Fisher returned parameters "
                f"{list(parameter_names)}, expected {list(PV_FISHER_PARAMETERS)}."
            )
        idx_fs8 = parameter_names.index("fs8")
        parameter_covariance = np.asarray(
            fisher_result["parameter_covariance"], dtype=float
        )
        conditional_sigma = np.asarray(fisher_result["conditional_sigma"], dtype=float)
        sigma_fs8 = float(np.sqrt(parameter_covariance[idx_fs8, idx_fs8]))
        sigma_fs8_conditional = float(conditional_sigma[idx_fs8])
        out.append(
            {
                "z_min": float(zmin),
                "z_max": float(zmax),
                "zeff": zeff,
                "sigma_fs8": sigma_fs8,
                "sigma_fs8_conditional": sigma_fs8_conditional,
                "marginalization_degradation": sigma_fs8 / sigma_fs8_conditional,
                "N": int(len(df_sub)),
                "fisher_rank": int(fisher_result["rank"]),
                "fisher_condition_number_scaled": float(
                    fisher_result["condition_number_scaled"]
                ),
                "parameter_scales_json": json.dumps(
                    fisher_result["parameter_scales"], sort_keys=True
                ),
                **common_metadata,
            }
        )

    return pd.DataFrame(out)


def compute_fs8_sigma_by_ranges_paper(
    flow_df: FlowInput,
    z_ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    power_spectrum_dict: dict,
    *,
    size_batch: int = 10_000,
    number_worker: int = 2,
    sigv_kms: float = 300.0,
) -> pd.DataFrame:
    """Reproduce the Fisher definition used in the v7 paper notebook.

    This intentionally follows ``twilight_cosmology_v7_fsigma.ipynb``:
    FLIP constructs the Fisher matrix for ``fs8``, ``sigv``, and ``sigma_M``
    at ``sigma_M=0`` and the notebook applies a pseudoinverse.  The historical
    power-spectrum cutoff is supplied by the caller (normally kmax=0.2 h/Mpc).
    The method is retained as an explicit compatibility mode; the stricter
    ``compute_fs8_sigma_by_ranges`` routine uses an identifiable added-variance
    nuisance and records a fully ranked three-parameter marginalization.
    """

    import flip

    spectrum_metadata = _power_spectrum_metadata(power_spectrum_dict)
    parameter_dict = {"fs8": 1.0, "sigv": float(sigv_kms), "sigma_M": 0.0}
    prepared_cache: dict[int, pd.DataFrame] = {}
    per_range_frames = (
        {
            (float(z_min), float(z_max)): frame
            for (z_min, z_max), frame in flow_df.items()
        }
        if isinstance(flow_df, Mapping)
        else None
    )
    out: list[dict[str, object]] = []
    for zmin, zmax in z_ranges:
        if per_range_frames is None:
            df_source = flow_df
        else:
            key = (float(zmin), float(zmax))
            if key not in per_range_frames:
                raise KeyError(f"Missing flow sample for z-range {key}.")
            df_source = per_range_frames[key]

        cache_key = id(df_source)
        if cache_key not in prepared_cache:
            df_all = df_source.copy()
            if "z" not in df_all.columns:
                df_all["z"] = df_all["zobs"]
            df_all["dmu_error_base"] = sigma_mu_per_sn(
                df_all, sigma_vpec_kms=0
            ).astype(float)
            prepared_cache[cache_key] = df_all
        df_all = prepared_cache[cache_key]
        zobs_all = pd.to_numeric(df_all["zobs"], errors="coerce")
        df_sub = df_all.loc[(zobs_all >= zmin) & (zobs_all < zmax)].copy()
        if df_sub.empty:
            out.append(
                {
                    "z_min": float(zmin),
                    "z_max": float(zmax),
                    "zeff": np.nan,
                    "sigma_fs8": np.nan,
                    "N": 0,
                    "fisher_rank": 0,
                    "fisher_condition_number": np.nan,
                    "fisher_method": "v7_notebook_flip_pseudoinverse",
                    "fisher_parameters": "[]",
                    "k_min_hmpc": spectrum_metadata.get("k_min_hmpc", np.nan),
                    "k_max_hmpc": spectrum_metadata.get("k_max_hmpc", np.nan),
                    "n_k": spectrum_metadata.get("n_k", np.nan),
                    "sigmau_hinv_mpc": spectrum_metadata.get(
                        "sigmau_hinv_mpc", np.nan
                    ),
                }
            )
            continue

        sig = pd.to_numeric(df_sub["dmu_error_base"], errors="coerce").to_numpy(float)
        zobs = pd.to_numeric(df_sub["zobs"], errors="coerce").to_numpy(float)
        required_columns = ("ra", "dec", "rcom_zobs", "hubble_norm")
        good = np.isfinite(sig) & np.isfinite(zobs) & (sig > 0)
        for column in required_columns:
            if column not in df_sub:
                raise KeyError(
                    f"PV sample for {zmin} <= z < {zmax} is missing {column}."
                )
            good &= np.isfinite(
                pd.to_numeric(df_sub[column], errors="coerce").to_numpy(float)
            )
        if not np.all(good):
            raise ValueError(
                f"PV sample for {zmin} <= z < {zmax} contains "
                f"{int(np.count_nonzero(~good))} invalid row(s)."
            )

        weight = 1.0 / sig**2
        zeff = float(np.sum(weight * zobs) / np.sum(weight))
        df_sub["dmu_error"] = df_sub["dmu_error_base"]
        data_fisher = flip.data_vector.FisherVelFromHDres(
            df_sub.to_dict(orient="list")
        )
        covariance = data_fisher.compute_covariance(
            "carreres23",
            power_spectrum_dict,
            size_batch=size_batch,
            number_worker=number_worker,
        )
        fisher = flip.fisher.FisherMatrix.init_from_covariance(
            covariance,
            data_fisher,
            parameter_dict,
            fisher_properties={"inversion_method": "inverse"},
        )
        parameter_names, fisher_matrix = fisher.compute_fisher_matrix()
        fisher_matrix = np.asarray(fisher_matrix, dtype=float)
        idx_fs8 = list(parameter_names).index("fs8")
        parameter_covariance = np.linalg.pinv(fisher_matrix)
        sigma_fs8 = float(np.sqrt(parameter_covariance[idx_fs8, idx_fs8]))
        condition_number = float(np.linalg.cond(fisher_matrix))
        out.append(
            {
                "z_min": float(zmin),
                "z_max": float(zmax),
                "zeff": zeff,
                "sigma_fs8": sigma_fs8,
                "N": int(len(df_sub)),
                "fisher_rank": int(np.linalg.matrix_rank(fisher_matrix)),
                "fisher_condition_number": condition_number,
                "fisher_method": "v7_notebook_flip_pseudoinverse",
                "fisher_parameters": json.dumps(list(parameter_names)),
                "k_min_hmpc": spectrum_metadata.get("k_min_hmpc", np.nan),
                "k_max_hmpc": spectrum_metadata.get("k_max_hmpc", np.nan),
                "n_k": spectrum_metadata.get("n_k", np.nan),
                "sigmau_hinv_mpc": spectrum_metadata.get(
                    "sigmau_hinv_mpc", np.nan
                ),
            }
        )

    return pd.DataFrame(out)


def run_sigv_scan(
    flow_samples: dict[str, FlowInput],
    *,
    z_ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...] = DEFAULT_Z_RANGES,
    sigv_values: list[float] | tuple[float, ...] = (150, 200, 250, 300, 350, 400, 500),
    sigmau_fiducial: float = 21.0,
    size_batch: int = 10_000,
    number_worker: int = 2,
) -> pd.DataFrame:
    """Scan the fiducial ``sigv`` while marginalizing all PV nuisances."""

    power_spectrum_dict = build_velocity_power_spectrum(
        sigmau_fiducial=sigmau_fiducial
    )
    frames: list[pd.DataFrame] = []
    for sigv in sigv_values:
        parameter_dict = {"fs8": 1.0, "sigv": float(sigv), "sigma_M2": 0.0}
        for scenario, flow_df in flow_samples.items():
            frame = compute_fs8_sigma_by_ranges(
                flow_df,
                z_ranges,
                power_spectrum_dict,
                size_batch=size_batch,
                number_worker=number_worker,
                parameter_dict=parameter_dict,
            ).copy()
            frame["scenario"] = scenario
            frame["sigv_kms"] = float(sigv)
            frame["sigmau_fiducial"] = float(sigmau_fiducial)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_sigmau_scan(
    flow_samples: dict[str, FlowInput],
    *,
    z_ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...] = DEFAULT_Z_RANGES,
    sigmau_values: list[float] | tuple[float, ...] = (15, 21, 30),
    sigv_kms: float = 300.0,
    size_batch: int = 10_000,
    number_worker: int = 2,
) -> pd.DataFrame:
    """Scan the velocity-damping scale used in the FLIP power spectrum."""

    frames: list[pd.DataFrame] = []
    parameter_dict = {"fs8": 1.0, "sigv": float(sigv_kms), "sigma_M2": 0.0}
    for sigmau in sigmau_values:
        power_spectrum_dict = build_velocity_power_spectrum(
            sigmau_fiducial=float(sigmau)
        )
        for scenario, flow_df in flow_samples.items():
            frame = compute_fs8_sigma_by_ranges(
                flow_df,
                z_ranges,
                power_spectrum_dict,
                size_batch=size_batch,
                number_worker=number_worker,
                parameter_dict=parameter_dict,
            ).copy()
            frame["scenario"] = scenario
            frame["sigmau_fiducial"] = float(sigmau)
            frame["sigv_kms"] = float(sigv_kms)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_scan(
    scan_df: pd.DataFrame,
    *,
    parameter_col: str,
    baseline_value: float,
) -> pd.DataFrame:
    """Pivot scan results into WFD vs WFD+Twilight summaries."""

    base = scan_df[["scenario", parameter_col, "z_min", "z_max", "zeff", "sigma_fs8", "N"]]
    pivot = (
        base.pivot_table(
            index=[parameter_col, "z_min", "z_max"],
            columns="scenario",
            values=["zeff", "sigma_fs8", "N"],
            aggfunc="first",
        )
        .sort_index()
        .reset_index()
    )
    pivot.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in pivot.columns
    ]
    renamed = pivot.rename(
        columns={
            "sigma_fs8_WFD Y1": "sigma_fs8_wfd",
            "sigma_fs8_WFD+Twilight Y1": "sigma_fs8_wfd_twilight",
            "zeff_WFD Y1": "zeff_wfd",
            "zeff_WFD+Twilight Y1": "zeff_wfd_twilight",
            "N_WFD Y1": "N_wfd",
            "N_WFD+Twilight Y1": "N_wfd_twilight",
        }
    )
    renamed["precision_gain"] = (
        renamed["sigma_fs8_wfd"] / renamed["sigma_fs8_wfd_twilight"]
    )
    renamed["improvement_pct"] = 100.0 * (
        1.0 - renamed["sigma_fs8_wfd_twilight"] / renamed["sigma_fs8_wfd"]
    )

    baseline = renamed.loc[np.isclose(renamed[parameter_col], float(baseline_value))].copy()
    baseline = baseline.rename(
        columns={
            "sigma_fs8_wfd": "sigma_fs8_wfd_baseline",
            "sigma_fs8_wfd_twilight": "sigma_fs8_wfd_twilight_baseline",
            "precision_gain": "precision_gain_baseline",
        }
    )[
        [
            "z_min",
            "z_max",
            "sigma_fs8_wfd_baseline",
            "sigma_fs8_wfd_twilight_baseline",
            "precision_gain_baseline",
        ]
    ]
    merged = renamed.merge(baseline, on=["z_min", "z_max"], how="left")
    merged["delta_wfd_pct_vs_baseline"] = 100.0 * (
        merged["sigma_fs8_wfd"] / merged["sigma_fs8_wfd_baseline"] - 1.0
    )
    merged["delta_wfd_twilight_pct_vs_baseline"] = 100.0 * (
        merged["sigma_fs8_wfd_twilight"] / merged["sigma_fs8_wfd_twilight_baseline"]
        - 1.0
    )
    merged["delta_precision_gain_pct_vs_baseline"] = 100.0 * (
        merged["precision_gain"] / merged["precision_gain_baseline"] - 1.0
    )
    return merged


def lowz_robustness_summary(
    sigv_summary: pd.DataFrame,
    sigmau_summary: pd.DataFrame,
) -> str:
    """Return a short markdown summary for the low-z PV robustness scan."""

    lowz_sigv = sigv_summary.loc[
        np.isclose(sigv_summary["z_min"], 0.02)
        & np.isclose(sigv_summary["z_max"], 0.06)
    ].copy()
    lowz_sigmau = sigmau_summary.loc[
        np.isclose(sigmau_summary["z_min"], 0.02)
        & np.isclose(sigmau_summary["z_max"], 0.06)
    ].copy()
    if lowz_sigv.empty or lowz_sigmau.empty:
        raise ValueError("Low-z bin (0.02, 0.06) is missing from the scan summary.")

    sigv_gain_min = lowz_sigv["precision_gain"].min()
    sigv_gain_max = lowz_sigv["precision_gain"].max()
    sigv_gain_shift = lowz_sigv["delta_precision_gain_pct_vs_baseline"].abs().max()
    sigv_wfd_shift = lowz_sigv["delta_wfd_pct_vs_baseline"].abs().max()
    sigv_tw_shift = lowz_sigv["delta_wfd_twilight_pct_vs_baseline"].abs().max()

    sigmau_gain_min = lowz_sigmau["precision_gain"].min()
    sigmau_gain_max = lowz_sigmau["precision_gain"].max()
    sigmau_gain_shift = lowz_sigmau["delta_precision_gain_pct_vs_baseline"].abs().max()
    sigmau_wfd_shift = lowz_sigmau["delta_wfd_pct_vs_baseline"].abs().max()
    sigmau_tw_shift = lowz_sigmau["delta_wfd_twilight_pct_vs_baseline"].abs().max()

    cumulative_sigv = sigv_summary.loc[
        np.isclose(sigv_summary["z_min"], 0.02)
        & np.isclose(sigv_summary["z_max"], 0.14)
    ].copy()
    cumulative_gain_min = cumulative_sigv["precision_gain"].min()
    cumulative_gain_max = cumulative_sigv["precision_gain"].max()

    return (
        "We tested the PV forecast against the adopted 300 km s$^{-1}$ scale by "
        "holding the per-SN distance-error model fixed and sweeping the fiducial "
        "uncorrelated velocity dispersion over $150\\leq\\sigma_v\\leq500$ "
        "km s$^{-1}$. At every grid point, the covariance Fisher is inverted "
        "after marginalizing over both $\\sigma_v$ and a zero-fiducial added "
        "magnitude-variance nuisance, with no external priors. "
        "In the key $0.02\\leq z<0.06$ bin, the absolute one-year forecasts move by "
        f"only {sigv_wfd_shift:.1f}\\% (WFD) and {sigv_tw_shift:.1f}\\% "
        f"(WFD+Twilight) around the $\\sigma_v=300$ km s$^{{-1}}$ baseline, while "
        f"the Twilight precision gain stays between $\\times{sigv_gain_min:.2f}$ and "
        f"$\\times{sigv_gain_max:.2f}$ (within {sigv_gain_shift:.1f}\\%). "
        "A separate damping scan with $15\\leq\\sigma_u\\leq30\\ h^{-1}\\,{\\rm Mpc}$ "
        f"at fixed $\\sigma_v=300$ km s$^{{-1}}$ gives a low-z gain range of "
        f"$\\times{sigmau_gain_min:.2f}$--$\\times{sigmau_gain_max:.2f}$; the "
        f"corresponding absolute forecasts shift by {sigmau_wfd_shift:.1f}\\% "
        f"(WFD) and {sigmau_tw_shift:.1f}\\% (WFD+Twilight), and the gain changes "
        f"by at most {sigmau_gain_shift:.1f}\\%. The cumulative "
        f"$0.02\\leq z<0.14$ gain remains $\\times{cumulative_gain_min:.2f}$--"
        f"$\\times{cumulative_gain_max:.2f}$. "
        "The main conclusion that Twilight substantially improves the one-year "
        "$f\\sigma_8$ precision therefore does not depend sensitively on the "
        "fiducial 300 km s$^{-1}$ choice."
    )


def plot_sigv_scan(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a line plot of σ(fs8) vs sigv for WFD and WFD+Twilight."""

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    color_map = {
        (0.02, 0.06): "#1f77b4",
        (0.06, 0.10): "#2ca02c",
        (0.02, 0.14): "#ff7f0e",
    }
    for (z_min, z_max), group in summary_df.groupby(["z_min", "z_max"], sort=True):
        group = group.sort_values("sigv_kms")
        color = color_map.get((float(z_min), float(z_max)), "#444444")
        label = f"{z_min:.2f}≤z<{z_max:.2f}"
        ax.plot(
            group["sigv_kms"],
            group["sigma_fs8_wfd"],
            linestyle="--",
            color=color,
            marker="o",
            label=f"WFD, {label}",
        )
        ax.plot(
            group["sigv_kms"],
            group["sigma_fs8_wfd_twilight"],
            linestyle="-",
            color=color,
            marker="o",
            label=f"WFD+Twilight, {label}",
        )
    ax.set_xlabel(r"$\sigma_v$ [km s$^{-1}$]")
    ax.set_ylabel(r"$\sigma(f\sigma_8)$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_sigv_gain(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a line plot of WFD/WFD+Twilight precision gain vs sigv."""

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=180)
    color_map = {
        (0.02, 0.06): "#1f77b4",
        (0.06, 0.10): "#2ca02c",
        (0.02, 0.14): "#ff7f0e",
    }
    for (z_min, z_max), group in summary_df.groupby(["z_min", "z_max"], sort=True):
        group = group.sort_values("sigv_kms")
        color = color_map.get((float(z_min), float(z_max)), "#444444")
        ax.plot(
            group["sigv_kms"],
            group["precision_gain"],
            color=color,
            marker="o",
            label=f"{z_min:.2f}≤z<{z_max:.2f}",
        )
    ax.axhline(1.0, color="k", linewidth=1.0, linestyle=":")
    ax.set_xlabel(r"$\sigma_v$ [km s$^{-1}$]")
    ax.set_ylabel("WFD / (WFD+Twilight) precision gain")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_v7_pv_samples(
    *,
    include_multiyear: bool = False,
    wfd_fitres_path: str | Path | None = None,
    wfd_twboost_fitres_path: str | Path | None = None,
    twilight_fitres_path: str | Path | None = None,
    wfd_simlib_path: str | Path | None = None,
    wfd_twboost_simlib_path: str | Path | None = None,
    twilight_simlib_path: str | Path | None = None,
) -> dict[str, FlowInput]:
    """Rebuild the PV samples used in the v7 fsigma8 forecast notebook.

    The optional FITRES and SIMLIB paths allow a corrected one-year rerun to
    remain separate from the historical products.  Omitting every override
    preserves the original v7 path discovery and file choices.
    """

    root = repo_root()
    sim3_legacy_base = root / "data" / "simlib_code_from_eplsst" / "3_sim_data_legacy"
    sim3_lcfit_base = root / "data" / "simlib_code_from_eplsst" / "_3sim_data" / "2_LCFIT"
    if wfd_fitres_path is None:
        wfd_fitres_path = first_existing_path(
            sim3_lcfit_base
            / "FIT_WFDORG_WFDORG23"
            / "output"
            / "PIP_TZ_3SIM_TWI_WFDORG23"
            / "FITOPT000.FITRES.gz",
            sim3_legacy_base / "PIP_TZ_3SIM_TWI_WFDORG23" / "FITOPT000.FITRES",
        )
    else:
        wfd_fitres_path = Path(wfd_fitres_path)
    if wfd_twboost_fitres_path is None:
        wfd_twboost_fitres_path = first_existing_path(
            sim3_lcfit_base
            / "FIT_WFDADDTW_WFDADDTW23"
            / "output"
            / "PIP_TZ_3SIM_TWI_WFDADDTW23"
            / "FITOPT000.FITRES.gz",
            sim3_legacy_base / "PIP_TZ_3SIM_TWI_WFDADDTW23" / "FITOPT000.FITRES",
        )
    else:
        wfd_twboost_fitres_path = Path(wfd_twboost_fitres_path)
    if twilight_fitres_path is None:
        twilight_fitres_path = first_existing_path(
            sim3_lcfit_base
            / "FIT_TWI_TWI23"
            / "output"
            / "PIP_TZ_3SIM_TWI_TWI23"
            / "FITOPT000.FITRES.gz",
            sim3_legacy_base / "PIP_TZ_3SIM_TWI_TWI23" / "FITOPT000.FITRES",
        )
    else:
        twilight_fitres_path = Path(twilight_fitres_path)

    if wfd_simlib_path is None:
        wfd_simlib_path = first_existing_path(
            root
            / "data"
            / "simlib_code_from_eplsst"
            / "pipeline_out_simlib"
            / "WFD_baseline_1Y_first_win2022-12-01_to_2023-12-01_start2021-11-01.SIMLIB",
            root
            / "data"
            / "simlib_code_from_eplsst"
            / "wfd"
            / "data_use"
            / "WFD_1Y_first_win2022-12-01_to_2023-12-01_start2021-11-01_WITH_SALT2XC.SIMLIB",
        )
    else:
        wfd_simlib_path = Path(wfd_simlib_path)
    if wfd_twboost_simlib_path is None:
        wfd_twboost_simlib_path = wfd_simlib_path
    else:
        wfd_twboost_simlib_path = Path(wfd_twboost_simlib_path)
    if twilight_simlib_path is None:
        twilight_simlib_path = first_existing_path(
            root
            / "data"
            / "simlib_code_from_eplsst"
            / "pipeline_out_simlib"
            / "TWILIGHT_only_1Y_first_win2022-12-01_to_2023-12-01_start2021-11-01.SIMLIB",
            root
            / "data"
            / "simlib_code_from_eplsst"
            / "wfd"
            / "data_use"
            / "TWILIGHT_1Y_first_win2022-12-01_to_2023-12-01_start2021-11-01.SIMLIB",
        )
    else:
        twilight_simlib_path = Path(twilight_simlib_path)
    ep_highz_fitres_path = (
        root
        / "Motivation"
        / "data_snana_EP_LSST"
        / "2_LCFIT"
        / "PIP_EP-LSST_LSST_P21_HIGHZ"
        / "FITOPT000.FITRES"
    )
    chris_wfd_fitres_path = (
        root / "Motivation" / "data_chris" / "WFD" / "FITOPT000.FITRES.gz"
    )

    fit_wfd_y1 = standardize_fitres(read_fitres_any(wfd_fitres_path))
    fit_wfd_y1["SRC"] = "WFDORG23"
    fit_wfd_y1["z"] = pd.to_numeric(fit_wfd_y1.get("z", np.nan), errors="coerce")

    fit_wfd_y1_twboost = standardize_fitres(
        read_fitres_any(wfd_twboost_fitres_path)
    )
    fit_wfd_y1_twboost["SRC"] = "WFDADDTW23"
    fit_wfd_y1_twboost["z"] = pd.to_numeric(
        fit_wfd_y1_twboost.get("z", np.nan), errors="coerce"
    )

    fit_qc_wfd = ross_qc_with_report(
        fit_wfd_y1,
        fitres_only=True,
        check_sampling=True,
        check_saturation=True,
        add_qc_columns=True,
        verbose=False,
        require_pm10_snr5=True,
    )
    fit_qc_wfd_twboost = ross_qc_with_report(
        fit_wfd_y1_twboost,
        fitres_only=True,
        check_sampling=True,
        check_saturation=True,
        add_qc_columns=True,
        verbose=False,
        require_pm10_snr5=True,
    )

    fit_qc_wfd_fixed, _ = fix_ra_dec_from_simlib(
        fit_qc_wfd,
        wfd_simlib_path,
        id_col="SIM_LIBID",
        ra_col="RA",
        dec_col="DEC",
        overwrite="all",
        require_all_mapped=True,
        redshift_col="z",
        require_redshift_match=True,
    )
    fit_qc_wfd_twboost_fixed, _ = fix_ra_dec_from_simlib(
        fit_qc_wfd_twboost,
        wfd_twboost_simlib_path,
        id_col="SIM_LIBID",
        ra_col="RA",
        dec_col="DEC",
        overwrite="all",
        require_all_mapped=True,
        redshift_col="z",
        require_redshift_match=True,
    )

    fit_ep_highz = standardize_fitres(read_fitres_any(ep_highz_fitres_path))
    fit_ep_highz["SRC"] = "EP_HIGHZ"
    fit_ep_highz["z"] = pd.to_numeric(fit_ep_highz.get("z", np.nan), errors="coerce")
    fit_ep_highz_tail = fit_ep_highz.loc[
        (fit_ep_highz["z"] >= 0.1) & (fit_ep_highz["z"] < 0.14)
    ].copy()
    if "RA" not in fit_ep_highz_tail.columns and "HOST_RA" in fit_ep_highz_tail.columns:
        fit_ep_highz_tail["RA"] = fit_ep_highz_tail["HOST_RA"]
    if "DEC" not in fit_ep_highz_tail.columns and "HOST_DEC" in fit_ep_highz_tail.columns:
        fit_ep_highz_tail["DEC"] = fit_ep_highz_tail["HOST_DEC"]
    fit_qc_ep_tail = ross_qc_with_report(
        fit_ep_highz_tail,
        fitres_only=True,
        check_sampling=True,
        check_saturation=True,
        add_qc_columns=True,
        verbose=False,
        require_pm10_snr5=True,
    )
    fit_qc_ep_tail["RA"] = pd.to_numeric(fit_qc_ep_tail["RA"], errors="coerce")
    fit_qc_ep_tail["DEC"] = pd.to_numeric(fit_qc_ep_tail["DEC"], errors="coerce")
    fit_qc_ep_tail = fit_qc_ep_tail.loc[
        fit_qc_ep_tail["RA"].between(0, 360)
        & fit_qc_ep_tail["DEC"].between(-90, 90)
    ].copy()

    tw_fitres = standardize_fitres(read_fitres_any(twilight_fitres_path))
    tw_fitres["SRC"] = "TWI23"
    tw_fitres["z"] = pd.to_numeric(tw_fitres.get("z", np.nan), errors="coerce")
    tw_fit_qc_all = ross_qc_with_report(
        tw_fitres,
        fitres_only=True,
        check_sampling=True,
        check_saturation=True,
        add_qc_columns=True,
        verbose=False,
        require_pm10_snr5=True,
    )
    tw_fit_qc_all_fixed, _ = fix_ra_dec_from_simlib(
        tw_fit_qc_all,
        twilight_simlib_path,
        id_col="SIM_LIBID",
        ra_col="RA",
        dec_col="DEC",
        overwrite="all",
        require_all_mapped=True,
        redshift_col="z",
        require_redshift_match=True,
        allow_unset_simlib_redshift=True,
    )

    z_split = 0.1
    z_max_comb = 0.14
    wfd_base_z = pd.to_numeric(fit_qc_wfd_fixed["z"], errors="coerce")
    wfd_twboost_z = pd.to_numeric(fit_qc_wfd_twboost_fixed["z"], errors="coerce")
    ep_tail_z = pd.to_numeric(fit_qc_ep_tail["z"], errors="coerce")
    fit_qc_wfd_with_highz = pd.concat(
        [
            fit_qc_wfd_fixed.loc[wfd_base_z < z_split],
            fit_qc_ep_tail.loc[(ep_tail_z >= z_split) & (ep_tail_z < z_max_comb)],
        ],
        ignore_index=True,
        sort=False,
    )
    wfd_part = pd.concat(
        [
            fit_qc_wfd_twboost_fixed.loc[wfd_twboost_z < z_split],
            fit_qc_ep_tail.loc[(ep_tail_z >= z_split) & (ep_tail_z < z_max_comb)],
        ],
        ignore_index=True,
        sort=False,
    )
    combined = pd.concat(
        [wfd_part, tw_fit_qc_all_fixed], ignore_index=True, sort=False
    )

    wfd_lowz_ready = make_flip_ready(fit_qc_wfd_with_highz)
    combined_ready = make_flip_ready(combined)

    out: dict[str, FlowInput] = {
        "WFD Y1": wfd_lowz_ready,
        "WFD+Twilight Y1": combined_ready,
    }
    if include_multiyear:
        fit_chris_wfd = read_fitres_lowz_streaming(
            chris_wfd_fitres_path,
            z_min=0.0,
            z_max=0.14,
        )
        fit_chris_wfd["SRC"] = "CHRIS_WFD"
        fit_chris_lowz = fit_chris_wfd.loc[
            (fit_chris_wfd["z"] > 0) & (fit_chris_wfd["z"] < 0.14)
        ].copy()
        multiyear_ready: dict[int, pd.DataFrame] = {}
        for years in (3, 5):
            t0, t1 = densest_k_year_window(
                fit_chris_lowz,
                time_col="PKMJD",
                k_years=float(years),
            )
            head = slice_by_window(
                fit_chris_lowz,
                t0,
                t1,
                time_col="PKMJD",
            )
            fit_qc = ross_qc_with_report(
                head,
                fitres_only=True,
                check_sampling=True,
                check_saturation=True,
                add_qc_columns=True,
                verbose=False,
                require_pm10_snr5=True,
            )
            multiyear_ready[years] = make_flip_ready(filter_valid_radec(fit_qc))
        wfd_y3_ready = multiyear_ready[3]
        wfd_y5_ready = multiyear_ready[5]
        out["WFD Y3"] = wfd_y3_ready
        out["WFD Y5"] = wfd_y5_ready
    return out


def compute_forecast_grid(
    flow_samples: Mapping[str, FlowInput],
    *,
    z_ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...] = DEFAULT_Z_RANGES,
    sigv_kms: float = 300.0,
    sigmau_fiducial: float = 21.0,
    size_batch: int = 10_000,
    number_worker: int = 2,
    forecast_definition: str = "marginalized",
) -> dict[str, pd.DataFrame]:
    """Compute forecasts using either the paper or stricter definition."""

    power_spectrum_dict = build_velocity_power_spectrum(
        kmax=0.2 if forecast_definition == "paper" else 1.0,
        sigmau_fiducial=sigmau_fiducial,
    )
    if forecast_definition == "paper":
        return {
            label: compute_fs8_sigma_by_ranges_paper(
                flow_df,
                z_ranges,
                power_spectrum_dict,
                size_batch=size_batch,
                number_worker=number_worker,
                sigv_kms=sigv_kms,
            )
            for label, flow_df in flow_samples.items()
        }
    if forecast_definition != "marginalized":
        raise ValueError(
            "`forecast_definition` must be either 'paper' or 'marginalized'."
        )
    parameter_dict = {"fs8": 1.0, "sigv": float(sigv_kms), "sigma_M2": 0.0}
    return {
        label: compute_fs8_sigma_by_ranges(
            flow_df,
            z_ranges,
            power_spectrum_dict,
            size_batch=size_batch,
            number_worker=number_worker,
            parameter_dict=parameter_dict,
        )
        for label, flow_df in flow_samples.items()
    }


def flatten_forecast_grid(results_by_label: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert forecast results into a single long-form table."""

    frames: list[pd.DataFrame] = []
    for label, frame in results_by_label.items():
        tmp = frame.copy()
        tmp["scenario"] = label
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(
            columns=["scenario", "z_min", "z_max", "zeff", "sigma_fs8", "N"]
        )
    return pd.concat(frames, ignore_index=True)


def sigma_cv_example(z: float) -> float:
    """Illustrative cosmic-variance floor used in the v7 notebook plot."""

    return 0.015 + 0.11 * np.exp(-float(z) / 0.03)


def plot_fs8_overview(
    results_by_label: Mapping[str, pd.DataFrame],
    output_path: str | Path,
    *,
    range_colors: list[str] | None = None,
    year_size_map: Mapping[str, float] | None = None,
    combine_marker: str = "D",
    cosmic_variance_fn=sigma_cv_example,
    title: str | None = None,
    y_max: float | None = None,
    band_base: tuple[float, float] = (-0.08, -0.02),
    band_gap: float = 0.012,
    legend_loc: str = "upper right",
    legend_fontsize: float = 12,
    show_grid: bool = False,
    x_pad: float = 0.005,
    touch_epsilon: float = 1e-6,
) -> None:
    """Save the notebook-style fsigma8 overview plot used in the paper."""

    plt.rcParams.update(PLOT_STYLE)

    def _norm_pair(a: float, b: float, ndig: int = 6) -> tuple[float, float]:
        return (round(float(a), ndig), round(float(b), ndig))

    all_ranges: list[tuple[float, float]] = []
    for df in results_by_label.values():
        if df is None or df.empty:
            continue
        for row in df[["z_min", "z_max"]].dropna().itertuples(index=False):
            zr = _norm_pair(row.z_min, row.z_max)
            if zr not in all_ranges:
                all_ranges.append(zr)
    all_ranges.sort()
    if not all_ranges:
        raise ValueError("No redshift ranges found.")

    if range_colors is None:
        base = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "gold",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:olive",
            "tab:cyan",
        ]
        range_colors = (base * ((len(all_ranges) + len(base) - 1) // len(base)))[
            : len(all_ranges)
        ]
    range_color_map = {zr: c for zr, c in zip(all_ranges, range_colors)}

    year_size_map = dict(
        year_size_map
        or {
            "WFD Y1": 40,
            "WFD Y3": 80,
            "WFD Y5": 120,
            "WFD Y1 + 1 Year Twilight": 40,
        }
    )

    def _size_for_label(label: str) -> float:
        if label in year_size_map:
            return float(year_size_map[label])
        low = label.lower()
        if "5" in low:
            return float(year_size_map.get("WFD Y5", 120))
        if "3" in low:
            return float(year_size_map.get("WFD Y3", 80))
        return float(year_size_map.get("WFD Y1", 40))

    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)

    intervals = sorted(all_ranges, key=lambda ab: (ab[0], ab[1] - ab[0]))
    rows_last_end: list[float] = []
    rows_members: list[list[tuple[float, float]]] = []
    for a, b in intervals:
        placed = False
        for idx, last_end in enumerate(rows_last_end):
            if a >= (last_end - touch_epsilon):
                rows_members[idx].append((a, b))
                rows_last_end[idx] = max(last_end, b)
                placed = True
                break
        if not placed:
            rows_members.append([(a, b)])
            rows_last_end.append(b)

    band_h = band_base[1] - band_base[0]
    x_min = min(r[0] for r in all_ranges)
    x_max = max(r[1] for r in all_ranges)
    for row_idx, row in enumerate(rows_members):
        y0 = band_base[0] - row_idx * (band_h + band_gap)
        y1 = y0 + band_h
        for a, b in row:
            ax.fill_between(
                [a, b],
                y0,
                y1,
                color=range_color_map[(a, b)],
                alpha=0.95,
                linewidth=0,
            )
    lowest_band_bottom = band_base[0] - (len(rows_members) - 1) * (band_h + band_gap)

    if callable(cosmic_variance_fn):
        xs = np.linspace(x_min, x_max, 500)
        ys = np.array([cosmic_variance_fn(xx) for xx in xs], dtype=float)
        ax.plot(xs, ys, "k--", lw=1.8, label="Cosmic variance")

    def _zcol(df: pd.DataFrame) -> str:
        if "zeff" in df.columns:
            return "zeff"
        if "z_eff" in df.columns:
            return "z_eff"
        raise KeyError("DataFrame must contain 'zeff' or 'z_eff'.")

    for label, df in results_by_label.items():
        if df is None or df.empty:
            continue
        zc = _zcol(df)
        d = df.dropna(subset=[zc, "sigma_fs8", "z_min", "z_max"]).copy()
        if d.empty:
            continue
        colors = [
            range_color_map.get(_norm_pair(a, b), "k")
            for a, b in d[["z_min", "z_max"]].itertuples(index=False)
        ]
        marker = combine_marker if "+" in label else "o"
        ax.scatter(
            d[zc].to_numpy(float),
            d["sigma_fs8"].to_numpy(float),
            s=_size_for_label(label),
            c=colors,
            marker=marker,
            edgecolors="k",
            linewidths=0.6,
            zorder=3,
        )

    ax.set_xlabel(r"Observed redshift, $z_{\mathrm{obs}}$")
    ax.set_ylabel(
        r"Relative uncertainty on $f\sigma_8$, "
        r"$\sigma_{f\sigma_8}/(f\sigma_8)_{\mathrm{fid}}$"
    )
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if show_grid:
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    cur_top = ax.get_ylim()[1]
    ax.set_ylim(lowest_band_bottom - 0.02, y_max if y_max is not None else max(0.8, cur_top))
    if title:
        ax.set_title(title)

    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    handles = [mpatches.Patch(facecolor="lightgray", edgecolor="lightgray")]
    labels = ["Redshift range"]
    if callable(cosmic_variance_fn):
        handles.append(Line2D([0], [0], color="k", linestyle="--", lw=1.8))
        labels.append("Illustrative cosmic variance")
    for scenario, legend_label in (
        ("WFD Y1", "WFD 1 yr"),
        ("WFD Y3", "WFD 3 yr"),
        ("WFD Y5", "WFD 5 yr"),
    ):
        if scenario in results_by_label:
            handles.append(
                plt.scatter(
                    [],
                    [],
                    s=_size_for_label(scenario),
                    c="white",
                    edgecolors="k",
                )
            )
            labels.append(legend_label)
    for label in results_by_label:
        if "+" in label:
            handles.append(
                plt.scatter(
                    [],
                    [],
                    s=_size_for_label(label),
                    c="white",
                    edgecolors="k",
                    marker=combine_marker,
                )
            )
            labels.append(
                {
                    "WFD Y1 + 1 Year Twilight": "WFD 1 yr + 1 yr Twilight",
                    "WFD+Twilight Y1": "WFD 1 yr + 1 yr Twilight",
                }.get(label, label)
            )
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc=legend_loc,
        bbox_to_anchor=(1.0, 1.0),
        fontsize=legend_fontsize,
        labelspacing=0.3,
        handletextpad=0.5,
        borderpad=0.3,
        markerscale=1.0,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
