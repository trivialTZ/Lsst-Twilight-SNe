"""Input/output helpers for twilight cosmology notebooks.

Additions:
- ``load_snana_fitres_ascii``: robust reader for SNANA FITRES ASCII that
  standardizes ``z`` column (maps ``zHD``/``zCMB``/``z`` to ``z``).
- ``load_snana_head_fits``: reader for SNANA ``*_HEAD.FITS`` that exposes a
  single ``z`` column suitable for detection-count histogramming.
"""

from __future__ import annotations

import gzip
from pathlib import Path
import re
from typing import Literal

import numpy as np
import pandas as pd
from astropy.io import fits

try:
    from astropy.table import Table  # for HEAD FITS convenience
except Exception:  # pragma: no cover - optional import
    Table = None

try:  # optional ROOT reader
    import uproot  # type: ignore
except Exception:  # pragma: no cover - uproot optional
    uproot = None


def _to_int64_safe(x: object) -> object:
    """Convert arbitrary value to ``Int64`` or ``pd.NA``."""
    if pd.isna(x):
        return pd.NA
    try:
        if isinstance(x, bytes):
            x = x.decode(errors="ignore")
        if isinstance(x, str):
            x = x.strip()
        return np.int64(int(float(x)))
    except Exception:
        return pd.NA


def _clean_chars_inplace(df: pd.DataFrame) -> None:
    for c in df.columns:
        if df[c].dtype.kind in ("S", "O", "U"):
            try:
                df[c] = df[c].astype(str).str.strip()
            except Exception:
                pass


def _to_numeric_if_possible(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s)
    except Exception:
        return s


def _first_finite_numeric(df: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    """Return the first finite numeric value available in each row."""

    result = pd.Series(np.nan, index=df.index, dtype=float)
    for name in names:
        if name not in df.columns:
            continue
        candidate = pd.to_numeric(df[name], errors="coerce").astype(float)
        candidate = candidate.replace([np.inf, -np.inf], np.nan)
        result = result.where(result.notna(), candidate)
    return result


def _variance_from_error(df: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    """Return the square of the first finite error available in each row."""

    error = _first_finite_numeric(df, names)
    return error.pow(2)


def salt_covariance_mB_x1_c(
    df: pd.DataFrame,
    *,
    missing: Literal["raise", "zero"] = "raise",
) -> pd.DataFrame:
    """Return each SALT covariance in the ``(mB, x1, c)`` basis.

    SNANA FITRES files commonly store the amplitude cross-covariances as
    ``COV_x1_x0`` and ``COV_c_x0`` even when they also report ``mB`` and
    ``mBERR``.  For ``mB = -2.5 log10(x0) + constant``, this function applies

    ``dmB/dx0 = -2.5 / (ln(10) x0)``

    row by row to transform those terms.  A finite covariance already supplied
    in the ``mB`` basis takes precedence; an ``x0``-basis value only fills rows
    where the direct value is absent.  The returned columns are ``VAR_mB``,
    ``VAR_x1``, ``VAR_c``, ``COV_mB_x1``, ``COV_mB_c``, and ``COV_x1_c``.

    Parameters
    ----------
    df : pandas.DataFrame
        FITRES-like rows containing SALT errors/covariances.  Covariance names
        are accepted in either order (for example, ``COV_x1_x0`` or
        ``COV_x0_x1``).
    missing : {"raise", "zero"}, optional
        Missing covariance elements raise a descriptive ``ValueError`` by
        default.  ``"zero"`` is an explicit opt-in for legacy/toy tables; it
        never suppresses an available ``x0``-basis covariance.

    Returns
    -------
    pandas.DataFrame
        One covariance row per input row, with the same index as ``df``.
    """

    if missing not in {"raise", "zero"}:
        raise ValueError("missing must be either 'raise' or 'zero'")

    x0 = _first_finite_numeric(df, ("x0", "X0"))
    dm_b_dx0 = pd.Series(np.nan, index=df.index, dtype=float)
    # SALT amplitudes must be positive for the logarithmic magnitude mapping.
    valid_x0 = x0.notna() & (x0 > 0.0)
    dm_b_dx0.loc[valid_x0] = -2.5 / (np.log(10.0) * x0.loc[valid_x0])

    var_x0 = _first_finite_numeric(df, ("COV_x0_x0", "COV_X0_X0", "VAR_x0", "VAR_X0"))
    var_x0 = var_x0.where(var_x0.notna(), _variance_from_error(df, ("x0ERR", "X0ERR")))

    var_mb = _first_finite_numeric(df, ("COV_mB_mB", "COV_MB_MB", "VAR_mB", "VAR_MB"))
    var_mb = var_mb.where(var_mb.notna(), _variance_from_error(df, ("mBERR", "MBERR")))
    var_mb = var_mb.where(var_mb.notna(), dm_b_dx0.pow(2) * var_x0)

    var_x1 = _first_finite_numeric(df, ("COV_x1_x1", "COV_X1_X1", "VAR_x1", "VAR_X1"))
    var_x1 = var_x1.where(var_x1.notna(), _variance_from_error(df, ("x1ERR", "X1ERR")))

    var_c = _first_finite_numeric(df, ("COV_c_c", "COV_C_C", "VAR_c", "VAR_C"))
    var_c = var_c.where(var_c.notna(), _variance_from_error(df, ("cERR", "CERR")))

    cov_x0_x1 = _first_finite_numeric(
        df,
        ("COV_x0_x1", "COV_x1_x0", "COV_X0_X1", "COV_X1_X0"),
    )
    cov_x0_c = _first_finite_numeric(
        df,
        ("COV_x0_c", "COV_c_x0", "COV_X0_C", "COV_C_X0"),
    )

    cov_mb_x1 = _first_finite_numeric(
        df,
        ("COV_mB_x1", "COV_x1_mB", "COV_MB_X1", "COV_X1_MB"),
    )
    cov_mb_x1 = cov_mb_x1.where(cov_mb_x1.notna(), dm_b_dx0 * cov_x0_x1)

    cov_mb_c = _first_finite_numeric(
        df,
        ("COV_mB_c", "COV_c_mB", "COV_MB_C", "COV_C_MB"),
    )
    cov_mb_c = cov_mb_c.where(cov_mb_c.notna(), dm_b_dx0 * cov_x0_c)

    cov_x1_c = _first_finite_numeric(
        df,
        ("COV_x1_c", "COV_c_x1", "COV_x1c", "COV_X1_C", "COV_C_X1"),
    )

    covariance = pd.DataFrame(
        {
            "VAR_mB": var_mb,
            "VAR_x1": var_x1,
            "VAR_c": var_c,
            "COV_mB_x1": cov_mb_x1,
            "COV_mB_c": cov_mb_c,
            "COV_x1_c": cov_x1_c,
        },
        index=df.index,
        dtype=float,
    )

    negative_variances = {
        name: int((covariance[name] < 0.0).sum())
        for name in ("VAR_mB", "VAR_x1", "VAR_c")
        if bool((covariance[name] < 0.0).any())
    }
    if negative_variances:
        details = ", ".join(
            f"{name} ({count} row(s))" for name, count in negative_variances.items()
        )
        raise ValueError(f"SALT covariance contains negative variances: {details}")

    missing_counts = covariance.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        if missing == "raise":
            details = ", ".join(
                f"{name} ({int(count)} row(s))"
                for name, count in missing_counts.items()
            )
            raise ValueError(
                "Could not construct a complete (mB, x1, c) SALT covariance: "
                f"missing {details}. Provide direct mB-basis terms or x0 and "
                "the corresponding x0-basis covariances."
            )
        covariance = covariance.fillna(0.0)

    return covariance


# -----------------------------------------------------------------------------
# FITS/ASCII readers
# -----------------------------------------------------------------------------


def read_head_fits(path: str | Path) -> pd.DataFrame:
    """Read HEAD FITS table and return a standardized DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to a HEAD FITS file.

    Returns
    -------
    pandas.DataFrame
        Standardized HEAD table with expected columns (e.g., ``CID``).
    """
    with fits.open(path) as hdul:
        arr = np.array(hdul[1].data)
    # NumPy 2.0: ndarray.newbyteorder removed; use view with dtype.newbyteorder()
    arr_native = arr.byteswap().view(arr.dtype.newbyteorder())
    df = pd.DataFrame(arr_native)
    _clean_chars_inplace(df)
    if "SNID" in df.columns:
        df["ID_int"] = pd.Series([_to_int64_safe(v) for v in df["SNID"]], dtype="Int64")
    else:
        df["ID_int"] = pd.Series([_to_int64_safe(v) for v in df.index], dtype="Int64")
    zcol = (
        "REDSHIFT_FINAL"
        if "REDSHIFT_FINAL" in df.columns
        else ("REDSHIFT_TRUE" if "REDSHIFT_TRUE" in df.columns else None)
    )
    if zcol is not None:
        df["z"] = pd.to_numeric(df[zcol], errors="coerce")
    if "PEAKMJD" in df.columns:
        df["PEAKMJD"] = pd.to_numeric(df["PEAKMJD"], errors="coerce")
        if "PKMJD" not in df.columns:
            df["PKMJD"] = df["PEAKMJD"]
    for c in ("NOBS", "PTROBS_MIN", "PTROBS_MAX"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def read_fitres_ascii_gz(path: Path) -> pd.DataFrame:
    """Parse SNANA FITRES ASCII (``.FITRES.gz``) robustly."""
    with gzip.open(path, "rt") as f:
        lines = f.readlines()
    names: list[str] | None = None
    start = 0
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("VARNAMES:"):
            names = line.strip().split()[1:]
            start = i + 1
            break
    if names is None:
        raise RuntimeError(f"VARNAMES not found in {path}")
    rows = []
    for line in lines[start:]:
        s = line.strip()
        if (
            not s
            or s.startswith("#")
            or s.upper().startswith(("VARNAMES", "NVAR", "END", "VERSION", "SNANA"))
        ):
            continue
        toks = s.split()
        if toks and toks[0].endswith(":"):
            toks = toks[1:]
        if len(toks) < len(names):
            continue
        rows.append(toks[: len(names)])
    df = pd.DataFrame(rows, columns=names)
    for c in df.columns:
        df[c] = _to_numeric_if_possible(df[c])
    return df


def read_fitres_ascii_text(path: Path) -> pd.DataFrame:
    """Parse SNANA FITRES ASCII (plain .FITRES) robustly.

    Mirrors read_fitres_ascii_gz but for uncompressed text files.
    """
    with open(path, "rt") as f:
        lines = f.readlines()
    names: list[str] | None = None
    start = 0
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("VARNAMES:"):
            names = line.strip().split()[1:]
            start = i + 1
            break
    if names is None:
        raise RuntimeError(f"VARNAMES not found in {path}")
    rows = []
    for line in lines[start:]:
        s = line.strip()
        if (
            not s
            or s.startswith("#")
            or s.upper().startswith(("VARNAMES", "NVAR", "END", "VERSION", "SNANA"))
        ):
            continue
        toks = s.split()
        if toks and toks[0].endswith(":"):
            toks = toks[1:]
        if len(toks) < len(names):
            continue
        rows.append(toks[: len(names)])
    df = pd.DataFrame(rows, columns=names)
    for c in df.columns:
        df[c] = _to_numeric_if_possible(df[c])
    return df


def read_fitres_root(path: Path) -> pd.DataFrame:
    """Read FITRES from a ROOT file."""
    if uproot is None:
        raise RuntimeError("uproot is not available to read ROOT FITRES.")
    with uproot.open(path) as f:
        tree = None
        for key in ("FITRES", "FITOPT000", "FITRES/FITRES"):
            if key in f:
                tree = f[key]
                break
        if tree is None:
            for _, obj in f.items():
                if hasattr(obj, "arrays"):
                    tree = obj
                    break
        if tree is None:
            raise RuntimeError(f"No TTree found in {path}")
        df = tree.arrays(library="pd")
    _clean_chars_inplace(df)
    return df.reset_index(drop=True)


def read_fitres_any(path: str | Path) -> pd.DataFrame:
    """Read FITRES (ASCII .FITRES[.gz] or ROOT) and standardize columns.

    Supports the following, preferring ASCII over ROOT if multiple exist:
      - *.FITRES.gz (compressed ASCII)
      - *.FITRES (plain ASCII)
      - *.ROOT*   (ROOT file via uproot, if available)
    """
    p = Path(path)
    candidates: list[Path] = []
    if p.is_file():
        candidates.append(p)
    elif p.is_dir():
        candidates += sorted(p.glob("*.FITRES.gz"))
        candidates += sorted(p.glob("*.FITRES"))
        candidates += sorted(p.glob("*.ROOT*"))
    if not candidates:
        print(f"[WARN] No FITRES found at {path}")
        return pd.DataFrame()
    # Prefer compressed ASCII, then plain ASCII, then ROOT
    for cand in candidates:
        if cand.suffixes[-2:] == [".FITRES", ".gz"] or cand.name.upper().endswith(
            ".FITRES.GZ"
        ):
            try:
                return read_fitres_ascii_gz(cand)
            except Exception as e:
                print(f"[WARN] Could not read {cand} as FITRES.gz: {e}")
    for cand in candidates:
        if cand.suffixes[-1:] == [".FITRES"] or cand.name.upper().endswith(".FITRES"):
            try:
                return read_fitres_ascii_text(cand)
            except Exception as e:
                print(f"[WARN] Could not read {cand} as FITRES text: {e}")
    for cand in candidates:
        if ".ROOT" in cand.name.upper() and uproot is not None:
            try:
                return read_fitres_root(cand)
            except Exception as e:
                print(f"[WARN] Could not read {cand} as ROOT: {e}")
    print(f"[WARN] No readable FITRES among: {[str(pp) for pp in candidates]}")
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# New loaders for SNANA Twilight data
# -----------------------------------------------------------------------------


def _read_fitres_ascii(path: str | Path) -> pd.DataFrame:
    """Robust reader for SNANA FITRES ASCII (whitespace separated).

    This path supports FITRES files where column names are provided on one of
    the leading comment lines (starting with ``#``). It chooses the longest
    such comment line as the header and parses the remainder as whitespace-
    separated values.
    """
    path = str(path)
    # Detect SNANA 'VARNAMES:' style first and defer to the robust parser above
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(2048)
        if "VARNAMES:" in head.upper():
            return read_fitres_ascii_text(Path(path))
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = None
        for line in f:
            if line.strip().startswith("#"):
                txt = line.strip().lstrip("#").strip()
                if header is None or (len(txt.split()) > len(header.split())):
                    header = txt
            else:
                break
    if header is None:
        # Fall back to pandas sniffing the first data line
        df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#")
        return df
    cols = re.split(r"\s+", header.strip())
    df = pd.read_csv(
        path, sep=r"\s+", engine="python", comment="#", names=cols, header=None
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="all")
    return df


def load_snana_fitres_ascii(
    path: str | Path,
    *,
    prefer_z: tuple[str, ...] = ("zHD", "zCMB", "z"),
) -> pd.DataFrame:
    """Load a SNANA FITRES ASCII into a DataFrame with standardized columns.

    Ensures there is a ``z`` column, chosen from ``prefer_z`` in order. SALT2
    error columns (``mBERR``, ``x1ERR``, ``cERR``, and covariances) are left
    unchanged to plug directly into SALT2 propagation downstream.
    """
    df = _read_fitres_ascii(path)
    df.columns = [c.strip() for c in df.columns]
    z_col = None
    for c in prefer_z:
        if c in df.columns:
            z_col = c
            break
    if z_col is None:
        # Fallback: use the first column that looks like redshift
        candidates = [c for c in df.columns if c.lower().startswith("z")]
        if candidates:
            z_col = candidates[0]
    if z_col is None:
        raise ValueError("No redshift column found in FITRES.")
    df = df.copy()
    df["z"] = pd.to_numeric(df[z_col], errors="coerce")
    return df


def load_snana_head_fits(
    path: str | Path,
    *,
    prefer_z: tuple[str, ...] = (
        "REDSHIFT_FINAL",
        "REDSHIFT_TRUE",
        "zCMB",
        "SIM_ZCMB",
        "zHD",
        "zHEL",
        "Z",
    ),
) -> pd.DataFrame:
    """Load a SNANA ``*_HEAD.FITS`` and expose a single ``z`` column.

    This is intended for building detection-level redshift histograms (``N_det``)
    to set Twilight promotion counts per bin.

    Requires ``astropy``; if unavailable, raises a helpful error.
    """
    if Table is None:
        raise RuntimeError("astropy is required to read FITS HEAD files.")
    t = Table.read(str(path), format="fits")
    df = t.to_pandas()
    z_col = None
    # First, honor explicit preference order
    for c in prefer_z:
        if c in df.columns:
            z_col = c
            break
    # Next, common SNANA HEAD names if not in prefer_z provided by caller
    if z_col is None:
        for c in ("REDSHIFT_FINAL", "REDSHIFT_TRUE", "REDSHIFT_HELIO"):
            if c in df.columns:
                z_col = c
                break
    if z_col is None:
        # Last resort: any column that looks like redshift
        candidates = [c for c in df.columns if c.lower().startswith("z")]
        if candidates:
            z_col = candidates[0]
    if z_col is None:
        raise ValueError("No plausible redshift column found in HEAD.FITS.")
    return pd.DataFrame({"z": pd.to_numeric(df[z_col], errors="coerce")})


def standardize_fitres(df: pd.DataFrame) -> pd.DataFrame:
    """Rename/map FITRES columns to canonical SALT2 names."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ID_int", "z", "PKMJD"])
    _clean_chars_inplace(df)
    if "CIDint" in df.columns:
        df["ID_int"] = pd.to_numeric(df["CIDint"], errors="coerce").astype("Int64")
    elif "CID" in df.columns:
        df["ID_int"] = pd.Series([_to_int64_safe(v) for v in df["CID"]], dtype="Int64")
    elif "SNID" in df.columns:
        df["ID_int"] = pd.Series([_to_int64_safe(v) for v in df["SNID"]], dtype="Int64")
    else:
        df["ID_int"] = pd.Series([_to_int64_safe(v) for v in df.index], dtype="Int64")
    z = None
    for zc in ("zHD", "zCMB", "z", "ZCMB", "Z"):
        if zc in df.columns:
            z = pd.to_numeric(df[zc], errors="coerce")
            break
    df["z"] = z
    pk = None
    for pc in ("PKMJD", "PKMJD_SALT2", "PKMJD_SNIa"):
        if pc in df.columns:
            pk = pd.to_numeric(df[pc], errors="coerce")
            break
    df["PKMJD"] = pk
    return df


def read_phot_fits(path: str | Path) -> pd.DataFrame:
    """Read PHOT FITS and return a standardized DataFrame (epoch-level)."""
    with fits.open(path) as hdul:
        arr = np.array(hdul[1].data)
    # NumPy 2.0: ndarray.newbyteorder removed; use view with dtype.newbyteorder()
    arr_native = arr.byteswap().view(arr.dtype.newbyteorder())
    df = pd.DataFrame(arr_native)
    for c in ("MJD", "FLUXCAL", "FLUXCALERR", "FLUX", "FLUXERR", "PHOTFLAG"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
