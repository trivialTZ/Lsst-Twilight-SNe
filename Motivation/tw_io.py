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
        if (
            cand.suffixes[-2:] == [".FITRES", ".gz"]
            or cand.name.upper().endswith(".FITRES.GZ")
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
