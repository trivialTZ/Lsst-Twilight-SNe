"""Input/output helpers for twilight cosmology notebooks."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

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
    df = pd.DataFrame(arr.byteswap().newbyteorder())
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
    """Read FITRES (ASCII ``.gz`` or ROOT) and standardize columns to current schema."""
    p = Path(path)
    candidates: list[Path] = []
    if p.is_file():
        candidates.append(p)
    elif p.is_dir():
        candidates += sorted(p.glob("*.FITRES.gz"))
        candidates += sorted(p.glob("*.ROOT*"))
    if not candidates:
        print(f"[WARN] No FITRES found at {path}")
        return pd.DataFrame()
    for cand in candidates:
        if cand.suffixes[-2:] == [".FITRES", ".gz"] or cand.name.upper().endswith(
            ".FITRES.GZ"
        ):
            return read_fitres_ascii_gz(cand)
    for cand in candidates:
        if ".ROOT" in cand.name.upper() and uproot is not None:
            return read_fitres_root(cand)
    print(f"[WARN] No readable FITRES among: {[str(pp) for pp in candidates]}")
    return pd.DataFrame()


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
    df = pd.DataFrame(arr.byteswap().newbyteorder())
    for c in ("MJD", "FLUXCAL", "FLUXCALERR", "FLUX", "FLUXERR", "PHOTFLAG"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
