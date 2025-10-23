"""Utilities for reading SNANA SIMLIB coordinate metadata.

This provides a lightweight parser for SIMLIB text files to extract
per-``LIBID`` sky coordinates (``RA``/``DEC`` in degrees), as well as a
convenience function to attach those coordinates to a Twilight FITRES-
derived DataFrame that includes ``SIM_LIBID``.

The parser is intentionally simple and robust to minor formatting changes:
- Detects the start of a block with ``LIBID: <int>``
- Within a block, finds any line containing ``RA: <float>`` and ``DEC: <float>``
- Optionally parses ``REDSHIFT`` and ``PEAKMJD`` if present

No external dependencies beyond pandas/numpy are required.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


_RE_LIBID = re.compile(r"^\s*LIBID:\s*(\d+)")
_RE_RADEC = re.compile(r"RA:\s*([+-]?(?:\d+\.?\d*|\d*\.\d+))\s+DEC:\s*([+-]?(?:\d+\.?\d*|\d*\.\d+))")
_RE_REDSHIFT = re.compile(r"REDSHIFT:\s*([+-]?(?:\d+\.?\d*|\d*\.\d+))")
_RE_PEAKMJD = re.compile(r"PEAKMJD:\s*([+-]?(?:\d+\.?\d*|\d*\.\d+))")


def read_simlib_coords(path: str | Path) -> pd.DataFrame:
    """Parse a SIMLIB file and return ``LIBID``→``RA``/``DEC`` mapping.

    Parameters
    ----------
    path : str or Path
        Path to a SIMLIB text file.

    Returns
    -------
    pandas.DataFrame
        Columns: ``LIBID`` (Int64), ``RA`` (float, degrees in [0, 360)),
        ``DEC`` (float, degrees), and optionally ``REDSHIFT`` and ``PEAKMJD``
        if they are present in the file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SIMLIB not found: {p}")
    rows: List[Dict[str, float]] = []
    current: Dict[str, float] | None = None

    with open(p, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_id = _RE_LIBID.search(line)
            if m_id:
                # Save previous block if it had coords
                if current and "RA" in current and "DEC" in current:
                    rows.append(current)
                current = {"LIBID": int(m_id.group(1))}
                continue
            if current is None:
                continue
            # Parse RA/DEC if present on this line
            m_rd = _RE_RADEC.search(line)
            if m_rd:
                ra = float(m_rd.group(1))
                dec = float(m_rd.group(2))
                # Normalize RA to [0, 360)
                ra = float(np.mod(ra, 360.0))
                current["RA"] = ra
                current["DEC"] = dec
            # Optional extras
            m_z = _RE_REDSHIFT.search(line)
            if m_z:
                try:
                    current["REDSHIFT"] = float(m_z.group(1))
                except Exception:
                    pass
            m_pk = _RE_PEAKMJD.search(line)
            if m_pk:
                try:
                    current["PEAKMJD"] = float(m_pk.group(1))
                except Exception:
                    pass

    if current and "RA" in current and "DEC" in current:
        rows.append(current)

    if not rows:
        return pd.DataFrame(columns=["LIBID", "RA", "DEC"]).astype({"LIBID": "Int64"})

    df = pd.DataFrame(rows)
    # Enforce dtypes and expected ranges
    df["LIBID"] = pd.to_numeric(df["LIBID"], errors="coerce").astype("Int64")
    for c in ("RA", "DEC"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Normalize RA to [0, 360)
    if "RA" in df:
        df["RA"] = np.mod(df["RA"], 360.0)
    return df


def attach_coords_from_simlib(
    df_tw: pd.DataFrame, simlib_path: str | Path
) -> pd.DataFrame:
    """Attach RA/DEC from SIMLIB to a Twilight FITRES-like table.

    Parameters
    ----------
    df_tw : pandas.DataFrame
        Table containing at least ``ID_int`` and ``SIM_LIBID`` columns.
    simlib_path : str or Path
        Path to the Twilight SIMLIB file.

    Returns
    -------
    pandas.DataFrame
        A two-column mapping ``['ID_int','RA','DEC']`` suitable for merging
        back onto a QC'ed Twilight subset.
    """
    coords = read_simlib_coords(simlib_path)
    if coords.empty:
        return pd.DataFrame({"ID_int": pd.Series(dtype="Int64"), "RA": [], "DEC": []})
    # Ensure proper types for join
    left = df_tw[["ID_int", "SIM_LIBID"]].copy()
    left["SIM_LIBID"] = pd.to_numeric(left["SIM_LIBID"], errors="coerce").astype("Int64")
    coords = coords[["LIBID", "RA", "DEC"]].copy()
    coords["LIBID"] = pd.to_numeric(coords["LIBID"], errors="coerce").astype("Int64")
    out = left.merge(coords, left_on="SIM_LIBID", right_on="LIBID", how="left")
    out = out.drop(columns=["SIM_LIBID", "LIBID"]).drop_duplicates(subset=["ID_int"]) 
    return out

