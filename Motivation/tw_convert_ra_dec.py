import re
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Tuple, Dict


def parse_simlib_ra_dec(simlib_path: str | Path) -> pd.DataFrame:
    libid_pat = re.compile(r'\bLIBID\b\s*[:=]\s*(\d+)')
    ra_pat    = re.compile(r'\bRA\b\s*[:=]\s*([+-]?\d+(?:\.\d*)?)')
    dec_pat   = re.compile(r'\bDEC\b\s*[:=]\s*([+-]?\d+(?:\.\d*)?)', re.IGNORECASE)

    records = []
    cur = {"LIBID": None, "RA": None, "DEC": None}

    with open(simlib_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m_id = libid_pat.search(line)
            if m_id:
                if cur["LIBID"] is not None and cur["RA"] is not None and cur["DEC"] is not None:
                    records.append((int(cur["LIBID"]), float(cur["RA"]), float(cur["DEC"])))
                cur = {"LIBID": int(m_id.group(1)), "RA": None, "DEC": None}
                m_ra = ra_pat.search(line); m_dec = dec_pat.search(line)
                if m_ra:  cur["RA"]  = float(m_ra.group(1))
                if m_dec: cur["DEC"] = float(m_dec.group(1))
                continue
            if cur["LIBID"] is not None:
                m_ra = ra_pat.search(line)
                if m_ra:  cur["RA"]  = float(m_ra.group(1))
                m_dec = dec_pat.search(line)
                if m_dec: cur["DEC"] = float(m_dec.group(1))
        if cur["LIBID"] is not None and cur["RA"] is not None and cur["DEC"] is not None:
            records.append((int(cur["LIBID"]), float(cur["RA"]), float(cur["DEC"])))

    return pd.DataFrame(records, columns=["LIBID", "RA_simlib", "DEC_simlib"])


def fix_ra_dec_from_simlib(df: pd.DataFrame,
                           simlib_path: str | Path,
                           id_col: str = "SIM_LIBID",
                           ra_col: str = "RA",
                           dec_col: str = "DEC",
                           overwrite: str = "all") -> tuple[pd.DataFrame, dict]:
    """
    Replace RA/DEC in `df` using RA/DEC from SIMLIB matched by `id_col`.
    overwrite: 'all' or 'nan_only'
    """
    if id_col not in df.columns:
        raise KeyError(f"`{id_col}` not found in DataFrame.")

    lib_df = parse_simlib_ra_dec(simlib_path)
    if lib_df.empty:
        raise ValueError("Parsed SIMLIB is empty—no (LIBID, RA, DEC) found.")

    map_df = lib_df.rename(columns={"LIBID": id_col})
    merged = df.merge(map_df, on=id_col, how="left", sort=False, copy=True, validate="m:1")

    before_ra_nan = int(merged[ra_col].isna().sum())
    before_dec_nan = int(merged[dec_col].isna().sum())

    if overwrite == "all":
        mask_ra  = merged["RA_simlib"].notna()
        mask_dec = merged["DEC_simlib"].notna()
        merged.loc[mask_ra,  ra_col]  = merged.loc[mask_ra,  "RA_simlib"].to_numpy()
        merged.loc[mask_dec, dec_col] = merged.loc[mask_dec, "DEC_simlib"].to_numpy()
    elif overwrite == "nan_only":
        fill_ra_mask  = merged[ra_col].isna()  & merged["RA_simlib"].notna()
        fill_dec_mask = merged[dec_col].isna() & merged["DEC_simlib"].notna()
        merged.loc[fill_ra_mask,  ra_col]  = merged.loc[fill_ra_mask,  "RA_simlib"].to_numpy()
        merged.loc[fill_dec_mask, dec_col] = merged.loc[fill_dec_mask, "DEC_simlib"].to_numpy()
    else:
        raise ValueError("`overwrite` must be 'all' or 'nan_only'.")

    after_ra_nan = int(merged[ra_col].isna().sum())
    after_dec_nan = int(merged[dec_col].isna().sum())

    # Safe change counts (compare by values to avoid index label issues)
    n_ra_changed  = int((merged[ra_col].to_numpy()  != df[ra_col].to_numpy()).sum())
    n_dec_changed = int((merged[dec_col].to_numpy() != df[dec_col].to_numpy()).sum())

    stats = dict(
        n_rows=len(df),
        n_with_mapping=map_df[id_col].nunique(),
        ra_nan_before=before_ra_nan, ra_nan_after=after_ra_nan,
        dec_nan_before=before_dec_nan, dec_nan_after=after_dec_nan,
        n_ra_changed=n_ra_changed, n_dec_changed=n_dec_changed
    )

    merged = merged.drop(columns=["RA_simlib", "DEC_simlib"])
    merged = merged[df.columns]  # restore column order

    return merged, stats
import pandas as pd
import numpy as np
from typing import Tuple, Dict

def promote_host_coords_to_ra_dec(
    df: pd.DataFrame,
    host_ra_col: str = "HOST_RA",
    host_dec_col: str = "HOST_DEC",
    ra_col: str = "RA",
    dec_col: str = "DEC",
    invalid_sentinel: float = -999.0,
    drop_invalid_host: bool = True,   # drop rows with invalid HOST_* first
    inplace: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Promote HOST_RA/HOST_DEC to RA/DEC when RA/DEC are missing/invalid.
    Optionally drop rows whose HOST_* are invalid before promotion.

    Returns (new_df, stats).
    """
    if not inplace:
        df = df.copy()

    # make sure columns exist
    if ra_col not in df.columns:
        df[ra_col] = np.nan
    if dec_col not in df.columns:
        df[dec_col] = np.nan

    # coerce to numeric
    for c in (host_ra_col, host_dec_col, ra_col, dec_col):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # if requested, drop invalid hosts first
    if (host_ra_col in df.columns) and (host_dec_col in df.columns):
        host_invalid_mask = (
            df[host_ra_col].isna() | df[host_dec_col].isna() |
            (df[host_ra_col] == invalid_sentinel) | (df[host_dec_col] == invalid_sentinel)
        )
        dropped = int(host_invalid_mask.sum()) if drop_invalid_host else 0
        if drop_invalid_host and dropped:
            df = df.loc[~host_invalid_mask].reset_index(drop=True)
    else:
        # no host columns; nothing to promote
        return df, {
            "n_rows": len(df),
            "n_candidates": 0,
            "n_ra_filled": 0,
            "n_dec_filled": 0,
            "n_both_filled": 0,
            "n_host_dropped": 0
        }

    valid_host = (
        df[host_ra_col].notna() & df[host_dec_col].notna() &
        (df[host_ra_col] != invalid_sentinel) & (df[host_dec_col] != invalid_sentinel)
    )

    invalid_ra  = df[ra_col].isna()  | (df[ra_col]  == invalid_sentinel)
    invalid_dec = df[dec_col].isna() | (df[dec_col] == invalid_sentinel)

    fill_ra_mask   = valid_host & invalid_ra
    fill_dec_mask  = valid_host & invalid_dec
    fill_both_mask = valid_host & invalid_ra & invalid_dec

    df.loc[fill_ra_mask,  ra_col]  = df.loc[fill_ra_mask,  host_ra_col].to_numpy()
    df.loc[fill_dec_mask, dec_col] = df.loc[fill_dec_mask, host_dec_col].to_numpy()

    stats = {
        "n_rows": len(df),
        "n_candidates": int(valid_host.sum()),
        "n_ra_filled": int(fill_ra_mask.sum()),
        "n_dec_filled": int(fill_dec_mask.sum()),
        "n_both_filled": int(fill_both_mask.sum()),
        "n_host_dropped": int(dropped) if drop_invalid_host else 0,
    }
    return df, stats
