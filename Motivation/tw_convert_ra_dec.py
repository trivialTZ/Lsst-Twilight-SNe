import re
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Tuple, Dict


def parse_simlib_ra_dec(simlib_path: str | Path) -> pd.DataFrame:
    libid_pat = re.compile(r'\bLIBID\b\s*[:=]\s*(\d+)')
    ra_pat    = re.compile(r'\bRA\b\s*[:=]\s*([+-]?\d+(?:\.\d*)?)')
    dec_pat   = re.compile(r'\bDEC\b\s*[:=]\s*([+-]?\d+(?:\.\d*)?)', re.IGNORECASE)
    redshift_pat = re.compile(
        r'\bREDSHIFT\b\s*[:=]\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)',
        re.IGNORECASE,
    )

    records = []
    cur = {"LIBID": None, "RA": None, "DEC": None, "REDSHIFT": None}

    with open(simlib_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m_id = libid_pat.search(line)
            if m_id:
                if cur["LIBID"] is not None and cur["RA"] is not None and cur["DEC"] is not None:
                    records.append(
                        (
                            int(cur["LIBID"]),
                            float(cur["RA"]),
                            float(cur["DEC"]),
                            np.nan if cur["REDSHIFT"] is None else float(cur["REDSHIFT"]),
                        )
                    )
                cur = {"LIBID": int(m_id.group(1)), "RA": None, "DEC": None, "REDSHIFT": None}
                m_ra = ra_pat.search(line); m_dec = dec_pat.search(line)
                if m_ra:  cur["RA"]  = float(m_ra.group(1))
                if m_dec: cur["DEC"] = float(m_dec.group(1))
                m_redshift = redshift_pat.search(line)
                if m_redshift: cur["REDSHIFT"] = float(m_redshift.group(1))
                continue
            if cur["LIBID"] is not None:
                m_ra = ra_pat.search(line)
                if m_ra:  cur["RA"]  = float(m_ra.group(1))
                m_dec = dec_pat.search(line)
                if m_dec: cur["DEC"] = float(m_dec.group(1))
                m_redshift = redshift_pat.search(line)
                if m_redshift: cur["REDSHIFT"] = float(m_redshift.group(1))
        if cur["LIBID"] is not None and cur["RA"] is not None and cur["DEC"] is not None:
            records.append(
                (
                    int(cur["LIBID"]),
                    float(cur["RA"]),
                    float(cur["DEC"]),
                    np.nan if cur["REDSHIFT"] is None else float(cur["REDSHIFT"]),
                )
            )

    return pd.DataFrame(
        records,
        columns=["LIBID", "RA_simlib", "DEC_simlib", "REDSHIFT_simlib"],
    )


def fix_ra_dec_from_simlib(df: pd.DataFrame,
                           simlib_path: str | Path,
                           id_col: str = "SIM_LIBID",
                           ra_col: str = "RA",
                           dec_col: str = "DEC",
                           overwrite: str = "all",
                           require_all_mapped: bool = False,
                           redshift_col: str | None = None,
                           require_redshift_match: bool = False,
                           redshift_atol: float = 5.1e-6,
                           allow_unset_simlib_redshift: bool = False,
                           simlib_redshift_set_threshold: float = 1.0e-5,
) -> tuple[pd.DataFrame, dict]:
    """
    Replace RA/DEC in `df` using RA/DEC from SIMLIB matched by `id_col`.
    overwrite: 'all' or 'nan_only'

    SNANA applies a per-LIBID ``REDSHIFT`` only when it is greater than
    1e-5.  With ``allow_unset_simlib_redshift=True``, finite SIMLIB values
    at or below that threshold are therefore treated as an explicit
    "generate from the input redshift distribution" sentinel and exempted
    from the equality check. Missing/non-numeric SIMLIB redshifts still fail
    when ``require_redshift_match`` is enabled.
    """
    if id_col not in df.columns:
        raise KeyError(f"`{id_col}` not found in DataFrame.")
    if require_redshift_match and redshift_col is None:
        raise ValueError("`redshift_col` is required when checking SIMLIB redshifts.")
    if redshift_col is not None and redshift_col not in df.columns:
        raise KeyError(f"`{redshift_col}` not found in DataFrame.")

    lib_df = parse_simlib_ra_dec(simlib_path)
    if lib_df.empty:
        raise ValueError("Parsed SIMLIB is empty—no (LIBID, RA, DEC) found.")

    map_df = lib_df.rename(columns={"LIBID": id_col})
    merged = df.merge(map_df, on=id_col, how="left", sort=False, copy=True, validate="m:1")

    mapped_rows = merged["RA_simlib"].notna() & merged["DEC_simlib"].notna()
    n_rows_mapped = int(mapped_rows.sum())
    n_rows_unmapped = int((~mapped_rows).sum())
    if require_all_mapped and n_rows_unmapped:
        missing_ids = (
            merged.loc[~mapped_rows, id_col]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"SIMLIB coordinate mapping is incomplete for {simlib_path}: "
            f"mapped {n_rows_mapped}/{len(merged)} rows; "
            f"first missing {id_col} values={missing_ids}"
        )

    n_redshift_compared = 0
    n_redshift_mismatch = 0
    n_redshift_unset_exempt = 0
    if redshift_col is not None:
        fitres_redshift = pd.to_numeric(merged[redshift_col], errors="coerce")
        simlib_redshift = pd.to_numeric(merged["REDSHIFT_simlib"], errors="coerce")
        simlib_redshift_unset = (
            mapped_rows
            & simlib_redshift.notna()
            & (simlib_redshift <= float(simlib_redshift_set_threshold))
            & bool(allow_unset_simlib_redshift)
        )
        redshift_required = mapped_rows & ~simlib_redshift_unset
        comparable = (
            redshift_required
            & fitres_redshift.notna()
            & simlib_redshift.notna()
        )
        redshift_match = np.isclose(
            fitres_redshift.to_numpy(float),
            simlib_redshift.to_numpy(float),
            rtol=0.0,
            atol=float(redshift_atol),
            equal_nan=False,
        )
        n_redshift_compared = int(comparable.sum())
        n_redshift_mismatch = int((comparable & ~redshift_match).sum())
        n_redshift_unset_exempt = int(simlib_redshift_unset.sum())
        if require_redshift_match:
            missing_redshift = redshift_required & ~comparable
            if bool(missing_redshift.any()) or n_redshift_mismatch:
                bad = merged.loc[missing_redshift | (comparable & ~redshift_match), id_col]
                bad_ids = bad.dropna().astype(str).drop_duplicates().head(10).tolist()
                raise ValueError(
                    f"SIMLIB redshift mapping is inconsistent for {simlib_path}: "
                    f"compared {n_redshift_compared}/{n_rows_mapped} mapped rows, "
                    f"found {n_redshift_mismatch} mismatches; "
                    f"first inconsistent {id_col} values={bad_ids}"
                )

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
        n_rows_mapped=n_rows_mapped,
        n_rows_unmapped=n_rows_unmapped,
        n_redshift_compared=n_redshift_compared,
        n_redshift_mismatch=n_redshift_mismatch,
        n_redshift_unset_exempt=n_redshift_unset_exempt,
        ra_nan_before=before_ra_nan, ra_nan_after=after_ra_nan,
        dec_nan_before=before_dec_nan, dec_nan_after=after_dec_nan,
        n_ra_changed=n_ra_changed, n_dec_changed=n_dec_changed
    )

    merged = merged.drop(columns=["RA_simlib", "DEC_simlib", "REDSHIFT_simlib"])
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
