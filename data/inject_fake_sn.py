#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inject fake SNe into an ATLAS-like CSV while preserving column order/types.

Updated behavior for time columns (requested):
- All injected time fields are written as MJD (UTC) with high precision.
  Specifically, we write ``creationdate``, ``time_received``, ``lastmodified``,
  and ``discoverydate`` as MJD strings with 8 decimal places (e.g., "60250.12345678").
  Existing/original rows are left untouched.

Other behaviors:
- source_group     = "fake inject"; source_groupid kept dtype-compatible.
- objid            = integer with '9999' prefix (auto-expands suffix width if needed).
- Robust column-name resolving + Unicode cleanup.
- Optional biasing of injected discovery times near a given cut-night or year end.

Note: In the planner, set ``cfg.disc_col = "creationdate"`` if you want to
use the injected creation date as discovery time. The scheduler supports MJD
input per-row (mixed with strings) and converts to UTC datetimes internally.
"""

# -------- progress bar (optional) --------
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # if not installed, just skip the bar

import argparse
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import unicodedata
import re
from astropy.time import Time
from datetime import timezone as _dt_timezone

# ---------------- config ----------------
CUM_TARGETS = (297, 1409, 3660)         # cumulative per year at z<=0.06, z<=0.10, z<=0.14
BINS = [(0.02, 0.06), (0.06, 0.10), (0.10, 0.14)]  # disjoint
BIN_TARGETS = (CUM_TARGETS[0], CUM_TARGETS[1]-CUM_TARGETS[0], CUM_TARGETS[2]-CUM_TARGETS[1])
FAKE_SOURCE_GROUP = "fake inject"
FAKE_SOURCE_GROUPID_INT = 9999          # if original column is numeric -> 9999 else "9999"

# ---------------- unicode/date helpers ----------------
_ALLOWED_ASCII = set("0123456789:/ -T")

def _ascii_nfkc_clean(s: pd.Series) -> pd.Series:
    """NFKC normalize, drop zero-width/NBSP/BOM/WORD JOINER, keep only safe ASCII."""
    def clean_one(x):
        if not isinstance(x, str):
            x = str(x)
        x = unicodedata.normalize("NFKC", x)
        # remove ZW*, NBSP, BOM, WORD JOINER
        x = re.sub(r"[\u200B-\u200D\uFEFF\u2060\u00A0]", "", x)
        # keep only allowed ASCII
        x = "".join(ch for ch in x if ch in _ALLOWED_ASCII).strip()
        return x
    return s.astype(str).map(clean_one)

def _parse_ymdhm_loose(series: pd.Series) -> pd.Series:
    """Parse strings like 'YYYY/M/D H:MM' after cleanup; return naive Timestamp."""
    s = _ascii_nfkc_clean(series.fillna(""))
    parts = s.str.extract(r"^\s*(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})\s*$")
    ok = parts.notna().all(axis=1)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if ok.any():
        dt = pd.to_datetime(
            parts[ok].apply(
                lambda r: f"{int(r[0]):04d}-{int(r[1])}-{int(r[2])} {int(r[3])}:{int(r[4])}",
                axis=1,
            ),
            errors="coerce",
        )
        out.loc[ok.index[ok]] = dt
    rest = ~ok
    if rest.any():
        out.loc[rest] = pd.to_datetime(s[rest], errors="coerce")
    return out

def fmt_ymdhm(dt: datetime) -> str:
    """ASCII 'YYYY/MM/DD HH:MM' (ALL zero-padded)."""
    return f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}"

def _norm_name(s: str) -> str:
    """Normalize a header to compare: NFKC, strip ZW*/NBSP/BOM/WORD JOINER, strip spaces, lower."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u200B-\u200D\uFEFF\u2060\u00A0]", "", s)
    return s.strip().lower()

def _resolve_time_cols(df: pd.DataFrame) -> tuple[str, str, str, str]:
    """
    Return exact column names in df for:
      (creationdate, time_received, lastmodified, discoverydate)
    by matching case-insensitively and ignoring unicode noise.
    If absent, return canonical clean name (we will add it).
    """
    targets = {"creationdate": None, "time_received": None, "lastmodified": None, "discoverydate": None}
    norm_map = {col: _norm_name(col) for col in df.columns}
    for col, n in norm_map.items():
        if n in targets and targets[n] is None:
            targets[n] = col

    # fallbacks
    def pick_fallback(cands, key):
        if targets[key] is not None:
            return
        for col, n in norm_map.items():
            if n in cands:
                targets[key] = col
                return

    pick_fallback({"creation_date", "created_at", "create_time"}, "creationdate")
    pick_fallback({"time received", "timereceived", "received_time", "receivedtime"}, "time_received")
    pick_fallback({"last modified", "last_modified", "modified"}, "lastmodified")
    pick_fallback({"discovery_date", "discovery time", "discoverytime", "disc_date"}, "discoverydate")

    cre = targets["creationdate"] or "creationdate"
    rcv = targets["time_received"] or "time_received"
    mod = targets["lastmodified"] or "lastmodified"
    dsc = targets["discoverydate"] or "discoverydate"
    return cre, rcv, mod, dsc


# ---- MJD helpers ----
def _dt_to_mjd_str(dt: datetime, precision: int = 8) -> str:
    """Convert a naive or tz-aware datetime (assumed UTC if naive) to MJD string.

    Parameters
    ----------
    dt : datetime
        Datetime to convert. If naive, it is treated as UTC.
    precision : int
        Number of decimal places for MJD string.

    Returns
    -------
    str
        MJD in days as a string with fixed decimal places.
    """
    if dt.tzinfo is None:
        # Treat naive as UTC
        mjd = Time(dt, scale="utc").mjd
    else:
        mjd = Time(dt.astimezone(tz=None), scale="utc").mjd
    fmt = f"%.{max(0, int(precision))}f"
    return fmt % mjd

def rand_letters(n: int = 3) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=n))

def make_name(year: int) -> str:
    return f"{year % 100:02d}fk{rand_letters(3)}"

def make_objids(existing: set, n: int, prefix: str = "9999", total_len: int | None = None) -> List[int]:
    """
    Allocate n unique integer objids with the given prefix.
    If total_len is None, automatically choose minimal suffix width so that
    capacity >= n after excluding already-used IDs. Vectorized, no trial loop.
    """
    existing_ints: set[int] = set()
    for oid in existing:
        try:
            existing_ints.add(int(oid))
        except Exception:
            s = str(oid)
            if s.isdigit():
                existing_ints.add(int(s))

    width = max(1, (total_len - len(prefix)) if (total_len is not None) else 4)
    while True:
        cap = 10 ** width
        base = int(prefix) * cap
        upper = base + cap - 1

        used_suffixes = set()
        for oid in existing_ints:
            if base <= oid <= upper:
                used_suffixes.add(oid - base)

        available = cap - len(used_suffixes)
        if n <= available:
            if not used_suffixes:
                choices = np.random.choice(cap, size=n, replace=False)
            else:
                all_suffix = np.arange(cap, dtype=np.int64)
                used_arr = np.fromiter(used_suffixes, dtype=np.int64) if used_suffixes else np.array([], dtype=np.int64)
                free_suffix = np.setdiff1d(all_suffix, used_arr, assume_unique=False)
                choices = np.random.choice(free_suffix, size=n, replace=False)
            return (base + choices).astype(np.int64).tolist()

        width += 1  # expand suffix width and retry

def sample_month_day_time(year: int,
                          base_dt: Optional[pd.Series] = None,
                          bias_near_cut: Optional[datetime] = None,
                          bias_end_of_year: bool = False,
                          bias_window_days: int = 120) -> datetime:
    """
    Pick a month/day/time for 'year' with optional biasing.
    """
    if bias_near_cut is not None:
        end = datetime(bias_near_cut.year, bias_near_cut.month, bias_near_cut.day, 23, 59)
        start = end - timedelta(days=max(1, bias_window_days))
    elif bias_end_of_year:
        end = datetime(year, 12, 31, 23, 59)
        start = end - timedelta(days=max(1, bias_window_days))
    else:
        start = None
        end = None

    if start is not None:
        delta_min = int((end - start).total_seconds() // 60)
        off = random.randint(0, max(1, delta_min))
        dt = start + timedelta(minutes=off)
        return dt.replace(second=0, microsecond=0)

    if base_dt is not None and base_dt.notna().any():
        pick = pd.to_datetime(base_dt.dropna().sample(1).iloc[0]).to_pydatetime()
        m, d, h, M = pick.month, pick.day, pick.hour, pick.minute
    else:
        m = random.randint(1, 12)
        d = random.randint(1, 28 if m == 2 else (30 if m in {4,6,9,11} else 31))
        h = random.randint(0, 23)
        M = random.randint(0, 59)
    if m == 2 and d == 29:
        try:
            datetime(year, 2, 29)
        except ValueError:
            d = 28
    return datetime(year, m, d, h, M)

def summarize_by_year(df: pd.DataFrame, year_series: pd.Series) -> pd.DataFrame:
    rows = []
    yy = year_series.dropna().astype(int)
    if yy.empty:
        return pd.DataFrame(columns=["year","N(0.02,0.06]","N(0.06,0.10]","N(0.10,0.14]","N<=0.06","N<=0.10","N<=0.14","total"])
    for y in sorted(yy.unique()):
        idx = (year_series == y)
        sub = df.loc[idx]
        s = sub.loc[sub["redshift"].between(0.02, 0.14, inclusive="both"), "redshift"]
        n1 = ((s > 0.02) & (s <= 0.06)).sum()
        n2 = ((s > 0.06) & (s <= 0.10)).sum()
        n3 = ((s > 0.10) & (s <= 0.14)).sum()
        rows.append({
            "year": int(y),
            "N(0.02,0.06]": int(n1),
            "N(0.06,0.10]": int(n2),
            "N(0.10,0.14]": int(n3),
            "N<=0.06": int(n1),
            "N<=0.10": int(n1+n2),
            "N<=0.14": int(n1+n2+n3),
            "total": int(sub.shape[0]),
        })
    return pd.DataFrame(rows)

# ---------- NEW: strong normalization for time columns (fixes the 2 bad rows) ----------
_TIME_PAT = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?\s*$")

def _normalize_time_str_series(s: pd.Series) -> pd.Series:
    """
    Normalize a time-like series to strict 'YYYY/MM/DD HH:MM' ASCII strings.
    Handles pandas.Timestamp, numpy.datetime64, and common string variants
    like 'YYYY-MM-DD HH:MM:SS' or 'YYYY/M/D H:MM'.
    """
    def norm_one(x):
        if pd.isna(x):
            return x
        if isinstance(x, (pd.Timestamp, datetime, np.datetime64)):
            return fmt_ymdhm(pd.to_datetime(x).to_pydatetime())
        xs = str(x)
        xs = _ascii_nfkc_clean(pd.Series([xs]))[0]  # cleanup control chars/fullwidth
        if _TIME_PAT.match(xs):
            dt = pd.to_datetime(xs, errors="coerce")
            if pd.isna(dt):
                return xs  # keep; will fail validation if truly bad
            return fmt_ymdhm(pd.to_datetime(dt).to_pydatetime())
        # fallback: try generic parse
        dt = pd.to_datetime(xs, errors="coerce")
        if pd.isna(dt):
            return xs
        return fmt_ymdhm(pd.to_datetime(dt).to_pydatetime())
    return s.map(norm_one)

# -------------- main proc --------------
def generate_augmented_csv_like_input(
    src: str,
    out: str,
    years: List[int],
    seed: Optional[int] = None,
    verbose: bool = True,
    *,
    bias_near_cut: Optional[str] = None,   # "YYYY-MM-DD"
    bias_end_of_year: bool = False,
    bias_window_days: int = 120,
    validate: bool = True,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    df = pd.read_csv(src, low_memory=False)

    # Resolve exact time column names from *original* CSV (robust to unicode noise)
    cre_col, rcv_col, mod_col, dsc_col = _resolve_time_cols(df)
    if verbose:
        print(f"[cols] creation={cre_col!r} time_received={rcv_col!r} lastmodified={mod_col!r} discovery={dsc_col!r}")

    # Preserve original order, but ensure time columns we will write exist in the final order
    col_order = list(df.columns)
    col_order_out = col_order.copy()
    for c in (cre_col, rcv_col, mod_col, dsc_col):
        if c not in col_order_out:
            col_order_out.append(c)

    # Parse baseline discovery-time distribution from cleaned creation/time_received
    creation_dt = _parse_ymdhm_loose(df[cre_col]) if cre_col in df.columns else pd.Series(pd.NaT, index=df.index)
    recv_dt     = _parse_ymdhm_loose(df[rcv_col]) if rcv_col in df.columns else pd.Series(pd.NaT, index=df.index)
    base_dt     = creation_dt.fillna(recv_dt)
    year_series = base_dt.dt.year

    # discoverydate raw choices were previously used to mimic original style.
    # With MJD output requested, we ignore original styles for injected rows.

    # redshift pools
    pools = {
        0: df[(df["redshift"] > BINS[0][0]) & (df["redshift"] <= BINS[0][1])].copy(),
        1: df[(df["redshift"] > BINS[1][0]) & (df["redshift"] <= BINS[1][1])].copy(),
        2: df[(df["redshift"] > BINS[2][0]) & (df["redshift"] <= BINS[2][1])].copy(),
    }
    fallback_pool = df[df["redshift"].apply(lambda x: isinstance(x, (int, float, np.floating)) and np.isfinite(x))]
    if fallback_pool.empty:
        raise ValueError("No finite redshift values; cannot synthesize.")

    # uniqueness sets
    existing_objids = set(pd.to_numeric(df["objid"], errors="coerce").dropna().astype(int).tolist())
    existing_names  = set(df["name"].astype(str).tolist())

    years_in_data = set(year_series.dropna().astype(int).tolist())
    years_all = sorted(set(years_in_data).union(set(years)))

    cut_dt = None
    if bias_near_cut:
        cut_dt = datetime.strptime(bias_near_cut, "%Y-%m-%d")

    # ---------------------------
    # Pass 1: scan deficits (progress bar)
    # ---------------------------
    year_deficits: Dict[int, List[int]] = {}
    total_inject = 0
    for y in years_all:
        curr = df.loc[(year_series == y)]
        counts = [
            ((curr["redshift"] > BINS[0][0]) & (curr["redshift"] <= BINS[0][1])).sum(),
            ((curr["redshift"] > BINS[1][0]) & (curr["redshift"] <= BINS[1][1])).sum(),
            ((curr["redshift"] > BINS[2][0]) & (curr["redshift"] <= BINS[2][1])).sum(),
        ]
        deficits = [max(0, BIN_TARGETS[i] - int(counts[i])) for i in range(3)]
        year_deficits[y] = deficits
        total_inject += sum(deficits)
        if verbose:
            print(f"[scan year {y}] have={counts} need={deficits}")
    if verbose:
        print(f"[scan total] will inject: {total_inject} rows")

    # progress bar
    pbar = tqdm(total=total_inject, desc="Injecting fake SNe", unit="row") if (tqdm and total_inject > 0) else None

    # ---------------------------
    # Pass 2: inject
    # ---------------------------
    new_rows: List[Dict] = []
    for y in years_all:
        deficits = year_deficits[y]
        for i, need in enumerate(deficits):
            if need == 0:
                continue
            base_pool = pools[i] if not pools[i].empty else fallback_pool
            base_pool = base_pool.reset_index(drop=True)
            idxs = np.random.randint(0, len(base_pool), size=need)
            new_objids = make_objids(existing_objids, need, prefix="9999", total_len=None)

            if not pools[i].empty and pools[i]["redshift"].notna().any():
                z_samples = pools[i]["redshift"].dropna().values
                use_empirical = True
            else:
                z_low, z_high = BINS[i]
                use_empirical = False

            for j in range(need):
                base = base_pool.iloc[idxs[j]]
                row = {c: base[c] if c in base else np.nan for c in col_order_out}

                # unique ids/names
                oid = int(new_objids[j])
                name = make_name(y)
                while name in existing_names:
                    name = make_name(y)

                row["objid"] = oid
                row["name"] = name
                row["internal_names"] = name

                # redshift
                if use_empirical:
                    row["redshift"] = float(np.random.choice(z_samples))
                else:
                    row["redshift"] = float(np.random.uniform(z_low + 1e-4, z_high - 1e-4))

                # choose date
                dt = sample_month_day_time(
                    y,
                    base_dt=base_dt,
                    bias_near_cut=cut_dt if (cut_dt and cut_dt.year == y) else None,
                    bias_end_of_year=bias_end_of_year,
                    bias_window_days=int(bias_window_days),
                )

                # write time columns (MJD, high precision)
                row[dsc_col] = _dt_to_mjd_str(dt)
                row[cre_col] = _dt_to_mjd_str(dt)
                row[rcv_col] = _dt_to_mjd_str(dt)
                row[mod_col] = _dt_to_mjd_str(dt + timedelta(days=random.randint(0, 10)))

                # group info
                row["source_group"] = FAKE_SOURCE_GROUP
                if "source_groupid" in df.columns:
                    src_dtype = df["source_groupid"].dtype
                    if np.issubdtype(src_dtype, np.integer):
                        row["source_groupid"] = int(FAKE_SOURCE_GROUPID_INT)
                    elif np.issubdtype(src_dtype, np.floating):
                        row["source_groupid"] = float(FAKE_SOURCE_GROUPID_INT)
                    else:
                        row["source_groupid"] = str(FAKE_SOURCE_GROUPID_INT)

                existing_objids.add(oid)
                existing_names.add(name)
                new_rows.append(row)

                if pbar:
                    pbar.update(1)

    if pbar:
        pbar.close()

    # ---- build injected block (times already MJD strings) ----
    inj_df = pd.DataFrame(new_rows)
    # Ensure missing columns exist (if any)
    for c in (cre_col, rcv_col, mod_col, dsc_col):
        if c not in inj_df.columns:
            inj_df[c] = np.nan

    # concat back, keep original order + appended time cols
    out_df = pd.concat([df, inj_df], ignore_index=True)
    out_df = out_df[[c for c in col_order_out if c in out_df.columns] + [c for c in out_df.columns if c not in col_order_out]]

    if validate:
        # Re-resolve and validate against the actual creation column
        cre_col_out, _, _, _ = _resolve_time_cols(out_df)
        # Identify injected rows robustly: prefer explicit source_group flag.
        if "source_group" in out_df.columns:
            inj = out_df["source_group"].astype(str) == FAKE_SOURCE_GROUP
        else:
            # Fallback: prefix match (least preferred; may catch originals)
            inj = out_df["objid"].astype(str).str.startswith("9999")

        vals = out_df.loc[inj, cre_col_out]
        # Parse as numeric MJD first
        num = pd.to_numeric(vals, errors="coerce")
        ok_mjd = num.between(30000, 90000, inclusive="both") & num.notna()
        bad_mask = pd.Series(True, index=vals.index)
        # Consider all in-range numerics as valid MJD
        bad_mask.loc[ok_mjd[ok_mjd].index] = False

        # Fallback to strict fixed-format parse (legacy) for any leftovers
        rest = bad_mask & vals.notna()
        if rest.any():
            inj_str = _ascii_nfkc_clean(vals.astype(str))
            parsed = pd.to_datetime(inj_str, format="%Y/%m/%d %H:%M", utc=True, errors="coerce")
            good2 = parsed.notna()
            bad_mask.loc[good2[good2].index] = False

        bad = int(bad_mask.sum())
        if bad:
            bad_samples = vals[bad_mask].head(10).astype(str).tolist()
            # Do not block; only warn and continue as requested.
            print(
                f"[validate:warn] {bad} injected rows have time in {cre_col_out!r} that is neither valid MJD nor strict 'YYYY/MM/DD HH:MM'; examples: {bad_samples}"
            )

    out_df.to_csv(out, index=False)

    # print summaries
    # Robust summary year extraction: handle both legacy strings and MJD
    def _to_year_mixed(s: pd.Series) -> pd.Series:
        if s is None or s.empty:
            return pd.Series(dtype=float)
        num = pd.to_numeric(s, errors="coerce")
        years = pd.Series(np.nan, index=s.index)
        mask_mjd = num.between(30000, 90000, inclusive="both") & num.notna()
        if mask_mjd.any():
            try:
                dt = pd.to_datetime(Time(num[mask_mjd].values, format="mjd").to_datetime(_dt_timezone.utc))
                years.loc[mask_mjd] = dt.dt.year.astype(float)
            except Exception:
                pass
        mask_rest = ~mask_mjd
        if mask_rest.any():
            dt2 = pd.to_datetime(s[mask_rest], errors="coerce", utc=True)
            if hasattr(dt2, "dt"):
                years.loc[mask_rest] = dt2.dt.year.astype(float)
        return years

    before_year = _to_year_mixed(df[cre_col]) if cre_col in df.columns else pd.Series(dtype=float)
    after_year  = _to_year_mixed(out_df[cre_col_out]) if cre_col_out in out_df.columns else pd.Series(dtype=float)
    print("\n=== SUMMARY BEFORE ===")
    print(summarize_by_year(df, before_year))
    print("\n=== SUMMARY AFTER ===")
    print(summarize_by_year(out_df, after_year))

def parse_args():
    ap = argparse.ArgumentParser(description="Inject fake SNe while preserving input CSV style and writing planner-safe time columns.")
    ap.add_argument("--src", type=str, default="/Users/tz/Documents/GitHub/Lsst-Twilight-SNe/data/ATLAS_2021_to25_cleaned.csv", help="Input CSV path")
    ap.add_argument("--out", type=str, default="/Users/tz/Documents/GitHub/Lsst-Twilight-SNe/data/ATLAS_2021_to25_with_fakes_like_input.csv", help="Output CSV path")
    ap.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025], help="Years to target")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    # biasing knobs
    ap.add_argument("--bias-near-cut", type=str, default=None, help="YYYY-MM-DD; cluster injected times within --bias-window-days before this night")
    ap.add_argument("--bias-end-of-year", action="store_true", help="Cluster injected times within --bias-window-days before Dec 31 of each year")
    ap.add_argument("--bias-window-days", type=int, default=120, help="Window size for clustering (days)")
    ap.add_argument("--no-validate", action="store_true", help="Skip post-write parse validation")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generate_augmented_csv_like_input(
        src=args.src,
        out=args.out,
        years=args.years,
        seed=args.seed,
        bias_near_cut=args.bias_near_cut,
        bias_end_of_year=args.bias_end_of_year,
        bias_window_days=args.bias_window_days,
        validate=not args.no_validate,
    )
