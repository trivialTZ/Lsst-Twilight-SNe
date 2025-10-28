"""Quality-control utilities for twilight cosmology analysis.

Add FITRES-only QC mode:
- Use FIT-level thresholds (FITPROB, x1, c, PKMJDERR, x1ERR, cERR).
- Replace PHOT-based sampling with FITRES proxies (SNRMAX1/2/3, SNRSUM, PKMJDERR).
- Skip saturation in fitres_only mode.
- Optionally annotate per-step QC boolean columns for audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits

# -----------------------------------------------------------------------------
# Saturation helpers
# -----------------------------------------------------------------------------


def discover_sat_mask_from_headers(phot_path: Path) -> int | None:
    """Return ``PHOTFLAG_SATURATE`` mask from FITS headers if present."""
    try:
        with fits.open(phot_path) as hdus:
            for h in hdus:
                if "PHOTFLAG_SATURATE" in h.header:
                    return int(h.header["PHOTFLAG_SATURATE"])
    except Exception:
        print("no PHOTFLAG_SATURATE")
        pass
    return None


def epoch_saturation_mask_df(
    phot_df: pd.DataFrame, phot_path: Path | None = None
) -> np.ndarray:
    """Epoch-level saturation boolean mask for a PHOT table."""
    for c in ("FLUXCAL", "FLUXCALERR", "FLUX", "FLUXERR", "PHOTFLAG"):
        if c in phot_df.columns:
            phot_df[c] = pd.to_numeric(phot_df[c], errors="coerce")
    if "PHOTFLAG" in phot_df.columns and phot_path is not None:
        sat_mask = discover_sat_mask_from_headers(phot_path)
        if sat_mask is not None:
            return (phot_df["PHOTFLAG"].fillna(0).astype(np.int64) & int(sat_mask)) > 0
    if {"FLUXCAL", "FLUXCALERR"}.issubset(phot_df.columns):
        return (phot_df["FLUXCAL"].abs() < 1e-6) & (phot_df["FLUXCALERR"].abs() > 1e7)
    if {"FLUX", "FLUXERR"}.issubset(phot_df.columns):
        return (phot_df["FLUX"].abs() < 1e-6) & (phot_df["FLUXERR"].abs() > 1e7)
    return np.zeros(len(phot_df), dtype=bool)


def _slice_indices_from_pointers(pmin: int, pmax: int, n_rows: int) -> tuple[int, int]:
    pmin = int(pmin) if pd.notna(pmin) else 0
    pmax = int(pmax) if pd.notna(pmax) else -1
    a = max(0, pmin - 1)
    b = min(n_rows, pmax)
    if b > a:
        return a, b
    a = max(0, pmin)
    b = min(n_rows, pmax + 1)
    if b > a:
        return a, b
    return 0, 0


def nosat_mask_for_run(
    fit_df: pd.DataFrame,
    head_df: pd.DataFrame,
    phot_df: pd.DataFrame,
    phot_path: Path | None = None,
    rest_window: Tuple[float, float] | None = None,
) -> pd.Series:
    """Return boolean mask True if SN has no saturated epochs in window."""
    ok = pd.Series(True, index=fit_df.index, dtype=bool)
    if phot_df is None or phot_df.empty or head_df is None or head_df.empty:
        return ok
    sat_epoch = epoch_saturation_mask_df(phot_df, phot_path=phot_path)
    n_rows = len(phot_df)
    need_cols = {"ID_int", "PTROBS_MIN", "PTROBS_MAX"}
    if not need_cols.issubset(head_df.columns):
        return ok
    head_idx = (
        head_df[list(need_cols)]
        .dropna()
        .astype({"PTROBS_MIN": "Int64", "PTROBS_MAX": "Int64"})
        .set_index("ID_int")
    )
    for i, row in fit_df.iterrows():
        sid = row.get("ID_int", pd.NA)
        if pd.isna(sid) or sid not in head_idx.index:
            continue
        pmin = head_idx.at[sid, "PTROBS_MIN"]
        pmax = head_idx.at[sid, "PTROBS_MAX"]
        a, b = _slice_indices_from_pointers(pmin, pmax, n_rows)
        if b <= a:
            continue
        if rest_window is not None and {"MJD"}.issubset(phot_df.columns):
            z = pd.to_numeric(row.get("z", np.nan), errors="coerce")
            pk = pd.to_numeric(row.get("PKMJD", np.nan), errors="coerce")
            if np.isfinite(z) and np.isfinite(pk):
                mjd_col = phot_df.columns.get_loc("MJD")
                t_rest = (phot_df.iloc[a:b, mjd_col].to_numpy() - pk) / (1.0 + z)
                mwin = (t_rest >= rest_window[0]) & (t_rest <= rest_window[1])
                sat_slice = sat_epoch[a:b]
                if np.any(sat_slice & mwin):
                    ok.at[i] = False
                continue
        if np.any(sat_epoch[a:b]):
            ok.at[i] = False
    return ok


def _sn_pm10_snr5_ok_for_run(
    fit_df: pd.DataFrame,
    head_df: pd.DataFrame,
    phot_df: pd.DataFrame,
    phase_days: Tuple[float, float] = (-10.0, 10.0),
    snr_thresh: float = 5.0,
) -> pd.Series:
    ok = pd.Series(False, index=fit_df.index, dtype=bool)
    if phot_df is None or phot_df.empty:
        return ok
    if {"FLUXCAL", "FLUXCALERR"}.issubset(phot_df.columns):
        flux_col, err_col = "FLUXCAL", "FLUXCALERR"
    elif {"FLUX", "FLUXERR"}.issubset(phot_df.columns):
        flux_col, err_col = "FLUX", "FLUXERR"
    else:
        return ok
    if not {"ID_int", "PTROBS_MIN", "PTROBS_MAX"}.issubset(head_df.columns):
        return ok
    head_idx = (
        head_df[["ID_int", "PTROBS_MIN", "PTROBS_MAX"]]
        .dropna()
        .astype({"PTROBS_MIN": "Int64", "PTROBS_MAX": "Int64"})
        .set_index("ID_int")
    )
    for i, row in fit_df.iterrows():
        sid = row.get("ID_int", pd.NA)
        if pd.isna(sid) or sid not in head_idx.index:
            continue
        pmin = head_idx.at[sid, "PTROBS_MIN"]
        pmax = head_idx.at[sid, "PTROBS_MAX"]
        if pd.isna(pmin) or pd.isna(pmax):
            continue
        lo = max(int(pmin) - 1, 0)
        hi = min(int(pmax), len(phot_df))
        if hi <= lo:
            continue
        sl = phot_df.iloc[lo:hi]
        z = row.get("z", np.nan)
        pk = row.get("PKMJD", np.nan)
        if not (np.isfinite(z) and np.isfinite(pk)) or sl.empty:
            continue
        snr = sl[flux_col].to_numpy() / np.maximum(sl[err_col].to_numpy(), 1e-9)
        t_rest = (sl["MJD"].to_numpy() - pk) / (1.0 + z)
        m = (t_rest >= phase_days[0]) & (t_rest <= phase_days[1]) & np.isfinite(snr)
        if np.count_nonzero(snr[m] > snr_thresh) >= 3:
            ok.loc[i] = True
    return ok


def densest_year_window(head: pd.DataFrame) -> Tuple[float, float]:
    """Return ``(t0, t1)`` MJD for densest ~1-year window based on ``PKMJD``."""
    mjd = pd.to_numeric(head.get("PKMJD", head.get("PEAKMJD")), errors="coerce")
    mjd = np.sort(mjd[np.isfinite(mjd)])
    width = 365.25
    if mjd.size == 0:
        return np.nan, np.nan
    j0 = 0
    best = (mjd[0], mjd[0] + width, 1)
    for j1 in range(mjd.size):
        while mjd[j1] - mjd[j0] > width:
            j0 += 1
        cnt = j1 - j0 + 1
        if cnt > best[2]:
            best = (mjd[j0], mjd[j0] + width, cnt)
    return best[0], best[1]


# -----------------------------------------------------------------------------
# FITRES-only sampling proxy
# -----------------------------------------------------------------------------
def fitres_proxy_sampling_mask(
    fit_df: pd.DataFrame,
    snrmax_min: float = 5.0,
    snrsum_min: Optional[float] = 20.0,
    require_pkmjderr: bool = True,
    pkmjderr_max: float = 1.0,
) -> pd.Series:
    """
    Build a sampling-quality proxy using only FITRES columns.
    Conditions (applied if corresponding columns exist):
      - max(SNRMAX1/2/3) >= snrmax_min
      - SNRSUM >= snrsum_min (if snrsum_min is not None)
      - PKMJDERR <= pkmjderr_max (if require_pkmjderr)
    """
    m = pd.Series(True, index=fit_df.index)
    # SNRMAX top-3
    snr_cols = [c for c in ("SNRMAX1", "SNRMAX2", "SNRMAX3") if c in fit_df.columns]
    if snr_cols:
        m &= fit_df[snr_cols].max(axis=1) >= float(snrmax_min)
    # SNRSUM
    if ("SNRSUM" in fit_df.columns) and (snrsum_min is not None):
        m &= pd.to_numeric(fit_df["SNRSUM"], errors="coerce") >= float(snrsum_min)
    # PKMJDERR
    if require_pkmjderr and ("PKMJDERR" in fit_df.columns):
        m &= pd.to_numeric(fit_df["PKMJDERR"], errors="coerce") <= float(pkmjderr_max)
    return m.fillna(False)


def ross_qc_with_report(
    fit_all: pd.DataFrame,
    *,
    head_hi: Optional[pd.DataFrame] = None,
    phot_hi: Optional[pd.DataFrame] = None,
    head_lo: Optional[pd.DataFrame] = None,
    phot_lo: Optional[pd.DataFrame] = None,
    phot_hi_path: Optional[Path] = None,
    phot_lo_path: Optional[Path] = None,
    # SALT2 / fit-level thresholds (None → skip that cut)
    fitprob_min: float | None = 0.05,
    x1_abs_max: float | None = 3.0,
    c_abs_max: float | None = 0.3,
    pkmjderr_max: float | None = 1.0,
    x1err_max: float | None = 1.0,
    cerr_max: float | None = 0.05,
    # Light-curve sampling checks
    phase_days: Tuple[float, float] = (-10.0, 10.0),
    snr_thresh: float = 5.0,
    require_pm10_snr5: bool = True,
    verbose: bool = False,
    # Mode & toggles
    fitres_only: bool = False,
    check_sampling: bool = True,
    check_saturation: bool = True,
    add_qc_columns: bool = True,
    # FITRES proxy thresholds (FITRES-only mode)
    fitres_snrmax_min: float = 5.0,
    fitres_snrsum_min: Optional[float] = 20.0,
    fitres_require_pkmjderr: bool = True,
) -> pd.DataFrame:
    """Apply Rosselli-style QC on FITRES using PHOT/HEAD checks.

    Parameters
    ----------
    fit_all : pandas.DataFrame
        All FITRES rows to quality-control.
    head_hi, phot_hi : pandas.DataFrame
        HEAD and PHOT tables for the high-redshift run.
    head_lo, phot_lo : pandas.DataFrame
        HEAD and PHOT tables for the low-redshift run.
    phot_hi_path, phot_lo_path : Path or None, optional
        Original PHOT FITS paths used to discover ``PHOTFLAG_SATURATE``
        header bits. If omitted, saturation is inferred from flux values.
    phase_days : tuple of float, optional
        Rest-frame phase window for saturation/SNR checks.
    snr_thresh : float, optional
        S/N threshold for the ``>=3 obs ±10d`` requirement.
    require_pm10_snr5 : bool, optional
        Enforce the ``>=3 obs ±10d`` cut if True.
    verbose : bool, optional
        Print keep counts for each cut.

    Returns
    -------
    pandas.DataFrame
        FITRES rows passing all quality cuts. If ``add_qc_columns=True``,
        includes per-step audit columns (boolean with possible pd.NA for skipped).
    """
    N0 = len(fit_all)
    # Helper to track cumulative keeps
    m_all = pd.Series(True, index=fit_all.index)

    def keep_and_print(mask: pd.Series, tag: str) -> None:
        nonlocal m_all
        m_all &= mask.fillna(False)
        if verbose:
            kept = int(m_all.sum())
            print(f"[ROSS QC] {tag:<18} keep={kept:6d}/{N0:6d} ({kept/N0:5.1%})")

    # Prepare audit columns (pandas nullable boolean to allow pd.NA)
    qc_cols: dict[str, pd.Series] = {
        "QC_fitprob": pd.Series(pd.NA, index=fit_all.index, dtype="boolean"),
        "QC_x1": pd.Series(pd.NA, index=fit_all.index, dtype="boolean"),
        "QC_c": pd.Series(pd.NA, index=fit_all.index, dtype="boolean"),
        "QC_pkmjderr": pd.Series(pd.NA, index=fit_all.index, dtype="boolean"),
        "QC_x1err": pd.Series(pd.NA, index=fit_all.index, dtype="boolean"),
        "QC_cerr": pd.Series(pd.NA, index=fit_all.index, dtype="boolean"),
        "QC_sampling_pm10snr5": pd.Series(pd.NA, index=fit_all.index, dtype="boolean"),
        "QC_sampling_fitres_proxy": pd.Series(pd.NA, index=fit_all.index, dtype="boolean"),
        "QC_nosat": pd.Series(pd.NA, index=fit_all.index, dtype="boolean"),
    }

    # ---------------------------
    # FIT-level cuts (FITRES only)
    # ---------------------------
    # FITPROB
    if (fitprob_min is not None) and ("FITPROB" in fit_all.columns):
        m_fitprob = pd.to_numeric(fit_all["FITPROB"], errors="coerce") > float(
            fitprob_min
        )
        qc_cols["QC_fitprob"] = m_fitprob.astype("boolean")
        keep_and_print(m_fitprob, f"FITPROB>{float(fitprob_min):.2f}")
    else:
        if verbose:
            print("[ROSS QC] skip FITPROB cut (column missing or threshold None)")
        qc_cols["QC_fitprob"] = pd.Series(pd.NA, index=fit_all.index, dtype="boolean")

    # |x1|
    if (x1_abs_max is not None) and ("x1" in fit_all.columns):
        m_x1 = (
            np.abs(pd.to_numeric(fit_all["x1"], errors="coerce"))
            <= float(x1_abs_max)
        )
        qc_cols["QC_x1"] = m_x1.astype("boolean")
        keep_and_print(m_x1, f"|x1|<={float(x1_abs_max):.2f}")
    else:
        if verbose:
            print("[ROSS QC] skip x1 cut (column missing or threshold None)")
        qc_cols["QC_x1"] = pd.Series(pd.NA, index=fit_all.index, dtype="boolean")

    # |c|
    if (c_abs_max is not None) and ("c" in fit_all.columns):
        m_c = (
            np.abs(pd.to_numeric(fit_all["c"], errors="coerce"))
            <= float(c_abs_max)
        )
        qc_cols["QC_c"] = m_c.astype("boolean")
        keep_and_print(m_c, f"|c|<={float(c_abs_max):.2f}")
    else:
        if verbose:
            print("[ROSS QC] skip c cut (column missing or threshold None)")
        qc_cols["QC_c"] = pd.Series(pd.NA, index=fit_all.index, dtype="boolean")

    # PKMJDERR
    if (pkmjderr_max is not None) and ("PKMJDERR" in fit_all.columns):
        m_pkmjd = (
            pd.to_numeric(fit_all["PKMJDERR"], errors="coerce")
            <= float(pkmjderr_max)
        )
        qc_cols["QC_pkmjderr"] = m_pkmjd.astype("boolean")
        keep_and_print(m_pkmjd, f"PKMJDERR<={float(pkmjderr_max):.1f}d")
    else:
        if verbose:
            print("[ROSS QC] skip PKMJDERR cut (column missing or threshold None)")
        qc_cols["QC_pkmjderr"] = pd.Series(pd.NA, index=fit_all.index, dtype="boolean")

    # x1ERR
    if (x1err_max is not None) and ("x1ERR" in fit_all.columns):
        m_x1e = (
            pd.to_numeric(fit_all["x1ERR"], errors="coerce")
            <= float(x1err_max)
        )
        qc_cols["QC_x1err"] = m_x1e.astype("boolean")
        keep_and_print(m_x1e, f"x1ERR<={float(x1err_max):.2f}")
    else:
        if verbose:
            print("[ROSS QC] skip x1ERR cut (column missing or threshold None)")
        qc_cols["QC_x1err"] = pd.Series(pd.NA, index=fit_all.index, dtype="boolean")

    # cERR
    if (cerr_max is not None) and ("cERR" in fit_all.columns):
        m_ce = (
            pd.to_numeric(fit_all["cERR"], errors="coerce")
            <= float(cerr_max)
        )
        qc_cols["QC_cerr"] = m_ce.astype("boolean")
        keep_and_print(m_ce, f"cERR<={float(cerr_max):.2f}")
    else:
        if verbose:
            print("[ROSS QC] skip cERR cut (column missing or threshold None)")
        qc_cols["QC_cerr"] = pd.Series(pd.NA, index=fit_all.index, dtype="boolean")

    # ------------------------------------
    # Sampling & saturation (mode dependent)
    # ------------------------------------
    if fitres_only:
        # FITRES-only: optional proxy sampling; saturation skipped
        if check_sampling:
            m_proxy = fitres_proxy_sampling_mask(
                fit_all,
                snrmax_min=fitres_snrmax_min,
                snrsum_min=fitres_snrsum_min,
                require_pkmjderr=fitres_require_pkmjderr,
                pkmjderr_max=float(pkmjderr_max) if pkmjderr_max is not None else 1.0,
            )
            qc_cols["QC_sampling_fitres_proxy"] = m_proxy.astype("boolean")
            keep_and_print(m_proxy, "FITRES proxy sampling")
        else:
            # sampling step disabled
            qc_cols["QC_sampling_fitres_proxy"] = pd.Series(
                pd.NA, index=fit_all.index, dtype="boolean"
            )
        if check_saturation and verbose:
            print("[ROSS QC] SKIP saturation: fitres_only=True")
        qc_cols["QC_nosat"] = pd.Series(pd.NA, index=fit_all.index, dtype="boolean")
    else:
        # Non FITRES-only: keep existing PHOT-based sampling & nosat logic (with toggles)
        ids_hi: set[int] = set()
        ids_lo: set[int] = set()
        fit_hi = fit_all.iloc[0:0]
        fit_lo = fit_all.iloc[0:0]
        if head_hi is not None:
            ids_hi = set(head_hi.get("ID_int", pd.Series(dtype="Int64")).dropna().astype("Int64"))
            fit_hi = fit_all[fit_all.get("ID_int").isin(ids_hi)] if "ID_int" in fit_all.columns else fit_all.iloc[0:0]
        if head_lo is not None:
            ids_lo = set(head_lo.get("ID_int", pd.Series(dtype="Int64")).dropna().astype("Int64"))
            fit_lo = fit_all[fit_all.get("ID_int").isin(ids_lo)] if "ID_int" in fit_all.columns else fit_all.iloc[0:0]

        if check_sampling and require_pm10_snr5:
            obs_ok_hi = (
                _sn_pm10_snr5_ok_for_run(
                    fit_hi, head_hi, phot_hi, phase_days, snr_thresh
                )
                if (head_hi is not None and phot_hi is not None)
                else pd.Series(False, index=fit_hi.index)
            )
            obs_ok_lo = (
                _sn_pm10_snr5_ok_for_run(
                    fit_lo, head_lo, phot_lo, phase_days, snr_thresh
                )
                if (head_lo is not None and phot_lo is not None)
                else pd.Series(False, index=fit_lo.index)
            )
            m_obs = pd.Series(False, index=fit_all.index)
            m_obs.loc[fit_hi.index] = obs_ok_hi.values
            m_obs.loc[fit_lo.index] = obs_ok_lo.values
            qc_cols["QC_sampling_pm10snr5"] = m_obs.astype("boolean")
            keep_and_print(m_obs, ">=3 obs ±10d")
        else:
            qc_cols["QC_sampling_pm10snr5"] = pd.Series(
                pd.NA, index=fit_all.index, dtype="boolean"
            )

        if check_saturation:
            nosat_hi = (
                nosat_mask_for_run(
                    fit_hi, head_hi, phot_hi, phot_path=phot_hi_path, rest_window=phase_days
                )
                if (head_hi is not None and phot_hi is not None)
                else pd.Series(True, index=fit_hi.index)
            )
            nosat_lo = (
                nosat_mask_for_run(
                    fit_lo, head_lo, phot_lo, phot_path=phot_lo_path, rest_window=phase_days
                )
                if (head_lo is not None and phot_lo is not None)
                else pd.Series(True, index=fit_lo.index)
            )
            m_nosat = pd.Series(False, index=fit_all.index)
            m_nosat.loc[fit_hi.index] = nosat_hi.values
            m_nosat.loc[fit_lo.index] = nosat_lo.values
            qc_cols["QC_nosat"] = m_nosat.astype("boolean")
            keep_and_print(m_nosat, "no saturation")
        else:
            qc_cols["QC_nosat"] = pd.Series(pd.NA, index=fit_all.index, dtype="boolean")

    kept = int(m_all.sum())
    if verbose:
        print(f"[ROSS QC] TOTAL          keep={kept:6d}/{N0:6d} ({kept/N0:5.1%})")
    out = fit_all.loc[m_all].copy()
    if add_qc_columns:
        for k, v in qc_cols.items():
            # Reindex to keep only rows we kept
            out[k] = v.reindex(out.index)
    return out

# Example usage (FITRES-only mode):
# fit_qc = ross_qc_with_report(
#     fit_all,
#     head_hi=None, phot_hi=None, head_lo=None, phot_lo=None,
#     fitres_only=True,
#     fitres_snrmax_min=5.0, fitres_snrsum_min=20.0,
#     fitres_require_pkmjderr=True,
#     add_qc_columns=True, verbose=True,
# )
