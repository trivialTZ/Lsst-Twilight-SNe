"""Quality-control utilities for twilight cosmology analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

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
                t_rest = (phot_df.loc[a:b, "MJD"].to_numpy() - pk) / (1.0 + z)
                mwin = (t_rest >= rest_window[0]) & (t_rest <= rest_window[1])
                if np.any(sat_epoch[a:b][mwin]):
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


def ross_qc_with_report(
    fit_all: pd.DataFrame,
    *,
    head_hi: pd.DataFrame,
    phot_hi: pd.DataFrame,
    head_lo: pd.DataFrame,
    phot_lo: pd.DataFrame,
    phot_hi_path: Path | None = None,
    phot_lo_path: Path | None = None,
    phase_days: Tuple[float, float] = (-10.0, 10.0),
    snr_thresh: float = 5.0,
    require_pm10_snr5: bool = True,
    verbose: bool = False,
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
        FITRES rows passing all quality cuts.
    """
    N0 = len(fit_all)

    def keep_and_print(mask: pd.Series, tag: str) -> pd.Series:
        if verbose:
            kept = int(mask.sum())
            print(f"[ROSS QC] {tag:<18} keep={kept:6d}/{N0:6d} ({kept/N0:5.1%})")
        return mask

    m_all = pd.Series(True, index=fit_all.index)
    m_fitprob = (
        fit_all["FITPROB"].fillna(0) > 0.05
        if "FITPROB" in fit_all.columns
        else pd.Series(True, index=fit_all.index)
    )
    keep_and_print(m_fitprob, "FITPROB cut")
    m_all &= m_fitprob
    m_x1 = np.abs(pd.to_numeric(fit_all.get("x1", np.nan), errors="coerce")) <= 3.0
    keep_and_print(m_x1, "x1 range")
    m_all &= m_x1
    m_c = np.abs(pd.to_numeric(fit_all.get("c", np.nan), errors="coerce")) <= 0.3
    keep_and_print(m_c, "c range")
    m_all &= m_c
    m_pkmjd = pd.to_numeric(fit_all.get("PKMJDERR", np.nan), errors="coerce") <= 1.0
    keep_and_print(m_pkmjd, "PKMJDERR<=1d")
    m_all &= m_pkmjd
    m_x1e = pd.to_numeric(fit_all.get("x1ERR", np.nan), errors="coerce") <= 1.0
    keep_and_print(m_x1e, "x1ERR<=1")
    m_all &= m_x1e
    m_ce = pd.to_numeric(fit_all.get("cERR", np.nan), errors="coerce") <= 0.05
    keep_and_print(m_ce, "cERR<=0.05")
    m_all &= m_ce
    ids_hi = set(head_hi["ID_int"].dropna().astype("Int64"))
    ids_lo = set(head_lo["ID_int"].dropna().astype("Int64"))
    fit_hi = fit_all[fit_all["ID_int"].isin(ids_hi)]
    fit_lo = fit_all[fit_all["ID_int"].isin(ids_lo)]
    if require_pm10_snr5:
        obs_ok_hi = _sn_pm10_snr5_ok_for_run(
            fit_hi, head_hi, phot_hi, phase_days, snr_thresh
        )
        obs_ok_lo = _sn_pm10_snr5_ok_for_run(
            fit_lo, head_lo, phot_lo, phase_days, snr_thresh
        )
        m_obs = pd.Series(False, index=fit_all.index)
        m_obs.loc[fit_hi.index] = obs_ok_hi.values
        m_obs.loc[fit_lo.index] = obs_ok_lo.values
        keep_and_print(m_obs, ">=3 obs ±10d")
        m_all &= m_obs
    nosat_hi = nosat_mask_for_run(
        fit_hi, head_hi, phot_hi, phot_path=phot_hi_path, rest_window=phase_days
    )
    nosat_lo = nosat_mask_for_run(
        fit_lo, head_lo, phot_lo, phot_path=phot_lo_path, rest_window=phase_days
    )
    m_nosat = pd.Series(False, index=fit_all.index)
    m_nosat.loc[fit_hi.index] = nosat_hi.values
    m_nosat.loc[fit_lo.index] = nosat_lo.values
    keep_and_print(m_nosat, "no saturation")
    m_all &= m_nosat
    kept = int(m_all.sum())
    if verbose:
        print(f"[ROSS QC] TOTAL          keep={kept:6d}/{N0:6d} ({kept/N0:5.1%})")
    return fit_all.loc[m_all].copy()
