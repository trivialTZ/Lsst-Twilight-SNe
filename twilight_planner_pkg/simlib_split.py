"""Post-process twilight planner outputs into SIMLIBs.

This module exists for the "planner is slow" workflow: once you already have
the per-visit CSV (or DataFrame) written by the planner, you can generate one
or more SIMLIB files without re-running the scheduler.

Typical use for the WFD+TNS notebook
-----------------------------------
- Run the planner once on a combined catalog.
- Use the combined catalog's ``source`` column to split the twilight plan into:
  1) a TNS-only twilight SIMLIB
  2) a WFD-linked twilight SIMLIB (optionally preserving WFD LIBIDs)
"""

from __future__ import annotations

from pathlib import Path
import gzip
import math
from typing import Optional, Tuple

import pandas as pd

from .config import PlannerConfig
from .simlib_writer import SimlibHeader, SimlibWriter


def simlib_header_from_config(cfg: PlannerConfig) -> SimlibHeader:
    """Build a :class:`~twilight_planner_pkg.simlib_writer.SimlibHeader` from config."""

    npe = getattr(cfg, "simlib_npe_pixel_saturate", 80000)
    try:
        npe_int = int(round(float(npe)))
    except Exception:
        npe_int = 80000
    return SimlibHeader(
        SURVEY=str(getattr(cfg, "simlib_survey", "LSST")),
        FILTERS=str(getattr(cfg, "simlib_filters", "grizy")),
        PIXSIZE=float(getattr(cfg, "simlib_pixsize", 0.2)),
        NPE_PIXEL_SATURATE=npe_int,
        PHOTFLAG_SATURATE=int(getattr(cfg, "simlib_photflag_saturate", 2048)),
        PSF_UNIT=str(getattr(cfg, "simlib_psf_unit", "PIXEL")),
    )


def _as_float(val) -> Optional[float]:
    try:
        out = float(val)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _timestamp_to_mjd(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return float(ts.to_julian_date() - 2400000.5)
    except Exception:
        return None


def _normalize_name(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    name = str(val).strip()
    return name if name else None


def _infer_name_col(df: pd.DataFrame) -> str:
    for col in ("SN", "Name", "name"):
        if col in df.columns:
            return col
    raise ValueError("Could not find a name column in plan df (expected one of: SN, Name, name)")


def _row_to_epoch(row: pd.Series, *, default_zpt_err_mag: float) -> Optional[dict]:
    mjd = None
    for col in ("visit_start_utc", "sn_start_utc", "best_twilight_time_utc"):
        if col in row:
            mjd = _timestamp_to_mjd(row[col])
            if mjd is not None:
                break
    if mjd is None:
        return None

    band_val = None
    for col in ("filter", "band", "FLT"):
        if col in row and row[col] is not None:
            band_val = str(row[col]).strip()
            break
    if not band_val:
        return None
    band = band_val.lower()
    if band == "y":
        band = "y"

    gain = _as_float(row.get("GAIN", row.get("gain")))
    noise = _as_float(row.get("RDNOISE", row.get("read_noise_e")))
    skysig = _as_float(row.get("SKYSIG"))
    psf1 = _as_float(row.get("PSF1_pix"))
    psf2 = _as_float(row.get("PSF2_pix"))
    psfratio = _as_float(row.get("PSFRATIO"))
    zpavg = _as_float(row.get("ZPT"))

    zperr = _as_float(row.get("zpt_err_mag", row.get("ZPTERR")))
    if zperr is None:
        zperr = float(default_zpt_err_mag)

    nexpose = row.get("nexpose", 1)
    try:
        nexpose_int = int(nexpose)
    except Exception:
        nexpose_int = 1
    if nexpose_int < 1:
        nexpose_int = 1

    return {
        "mjd": mjd,
        "band": band,
        "gain": gain if gain is not None else 0.0,
        "noise": noise if noise is not None else 0.0,
        "skysig": skysig if skysig is not None else 0.0,
        "psf1": psf1 if psf1 is not None else 0.0,
        "psf2": psf2 if psf2 is not None else 0.0,
        "psfratio": psfratio if psfratio is not None else 0.0,
        "zpavg": zpavg if zpavg is not None else 0.0,
        "zperr": zperr,
        "mag": -99.0,
        "nexpose": nexpose_int,
    }


def _group_plan_epochs(
    df: pd.DataFrame,
    *,
    default_zpt_err_mag: float,
) -> Tuple[list[str], dict[str, dict]]:
    name_col = _infer_name_col(df)
    order: list[str] = []
    grouped: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = _normalize_name(row.get(name_col))
        if not name:
            continue
        if name not in grouped:
            grouped[name] = {
                "epochs": [],
                "ra": None,
                "dec": None,
                "redshift": None,
                "peakmjd": None,
                "libid_hint": None,
            }
            order.append(name)
        info = grouped[name]
        if info["libid_hint"] is None and "wfd_libid" in row and pd.notna(row["wfd_libid"]):
            try:
                info["libid_hint"] = int(float(row["wfd_libid"]))
            except Exception:
                info["libid_hint"] = None
        if info["ra"] is None and "RA_deg" in row and pd.notna(row["RA_deg"]):
            info["ra"] = _as_float(row["RA_deg"])
        if info["dec"] is None and "Dec_deg" in row and pd.notna(row["Dec_deg"]):
            info["dec"] = _as_float(row["Dec_deg"])
        if info["redshift"] is None and "redshift" in row and pd.notna(row["redshift"]):
            info["redshift"] = _as_float(row["redshift"])
        ep = _row_to_epoch(row, default_zpt_err_mag=default_zpt_err_mag)
        if ep:
            info["epochs"].append(ep)
    return order, grouped


def _open_text_write(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")


def write_twilight_simlib(
    twilight_plan_df: pd.DataFrame,
    out_path: str | Path,
    *,
    header: Optional[SimlibHeader] = None,
    preserve_ids: bool = False,
    default_zpt_err_mag: float = 0.01,
) -> Path:
    """Write a SIMLIB from the planner's per-visit output table.

    Parameters
    ----------
    twilight_plan_df:
        Per-visit DataFrame (e.g. ``perSN_df`` or the on-disk plan CSV).
    out_path:
        Output SIMLIB path. If it ends with ``.gz``, output is gzip-compressed.
    header:
        SIMLIB header. If ``None``, uses :class:`~twilight_planner_pkg.simlib_writer.SimlibHeader`
        defaults.
    preserve_ids:
        If True, preserve LIBIDs provided via the ``wfd_libid`` column (when present).
    default_zpt_err_mag:
        Used when the input does not contain ``zpt_err_mag`` or ``ZPTERR``.
    """

    header = header or SimlibHeader()
    order, grouped = _group_plan_epochs(
        twilight_plan_df, default_zpt_err_mag=default_zpt_err_mag
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp = _open_text_write(out_path)
    writer = SimlibWriter(fp, header, preserve_ids=preserve_ids)
    writer.write_header()

    libid_counter = 1
    for name in order:
        info = grouped[name]
        epochs = info.get("epochs") or []
        if not epochs:
            continue
        epochs_sorted = sorted(epochs, key=lambda e: (float(e["mjd"]), str(e["band"])))
        ra = info.get("ra")
        dec = info.get("dec")
        if ra is None or dec is None:
            raise ValueError(f"Missing RA/Dec for {name}; cannot write SIMLIB block")
        libid_val = libid_counter
        if preserve_ids and info.get("libid_hint") is not None:
            libid_val = int(info["libid_hint"])

        writer.start_libid(
            libid_val,
            float(ra),
            float(dec),
            len(epochs_sorted),
            comment=name,
            redshift=info.get("redshift"),
            peakmjd=info.get("peakmjd"),
        )
        for ep in epochs_sorted:
            writer.add_epoch(**ep)
        writer.end_libid()
        if not preserve_ids:
            libid_counter += 1
        else:
            libid_counter += 1
    writer.close()
    return out_path


def split_twilight_plan_by_source(
    twilight_plan_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    *,
    out_tns_path: str | Path,
    out_wfd_path: str | Path,
    header: Optional[SimlibHeader] = None,
    source_col: str = "source",
    catalog_name_col: str = "Name",
    tns_value: str = "TNS",
    wfd_value: str = "WFD",
    preserve_wfd_libids: bool = True,
    default_zpt_err_mag: float = 0.01,
    require_all_mapped: bool = True,
) -> tuple[Path, Path]:
    """Split a twilight plan into two SIMLIB files using a catalog ``source`` column."""

    if source_col not in catalog_df.columns:
        raise ValueError(f"catalog_df missing required column {source_col!r}")
    if catalog_name_col not in catalog_df.columns:
        raise ValueError(f"catalog_df missing required column {catalog_name_col!r}")

    plan_name_col = _infer_name_col(twilight_plan_df)
    cat = catalog_df.copy()
    cat["__name_norm"] = cat[catalog_name_col].astype("string").str.strip()
    cat = cat[cat["__name_norm"].notna()].copy()
    keep_cols = ["__name_norm", source_col]
    for extra in ("wfd_libid", "redshift", "RA_deg", "Dec_deg"):
        if extra in cat.columns:
            keep_cols.append(extra)
    cat = cat[keep_cols].drop_duplicates(subset=["__name_norm"], keep="first")

    plan = twilight_plan_df.copy()
    plan["__name_norm"] = plan[plan_name_col].astype("string").str.strip()
    merged = plan.merge(cat, on="__name_norm", how="left", suffixes=("", "_cat"))

    missing = merged[merged[source_col].isna()]["__name_norm"].dropna().unique().tolist()
    if missing and require_all_mapped:
        missing_preview = ", ".join(missing[:10])
        raise ValueError(
            f"{len(missing)} targets in twilight_plan_df had no match in catalog_df by name; "
            f"first few: {missing_preview}"
        )

    tns_df = merged[merged[source_col] == tns_value].copy()
    wfd_df = merged[merged[source_col] == wfd_value].copy()

    out_tns = write_twilight_simlib(
        tns_df,
        out_tns_path,
        header=header,
        preserve_ids=False,
        default_zpt_err_mag=default_zpt_err_mag,
    )
    out_wfd = write_twilight_simlib(
        wfd_df,
        out_wfd_path,
        header=header,
        preserve_ids=bool(preserve_wfd_libids),
        default_zpt_err_mag=default_zpt_err_mag,
    )
    return out_tns, out_wfd


def split_twilight_plan_csv_by_source(
    *,
    plan_csv: str | Path,
    catalog_csv: str | Path,
    out_tns_path: str | Path,
    out_wfd_path: str | Path,
    header: Optional[SimlibHeader] = None,
    **kwargs,
) -> tuple[Path, Path]:
    """File-path wrapper around :func:`split_twilight_plan_by_source`."""

    plan_df = pd.read_csv(plan_csv)
    cat_df = pd.read_csv(catalog_csv)
    return split_twilight_plan_by_source(
        plan_df,
        cat_df,
        out_tns_path=out_tns_path,
        out_wfd_path=out_wfd_path,
        header=header,
        **kwargs,
    )

