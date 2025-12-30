"""Merge a WFD SIMLIB with twilight planner outputs."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Dict, List, Optional

import pandas as pd

from .simlib_reader import (
    SimlibDocument,
    SimlibEntry,
    SimlibEpoch,
    entry_name,
    parse_simlib,
)
from .simlib_writer import SimlibHeader, SimlibWriter


def _copy_entry(entry: SimlibEntry) -> SimlibEntry:
    """Return a deep-ish copy of a parsed SIMLIB entry."""

    epochs = [
        SimlibEpoch(
            mjd=e.mjd,
            band=e.band,
            id_expose=e.id_expose,
            gain=e.gain,
            noise=e.noise,
            skysig=e.skysig,
            psf1=e.psf1,
            psf2=e.psf2,
            psfratio=e.psfratio,
            zpavg=e.zpavg,
            zperr=e.zperr,
            mag=e.mag,
            nexpose=getattr(e, "nexpose", 1),
        )
        for e in entry.epochs
    ]
    return SimlibEntry(
        libid=entry.libid,
        ra_deg=entry.ra_deg,
        dec_deg=entry.dec_deg,
        nobs=entry.nobs,
        redshift=entry.redshift,
        peakmjd=entry.peakmjd,
        comment=entry.comment,
        match_tns_name=entry.match_tns_name,
        epochs=epochs,
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


def _row_to_epoch(row: pd.Series) -> Optional[SimlibEpoch]:
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
    band = ""
    if band_val:
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
    nexpose = row.get("nexpose", 1)
    try:
        nexpose_int = int(nexpose)
    except Exception:
        nexpose_int = 1
    if nexpose_int < 1:
        nexpose_int = 1
    return SimlibEpoch(
        mjd=mjd,
        band=band,
        gain=gain if gain is not None else 0.0,
        noise=noise if noise is not None else 0.0,
        skysig=skysig if skysig is not None else 0.0,
        psf1=psf1 if psf1 is not None else 0.0,
        psf2=psf2 if psf2 is not None else 0.0,
        psfratio=psfratio if psfratio is not None else 0.0,
        zpavg=zpavg if zpavg is not None else 0.0,
        zperr=zperr if zperr is not None else 0.0,
        nexpose=nexpose_int,
    )


def _group_twilight_epochs(df: pd.DataFrame) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = None
        for key in ("SN", "Name", "name"):
            if key in row and pd.notna(row[key]):
                name = str(row[key])
                break
        if not name:
            continue
        info = grouped.setdefault(
            name,
            {"epochs": [], "ra": None, "dec": None, "redshift": None, "peakmjd": None, "libid_hint": None},
        )
        if "wfd_libid" in row:
            libid_hint = _as_float(row["wfd_libid"])
            if libid_hint is not None:
                info["libid_hint"] = int(libid_hint)
        if info["ra"] is None and "RA_deg" in row:
            info["ra"] = _as_float(row["RA_deg"])
        if info["dec"] is None and "Dec_deg" in row:
            info["dec"] = _as_float(row["Dec_deg"])
        if info["redshift"] is None and "redshift" in row:
            info["redshift"] = _as_float(row["redshift"])
        ep = _row_to_epoch(row)
        if ep:
            info["epochs"].append(ep)
    return grouped


def _build_preamble(doc: SimlibDocument, source_path: Path, n_entries: int) -> List[str]:
    """Compose the header for the merged SIMLIB."""

    lines: List[str] = []
    lines.extend(doc.documentation)
    for line in doc.global_header:
        stripped = line.strip()
        if stripped.upper().startswith("BEGIN LIBGEN"):
            break
        if stripped.startswith("NLIBID:"):
            lines.append(f"NLIBID:      {n_entries}")
        else:
            lines.append(line)
    lines.append("")
    lines.extend(
        [
            "DOCUMENTATION:",
            f"  NOTE: Merged twilight planner output into WFD SIMLIB: {source_path}",
            "  SOURCE: twilight_planner_pkg.simlib_merge",
            "DOCUMENTATION_END:",
            "",
        ]
    )
    return lines


def merge_simlib_with_twilight(
    wfd_simlib_path: str | Path, twilight_df: pd.DataFrame, out_path: str | Path
) -> Path:
    """Merge a WFD SIMLIB with twilight planner epochs."""

    doc = parse_simlib(wfd_simlib_path)
    wfd_entries: Dict[int, SimlibEntry] = {}
    name_by_libid: Dict[int, str] = {}
    for entry in doc.entries:
        name = entry_name(entry)
        wfd_entries[entry.libid] = _copy_entry(entry)
        name_by_libid[entry.libid] = name

    max_wfd_libid = max(wfd_entries) if wfd_entries else 0
    twilight_grouped = _group_twilight_epochs(twilight_df)
    merged: Dict[int, SimlibEntry] = dict(wfd_entries)
    has_wfd_epochs: Dict[int, bool] = {lid: True for lid in merged}
    has_twilight_epochs: Dict[int, bool] = {lid: False for lid in merged}
    name_to_libid: Dict[str, int] = {v: k for k, v in name_by_libid.items()}
    name_to_output_libid: Dict[str, int] = {}
    next_libid = max_wfd_libid

    for name, info in twilight_grouped.items():
        if not info["epochs"]:
            continue
        libid: Optional[int] = None
        if info.get("libid_hint") is not None and info["libid_hint"] in merged:
            libid = int(info["libid_hint"])
            existing_name = name_by_libid.get(libid)
            if existing_name and existing_name != name:
                raise ValueError(f"LIBID collision: {libid} already mapped to {existing_name}")
        elif name in name_to_libid:
            libid = name_to_libid[name]
        if libid is None:
            next_libid += 1
            libid = next_libid
            merged[libid] = SimlibEntry(
                libid=libid,
                ra_deg=info.get("ra"),
                dec_deg=info.get("dec"),
                redshift=info.get("redshift"),
                peakmjd=info.get("peakmjd"),
                comment="",
                match_tns_name=name,
                epochs=[],
            )
            has_wfd_epochs[libid] = False
            name_by_libid[libid] = name
        if name in name_to_output_libid and name_to_output_libid[name] != libid:
            raise ValueError(f"Multiple LIBIDs assigned to {name}: {name_to_output_libid[name]} vs {libid}")
        name_to_output_libid[name] = libid
        entry = merged[libid]
        if info.get("ra") is not None and entry.ra_deg is None:
            entry.ra_deg = info["ra"]
        if info.get("dec") is not None and entry.dec_deg is None:
            entry.dec_deg = info["dec"]
        entry.epochs.extend(info["epochs"])
        has_twilight_epochs[libid] = True

    total_entries = len(merged)
    preamble_lines = _build_preamble(doc, Path(wfd_simlib_path), total_entries)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fp:
        for line in preamble_lines:
            fp.write(str(line).rstrip("\n") + "\n")
        writer = SimlibWriter(fp, SimlibHeader(), preserve_ids=True)
        for libid in sorted(merged.keys()):
            entry = merged[libid]
            epochs_sorted = sorted(entry.epochs, key=lambda e: e.mjd)
            src_tag = "WFD" if has_wfd_epochs.get(libid, False) else "TWILIGHT"
            if has_wfd_epochs.get(libid, False) and has_twilight_epochs.get(libid, False):
                src_tag = "WFD+TWILIGHT"
            base_comment = entry.comment or name_by_libid.get(libid, "")
            if not has_wfd_epochs.get(libid, False):
                base_comment = f"name={name_by_libid.get(libid, base_comment)}"
            comment = " | ".join([c for c in [base_comment, f"SOURCE={src_tag}"] if c])
            ra = entry.ra_deg if entry.ra_deg is not None else 0.0
            dec = entry.dec_deg if entry.dec_deg is not None else 0.0
            writer.start_libid(
                libid,
                ra,
                dec,
                len(epochs_sorted),
                comment=comment,
                redshift=entry.redshift,
                peakmjd=entry.peakmjd,
            )
            for ep in epochs_sorted:
                writer.add_epoch(
                    ep.mjd,
                    ep.band,
                    ep.gain if ep.gain is not None else 0.0,
                    ep.noise if ep.noise is not None else 0.0,
                    ep.skysig if ep.skysig is not None else 0.0,
                    ep.psf1 if ep.psf1 is not None else 0.0,
                    ep.psf2 if ep.psf2 is not None else 0.0,
                    ep.psfratio if ep.psfratio is not None else 0.0,
                    ep.zpavg if ep.zpavg is not None else 0.0,
                    ep.zperr if ep.zperr is not None else 0.0,
                    nexpose=getattr(ep, "nexpose", 1),
                )
            writer.end_libid()
        writer.close()
    return out_path
