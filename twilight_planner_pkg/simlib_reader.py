"""Lightweight SIMLIB parser for WFD-derived cadence libraries.

The reader is intentionally minimal and keeps no external dependencies beyond
``pandas`` for the catalog helper. It understands the SIMLIB structure produced
by SNANA and the existing writer in this repo, extracting global documentation,
per-LIBID metadata, and per-epoch rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable, List, Optional, TextIO

import pandas as pd


@dataclass
class SimlibEpoch:
    """Single `S:` row within a LIBID block."""

    mjd: float
    band: str
    id_expose: Optional[str] = None
    gain: Optional[float] = None
    noise: Optional[float] = None
    skysig: Optional[float] = None
    psf1: Optional[float] = None
    psf2: Optional[float] = None
    psfratio: Optional[float] = None
    zpavg: Optional[float] = None
    zperr: Optional[float] = None
    mag: Optional[float] = None
    nexpose: int = 1


@dataclass
class SimlibEntry:
    """Parsed metadata and epochs for one LIBID."""

    libid: int
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    mwebv: Optional[float] = None
    nobs: Optional[int] = None
    redshift: Optional[float] = None
    peakmjd: Optional[float] = None
    comment: str = ""
    match_tns_name: Optional[str] = None
    epochs: List[SimlibEpoch] = field(default_factory=list)


@dataclass
class SimlibDocument:
    """Container for a parsed SIMLIB file."""

    documentation: List[str]
    global_header: List[str]
    entries: List[SimlibEntry]


_FLOAT = r"[+-]?\d+(?:\.\d+)?"
_NAME_RE = re.compile(r"name=([A-Za-z0-9_.-]+)")


def _extract_float(line: str, key: str) -> Optional[float]:
    """Return the first float following ``key`` in ``line``."""

    try:
        pattern = rf"{re.escape(key)}\s*:?\s*({_FLOAT})"
        m = re.search(pattern, line)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _parse_match_tns(line: str) -> Optional[str]:
    """Extract a MATCH_TNS name if present."""

    m = _NAME_RE.search(line)
    if m:
        return m.group(1)
    return None


def _normalize_band(band: str) -> str:
    """Normalize SIMLIB band tokens to planner convention (y not Y)."""

    b = str(band).strip()
    if not b:
        return b
    if b.lower() == "y":
        return "y"
    return b.lower()


def _parse_epoch(line: str) -> Optional[SimlibEpoch]:
    """Parse an ``S:`` row into a SimlibEpoch."""

    parts = line.split()
    if len(parts) < 4 or not parts[0].startswith("S:"):
        return None

    def _f(idx: int) -> Optional[float]:
        try:
            return float(parts[idx])
        except Exception:
            return None

    mjd = _f(1)
    if mjd is None:
        return None
    id_expose = parts[2] if len(parts) > 2 else None
    band = _normalize_band(parts[3]) if len(parts) > 3 else ""
    nexpose_val = 1
    if id_expose and "*" in id_expose:
        try:
            nexpose_val = int(float(id_expose.split("*")[-1]))
        except Exception:
            nexpose_val = 1
    return SimlibEpoch(
        mjd=float(mjd),
        band=band,
        id_expose=id_expose,
        gain=_f(4),
        noise=_f(5),
        skysig=_f(6),
        psf1=_f(7),
        psf2=_f(8),
        psfratio=_f(9),
        zpavg=_f(10),
        zperr=_f(11),
        mag=_f(12),
        nexpose=nexpose_val,
    )


def _iter_lines(source: str | Path | TextIO | Iterable[str]) -> List[str]:
    """Return all lines from a path, open file, or iterable."""

    if hasattr(source, "read"):
        return list(source)  # type: ignore[arg-type]
    if isinstance(source, (str, Path)):
        return Path(source).read_text().splitlines()
    return list(source)


def parse_simlib(source: str | Path | TextIO | Iterable[str]) -> SimlibDocument:
    """Parse a SIMLIB file into a structured object."""

    lines = _iter_lines(source)
    documentation: List[str] = []
    header: List[str] = []
    entries: List[SimlibEntry] = []
    in_doc = False
    seen_begin = False
    idx = 0
    # Preamble: DOCUMENTATION + global header up to BEGIN LIBGEN
    while idx < len(lines):
        raw = lines[idx].rstrip("\n")
        stripped = raw.strip()
        if stripped.startswith("DOCUMENTATION:"):
            in_doc = True
            documentation.append(raw)
            idx += 1
            continue
        if in_doc:
            documentation.append(raw)
            if stripped.startswith("DOCUMENTATION_END"):
                in_doc = False
            idx += 1
            continue
        header.append(raw)
        if stripped.startswith("BEGIN LIBGEN"):
            seen_begin = True
            idx += 1
            break
        idx += 1

    if not seen_begin:
        return SimlibDocument(documentation, header, entries)

    # Parse LIBID blocks
    while idx < len(lines):
        raw = lines[idx].rstrip("\n")
        stripped = raw.strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith("END_OF_SIMLIB"):
            break
        if not stripped.startswith("LIBID"):
            idx += 1
            continue
        # New LIBID block
        comment = ""
        if "#" in raw:
            comment = raw.split("#", 1)[1].strip()
        libid = _extract_float(raw, "LIBID")
        libid_int = int(libid) if libid is not None else 0
        entry = SimlibEntry(libid=libid_int, comment=comment)
        idx += 1
        while idx < len(lines):
            raw_line = lines[idx].rstrip("\n")
            stripped_line = raw_line.strip()
            if not stripped_line:
                idx += 1
                continue
            if stripped_line.startswith("END_LIBID"):
                idx += 1
                break
            if stripped_line.startswith("LIBID"):
                # Unexpected new block; step back so outer loop can handle.
                break
            if stripped_line.startswith("# MATCH_TNS"):
                name = _parse_match_tns(stripped_line)
                if name:
                    entry.match_tns_name = name
                idx += 1
                continue
            if stripped_line.startswith("#"):
                idx += 1
                continue
            if "RA:" in raw_line:
                ra = _extract_float(raw_line, "RA:")
                dec = _extract_float(raw_line, "DEC:")
                mw = _extract_float(raw_line, "MWEBV:")
                if ra is not None:
                    entry.ra_deg = ra
                if dec is not None:
                    entry.dec_deg = dec
                if mw is not None:
                    entry.mwebv = mw
                idx += 1
                continue
            if "NOBS:" in raw_line:
                nobs = _extract_float(raw_line, "NOBS:")
                if nobs is not None:
                    entry.nobs = int(nobs)
                rz = _extract_float(raw_line, "REDSHIFT:")
                if rz is not None:
                    entry.redshift = rz
                pk = _extract_float(raw_line, "PEAKMJD:")
                if pk is not None:
                    entry.peakmjd = pk
                idx += 1
                continue
            if stripped_line.startswith("S:"):
                epoch = _parse_epoch(raw_line)
                if epoch:
                    entry.epochs.append(epoch)
                idx += 1
                continue
            idx += 1
        entries.append(entry)

    return SimlibDocument(documentation, header, entries)


def _entry_name(entry: SimlibEntry) -> str:
    """Return a stable name for an entry with MATCH_TNS preference."""

    if entry.match_tns_name:
        return entry.match_tns_name
    # Fallback: attempt to extract from the LIBID comment if formatted similarly
    comment_name = _parse_match_tns(entry.comment) or ""
    if comment_name:
        return comment_name
    return f"WFD_LIBID_{entry.libid}"


def entry_name(entry: SimlibEntry) -> str:
    """Public helper to resolve the preferred name for a SIMLIB entry."""

    return _entry_name(entry)


def simlib_to_catalog_df(source: str | Path | SimlibDocument) -> pd.DataFrame:
    """Convert a SIMLIB into a catalog-like DataFrame."""

    doc = parse_simlib(source) if not isinstance(source, SimlibDocument) else source
    rows = []
    for entry in doc.entries:
        rows.append(
            {
                "Name": _entry_name(entry),
                "RA_deg": entry.ra_deg,
                "Dec_deg": entry.dec_deg,
                "redshift": entry.redshift,
                "wfd_libid": entry.libid,
                "source": "WFD",
            }
        )
    return pd.DataFrame(rows)


def simlib_visits_by_name(source: str | Path | SimlibDocument) -> dict[str, dict[str, List[float]]]:
    """Return a mapping {name: {band: sorted_mjd_list}} from a SIMLIB."""

    doc = parse_simlib(source) if not isinstance(source, SimlibDocument) else source
    visits: dict[str, dict[str, List[float]]] = {}
    for entry in doc.entries:
        name = _entry_name(entry)
        band_map = visits.setdefault(name, {})
        for epoch in entry.epochs:
            band = _normalize_band(epoch.band)
            if not band:
                continue
            band_map.setdefault(band, []).append(float(epoch.mjd))
    # Sort MJD lists for determinism
    for band_map in visits.values():
        for band, mjds in list(band_map.items()):
            band_map[band] = sorted(mjds)
    return visits
