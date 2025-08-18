from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from astropy.time import Time

from .astro_utils import parse_sn_type_to_window_days, validate_coords
from .config import PlannerConfig

# mypy: ignore-errors


def _normalize_col_names(names):
    """Canonicalize column names for fuzzy matching.

    Parameters
    ----------
    names : Iterable[str]
        Raw column names from the input table.

    Returns
    -------
    list[tuple[str, str]]
        Sequence of (original, normalized) pairs where the normalized
        string is lower-case and alphanumeric only.
    """
    out = []
    for n in names:
        key = "".join(ch for ch in str(n).lower() if ch.isalnum())
        out.append((n, key))
    return out


_RA_SYNONYMS = [
    "ra",
    "radeg",
    "raj2000",
    "ra_deg",
    "ra(deg)",
    "ra_deg_j2000",
    "ra_j2000",
]
_DEC_SYNONYMS = [
    "dec",
    "decl",
    "declination",
    "decdeg",
    "dej2000",
    "dec_deg",
    "dec(deg)",
    "dec_j2000",
]
_DISC_DATE_SYNONYMS = [
    "discoverydate",
    "discdate",
    "discoverdate",
    "firstdetected",
    "firstdiscoverydate",
    "date",
    "utcdate",
    "isodate",
    "discoverymjd",
    "discmjd",
    "firstmjd",
    "mjd",
]
_NAME_SYNONYMS = [
    "name",
    "objname",
    "objectname",
    "atlasname",
    "tnsname",
    "id",
    "snname",
]
_TYPE_SYNONYMS = ["type", "sntype", "class", "tnsclass", "subtype"]

_MAG_SYNONYMS = {
    "g": ["gmag", "mag_g", "atlas_gmag", "last_mag_g", "gmaglast"],
    "r": ["rmag", "mag_r", "atlas_rmag", "last_mag_r", "rmaglast"],
    "i": ["imag", "mag_i", "atlas_imag", "last_mag_i", "imaglast"],
    "z": ["zmag", "mag_z", "atlas_zmag", "last_mag_z", "zmaglast"],
}


def _fuzzy_pick(df: pd.DataFrame, synonyms: List[str]) -> Optional[str]:
    """Select a column whose normalized name matches any synonym.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table to search.
    synonyms : list[str]
        Candidate names to match against.

    Returns
    -------
    str or None
        The first matching column name, or ``None`` if no match is found.
    """
    canon = dict(_normalize_col_names(df.columns))
    for syn in synonyms:
        syn_key = "".join(ch for ch in syn.lower() if ch.isalnum())
        for orig, key in canon.items():
            if key == syn_key:
                return orig
    return None


def resolve_columns(
    df: pd.DataFrame, cfg: PlannerConfig
) -> Tuple[str, str, Optional[str], str, Optional[str]]:
    """Determine key column names in an input catalog.

    Parameters
    ----------
    df : pandas.DataFrame
        Catalog of transients.
    cfg : PlannerConfig
        Configuration optionally providing explicit column names.

    Returns
    -------
    tuple
        ``(ra_col, dec_col, disc_col, name_col, type_col)``.
        Raises ``KeyError`` if required columns are missing.
    """
    ra_col = cfg.ra_col if (cfg.ra_col in df.columns) else _fuzzy_pick(df, _RA_SYNONYMS)
    dec_col = (
        cfg.dec_col if (cfg.dec_col in df.columns) else _fuzzy_pick(df, _DEC_SYNONYMS)
    )
    disc_col = (
        cfg.disc_col
        if (cfg.disc_col in df.columns)
        else _fuzzy_pick(df, _DISC_DATE_SYNONYMS)
    )
    name_col = (
        cfg.name_col
        if (cfg.name_col in df.columns)
        else _fuzzy_pick(df, _NAME_SYNONYMS)
    )
    type_col = (
        cfg.type_col
        if (cfg.type_col in df.columns)
        else _fuzzy_pick(df, _TYPE_SYNONYMS)
    )

    missing = []
    if ra_col is None:
        missing.append("RA")
    if dec_col is None:
        missing.append("Dec")
    if name_col is None:
        missing.append("Name")

    if missing:
        raise KeyError(
            f"Required column(s) not found or auto-detected: {', '.join(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    return ra_col, dec_col, disc_col, name_col, type_col


def _parse_ra_value(val) -> float:
    """Convert a single right ascension value to degrees.

    Parameters
    ----------
    val : object
        RA expressed as degrees, hours, radians, or a sexagesimal string.

    Returns
    -------
    float
        Right ascension in degrees, or ``numpy.nan`` if parsing fails.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (float, int, np.floating, np.integer)):
        x = float(val)
        if 0.0 <= x <= 2 * np.pi + 0.05:  # radians
            return float(np.degrees(x)) % 360.0
        if 0.0 <= x <= 24.1:  # hours
            return (x * 15.0) % 360.0
        return x % 360.0  # degrees
    try:
        ang = Angle(str(val))
        if ang.unit == u.radian:
            return float(ang.to(u.deg).value) % 360.0
        return float(ang.wrap_at(360 * u.deg).degree)
    except Exception:
        return np.nan


def _parse_dec_value(val) -> float:
    """Convert a single declination value to degrees.

    Parameters
    ----------
    val : object
        Declination expressed as degrees, radians, or a sexagesimal string.

    Returns
    -------
    float
        Declination in degrees, or ``numpy.nan`` if parsing fails.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (float, int, np.floating, np.integer)):
        x = float(val)
        if (-np.pi / 2 - 0.05) <= x <= (np.pi / 2 + 0.05):  # radians
            return float(np.degrees(x))
        return x
    try:
        ang = Angle(str(val))
        return float(ang.to(u.deg).value)
    except Exception:
        return np.nan


def quick_unit_report(df: pd.DataFrame, ra_col: str, dec_col: str) -> None:
    """Print the numeric ranges of the RA and Dec columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Table containing coordinate columns.
    ra_col : str
        Name of the right ascension column.
    dec_col : str
        Name of the declination column.
    """
    ra = pd.to_numeric(df[ra_col], errors="coerce")
    dec = pd.to_numeric(df[dec_col], errors="coerce")
    print(f"RA raw (numeric) range:  min={np.nanmin(ra):.6f}, max={np.nanmax(ra):.6f}")
    print(
        f"Dec raw (numeric) range: min={np.nanmin(dec):.6f}, max={np.nanmax(dec):.6f}"
    )


def _infer_units(ra_num: pd.Series, dec_num: pd.Series):
    """Guess the units of raw RA and Dec columns.

    Parameters
    ----------
    ra_num : pandas.Series
        Numeric right ascension values.
    dec_num : pandas.Series
        Numeric declination values.

    Returns
    -------
    tuple
        ``(ra_unit, dec_unit, notes)`` where units are ``'deg'``, ``'hour'``,
        ``'rad'``, or ``'unknown'`` and ``notes`` is a list of warnings.
    """
    notes = []
    ra_unit = "unknown"
    if ra_num.notna().any():
        ra_max = float(np.nanmax(ra_num))
        if ra_max > 50.0:
            ra_unit = "deg"
        elif 0.0 <= ra_max <= 24.2:
            ra_unit = "hour"
        elif ra_max <= (2 * np.pi + 0.2):
            ra_unit = "rad"
        else:
            ra_unit = "deg"
        if 24.2 < ra_max <= 50.0:
            notes.append(
                f"RA numeric max={ra_max:.3f} is between 24h and 50°, ambiguous. Assuming degrees."
            )
    else:
        notes.append(
            "RA has non-numeric values; falling back to string parsing via astropy Angle."
        )

    dec_unit = "unknown"
    if dec_num.notna().any():
        dec_abs_max = float(np.nanmax(np.abs(dec_num)))
        if dec_abs_max <= 90.5:
            dec_unit = "deg"
        elif dec_abs_max <= 1.7:
            dec_unit = "rad"
        else:
            dec_unit = "unknown"
            notes.append(
                f"Dec |max|={dec_abs_max:.3f} looks suspicious; trying robust parsing per value."
            )
        if 1.5 <= dec_abs_max <= 1.7:
            notes.append(
                "Dec |max| ~ 1.57 rad (90°). If in radians, will be converted."
            )
    else:
        notes.append(
            "Dec has non-numeric values; falling back to string parsing via astropy Angle."
        )

    return ra_unit, dec_unit, notes


def unit_report_from_df(df: pd.DataFrame, cfg: PlannerConfig) -> dict:
    """Infer coordinate units and report column choices.

    Parameters
    ----------
    df : pandas.DataFrame
        Input catalog.
    cfg : PlannerConfig
        Configuration with optional column overrides.

    Returns
    -------
    dict
        Summary including chosen columns, min/max values,
        inferred units, and notes.
    """
    ra_col, dec_col, disc_col, name_col, type_col = resolve_columns(df, cfg)
    ra_num = pd.to_numeric(df[ra_col], errors="coerce")
    dec_num = pd.to_numeric(df[dec_col], errors="coerce")
    ra_min = float(np.nanmin(ra_num)) if ra_num.notna().any() else float("nan")
    ra_max = float(np.nanmax(ra_num)) if ra_num.notna().any() else float("nan")
    dec_min = float(np.nanmin(dec_num)) if dec_num.notna().any() else float("nan")
    dec_max = float(np.nanmax(dec_num)) if dec_num.notna().any() else float("nan")
    print(f"RA raw (numeric) range:  min={ra_min:.6f}, max={ra_max:.6f}")
    print(f"Dec raw (numeric) range: min={dec_min:.6f}, max={dec_max:.6f}")
    ra_unit, dec_unit, notes = _infer_units(ra_num, dec_num)
    print(f"→ Inferred RA unit:  {ra_unit}")
    print(f"→ Inferred Dec unit: {dec_unit}")
    if notes:
        print("Warnings / notes:")
        for n in notes:
            print(" -", n)
    return {
        "columns": {
            "ra": ra_col,
            "dec": dec_col,
            "disc": disc_col,
            "name": name_col,
            "type": type_col,
        },
        "ra": {"min": ra_min, "max": ra_max, "unit_inferred": ra_unit},
        "dec": {"min": dec_min, "max": dec_max, "unit_inferred": dec_unit},
        "notes": notes,
    }


def normalize_ra_dec_to_degrees(
    df: pd.DataFrame, ra_col: str, dec_col: str, name_col: str | None = None
) -> pd.DataFrame:
    """Normalize RA/Dec columns to degrees.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    ra_col : str
        Column containing right ascension.
    dec_col : str
        Column containing declination.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with ``'RA_deg'`` and ``'Dec_deg'`` columns in degrees.
    """
    out = df.copy()
    ra_num = pd.to_numeric(out[ra_col], errors="coerce")
    dec_num = pd.to_numeric(out[dec_col], errors="coerce")
    if ra_num.notna().any():
        ra_max = float(np.nanmax(ra_num))
        if ra_max > 50.0:
            ra_deg = ra_num.astype(float)
        elif ra_max <= 24.2 and ra_max >= 0.0:
            ra_deg = ra_num.astype(float) * 15.0
        elif ra_max <= (2 * np.pi + 0.2):
            ra_deg = np.degrees(ra_num.astype(float))
        else:
            ra_deg = ra_num.astype(float)
    else:
        ra_deg = pd.Series(
            [_parse_ra_value(v) for v in out[ra_col].values],
            index=out.index,
            dtype=float,
        )

    if dec_num.notna().any():
        dec_abs_max = float(np.nanmax(np.abs(dec_num)))
        if dec_abs_max <= 90.5:
            dec_deg = dec_num.astype(float)
        elif dec_abs_max <= 1.7:
            dec_deg = np.degrees(dec_num.astype(float))
        else:
            dec_deg = pd.Series(
                [_parse_dec_value(v) for v in out[dec_col].values],
                index=out.index,
                dtype=float,
            )
    else:
        dec_deg = pd.Series(
            [_parse_dec_value(v) for v in out[dec_col].values],
            index=out.index,
            dtype=float,
        )

    ra_vals = ra_deg.astype(float).values
    dec_vals = dec_deg.astype(float).values

    def _val(idx: int, ra: float, dec: float) -> Tuple[float, float]:
        snid = out[name_col].iloc[idx] if name_col and name_col in out.columns else idx
        try:
            return validate_coords(ra, dec)
        except ValueError as e:
            raise ValueError(f"SN {snid}: {e}")

    validated = [_val(i, ra_vals[i], dec_vals[i]) for i in range(len(out))]
    ra_norm, dec_clamped = zip(*validated)
    out["RA_deg"] = ra_norm
    out["Dec_deg"] = dec_clamped
    return out


def _parse_discovery_to_datetime(series: pd.Series) -> pd.Series:
    """Parse discovery date values into timezone-aware datetimes.

    Parameters
    ----------
    series : pandas.Series
        Raw discovery timestamps or MJDs.

    Returns
    -------
    pandas.Series
        Series of UTC datetimes (``datetime64[ns, UTC]``).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
        mask_mjd = (numeric >= 30000) & (numeric <= 90000)
        if mask_mjd.any():
            dt_mjd = pd.to_datetime(
                Time(numeric[mask_mjd].values, format="mjd").to_datetime("utc")
            )
            out.loc[mask_mjd.index[mask_mjd]] = dt_mjd
        mask_jd = (numeric >= 2400000) & (numeric <= 2500000)
        if mask_jd.any():
            dt_jd = pd.to_datetime(
                Time(numeric[mask_jd].values, format="jd").to_datetime("utc")
            )
            out.loc[mask_jd.index[mask_jd]] = dt_jd
        remaining = series[~(mask_mjd | mask_jd)]
        if remaining.size:
            out.loc[remaining.index] = pd.to_datetime(
                remaining, utc=True, errors="coerce"
            )
        return out
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC")
    return dt


def standardize_columns(df: pd.DataFrame, cfg: PlannerConfig) -> pd.DataFrame:
    """Standardize column names and units for planning.

    Parameters
    ----------
    df : pandas.DataFrame
        Input supernova catalog.
    cfg : PlannerConfig
        Configuration describing expected columns and units.

    Returns
    -------
    pandas.DataFrame
        Normalized table with RA/Dec in degrees and helper columns
        (``'Name'``, ``'SN_type_raw'``, ``'discovery_datetime'``).
    """
    ra_col, dec_col, disc_col, name_col, type_col = resolve_columns(df, cfg)
    try:
        quick_unit_report(df, ra_col, dec_col)
    except Exception:
        pass
    df = normalize_ra_dec_to_degrees(df, ra_col, dec_col, name_col)
    if disc_col and disc_col in df.columns:
        parsed = _parse_discovery_to_datetime(df[disc_col])
    else:
        parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    df["discovery_datetime"] = parsed
    df["Name"] = (
        df[name_col].astype(str)
        if name_col in df.columns
        else [f"SN_{i:05d}" for i in range(len(df))]
    )
    df["SN_type_raw"] = (
        df[type_col].astype(str) if (type_col and type_col in df.columns) else np.nan
    )
    df["typical_lifetime_days"] = df["SN_type_raw"].apply(
        lambda t: parse_sn_type_to_window_days(t, cfg)
    )
    return df


def extract_current_mags(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Extract per-target magnitudes if columns are present.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table after :func:`standardize_columns`.

    Returns
    -------
    dict
        Mapping from SN name to ``{band: mag}`` for available bands.
    """
    import math

    canon = dict(_normalize_col_names(df.columns))
    band_cols: Dict[str, str] = {}
    for band, syns in _MAG_SYNONYMS.items():
        for syn in syns:
            key = "".join(ch for ch in syn.lower() if ch.isalnum())
            for orig, norm in canon.items():
                if norm == key:
                    band_cols[band] = orig
                    break
            if band_cols.get(band):
                break

    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        name = row.get("Name")
        mags: Dict[str, float] = {}
        for band, col in band_cols.items():
            try:
                val = float(row[col])
                if not math.isnan(val):
                    mags[band] = float(val)
            except Exception:
                continue
        if mags:
            out[str(name)] = mags
    return out
