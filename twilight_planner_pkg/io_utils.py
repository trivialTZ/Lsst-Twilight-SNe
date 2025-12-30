from __future__ import annotations

from datetime import timezone as _dt_timezone
from typing import Dict, List, Optional, Tuple
import re

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from astropy.time import Time

from .astro_utils import (
    parse_sn_type_to_window_days,
    peak_mag_from_redshift,
    validate_coords,
)
from .config import PlannerConfig
from .config import (
    DISCOVERY_ATLAS_LINEAR,
    DISCOVERY_COLOR_PRIORS_MIN,
    DISCOVERY_COLOR_PRIORS_MAX,
    DISCOVERY_LINEAR_COEFFS,
    CLEAR_ZEROPOINT,
)

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
    # TNS variant
    "discoverydateut",
    # generic fallbacks
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
_TYPE_SYNONYMS = [
    "type",
    "sntype",
    "class",
    "tnsclass",
    "subtype",
    "sn_type_raw",
    # TNS variant
    "objtype",
]

_MAG_SYNONYMS = {
    "g": [
        "gmag",
        "mag_g",
        "atlas_gmag",
        "last_mag_g",
        "gmaglast",
        # additional common variants
        "atlas_mag_g",
        "magpsf_g",
        "psfmag_g",
    ],
    "r": [
        "rmag",
        "mag_r",
        "atlas_rmag",
        "last_mag_r",
        "rmaglast",
        # additional common variants
        "atlas_mag_r",
        "magpsf_r",
        "psfmag_r",
    ],
    "i": [
        "imag",
        "mag_i",
        "atlas_imag",
        "last_mag_i",
        "imaglast",
        # additional common variants
        "atlas_mag_i",
        "magpsf_i",
        "psfmag_i",
    ],
    "z": [
        "zmag",
        "mag_z",
        "atlas_zmag",
        "last_mag_z",
        "zmaglast",
        # additional common variants
        "atlas_mag_z",
        "magpsf_z",
        "psfmag_z",
    ],
    "y": [
        "ymag",
        "mag_y",
        "atlas_ymag",
        "last_mag_y",
        "ymaglast",
        # additional common variants
        "atlas_mag_y",
        "magpsf_y",
        "psfmag_y",
    ],
}

# Fallback based on discovery magnitude
_DISC_MAG_SYNONYMS = [
    "discoverymag",
    "disc_mag",
    "discovermag",
    # TNS variant
    "discoverymagflux",
]
_DISC_FILT_SYNONYMS = [
    "discmagfilter",
    "discoverymagfilter",
    "disc_filter",
    # common variants
    "filter",
    "band",
    # TNS variants
    "discoveryfilter",
    "discfilter",
]


def _pick_filter_column(df: pd.DataFrame) -> Optional[str]:
    """Pick a discovery filter column from common synonyms if present."""
    canon = dict(_normalize_col_names(df.columns))
    for syn in _DISC_FILT_SYNONYMS:
        key = "".join(ch for ch in syn.lower() if ch.isalnum())
        for orig, norm in canon.items():
            if norm == key:
                return orig
    return None


def _norm_filter_string(x: object) -> str:
    """Normalize a free-form filter string to a compact token.

    Examples: "Orange" → "o", "ATLAS-o" → "o", "g-ZTF" → "g", "PS1 r" → "r".
    """
    s = str(x).strip().lower()
    # remove common prefixes/suffixes and spaces, plus common survey/instrument tags
    for rm in [
        "atlas-",
        "-atlas",
        "atlas",
        "ps1",
        "ps ",
        "_ps1",
        "-p1",
        "p1",
        "gpc1",
        "gpc2",
        "-gpc1",
        "-gpc2",
        "ztf",
        "-ztf",
        "lsst",
        "panstarrs",
        "blackgem",
        "bg",
        "goto",
        "-goto",
        "crts",
        "-crts",
        "sedm",
        "-sedm",
        " ",
    ]:
        s = s.replace(rm, "")
    alias = {
        "orange": "o",
        "cyan": "c",
        "o": "o",
        "c": "c",
        "gpc1": "g",
        "rpc1": "r",
        "ipc1": "i",
        "zpc1": "z",
        "ypc1": "y",
        "g": "g",
        "r": "r",
        "i": "i",
        "z": "z",
        "y": "y",
    }
    if s in alias:
        return alias[s]
    # Fallback: pick the first plausible band letter
    for ch in s:
        if ch in "ugrizyoc":
            return ch
    return s


def _to_r_from_atlas(
    discovery_mag: float, disc_filter: str, assumed_gr: float = 0.0
) -> Optional[float]:
    """Approximate r-band mag from ATLAS c/o using simple color relations.

    With no color (g-r≈0), this reduces to r≈c or r≈o.
    """
    try:
        dm = float(discovery_mag)
    except Exception:
        return None
    f = str(disc_filter).lower()
    if f == "c":
        return dm - 0.47 * float(assumed_gr)
    if f == "o":
        return dm + 0.26 * float(assumed_gr)
    return None


# ---------------------------------------------------------------------------
# Heterogeneous discovery photometry ingestion and mapping
# ---------------------------------------------------------------------------

# Canonical tags for heterogeneous discovery filters
_FILTER_SYNONYMS = {
    r"^\s*w[\-\s]?p?1\s*$": "PS1_w",
    r"^\s*l[\-\s]?goto\s*$": "GOTO_L",
    r"^\s*bg[\-\s]?q[\-\s]?(blackgem)?$": "BG_q",
    r"^\s*clear[\-\s]?$": "Clear",
    r"^\s*v[\-\s]?crts[\-\s]?(crts)?$": "CRTS_V",
}


def normalize_filter_name(s: str | object) -> Optional[str]:
    """Normalize free-form discovery filter strings to canonical tags.

    Returns one of: 'PS1_w', 'GOTO_L', 'BG_q', 'Clear', 'CRTS_V', or
    LSST bands 'g'/'r' (and maps others to None for downstream defaulting).
    Matching is case/space/punctuation insensitive.
    """
    try:
        t = str(s).strip().lower()
    except Exception:
        return None
    for pat, tag in _FILTER_SYNONYMS.items():
        if re.match(pat, t):
            return tag
    # Allow direct LSST-like single-letter band identifiers
    if t in {"g", "r"}:
        return t
    # tolerate strings like 'g-ztf', 'r-ztf'
    if t.startswith("g"):
        return "g"
    if t.startswith("r"):
        return "r"
    return None


def _clip_color(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        xv = float(x)
    except Exception:
        return None
    if np.isnan(xv):
        return None
    return float(np.clip(xv, DISCOVERY_COLOR_PRIORS_MIN, DISCOVERY_COLOR_PRIORS_MAX))


def map_to_lsst(
    tag: str,
    mag: float,
    colors: Dict[str, Optional[float]] | None,
    coeffs: Optional[Dict[str, float]] = None,
) -> Tuple[str, float]:
    """Map a heterogeneous discovery-band magnitude to an LSST band/magnitude.

    Parameters
    ----------
    tag : str
        Canonical filter tag (e.g., 'PS1_w', 'GOTO_L', 'BG_q', 'Clear', 'CRTS_V', 'g', 'r').
    mag : float
        Input discovery magnitude in the given tag.
    colors : dict
        Optional per-object colors with keys 'g-r' and 'r-i'.
    coeffs : dict or None
        Optional override coefficients {a, b, c?}. If None, defaults are used.

    Returns
    -------
    (lsst_band, mag_transformed)
    """
    # Direct pass-through for g/r if requested
    if tag in {"g", "r"}:
        return tag, float(mag)

    colors = colors or {}
    gr = _clip_color(colors.get("g-r"))
    ri = _clip_color(colors.get("r-i"))

    def _apply(default_key: str) -> Tuple[str, float]:
        default = DISCOVERY_LINEAR_COEFFS.get(default_key, {})
        target = str(default.get("target", "r"))
        a = float((coeffs or default).get("a", 0.0))
        b = float((coeffs or default).get("b", 0.0))
        c = float((coeffs or default).get("c", 0.0))
        dm = a
        if gr is not None:
            dm += b * gr
        if (c != 0.0) and (ri is not None):
            dm += c * ri
        return target, float(mag + dm)

    if tag == "PS1_w":
        return _apply("PS1_w")
    if tag == "GOTO_L":
        return _apply("GOTO_L")
    if tag == "BG_q":
        return _apply("BG_q")
    if tag in {"Clear", "CRTS_V"}:
        # Treat Clear as V (CV) unless overriden to CR. CRTS_V is V.
        if (tag == "CRTS_V") or (CLEAR_ZEROPOINT.upper() == "CV"):
            return _apply("CV_to")
        else:
            return _apply("CR_to")

    # Unknown tag: conservatively map to r unchanged
    return "r", float(mag)


def _fit_linear_coeffs_for_tag(
    tag: str, calibrators: Optional[pd.DataFrame]
) -> Optional[Dict[str, float]]:
    """Optionally fit (a,b[,c]) for a given tag using calibrators.

    Expects calibrators with columns at least: 'tag', 'mag' (input discovery mag),
    and per-object reference magnitudes 'g','r','i'. The target band is implied by
    DISCOVERY_LINEAR_COEFFS[target]['target'] for the tag.

    Returns None if inputs are insufficient.
    """
    if calibrators is None or tag not in DISCOVERY_LINEAR_COEFFS:
        return None
    df = calibrators
    try:
        df = df[df["tag"] == tag]
    except Exception:
        return None
    need_cols = {"mag", "g", "r"}
    if not need_cols.issubset(set(df.columns)):
        return None
    cfg_default = DISCOVERY_LINEAR_COEFFS.get(tag, {})
    target = str(cfg_default.get("target", "r"))
    if target not in {"g", "r"}:
        target = "r"
    # Build design matrix: [1, (g-r), (r-i)?]
    try:
        gr = (df["g"] - df["r"]).astype(float)
        ri = None
        if "i" in df.columns:
            ri = (df["r"] - df["i"]).astype(float)
        y = (df[target] - df["mag"]).astype(float)
        # clip colors to priors range
        gr = gr.clip(DISCOVERY_COLOR_PRIORS_MIN, DISCOVERY_COLOR_PRIORS_MAX)
        if ri is not None:
            ri = ri.clip(DISCOVERY_COLOR_PRIORS_MIN, DISCOVERY_COLOR_PRIORS_MAX)
        X_cols = [np.ones_like(gr.values), gr.values]
        if ri is not None and ("c" in cfg_default):
            X_cols.append(ri.values)
        X = np.vstack(X_cols).T
        # Robust-ish: iterative sigma clip on residuals
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y.values)
        Xw = X[mask]
        yw = y.values[mask]
        if Xw.shape[0] < Xw.shape[1] + 3:
            return None
        for _ in range(2):
            coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            resid = yw - Xw @ coef
            s = np.nanstd(resid)
            if not np.isfinite(s) or s <= 0:
                break
            keep = np.abs(resid) < 3.0 * s
            if keep.sum() < Xw.shape[1] + 3:
                break
            Xw = Xw[keep]
            yw = yw[keep]
        # Map to a,b,c
        out: Dict[str, float] = {"a": float(coef[0]), "b": float(coef[1])}
        if len(coef) >= 3:
            out["c"] = float(coef[2])
        return out
    except Exception:
        return None


def read_discovery_csv(
    path: str,
    ref_colors_df: Optional[pd.DataFrame] = None,
    calibrators_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Read a mixed-filter discovery CSV and normalize to LSST g/r magnitudes.

    Returns columns: mjd, ra, dec, lsst_band, mag_lsst, magerr

    - Accepts liberal headers for mjd, mag, magerr/e_mag, filter/band, ra, dec, survey.
    - Does not mutate input RA/Dec.
    - If DISCOVERY_ATLAS_LINEAR and calibrators are provided, fits per-tag
      (a,b[,c]) and applies; otherwise falls back to static coefficients.
    - Fitting is skipped gracefully if calibrators are unavailable.
    """
    df = pd.read_csv(path)
    # tolerant header mapping
    def _pick(colset: List[str], keys: List[str]) -> Optional[str]:
        keys_norm = {"".join(ch for ch in k.lower() if ch.isalnum()): k for k in colset}
        for want in keys:
            w = "".join(ch for ch in want.lower() if ch.isalnum())
            for knorm, orig in keys_norm.items():
                if knorm == w:
                    return orig
        return None

    cols = list(df.columns)
    c_mjd = _pick(cols, ["mjd", "jd"])
    c_mag = _pick(cols, ["mag", "magnitude"])  # discovery magnitude value
    c_meg = _pick(cols, ["magerr", "e_mag", "mag_error"])  # uncertainty
    c_fil = _pick(cols, ["filter", "band", "filt", "discmagfilter", "discoveryfilter"])
    c_ra = _pick(cols, ["ra"])  # do not mutate RA
    c_dec = _pick(cols, ["dec", "decl", "declination"])  # do not mutate Dec
    # Optional survey (unused but tolerated)
    c_surv = _pick(cols, ["survey"])  # noqa: F841

    need = {"mjd": c_mjd, "mag": c_mag, "magerr": c_meg, "filter": c_fil, "ra": c_ra, "dec": c_dec}
    missing = [k for k, v in need.items() if v is None]
    if missing:
        raise ValueError(f"Missing required column(s) {missing} in {path}")

    df = df.rename(columns={c_mjd: "mjd", c_mag: "mag", c_meg: "magerr", c_fil: "filter", c_ra: "ra", c_dec: "dec"})

    # Normalize filter tags
    df["tag"] = df["filter"].apply(normalize_filter_name)
    # Provide per-object colors if available
    if ref_colors_df is not None:
        # Expect columns 'g-r','r-i' keyed by index or an identifier; fallback to NaNs
        for c in ("g-r", "r-i"):
            if c not in ref_colors_df.columns:
                ref_colors_df[c] = np.nan
        try:
            df = df.join(ref_colors_df[["g-r", "r-i"]], how="left")
        except Exception:
            # If join fails, just fill NaNs
            df["g-r"] = np.nan
            df["r-i"] = np.nan
    else:
        df["g-r"] = np.nan
        df["r-i"] = np.nan

    # Optionally fit per-tag coefficients
    fit_by_tag: Dict[str, Dict[str, float]] = {}
    if DISCOVERY_ATLAS_LINEAR and calibrators_df is not None:
        try:
            tags = sorted(set(df["tag"].dropna().astype(str)))
        except Exception:
            tags = []
        for t in tags:
            co = _fit_linear_coeffs_for_tag(t, calibrators_df)
            if co is not None:
                fit_by_tag[t] = co

    # Transform per-row
    out_band: List[str] = []
    out_mag: List[float] = []
    for tag, mag, gr, ri in zip(df["tag"], df["mag"], df["g-r"], df["r-i"]):
        t = tag if isinstance(tag, str) else None
        if t is None:
            # Unknown tag → default to r unchanged
            out_band.append("r")
            try:
                out_mag.append(float(mag))
            except Exception:
                out_mag.append(np.nan)
            continue
        co = fit_by_tag.get(t)
        band, mt = map_to_lsst(t, float(mag), {"g-r": gr, "r-i": ri}, coeffs=co)
        # Ensure only g/r bands are emitted
        band = band if band in {"g", "r"} else "r"
        out_band.append(band)
        out_mag.append(mt)

    df["lsst_band"] = out_band
    df["mag_lsst"] = out_mag
    return df[["mjd", "ra", "dec", "lsst_band", "mag_lsst", "magerr"]]


def infer_src_mag_from_discovery_pro(
    df: pd.DataFrame,
    planned_bands: List[str],
    *,
    policy: str = "atlas_transform",
    assumed_gr: float = 0.0,
    margin_mag: float = 0.2,
) -> Dict[str, Dict[str, float]]:
    """Infer per-band source magnitudes from discovery magnitude.

    Parameters
    ----------
    df : pandas.DataFrame
        Catalog after :func:`standardize_columns`.
    planned_bands : list[str]
        Bands that the planner intends to observe (cfg.filters).
    policy : str
        "copy" to copy discoverymag into all bands minus ``margin_mag``;
        "atlas_transform" to convert ATLAS c/o to r (using ``assumed_gr``),
        copy to others with a small margin.
    assumed_gr : float
        Assumed g-r color used in ATLAS c/o → r transformation.
    margin_mag : float
        Safety margin (mag) to make copied mags brighter (conservative).
    """
    # Pick discovery magnitude column
    disc_mag_col = None
    canon = dict(_normalize_col_names(df.columns))
    for syn in _DISC_MAG_SYNONYMS:
        key = "".join(ch for ch in syn.lower() if ch.isalnum())
        for orig, norm in canon.items():
            if norm == key:
                disc_mag_col = orig
                break
        if disc_mag_col:
            break
    if disc_mag_col is None:
        return {}

    filt_col = _pick_filter_column(df)
    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        name = str(row.get("Name", row.name))
        dm = row.get(disc_mag_col)
        try:
            dm = float(dm)
        except Exception:
            continue
        raw_f = row.get(filt_col, "") if filt_col else ""
        # First try richer normalization/mapping to LSST g/r with linear terms
        tag = normalize_filter_name(raw_f)
        if tag is not None:
            try:
                lsst_band, dm_lsst = map_to_lsst(tag, float(dm), colors=None, coeffs=None)
                disc_f = str(lsst_band).lower()
                dm = float(dm_lsst)
            except Exception:
                disc_f = _norm_filter_string(raw_f)
        else:
            disc_f = _norm_filter_string(raw_f)
        src: Dict[str, float] = {}

        if policy == "copy":
            for b in planned_bands:
                src[b] = dm - float(margin_mag)
        elif policy == "atlas_transform":
            # Determine r first
            if disc_f == "r":
                rmag = dm
            elif disc_f == "g":
                rmag = dm - float(assumed_gr)
            elif disc_f in ("c", "o"):
                rmag = _to_r_from_atlas(dm, disc_f, assumed_gr=assumed_gr)
                if rmag is None:
                    rmag = dm - float(margin_mag)
            else:
                rmag = dm - float(margin_mag)

            for b in planned_bands:
                if str(b).lower() == "r":
                    src[b] = float(rmag)
                else:
                    src[b] = float(rmag) - float(margin_mag)
        else:  # atlas_priors (use color prior ranges for conservative bright estimate)
            # pull from cfg-like attributes via closure? Not available here; use defaults
            # Overload via a small helper that reads from a global config isn't ideal.
            # Instead, we support atlas_priors in build_mag_lookup_with_fallback below where cfg is present.
            # Here we just fallback to atlas_transform behavior.
            if disc_f == "r":
                rmag = dm
            elif disc_f == "g":
                rmag = dm - float(assumed_gr)
            elif disc_f in ("c", "o"):
                rmag = _to_r_from_atlas(dm, disc_f, assumed_gr=assumed_gr)
                if rmag is None:
                    rmag = dm - float(margin_mag)
            else:
                rmag = dm - float(margin_mag)
            for b in planned_bands:
                if str(b).lower() == "r":
                    src[b] = float(rmag)
                else:
                    src[b] = float(rmag) - float(margin_mag)

        out[name] = src
    return out


def _merge_mag_maps(
    base: Dict[str, Dict[str, float]], fallback: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Merge two {Name: {band: mag}} maps, preferring values from ``base``."""
    out: Dict[str, Dict[str, float]] = {}
    names = set(base.keys()) | set(fallback.keys())
    for n in names:
        bmap = dict(fallback.get(n, {}))
        bmap.update(base.get(n, {}))  # base wins
        if bmap:
            out[n] = bmap
    return out


def build_mag_lookup_with_fallback(
    df: pd.DataFrame, cfg: PlannerConfig
) -> Dict[str, Dict[str, float]]:
    """Return per-target per-band magnitudes with discovery fallback.

    This wraps :func:`extract_current_mags` and, when necessary, fills missing
    bands or targets using discovery magnitude heuristics. Fallback is active
    only when the input contains a discovery magnitude column.
    """
    base = extract_current_mags(df)
    # Only activate fallback if the config allows and discovery mag is present
    if not getattr(cfg, "use_discovery_fallback", True):
        return base
    planned = list(getattr(cfg, "filters", []) or [])
    if not planned:
        planned = ["r", "i", "z", "y", "g"]
    policy = str(getattr(cfg, "discovery_policy", "atlas_priors") or "atlas_priors")
    assumed_gr = float(getattr(cfg, "discovery_assumed_gr", 0.0))
    margin = float(getattr(cfg, "discovery_margin_mag", 0.2))

    fb_disc: Dict[str, Dict[str, float]] = {}
    if policy == "atlas_priors":
        fb_disc = _infer_from_discovery_with_priors(df, planned, cfg)
    else:
        fb_disc = infer_src_mag_from_discovery_pro(
            df, planned, policy=policy, assumed_gr=assumed_gr, margin_mag=margin
        )

    try:
        fb_peak = _infer_peak_from_redshift(df, planned, cfg)
    except Exception:
        fb_peak = {}

    fallback_combined: Dict[str, Dict[str, float]] = {}
    if fb_disc or fb_peak:
        names = set(fb_disc.keys()) | set(fb_peak.keys())
        for name in names:
            band_map: Dict[str, float] = {}
            disc_map = fb_disc.get(name, {})
            peak_map = fb_peak.get(name, {})
            bands = set(disc_map.keys()) | set(peak_map.keys())
            for band in bands:
                candidates = []
                if band in disc_map:
                    candidates.append(float(disc_map[band]))
                if band in peak_map:
                    candidates.append(float(peak_map[band]))
                if candidates:
                    band_map[band] = min(candidates)
            if band_map:
                fallback_combined[name] = band_map

    fallback_source = fallback_combined or fb_disc or fb_peak
    merged = _merge_mag_maps(base, fallback_source) if fallback_source else base

    # Enforce completeness if requested: every Name in df must have values for all planned bands
    if getattr(cfg, "discovery_error_on_missing", True):
        normalized_cols = {norm for _, norm in _normalize_col_names(df.columns)}
        has_discovery_mag = any("discoverymag" in norm for norm in normalized_cols)
        has_discovery_filter = any(
            "".join(ch for ch in syn.lower() if ch.isalnum()) in normalized_cols
            for syn in _DISC_FILT_SYNONYMS
        )
        planned_set = set(str(b).lower() for b in planned)
        missing: list[str] = []
        incomplete: list[str] = []
        for _, row in df.iterrows():
            name = str(row.get("Name", row.name))
            bandmap = merged.get(name)
            if not bandmap:
                missing.append(name)
                continue
            have = set(str(b).lower() for b in bandmap.keys())
            # restrict to planned bands present in priors/generation
            need = {b for b in planned_set if b in {"u", "g", "r", "i", "z", "y"}}
            if not need.issubset(have):
                incomplete.append(name)
        if missing or incomplete:
            details = []
            if missing:
                details.append(
                    f"missing mags for: {', '.join(missing[:5])}{'...' if len(missing)>5 else ''}"
                )
            if incomplete:
                details.append(
                    f"incomplete mags for: {', '.join(incomplete[:5])}{'...' if len(incomplete)>5 else ''}"
                )
            if has_discovery_mag or has_discovery_filter:
                raise ValueError(
                    "Discovery fallback failed to produce per-band magnitudes for all targets: "
                    + "; ".join(details)
                )

    return merged


def _color_extreme(
    color: str, sign_k: int, cfg: PlannerConfig, is_non_ia: bool
) -> float:
    """Return the color extreme that minimizes the target-band magnitude.

    If the relation is ``target = base + sign_k * color``, choose the extreme
    of ``color`` that minimizes ``target``. For ``sign_k=+1`` choose the lower
    bound; for ``sign_k=-1`` choose the upper bound. If ``is_non_ia`` widen the
    chosen extreme by ``cfg.discovery_non_ia_widen_mag`` further in the same
    direction (more extreme), for saturation safety.
    """
    cmin = cfg.discovery_color_priors_min.get(color, 0.0)
    cmax = cfg.discovery_color_priors_max.get(color, 0.0)
    widen = float(getattr(cfg, "discovery_non_ia_widen_mag", 0.0) or 0.0)
    if sign_k >= 0:
        val = float(cmin)
        if is_non_ia:
            val -= widen
        return val
    else:
        val = float(cmax)
        if is_non_ia:
            val += widen
        return val


def _is_non_ia_type(row: pd.Series) -> bool:
    t = str(row.get("SN_type_raw", "")).strip().lower()
    if not t or t in {"nan", "none", "unknown"}:
        return False
    return "ia" not in t


def _infer_from_discovery_with_priors(
    df: pd.DataFrame, planned_bands: List[str], cfg: PlannerConfig
) -> Dict[str, Dict[str, float]]:
    """Infer per-band mags from discovery mag using ATLAS c/o→r color terms and priors.

    Implements conservative (bright-side) choice of color endpoints to avoid
    saturation. Converts discovery filter to r, then extrapolates to other
    planned bands via chained colors and an extra safety margin in y.
    """
    # Pick discovery magnitude column
    disc_mag_col = None
    canon = dict(_normalize_col_names(df.columns))
    for syn in _DISC_MAG_SYNONYMS:
        key = "".join(ch for ch in syn.lower() if ch.isalnum())
        for orig, norm in canon.items():
            if norm == key:
                disc_mag_col = orig
                break
        if disc_mag_col:
            break
    if disc_mag_col is None:
        return {}

    filt_col = _pick_filter_column(df)
    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        name = str(row.get("Name", row.name))
        dm = row.get(disc_mag_col)
        try:
            dm = float(dm)
        except Exception:
            continue
        disc_f = _norm_filter_string(row.get(filt_col, "")) if filt_col else ""
        non_ia = _is_non_ia_type(row)

        # 1) Find r from discovery
        rmag: Optional[float]
        if disc_f == "r":
            rmag = dm
        elif disc_f == "g":
            # r = g - (g-r) → sign_k = -1
            gr = _color_extreme("g-r", -1, cfg, non_ia)
            rmag = dm - gr
        elif disc_f in ("c", "o"):
            pars = cfg.discovery_atlas_linear.get(disc_f, {"alpha": 0.0, "beta": 0.0})
            alpha = float(pars.get("alpha", 0.0))
            beta = float(pars.get("beta", 0.0))
            # r = m_disc + alpha + beta*(g-r); choose (g-r) extreme based on sign(beta)
            gr = _color_extreme("g-r", -1 if beta < 0 else +1, cfg, non_ia)
            rmag = dm + alpha + beta * gr
        else:
            # unknown discovery filter → conservative copy with margin
            rmag = dm - float(cfg.discovery_margin_mag)

        # 2) Extrapolate to planned bands using priors (bright-side extremes)
        src: Dict[str, float] = {}
        # r is anchor
        if "r" in [b.lower() for b in planned_bands]:
            src["r"] = float(rmag)

        # g = r + (g-r) → sign_k = +1
        if any(str(b).lower() == "g" for b in planned_bands):
            gr = _color_extreme("g-r", +1, cfg, non_ia)
            src["g"] = float(rmag + gr)

        # i = r - (r-i) → sign_k = -1
        if any(str(b).lower() == "i" for b in planned_bands):
            ri = _color_extreme("r-i", -1, cfg, non_ia)
            src["i"] = float(rmag - ri)

        # z = i - (i-z) → compute i if missing
        need_z = any(str(b).lower() == "z" for b in planned_bands)
        if need_z:
            if "i" not in src:
                ri = _color_extreme("r-i", -1, cfg, non_ia)
                src["i"] = float(rmag - ri)
            iz = _color_extreme("i-z", -1, cfg, non_ia)
            src["z"] = float(src["i"] - iz)

        # y = z - (z-y) - Δy
        need_y = any(str(b).lower() == "y" for b in planned_bands)
        if need_y:
            if "z" not in src:
                # build z through i
                if "i" not in src:
                    ri = _color_extreme("r-i", -1, cfg, non_ia)
                    src["i"] = float(rmag - ri)
                iz = _color_extreme("i-z", -1, cfg, non_ia)
                src["z"] = float(src["i"] - iz)
            zy = _color_extreme("z-y", -1, cfg, non_ia)
            y_extra = float(getattr(cfg, "discovery_y_extra_margin_mag", 0.25) or 0.0)
            src["y"] = float(src["z"] - zy - y_extra)

        # u = g + (u-g) → sign_k = +1; build g if missing
        need_u = any(str(b).lower() == "u" for b in planned_bands)
        if need_u:
            if "g" not in src:
                gr = _color_extreme("g-r", +1, cfg, non_ia)
                src["g"] = float(rmag + gr)
            ug = _color_extreme("u-g", +1, cfg, non_ia)
            src["u"] = float(src["g"] + ug)

        # Map to requested bands
        out[name] = {
            b: src[b.lower()]
            for b in (str(x).lower() for x in planned_bands)
            if b in src
        }

    return out


def _infer_peak_from_redshift(
    df: pd.DataFrame, planned_bands: List[str], cfg: PlannerConfig
) -> Dict[str, Dict[str, float]]:
    """Infer conservative peak magnitudes from redshift when present."""

    planned = [str(b).lower() for b in planned_bands]
    if not planned:
        return {}

    H0 = float(getattr(cfg, "H0_km_s_Mpc", 70.0))
    Om = float(getattr(cfg, "Omega_m", 0.3))
    Ol = float(getattr(cfg, "Omega_L", 0.7))
    MB = float(getattr(cfg, "MB_absolute", -19.36))
    alpha = float(getattr(cfg, "SALT2_alpha", 0.14))
    beta = float(getattr(cfg, "SALT2_beta", 3.1))
    K0 = float(getattr(cfg, "Kcorr_approx_mag", 0.0))
    K_by_filter = getattr(cfg, "Kcorr_approx_mag_by_filter", None)
    margin = float(getattr(cfg, "peak_extra_bright_margin_mag", 0.3))

    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        name = str(row.get("Name", row.name))
        z_val = row.get("redshift")
        try:
            zf = float(z_val)
        except Exception:
            continue
        if not np.isfinite(zf) or zf <= 0.0:
            continue
        band_map: Dict[str, float] = {}
        for band in planned:
            k_eff = K0
            if isinstance(K_by_filter, dict):
                k_eff = float(K_by_filter.get(band, K0))
            try:
                m_peak = peak_mag_from_redshift(
                    zf,
                    band,
                    MB=MB,
                    alpha=alpha,
                    beta=beta,
                    H0=H0,
                    Om=Om,
                    Ol=Ol,
                    K_approx=k_eff,
                )
            except Exception:
                continue
            band_map[band] = float(m_peak - margin)
        if band_map:
            out[name] = band_map
    return out


# Common redshift column synonyms (first match wins)
_REDSHIFT_SYNONYMS = [
    "redshift",
    "z",
    "zsn",
    "zspec",
    "z_spec",
    "zphot",
    "z_phot",
    "zbest",
    "z_best",
    "hostz",
    "host_z",
]


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
        s = str(val).strip()
        # Heuristic: sexagesimal with ":" and leading field < 24 → hours
        if ":" in s:
            try:
                lead = float(s.split(":", 1)[0].replace("+", "").replace("-", ""))
            except Exception:
                lead = None
            if lead is not None and 0.0 <= lead < 24.0:
                ang = Angle(s, unit=u.hourangle)
                return float(ang.to(u.deg).value) % 360.0
        ang = Angle(s)
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
    # Guard against all-NaN numeric views (e.g., sexagesimal strings)
    if ra.notna().any():
        try:
            print(
                f"RA raw (numeric) range:  min={np.nanmin(ra):.6f}, max={np.nanmax(ra):.6f}"
            )
        except Exception:
            pass
    else:
        # Try parsing a sample of values as sexagesimal hours
        try:
            vals = df[ra_col].astype(str).head(100).tolist()
            parsed = np.array([_parse_ra_value(v) for v in vals], dtype=float)
            finite = parsed[np.isfinite(parsed)]
            if finite.size:
                print(
                    f"RA parsed (deg) sample:  min={finite.min():.6f}, max={finite.max():.6f}"
                )
            else:
                print("RA appears non-numeric/empty; skipping unit report.")
        except Exception:
            print("RA appears non-numeric; skipping unit report.")

    if dec.notna().any():
        try:
            print(
                f"Dec raw (numeric) range: min={np.nanmin(dec):.6f}, max={np.nanmax(dec):.6f}"
            )
        except Exception:
            pass
    else:
        try:
            vals = df[dec_col].astype(str).head(100).tolist()
            parsed = np.array([_parse_dec_value(v) for v in vals], dtype=float)
            finite = parsed[np.isfinite(parsed)]
            if finite.size:
                print(
                    f"Dec parsed (deg) sample: min={finite.min():.6f}, max={finite.max():.6f}"
                )
            else:
                print("Dec appears non-numeric/empty; skipping unit report.")
        except Exception:
            print("Dec appears non-numeric; skipping unit report.")


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
    """Parse discovery date values into UTC, supporting mixed formats per-row.

    This function is tolerant to heterogeneous time encodings within the same
    column. It recognizes:
    - ISO-like strings (with or without timezone); tz-aware values are
      converted to UTC, tz-naive are assumed to be UTC.
    - Modified Julian Date (MJD) in days (roughly 30,000–90,000).
    - Julian Date (JD) in days (2,400,000–2,500,000).
    - Unix epoch timestamps in seconds, milliseconds, or microseconds. A
      plausibility check retains only conversions yielding years in [1900, 2200].
    - Digit-coded calendar numbers (e.g., YYYYMMDD or YYYYMMDDhhmmss) via a
      string fallback.

    Returns a tz-aware ``datetime64[ns, UTC]`` series.
    """
    # Prepare output (tz-aware) and numeric view for masks
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    numeric = pd.to_numeric(series, errors="coerce")

    # Helper: keep only dates within a reasonable human time span
    def _in_bounds(dt: pd.Series) -> pd.Series:
        try:
            years = dt.dt.year
        except Exception:
            return pd.Series(False, index=dt.index)
        return (years >= 1900) & (years <= 2200)

    # 1) MJD (days)
    mask_mjd = numeric.between(30000, 90000, inclusive="both")
    if mask_mjd.any():
        dt = pd.to_datetime(
            Time(numeric[mask_mjd].values, format="mjd").to_datetime(_dt_timezone.utc)
        )
        out.loc[mask_mjd] = dt

    # 2) JD (days)
    mask_jd = numeric.between(2400000, 2500000, inclusive="both")
    if mask_jd.any():
        dt = pd.to_datetime(
            Time(numeric[mask_jd].values, format="jd").to_datetime(_dt_timezone.utc)
        )
        out.loc[mask_jd] = dt

    # 3) Unix epoch (seconds, milliseconds, microseconds)
    mask_num_remaining = out.isna() & numeric.notna()
    if mask_num_remaining.any():
        nums = numeric[mask_num_remaining]
        # seconds
        try:
            dt_s = pd.to_datetime(
                nums, unit="s", origin="unix", utc=True, errors="coerce"
            )
        except TypeError:
            dt_s = pd.Series(pd.NaT, index=nums.index, dtype="datetime64[ns, UTC]")
        ok_s = _in_bounds(dt_s) & dt_s.notna()
        out.loc[ok_s.index[ok_s]] = dt_s[ok_s]

        # milliseconds
        mask_after_s = out.isna() & numeric.notna()
        if mask_after_s.any():
            nums_ms = numeric[mask_after_s]
            try:
                dt_ms = pd.to_datetime(
                    nums_ms, unit="ms", origin="unix", utc=True, errors="coerce"
                )
            except TypeError:
                dt_ms = pd.Series(
                    pd.NaT, index=nums_ms.index, dtype="datetime64[ns, UTC]"
                )
            ok_ms = _in_bounds(dt_ms) & dt_ms.notna()
            out.loc[ok_ms.index[ok_ms]] = dt_ms[ok_ms]

        # microseconds
        mask_after_ms = out.isna() & numeric.notna()
        if mask_after_ms.any():
            nums_us = numeric[mask_after_ms]
            try:
                dt_us = pd.to_datetime(
                    nums_us, unit="us", origin="unix", utc=True, errors="coerce"
                )
            except TypeError:
                dt_us = pd.Series(
                    pd.NaT, index=nums_us.index, dtype="datetime64[ns, UTC]"
                )
            ok_us = _in_bounds(dt_us) & dt_us.notna()
            out.loc[ok_us.index[ok_us]] = dt_us[ok_us]

    # 4) Fallback: parse the rest via pandas (treat remaining numerics as strings)
    mask_remaining = out.isna()
    if mask_remaining.any():
        # Convert numerics to strings so that values like 20240101 are read as calendar dates
        remainder = series[mask_remaining]
        as_str = remainder.astype(str)
        # Pandas may infer a single format from the first values, which breaks
        # truly mixed-format columns (e.g., "YYYY/MM/DD HH:MM" + ISO-8601). Use
        # the mixed-format parser when available.
        try:
            dt_fallback = pd.to_datetime(as_str, utc=True, errors="coerce", format="mixed")
        except TypeError:
            dt_fallback = pd.to_datetime(as_str, utc=True, errors="coerce")
        # If tz is missing, localize to UTC (pd.to_datetime with utc=True already does this)
        out.loc[mask_remaining] = dt_fallback

    return out


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
    # Backward-compatible handling of WFD SIMLIB-derived catalogs:
    # some pipelines store provenance in the `type` column as "WFD" while
    # also providing `source="WFD"`. When present, treat this as Ia-like so
    # `only_ia` filtering works without requiring users to rewrite catalogs.
    try:
        if "source" in df.columns and type_col and type_col in df.columns:
            src_norm = df["source"].astype(str).str.strip().str.lower()
            type_norm = df[type_col].astype(str).str.strip().str.lower()
            mask = src_norm.eq("wfd") & type_norm.eq("wfd")
            if mask.any():
                df.loc[mask, "SN_type_raw"] = "WFD_Ia"
    except Exception:
        pass
    df["typical_lifetime_days"] = df["SN_type_raw"].apply(
        lambda t: parse_sn_type_to_window_days(t, cfg)
    )
    # Normalized redshift column (best-effort): look for configured column first,
    # else try common synonyms. Non-finite or negative values become NaN.
    red_col = (
        cfg.redshift_column
        if (cfg.redshift_column in df.columns)
        else _fuzzy_pick(df, _REDSHIFT_SYNONYMS)
    )
    if red_col and red_col in df.columns:
        z_num = pd.to_numeric(df[red_col], errors="coerce")
        # reject negatives and absurdly large values
        z_num = z_num.where((z_num >= 0.0) & (z_num < 10.0))
        df["redshift"] = z_num.astype(float)
    else:
        df["redshift"] = np.nan
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
