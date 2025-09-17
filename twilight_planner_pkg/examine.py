from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd

from .config import PlannerConfig
from .io_utils import build_mag_lookup_with_fallback, standardize_columns
from .io_utils import _MAG_SYNONYMS, _normalize_col_names  # type: ignore
from .photom_rubin import (
    PhotomConfig,
    airmass_from_alt_deg,
    central_pixel_electrons,
    central_pixel_fraction_gaussian,
    electrons_per_pixel_from_sb,
    epoch_zeropoints,
)


def build_saturation_df(
    perSN_df: pd.DataFrame,
    csv_path: str,
    cfg: PlannerConfig,
) -> pd.DataFrame:
    """Return a diagnostic DataFrame with per-visit saturation components.

    For each planned visit in ``perSN_df``, compute central-pixel electrons
    contributed by the SN point source (``e_src``), the sky background
    (``e_sky``), and the host galaxy diffuse light (``e_host``). Also report
    their sum (``total_e``), the saturation limit used by the planner, and
    whether the plan applied a saturation guard.

    To mirror planning-time calculations, this function:
    - reconstructs the per-filter effective FWHM, pixel scale and gain from
      ``PlannerConfig`` and ``PhotomConfig`` defaults;
    - uses the row's recorded ``ZPT`` (ADU) and ``GAIN`` to recover the per-row
      electron zeropoint (``ZPT_pe``) for the actual exposure time;
    - falls back to recomputing zeropoints from exposure, airmass and extinction
      when the row lacks ``ZPT``/``GAIN``;
    - uses the row's ``sky_mag_arcsec2`` for the sky term;
    - uses the configured default host SB (if enabled) when no per-target host
      SB was available at planning time.
    """

    # Source magnitudes per target (by filter)
    # Use discovery‑magnitude fallback so src_mag is populated even when
    # explicit per‑band columns (e.g., gmag/rmag) are missing in the input CSV.
    raw = pd.read_csv(csv_path)
    cat = standardize_columns(raw, cfg)
    mag_lookup: Dict[str, Dict[str, float]] = build_mag_lookup_with_fallback(cat, cfg)

    phot_cfg = PhotomConfig(
        pixel_scale_arcsec=cfg.pixel_scale_arcsec,
        zpt1s=cfg.zpt1s or None,
        k_m=cfg.k_m or None,
        fwhm_eff=cfg.fwhm_eff or None,
        read_noise_e=cfg.read_noise_e,
        gain_e_per_adu=cfg.gain_e_per_adu,
        zpt_err_mag=cfg.zpt_err_mag,
        npe_pixel_saturate=int(getattr(cfg, "simlib_npe_pixel_saturate", 100_000) or 100_000),
        npe_pixel_warn_nonlinear=int(
            0.8 * (getattr(cfg, "simlib_npe_pixel_saturate", 100_000) or 100_000)
        ),
    )

    rows: List[Dict] = []
    for _, r in perSN_df.iterrows():
        name = str(r.get("SN", "")).strip()
        band = str(r.get("filter", "")).strip().lower()
        t_exp = float(r.get("t_exp_s", r.get("exposure_s", float(cfg.exposure_by_filter.get(band, 0.0)))))
        alt = float(r.get("alt_deg", math.nan))
        air = airmass_from_alt_deg(alt) if (alt == alt) else math.nan  # NaN check
        gain = float(r.get("GAIN", getattr(cfg, "gain_e_per_adu", 1.0)))
        zptavg = r.get("ZPT")
        sky_mag = r.get("sky_mag_arcsec2")
        sat_guard = bool(r.get("saturation_guard_applied", False))
        warn_nl = bool(r.get("warn_nonlinear", False))

        # Reconstruct ZPT_pe in electrons
        if pd.notna(zptavg):
            zpt_pe = float(zptavg) + 2.5 * math.log10(max(1e-6, gain))
        else:
            # Fallback: derive from exposure, extinction and airmass
            if band in (phot_cfg.k_m or {}):
                X = air if (air == air) else 1.0
                zpt_pe, _ = epoch_zeropoints(
                    (phot_cfg.zpt1s or {}).get(band, 28.0),
                    max(1e-3, t_exp),
                    (phot_cfg.k_m or {}).get(band, 0.1),
                    X,
                    gain,
                )
            else:
                zpt_pe = 28.0  # conservative fallback

        # Central pixel fraction from Gaussian PSF
        fwhm = (phot_cfg.fwhm_eff or {}).get(band, 0.85)
        frac = central_pixel_fraction_gaussian(fwhm, phot_cfg.pixel_scale_arcsec)

        # Source magnitude in this band (may be missing)
        mags = mag_lookup.get(name, {})
        src_mag = mags.get(band)
        e_src = (
            float(central_pixel_electrons(float(src_mag), zpt_pe, frac))
            if (src_mag is not None and pd.notna(src_mag))
            else math.nan
        )

        npe_sat = float(phot_cfg.npe_pixel_saturate)
        frac_safe = max(1e-6, frac)
        m_sat = float(zpt_pe - 2.5 * math.log10(npe_sat / frac_safe))
        margin_mag = (
            float(src_mag) - m_sat
            if (src_mag is not None and pd.notna(src_mag))
            else math.nan
        )

        # Sky electrons per pixel (requires sky surface brightness)
        e_sky = (
            float(electrons_per_pixel_from_sb(float(sky_mag), zpt_pe, phot_cfg.pixel_scale_arcsec))
            if pd.notna(sky_mag)
            else math.nan
        )

        # Host electrons per pixel: use configured default host SB if enabled
        mu_host = None
        if getattr(cfg, "use_default_host_sb", False):
            mu_host = (getattr(cfg, "default_host_mu_arcsec2_by_filter", {}) or {}).get(band, 22.0)
        e_host = (
            float(electrons_per_pixel_from_sb(float(mu_host), zpt_pe, phot_cfg.pixel_scale_arcsec))
            if mu_host is not None
            else 0.0
        )

        # Sum contributions; if e_src is NaN, ignore it in the sum but mark missing
        missing_src = not pd.notna(src_mag)
        e_src_safe = 0.0 if (e_src != e_src) else float(e_src)
        e_sky_safe = 0.0 if (e_sky != e_sky) else float(e_sky)
        total_e = e_src_safe + e_host + e_sky_safe
        rate_e_per_s = (e_src_safe + e_sky_safe + e_host) / max(1e-6, float(t_exp))
        t_sat = npe_sat / max(1e-6, rate_e_per_s)

        # Also compute contributions at the baseline (max) exposure time
        base_t = float(cfg.exposure_by_filter.get(band, t_exp))
        if base_t <= 0:
            base_t = t_exp
        # Zeropoint for baseline exposure at same airmass
        try:
            Xb = air if (air == air) else 1.0
            zpt_pe_base, _ = epoch_zeropoints(
                (phot_cfg.zpt1s or {}).get(band, 28.0),
                max(1e-3, base_t),
                (phot_cfg.k_m or {}).get(band, 0.1),
                Xb,
                gain,
            )
        except Exception:
            zpt_pe_base = zpt_pe
        e_src_base = (
            float(central_pixel_electrons(float(src_mag), zpt_pe_base, frac))
            if (src_mag is not None and pd.notna(src_mag))
            else math.nan
        )
        e_sky_base = (
            float(electrons_per_pixel_from_sb(float(sky_mag), zpt_pe_base, phot_cfg.pixel_scale_arcsec))
            if pd.notna(sky_mag)
            else math.nan
        )
        e_host_base = (
            float(electrons_per_pixel_from_sb(float(mu_host), zpt_pe_base, phot_cfg.pixel_scale_arcsec))
            if mu_host is not None
            else 0.0
        )
        e_src_base_safe = 0.0 if (e_src_base != e_src_base) else float(e_src_base)
        e_sky_base_safe = 0.0 if (e_sky_base != e_sky_base) else float(e_sky_base)
        total_e_base = e_src_base_safe + e_host_base + e_sky_base_safe

        rows.append(
            {
                "SN": name,
                "filter": band,
                "t_exp_s": float(t_exp),
                "t_exp_s_base": float(base_t),
                "airmass": float(air) if (air == air) else math.nan,
                "alt_deg": float(alt) if (alt == alt) else math.nan,
                "sky_mag_arcsec2": float(sky_mag) if pd.notna(sky_mag) else math.nan,
                "src_mag": float(src_mag) if (src_mag is not None and pd.notna(src_mag)) else math.nan,
                "host_mu_arcsec2": float(mu_host) if mu_host is not None else math.nan,
                "GAIN": float(gain),
                "ZPTAVG": float(zptavg) if pd.notna(zptavg) else math.nan,
                "ZPT_pe": float(zpt_pe),
                "frac_central": float(frac),
                "e_src": float(e_src) if (e_src == e_src) else math.nan,
                "e_sky": float(e_sky) if (e_sky == e_sky) else math.nan,
                "e_host": float(e_host),
                "total_e": float(total_e),
                "m_sat": float(m_sat),
                "margin_mag": float(margin_mag) if margin_mag == margin_mag else math.nan,
                "t_sat_s": float(t_sat),
                "ZPT_pe_base": float(zpt_pe_base),
                "e_src_base": float(e_src_base) if (e_src_base == e_src_base) else math.nan,
                "e_sky_base": float(e_sky_base) if (e_sky_base == e_sky_base) else math.nan,
                "e_host_base": float(e_host_base),
                "total_e_base": float(total_e_base),
                "sat_limit": float(phot_cfg.npe_pixel_saturate),
                "warn_limit": float(phot_cfg.npe_pixel_warn_nonlinear),
                "sat_guard_applied": bool(sat_guard),
                "warn_nonlinear": bool(warn_nl),
                "missing_src_mag": bool(missing_src),
            }
        )

    df = pd.DataFrame.from_records(rows)
    # Helpful deltas
    df["margin_e"] = df["sat_limit"] - df["total_e"]
    df["over_sat"] = df["total_e"] - df["sat_limit"]
    df["margin_e_base"] = df["sat_limit"] - df["total_e_base"]
    df["over_sat_base"] = df["total_e_base"] - df["sat_limit"]
    return df


def diagnose_mag_mapping(csv_path: str, cfg: PlannerConfig):
    """Diagnose why ``src_mag`` may be NaN by reporting column matches.

    Returns a tuple ``(band_to_col, maglike_cols, preview_df)`` where
    - ``band_to_col`` maps each band to the column chosen via the internal
      synonym list;
    - ``maglike_cols`` is a list of columns whose normalized names contain
      the substring ``'mag'`` (useful hints if no synonym matched);
    - ``preview_df`` shows the first few rows for the matched columns.
    """
    import pandas as pd

    raw = pd.read_csv(csv_path)
    df = standardize_columns(raw, cfg)
    canon = dict(_normalize_col_names(df.columns))

    band_to_col: Dict[str, str | None] = {b: None for b in _MAG_SYNONYMS.keys()}
    for band, syns in _MAG_SYNONYMS.items():
        for syn in syns:
            key = "".join(ch for ch in syn.lower() if ch.isalnum())
            for orig, norm in canon.items():
                if norm == key:
                    band_to_col[band] = orig
                    break
            if band_to_col[band] is not None:
                break

    maglike_cols = [orig for orig, norm in canon.items() if "mag" in norm]

    cols = [c for c in band_to_col.values() if c is not None]
    preview = df[["Name"] + cols].head(10) if cols else df.head(5)
    return band_to_col, maglike_cols, preview
