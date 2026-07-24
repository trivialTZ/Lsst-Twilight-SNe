"""Filter selection policy using an m5/SNR model aligned with LSST practice."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

from .photom_rubin import PhotomConfig, point_source_limiting_mag
from .sky_model import SkyModelConfig, SimpleSkyProvider, sky_mag_arcsec2

MIN_SNR = 5.0


def _m5_scale(
    band: str,
    *,
    m_sky_arcsec2: float,
    seeing_fwhm_arcsec: float,
    airmass: float,
    t_vis_s: float,
    k_m: Optional[float] = None,
    phot_cfg: Optional[PhotomConfig] = None,
) -> float:
    """Return physical 5-sigma depth from the SIMLIB photometric model.

    This wrapper is retained for scheduler call-site compatibility, but the
    calculation lives in :func:`photom_rubin.point_source_limiting_mag` so the
    feasibility gate and generated SIMLIB cannot drift to different noise
    models.
    """

    cfg = phot_cfg or PhotomConfig()
    return point_source_limiting_mag(
        band,
        t_exp_s=float(t_vis_s),
        airmass=float(airmass),
        sky_mag_arcsec2=float(m_sky_arcsec2),
        cfg=cfg,
        fwhm_eff_arcsec=float(seeing_fwhm_arcsec),
        k_m=k_m,
        snr=MIN_SNR,
    )


def heuristic_filters_from_sun_alt(sun_alt_deg: float) -> List[str]:
    """Utility ordering; not used as a feasibility fallback."""
    if sun_alt_deg > -12:
        return ["i", "r", "z", "y"]
    if sun_alt_deg > -15:
        return ["r", "i", "z", "y"]
    return ["g", "r", "i", "z", "y"]


def allowed_filters_for_window(
    target_mag_dict: Dict[str, float],
    sun_alt_deg: float,
    moon_alt_deg: float,
    moon_phase: float,
    moon_sep_deg: float,
    airmass: float,
    seeing_default: float,
    *,
    exposure_by_filter: Optional[Dict[str, float]] = None,
    phot_cfg: Optional["PhotomConfig"] = None,
    sky_provider=None,
    sky_cfg: Optional[SkyModelConfig] = None,
    mjd: Optional[float] = None,
    ra_deg: Optional[float] = None,
    dec_deg: Optional[float] = None,
    sun_sep_deg: Optional[float] = None,
    sun_az_rel_deg: Optional[float] = None,
    sky_lookup: Optional[Callable[..., float]] = None,
) -> List[str]:
    """Select filters that are predicted to reach the target's magnitude.

    Parameters
    ----------
    target_mag_dict : dict of str to float
        Target magnitudes by filter. Missing filters fall back to the ``"r"``
        entry or ``21.5`` if absent.
    sun_alt_deg : float
        Altitude of the Sun in degrees.
    moon_alt_deg : float
        Altitude of the Moon in degrees.
    moon_phase : float
        Fractional lunar illumination in ``[0, 1]``.
    moon_sep_deg : float
        Separation between target and Moon in degrees.
    airmass : float
        Airmass along the line of sight.
    seeing : float
        Seeing FWHM in arcseconds.

    Returns
    -------
    list of str
        Filters deemed viable. When no band passes, an empty list is returned
        and the window policy applies separately.

    Notes
    -----
    This selector is a pruning step; final use is constrained by the
    user-configured ``sun_alt_policy`` at the window stage.
    """

    if phot_cfg is None:
        phot_cfg = PhotomConfig()

    exp_map = exposure_by_filter or {}
    sky_cfg_local = sky_cfg or SkyModelConfig()
    allowed: List[str] = []
    snr_threshold = 2.5 * math.log10(max(1e-3, MIN_SNR / 5.0))

    use_provider = None
    if sky_provider is not None and not isinstance(sky_provider, SimpleSkyProvider):
        use_provider = sky_provider

    # Order candidate filters using a simple twilight heuristic and then add any
    # explicit per-target magnitudes so that we never invent filters when no
    # photometry is available (important in bright twilight tests).
    heuristic_order = heuristic_filters_from_sun_alt(sun_alt_deg)
    seen: set[str] = set()
    candidate_order: List[str] = []

    def _add_candidate(name: str | None) -> None:
        if not name:
            return
        filt_norm = str(name).strip().lower()
        if not filt_norm:
            return
        if filt_norm in seen:
            return
        seen.add(filt_norm)
        candidate_order.append(filt_norm)

    for filt_name in heuristic_order:
        _add_candidate(filt_name)
    for filt_name in target_mag_dict.keys():
        _add_candidate(filt_name)

    if not candidate_order:
        candidate_order = ["g", "r", "i", "z", "y"]

    for filt in candidate_order:
        target_mag = target_mag_dict.get(filt, target_mag_dict.get("r", 21.5))
        t_vis = float(exp_map.get(filt, 30.0))
        if phot_cfg is not None and phot_cfg.fwhm_eff is not None:
            seeing_fwhm = float(phot_cfg.fwhm_eff.get(filt, seeing_default))
        else:
            seeing_fwhm = float(seeing_default)
        k_band = 0.0
        if phot_cfg is not None and phot_cfg.k_m is not None:
            k_band = float(phot_cfg.k_m.get(filt, 0.0))

        # Sky brightness per band
        m_sky: Optional[float] = None
        if (
            sky_lookup is not None
            and mjd is not None
            and ra_deg is not None
            and dec_deg is not None
        ):
            try:
                m_sky = float(
                    sky_lookup(
                        sky_provider=use_provider,
                        sky_cfg=sky_cfg_local,
                        ra_deg=float(ra_deg),
                        dec_deg=float(dec_deg),
                        band=filt,
                        airmass=float(airmass),
                        mjd=float(mjd),
                        minutes=1,
                        sun_alt_deg=float(sun_alt_deg),
                        sun_sep_deg=(
                            float(sun_sep_deg) if sun_sep_deg is not None else None
                        ),
                        sun_az_rel_deg=(
                            float(sun_az_rel_deg)
                            if sun_az_rel_deg is not None
                            else None
                        ),
                        moon_alt_deg=float(moon_alt_deg),
                        moon_phase=float(moon_phase),
                        moon_sep_deg=float(moon_sep_deg),
                        k_band=float(k_band),
                    )
                )
            except Exception:
                m_sky = None
        if m_sky is None:
            if use_provider is not None:
                try:
                    m_sky = float(
                        use_provider.sky_mag(
                            mjd, ra_deg, dec_deg, filt, airmass  # type: ignore[arg-type]
                        )
                    )
                except Exception:
                    m_sky = None
            if m_sky is None:
                m_sky = sky_mag_arcsec2(
                    filt,
                    sky_cfg_local,
                    sun_alt_deg,
                    moon_alt_deg,
                    moon_phase,
                    moon_sep_deg,
                    airmass,
                    sun_sep_deg=sun_sep_deg,
                    sun_az_rel_deg=sun_az_rel_deg,
                    k_band=k_band,
                )

        m5 = _m5_scale(
            filt,
            m_sky_arcsec2=m_sky,
            seeing_fwhm_arcsec=seeing_fwhm,
            airmass=airmass,
            t_vis_s=t_vis,
            k_m=k_band,
            phot_cfg=phot_cfg,
        )

        if float(m5) - float(target_mag) >= snr_threshold:
            allowed.append(filt)

    return allowed
