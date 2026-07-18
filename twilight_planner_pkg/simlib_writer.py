from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, TextIO


PSF_UNIT_PIXEL_SIGMA = "PIXEL_SIGMA"
PSF_UNIT_ARCSEC_FWHM = "ARCSEC_FWHM"
SKYSIG_UNIT_ADU_PER_SQPIXEL = "ADU_PER_SQPIXEL"
SKYSIG_UNIT_ADU_PER_SQARCSEC = "ADU_PER_SQARCSEC"
FWHM_SIGMA_RATIO = 2.355


def normalize_psf_unit(value: str) -> str:
    """Return a SNANA-supported PSF unit for three-column Gaussian rows."""

    unit = str(value).strip().upper()
    aliases = {
        "PIXEL": PSF_UNIT_PIXEL_SIGMA,
        "PIXELS": PSF_UNIT_PIXEL_SIGMA,
        "PIXEL_SIGMA": PSF_UNIT_PIXEL_SIGMA,
        "NORM": PSF_UNIT_PIXEL_SIGMA,
        "ARCSEC": PSF_UNIT_ARCSEC_FWHM,
        "ARCSEC_FWHM": PSF_UNIT_ARCSEC_FWHM,
    }
    try:
        return aliases[unit]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported PSF_UNIT {value!r}; expected PIXEL_SIGMA or ARCSEC_FWHM"
        ) from exc


def normalize_skysig_unit(value: str) -> str:
    """Return a SNANA-supported sky-RMS surface unit."""

    unit = str(value).strip().upper()
    aliases = {
        "ADU_PER_PIXEL": SKYSIG_UNIT_ADU_PER_SQPIXEL,
        "ADU_PER_SQPIXEL": SKYSIG_UNIT_ADU_PER_SQPIXEL,
        "ADU_PER_ARCSEC": SKYSIG_UNIT_ADU_PER_SQARCSEC,
        "ADU_PER_SQARCSEC": SKYSIG_UNIT_ADU_PER_SQARCSEC,
    }
    try:
        return aliases[unit]
    except KeyError as exc:
        raise ValueError(
            "Unsupported SKYSIG_UNIT "
            f"{value!r}; expected ADU_PER_SQPIXEL or ADU_PER_SQARCSEC"
        ) from exc


def psf_sigma_pixels_to_simlib(
    value: float, *, pixel_scale_arcsec: float, psf_unit: str
) -> float:
    """Convert Gaussian sigma in pixels to the requested SIMLIB PSF unit."""

    unit = normalize_psf_unit(psf_unit)
    if unit == PSF_UNIT_PIXEL_SIGMA:
        return float(value)
    return float(value) * float(pixel_scale_arcsec) * FWHM_SIGMA_RATIO


def psf_sigma_pixels_from_simlib(
    value: float, *, pixel_scale_arcsec: float, psf_unit: str
) -> float:
    """Convert a SIMLIB PSF value to Gaussian sigma in pixels."""

    unit = normalize_psf_unit(psf_unit)
    if unit == PSF_UNIT_PIXEL_SIGMA:
        return float(value)
    denom = float(pixel_scale_arcsec) * FWHM_SIGMA_RATIO
    if not math.isfinite(denom) or denom <= 0.0:
        raise ValueError("pixel_scale_arcsec must be finite and positive")
    return float(value) / denom


def skysig_adu_per_pixel_to_simlib(
    value: float, *, pixel_scale_arcsec: float, skysig_unit: str
) -> float:
    """Convert sky RMS in ADU/pixel to the requested SIMLIB sky unit."""

    unit = normalize_skysig_unit(skysig_unit)
    if unit == SKYSIG_UNIT_ADU_PER_SQPIXEL:
        return float(value)
    pixel_scale = float(pixel_scale_arcsec)
    if not math.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise ValueError("pixel_scale_arcsec must be finite and positive")
    # SNANA converts ADU/sqrt(arcsec^2) back to ADU/pixel by multiplying
    # by PIXSIZE, so serialization performs the inverse operation.
    return float(value) / pixel_scale


def skysig_adu_per_pixel_from_simlib(
    value: float, *, pixel_scale_arcsec: float, skysig_unit: str
) -> float:
    """Convert a SIMLIB sky RMS value to ADU/pixel."""

    unit = normalize_skysig_unit(skysig_unit)
    if unit == SKYSIG_UNIT_ADU_PER_SQPIXEL:
        return float(value)
    pixel_scale = float(pixel_scale_arcsec)
    if not math.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise ValueError("pixel_scale_arcsec must be finite and positive")
    return float(value) * pixel_scale


@dataclass
class SimlibHeader:
    """Header information for a SIMLIB file."""

    SURVEY: str = "LSST_TWILIGHT"
    FILTERS: str = "ugrizY"
    PIXSIZE: float = 0.200
    NPE_PIXEL_SATURATE: int = 80000
    PHOTFLAG_SATURATE: int = 2048
    PSF_UNIT: str = PSF_UNIT_PIXEL_SIGMA
    SKYSIG_UNIT: str = SKYSIG_UNIT_ADU_PER_SQPIXEL

    def __post_init__(self) -> None:
        # SNANA expects the y-band to be written as an uppercase 'Y'
        # even though the rest of the bands are lowercase. Normalise
        # here so callers do not have to remember the convention.
        try:
            filt_str = str(self.FILTERS)
        except Exception:
            filt_str = ""
        self.FILTERS = "".join("Y" if c == "y" else c for c in filt_str)
        self.PSF_UNIT = normalize_psf_unit(self.PSF_UNIT)
        self.SKYSIG_UNIT = normalize_skysig_unit(self.SKYSIG_UNIT)
        if not math.isfinite(float(self.PIXSIZE)) or float(self.PIXSIZE) <= 0.0:
            raise ValueError("SimlibHeader.PIXSIZE must be finite and positive")


class SimlibWriter:
    """Minimal SNANA SIMLIB writer."""

    def __init__(self, fp: TextIO, header: SimlibHeader, *, preserve_ids: bool = False):
        """Create a writer.

        Parameters
        ----------
        fp : TextIO
            Open file-like object for writing.
        header : SimlibHeader
            Header configuration.
        preserve_ids : bool, optional
            When ``True``, LIBID numbers provided to :meth:`start_libid` are
            preserved in the output. When ``False`` (default), LIBIDs are
            renumbered sequentially as in the original implementation.
        """

        self.fp = fp
        self.header = header
        # Per-LIBID running index for epoch IDs when streaming (unused in buffered mode)
        self._idx = 0
        # Track the "open" group key while the caller is between start_libid/end_libid.
        self._current_key: Optional[str] = None
        # Buffered groups keyed by SN key (comment if present, else RA/DEC key).
        # Insertion order is preserved (Python 3.7+), used for sequential LIBID renumbering.
        self._groups: dict[str, dict] = {}
        # Preserve first-seen order of SN keys (explicit list for clarity).
        self._order: list[str] = []
        self._preserve_ids = bool(preserve_ids)

    def write_header(self) -> None:
        """Write the SIMLIB DOCUMENTATION and global header (no LIBID blocks).

        Per SNANA conventions, a DOCUMENTATION block and the global header
        come first. The "BEGIN LIBGEN" marker is emitted at the start of
        close(), just before the per-LIBID blocks are written.
        """

        h = self.header
        w = self.fp.write
        # Required DOCUMENTATION block (minimal content)
        w("DOCUMENTATION:\n")
        w("  PURPOSE: LSST twilight cadence library for SNANA simulation\n")
        w("  INTENT: Nominal\n")
        w("  USAGE_KEY: SIMLIB_FILE\n")
        w("  NOTE: NOISE is read noise in electrons/pixel.\n")
        w("  NOTE: SKYSIG excludes read noise and follows SKYSIG_UNIT.\n")
        w("  NOTE: PSF1/PSF2 follow PSF_UNIT.\n")
        w("DOCUMENTATION_END:\n\n")
        # Global header keys
        w("# Generated by twilight_planner\n")
        w(f"SURVEY:   {h.SURVEY}\n")
        w(f"FILTERS:  {h.FILTERS}\n")
        w(f"PIXSIZE:  {h.PIXSIZE:.3f}\n")
        w(f"PSF_UNIT: {h.PSF_UNIT}\n")
        w(f"SKYSIG_UNIT: {h.SKYSIG_UNIT}\n")
        w(f"NPE_PIXEL_SATURATE: {h.NPE_PIXEL_SATURATE}\n")
        w(f"PHOTFLAG_SATURATE:  {h.PHOTFLAG_SATURATE}\n")
        w("\n")

    def start_libid(
        self,
        libid: int,
        ra_deg: float,
        dec_deg: float,
        nobs: int,
        comment: str = "",
        *,
        mwebv: float | None = None,
        redshift: float | None = None,
        peakmjd: float | None = None,
    ) -> None:
        """Begin a new logical LIBID group for a target (BUFFERED; no file I/O).

        We aggregate all subsequent add_epoch() calls into a per-SN buffer keyed by
        `comment` (preferred) or by a RA/DEC-derived fallback key if `comment` is empty.
        The actual SIMLIB block for this SN is written only at close(), with NOBS
        recomputed from the buffered epochs. Unless ``preserve_ids=True`` was set at
        construction time, LIBIDs are renumbered sequentially.
        """
        # Choose a stable group key: prefer the SN name/comment; else RA/DEC key.
        key = comment.strip() if comment else f"{ra_deg:.6f},{dec_deg:.6f}"
        self._current_key = key
        # First time we see this SN key => create the group and remember insertion order.
        if key not in self._groups:
            self._groups[key] = {
                # We ignore the caller's libid when writing; we renumber at close().
                "first_seen_libid": libid,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "comment": comment.strip() if comment else "",
                "mwebv": mwebv,
                "redshift": redshift,
                "peakmjd": peakmjd,
                "epochs": [],  # list of dicts with epoch fields
            }
            self._order.append(key)
        else:
            # If RA/DEC vary across visits for the same key, keep first-seen values.
            # (Optional sanity: could assert closeness; here we just ignore new coords.)
            # Merge in redshift/peakmjd if missing.
            g = self._groups[key]
            if g.get("redshift") is None and redshift is not None:
                g["redshift"] = redshift
            if g.get("peakmjd") is None and peakmjd is not None:
                g["peakmjd"] = peakmjd
            if g.get("mwebv") is None and mwebv is not None:
                g["mwebv"] = mwebv
        # Reset per-LIBID epoch counter used ONLY when streaming (we re-ID at flush).
        self._idx = 0

    def add_epoch(
        self,
        mjd: float,
        band: str,
        gain: float,
        noise: float,
        skysig: float,
        psf1: float,
        psf2: float,
        psfratio: float,
        zpavg: float,
        zperr: float,
        *,
        mag: float = -99.0,
        nexpose: int = 1,
    ) -> None:
        """Buffer an ``S:`` row in planner-canonical units.

        ``noise`` is read noise in electrons/pixel, ``skysig`` is sky RMS in
        ADU/pixel, and ``psf1``/``psf2`` are Gaussian sigmas in pixels.  The
        latter two are converted at serialization according to the global
        header; ``noise`` is never divided by gain.
        """
        if self._current_key is None:
            # Defensive: if caller forgot start_libid(), ignore silently or raise.
            # We choose to no-op to preserve backward compatibility.
            return
        self._groups[self._current_key]["epochs"].append(
            {
                "mjd": mjd,
                "band": band,
                "gain": gain,
                "noise": noise,
                "skysig": skysig,
                "psf1": psf1,
                "psf2": psf2,
                "psfratio": psfratio,
                "zpavg": zpavg,
                "zperr": zperr,
                "mag": mag,
                "nexpose": max(1, int(round(nexpose))),
            }
        )
        # Keep a local counter to mimic old behavior (unused at flush).
        self._idx += 1

    def end_libid(self) -> None:
        """Terminate the current logical LIBID group (BUFFERED; no file I/O)."""
        self._current_key = None

    def close(self) -> None:
        """Flush buffered groups as one LIBID per SN, then close the file handle."""
        # Emit groups in first-seen order with sequential LIBIDs starting at 1.
        w = self.fp.write
        h = self.header
        # Per SNANA format, mark the start of LIBGEN now (after global header)
        w("BEGIN LIBGEN\n\n")
        libid_out = 1
        for key in self._order:
            g = self._groups[key]
            epochs = g["epochs"]
            if not epochs:
                continue  # no observations buffered for this SN
            ra = g["ra_deg"]
            dec = g["dec_deg"]
            comment = g["comment"]
            mwebv = g.get("mwebv")
            nobs = len(epochs)
            # REDSHIFT and PEAKMJD (optional):
            z = g.get("redshift")
            pmjd = g.get("peakmjd")
            if pmjd is None:
                try:
                    mjds = [
                        float(e.get("mjd")) for e in epochs if e.get("mjd") is not None
                    ]
                    if mjds:
                        pmjd = 0.5 * (min(mjds) + max(mjds))
                except Exception:
                    pmjd = None
            # Header for this LIBID block.  The SN comment (if any) is appended on
            # the LIBID line to mirror typical SNANA formatting like:
            #   LIBID:      236     # from SNID=000236
            w("#--------------------------------------------\n")
            libid_val = libid_out
            if self._preserve_ids:
                try:
                    libid_val = int(g.get("first_seen_libid", libid_out))
                except Exception:
                    libid_val = libid_out
            libid_line = f"LIBID: {libid_val:7d}"
            if comment:
                libid_line += f"     # {comment}"
            w(f"{libid_line}\n")
            nobs_line = (
                f"NOBS: {nobs}  PIXSIZE:  {h.PIXSIZE:.3f}  RA: {ra:11.6f}  DEC: {dec:11.6f}"
            )
            if isinstance(mwebv, (int, float)):
                nobs_line += f"  MWEBV:  {float(mwebv):.4f}"
            w(f"{nobs_line}\n")
            if (z is not None) or (pmjd is not None):
                z_str = f"{z:.5f}" if isinstance(z, (int, float)) else ""
                pmjd_str = f"{pmjd:11.3f}" if isinstance(pmjd, (int, float)) else ""
                w(f"REDSHIFT:  {z_str:>7s}     PEAKMJD: {pmjd_str}\n")
            w(
                "#     MJD        ID+NEXPOSE FLT GAIN NOISE SKYSIG PSF1 PSF2 PSFRATIO ZPTAVG ZPTERR\n"
            )
            # Epochs (re-ID from 1..N to guarantee consistency)
            for idx, e in enumerate(epochs, start=1):
                # Normalize band: emit 'Y' (uppercase) for y-band to match SNANA expectations
                band = e.get("band", "")
                if band == "y":
                    band = "Y"
                nexpose_val = e.get("nexpose", 1)
                try:
                    nexpose = int(round(float(nexpose_val)))
                except Exception:
                    nexpose = 1
                if nexpose < 1:
                    nexpose = 1
                id_field = f"{idx}*{nexpose}" if nexpose > 1 else f"{idx}"
                skysig_out = skysig_adu_per_pixel_to_simlib(
                    e["skysig"],
                    pixel_scale_arcsec=h.PIXSIZE,
                    skysig_unit=h.SKYSIG_UNIT,
                )
                psf1_out = psf_sigma_pixels_to_simlib(
                    e["psf1"],
                    pixel_scale_arcsec=h.PIXSIZE,
                    psf_unit=h.PSF_UNIT,
                )
                psf2_out = psf_sigma_pixels_to_simlib(
                    e["psf2"],
                    pixel_scale_arcsec=h.PIXSIZE,
                    psf_unit=h.PSF_UNIT,
                )
                w(
                    f"S: {e['mjd']:11.4f}  {id_field:>11s}  {band:1s}  "
                    f"{e['gain']:5.2f}  {e['noise']:5.2f}  {skysig_out:7.2f}  "
                    f"{psf1_out:7.3f}  {psf2_out:7.3f}  {e['psfratio']:7.3f}  "
                    f"{e['zpavg']:6.3f}  {e['zperr']:5.3f}\n"
                )
            w("END_LIBID:\n\n")
            if not self._preserve_ids:
                libid_out += 1
        # End marker then close
        w("END_OF_SIMLIB:\n")
        self.fp.close()
