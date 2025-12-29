from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Iterable, Literal, overload

import numpy as np
import pandas as pd
from astropy.io import fits


_DATACLASS_KWARGS: dict[str, bool] = {}
if "slots" in inspect.signature(dataclass).parameters:
    _DATACLASS_KWARGS["slots"] = True


@dataclass(frozen=True, **_DATACLASS_KWARGS)
class SnanaFitsPair:
    head_path: Path
    phot_path: Path


def _first_bintable_hdu_index(hdul: fits.HDUList) -> int:
    for i, hdu in enumerate(hdul):
        if isinstance(hdu, fits.BinTableHDU):
            return i
    raise ValueError("No BinTableHDU found in FITS file.")


@overload
def read_fits_bintable(path: str | Path, *, ext: int | str | None = None) -> fits.FITS_rec: ...


def read_fits_bintable(path: str | Path, *, ext: int | str | None = None) -> fits.FITS_rec:
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        if ext is None:
            ext = _first_bintable_hdu_index(hdul)
        return hdul[ext].data  # type: ignore[return-value]


def _maybe_decode_bytes(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "a"}:
        decoded = np.char.decode(values, "utf-8", errors="replace")
        return np.char.strip(decoded)
    if values.dtype.kind == "U":
        return np.char.strip(values)
    return values


def _ensure_native_endian(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"O", "U", "S", "a"}:
        return values
    if values.dtype.byteorder in {"=", "|"}:
        return values
    # FITS tables are often big-endian; pandas doesn't accept big-endian buffers.
    swapped = values.byteswap()
    return swapped.view(values.dtype.newbyteorder("="))


def fits_rec_to_dataframe(table: fits.FITS_rec | np.ndarray) -> pd.DataFrame:
    names = getattr(table, "names", None)
    if names is None:
        names = list(getattr(table, "dtype").names or [])

    data: dict[str, np.ndarray] = {}
    for name in names:
        values = np.asarray(table[name])
        values = _ensure_native_endian(values)
        data[name] = _maybe_decode_bytes(values)
    return pd.DataFrame(data)


def _find_list_file(genversion_dir: Path) -> Path | None:
    list_paths = sorted(genversion_dir.glob("*.LIST"))
    if len(list_paths) == 1:
        return list_paths[0]
    return None


def _resolve_head_to_phot(head_path: Path) -> Path:
    name = head_path.name
    if "_HEAD.FITS.gz" in name:
        phot_name = name.replace("_HEAD.FITS.gz", "_PHOT.FITS.gz")
    elif "_HEAD.FITS" in name:
        phot_name = name.replace("_HEAD.FITS", "_PHOT.FITS")
    else:
        raise ValueError(f"Unrecognized HEAD filename pattern: {head_path.name}")

    phot_path = head_path.with_name(phot_name)
    if phot_path.exists():
        return phot_path

    # Fall back: if we found a .gz HEAD but only uncompressed PHOT exists, or vice-versa.
    if phot_path.suffixes[-2:] == [".FITS", ".gz"]:
        alt = phot_path.with_suffix("")  # drop .gz
        if alt.exists():
            return alt
    if phot_path.suffix == ".FITS":
        alt = phot_path.with_suffix(".FITS.gz")
        if alt.exists():
            return alt

    raise FileNotFoundError(f"Could not find matching PHOT for HEAD: {head_path}")


def list_snana_fits_pairs(genversion_dir: str | Path) -> list[SnanaFitsPair]:
    genversion_dir = Path(genversion_dir)
    list_file = _find_list_file(genversion_dir)
    head_paths: list[Path] = []

    if list_file is not None:
        for line in list_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            head_paths.append(genversion_dir / line)
    else:
        head_paths = sorted(genversion_dir.glob("*_HEAD.FITS*"))

    pairs: list[SnanaFitsPair] = []
    for head_path in head_paths:
        if not head_path.exists():
            raise FileNotFoundError(f"HEAD file listed but not found: {head_path}")
        pairs.append(SnanaFitsPair(head_path=head_path, phot_path=_resolve_head_to_phot(head_path)))
    return pairs


PointerBase = Literal[0, 1, "auto"]


def _score_pointer_base_for_row(
    *,
    pointer_base: Literal[0, 1],
    phot: fits.FITS_rec,
    pmin: int,
    pmax: int,
    nobs: int,
    marker_mjd: float,
) -> float:
    nphot = len(phot)

    if pointer_base == 1:
        start = pmin - 1
        stop = pmax
        if start < 0 or stop < 0 or start > stop:
            return 1e9
        obs_len = max(0, min(stop, nphot) - max(start, 0))
        marker_ok = 0 <= pmax < nphot and float(phot[pmax]["MJD"]) == marker_mjd
        return abs(obs_len - nobs) + (0.0 if marker_ok else 5.0)

    # pointer_base == 0: assume pointers refer to 0-based obs slice endpoints, inclusive.
    start = pmin
    stop_inclusive = pmax
    if start < 0 or stop_inclusive < 0 or start > stop_inclusive:
        return 1e9
    stop = stop_inclusive + 1
    sl = phot[max(start, 0) : min(stop, nphot)]
    mjd = np.asarray(sl["MJD"], dtype=float)
    marker_count = int(np.sum(mjd == marker_mjd))
    obs_len = len(sl) - marker_count
    marker_ok = marker_count >= 1
    return abs(obs_len - nobs) + (0.0 if marker_ok else 5.0)


def infer_pointer_base(
    head: fits.FITS_rec,
    phot: fits.FITS_rec,
    *,
    marker_mjd: float = -777.0,
    sample_size: int = 50,
    random_state: int = 0,
) -> Literal[0, 1]:
    if len(head) == 0:
        raise ValueError("HEAD table is empty.")
    if len(phot) == 0:
        raise ValueError("PHOT table is empty.")

    rng = np.random.default_rng(random_state)
    idxs = np.arange(len(head))
    if len(idxs) > sample_size:
        idxs = rng.choice(idxs, size=sample_size, replace=False)

    score_0 = 0.0
    score_1 = 0.0
    for i in idxs:
        row = head[i]
        pmin = int(row["PTROBS_MIN"])
        pmax = int(row["PTROBS_MAX"])
        nobs = int(row["NOBS"])
        score_0 += _score_pointer_base_for_row(
            pointer_base=0, phot=phot, pmin=pmin, pmax=pmax, nobs=nobs, marker_mjd=marker_mjd
        )
        score_1 += _score_pointer_base_for_row(
            pointer_base=1, phot=phot, pmin=pmin, pmax=pmax, nobs=nobs, marker_mjd=marker_mjd
        )

    return 1 if score_1 <= score_0 else 0


def annotate_phot_with_snid(
    head: fits.FITS_rec,
    phot: fits.FITS_rec,
    *,
    pointer_base: PointerBase = "auto",
    marker_mjd: float = -777.0,
    validate: bool = True,
) -> np.ndarray:
    if pointer_base == "auto":
        pointer_base = infer_pointer_base(head, phot, marker_mjd=marker_mjd)

    snid_by_row = np.full(len(phot), fill_value=-1, dtype=np.int64)

    for row in head:
        snid = int(row["SNID"])
        pmin = int(row["PTROBS_MIN"])
        pmax = int(row["PTROBS_MAX"])

        if pointer_base == 1:
            start = pmin - 1
            stop = pmax
            marker_idx = pmax
        else:
            start = pmin
            stop = pmax + 1
            marker_idx = None

        if start < 0 or stop < 0 or start > stop:
            raise ValueError(f"Invalid pointers for SNID={snid}: PTROBS_MIN={pmin}, PTROBS_MAX={pmax}")
        if stop > len(phot):
            raise ValueError(
                f"Pointers exceed PHOT length for SNID={snid}: stop={stop}, len(PHOT)={len(phot)}"
            )

        snid_by_row[start:stop] = snid

        if validate and pointer_base == 1:
            if not (0 <= marker_idx < len(phot)) or float(phot[marker_idx]["MJD"]) != marker_mjd:
                raise ValueError(
                    f"Missing/incorrect marker row for SNID={snid}: expected PHOT[{marker_idx}].MJD={marker_mjd}"
                )

    if validate:
        mjd = np.asarray(phot["MJD"], dtype=float)
        marker_rows = mjd == marker_mjd
        if np.any(marker_rows & (snid_by_row != -1)):
            raise ValueError("Marker rows were assigned a SNID; pointer convention likely wrong.")

        head_df = fits_rec_to_dataframe(head)[["SNID", "NOBS"]].copy()
        obs_counts = (
            pd.Series(snid_by_row[~marker_rows])
            .value_counts(dropna=False)
            .rename_axis("SNID")
            .reset_index(name="NOBS_PHOT")
        )
        head_df["SNID"] = head_df["SNID"].astype(int)
        merged = head_df.merge(obs_counts, on="SNID", how="left").fillna({"NOBS_PHOT": 0})
        mismatch = merged[merged["NOBS"].astype(int) != merged["NOBS_PHOT"].astype(int)]
        if len(mismatch) > 0:
            ex = mismatch.head(5).to_dict(orient="records")
            raise ValueError(f"NOBS mismatch for {len(mismatch)} SNe (showing up to 5): {ex}")

    return snid_by_row


def build_combined_dataframe(
    head: fits.FITS_rec,
    phot: fits.FITS_rec,
    *,
    head_columns: Iterable[str] | None = None,
    phot_columns: Iterable[str] | None = None,
    pointer_base: PointerBase = "auto",
    marker_mjd: float = -777.0,
    validate: bool = True,
) -> pd.DataFrame:
    snid_by_row = annotate_phot_with_snid(
        head, phot, pointer_base=pointer_base, marker_mjd=marker_mjd, validate=validate
    )

    phot_df = fits_rec_to_dataframe(phot)
    phot_df["SNID"] = snid_by_row

    if phot_columns is not None:
        phot_columns = list(phot_columns)
        keep = ["SNID", *phot_columns]
        phot_df = phot_df[keep]

    phot_df = phot_df[(phot_df["SNID"] != -1) & (phot_df["MJD"].astype(float) != marker_mjd)].copy()

    head_df = fits_rec_to_dataframe(head)
    head_df["SNID"] = head_df["SNID"].astype(int)

    if head_columns is not None:
        head_columns = list(head_columns)
        head_df = head_df[["SNID", *head_columns]]

    combined = phot_df.merge(head_df, on="SNID", how="left", validate="many_to_one")
    return combined
