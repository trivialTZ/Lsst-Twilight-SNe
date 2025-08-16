"""Reporting utilities for nightly metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def summarize_night(
    plan_df: pd.DataFrame,
    twilight: float | dict | pd.DataFrame,
    outdir: str | None = None,
) -> Dict[str, Any]:
    """Compute nightly efficiency and science KPIs from a plan.

    Parameters
    ----------
    plan_df : pandas.DataFrame
        DataFrame describing the executed or planned observations. Required
        columns are ``exposure_s``, ``readout_s``, ``filter_changes_s``,
        ``cross_filter_change_s``, ``slew_s``, ``filter`` and ``Name``. Optional
        columns ``airmass`` and ``moon_sep`` enable additional metrics.
    twilight : float | dict | pandas.DataFrame
        Total twilight duration in seconds (float), per-window durations
        (mapping of label→seconds), or a DataFrame with columns
        ``twilight_window`` and ``window_duration_s``.
    outdir : str, optional
        Directory to which ``nightly_metrics.csv`` will be written. Defaults to
        current directory when ``None``.

    Returns
    -------
    dict
        Mapping of metric names to values. The same information is written to
        ``twilight_outputs/nightly_metrics.csv``.

    Notes
    -----
    Key performance indicators include

    ``science_efficiency``
        ``science_exptime_s / total_used_s``
    ``twilight_utilization``
        ``total_used_s / twilight_total``
    ``color_completeness_frac``
        Fraction of supernovae observed in at least two filters.
    """
    totals = {
        "science_exptime_s": float(
            plan_df.get("exposure_s", pd.Series(dtype=float)).sum()
        ),
        "readout_s": float(plan_df.get("readout_s", pd.Series(dtype=float)).sum()),
        "filter_change_s": float(
            plan_df.get("filter_changes_s", pd.Series(dtype=float)).sum()
            + plan_df.get("cross_filter_change_s", pd.Series(dtype=float)).sum()
        ),
        "slew_s": float(plan_df.get("slew_s", pd.Series(dtype=float)).sum()),
    }
    totals["overhead_s"] = (
        totals["readout_s"] + totals["filter_change_s"] + totals["slew_s"]
    )
    totals["total_used_s"] = totals["science_exptime_s"] + totals["overhead_s"]

    if isinstance(twilight, (int, float)):
        twilight_total = float(twilight)
        per_window: dict[str, float] = {}
    elif isinstance(twilight, dict):
        twilight_total = float(sum(twilight.values()))
        per_window = {k: float(v) for k, v in twilight.items()}
    else:
        twilight_total = float(twilight["window_duration_s"].sum())
        per_window = {
            w: float(
                twilight.loc[
                    twilight["twilight_window"] == w, "window_duration_s"
                ].sum()
            )
            for w in twilight["twilight_window"].unique()
        }

    kpi: Dict[str, Any] = {}
    kpi["science_efficiency"] = totals["science_exptime_s"] / max(
        1.0, totals["total_used_s"]
    )
    kpi["twilight_utilization"] = totals["total_used_s"] / max(1.0, twilight_total)
    if "airmass" in plan_df:
        kpi["median_airmass"] = float(plan_df["airmass"].median())
    if "moon_sep" in plan_df:
        kpi["median_moon_sep_deg"] = float(plan_df["moon_sep"].median())
    kpi["filter_swaps"] = int(
        (plan_df["filter"] != plan_df["filter"].shift()).sum() - 1
        if len(plan_df)
        else 0
    )
    kpi["filters_used"] = (
        sorted(plan_df["filter"].unique().tolist()) if len(plan_df) else []
    )
    per_sn = (
        plan_df.groupby("Name")["filter"].agg(lambda s: set(s))
        if len(plan_df)
        else pd.Series()
    )
    kpi["color_completeness_frac"] = (
        float(per_sn.apply(lambda s: len(s) >= 2).mean()) if len(per_sn) else 0.0
    )

    metrics = {**totals, **kpi}
    metrics["twilight_total_s"] = twilight_total
    if per_window:
        metrics["twilight_per_window_s"] = per_window
    out_path = Path(outdir or ".") / "nightly_metrics.csv"
    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    return metrics
