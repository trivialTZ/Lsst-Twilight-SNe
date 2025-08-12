"""Reporting utilities for nightly metrics."""
from __future__ import annotations

from typing import Dict, Any
import pandas as pd


def summarize_night(plan_df: pd.DataFrame, twilight_window_s: float) -> Dict[str, Any]:
    """Compute nightly efficiency and science KPIs from a plan.

    Parameters
    ----------
    plan_df : pandas.DataFrame
        DataFrame describing the executed or planned observations. Required
        columns are ``exposure_s``, ``readout_s``, ``filter_changes_s``,
        ``cross_filter_change_s``, ``slew_s``, ``filter`` and ``Name``. Optional
        columns ``airmass`` and ``moon_sep`` enable additional metrics.
    twilight_window_s : float
        Duration of the available twilight window in seconds.

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
        ``total_used_s / twilight_window_s``
    ``color_completeness_frac``
        Fraction of supernovae observed in at least two filters.
    """
    totals = {
        "science_exptime_s": float(plan_df.get("exposure_s", pd.Series(dtype=float)).sum()),
        "readout_s": float(plan_df.get("readout_s", pd.Series(dtype=float)).sum()),
        "filter_change_s": float(
            plan_df.get("filter_changes_s", pd.Series(dtype=float)).sum()
            + plan_df.get("cross_filter_change_s", pd.Series(dtype=float)).sum()
        ),
        "slew_s": float(plan_df.get("slew_s", pd.Series(dtype=float)).sum()),
    }
    totals["overhead_s"] = totals["readout_s"] + totals["filter_change_s"] + totals["slew_s"]
    totals["total_used_s"] = totals["science_exptime_s"] + totals["overhead_s"]

    kpi: Dict[str, Any] = {}
    kpi["science_efficiency"] = totals["science_exptime_s"] / max(1.0, totals["total_used_s"])
    kpi["twilight_utilization"] = totals["total_used_s"] / max(1.0, twilight_window_s)
    if "airmass" in plan_df:
        kpi["median_airmass"] = float(plan_df["airmass"].median())
    if "moon_sep" in plan_df:
        kpi["median_moon_sep_deg"] = float(plan_df["moon_sep"].median())
    kpi["filter_swaps"] = int((plan_df["filter"] != plan_df["filter"].shift()).sum() - 1 if len(plan_df) else 0)
    kpi["filters_used"] = sorted(plan_df["filter"].unique().tolist()) if len(plan_df) else []
    per_sn = plan_df.groupby("Name")["filter"].agg(lambda s: set(s)) if len(plan_df) else pd.Series()
    kpi["color_completeness_frac"] = (
        float(per_sn.apply(lambda s: len(s) >= 2).mean()) if len(per_sn) else 0.0
    )

    metrics = {**totals, **kpi}
    pd.DataFrame([metrics]).to_csv("twilight_outputs/nightly_metrics.csv", index=False)
    return metrics
