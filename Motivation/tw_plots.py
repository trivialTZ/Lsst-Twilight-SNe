"""Plotting helpers for twilight cosmology analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .tw_constants import COLORS, FID_CENTERS


def _apply_aas_style() -> None:
    """AAS-journal-friendly Matplotlib defaults (serif/Roman, larger labels)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "axes.linewidth": 1.2,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "legend.fontsize": 14,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "savefig.dpi": 300,
        }
    )


def _param_label(p: str) -> str:
    return {"Om": r"$\Omega_m$", "w0": r"$w_0$", "wa": r"$w_a$", "M": r"$\mathcal{M}$"}.get(p, p)


def _ymax_padded(values: list[float], *, pad_frac: float = 0.12) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return None
    vmax = max(finite)
    if vmax <= 0:
        return None
    return vmax * (1.0 + pad_frac)


def ellipse_points_from_cov(
    C2: np.ndarray, n_sigma: float = 1.0, n_pts: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    delta = (
        2.30
        if np.isclose(n_sigma, 1.0)
        else (6.17 if np.isclose(n_sigma, 2.0) else n_sigma**2)
    )
    vals, vecs = np.linalg.eigh(C2)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    a = np.sqrt(delta * vals[0])
    b = np.sqrt(delta * vals[1])
    t = np.linspace(0, 2 * np.pi, n_pts)
    circ = np.stack([a * np.cos(t), b * np.sin(t)], axis=0)
    pts = vecs @ circ
    return pts[0], pts[1]


def show_fig_y1_nz(
    z_mid: np.ndarray,
    N_det: np.ndarray,
    N_cos: np.ndarray,
    *,
    N_cos_tw: np.ndarray | None = None,
    DZ: float,
    Z_TW_MIN: float,
    Z_TW_MAX: float,
    Z_MAX: float,
) -> None:
    _apply_aas_style()
    plt.figure(figsize=(10, 6))
    plt.plot(z_mid, N_det, label=f"Detection (Y1, N={int(N_det.sum())})")
    plt.plot(z_mid, N_cos, label=f"Cosmology (Y1, N={int(N_cos.sum())})")
    if N_cos_tw is not None:
        plt.plot(z_mid, N_cos_tw, label=f"Twilight (Y1, N={int(N_cos_tw.sum())})")
    plt.axvspan(Z_TW_MIN, Z_TW_MAX, color="k", alpha=0.06, label="Twilight Window")
    plt.xlim(0.0, Z_MAX)
    plt.ylim(bottom=0)
    plt.xlabel("Redshift z")
    plt.ylabel("Count per Δz")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _bin_mask_for_range(
    z_mid: np.ndarray,
    z_min: float,
    z_max: float,
    *,
    dz: float | None = None,
    include_partial: bool = True,
) -> np.ndarray:
    """Boolean mask selecting bins within a redshift range.

    If ``dz`` is provided, bins are treated as intervals of width ``dz``
    centered at ``z_mid``. Otherwise, selection falls back to midpoint
    inclusion: ``z_min <= z_mid < z_max``.

    Parameters
    ----------
    z_mid : array-like
        Midpoints of the redshift bins.
    z_min, z_max : float
        Redshift range to select.
    dz : float or None, optional
        Bin width. If provided, interval logic is used; otherwise midpoint logic.
    include_partial : bool, optional
        When using interval logic, include bins that partially overlap the range
        if True; if False, only bins fully contained in the range are selected.

    Returns
    -------
    numpy.ndarray (bool)
        Mask over bins.
    """
    z_mid = np.asarray(z_mid, dtype=float)
    if dz is None:
        return (z_mid >= z_min) & (z_mid < z_max)
    half = 0.5 * float(dz)
    lo = z_mid - half
    hi = z_mid + half
    if include_partial:
        return (hi > z_min) & (lo < z_max)
    return (lo >= z_min) & (hi <= z_max)


def summarize_y1_counts_in_range(
    z_mid: np.ndarray,
    N_det: np.ndarray,
    N_cos: np.ndarray,
    *,
    N_cos_tw: np.ndarray | None = None,
    z_min: float,
    z_max: float,
    dz: float | None = None,
    include_partial: bool = True,
) -> dict[str, int]:
    """Summarize detection/cosmology counts within a redshift range.

    Parameters
    ----------
    z_mid : array-like
        Midpoints of the redshift bins.
    N_det, N_cos : array-like
        Counts per bin for detections and cosmology-quality SNe.
    N_cos_tw : array-like or None, optional
        Counts per bin for cosmology with Twilight promotion. If omitted,
        this key is not included in the output.
    z_min, z_max : float
        Inclusive/exclusive selection bounds: ``[z_min, z_max)`` for midpoint
        logic, or interval overlap when ``dz`` is provided.
    dz : float or None, optional
        Bin width. If provided, use interval logic via bin edges inferred from
        ``z_mid`` and ``dz``. If omitted, use midpoint logic.
    include_partial : bool, optional
        When interval logic is used (``dz`` provided), include bins that
        partially overlap the range if True; otherwise require full containment.

    Returns
    -------
    dict
        Dictionary with integer sums for keys: ``"N_det"``, ``"N_cos"``, and
        optionally ``"N_cos_tw"`` if provided.

    Examples
    --------
    >>> summarize_y1_counts_in_range(z_mid, N_det, N_cos, N_cos_tw=N_cos_tw,
    ...                              z_min=0.02, z_max=0.14, dz=DZ)
    {'N_det': 385, 'N_cos': 316, 'N_cos_tw': 385}
    """
    m = _bin_mask_for_range(z_mid, z_min, z_max, dz=dz, include_partial=include_partial)
    out: dict[str, int] = {
        "N_det": int(np.rint(np.asarray(N_det)[m].sum())),
        "N_cos": int(np.rint(np.asarray(N_cos)[m].sum())),
    }
    if N_cos_tw is not None:
        out["N_cos_tw"] = int(np.rint(np.asarray(N_cos_tw)[m].sum()))
    return out


def plot_lcdm_1d(
    res: dict, *, param: str = "Om", title: str = "ΛCDM Forecast Comparison"
):
    """1D Gaussian PDFs for a single parameter from forecast results."""
    _apply_aas_style()
    plt.figure(dpi=150)
    ax = plt.gca()
    x0 = FID_CENTERS[param]
    sigs = {}
    for label in res:
        labs = res[label]["labels"]
        C = res[label]["cov"]
        i = labs.index(param)
        sigs[label] = np.sqrt(C[i, i])
    span = 5.0 * max(sigs.values())
    xs = np.linspace(x0 - 3 * span, x0 + 3 * span, 800)
    for label, s in sigs.items():
        ys = np.exp(-0.5 * ((xs - x0) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        ax.plot(xs, ys, label=f"{label} (σ={s:.4g})", color=COLORS[label]["primary"])
    ax.set_xlabel(_param_label(param))
    ax.set_ylabel("Gaussian (norm.)")
    ax.legend(frameon=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    return plt.gcf(), ax


def plot_corner(
    res: dict,
    *,
    params: tuple[str, ...] = ("Om", "w0", "wa"),
    title: str = "w0waCDM Forecast",
):
    _apply_aas_style()
    # Use the first scenario as the label reference (param ordering is shared).
    labs = next(iter(res.values()))["labels"]
    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.2 * n), dpi=150)
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            if r < c:
                ax.axis("off")
                continue
            p_i = params[c]
            if r == c:
                for label in res:
                    C = res[label]["cov"]
                    i = labs.index(p_i)
                    s = np.sqrt(C[i, i])
                    x0 = FID_CENTERS[p_i]
                    xs = np.linspace(x0 - 5 * s, x0 + 5 * s, 600)
                    ys = np.exp(-0.5 * ((xs - x0) / s) ** 2) / (s * np.sqrt(2 * np.pi))
                    ax.plot(xs, ys, color=COLORS[label]["primary"], lw=2)
                ax.set_ylabel("")
                ax.set_xlabel(_param_label(p_i))
            else:
                p_x, p_y = params[c], params[r]
                i, j = labs.index(p_x), labs.index(p_y)
                for label in res:
                    C = res[label]["cov"]
                    C2 = C[np.ix_([i, j], [i, j])]
                    ex, ey = ellipse_points_from_cov(C2, n_sigma=1.0)
                    ex2, ey2 = ellipse_points_from_cov(C2, n_sigma=2.0)
                    ax.plot(
                        FID_CENTERS[p_x] + ex,
                        FID_CENTERS[p_y] + ey,
                        color=COLORS[label]["primary"],
                        lw=2,
                    )
                    ax.plot(
                        FID_CENTERS[p_x] + ex2,
                        FID_CENTERS[p_y] + ey2,
                        color=COLORS[label]["primary"],
                        lw=1.5,
                        linestyle="--",
                        alpha=0.65,
                    )
                ax.set_xlabel(_param_label(p_x))
                ax.set_ylabel(_param_label(p_y))
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=COLORS[label]["primary"], lw=2, label=f"{label}")
        for label in res
    ]
    leg_ax = axes[0, n - 1] if n > 1 else axes[0, 0]
    if n > 1:
        leg_ax.axis("off")
    leg_ax.legend(handles=handles, frameon=False, loc="upper right")
    if title:
        fig.suptitle(title, y=0.98)
    fig.tight_layout()
    plt.show()
    return fig, axes


def plot_lcdm_1d_comparison(
    res_map: dict, *, param: str = "Om", title: str = "ΛCDM Forecast Comparison"
):
    _apply_aas_style()
    plt.figure(dpi=150)
    ax = plt.gca()
    x0 = FID_CENTERS[param]
    max_s = 0.0
    for res in res_map.values():
        labs = res["labels"]
        C = res["cov"]
        i = labs.index(param)
        max_s = max(max_s, np.sqrt(C[i, i]))
    span = 5.0 * max_s
    xs = np.linspace(x0 - 3 * span, x0 + 3 * span, 800)
    for label, res in res_map.items():
        labs = res["labels"]
        C = res["cov"]
        i = labs.index(param)
        s = np.sqrt(C[i, i])
        ys = np.exp(-0.5 * ((xs - x0) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        ax.plot(xs, ys, label=f"{label} (σ={s:.4g})", color=COLORS[label]["primary"])
    ax.set_xlabel(_param_label(param))
    ax.set_ylabel("Gaussian (norm.)")
    ax.legend(frameon=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_corner_comparison(
    res_map: dict,
    *,
    params: tuple[str, ...] = ("Om", "w0", "wa"),
    title: str = "w0waCDM Forecast Comparison (1σ/2σ)",
):
    _apply_aas_style()
    # Use the first scenario as the label reference (param ordering is shared).
    labs_ref = next(iter(res_map.values()))["labels"]
    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.2 * n), dpi=150)
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            if r < c:
                ax.axis("off")
                continue
            p_i = params[c]
            if r == c:
                for label, res in res_map.items():
                    C = res["cov"]
                    i = labs_ref.index(p_i)
                    s = np.sqrt(C[i, i])
                    x0 = FID_CENTERS[p_i]
                    xs = np.linspace(x0 - 5 * s, x0 + 5 * s, 600)
                    ys = np.exp(-0.5 * ((xs - x0) / s) ** 2) / (s * np.sqrt(2 * np.pi))
                    ax.plot(xs, ys, color=COLORS[label]["primary"], lw=2)
                ax.set_ylabel("")
                ax.set_xlabel(_param_label(p_i))
            else:
                p_x, p_y = params[c], params[r]
                i, j = labs_ref.index(p_x), labs_ref.index(p_y)
                for label, res in res_map.items():
                    C = res["cov"]
                    C2 = C[np.ix_([i, j], [i, j])]
                    ex, ey = ellipse_points_from_cov(C2, n_sigma=1.0)
                    ex2, ey2 = ellipse_points_from_cov(C2, n_sigma=2.0)
                    ax.plot(
                        FID_CENTERS[p_x] + ex,
                        FID_CENTERS[p_y] + ey,
                        color=COLORS[label]["primary"],
                        lw=2,
                    )
                    ax.plot(
                        FID_CENTERS[p_x] + ex2,
                        FID_CENTERS[p_y] + ey2,
                        color=COLORS[label]["primary"],
                        lw=1.5,
                        linestyle="--",
                        alpha=0.65,
                    )
                ax.set_xlabel(_param_label(p_x))
                ax.set_ylabel(_param_label(p_y))
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=COLORS[label]["primary"], lw=2, label=f"{label}")
        for label in res_map
    ]
    leg_ax = axes[0, n - 1] if n > 1 else axes[0, 0]
    if n > 1:
        leg_ax.axis("off")
    leg_ax.legend(handles=handles, frameon=False, loc="upper right")
    if title:
        fig.suptitle(title, y=0.98)
    fig.tight_layout()
    plt.show()


def plot_w0wa_metrics_bars(
    res: dict,
    *,
    title: str = "w0waCDM: WFD vs WFD+Twilight",
    scenario_order: tuple[str, str] = ("WFD", "WFD+Twilight"),
    scenario_colors: dict[str, str] | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (10.2, 6.4),
    show_legend: bool = True,
):
    """Bar-plot FoM and 1σ errors for WFD vs WFD+Twilight.

    Creates a 2×2 figure:
    - FoM and σ for (w0, wa)
    - FoM and σ for (Om, w0, wa)

    Notes
    -----
    The FoM is computed as ``1/sqrt(det(C_sub))`` for the selected
    parameter sub-covariance. For 2D (w0, wa) this matches the usual
    DETF FoM up to an overall constant (which cancels in ratios).
    """
    _apply_aas_style()
    from .tw_fisher import compute_w0wa_metrics

    m = compute_w0wa_metrics(res)
    sigmas = m["sigmas"]
    fom_2d = m["fom_2d"]
    fom_3d = m["fom_3d"]

    scenarios = [s for s in scenario_order if s in sigmas]
    if len(scenarios) < 2:
        raise ValueError(f"Expected at least 2 scenarios in {scenario_order}; got {list(sigmas)}")

    def _pname(p: str) -> str:
        return {"Om": r"\Omega_m", "w0": r"w_0", "wa": r"w_a"}.get(p, p)

    def _sigma_xtick(p: str) -> str:
        return rf"$\sigma({_pname(p)})$"

    def _fmt_fom(v: float) -> str:
        if not np.isfinite(v):
            return "nan"
        av = abs(v)
        if av >= 100:
            return f"{v:.0f}"
        if av >= 10:
            return f"{v:.1f}"
        return f"{v:.3g}"

    def _fmt_sig(v: float) -> str:
        if not np.isfinite(v):
            return "nan"
        av = abs(v)
        if av >= 0.1:
            return f"{v:.3f}"
        if av >= 0.01:
            return f"{v:.4f}"
        return f"{v:.3g}"

    def _bar_height(v: float) -> float:
        return float(v) if np.isfinite(v) else 0.0

    def _annotate(ax, bars, values, fmt_fn):
        for b, v in zip(bars, values, strict=False):
            h = b.get_height()
            ax.annotate(
                fmt_fn(float(v)),
                (b.get_x() + b.get_width() / 2.0, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    def _color_for(scenario: str) -> str | None:
        if scenario_colors and scenario in scenario_colors:
            return scenario_colors[scenario]
        if scenario in COLORS:
            return COLORS[scenario]["primary"]
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=150)

    # ---- FoM panels
    x = np.arange(len(scenarios))
    colors = [_color_for(s) for s in scenarios]

    vals_2d = [fom_2d[s] for s in scenarios]
    bars = axes[0, 0].bar(x, [_bar_height(v) for v in vals_2d], color=colors)
    _annotate(axes[0, 0], bars, vals_2d, _fmt_fom)
    axes[0, 0].set_xticks(x, scenarios, rotation=0)
    axes[0, 0].set_ylabel("FoM")
    axes[0, 0].set_title(r"FoM: $(w_0,w_a)$")
    ymax = _ymax_padded([float(v) for v in vals_2d])
    if ymax is None:
        axes[0, 0].set_ylim(bottom=0.0)
    else:
        axes[0, 0].set_ylim(bottom=0.0, top=ymax)

    vals_3d = [fom_3d[s] for s in scenarios]
    bars = axes[1, 0].bar(x, [_bar_height(v) for v in vals_3d], color=colors)
    _annotate(axes[1, 0], bars, vals_3d, _fmt_fom)
    axes[1, 0].set_xticks(x, scenarios, rotation=0)
    axes[1, 0].set_ylabel("FoM")
    axes[1, 0].set_title(r"FoM: $(\Omega_m,w_0,w_a)$")
    ymax = _ymax_padded([float(v) for v in vals_3d])
    if ymax is None:
        axes[1, 0].set_ylim(bottom=0.0)
    else:
        axes[1, 0].set_ylim(bottom=0.0, top=ymax)

    # ---- σ panels
    def _plot_sig_panel(ax, params: list[str]):
        group_x = np.arange(len(params))
        width = 0.36
        offsets = (-width / 2.0, width / 2.0) if len(scenarios) == 2 else np.linspace(
            -width, width, len(scenarios)
        )

        all_vals: list[float] = []
        for i, s in enumerate(scenarios):
            vals = [sigmas[s].get(p, float("nan")) for p in params]
            all_vals.extend([float(v) for v in vals])
            bars = ax.bar(
                group_x + offsets[i],
                [_bar_height(v) for v in vals],
                width=width,
                label=s,
                color=_color_for(s),
                edgecolor="k",
                linewidth=0.3,
            )
            _annotate(ax, bars, vals, _fmt_sig)

        ax.set_xticks(group_x, [_sigma_xtick(p) for p in params])
        ax.set_ylabel("1σ error")
        ymax = _ymax_padded(all_vals)
        if ymax is None:
            ax.set_ylim(bottom=0.0)
        else:
            ax.set_ylim(bottom=0.0, top=ymax)

    _plot_sig_panel(axes[0, 1], ["w0", "wa"])
    axes[0, 1].set_title(r"Errors: $(w_0,w_a)$")

    _plot_sig_panel(axes[1, 1], ["Om", "w0", "wa"])
    axes[1, 1].set_title(r"Errors: $(\Omega_m,w_0,w_a)$")

    # Shared legend
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if show_legend and handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    if title:
        fig.suptitle(title, y=0.99)
    right = 0.92 if show_legend else 1.0
    top = 0.96 if title else 1.0
    fig.tight_layout(rect=(0, 0, right, top))
    if show:
        plt.show()
    return fig, axes


def plot_w0wa_metrics_bars_multi(
    res_map: dict[str, dict],
    *,
    title: str = "w0waCDM: multi-scenario",
    scenario_order: tuple[str, ...] | None = None,
    scenario_colors: dict[str, str] | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (10.2, 6.4),
    show_legend: bool = True,
):
    """Bar-plot FoM and 1σ errors for multiple scenarios.

    Parameters
    ----------
    res_map : dict
        Mapping from scenario name to a forecast block with keys:
        - ``labels``: list[str]
        - ``cov``: ndarray

        This matches the per-scenario values inside ``run_binned_forecast``
        output (e.g., ``res['WFD']`` or ``res['WFD+Twilight']``).
    """
    _apply_aas_style()

    scenarios = list(scenario_order) if scenario_order is not None else list(res_map)
    scenarios = [s for s in scenarios if s in res_map]
    if len(scenarios) < 1:
        raise ValueError("No scenarios provided in res_map.")

    labels = list(res_map[scenarios[0]].get("labels", []))
    required = ("Om", "w0", "wa")
    if not all(p in labels for p in required):
        raise ValueError(f"required params {required} not found in labels: {labels}")

    for s in scenarios[1:]:
        labs_s = list(res_map[s].get("labels", []))
        if labs_s != labels:
            raise ValueError(
                "All scenarios must share identical parameter labels; "
                f"got {scenarios[0]}={labels} vs {s}={labs_s}"
            )

    def _subcov(C: np.ndarray, params: tuple[str, ...]) -> np.ndarray:
        idx = [labels.index(p) for p in params]
        return C[np.ix_(idx, idx)]

    def _det_safe(C: np.ndarray) -> float:
        try:
            det = float(np.linalg.det(C))
        except np.linalg.LinAlgError:
            eps = 1e-15 * float(np.trace(C)) / C.shape[0]
            det = float(np.linalg.det(C + eps * np.eye(C.shape[0])))
        if det < 0 and abs(det) < 1e-18:
            det = 0.0
        return det

    def _fom(C_sub: np.ndarray) -> float:
        det = _det_safe(C_sub)
        if (not np.isfinite(det)) or det <= 0:
            return float("nan")
        return float(1.0 / np.sqrt(det))

    def _color_for(s: str) -> str | None:
        if scenario_colors and s in scenario_colors:
            return scenario_colors[s]
        if s in COLORS:
            return COLORS[s]["primary"]
        return None

    def _fmt_fom(v: float) -> str:
        if not np.isfinite(v):
            return "nan"
        av = abs(v)
        if av >= 100:
            return f"{v:.0f}"
        if av >= 10:
            return f"{v:.1f}"
        return f"{v:.3g}"

    def _fmt_sig(v: float) -> str:
        if not np.isfinite(v):
            return "nan"
        av = abs(v)
        if av >= 0.1:
            return f"{v:.3f}"
        if av >= 0.01:
            return f"{v:.4f}"
        return f"{v:.3g}"

    def _bar_height(v: float) -> float:
        return float(v) if np.isfinite(v) else 0.0

    def _annotate(ax, bars, values, fmt_fn):
        for b, v in zip(bars, values, strict=False):
            h = b.get_height()
            ax.annotate(
                fmt_fn(float(v)),
                (b.get_x() + b.get_width() / 2.0, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    sigmas: dict[str, dict[str, float]] = {}
    fom_2d: dict[str, float] = {}
    fom_3d: dict[str, float] = {}
    for s in scenarios:
        C = np.asarray(res_map[s]["cov"], dtype=float)
        sigmas[s] = {p: float(np.sqrt(C[i, i])) for i, p in enumerate(labels)}
        fom_2d[s] = _fom(_subcov(C, ("w0", "wa")))
        fom_3d[s] = _fom(_subcov(C, ("Om", "w0", "wa")))

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=150)

    x = np.arange(len(scenarios))
    colors = [_color_for(s) for s in scenarios]

    vals_2d = [fom_2d[s] for s in scenarios]
    bars = axes[0, 0].bar(x, [_bar_height(v) for v in vals_2d], color=colors)
    _annotate(axes[0, 0], bars, vals_2d, _fmt_fom)
    axes[0, 0].set_xticks(x, scenarios, rotation=0)
    axes[0, 0].set_ylabel("FoM")
    axes[0, 0].set_title(r"FoM: $(w_0,w_a)$")
    ymax = _ymax_padded([float(v) for v in vals_2d])
    if ymax is None:
        axes[0, 0].set_ylim(bottom=0.0)
    else:
        axes[0, 0].set_ylim(bottom=0.0, top=ymax)

    vals_3d = [fom_3d[s] for s in scenarios]
    bars = axes[1, 0].bar(x, [_bar_height(v) for v in vals_3d], color=colors)
    _annotate(axes[1, 0], bars, vals_3d, _fmt_fom)
    axes[1, 0].set_xticks(x, scenarios, rotation=0)
    axes[1, 0].set_ylabel("FoM")
    axes[1, 0].set_title(r"FoM: $(\Omega_m,w_0,w_a)$")
    ymax = _ymax_padded([float(v) for v in vals_3d])
    if ymax is None:
        axes[1, 0].set_ylim(bottom=0.0)
    else:
        axes[1, 0].set_ylim(bottom=0.0, top=ymax)

    def _pname(p: str) -> str:
        return {"Om": r"\Omega_m", "w0": r"w_0", "wa": r"w_a"}.get(p, p)

    def _sigma_xtick(p: str) -> str:
        return rf"$\sigma({_pname(p)})$"

    def _plot_sig_panel(ax, params: list[str]):
        group_x = np.arange(len(params))
        width = min(0.22, 0.8 / max(1, len(scenarios)))
        offsets = (np.arange(len(scenarios)) - (len(scenarios) - 1) / 2.0) * width

        all_vals: list[float] = []
        for i, s in enumerate(scenarios):
            vals = [sigmas[s].get(p, float("nan")) for p in params]
            all_vals.extend([float(v) for v in vals])
            bars = ax.bar(
                group_x + offsets[i],
                [_bar_height(v) for v in vals],
                width=width,
                label=s,
                color=_color_for(s),
                edgecolor="k",
                linewidth=0.3,
            )
            _annotate(ax, bars, vals, _fmt_sig)

        ax.set_xticks(group_x, [_sigma_xtick(p) for p in params])
        ax.set_ylabel("1σ error")
        ymax = _ymax_padded(all_vals)
        if ymax is None:
            ax.set_ylim(bottom=0.0)
        else:
            ax.set_ylim(bottom=0.0, top=ymax)

    _plot_sig_panel(axes[0, 1], ["w0", "wa"])
    axes[0, 1].set_title(r"Errors: $(w_0,w_a)$")

    _plot_sig_panel(axes[1, 1], ["Om", "w0", "wa"])
    axes[1, 1].set_title(r"Errors: $(\Omega_m,w_0,w_a)$")

    handles, labels_leg = axes[0, 1].get_legend_handles_labels()
    if show_legend and handles:
        fig.legend(handles, labels_leg, loc="upper right", frameon=False)

    if title:
        fig.suptitle(title, y=0.99)
    right = 0.92 if show_legend else 1.0
    top = 0.96 if title else 1.0
    fig.tight_layout(rect=(0, 0, right, top))
    if show:
        plt.show()
    return fig, axes


def plot_fs8_scan(
    flow_df,
    *,
    power_spectrum_dict: dict | None = None,
    sigma_u: float | None = None,
    sig_floor_list: np.ndarray | None = None,
    zlim_list: np.ndarray | None = None,
    fisher_properties: dict | None = None,
    parameter_dict: dict | None = None,
    size_batch: int = 10_000,
    number_worker: int = 20,
    zmin: float = 0.02,
    quiet: bool = True,
    title: str = "fσ8 forecast vs error floor and zmax",
    sigma_ref: float | None = None,
    sigma_label: str | None = None,
    relative: bool = False,
    fs8_ref: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scan σ(fσ8) vs. added error floor and zmax using FLIP.

    Parameters
    ----------
    flow_df : pandas.DataFrame
        Table containing at least the FLIP-required columns:
        ``ra`` (rad), ``dec`` (rad), ``zobs``, ``rcom_zobs``,
        ``hubble_norm``, and ``dmu_error`` (mag). If ``dmu_error_base``
        is not present, it will be set equal to ``dmu_error``.
    power_spectrum_dict : dict, optional
        Precomputed dict for FLIP covariance, e.g.,
        ``{"vv": [[k, Pvv(k)]]}``. If None, a default will be computed
        with CLASS using Planck-like values and growth-rate normalization.
    sigma_u : float, optional
        Damping scale in ``h^{-1} Mpc`` for the velocity power spectrum
        Gaussian suppression factor ``Du(k, sigma_u)``. Ignored if
        ``power_spectrum_dict`` is supplied. Default: 21.0.
    sig_floor_list : array-like, optional
        Values of an additional per-SN error floor (mag) to be COMBINED IN
        QUADRATURE with ``dmu_error_base`` when scanning. Default is
        linspace(0.035, 0.10, 6).
    zlim_list : array-like, optional
        Upper redshift limits to include in the scan (lower bound is ``zmin``).
        Default is [0.04, 0.05, 0.06, 0.08, 0.10].
    fisher_properties : dict, optional
        Fisher options; defaults to ``{"inversion_method": "inverse"}``.
    parameter_dict : dict, optional
        Free parameters for FLIP Fisher; defaults to
        ``{"fs8": 1.0, "sigv": 300.0, "sigma_M": 0.0}``.
    size_batch : int, optional
        FLIP covariance batch size.
    number_worker : int, optional
        Number of workers for FLIP covariance.
    zmin : float, optional
        Lower redshift cut for the scan.
    title : str, optional
        Plot title.

    sigma_ref : float, optional
        Reference value for the rms distance-modulus scatter (e.g., DEBASS
        benchmark). Drawn as a vertical dashed line if provided.
    sigma_label : str, optional
        Custom x-axis label. Defaults to ``r"$\langle\sigma_\mu\rangle$ [mag]"``.
    relative : bool, optional
        If True, divide the y-values by ``fs8_ref`` so the plot shows the
        relative error ``σ(fσ8)/fσ8``. Default False.
    fs8_ref : float, optional
        Reference value used when ``relative`` is True. Default 1.0.

    Returns
    -------
    tuple of numpy.ndarray
        ``(sigma_fs8, sigma_mu_rms)`` with shape ``(len(zlim_list),
        len(sig_floor_list))``. The first entry is σ(fσ8); the second is the
        rms of the per-SN ``dmu_error`` values after the additive floor is
        applied.
    """
    import os as _os
    import logging as _logging
    import numpy as _np
    import matplotlib.pyplot as _plt

    # Prefer CPU to avoid TPU/CUDA backend noise in logs
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # If possible, set JAX platform explicitly before flip/jax import
    try:  # pragma: no cover
        from jax import config as _jax_config  # type: ignore
        _jax_config.update("jax_platform_name", "cpu")
    except Exception:
        pass

    if quiet:
        # Reduce INFO chatter from libraries
        _logging.getLogger().setLevel(_logging.WARNING)
        _logging.getLogger("jax._src.xla_bridge").setLevel(_logging.ERROR)
        _logging.getLogger("root").setLevel(_logging.WARNING)

    try:
        import flip  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "flip is required for plot_fs8_scan; install flip-peculiar-velocity"
        ) from e

    df = flow_df.copy()
    if "dmu_error_base" not in df.columns:
        df["dmu_error_base"] = df["dmu_error"]

    if sig_floor_list is None:
        sig_floor_list = _np.linspace(0.035, 0.10, 6)
    else:
        sig_floor_list = _np.asarray(sig_floor_list, dtype=float)

    if zlim_list is None:
        zlim_list = _np.array([0.04, 0.05, 0.06, 0.08, 0.10], dtype=float)
    else:
        zlim_list = _np.asarray(zlim_list, dtype=float)

    if fisher_properties is None:
        fisher_properties = {"inversion_method": "inverse"}
    if parameter_dict is None:
        parameter_dict = {"fs8": 1.0, "sigv": 300.0, "sigma_M": 0.0}

    # Power spectrum if not provided
    if power_spectrum_dict is None:
        # Default to CLASS growth-rate normalization with Planck-like params
        kh, _, _, ptt, _ = flip.power_spectra.compute_power_spectra(
            "class_engine",
            {
                "h": 0.6766,
                "sigma8": 0.8102,
                "n_s": 0.9665,
                "omega_b": 0.02242,
                "omega_cdm": 0.11933,
            },
            0,
            1e-5,
            0.2,
            1500,
            normalization_power_spectrum="growth_rate",
        )
        sigmau_fiducial = 21.0 if sigma_u is None else float(sigma_u)
        power_spectrum_dict = {
            "vv": [[kh, ptt * flip.utils.Du(kh, sigmau_fiducial) ** 2]]
        }

    fs8_grid = _np.full((len(zlim_list), len(sig_floor_list)), _np.nan, dtype=float)
    sig_rms_grid = _np.full_like(fs8_grid, _np.nan, dtype=float)

    for ii, zl in enumerate(zlim_list):
        m = (df["zobs"] >= float(zmin)) & (df["zobs"] <= float(zl))
        sub = df.loc[m].copy()
        if sub.empty:
            continue
        base_errors = sub["dmu_error_base"].to_numpy(dtype=float)
        for jj, sf in enumerate(sig_floor_list):
            # Combine the additional floor in quadrature with baseline errors
            sub["dmu_error"] = _np.sqrt(_np.square(base_errors) + float(sf) ** 2)
            sig_rms_grid[ii, jj] = float(
                _np.sqrt(_np.mean(_np.square(sub["dmu_error"].to_numpy(dtype=float))))
            )
            DF = flip.data_vector.FisherVelFromHDres(sub.to_dict(orient="list"))
            Cfit = DF.compute_covariance(
                "carreres23",
                power_spectrum_dict,
                size_batch=size_batch,
                number_worker=number_worker,
            )
            FF = flip.fisher.FisherMatrix.init_from_covariance(
                Cfit, DF, parameter_dict, fisher_properties=fisher_properties
            )
            names, Fm = FF.compute_fisher_matrix()
            try:
                idx = list(names).index("fs8")
            except ValueError:
                idx = 0
            Covm = _np.linalg.pinv(Fm)
            fs8_grid[ii, jj] = float(_np.sqrt(Covm[idx, idx]))

    # Plot
    _plt.figure(figsize=(7, 5), dpi=150)
    xlabel = sigma_label or r"$\langle\sigma_\mu\rangle$ [mag]"
    for ii, zl in enumerate(zlim_list):
        xvals = sig_rms_grid[ii]
        mask = _np.isfinite(xvals) & _np.isfinite(fs8_grid[ii])
        if not mask.any():
            continue
        yvals = fs8_grid[ii, mask]
        if relative:
            yvals = yvals / float(fs8_ref)
        _plt.plot(
            xvals[mask],
            yvals,
            marker="o",
            label=f"z≤{float(zl):.3f}",
        )
    if sigma_ref is not None:
        _plt.axvline(float(sigma_ref), color="k", linestyle="--", linewidth=1.0)
    _plt.xlabel(xlabel)
    _plt.ylabel("$\\sigma_{f\\sigma_8}$" if not relative else r"$\\sigma_{f\\sigma_8}/f\\sigma_8$")
    _plt.legend(frameon=False)
    if title:
        _plt.title(title)
    _plt.grid(alpha=0.2)
    _plt.tight_layout()
    _plt.show()

    return fs8_grid, sig_rms_grid


def compute_fs8_scan_grid(
    flow_df,
    *,
    power_spectrum_dict: dict | None = None,
    sigma_u: float | None = None,
    sig_floor_list: np.ndarray | None = None,
    zlim_list: np.ndarray | None = None,
    fisher_properties: dict | None = None,
    parameter_dict: dict | None = None,
    size_batch: int = 10_000,
    number_worker: int = 20,
    zmin: float = 0.02,
    quiet: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute σ(fσ8) grid and corresponding RMS(σμ) grid without plotting.

    Returns (fs8_grid, sig_rms_grid, sig_floor_list, zlim_list).
    API and science match plot_fs8_scan.
    """
    import os as _os
    import logging as _logging
    import numpy as _np
    try:
        import flip  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "flip is required for compute_fs8_scan_grid; install flip-peculiar-velocity"
        ) from e

    _os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    if quiet:
        _logging.getLogger().setLevel(_logging.WARNING)
        _logging.getLogger("jax._src.xla_bridge").setLevel(_logging.ERROR)
        _logging.getLogger("root").setLevel(_logging.WARNING)

    df = flow_df.copy()
    if "dmu_error_base" not in df.columns:
        df["dmu_error_base"] = df["dmu_error"]

    sig_floor_list = (
        _np.linspace(0.035, 0.10, 6)
        if sig_floor_list is None
        else _np.asarray(sig_floor_list, dtype=float)
    )
    zlim_list = (
        _np.array([0.04, 0.05, 0.06, 0.08, 0.10], dtype=float)
        if zlim_list is None
        else _np.asarray(zlim_list, dtype=float)
    )

    fisher_properties = fisher_properties or {"inversion_method": "inverse"}
    parameter_dict = parameter_dict or {"fs8": 1.0, "sigv": 300.0, "sigma_M": 0.0}

    if power_spectrum_dict is None:
        kh, _, _, ptt, _ = flip.power_spectra.compute_power_spectra(
            "class_engine",
            {
                "h": 0.6766,
                "sigma8": 0.8102,
                "n_s": 0.9665,
                "omega_b": 0.02242,
                "omega_cdm": 0.11933,
            },
            0,
            1e-5,
            0.2,
            1500,
            normalization_power_spectrum="growth_rate",
        )
        sigmau_fiducial = 21.0 if sigma_u is None else float(sigma_u)
        power_spectrum_dict = {
            "vv": [[kh, ptt * flip.utils.Du(kh, sigmau_fiducial) ** 2]]
        }

    fs8_grid = _np.full((len(zlim_list), len(sig_floor_list)), _np.nan, dtype=float)
    sig_rms_grid = _np.full_like(fs8_grid, _np.nan, dtype=float)

    for ii, zl in enumerate(zlim_list):
        m = (df["zobs"] >= float(zmin)) & (df["zobs"] <= float(zl))
        sub = df.loc[m].copy()
        if sub.empty:
            continue
        base_errors = sub["dmu_error_base"].to_numpy(dtype=float)
        for jj, sf in enumerate(sig_floor_list):
            # Combine the floor in quadrature with the baseline per-SN error
            # (baseline typically already includes SALT2 stat + sigma_int).
            sub["dmu_error"] = _np.sqrt(_np.square(base_errors) + float(sf) ** 2)
            sig_rms_grid[ii, jj] = float(
                _np.sqrt(_np.mean(_np.square(sub["dmu_error"].to_numpy(dtype=float))))
            )
            DF = flip.data_vector.FisherVelFromHDres(sub.to_dict(orient="list"))
            Cfit = DF.compute_covariance(
                "carreres23",
                power_spectrum_dict,
                size_batch=size_batch,
                number_worker=number_worker,
            )
            FF = flip.fisher.FisherMatrix.init_from_covariance(
                Cfit, DF, parameter_dict, fisher_properties=fisher_properties
            )
            names, Fm = FF.compute_fisher_matrix()
            try:
                idx = list(names).index("fs8")
            except ValueError:
                idx = 0
            Covm = _np.linalg.pinv(Fm)
            fs8_grid[ii, jj] = float(_np.sqrt(Covm[idx, idx]))

    return fs8_grid, sig_rms_grid, sig_floor_list, zlim_list


def plot_fs8_scan_compare(
    flow_wfd,
    flow_combined,
    *,
    power_spectrum_dict: dict | None = None,
    sigma_u: float | None = None,
    sig_floor_list: np.ndarray | None = None,
    zlim_list: np.ndarray | None = None,
    fisher_properties: dict | None = None,
    parameter_dict: dict | None = None,
    size_batch: int = 10_000,
    number_worker: int = 20,
    zmin: float = 0.02,
    quiet: bool = True,
    sigma_ref: float | None = None,
    title: str = "fσ8 vs ⟨σμ⟩: WFD Y1 (dashed) vs WFD+Twilight Y1 (solid)",
    relative: bool = False,
    fs8_ref: float = 1.0,
):
    """Overlay fσ8 scans: dashed WFD vs solid WFD+Twilight with matched colors per z-limit.

    Uses the same science/definitions as plot_fs8_scan (linear error floor → per-SN
    dμ rms on the x-axis; Fisher σ(fσ8) on the y-axis).
    """
    import matplotlib.pyplot as _plt
    import numpy as _np

    fs8_wfd, rms_wfd, sf_list, zl_list = compute_fs8_scan_grid(
        flow_wfd,
        power_spectrum_dict=power_spectrum_dict,
        sigma_u=sigma_u,
        sig_floor_list=sig_floor_list,
        zlim_list=zlim_list,
        fisher_properties=fisher_properties,
        parameter_dict=parameter_dict,
        size_batch=size_batch,
        number_worker=number_worker,
        zmin=zmin,
        quiet=quiet,
    )
    fs8_tw, rms_tw, _, _ = compute_fs8_scan_grid(
        flow_combined,
        power_spectrum_dict=power_spectrum_dict,
        sigma_u=sigma_u,
        sig_floor_list=sf_list,
        zlim_list=zl_list,
        fisher_properties=fisher_properties,
        parameter_dict=parameter_dict,
        size_batch=size_batch,
        number_worker=number_worker,
        zmin=zmin,
        quiet=quiet,
    )

    _plt.figure(figsize=(8, 5.5), dpi=150)
    color_cycle = _plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0","C1","C2","C3","C4"])
    for i, zl in enumerate(zl_list):
        color = color_cycle[i % len(color_cycle)]
        # WFD dashed
        mask = _np.isfinite(rms_wfd[i]) & _np.isfinite(fs8_wfd[i])
        if mask.any():
            yvals = fs8_wfd[i][mask]
            if relative:
                yvals = yvals / float(fs8_ref)
            _plt.plot(rms_wfd[i][mask], yvals, linestyle="--", color=color, linewidth=2.0)
        # Combined solid
        mask2 = _np.isfinite(rms_tw[i]) & _np.isfinite(fs8_tw[i])
        if mask2.any():
            yvals2 = fs8_tw[i][mask2]
            if relative:
                yvals2 = yvals2 / float(fs8_ref)
            _plt.plot(rms_tw[i][mask2], yvals2, linestyle="-", color=color, linewidth=2.0, label=f"z≤{float(zl):.3f}")

    if sigma_ref is not None:
        _plt.axvline(float(sigma_ref), color="k", linestyle="--", linewidth=1.0)

    # Build a second legend for dataset styles
    from matplotlib.lines import Line2D as _Line2D
    style_handles = [
        _Line2D([0], [0], color="k", linestyle="--", lw=2.0, label="WFD Y1"),
        _Line2D([0], [0], color="k", linestyle="-", lw=2.0, label="WFD+Twilight Y1"),
    ]

    _plt.xlabel(r"$\langle\sigma_\mu\rangle$ [mag]")
    _plt.ylabel(r"$\sigma_{f\sigma_8}$" if not relative else r"$\sigma_{f\sigma_8}/f\sigma_8$")
    _plt.title(title)
    leg1 = _plt.legend(frameon=False, title="z max", loc="upper left")
    _plt.gca().add_artist(leg1)
    _plt.legend(handles=style_handles, frameon=False, loc="lower right", title="dataset")
    _plt.grid(alpha=0.2)
    _plt.tight_layout()
    _plt.show()

    return (fs8_wfd, rms_wfd), (fs8_tw, rms_tw), sf_list, zl_list
