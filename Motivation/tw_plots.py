"""Plotting helpers for twilight cosmology analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .tw_constants import COLORS, FID_CENTERS


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
    N_cos_tw: np.ndarray,
    DZ: float,
    Z_TW_MIN: float,
    Z_TW_MAX: float,
    Z_MAX: float,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(z_mid, N_det, label=f"Detection (Y1, N={int(N_det.sum())})")
    plt.plot(z_mid, N_cos, label=f"Cosmology (Y1, N={int(N_cos.sum())})")
    plt.plot(z_mid, N_cos_tw, label=f"Cosmo + Twilight (Y1, N={int(N_cos_tw.sum())})")
    plt.axvspan(Z_TW_MIN, Z_TW_MAX, color="k", alpha=0.06, label="Twilight Promotion")
    plt.xlim(0.0, Z_MAX)
    plt.ylim(bottom=0)
    plt.xlabel("Redshift z")
    plt.ylabel("Count per Δz")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lcdm_1d(
    res: dict, *, param: str = "Om", title: str = "ΛCDM Forecast Comparison"
):
    """1D Gaussian PDFs for a single parameter from forecast results."""
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
    ax.set_xlabel(param)
    ax.set_ylabel("Gaussian (norm.)")
    ax.legend(frameon=False)
    if title:
        plt.title(title)
    plt.show()


def plot_corner(
    res: dict,
    *,
    params: tuple[str, ...] = ("Om", "w0", "wa"),
    title: str = "w0waCDM Forecast",
):
    labs = res["WFD"]["labels"]
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
                ax.set_ylabel("PDF")
                ax.set_xlabel(p_i)
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
                        color=COLORS[label]["light"],
                        lw=1.5,
                    )
                ax.set_xlabel(p_x)
                ax.set_ylabel(p_y)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=COLORS[label]["primary"], lw=2, label=f"{label} 1σ")
        for label in res
    ]
    axes[0, 0].legend(handles=handles, frameon=False, loc="upper right")
    fig.suptitle(title, y=0.93)
    fig.tight_layout()
    plt.show()


def plot_lcdm_1d_comparison(
    res_map: dict, *, param: str = "Om", title: str = "ΛCDM Forecast Comparison"
):
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
    ax.set_xlabel(param)
    ax.set_ylabel("Gaussian (norm.)")
    ax.legend(frameon=False)
    if title:
        plt.title(title)
    plt.show()


def plot_corner_comparison(
    res_map: dict,
    *,
    params: tuple[str, ...] = ("Om", "w0", "wa"),
    title: str = "w0waCDM Forecast Comparison (1σ/2σ)",
):
    labs_ref = res_map["WFD"]["labels"]
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
                ax.set_ylabel("PDF")
                ax.set_xlabel(p_i)
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
                        color=COLORS[label]["light"],
                        lw=1.5,
                    )
                ax.set_xlabel(p_x)
                ax.set_ylabel(p_y)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=COLORS[label]["primary"], lw=2, label=f"{label} 1σ")
        for label in res_map
    ]
    axes[0, 0].legend(handles=handles, frameon=False, loc="upper right")
    fig.suptitle(title, y=0.93)
    fig.tight_layout()
    plt.show()
