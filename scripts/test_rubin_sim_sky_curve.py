#!/usr/bin/env python3
"""
Probe rubin_sim SkyModel via our RubinSkyProvider and print a short
Sun-altitude → sky-mu table for a chosen band. Falls back to the simple model
if rubin_sim is not available, but clearly labels which path was used.

Usage:
  python scripts/test_rubin_sim_sky_curve.py [band] [--plot]

Examples:
  python scripts/test_rubin_sim_sky_curve.py g
  python scripts/test_rubin_sim_sky_curve.py r --plot
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str]) -> int:
    band = (argv[0] if argv and not argv[0].startswith("-") else "g").lower()
    do_plot = "--plot" in argv

    ROOT = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJ", str(ROOT))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from twilight_planner_pkg.sky_model import (
        RubinSkyProvider,
        SkyModelConfig,
        sky_mag_arcsec2,
    )

    alts = np.array([-12.0, -11.5, -11.0, -10.0, -9.0])
    X = 1.2
    used = "simple"
    try:
        sky = RubinSkyProvider()
        mu = np.array([
            sky.sky_mag(None, None, None, band=band, airmass=X, sun_alt_deg=float(a))
            for a in alts
        ])
        used = "rubin_sim"
    except Exception as e:
        print(f"rubin_sim unavailable, falling back to simple model — {type(e).__name__}: {e}")
        cfg = SkyModelConfig()
        mu = np.array([sky_mag_arcsec2(band, cfg, sun_alt_deg=float(a), airmass=X) for a in alts])

    print(f"band={band}, airmass={X}, model={used}")
    for a, m in zip(alts, mu):
        print(f"  SunAlt={a:5.1f} deg  mu={m:6.3f} mag/arcsec^2")

    if do_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5,3.2))
        plt.plot(alts, mu, marker='o')
        plt.gca().invert_yaxis()
        plt.xlabel("Sun altitude [deg]")
        plt.ylabel("Sky μ [mag/arcsec²]")
        plt.title(f"{band} — {used}")
        out = Path(os.environ.get("PROJ", ".")) / "figs" / f"test_rubin_sim_curve_{band}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=160)
        print(f"Wrote: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

