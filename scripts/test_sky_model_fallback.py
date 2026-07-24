#!/usr/bin/env python3
"""
Quick test runner to exercise the sky-model selection printouts in
plot/plot_fig5_alt_vs_mlim_and_skymag.py.

It forces the code path where rubin_sim is attempted first and then fails with
an IndexError, so the script prints:

  [sky] rubin_sim requested but failed — IndexError: list index out of range
  [sky] Using toy/simple sky model for band g.

Usage:
  python scripts/test_sky_model_fallback.py [--trace]

Use --trace to also print the full traceback that the plot script emits when
SKY_DEBUG_TRACE=1 is set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    want_trace = "--trace" in argv

    # Ensure repo root in PROJ so figures go to <repo>/figs
    ROOT = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJ", str(ROOT))
    # Ensure local package is importable
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Force the plotting code to try rubin_sim first
    os.environ["SKY_MODEL"] = "rubin"
    # Optionally show full traceback when the forced failure triggers
    if want_trace:
        os.environ["SKY_DEBUG_TRACE"] = "1"

    # Monkeypatch the rubin_sim vector helper to raise the specific error the
    # user reported. This simulates a rubin_sim call failure and exercises the
    # fallback + logging path without requiring rubin_sim to be installed.
    from twilight_planner_pkg import sky_model as sm

    def _boom(*args, **kwargs):  # noqa: D401
        raise IndexError("list index out of range")

    sm.rubin_sim_mu_sky_by_sun_alt = _boom  # type: ignore[attr-defined]

    # Run the figure generator (writes PNGs into <PROJ>/figs and prints model)
    from plot.plot_fig5_alt_vs_mlim_and_skymag import main as plot_fig5

    return int(plot_fig5())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
