from __future__ import annotations

"""Shared constants for twilight cosmology analysis."""

# Colors used in plots (preserve exact hex codes used)
COLORS = {
    "WFD": {"primary": "#1f77b4", "light": "#aec7e8"},
    "WFD+Twilight": {"primary": "#2ca02c", "light": "#98df8a"},
    # Direct Twilight-only label for clarity in new analyses
    "Twilight": {"primary": "#2ca02c", "light": "#98df8a"},
    # comparison color is added by notebook; export a slot here:
    "WFD+Twilight (Guess)": {"primary": "#ff7f0e", "light": "#ffbb78"},
}

# Fiducial centers used when drawing 1σ/2σ ellipses / 1D PDFs
FID_CENTERS = {"Om": 0.3, "w0": -1.0, "wa": 0.0, "M": 0.0}

# Default binning / ranges (keep notebook defaults)
DZ_DEFAULT = 0.01
Z_MAX_DEFAULT = 1.20
Z_TW_MIN_DEFAULT, Z_TW_MAX_DEFAULT = 0.02, 0.22
