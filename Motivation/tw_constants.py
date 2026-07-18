from __future__ import annotations

"""Shared constants for twilight cosmology analysis."""

# Colors used in plots (preserve exact hex codes used)
COLORS = {
    # Paper palette (user request): WFD=navy, Twilight=orange
    "WFD": {"primary": "#192D54", "light": "#AFB6C3"},
    "WFD+Twilight": {"primary": "#C65F10", "light": "#EBC7AB"},
    # Direct Twilight-only label for clarity in new analyses
    "Twilight": {"primary": "#C65F10", "light": "#EBC7AB"},
    # Add-on scenarios (e.g., adding DDF on top of baseline samples)
    "Twilight+DDF": {"primary": "#6A3D9A", "light": "#D1C4E9"},
    "WFD+Twilight+DDF": {"primary": "#6A3D9A", "light": "#D1C4E9"},
    # Backwards-compatible alias used in some notebooks
    "WFD+Twilight (Guess)": {"primary": "#C65F10", "light": "#EBC7AB"},
}

# Fiducial centers used when drawing 1σ/2σ ellipses / 1D PDFs
FID_CENTERS = {"Om": 0.3, "w0": -1.0, "wa": 0.0, "M": 0.0}

# Default binning / ranges (keep notebook defaults)
DZ_DEFAULT = 0.00
Z_MAX_DEFAULT = 2.2
Z_TW_MIN_DEFAULT, Z_TW_MAX_DEFAULT = 0.00, 0.25
