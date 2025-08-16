"""Simple entry point for the twilight planner.

Usage (example):
    python -m twilight_planner_pkg.main \
        --csv /mnt/data/ATLAS_2021_to25_cleaned.csv \
        --out /mnt/data/out \
        --start 2024-01-01 --end 2024-01-07 \
        --lat -30.2446 --lon -70.7494 --height 2663 \
        --strategy hybrid
"""

import argparse
from pathlib import Path

from .config import PlannerConfig
from .scheduler import plan_twilight_range_with_caps


def build_parser():
    """Construct the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with all planner options.
    """
    p = argparse.ArgumentParser(description="LSST Twilight Planner (modular)")
    p.add_argument("--csv", required=True, help="Path to input CSV with SN list")
    p.add_argument("--out", required=True, help="Output directory for CSV results")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD, UTC)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD, UTC)")
    # Site
    p.add_argument("--lat", type=float, required=True, help="Geodetic latitude (deg)")
    p.add_argument("--lon", type=float, required=True, help="Geodetic longitude (deg)")
    p.add_argument("--height", type=float, required=True, help="Elevation (m)")
    # Optional knobs
    p.add_argument("--min_alt", type=float, default=20.0)
    p.add_argument(
        "--twilight-sun-alt-min",
        type=float,
        default=-18.0,
        help="Minimum Sun altitude (deg) for twilight windows",
    )
    p.add_argument(
        "--twilight-sun-alt-max",
        type=float,
        default=0.0,
        help="Maximum Sun altitude (deg) for twilight windows",
    )
    p.add_argument(
        "--filters", default="g,r,i,z", help="Comma-separated filter list, e.g. g,r,i,z"
    )
    p.add_argument(
        "--exp",
        default="g:5,r:5,i:5,z:5",
        help="Exposure per filter seconds, e.g. g:5,r:5,i:5,z:5",
    )
    p.add_argument("--evening_cap", type=float, default=600.0)
    p.add_argument("--morning_cap", type=float, default=600.0)
    p.add_argument("--per_sn_cap", type=float, default=120.0)
    p.add_argument("--max_sn", type=int, default=10)
    p.add_argument("--twilight_step", type=int, default=2)
    p.add_argument(
        "--require_single_time",
        action="store_true",
        help="Require one time satisfying moon sep for all filters",
    )
    p.add_argument(
        "--strategy",
        choices=["hybrid", "lc"],
        default="hybrid",
        help="Priority strategy: 'hybrid' drops after quick color, 'lc' pursues full light curves",
    )
    p.add_argument(
        "--hybrid-detections",
        type=int,
        default=2,
        help="Detections across ≥2 filters before Hybrid goal is met",
    )
    p.add_argument(
        "--hybrid-exposure",
        type=float,
        default=300.0,
        help="Total exposure seconds before Hybrid goal is met",
    )
    p.add_argument(
        "--lc-detections",
        type=int,
        default=5,
        help="Detections required for LSST-only light-curve goal",
    )
    p.add_argument(
        "--lc-exposure",
        type=float,
        default=300.0,
        help="Total exposure seconds for LSST-only goal",
    )
    p.add_argument("--simlib-out", help="Path to write SNANA SIMLIB (optional)")
    p.add_argument("--simlib-survey", default="LSST_TWILIGHT")
    p.add_argument("--simlib-filters", default="ugrizY")
    p.add_argument("--simlib-pixsize", type=float, default=0.200)
    p.add_argument("--npe-pixel-saturate", type=int, default=120000)
    p.add_argument("--photflag-saturate", type=int, default=2048)
    p.add_argument("--psf-unit", default="NEA_PIXEL")
    p.add_argument("--gain", type=float, default=1.6)
    p.add_argument("--read-noise-e", type=float, default=9.0)
    p.add_argument("--allow-filter-changes-in-twilight", action="store_true")
    return p


def parse_exp_map(s: str):
    """Convert a comma-separated exposure mapping into a dictionary.

    Parameters
    ----------
    s : str
        Comma-separated list of ``filter:time`` pairs (e.g., ``"g:5,r:10"``).

    Returns
    -------
    dict[str, float]
        Mapping from filter name to exposure time in seconds.
    """
    m = {}
    if s.strip():
        for part in s.split(","):
            k, v = part.split(":")
            m[k.strip()] = float(v.strip())
    return m


def main():
    """Run the command-line planner.

    Parses arguments, builds a :class:`PlannerConfig`, and writes
    schedule files to the output directory.
    """
    args = build_parser().parse_args()
    exp_map = parse_exp_map(args.exp)
    cfg = PlannerConfig(
        lat_deg=args.lat,
        lon_deg=args.lon,
        height_m=args.height,
        min_alt_deg=args.min_alt,
        twilight_sun_alt_min_deg=args.twilight_sun_alt_min,
        twilight_sun_alt_max_deg=args.twilight_sun_alt_max,
        filters=[x.strip() for x in args.filters.split(",") if x.strip()],
        exposure_by_filter=exp_map,
        twilight_step_min=args.twilight_step,
        evening_cap_s=args.evening_cap,
        morning_cap_s=args.morning_cap,
        per_sn_cap_s=args.per_sn_cap,
        max_sn_per_night=args.max_sn,
        require_single_time_for_all_filters=args.require_single_time,
        priority_strategy=args.strategy,
        hybrid_detections=args.hybrid_detections,
        hybrid_exposure_s=args.hybrid_exposure,
        lc_detections=args.lc_detections,
        lc_exposure_s=args.lc_exposure,
        simlib_out=args.simlib_out,
        simlib_survey=args.simlib_survey,
        simlib_filters=args.simlib_filters,
        simlib_pixsize=args.simlib_pixsize,
        simlib_npe_pixel_saturate=args.npe_pixel_saturate,
        simlib_photflag_saturate=args.photflag_saturate,
        simlib_psf_unit=args.psf_unit,
        gain_e_per_adu=args.gain,
        read_noise_e=args.read_noise_e,
        allow_filter_changes_in_twilight=args.allow_filter_changes_in_twilight,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    plan_twilight_range_with_caps(
        csv_path=args.csv,
        outdir=args.out,
        start_date=args.start,
        end_date=args.end,
        cfg=cfg,
        verbose=True,
    )


if __name__ == "__main__":
    main()
