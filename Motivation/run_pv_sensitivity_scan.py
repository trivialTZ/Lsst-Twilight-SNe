"""Run PV robustness scans for the Year-1 WFD and WFD+Twilight samples."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

here = Path(__file__).resolve()
root = here.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from Motivation.tw_pv_sensitivity import (  # noqa: E402
    lowz_robustness_summary,
    load_v7_pv_samples,
    plot_sigv_gain,
    plot_sigv_scan,
    run_sigmau_scan,
    run_sigv_scan,
    summarize_scan,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=root / "Motivation" / "derived" / "v7_fsigma" / "pv_sensitivity",
    )
    parser.add_argument("--size-batch", type=int, default=10_000)
    parser.add_argument("--number-worker", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Build the Year-1 PV samples, run the scans, and save summaries."""

    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    flow_samples = load_v7_pv_samples()

    sigv_raw = run_sigv_scan(
        flow_samples,
        size_batch=args.size_batch,
        number_worker=args.number_worker,
    )
    sigv_summary = summarize_scan(
        sigv_raw,
        parameter_col="sigv_kms",
        baseline_value=300.0,
    )

    sigmau_raw = run_sigmau_scan(
        flow_samples,
        size_batch=args.size_batch,
        number_worker=args.number_worker,
    )
    sigmau_summary = summarize_scan(
        sigmau_raw,
        parameter_col="sigmau_fiducial",
        baseline_value=21.0,
    )

    sigv_raw.to_csv(args.outdir / "pv_sigv_scan_raw.csv", index=False)
    sigv_summary.to_csv(args.outdir / "pv_sigv_scan_summary.csv", index=False)
    sigmau_raw.to_csv(args.outdir / "pv_sigmau_scan_raw.csv", index=False)
    sigmau_summary.to_csv(args.outdir / "pv_sigmau_scan_summary.csv", index=False)

    plot_sigv_scan(sigv_summary, args.outdir / "pv_sigv_sigma_fs8.png")
    plot_sigv_gain(sigv_summary, args.outdir / "pv_sigv_precision_gain.png")

    summary_text = lowz_robustness_summary(sigv_summary, sigmau_summary)
    summary_path = args.outdir / "pv_sensitivity_summary.md"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    baseline_table = sigv_summary.loc[
        sigv_summary["sigv_kms"].eq(300.0),
        [
            "sigv_kms",
            "z_min",
            "z_max",
            "zeff_wfd",
            "zeff_wfd_twilight",
            "sigma_fs8_wfd",
            "sigma_fs8_wfd_twilight",
            "N_wfd",
            "N_wfd_twilight",
            "precision_gain",
            "improvement_pct",
        ],
    ].copy()
    baseline_table.to_csv(args.outdir / "pv_sigv_baseline_300.csv", index=False)

    print(f"Wrote {summary_path}")
    print(summary_text)
    print("\nBaseline (sigv=300 km/s):")
    print(baseline_table.to_string(index=False))


if __name__ == "__main__":
    main()
