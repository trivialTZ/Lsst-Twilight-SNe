
# Implementation Plan — LSST Twilight SNe

This plan tracks incremental, testable stages for the twilight planner and the saturation proof pipeline.
---

You’re right—cadence should be enforced per **filter**, not per-SN. Here’s a clean way to bolt that in without fighting your existing flow.

## Stage 5: Add “Unique-First” strategy (maximize distinct SNe per night)
**Why**: Provide a comparison cohort for the paper by sampling as many unique targets as possible, contrasting with cadence-preserving default.

**Scope**:
- New strategy flag `strategy="unique_first"`.
- Greedy selection to maximize the number of distinct SNe seen in a night, subject to altitude, moon, filter policy, and window caps.
- If time remains after covering unique SNe, optionally fill with second visits for color (configurable).
- Nightly summary should report: `unique_targets_observed`, `repeat_fraction`, and compare against default strategy if both are run.

**Files/Interfaces**:
- `twilight_planner_pkg/scheduler.py`: add a second prioritizer `prioritize_unique_first(...)`.
- `twilight_planner_pkg/main.py`: CLI `--strategy [default|unique_first]`.

**Config**:
- `cfg.unique_first_fill_with_color = True`
- `cfg.unique_lookback_days = 7` (avoid re-picking the same SN within a week, if feasible)

**Success Criteria**:
- On the same night and input list, `unique_first` yields a higher count of distinct SNe than the default strategy.
- Report includes side-by-side counts when both are executed.

**Tests**:
- `pytest twilight_planner_pkg/tests/test_strategy_unique_first.py`

**Docs**:
- README: add “Strategies” table comparing `default` vs `unique_first`.

**Status**: Planned ☐

---

## Stage 6: WFD/DDF saturation demonstration for low-z SNe
**Why**: Show that standard WFD/DDF exposures saturate bright nearby SNe, while twilight short exposures recover them.

**Scope**:
- In `motivation/`, add a reproducible notebook/script that:
  1) Computes peak-pixel or full-well risk vs. redshift for SNe Ia using WFD/DDF exposure assumptions,
  2) Compares against twilight `exptime_s = min(15 s, calculated)`,
  3) Outputs a figure and table: fraction of saturated visits vs. redshift bin; highlight the **z threshold** below which WFD/DDF underperform.
- Accept both synthetic SN grids and ATLAS 2021–2025 discoveries (if available) for realism.

**Inputs/Assumptions**:
- Use the project’s existing photometry/saturation model.
- Encode WFD/DDF exposure times, filter patterns, and overheads per current baseline used elsewhere in the repo.

**Deliverables**:
- `motivation/saturation_proof.ipynb` (or `.py`), saved outputs under `motivation/out/`.
- Figure: `saturation_vs_z.png`
- Table: `saturation_summary.csv` with columns `[z_bin, strategy, n_total, n_saturated, frac_saturated]`.

**Success Criteria**:
- Clear visualization of the redshift regime where WFD/DDF saturate while twilight does not.
- Notebook runs headless and produces deterministic outputs with the repo’s pinned deps.

**Tests**:
- `pytest motivation/tests/test_saturation_summary.py` (basic integrity checks: files exist, columns correct, monotonic trends sensible).

**Docs**:
- README (Motivation section) links the figure/table and briefly states the z-cut finding.
- Nightly reports optionally show predicted saturation risk for scheduled targets (stretch goal).

**Status**: Planned ☐

---

## Stage 7: Reporting & comparatives
**Goal**: Enhance nightly summaries to expose science-useful metrics for the paper.
**Add to nightly report**:
- Twilight window start/end/duration (UTC and local offset shown as `UTC±HH:MM`).
- Per-window time caps, utilization %, and whether caps bound the plan.
- Exposure stats: distribution of `exptime_s`, count at 15 s cap vs. shorter.
- Cadence compliance metrics (Stage 4).
- Unique-first vs default side-by-side counts when both are run (Stage 5).
- Optional: predicted saturation risk counts by filter.

**Tests**:  
- `pytest twilight_planner_pkg/tests/test_nightly_report_fields.py`

**Status**: Planned ☐

---

## Tracking
- Keep stages small and mergeable.
- Each stage: implement → add/green tests → update README + examples → land.
