
# Implementation Plan — LSST Twilight SNe

This plan tracks incremental, testable stages for the twilight planner and the saturation proof pipeline.

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

---

## Stage 8: Adaptive filter-per-visit heuristics (pre-work notes)
**Goal**: Keep twilight throughput high by only paying carousel overhead when it advances colour goals.

**Candidate refinements**:
- *Target-driven second band*: In `twilight_planner_pkg/scheduler.py:579-596`, gate the append of a second/third filter behind `PriorityTracker.color_deficit(...)` and `_has_only_blue_or_red(...)`. If the SN already has balanced colour coverage within `color_window_days`, leave `filters_pref` as a single band. This reuses the existing cadence bonus machinery, avoids new state, and simply trims the list before it flows into `choose_filters_with_cap`.
- *Overhead-aware acceptance test*: In `twilight_planner_pkg/astro_utils.py:666-706`, compute the marginal time cost of each additional filter (carousel change + guard). Reject filters whose extra cost exceeds (a) the remaining window slack or (b) a configurable ratio such as `exp_time / (filter_change_s + guard)` falling below a threshold. The ratio can default to 1.0 so existing behaviour is unchanged unless `filter_change_s` is large.
- *Colour backfill pass*: Repurpose the existing per-window backfill loop (`twilight_planner_pkg/scheduler.py:1460-2080`) to queue only the SNe still lacking colour pairs. First run keeps `filters_per_visit_cap=1`; the backfill call temporarily lifts the per-visit cap for a short list and respects the same cadence/swap policies. This isolates carousel-heavy work to a second phase without rewriting the main queue logic.

**Measurement**:
- Compare `nights_df` metrics (`n_planned`, `window_utilization`, `cap_utilization`) against current baseline.
- Add unit coverage around new gating paths (e.g., targeted tests in `tests/test_planner_features.py` exercising high `filter_change_s`).

**Status**: Planned ☐
