
# Implementation Plan — LSST Twilight SNe

This plan tracks incremental, testable stages for the twilight planner and the saturation proof pipeline.
---
## Stage 5: “Unique-First” nightly strategy (maximize distinct SNe; minimal code ripple)

**Goal**
Provide a paper comparison cohort by prioritizing as many **distinct** SNe as possible in a night, under the same altitude/moon/filter/cap constraints, with **no extra files** and only small edits centered in `priority.py`.

**Design Principles**

* **No new modules.** Only touch: `twilight_planner_pkg/priority.py` (core), plus tiny, safe wiring in `scheduler.py` and `config.py`.
* **First-class strategy token.** `"unique_first"` must be handled exactly like `"hybrid"` and `"lc"` (same call sites, same tracker object).
* **Tiny scoring rule.** If an SN has **never been observed** in this run (or is beyond a configurable lookback), score = **1.0**; otherwise **0.0**. No other heuristics.
* **Lookback default = 999 days.** Treats repeats as “off” for the duration of a typical run by default.
* **No scheduler surgery.** Do not add passes or re-planners. Optional “fill with color” remains a configurable placeholder (default True) but **inert** until we explicitly add a second pass in a later stage.

---

### Scope (what changes)

1. **`priority.py` (primary)**

   * Add a new strategy branch: `strategy == "unique_first"`.
   * Extend `_SNHistory` with:

     * `last_seen_mjd: Optional[float]` (set when any detection is recorded).
   * Extend `PriorityTracker` with:

     * `unique_lookback_days: float = 999.0`.
   * Update `record_detection(...)` to set `last_seen_mjd`.
   * Implement scoring for `"unique_first"`:

     * If `hist.detections == 0` → return `1.0`.
     * Else (repeat within run) → return `0.0`.
     * (If later we thread a `now_mjd`, allow: return `1.0` if `now_mjd - last_seen_mjd > unique_lookback_days`; otherwise `0.0`.)
   * Leave `"hybrid"` and `"lc"` behavior unchanged (only docstring/typing consistency as needed).

2. **`config.py` (tiny additions)**

   * Add:

     * `unique_first_fill_with_color: bool = True`  *(placeholder, currently no behavior change)*
     * `unique_lookback_days: int = 999` *(requested default)*

3. **`scheduler.py` (tiny wiring & summary only)**

   * When constructing `PriorityTracker`, pass through `cfg.unique_lookback_days`.
   * Keep planning flow identical; the changed priority scores naturally alter which SNe get picked.
   * Enrich per-window summary (CSV row) with:

     * `unique_targets_observed` = count of distinct SN IDs planned in the window.
     * `repeat_fraction` = `(n_planned - unique_targets_observed) / n_planned` (0 if no plans).

4. **CLI (if present already)**

   * If `main.py` already maps `--strategy`→`cfg.priority_strategy`, no changes needed.
   * Otherwise, add `--strategy [hybrid|lc|unique_first]` and set `cfg.priority_strategy` accordingly.

---

### Config (defaults)

```python
cfg.priority_strategy = "hybrid"              # unchanged default
cfg.unique_first_fill_with_color = True       # placeholder; inert for now
cfg.unique_lookback_days = 999                # your requested default
```

---

### Behavior Summary

* **unique\_first:**

  * First pass greedily chooses SNe with `detections == 0` (or older than lookback once we thread time), maximizing **distinct** targets within each twilight window and time cap.
  * No added scheduling passes; same cap, altitude, moon, filter, and overhead policies apply.

* **hybrid / lc:**

  * **Unchanged** in this stage. (If we later refine cadence or escalation, that’s a separate stage.)

---

### Success Criteria

* On the **same night** and **same input list**, running with `priority_strategy="unique_first"` yields a **greater** or **equal** number of distinct SNe than `hybrid` or `lc`.
* Nightly/Window summary includes `unique_targets_observed` and `repeat_fraction`.
* Existing `hybrid`/`lc` tests and behavior remain **unchanged**.

---

### Tests

Create `twilight_planner_pkg/tests/test_strategy_unique_first.py` with **small, deterministic** cases:

1. **Scoring unit tests (no scheduler):**

   * New SN → `score(..., "unique_first") == 1.0`.
   * After one `record_detection`, same SN → `score(..., "unique_first") == 0.0`.
   * Ensure `"hybrid"` and `"lc"` scores match previous expectations on the same histories.

2. **Integration-lite (scheduler harness / fake window):**

   * Inject a small set where 12 SNe are eligible but the cap limits to \~10 visits.
   * With `unique_first`, confirm `unique_targets_observed == n_planned`.
   * With `hybrid` or `lc`, confirm `unique_targets_observed <= n_planned` and typically **less than** the unique\_first run.

3. **Summary fields:**

   * Verify that the per-window summary row contains `unique_targets_observed` and `repeat_fraction` with correct values.

---

### Docs

* **README:** Add a **Strategies** comparison row:

  * **hybrid:** “Quick color/exposure, escalate Ia to LC.”
  * **lc:** “Always pursue LC depth.”
  * **unique\_first:** “Maximize distinct SNe per night; repeats suppressed (lookback default 999 d).”
* Briefly note `unique_first_fill_with_color` as a future enhancement switch.

---

### Backward Compatibility & Risk

* Default remains `hybrid`; no change unless users opt into `unique_first`.
* Changes are **additive** and **localized**.
* No data model migrations, no new files.

---

**Status:** Done ☑

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
