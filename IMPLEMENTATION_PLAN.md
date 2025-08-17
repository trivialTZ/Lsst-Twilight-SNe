
# Implementation Plan — LSST Twilight SNe

This plan tracks incremental, testable stages for the twilight planner and the saturation proof pipeline.
---

You’re right—cadence should be enforced per **filter**, not per-SN. Here’s a clean way to bolt that in without fighting your existing flow.

# Revised Stage 4 (per-filter cadence)
**Status**: Complete

* Enforce spacing only for the **same filter**: require `days_since_last(filter) ≥ cadence_days_target - cadence_jitter_days`.
* Allow different filters at the same epoch (so same-night g,r,i is fine).
* Add a soft “due-soon” bonus by filter to gently favor targets whose last visit in that filter is \~3 days ago (without hard-blocking others).
* Report cadence compliance per filter in the nightly summary.

# What to change (minimal, surgical)

## 1) Extend the priority tracker with per-filter timestamps

Add last-visit bookkeeping and helpers. (We reuse the existing tracker so we don’t introduce a second state holder.)&#x20;

**Patch sketch (priority.py):**

```diff
 dataclass
 class _SNHistory:
@@
     filters: Set[str] = field(default_factory=set)
     escalated: bool = False
+    # NEW: MJD of the most recent observation by filter
+    last_mjd_by_filter: Dict[str, float] = field(default_factory=dict)
 
@@
-    def record_detection(self, name: str, exposure_s: float, filters: List[str]) -> None:
+    def record_detection(self, name: str, exposure_s: float, filters: List[str], mjd: float | None = None) -> None:
@@
         hist.detections += len(filters)
         hist.exposure_s += exposure_s
         hist.filters.update(filters)
+        if mjd is not None:
+            for f in filters:
+                hist.last_mjd_by_filter[f] = float(mjd)
 
@@
     # alias for clarity
     update = record_detection
+
+    # ---- NEW: cadence helpers ----
+    def days_since(self, name: str, filt: str, now_mjd: float) -> Optional[float]:
+        """Days since last obs in this filter for SN `name` (None if never)."""
+        hist = self.history.get(name)
+        if not hist:
+            return None
+        last = hist.last_mjd_by_filter.get(filt)
+        return None if last is None else float(now_mjd - last)
+
+    def cadence_gate(self, name: str, filt: str, now_mjd: float,
+                     target_d: float, jitter_d: float) -> bool:
+        """True if allowed to observe `filt` now under per-filter cadence rule."""
+        ds = self.days_since(name, filt, now_mjd)
+        # If never observed in this filter, allow (so first-epoch colors work).
+        if ds is None:
+            return True
+        return ds >= (target_d - jitter_d)
+
+    def cadence_bonus(self, name: str, filt: str, now_mjd: float,
+                      target_d: float, sigma_d: float, weight: float = 0.25) -> float:
+        """Soft bonus (0..weight) peaking when ds≈target; Gaussian around target."""
+        ds = self.days_since(name, filt, now_mjd)
+        if ds is None:
+            return 0.0
+        # exp(-((|ds - target|)^2)/(2*sigma^2))
+        import math
+        return float(weight * math.exp(-((abs(ds - target_d) ** 2) / (2.0 * sigma_d**2))))
```

## 2) Gate “allowed filters” by per-filter cadence in the scheduler

* Compute the MJD we’re actually scheduling at (use the serialized **true** start time for the visit).
* Filter the candidate’s `allowed` list with `tracker.cadence_gate(...)` when cadence is enabled.
* Use `cadence_bonus` to order the **rest** of the filters (after the strategy-chosen first) so we naturally take the due filter sooner.
* Record MJDs into the tracker so subsequent decisions see updated recency.
* Emit per-filter Δt metrics per exposure for the nightly summary.



**Patch sketch (scheduler.py):**

```diff
@@
     tracker = PriorityTracker(
         hybrid_detections=cfg.hybrid_detections,
         hybrid_exposure_s=cfg.hybrid_exposure_s,
         lc_detections=cfg.lc_detections,
         lc_exposure_s=cfg.lc_exposure_s,
     )
+    # ---- NEW cadence config (with sane fallbacks) ----
+    cadence_on = bool(getattr(cfg, "cadence_enable", True))
+    cadence_per_filter = bool(getattr(cfg, "cadence_per_filter", True))
+    cad_target = float(getattr(cfg, "cadence_days_target", 3.0))
+    cad_jitter = float(getattr(cfg, "cadence_jitter_days", 0.25))
+    cad_tol = float(getattr(cfg, "cadence_days_tolerance", 0.5))  # for summary
+    cad_sigma = float(getattr(cfg, "cadence_bonus_sigma_days", 0.5))
+    cad_weight = float(getattr(cfg, "cadence_bonus_weight", 0.25))
@@
         for idx_w in sorted(set(top_global["best_window_index"].values)):
@@
-            # ---- NEW: per-window running clock (UTC) for true order ----
+            # ---- per-window running clock (UTC) for true, serialized order ----
             current_time_utc = pd.Timestamp(win["start"]).tz_convert("UTC")
             order_in_window = 0
@@
-                allowed = allowed_filters_for_window(
+                allowed = allowed_filters_for_window(
                     mag_lookup.get(row["Name"], {}),
                     sun_alt,
                     row["_moon_alt"],
                     row["_moon_phase"],
                     row["_moon_sep"],
                     airmass_from_alt_deg(row["max_alt_deg"]),
                     cfg.fwhm_eff or 0.7,
                 )
@@
                 moon_sep_ok = {
                     f: row["_moon_sep"]
                     >= effective_min_sep(
                         f,
                         row["_moon_alt"],
                         row["_moon_phase"],
                         cfg.min_moon_sep_by_filter,
                     )
                     for f in allowed
                 }
+                # ---- NEW: per-filter cadence gate ----
+                if cadence_on and cadence_per_filter:
+                    # Use the *window clock* moment as "now" to evaluate cadence
+                    now_mjd_for_gate = Time(current_time_utc).mjd
+                    allowed = [
+                        f for f in allowed
+                        if moon_sep_ok.get(f, False)
+                        and tracker.cadence_gate(row["Name"], f, now_mjd_for_gate, cad_target, cad_jitter)
+                    ]
+                else:
+                    allowed = [f for f in allowed if moon_sep_ok.get(f, False)]
 
                 first = pick_first_filter_for_target(
                     row["Name"],
                     row.get("SN_type_raw"),
                     tracker,
                     allowed,
                     cfg,
                     sun_alt_deg=sun_alt,
                     moon_sep_ok=moon_sep_ok,
                     current_mag=mag_lookup.get(row["Name"]),
                     current_filter=current_filter_by_window.get(idx_w),
                 )
                 if first is None:
                     continue
+                # ---- NEW: cadence-aware ordering of remaining filters ----
+                if cadence_on and cadence_per_filter:
+                    now_mjd_for_bonus = Time(current_time_utc).mjd
+                    def _urgency(f: str) -> float:
+                        return tracker.cadence_bonus(row["Name"], f, now_mjd_for_bonus, cad_target, cad_sigma, weight=cad_weight)
+                    rest = sorted([f for f in allowed if f != first], key=_urgency, reverse=True)
+                else:
+                    rest = [f for f in allowed if f != first]
 
                 cand = {
                     "Name": row["Name"],
@@
-                    "first_filter": first,
+                    "first_filter": first,
                     "sn_type": row.get("SN_type_raw"),
-                    "allowed": allowed,
+                    "allowed": [first] + rest,  # carry cadence-aware order forward
                     "moon_sep_ok": moon_sep_ok,
                     "moon_sep": float(row["_moon_sep"]),
                 }
@@
-                    sn_start_utc = current_time_utc
+                    sn_start_utc = current_time_utc
                     sn_end_utc = sn_start_utc + pd.to_timedelta(
                         timing["total_s"], unit="s"
                     )
+                    visit_mjd = Time(sn_start_utc.to_pydatetime()).mjd
@@
-                    filters_pref = [first] + [x for x in t["allowed"] if x != first]
+                    # `t["allowed"]` already cadence-ordered; ensure first is first
+                    filters_pref = [t["allowed"][0]] + [x for x in t["allowed"] if x != t["allowed"][0]]
@@
                     epochs = []
                     for f in filters_used:
                         exp_s = timing.get("exp_times", {}).get(
                             f, cfg.exposure_by_filter.get(f, 0.0)
                         )
@@
-                        mjd = (
-                            Time(t["best_time_utc"]).mjd
-                            if isinstance(t["best_time_utc"], (datetime, pd.Timestamp))
-                            else np.nan
-                        )
+                        # Use true serialized schedule time for MJD
+                        mjd = visit_mjd
@@
-                        if writer:
+                        if writer:
                             epochs.append(
                                 {
                                     "mjd": mjd,
                                     "band": f,
@@
-                        start_utc = pd.Timestamp(t["best_time_utc"])
-                        if start_utc.tzinfo is None:
-                            start_utc = start_utc.tz_localize("UTC")
-                        else:
-                            start_utc = start_utc.tz_convert("UTC")
-                        total_s = round(timing["total_s"], 2)
-                        end_utc = start_utc + pd.to_timedelta(total_s, unit="s")
+                        # Keep existing 'best_twilight_time_utc' for traceability,
+                        # but also include the true serialized visit start.
+                        start_utc = pd.Timestamp(t["best_time_utc"]).tz_localize("UTC") if pd.Timestamp(t["best_time_utc"]).tzinfo is None else pd.Timestamp(t["best_time_utc"]).tz_convert("UTC")
+                        total_s = round(timing["total_s"], 2)
+                        end_utc = pd.Timestamp(sn_end_utc)
+                        visit_start_iso = pd.Timestamp(sn_start_utc).isoformat()
@@
+                        # Cadence metrics: days since this filter BEFORE updating tracker
+                        dsf = tracker.days_since(t["Name"], f, visit_mjd) if (cadence_on and cadence_per_filter) else None
+                        gate_pass = (dsf is None) or (dsf >= (cad_target - cad_jitter)) if (cadence_on and cadence_per_filter) else True
+
                         row = {
@@
                             "best_twilight_time_utc": start_utc.isoformat(),
+                            "visit_start_utc": visit_start_iso,
                             "sn_end_utc": end_utc.isoformat(),
                             "filter": f,
@@
+                            "cadence_days_since": (round(float(dsf), 3) if dsf is not None else None),
+                            "cadence_target_d": cad_target if cadence_on else None,
+                            "cadence_gate_passed": bool(gate_pass),
@@
                     tracker.record_detection(
-                        t["Name"], timing["exposure_s"], filters_used
+                        t["Name"], timing["exposure_s"], filters_used, mjd=visit_mjd
                     )
@@
-            nights_rows.append(
+            # ---- NEW: per-filter cadence compliance summary for this window ----
+            # Compute per-filter median |Δt - target| and % within tolerance.
+            guard_rows = [
+                r for r in pernight_rows
+                if r["date"] == day.date().isoformat() and r["twilight_window"] == window_label_out
+            ]
+            cad_stats = {}
+            if cadence_on and cadence_per_filter:
+                byf: Dict[str, List[float]] = {}
+                within: Dict[str, List[bool]] = {}
+                for r in guard_rows:
+                    ds = r.get("cadence_days_since")
+                    f = r.get("filter")
+                    if ds is None or f is None:
+                        continue
+                    byf.setdefault(f, []).append(abs(ds - cad_target))
+                    within.setdefault(f, []).append((cad_target - cad_tol) <= ds <= (cad_target + cad_tol))
+                med_abs_err = {f: (float(np.median(v)) if v else np.nan) for f, v in byf.items()}
+                within_pct = {f: (float(100.0 * (sum(v) / max(1, len(v)))) if v else np.nan) for f, v in within.items()}
+                # Compact CSV maps for the summary row
+                cad_stats["cad_median_abs_err_by_filter_csv"] = ",".join(f"{k}:{med_abs_err[k]:.2f}" for k in sorted(med_abs_err))
+                cad_stats["cad_within_pct_by_filter_csv"] = ",".join(f"{k}:{within_pct[k]:.1f}" for k in sorted(within_pct))
+                # Overall aggregates across filters
+                all_abs = [x for v in byf.values() for x in v]
+                all_within = [x for v in within.values() for x in v]
+                cad_stats["cad_median_abs_err_all_d"] = (float(np.median(all_abs)) if all_abs else np.nan)
+                cad_stats["cad_within_pct_all"] = (float(100.0 * (sum(all_within) / max(1, len(all_within)))) if all_within else np.nan)
+
+            nights_rows.append(
                 {
@@
                     "median_alt_deg": (float(np.median(alts)) if alts else np.nan),
+                    # NEW cadence summaries (strings for quick scanning)
+                    **cad_stats,
                 }
             )
```

### Notes on choices

* **Gate at candidate-construction time**: we trim `allowed` by cadence before picking the first filter, so we never choose a same-filter revisit “too soon”.
* **Use the window clock (true order)**: we evaluate cadence with the *serialized* visit start (`current_time_utc` → `sn_start_utc`), not the idealized `best_time_utc`.
* **SIMLIB epochs** now use the true serialized MJD (more realistic than idealized). If you need to keep prior behavior, add a toggle `cfg.simlib_use_true_mjd`.

## 3) Config surface

Add these to `PlannerConfig` with defaults (back-compat safe):

```python
cadence_enable: bool = True
cadence_per_filter: bool = True
cadence_days_target: float = 3.0
cadence_days_tolerance: float = 0.5   # for compliance stats only
cadence_jitter_days: float = 0.25     # softening the hard gate
cadence_bonus_sigma_days: float = 0.5 # width of the bonus peak
cadence_bonus_weight: float = 0.25    # max additive bonus
```

## 4) Nightly report additions

You’ll now get, per window:

* `cad_median_abs_err_by_filter_csv` like `g:0.42,r:0.37,i:0.55`
* `cad_within_pct_by_filter_csv` like `g:82.0,r:79.0,i:76.0`
* global `cad_median_abs_err_all_d` and `cad_within_pct_all`

…and per-exposure columns:

* `visit_start_utc`, `cadence_days_since`, `cadence_gate_passed`, `cadence_target_d`

## 5) Tests

Create `tests/test_cadence_per_filter.py` (or replace your previous 3-day test):

* Build a tiny CSV with one SN.
* Force two twilight nights 1 day apart.
* Set `cadence_enable=True`, `cadence_per_filter=True`, `cadence_days_target=3.0`, `cadence_jitter_days=0.25`.
* Assert that:

  * on Night 1 you can schedule g,r,i;
  * on Night 2 (Δt≈1 d), **g** is filtered out by cadence gate but **r** or **i** is allowed;
  * recorded `cadence_days_since` for the allowed filter on Night 2 is \~1 d and `cadence_gate_passed=True` (because it’s the first for that filter);
  * once a given filter does have history, it won’t be re-used until Δt ≥ 2.75 d.

## 6) Docs

* In README “Prioritization Strategy”, add “Per-filter cadence constraint” explaining:

  * same-epoch multi-band allowed;
  * hard gate for the same filter;
  * soft bonus nudges due-soon filters;
  * per-filter compliance metrics in nightly summary.


---

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
