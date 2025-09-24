# Filter Change Logic (In-Window)

This note summarizes how the twilight planner decides when and how to change filters within a twilight window, and how the current notebook settings affect that behavior.

## Overview

- The scheduler first determines, per target and per window, which filters are feasible: SNR/m5 viability → intersect with Sun-alt policy → pass Moon-separation constraints.
- For each target, it chooses a “first filter” (the band to attempt first). If a visit uses multiple filters, that creates an internal (within-visit) filter change; otherwise, changes happen only between targets.
- Between targets, the scheduler maintains a per-window carousel state and accounts for the time/penalty of swapping filters. It limits the number of swaps per window and amortizes swap cost by remaining visit capacity in that window.

File references:
- First-filter selection: `twilight_planner_pkg/astro_utils.py:363`
- Legacy scheduler main loop: `twilight_planner_pkg/scheduler.py:1560`, cost model around `twilight_planner_pkg/scheduler.py:1676` and `:1756`
- New scheduler (refactor) candidate preparation: `twilight_planner_pkg/scheduler_new.py:359`
- Config knobs: `twilight_planner_pkg/config.py:66` (first_filter_order), `:71` (first_filter_bonus_weights), `:107` (min_moon_sep_by_filter), `:115` (require_single_time_for_all_filters), `:185` (max_swaps_per_window), `:47` (inter_exposure_min_s)

## Step 1 — Per-target feasible filters

- SNR/m5 gate using sky model and exposure time per band → `allowed_filters_for_window(...)`.
- Enforce window Sun-altitude policy (e.g., which filters are allowed at −12° to −8° Sun alt).
- Enforce per-filter Moon-separation requirement (scaled by Moon phase/altitude).

References:
- `twilight_planner_pkg/scheduler.py:439` (feasibility and policy gate)
- `twilight_planner_pkg/scheduler_new.py:362` (refactored path)

## Step 2 — First filter selection (per target)

Order of precedence (all work within the already-feasible set):

1) User hook (optional)
- If `cfg.pick_first_filter` is callable, it is invoked first and can directly return the chosen band. Context includes tracker, Sun alt, Moon OK flags, per-target mags, current carousel filter, and the best-time MJD.
- Reference: `twilight_planner_pkg/astro_utils.py:404`

2) Hybrid “unseen filter” preference (if still in Hybrid stage)
- If the target hasn’t yet achieved the hybrid color/exposure goal, the planner prefers a band not yet seen for that target.
- We score unseen candidates with the same filter bonus described below (cadence/diversity/cosmology) multiplied by optional per-filter weights, and pick the top.
- Reference: `twilight_planner_pkg/astro_utils.py:389` (history), logic continues through `:427-466`.

3) Scored ordering (general case)
- Every candidate band gets a score = `compute_filter_bonus(...) × first_filter_bonus_weights.get(band, 1.0)`.
  - `compute_filter_bonus` is the existing value function that already blends:
    - Cadence Gaussian bonus (target days, sigma, weight)
    - Cosmology/band boost (color deficit or per-filter diversity)
  - Weights are new and optional; use them to nudge the scoring (e.g., set z < 1.0).
- Ties are broken using `first_filter_order` and then a default red-to-blue list `[y, z, i, r, g, u]`.
- A tiny bias favors staying on the current carousel filter on exact ties.

References:
- Selection entry point and scoring: `twilight_planner_pkg/astro_utils.py:430-466`
- Underlying bonus function: `twilight_planner_pkg/priority.py:147`
- Config weights/ordering: `twilight_planner_pkg/config.py:66` and `:71`

## Step 3 — Within-visit filter changes (internal)

- Controlled by `filters_per_visit_cap` and `auto_color_pairing`.
- If the cap ≥ 2, the planner may add a second filter (pairing prefers opposite color group), paying internal filter-change overhead and enforcing the `inter_exposure_min_s` guard between consecutive exposures.
- When the cap = 1 (the notebook default), visits are single-band and there are no internal filter changes.

References:
- Per-visit selection and timing: `twilight_planner_pkg/scheduler.py:570-696`
- Guard/overhead calculation helpers: `twilight_planner_pkg/astro_utils.py:845-901`

## Step 4 — Between-target carousel swaps (cross-visit)

- The window maintains `state_filter` (current carousel filter) and counts swaps per window.
- When choosing the next target, costs include:
  - Slew time to the new field
  - If the new target’s first filter ≠ `state_filter`, a swap penalty is added
  - The swap penalty is amortized by remaining window capacity (more time left → penalty shared across expected future visits)
  - If the first filter provides strong color benefit, the penalty may be scaled down by `swap_cost_scale_color`
- Hard limit on swaps per window via `max_swaps_per_window` (default 2). During backfill, one extra swap is allowed if otherwise time would go unused.
- After a visit completes, `state_filter` becomes the last filter used in that visit.

References:
- State, swaps, and updates: `twilight_planner_pkg/scheduler.py:1560-1634` and `:2487-2498`
- Cost and amortization: `twilight_planner_pkg/scheduler.py:1676-1790`
- Backfill extra swap: `twilight_planner_pkg/scheduler.py:1837` (context in backfill loop)

## Current Notebook Settings and Effects

- `filters_per_visit_cap = 1`: visits are single-band → no internal (within-visit) filter changes; all changes are between targets.
- `allow_filter_changes_in_twilight = True`: carousel swaps are permitted in twilight.
- `max_swaps_per_window = 2`: limits costly carousel changes per window; backfill may allow one extra swap if needed to consume remaining time.
- `first_filter_bonus_weights = {'g': 1.0, 'r': 1.0, 'i': 1.0, 'z': 0.4}`: keeps g/r/i neutral and nudges z down; the scoring still uses the same cadence/diversity/cosmology machinery, only reweighted.
- `first_filter_order = ['i', 'r', 'g', 'z']`: tie-breaker list if scores are equal; does not bypass SNR/Moon/Sun constraints.

## Tuning Tips

- Prefer g/r/i equally but use z less often: keep the weights as above, or reduce z further (e.g., 0.3) if z still wins too often due to cadence/diversity.
- Reduce swap churn: lower `max_swaps_per_window`, or increase `swap_amortize_min` to amortize swap cost over more prospective visits.
- Encourage occasional color pairs in a single visit: set `filters_per_visit_cap ≥ 2` and leave `auto_color_pairing=True`.
- Make Moon avoidance stricter in a band: raise `min_moon_sep_by_filter[band]`.

## Cross-Visit Target Switching

This section details how the planner switches targets (and implicitly filters) between visits inside a window.

- Candidate pool per window
  - Build feasible filters per target (SNR/m5 → Sun-alt policy → Moon separation). Discard targets that end with an empty policy-allowed set.
  - References: scheduler path `twilight_planner_pkg/scheduler.py:439-468`; refactor path `twilight_planner_pkg/scheduler_new.py:362-386`.

- First-filter assignment per target
  - Use the selection flow described above to assign `first_filter` and keep `policy_allowed` for each candidate.
  - Reference: `twilight_planner_pkg/scheduler.py:471-525` (legacy) and `twilight_planner_pkg/scheduler_new.py:389-407` (refactor).

- Batch by first_filter
  - Group candidates by `first_filter` and create a per-window batch order (may be nudged by the first-filter cycle/palette mechanisms).
  - Reference: `twilight_planner_pkg/scheduler.py:1560-1675`.

- Cadence gating for costed selection
  - Before computing costs, restrict a target’s usable filters to those passing `cadence_gate` at the current moment; this can affect which band is optimal for the cost model.
  - Reference: `twilight_planner_pkg/scheduler.py:1687-1742`.

- Cost model to pick the next target
  - Slew cost: great-circle separation fed to `slew_time_seconds`.
  - Carousel swap penalty: if the chosen band ≠ current `state_filter`, add an amortized swap penalty.
    - Amortization: divide `filter_change_s` by k, estimated from time_left and a conservative per-visit wall-time; min amortization controlled by `swap_amortize_min`.
    - Color-aware scaling: if the chosen band yields clear color benefit (boost > 1), scale penalty by `swap_cost_scale_color` (< 1) to encourage that swap.
  - Hard limit: when `swap_count_by_window ≥ max_swaps_per_window`, any option that requires a swap is treated as infeasible (very high cost).
  - References: `twilight_planner_pkg/scheduler.py:1676-1790` (slew + amortized swap + scaling), `:1766-1770` (swap cap).

- Choose and execute
  - Pick the minimum-cost target in the active batch; if none feasible, move to the next batch or stop when capped.
  - Schedule the visit, update running clock, window sums, and emit plan/SIMLIB rows as configured.
  - References: `twilight_planner_pkg/scheduler.py:1560-1660, 2611`.

- State updates after a visit
  - Update `state_filter` to the last used filter in the visit; increment `swap_count_by_window` if a cross-visit swap occurred.
  - Track internal changes (within visit), slew times, sky mags, and the set of filters used for window KPIs.
  - References: `twilight_planner_pkg/scheduler.py:1625-1634, 2570-2571`.

- Backfill and repeats
  - After the first pass, run backfill/repeats with the same cost model to utilize remaining time.
  - A single extra swap beyond `max_swaps_per_window` is permitted during backfill if it prevents idle time.
  - Reference: `twilight_planner_pkg/scheduler.py:1837` (extra swap context) and backfill loops `:2046+`.

- Guards and pacing
  - `inter_exposure_min_s` enforces minimum gap; it also feeds the per-visit time estimate used for swap amortization.
  - `allow_filter_changes_in_twilight` must be true to permit carousel swaps inside twilight windows.
  - References: `twilight_planner_pkg/scheduler.py:691, 1680-1699` (gap/time estimate), config at `twilight_planner_pkg/config.py:364`.

- Strategy-neutral scoring
  - The first-filter scoring embeds your cadence/diversity/cosmology priorities and any per-band weights. The cross-visit selector then balances those priorities against slew and swap costs.
  - References: `twilight_planner_pkg/priority.py:147` (bonus), `twilight_planner_pkg/astro_utils.py:430-466` (weighted first-filter choice).
