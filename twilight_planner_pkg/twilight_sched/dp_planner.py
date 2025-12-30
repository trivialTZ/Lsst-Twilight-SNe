"""Dynamic programming batch planner for twilight windows.

Responsibility
--------------
- Prefix sums of scores and times per filter.
- Generate candidate filter sequences.
- Compute the optimal batch plan (sequence + counts) under window time/costs.

Hot path
--------
- ``_plan_batches_by_dp``
"""

from __future__ import annotations

import itertools
import math
from typing import List

import numpy as np

from ..config import PlannerConfig
from .types import PairItem


def _prefix_scores(per_filter: dict[str, list[PairItem]]) -> dict[str, np.ndarray]:
    """Return per-filter prefix sums of scores."""

    prefixes: dict[str, np.ndarray] = {}
    for filt, items in per_filter.items():
        run = [0.0]
        acc = 0.0
        for it in items:
            acc += it.score
            run.append(acc)
        prefixes[filt] = np.array(run, dtype=float)
    return prefixes


def _prefix_times(per_filter: dict[str, list[PairItem]]) -> dict[str, np.ndarray]:
    """Return per-filter prefix sums of approximate visit durations."""

    prefixes: dict[str, np.ndarray] = {}
    for filt, items in per_filter.items():
        run = [0.0]
        acc = 0.0
        for it in items:
            acc += it.approx_time_s
            run.append(acc)
        prefixes[filt] = np.array(run, dtype=float)
    return prefixes


def _generate_filter_sequences(
    per_filter: dict[str, list[PairItem]],
    length: int,
    forced_first: str | None,
    cfg: PlannerConfig,
) -> list[list[str]]:
    """Return candidate filter sequences respecting optional forced first filter."""

    if length <= 0:
        return []
    filters = [f for f, items in per_filter.items() if items]
    if not filters:
        return []
    sequences: list[list[str]] = []
    required_first: str | None
    if forced_first:
        required_first = forced_first
    else:
        start_filter = getattr(cfg, "start_filter", None)
        required_first = start_filter if start_filter in filters else None
    for seq in itertools.product(filters, repeat=length):
        if required_first and seq[0] != required_first:
            continue
        # Remove redundant consecutive duplicates which would imply zero-cost swaps.
        if any(seq[i] == seq[i + 1] for i in range(length - 1)):
            continue
        sequences.append(list(seq))
    return sequences


def _plan_batches_by_dp(
    per_filter: dict[str, list[PairItem]],
    prefix_scores: dict[str, np.ndarray],
    time_left_s: float,
    cfg: PlannerConfig,
    forced_first: str | None,
) -> tuple[list[str], list[int], float]:
    """Return optimal (filter sequence, visit counts, total score) via DP."""

    if not per_filter:
        return ([], [], 0.0)
    filters = [f for f, items in per_filter.items() if items]
    t_units = {}
    for filt, items in per_filter.items():
        if items:
            t_units[filt] = float(np.mean([it.approx_time_s for it in items]))
        else:
            t_units[filt] = float(
                cfg.inter_exposure_min_s + cfg.exposure_by_filter.get(filt, 0.0)
            )
    exposures = [
        max(0.0, t - cfg.inter_exposure_min_s) for t in t_units.values() if t > 0.0
    ]
    if exposures:
        median_exp = float(np.median(exposures))
    else:
        median_exp = float(np.median(list(cfg.exposure_by_filter.values()) or [15.0]))
    if getattr(cfg, "n_estimate_mode", "guard_plus_exp") == "per_filter":
        t_ref = float(np.mean(list(t_units.values())))
    else:
        t_ref = float(cfg.inter_exposure_min_s + median_exp)
    t_ref = max(t_ref, 1.0)
    N = int(max(0, time_left_s // t_ref))
    if N <= 0:
        return ([], [], 0.0)
    K = max(1, int(math.ceil(cfg.filter_change_s / max(1.0, t_ref))))
    swaps_cap = min(
        int(getattr(cfg, "max_swaps_per_window", 2)),
        int(time_left_s // max(cfg.filter_change_s, 1.0)),
        max(0, len(filters) - 1),
    )
    dp_max_swaps = getattr(cfg, "dp_max_swaps", None)
    if isinstance(dp_max_swaps, int):
        swaps_cap = min(swaps_cap, max(0, dp_max_swaps))

    

    # Replace single-pass selection with layered improvement search
    def _best_for_length(length: int) -> tuple[list[str], list[int], float] | None:
        seqs = _generate_filter_sequences(per_filter, length, forced_first, cfg)
        best: tuple[list[str], list[int], float] | None = None
        best_total_here = -np.inf
        for seq in seqs:
            counts = [0] * length
            total = 0.0
            remaining = float(time_left_s)
            for i, f in enumerate(seq):
                prefix = prefix_scores.get(f)
                if prefix is None or prefix.size <= 1:
                    continue
                lo, hi = 0, len(prefix) - 1
                t_unit = float(
                    cfg.inter_exposure_min_s + cfg.exposure_by_filter.get(f, 0.0)
                )
                if i == 0:
                    # Do not charge an entry swap for the first DP segment: the scheduler
                    # can pre-position the carousel before the window begins, so the DP
                    # should allocate the full on-sky window to visits. Swap costs are
                    # still charged between subsequent segments.
                    swap_cost = 0.0
                else:
                    swap_cost = cfg.filter_change_s
                seg_budget = max(0.0, remaining - swap_cost)
                max_k = int(seg_budget // max(t_unit, 1.0))
                if max_k <= 0:
                    continue
                take = hi if max_k >= hi else max_k
                counts[i] = take
                total += float(prefix[take])
                remaining -= swap_cost + take * t_unit
                if remaining <= 0.0:
                    break
            if total > best_total_here:
                best_total_here = total
                best = (list(seq), list(counts), float(total))
        return best

    # Layer 0: best plan with 0 swaps (length=1)
    prev = _best_for_length(1)
    if not prev:
        return ([], [], 0.0)
    best_seq, best_counts, best_total = prev

    # Increase allowed swaps one layer at a time; stop if no strict improvement
    theta = float(getattr(cfg, "dp_hysteresis_theta", 0.0) or 0.0)
    for length in range(2, swaps_cap + 2):
        cand = _best_for_length(length)
        if not cand:
            break
        seq_k, counts_k, total_k = cand
        if total_k <= best_total * (1.0 + theta):
            break
        prev_visits = int(sum(best_counts)) if best_counts else 0
        if prev_visits > 0 and seq_k and best_seq and seq_k[0] == best_seq[0]:
            avg_prev = best_total / prev_visits
            swap_count = max(0, len(seq_k) - 1)
            opp_cost = avg_prev * swap_count * K
            improvement = total_k - best_total
            if improvement <= opp_cost:
                break
            payoff_s = float(
                getattr(cfg, "min_batch_payoff_s", None) or cfg.filter_change_s
            )
            if payoff_s > 0.0:
                visits_per_payoff = payoff_s / max(1.0, t_ref)
                min_gain = avg_prev * visits_per_payoff
                if improvement <= min_gain:
                    break
        # Accept the improved layer
        best_seq, best_counts, best_total = seq_k, counts_k, total_k

    return (list(best_seq), list(best_counts), float(best_total))
