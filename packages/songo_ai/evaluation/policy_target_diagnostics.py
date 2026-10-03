"""Mesures de richesse et stabilité des cibles Policy produites par MCTS."""

from __future__ import annotations

import math
import statistics
from itertools import combinations
from typing import Sequence


def policy_entropy(policy: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in policy if value > 0.0)


def jensen_shannon(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("policies must have the same length")
    midpoint = [(left + right) / 2.0 for left, right in zip(first, second)]

    def kl(values: Sequence[float]) -> float:
        return sum(
            value * math.log(value / middle)
            for value, middle in zip(values, midpoint)
            if value > 0.0
        )

    return 0.5 * kl(first) + 0.5 * kl(second)


def summarize_policy_repetitions(
    per_state_policies: Sequence[Sequence[Sequence[float]]],
) -> dict:
    """Agrège des répétitions sans comparer des états différents entre eux."""

    if not per_state_policies or any(not runs for runs in per_state_policies):
        raise ValueError("at least one policy run is required for every state")
    flat = [tuple(float(value) for value in policy) for runs in per_state_policies for policy in runs]
    if any(len(policy) != 7 for policy in flat):
        raise ValueError("every policy must contain seven entries")
    supports = [sum(value > 0.0 for value in policy) for policy in flat]
    entropies = [policy_entropy(policy) for policy in flat]
    variances = []
    pairwise_js = []
    pairwise_agreement = []
    per_state_unique = []
    for runs in per_state_policies:
        normalized = [tuple(float(value) for value in policy) for policy in runs]
        per_state_unique.append(len(set(normalized)))
        for action in range(7):
            variances.append(statistics.pvariance(policy[action] for policy in normalized))
        for first, second in combinations(normalized, 2):
            pairwise_js.append(jensen_shannon(first, second))
            pairwise_agreement.append(
                max(range(7), key=first.__getitem__)
                == max(range(7), key=second.__getitem__)
            )
    return {
        "states": len(per_state_policies),
        "runs_per_state": sorted({len(runs) for runs in per_state_policies}),
        "policies": len(flat),
        "one_hot_fraction": sum(size == 1 for size in supports) / len(supports),
        "visited_actions_mean": statistics.fmean(supports),
        "visited_actions_median": statistics.median(supports),
        "entropy_mean": statistics.fmean(entropies),
        "entropy_median": statistics.median(entropies),
        "policy_component_variance_mean": statistics.fmean(variances),
        "pairwise_jensen_shannon_mean": (
            statistics.fmean(pairwise_js) if pairwise_js else 0.0
        ),
        "pairwise_jensen_shannon_max": max(pairwise_js, default=0.0),
        "pairwise_argmax_agreement": (
            statistics.fmean(pairwise_agreement) if pairwise_agreement else 1.0
        ),
        "unique_targets_per_state_mean": statistics.fmean(per_state_unique),
        "states_with_stable_target_fraction": (
            sum(count == 1 for count in per_state_unique) / len(per_state_unique)
        ),
    }
