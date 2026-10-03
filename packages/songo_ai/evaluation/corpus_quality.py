"""Diagnostics reproductibles et gate de qualité pour les corpus D_RL."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from typing import Sequence

from songo_ai.dataset import RLTrainingExample

from .policy_target_diagnostics import jensen_shannon, policy_entropy


def _top_margin(policy: Sequence[float]) -> float:
    ordered = sorted((float(value) for value in policy), reverse=True)
    return ordered[0] - ordered[1]


def policy_corpus_metrics(examples: Sequence[RLTrainingExample]) -> dict:
    if not examples:
        raise ValueError("examples must not be empty")
    if any(example.policy_target is None or example.visit_counts is None for example in examples):
        raise ValueError("all examples must contain policy_target and visit_counts")
    supports = [
        sum(value > 0.0 for value in example.policy_target) for example in examples
    ]
    legal_counts = [sum(example.legal_mask) for example in examples]
    visit_totals = [sum(example.visit_counts) for example in examples]
    return {
        "examples": len(examples),
        "one_hot_fraction": sum(size == 1 for size in supports) / len(examples),
        "support_mean": statistics.fmean(supports),
        "support_median": statistics.median(supports),
        "entropy_mean": statistics.fmean(
            policy_entropy(example.policy_target) for example in examples
        ),
        "legal_actions_mean": statistics.fmean(legal_counts),
        "visited_actions_mean": statistics.fmean(supports),
        "top1_top2_margin_mean": statistics.fmean(
            _top_margin(example.policy_target) for example in examples
        ),
        "visit_total_histogram": dict(sorted(Counter(visit_totals).items())),
        "support_histogram": dict(sorted(Counter(supports).items())),
        "legal_action_histogram": dict(sorted(Counter(legal_counts).items())),
        "visit_count_value_histogram": dict(
            sorted(
                Counter(
                    count
                    for example in examples
                    for count in example.visit_counts
                    if count > 0
                ).items()
            )
        ),
    }


def repeated_state_metrics(examples: Sequence[RLTrainingExample]) -> dict:
    groups = defaultdict(list)
    for example in examples:
        groups[(example.state.board, example.state.player_to_move)].append(example)
    repeated = [rows for rows in groups.values() if len(rows) > 1]
    pair_js = []
    pair_agreement = []
    component_variances = []
    for rows in repeated:
        policies = [example.policy_target for example in rows]
        for first, second in combinations(policies, 2):
            pair_js.append(jensen_shannon(first, second))
            pair_agreement.append(
                max(range(7), key=first.__getitem__)
                == max(range(7), key=second.__getitem__)
            )
        for action in range(7):
            component_variances.append(
                statistics.pvariance(policy[action] for policy in policies)
            )
    initial_key = (tuple([5] * 14 + [0, 0]), 1)
    initial = groups.get(initial_key, [])
    initial_policies = [example.policy_target for example in initial]
    initial_js = [
        jensen_shannon(first, second)
        for first, second in combinations(initial_policies, 2)
    ]
    initial_agreement = [
        max(range(7), key=first.__getitem__)
        == max(range(7), key=second.__getitem__)
        for first, second in combinations(initial_policies, 2)
    ]
    return {
        "unique_states": len(groups),
        "repeated_states": len(repeated),
        "repeated_occurrences": sum(len(rows) for rows in repeated),
        "pairwise_jensen_shannon_mean": statistics.fmean(pair_js) if pair_js else 0.0,
        "pairwise_argmax_agreement": (
            statistics.fmean(pair_agreement) if pair_agreement else 1.0
        ),
        "policy_component_variance_mean": (
            statistics.fmean(component_variances) if component_variances else 0.0
        ),
        "initial_state": {
            "occurrences": len(initial),
            "unique_targets": len(set(initial_policies)),
            "pairwise_jensen_shannon_mean": (
                statistics.fmean(initial_js) if initial_js else 0.0
            ),
            "pairwise_argmax_agreement": (
                statistics.fmean(initial_agreement) if initial_agreement else 1.0
            ),
            "policy_component_variance_mean": (
                statistics.fmean(
                    statistics.pvariance(policy[action] for policy in initial_policies)
                    for action in range(7)
                )
                if initial_policies
                else 0.0
            ),
        },
    }


def evaluate_corpus_quality_gate(
    baseline_policy: dict,
    baseline_repeated: dict,
    candidate_policy: dict,
    candidate_repeated: dict,
    *,
    serialization_valid: bool,
    terminal_labeled_examples: int,
) -> dict:
    """Gate relatif prévisible, indépendant d'un seuil absolu inventé."""

    checks = {
        "one_hot_halved": (
            candidate_policy["one_hot_fraction"]
            <= 0.5 * baseline_policy["one_hot_fraction"]
        ),
        "support_gain_at_least_one": (
            candidate_policy["support_mean"] >= baseline_policy["support_mean"] + 1.0
        ),
        "entropy_gain_at_least_50_percent": (
            candidate_policy["entropy_mean"] >= 1.5 * baseline_policy["entropy_mean"]
        ),
        "initial_js_halved": (
            candidate_repeated["initial_state"]["pairwise_jensen_shannon_mean"]
            <= 0.5
            * baseline_repeated["initial_state"]["pairwise_jensen_shannon_mean"]
        ),
        "serialization_valid": serialization_valid,
        "has_terminal_value_labels": terminal_labeled_examples > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "threshold_contract": {
            "one_hot": "candidate <= 50% of Lot 5",
            "support": "candidate >= Lot 5 + 1 action",
            "entropy": "candidate >= 150% of Lot 5",
            "initial_stability": "candidate initial-state JS <= 50% of Lot 5",
        },
    }
