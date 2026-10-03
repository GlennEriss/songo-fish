"""Statistiques inter-seeds et décision bornée du diagnostic MCTS Lot 11."""

from __future__ import annotations

import statistics
from itertools import combinations
from typing import Mapping, Sequence

from .policy_target_diagnostics import jensen_shannon, policy_entropy


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q10": quantile(values, 0.10),
        "q90": quantile(values, 0.90),
        "q95": quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def interseed_stability_report(
    per_state_policies: Sequence[Sequence[Sequence[float]]],
    legal_counts: Sequence[int],
) -> dict:
    if len(per_state_policies) != len(legal_counts) or not per_state_policies:
        raise ValueError("policies and legal_counts must describe the same non-empty battery")
    selected = [
        (runs, legal_count)
        for runs, legal_count in zip(per_state_policies, legal_counts)
        if legal_count > 1
    ]
    if not selected:
        raise ValueError("battery must contain at least one non-forced state")
    policies = [tuple(policy) for runs, _ in selected for policy in runs]
    pair_js = []
    pair_agreement = []
    per_state_js = []
    per_state_agreement = []
    for runs, _ in selected:
        js_values = []
        agreements = []
        for first, second in combinations(runs, 2):
            js_values.append(jensen_shannon(first, second))
            agreements.append(
                max(range(7), key=first.__getitem__)
                == max(range(7), key=second.__getitem__)
            )
        pair_js.extend(js_values)
        pair_agreement.extend(agreements)
        per_state_js.append(statistics.fmean(js_values) if js_values else 0.0)
        per_state_agreement.append(statistics.fmean(agreements) if agreements else 1.0)
    supports = [sum(value > 0.0 for value in policy) for policy in policies]
    entropies = [policy_entropy(policy) for policy in policies]
    margins = []
    for policy in policies:
        ordered = sorted(policy, reverse=True)
        margins.append(ordered[0] - ordered[1])
    return {
        "states": len(selected),
        "runs": len(policies),
        "seeds_per_state": sorted({len(runs) for runs, _ in selected}),
        "pairwise_js": _distribution(pair_js),
        "per_state_js": _distribution(per_state_js),
        "pairwise_argmax_agreement": _distribution(
            [float(value) for value in pair_agreement]
        ),
        "per_state_argmax_agreement": _distribution(per_state_agreement),
        "support": _distribution([float(value) for value in supports]),
        "entropy": _distribution(entropies),
        "top1_top2_margin": _distribution(margins),
        "one_hot_fraction": sum(value == 1 for value in supports) / len(supports),
    }


def choose_smallest_mcts_budget(curve: Mapping[int, dict]) -> dict:
    """Choisit 16/32/64 par critères relatifs, puis plateau vers le suivant."""

    budgets = (8, 16, 32, 64)
    if any(budget not in curve for budget in budgets):
        raise ValueError("curve must contain budgets 8, 16, 32 and 64")
    baseline = curve[8]
    decisions = {}
    for index, budget in enumerate((16, 32, 64)):
        metrics = curve[budget]
        quality = {
            "js_reduced_40_percent": (
                metrics["pairwise_js"]["mean"]
                <= 0.60 * baseline["pairwise_js"]["mean"]
            ),
            "one_hot_at_most_10_percent": metrics["one_hot_fraction"] <= 0.10,
            "support_at_least_three": metrics["support"]["mean"] >= 3.0,
            "argmax_gain_5_points": (
                metrics["pairwise_argmax_agreement"]["mean"]
                >= baseline["pairwise_argmax_agreement"]["mean"] + 0.05
            ),
        }
        if budget == 64:
            plateau = True
            plateau_metrics = {"last_candidate": True}
        else:
            next_budget = (16, 32, 64)[index + 1]
            following = curve[next_budget]
            plateau_metrics = {
                "next_budget": next_budget,
                "js_absolute_gain": metrics["pairwise_js"]["mean"]
                - following["pairwise_js"]["mean"],
                "support_gain": following["support"]["mean"]
                - metrics["support"]["mean"],
                "entropy_gain": following["entropy"]["mean"]
                - metrics["entropy"]["mean"],
                "argmax_gain": following["pairwise_argmax_agreement"]["mean"]
                - metrics["pairwise_argmax_agreement"]["mean"],
            }
            plateau = (
                plateau_metrics["js_absolute_gain"] <= 0.02
                and plateau_metrics["support_gain"] <= 0.5
                and plateau_metrics["entropy_gain"] <= 0.10
                and plateau_metrics["argmax_gain"] <= 0.05
            )
        decisions[budget] = {
            "quality": quality,
            "quality_passed": all(quality.values()),
            "plateau_to_next": plateau,
            "plateau_metrics": plateau_metrics,
        }
    eligible = [
        budget
        for budget in (16, 32, 64)
        if decisions[budget]["quality_passed"] and decisions[budget]["plateau_to_next"]
    ]
    if eligible:
        selected = eligible[0]
        fallback = False
    else:
        # Le lot doit produire un pilote : utilise le meilleur JS parmi les
        # candidats, sans prétendre que les critères diagnostiques ont réussi.
        selected = min(
            (16, 32, 64), key=lambda budget: curve[budget]["pairwise_js"]["mean"]
        )
        fallback = True
    return {
        "selected_budget": selected,
        "selected_by_fallback_best_js": fallback,
        "decisions": decisions,
    }


def final_g2_readiness_gate(baseline: dict, pilot: dict, *, valid: bool) -> dict:
    checks = {
        "one_hot_reduced_25_percent": (
            pilot["policy"]["one_hot_fraction"]
            <= 0.75 * baseline["policy"]["one_hot_fraction"]
        ),
        "support_gain_half_action": (
            pilot["policy"]["support_mean"]
            >= baseline["policy"]["support_mean"] + 0.5
        ),
        "entropy_gain_20_percent": (
            pilot["policy"]["entropy_mean"]
            >= 1.20 * baseline["policy"]["entropy_mean"]
        ),
        "initial_js_reduced_25_percent": (
            pilot["repeated"]["initial_state"]["pairwise_jensen_shannon_mean"]
            <= 0.75
            * baseline["repeated"]["initial_state"]["pairwise_jensen_shannon_mean"]
        ),
        "serialization_and_terminals_valid": valid,
    }
    return {"ready_for_g2": all(checks.values()), "checks": checks}
