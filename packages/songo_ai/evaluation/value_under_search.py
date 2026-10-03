"""Outils du Lot 28 pour isoler et mesurer la Value sous MCTS."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def value_disagreement(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    """Compare deux séries de Value évaluées sur les mêmes états."""
    if len(left) != len(right) or not left:
        raise ValueError("value series must be non-empty and aligned")
    if any(not math.isfinite(float(x)) for x in (*left, *right)):
        raise ValueError("values must be finite")
    delta = [float(a) - float(b) for a, b in zip(left, right)]
    return {
        "count": len(delta),
        "mean_signed_delta": statistics.fmean(delta),
        "mean_abs_delta": statistics.fmean(abs(x) for x in delta),
        "sign_disagreement": statistics.fmean(
            (a > 0) != (b > 0) for a, b in zip(left, right)
        ),
        "left_mean": statistics.fmean(left),
        "right_mean": statistics.fmean(right),
        "left_variance": statistics.pvariance(left),
        "right_variance": statistics.pvariance(right),
    }


def value_metrics(prediction: Sequence[float], target: Sequence[float]) -> dict[str, float]:
    """Métriques offline terminales, sans pseudo-label de recherche."""
    if len(prediction) != len(target) or not prediction:
        raise ValueError("prediction and target must be non-empty and aligned")
    error = [float(p) - float(t) for p, t in zip(prediction, target)]
    return {
        "count": len(error),
        "mse": statistics.fmean(x * x for x in error),
        "mae": statistics.fmean(abs(x) for x in error),
        "bias": statistics.fmean(error),
        "prediction_mean": statistics.fmean(prediction),
        "prediction_variance": statistics.pvariance(prediction),
        "prediction_mean_abs": statistics.fmean(abs(x) for x in prediction),
        "sign_accuracy": statistics.fmean(
            (1 if p > 0 else -1 if p < 0 else 0)
            == (1 if t > 0 else -1 if t < 0 else 0)
            for p, t in zip(prediction, target)
        ),
    }


def local_value_consistency(parent: Sequence[float], child: Sequence[float]) -> dict[str, float]:
    """Mesure ``|V(parent)+V(child)|`` lorsque le trait alterne."""
    if len(parent) != len(child) or not parent:
        raise ValueError("parent and child values must be non-empty and aligned")
    residual = [abs(float(a) + float(b)) for a, b in zip(parent, child)]
    return {
        "count": len(residual),
        "mean_abs_residual": statistics.fmean(residual),
        "median_abs_residual": statistics.median(residual),
        "sign_oscillation_rate": statistics.fmean(
            (a > 0) == (b > 0) and abs(a) > 0.1 and abs(b) > 0.1
            for a, b in zip(parent, child)
        ),
        "extreme_discontinuity_rate": statistics.fmean(x > 1.0 for x in residual),
    }


def search_sensitivity(rows: Sequence[dict]) -> dict[str, object]:
    """Relie ΔValue, ΔQ/visites et changement d'action racine."""
    if not rows:
        raise ValueError("search sensitivity requires rows")
    required = {"delta_value", "delta_q", "visit_js", "action_flip"}
    if any(not required <= set(row) for row in rows):
        raise ValueError("a search sensitivity row is incomplete")
    flipped = [row for row in rows if row["action_flip"]]
    stable = [row for row in rows if not row["action_flip"]]
    mean = lambda group, key: statistics.fmean(abs(float(x[key])) for x in group) if group else None
    return {
        "positions": len(rows),
        "flip_rate": len(flipped) / len(rows),
        "flipped": {
            "count": len(flipped),
            "mean_abs_delta_value": mean(flipped, "delta_value"),
            "mean_abs_delta_q": mean(flipped, "delta_q"),
            "mean_visit_js": mean(flipped, "visit_js"),
        },
        "stable": {
            "count": len(stable),
            "mean_abs_delta_value": mean(stable, "delta_value"),
            "mean_abs_delta_q": mean(stable, "delta_q"),
            "mean_visit_js": mean(stable, "visit_js"),
        },
    }


def flip_rate(actions_low: Sequence[int], actions_high: Sequence[int]) -> float:
    if len(actions_low) != len(actions_high) or not actions_low:
        raise ValueError("action series must be non-empty and aligned")
    return statistics.fmean(a != b for a, b in zip(actions_low, actions_high))
