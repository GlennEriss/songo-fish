"""Statistiques descriptives et calibration terminale de la Value SRN."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Mapping, Sequence


HORIZON_BUCKETS = ("0-5", "6-15", "16-30", "31-60", ">60")


def quantile(values: Sequence[float], probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("values must not be empty")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def value_distribution_statistics(values: Sequence[float]) -> dict:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("values must not be empty")
    if any(not math.isfinite(value) or not -1.000001 <= value <= 1.000001 for value in values):
        raise ValueError("Value predictions must be finite and in [-1, 1]")
    absolute = [abs(value) for value in values]
    count = len(values)
    return {
        "count": count,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "minimum": min(values),
        "q05": quantile(values, 0.05),
        "q25": quantile(values, 0.25),
        "q50": quantile(values, 0.50),
        "q75": quantile(values, 0.75),
        "q95": quantile(values, 0.95),
        "maximum": max(values),
        "mean_absolute_value": statistics.fmean(absolute),
        "fraction_abs_gt_0_5": sum(value > 0.5 for value in absolute) / count,
        "fraction_abs_gt_0_8": sum(value > 0.8 for value in absolute) / count,
        "fraction_abs_gt_0_95": sum(value > 0.95 for value in absolute) / count,
        "fraction_negative_saturated": sum(value <= -0.95 for value in values) / count,
        "fraction_positive_saturated": sum(value >= 0.95 for value in values) / count,
    }


def horizon_bucket(distance_to_terminal: int) -> str:
    if distance_to_terminal < 0:
        raise ValueError("distance_to_terminal must be non-negative")
    if distance_to_terminal <= 5:
        return "0-5"
    if distance_to_terminal <= 15:
        return "6-15"
    if distance_to_terminal <= 30:
        return "16-30"
    if distance_to_terminal <= 60:
        return "31-60"
    return ">60"


def calibration_metrics(predictions: Sequence[float], targets: Sequence[float]) -> dict:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must have the same non-zero length")
    predictions = [float(value) for value in predictions]
    targets = [float(value) for value in targets]
    if any(target not in (-1.0, 0.0, 1.0) for target in targets):
        raise ValueError("calibration targets must be -1, 0 or +1")
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    return {
        "count": len(errors),
        "mse": statistics.fmean(error * error for error in errors),
        "mae": statistics.fmean(abs(error) for error in errors),
        "sign_accuracy": statistics.fmean(
            float(
                (1 if prediction > 0.0 else -1 if prediction < 0.0 else 0)
                == (1 if target > 0.0 else -1 if target < 0.0 else 0)
            )
            for prediction, target in zip(predictions, targets)
        ),
        "prediction_mean": statistics.fmean(predictions),
        "target_mean": statistics.fmean(targets),
    }


def calibration_report(
    predictions_by_model: Mapping[str, Sequence[float]],
    targets: Sequence[float],
    distances_to_terminal: Sequence[int],
    players_to_move: Sequence[int],
) -> dict:
    size = len(targets)
    if size == 0:
        raise ValueError("calibration set must not be empty")
    if len(distances_to_terminal) != size or len(players_to_move) != size:
        raise ValueError("calibration metadata lengths do not match targets")
    if any(len(values) != size for values in predictions_by_model.values()):
        raise ValueError("prediction lengths do not match targets")

    result = {}
    for model_name, predictions in predictions_by_model.items():
        groups: dict[str, dict[str, list[int]]] = {
            "horizon": defaultdict(list),
            "player_to_move": defaultdict(list),
            "result": defaultdict(list),
        }
        for index, (target, distance, player) in enumerate(
            zip(targets, distances_to_terminal, players_to_move)
        ):
            groups["horizon"][horizon_bucket(int(distance))].append(index)
            groups["player_to_move"][f"P{int(player)}"].append(index)
            label = "win" if target == 1 else "draw" if target == 0 else "loss"
            groups["result"][label].append(index)

        stratified = {}
        for group_name, group_values in groups.items():
            stratified[group_name] = {}
            ordered_keys = (
                HORIZON_BUCKETS
                if group_name == "horizon"
                else sorted(group_values)
            )
            for key in ordered_keys:
                indices = group_values.get(key, [])
                if indices:
                    stratified[group_name][key] = calibration_metrics(
                        [predictions[index] for index in indices],
                        [targets[index] for index in indices],
                    )
        result[model_name] = {
            "global": calibration_metrics(predictions, targets),
            **stratified,
        }
    return result

