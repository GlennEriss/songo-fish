"""Diagnostics offline séparant les sorties Policy et Value de checkpoints SRN."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import torch

from songo_ai.model import SRNBatchCollator

from .policy_mcts_signal import analyze_policy_mcts_signal
from .value_diagnostics import calibration_metrics


def compare_policy_models(models: Mapping[str, torch.nn.Module], examples: Sequence) -> dict:
    """Compare chaque Policy au même ``pi_MCTS`` stocké, sans mutation."""

    if not models:
        raise ValueError("models must not be empty")
    return {
        name: analyze_policy_mcts_signal(model, examples)
        for name, model in models.items()
    }


def _ply_bucket(ply: int) -> str:
    if ply <= 30:
        return "ply_0_30"
    if ply <= 90:
        return "ply_31_90"
    return "ply_91_plus"


def compare_value_models(
    models: Mapping[str, torch.nn.Module], examples: Sequence, *, batch_size: int = 512
) -> dict:
    """Compare les Value à ``z`` et produit une calibration descriptive.

    Les exemples tronqués sans cible terminale sont exclus. La segmentation
    utilise uniquement le joueur physique, le résultat et le ply déjà stockés.
    """

    labeled = [example for example in examples if example.value_target is not None]
    if not models or not labeled:
        raise ValueError("models and terminally labeled examples are required")
    collator = SRNBatchCollator()
    predictions = {name: [] for name in models}
    for model in models.values():
        model.eval()
    with torch.no_grad():
        for start in range(0, len(labeled), batch_size):
            batch = collator(labeled[start : start + batch_size])
            for name, model in models.items():
                _, values = model(batch.graph)
                predictions[name].extend(float(value) for value in values.cpu().tolist())

    targets = [float(example.value_target) for example in labeled]
    dimensions = {
        "player_to_move": [f"P{example.state.player_to_move}" for example in labeled],
        "result": ["win" if target == 1 else "draw" if target == 0 else "loss" for target in targets],
        "ply": [_ply_bucket(int(example.metadata.get("ply", 0))) for example in labeled],
    }
    report = {}
    for name, values in predictions.items():
        groups = {}
        for dimension, labels in dimensions.items():
            indices = defaultdict(list)
            for index, label in enumerate(labels):
                indices[label].append(index)
            groups[dimension] = {
                label: calibration_metrics(
                    [values[index] for index in selected],
                    [targets[index] for index in selected],
                )
                for label, selected in sorted(indices.items())
            }
        global_metrics = calibration_metrics(values, targets)
        global_metrics["prediction_std"] = float(torch.tensor(values).std(unbiased=False).item())
        report[name] = {"global": global_metrics, **groups}
    return {"examples": len(labeled), "excluded_unlabeled": len(examples) - len(labeled), "models": report}
