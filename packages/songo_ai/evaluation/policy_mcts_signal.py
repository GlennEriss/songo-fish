"""Diagnostic passif Policy brute → cible MCTS stockée."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Sequence

import torch

from songo_ai.model import SRNBatchCollator, policy_probabilities

from .policy_target_diagnostics import jensen_shannon, policy_entropy


def _quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _distribution(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q05": _quantile(values, 0.05),
        "q25": _quantile(values, 0.25),
        "q75": _quantile(values, 0.75),
        "q95": _quantile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def _phase(ply: int) -> str:
    if ply <= 30:
        return "ply_0_30"
    if ply <= 90:
        return "ply_31_90"
    return "ply_91_plus"


def analyze_policy_mcts_signal(model, examples: Sequence, *, batch_size: int = 512) -> dict:
    """Compare P_G2 à pi_MCTS sans recalculer ni modifier une cible."""

    if not examples:
        raise ValueError("examples must not be empty")
    rows = []
    model.eval()
    collator = SRNBatchCollator()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            batch = collator(chunk)
            logits, _ = model(batch.graph)
            raw_policies = policy_probabilities(logits, batch.legal_mask).cpu().tolist()
            for example, raw in zip(chunk, raw_policies):
                target = list(example.policy_target)
                preferred = max(range(7), key=target.__getitem__)
                raw_argmax = max(range(7), key=raw.__getitem__)
                raw_top2 = set(sorted(range(7), key=raw.__getitem__, reverse=True)[:2])
                target_top2 = set(sorted(range(7), key=target.__getitem__, reverse=True)[:2])
                kl = sum(
                    target[action] * math.log(target[action] / max(raw[action], 1e-12))
                    for action in range(7)
                    if target[action] > 0.0
                )
                ply = int(example.metadata.get("ply", 0))
                rows.append({
                    "js": jensen_shannon(raw, target),
                    "kl_pi_to_policy": kl,
                    "argmax_agreement": float(preferred == raw_argmax),
                    "mcts_preferred_in_raw_top2": float(preferred in raw_top2),
                    "top2_overlap": len(raw_top2 & target_top2) / 2.0,
                    "raw_entropy": policy_entropy(raw),
                    "mcts_entropy": policy_entropy(target),
                    "preferred_probability_delta": target[preferred] - raw[preferred],
                    "phase": _phase(ply),
                    "legal_count": sum(example.legal_mask),
                    "player": example.state.player_to_move,
                })

    metric_names = (
        "js", "kl_pi_to_policy", "argmax_agreement",
        "mcts_preferred_in_raw_top2", "top2_overlap", "raw_entropy",
        "mcts_entropy", "preferred_probability_delta",
    )

    def summarize(indices):
        return {name: _distribution([rows[i][name] for i in indices]) for name in metric_names}

    subgroups = {}
    for dimension in ("phase", "legal_count", "player"):
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[str(row[dimension])].append(index)
        subgroups[dimension] = {
            key: summarize(indices) for key, indices in sorted(groups.items())
        }
    return {
        "examples": len(rows),
        "direction": "KL(pi_MCTS || P_G2)",
        "global": summarize(range(len(rows))),
        "subgroups": subgroups,
    }
