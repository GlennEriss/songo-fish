"""Metriques evenement rare, intervalles de confiance et bootstrap."""
from __future__ import annotations

import math

import numpy as np


def wilson(success: int, total: int, z: float = 1.96) -> list[float | None]:
    if total <= 0:
        return [None, None]
    p = success / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def average_precision(y: np.ndarray, scores: np.ndarray) -> float | None:
    """AP en escalier (pas d'interpolation trapezoidale optimiste)."""

    y = np.asarray(y, dtype=int)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / positives)


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float | None:
    """Mann-Whitney avec rangs moyens (ex-aequo comptes 1/2)."""

    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    i = 0
    sorted_scores = scores[order]
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def expected_calibration_error(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float | None:
    if len(y) == 0:
        return None
    y = np.asarray(y, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(error)


def _ratio(n: int, d: int) -> float | None:
    return n / d if d else None


def classification_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float, *, ece_bins: int = 10) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = p >= threshold
    truth = y.astype(bool)
    tp = int((pred & truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    fp = int((pred & ~truth).sum())
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "n": int(len(y)), "positives": int(truth.sum()), "threshold": threshold,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "recall": recall, "precision": _ratio(tp, tp + fp), "specificity": specificity,
        "npv": _ratio(tn, tn + fn), "fpr": _ratio(fp, fp + tn), "fnr": _ratio(fn, fn + tp),
        "balanced_accuracy": None if recall is None or specificity is None else 0.5 * (recall + specificity),
        "pr_auc": average_precision(y, p), "roc_auc": roc_auc(y, p),
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "ece": expected_calibration_error(y, p, ece_bins),
        "routing_rate": float(pred.mean()) if len(y) else None, "routed": int(pred.sum()),
    }


def proportion_intervals(m: dict) -> dict:
    return {
        "recall_wilson95": wilson(m["tp"], m["tp"] + m["fn"]),
        "precision_wilson95": wilson(m["tp"], m["tp"] + m["fp"]),
        "specificity_wilson95": wilson(m["tn"], m["tn"] + m["fp"]),
        "npv_wilson95": wilson(m["tn"], m["tn"] + m["fn"]),
        "fnr_wilson95": wilson(m["fn"], m["tp"] + m["fn"]),
        "fpr_wilson95": wilson(m["fp"], m["tn"] + m["fp"]),
        "routing_rate_wilson95": wilson(m["routed"], m["n"]),
        "positive_rate_wilson95": wilson(m["positives"], m["n"]),
    }


def bootstrap_ci(y: np.ndarray, scores: dict[str, np.ndarray], *, resamples: int, seed: int) -> dict:
    """IC95 bootstrap (percentile) pour AP/ROC-AUC et difference appariee."""

    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    names = list(scores)
    samples: dict[str, dict[str, list[float]]] = {n: {"pr_auc": [], "roc_auc": []} for n in names}
    diff: list[float] = []
    used = 0
    for _ in range(resamples):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        used += 1
        aps = {}
        for n in names:
            aps[n] = average_precision(yy, scores[n][idx])
            samples[n]["pr_auc"].append(aps[n])
            samples[n]["roc_auc"].append(roc_auc(yy, scores[n][idx]))
        if "combined" in aps and "classic" in aps:
            diff.append(aps["combined"] - aps["classic"])

    def ci(values: list[float]) -> list[float | None]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else [None, None]

    return {
        "resamples_requested": resamples,
        "resamples_used": used,
        "method": "nonparametric bootstrap, percentile CI95, resamples with a single class skipped",
        "models": {n: {k: ci(v) for k, v in samples[n].items()} for n in names},
        "ap_combined_minus_classic_ci95": ci(diff),
    }
