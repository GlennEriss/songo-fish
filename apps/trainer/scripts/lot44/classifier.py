"""Regression logistique L2 deterministe (Newton/IRLS), calibration et seuil."""
from __future__ import annotations

import math

import numpy as np

from .artifacts import canonical_hash
from .config import PROTOCOL


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float, balanced: bool, names: list[str]) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    mean = x.mean(axis=0) if len(x) else np.zeros(x.shape[1])
    scale = x.std(axis=0) if len(x) else np.ones(x.shape[1])
    scale[scale == 0] = 1.0
    base = {"family": "l2_logistic_regression", "features": names, "mean": mean.tolist(), "scale": scale.tolist(), "l2": l2, "balanced_class_weight": balanced, "n": len(y), "positives": positives}
    if positives == 0 or negatives == 0:
        prior = (positives + 0.5) / (len(y) + 1.0)
        return {**base, "status": "DEGENERATE_SINGLE_CLASS", "weights": [0.0] * x.shape[1], "intercept": math.log(prior / (1 - prior)), "iterations": 0, "converged": True}
    z = (x - mean) / scale
    design = np.hstack([z, np.ones((len(z), 1))])
    sample = np.where(y == 1, len(y) / (2.0 * positives), len(y) / (2.0 * negatives)) if balanced else np.ones(len(y))
    penalty = np.full(design.shape[1], l2)
    penalty[-1] = 0.0
    beta = np.zeros(design.shape[1])
    converged = False
    iterations = 0
    for iterations in range(1, 101):
        p = _sigmoid(design @ beta)
        gradient = design.T @ (sample * (p - y)) + penalty * beta
        hessian = (design * (sample * p * (1 - p))[:, None]).T @ design + np.diag(penalty + 1e-9)
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-10:
            converged = True
            break
    return {**base, "status": "FITTED", "weights": beta[:-1].tolist(), "intercept": float(beta[-1]), "iterations": iterations, "converged": converged}


def decision_function(model: dict, x: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, dtype=float) - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return z @ np.asarray(model["weights"]) + model["intercept"]


def predict_proba(model: dict, x: np.ndarray) -> np.ndarray:
    return _sigmoid(decision_function(model, x))


def stratified_folds(fingerprints: list[str], y: np.ndarray, *, k: int, seed: int) -> np.ndarray:
    """Folds stratifies deterministes (une position par partie => folds groupes)."""

    folds = np.zeros(len(y), dtype=int)
    for cls in (0, 1):
        idx = [i for i in range(len(y)) if int(y[i]) == cls]
        idx.sort(key=lambda i: canonical_hash({"seed": seed, "fold": fingerprints[i]}))
        for rank, i in enumerate(idx):
            folds[i] = rank % k
    return folds


def out_of_fold_scores(x: np.ndarray, y: np.ndarray, fingerprints: list[str], *, names: list[str], seed: int) -> tuple[np.ndarray | None, int]:
    positives = int(y.sum())
    k = min(PROTOCOL["cv_folds_max"], positives, len(y) - positives)
    if k < 2:
        return None, k
    folds = stratified_folds(fingerprints, y, k=k, seed=seed)
    scores = np.zeros(len(y))
    for fold in range(k):
        train = folds != fold
        model = fit_logistic(x[train], y[train], l2=PROTOCOL["l2_strength"], balanced=True, names=names)
        scores[~train] = decision_function(model, x[~train])
    return scores, k


def fit_calibrator(raw_scores: np.ndarray, y: np.ndarray) -> dict:
    """Platt sur les scores bruts de CALIBRATION, sinon identite documentee."""

    positives = int(np.asarray(y).sum())
    negatives = len(y) - positives
    minimum = PROTOCOL["calibration_min_class_count"]
    if positives >= minimum and negatives >= minimum:
        platt = fit_logistic(np.asarray(raw_scores).reshape(-1, 1), y, l2=1e-4, balanced=False, names=["raw_score"])
        return {"method": "PLATT", "model": platt, "justification": f"calibration has {positives} positives and {negatives} negatives (>= {minimum} each)"}
    return {"method": "IDENTITY", "model": None, "justification": f"calibration has {positives} positives / {negatives} negatives (< {minimum}); Platt would be unstable, sigmoid(raw score) is used uncalibrated"}


def apply_calibrator(calibrator: dict, raw_scores: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_scores, dtype=float)
    if calibrator["method"] == "PLATT":
        return predict_proba(calibrator["model"], raw.reshape(-1, 1))
    return _sigmoid(raw)


def choose_threshold(dev_scores: np.ndarray, dev_y: np.ndarray) -> dict:
    positives = np.asarray(dev_scores)[np.asarray(dev_y) == 1]
    if len(positives) == 0:
        return {"threshold": 0.0, "rule": "NO_DEVELOPMENT_POSITIVE_ROUTE_ALL", "development_positives": 0}
    floor = float(positives.min())
    return {"threshold": PROTOCOL["threshold_safety_factor"] * floor, "min_positive_score": floor, "rule": PROTOCOL["threshold_rule"], "development_positives": int(len(positives))}
