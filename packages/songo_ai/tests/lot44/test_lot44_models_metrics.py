import itertools

import numpy as np
import pytest

from lot44.accounting import compute_accounting, mcts_resume_audit, strategy_costs
from lot44.classifier import apply_calibrator, choose_threshold, decision_function, fit_calibrator, fit_logistic, out_of_fold_scores, predict_proba, stratified_folds
from lot44.config import PROTOCOL
from lot44.evaluation import average_precision, bootstrap_ci, classification_metrics, expected_calibration_error, proportion_intervals, roc_auc, wilson


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + 0.5 * rng.normal(size=n) > 1.2).astype(float)
    return x, y


def test_logistic_learns_signal_and_is_deterministic():
    x, y = toy()
    m1 = fit_logistic(x, y, l2=1.0, balanced=True, names=["a", "b", "c"])
    m2 = fit_logistic(x, y, l2=1.0, balanced=True, names=["a", "b", "c"])
    assert m1 == m2 and m1["status"] == "FITTED" and m1["converged"]
    assert m1["weights"][0] > 1.0 and abs(m1["weights"][1]) < m1["weights"][0]
    assert roc_auc(y, predict_proba(m1, x)) > 0.9


def test_l2_shrinks_weights():
    x, y = toy()
    weak = fit_logistic(x, y, l2=0.01, balanced=True, names=["a", "b", "c"])
    strong = fit_logistic(x, y, l2=100.0, balanced=True, names=["a", "b", "c"])
    assert np.linalg.norm(strong["weights"]) < np.linalg.norm(weak["weights"])


def test_degenerate_single_class_and_constant_feature():
    x = np.ones((10, 2))
    m = fit_logistic(x, np.zeros(10), l2=1.0, balanced=True, names=["a", "b"])
    assert m["status"] == "DEGENERATE_SINGLE_CLASS" and np.all(predict_proba(m, x) < 0.1)
    x2, y2 = toy(50)
    x2[:, 2] = 3.0
    m2 = fit_logistic(x2, y2, l2=1.0, balanced=True, names=["a", "b", "c"])
    assert np.isfinite(decision_function(m2, x2)).all()


def test_stratified_folds_and_oof():
    x, y = toy()
    fps = [f"{i:064x}" for i in range(len(y))]
    folds = stratified_folds(fps, y, k=5, seed=1)
    for f in range(5):
        assert abs(y[folds == f].sum() - y.sum() / 5) <= 1
    scores, k = out_of_fold_scores(x, y, fps, names=["a", "b", "c"], seed=1)
    assert k == 5 and scores.shape == y.shape and roc_auc(y, scores) > 0.85
    none, k = out_of_fold_scores(x[:10], np.r_[1.0, np.zeros(9)], fps[:10], names=["a", "b", "c"], seed=1)
    assert none is None and k == 1


def test_calibrator_rule_and_threshold_rule():
    raw = np.linspace(-3, 3, 40)
    y = (raw > 1).astype(float)
    platt = fit_calibrator(raw, y)
    assert platt["method"] == "PLATT"
    probs = apply_calibrator(platt, raw)
    assert np.all(np.diff(probs) >= 0)
    few = fit_calibrator(raw[:20], np.r_[np.zeros(17), np.ones(3)])
    assert few["method"] == "IDENTITY" and "Platt would be unstable" in few["justification"]
    t = choose_threshold(np.array([0.1, 0.5, 0.6, 0.9]), np.array([0, 1, 0, 1]))
    assert t["threshold"] == pytest.approx(PROTOCOL["threshold_safety_factor"] * 0.5)
    assert choose_threshold(np.array([0.2]), np.array([0]))["threshold"] == 0.0


def test_metrics_confusion_and_ratios():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1])
    m = classification_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["recall"] == pytest.approx(2 / 3) and m["precision"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(0.8) and m["npv"] == pytest.approx(0.8)
    assert m["fpr"] == pytest.approx(0.2) and m["fnr"] == pytest.approx(1 / 3)
    assert m["balanced_accuracy"] == pytest.approx(0.5 * (2 / 3 + 0.8))
    assert m["routing_rate"] == pytest.approx(3 / 8) and m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    empty = classification_metrics(np.zeros(4, dtype=int), np.zeros(4), 0.5)
    assert empty["recall"] is None and empty["pr_auc"] is None and empty["roc_auc"] is None


def test_average_precision_and_roc_against_brute_force():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.5, 0.5, 0.5])
    assert average_precision(y, s) == pytest.approx((1 / 1 + 2 / 3 + 3 / 6) / 3)
    pairs = [(a, b) for a, b in itertools.product(s[y == 1], s[y == 0])]
    brute = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs])
    assert roc_auc(y, s) == pytest.approx(brute)


def test_ece_and_wilson():
    assert expected_calibration_error(np.array([1, 0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert expected_calibration_error(np.array([0, 0]), np.array([0.95, 0.95])) == pytest.approx(0.95)
    assert wilson(0, 0) == [None, None]
    low, high = wilson(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-3) and high == 1.0
    low0, high0 = wilson(0, 15)
    assert low0 == 0.0 and high0 == pytest.approx(0.2039, abs=1e-3)


def test_zero_false_negatives_still_has_uncertainty():
    y = np.array([1] * 5 + [0] * 20)
    m = classification_metrics(y, y.astype(float), 0.5)
    ci = proportion_intervals(m)["recall_wilson95"]
    assert m["fn"] == 0 and ci[1] == 1.0 and ci[0] < 0.6


def test_bootstrap_is_seeded():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    scores = {"combined": np.linspace(1, 0, 10), "classic": np.linspace(0, 1, 10)}
    a = bootstrap_ci(y, scores, resamples=200, seed=3)
    assert a == bootstrap_ci(y, scores, resamples=200, seed=3)
    assert a["resamples_used"] > 150 and a["ap_combined_minus_classic_ci95"][0] is not None


def test_restart_and_resume_accounting():
    sims = {"l1": np.array([4096.0, 4096]), "l2": np.array([8192.0, 8192]), "l3": np.array([32768.0, 32768]), "ref": np.array([65536.0, 65536])}
    routed = np.array([True, False])
    assert strategy_costs(sims, routed, base="ladder") == {"restart": 2 * (4096 + 8192 + 32768) + 65536, "resume": 2 * 32768 + 32768}
    assert strategy_costs(sims, routed, base="l3") == {"restart": 2 * 32768 + 65536, "resume": 2 * 32768 + 32768}
    assert strategy_costs(sims, routed, base="ref") == {"restart": 2 * 65536, "resume": 2 * 65536}
    labels = [{"label": 1, "deeper_regret": 0.2, "high_severity": True}, {"label": 1, "deeper_regret": 0.03, "high_severity": False}]
    acc = compute_accounting(sims, labels, {"classifier": routed, "conservative": np.array([True, True])})["strategies"]
    assert acc["UNIFORM_32768"]["restart_total_simulations"] == 2 * 32768 and acc["UNIFORM_32768"]["recall"] == 0.0
    assert acc["UNIFORM_65536"]["restart_total_simulations"] == 2 * 65536 and acc["UNIFORM_65536"]["recall"] == 1.0
    router = acc["CALIBRATED_ROUTER_32768_TO_65536"]
    assert router["false_negatives"] == 1 and router["worst_false_negative_regret"] == 0.03 and router["high_severity_false_negatives"] == 0
    assert router["resume_semantics"] == "COUNTERFACTUAL_ESTIMATE" and router["restart_semantics"] == "MEASURED"
    assert acc["CONSERVATIVE_ROUTER"]["restart_total_simulations"] == 2 * (4096 + 8192 + 32768 + 65536)


def test_mcts_resume_is_not_supported_by_current_code():
    audit = mcts_resume_audit()
    assert audit["MCTS_RESUME_CURRENTLY_SUPPORTED"] == "NO"
    assert audit["search_many_rebuilds_root_each_call"] and not audit["accepts_existing_tree"]
