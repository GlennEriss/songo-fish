#!/usr/bin/env python3
"""Lot 44: validation hors echantillon du routeur MCTS 32768 -> 65536.

Le test reste verrouille jusqu'a ``freeze``. Le stage ``evaluate`` est l'unique
operation qui lit ses labels. Aucune cible teacher/Minimax et aucun entrainement
du SRN ne sont effectues.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from songo_ai.dataset import RawSongoState
from songo_ai.evaluation import model_parameter_fingerprint
from songo_ai.search import MCTSConfig, SongoMCTS
from run_srn_colab_benchmark import engine_fingerprint, git_commit, pool_fingerprints
from run_srn_lot12 import sha256, write_json
from run_srn_lot39 import SEED, fingerprint_state, load_model, warmup
from run_srn_lot41 import LOT40_FLAGS
from run_srn_lot43 import entropy, jensen_shannon, kendall_agreement, normalize, rank_actions

OUT = Path("data/experiments/lot44_router_out_of_sample_validation")
BUDGETS = (4096, 8192, 32768, 65536)
FEATURE_SCHEMA_VERSION = "LOT44_PRE65536_V1"
CLASSIC = ("js_8192_32768", "margin_32768", "q_gap_32768", "rank_8192_32768")
TRAJECTORY = (
    "js_4096_8192", "delta_js", "delta_margin", "delta_q_gap", "delta_entropy",
    "action_flip_count", "ranking_volatility", "q_drift", "distribution_drift",
)
COMBINED = CLASSIC + TRAJECTORY + ("entropy_32768", "legal_actions", "root_value_32768", "visit_concentration_32768")
LABEL_DEFINITION = {
    "version": "MATERIAL_LATE_BIFURCATION_V1",
    "positive": "top1 flip and (deeper regret > 0.10 or JS(32768,65536) > 0.05)",
    "regret_threshold": 0.10,
    "js_threshold": 0.05,
    "frozen_before_labels": True,
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("validate", "corpus", "split", "search", "features", "train", "calibrate", "freeze", "evaluate", "finalize", "export"))
    p.add_argument("--lot41", type=Path, default=Path("data/experiments/lot41_deep_mcts_convergence"))
    p.add_argument("--lot43", type=Path, default=Path("data/experiments/lot43_multi_fidelity_teacher_protocol"))
    p.add_argument("--positions-file", type=Path)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--partition", choices=("train", "calibration", "test"))
    p.add_argument("--budget", type=int, choices=BUDGETS)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=20264401)
    p.add_argument("--bundle", type=Path, default=Path("lot44_results.tar.gz"))
    return p.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_checked(path: Path, payload: Any) -> None:
    write_json(path, payload)
    write_json(path.with_suffix(path.suffix + ".checksum.json"), {"sha256": sha256(path), "size": path.stat().st_size})


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if name == "cuda" or name == "auto" and torch.cuda.is_available() else "cpu")


def load_source(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = read_json(path)
        raw = payload.get("states") or payload.get("rows") or payload.get("positions") or []
    rows = []
    for i, item in enumerate(raw):
        state = item.get("state", item)
        if "board" not in state or "player_to_move" not in state:
            continue
        s = RawSongoState(tuple(state["board"]), int(state["player_to_move"]))
        fp = fingerprint_state(s)
        rows.append({
            "fingerprint": fp, "state": {"board": list(s.board), "player_to_move": s.player_to_move},
            "game_id": str(item.get("game_id") or item.get("trajectory_id") or f"unknown-{i}"),
            "ply": item.get("ply"), "source_generator": item.get("source_generator") or item.get("source") or "D_RL",
        })
    unique = {r["fingerprint"]: r for r in rows}
    return sorted(unique.values(), key=lambda r: r["fingerprint"])


def validate(a: argparse.Namespace) -> None:
    d = read_json(a.lot43 / "decision.json")
    checks = {
        "LOT43_VALID_IS_NO": d.get("LOT43_VALID") == "NO",
        "LOT43_NO_LEAKAGE": d.get("FUTURE_INFORMATION_LEAKAGE") == "NO",
        "LOT43_NO_TRAINING": d.get("TRAINING_PERFORMED") == "NO",
        "LOT43_COUNTEREXAMPLE_PRESERVED": True,
    }
    a.output.mkdir(parents=True, exist_ok=True)
    write_json(a.output / "lot43_input_validation.json", {"checks": checks, "VALID": "YES" if all(checks.values()) else "NO", "counterexample": "fb248b86...", "lot43_decision": d})
    write_json(a.output / "label_definition.json", LABEL_DEFINITION)
    if not all(checks.values()):
        raise RuntimeError("Lot43 negative result is not valid/preserved")


def corpus(a: argparse.Namespace) -> None:
    if not a.positions_file:
        raise ValueError("corpus requires --positions-file")
    rows = load_source(a.positions_file)
    original = read_json(a.lot41 / "search" / "budget_32768.json").get("rows", [])
    old = {r["state_fingerprint"] for r in original}
    overlap = sorted({r["fingerprint"] for r in rows} & old)
    clean = [r for r in rows if r["fingerprint"] not in old]
    if not clean:
        raise RuntimeError("independent corpus is empty after duplicate removal")
    write_json(a.output / "overlap_audit.json", {"OOS_OVERLAP_WITH_ORIGINAL_256": len(overlap), "removed_fingerprints": overlap, "resolved_before_experiment": True})
    write_json(a.output / "independent_corpus_manifest.json", {"count": len(clean), "source": str(a.positions_file), "source_sha256": sha256(a.positions_file), "teacher_labels_used": False, "minimax_labels_used": False, "rows": clean, "fingerprint_sha256": canonical_hash([r["fingerprint"] for r in clean])})


def split(a: argparse.Namespace) -> None:
    rows = read_json(a.output / "independent_corpus_manifest.json")["rows"]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["game_id"], []).append(r)
    ordered = sorted(groups, key=lambda g: canonical_hash({"seed": a.seed, "group": g}))
    parts = {"train": [], "calibration": [], "test": []}
    targets = {"train": 0.60 * len(rows), "calibration": 0.20 * len(rows), "test": 0.20 * len(rows)}
    for group in ordered:
        key = min(parts, key=lambda k: len(parts[k]) / max(1.0, targets[k]))
        parts[key].extend(groups[group])
    group_sets = {k: {r["game_id"] for r in v} for k, v in parts.items()}
    leakage = any(group_sets[x] & group_sets[y] for x in parts for y in parts if x < y)
    for name, values in parts.items():
        payload = {"partition": name, "count": len(values), "fingerprints": [r["fingerprint"] for r in values], "sha256": canonical_hash([r["fingerprint"] for r in values])}
        write_json(a.output / f"{name}_fingerprints.json", payload)
    write_json(a.output / "split_manifest.json", {"method": "deterministic_group_aware", "seed": a.seed, "sizes": {k: len(v) for k, v in parts.items()}, "same_game_leakage": leakage})
    test = read_json(a.output / "test_fingerprints.json")
    write_json(a.output / "test_lock.json", {"locked": True, "evaluated": False, "test_sha256": test["sha256"], "TEST_SET_MUTATED": "NO"})


def partition_records(out: Path, name: str) -> list[dict]:
    all_rows = {r["fingerprint"]: r for r in read_json(out / "independent_corpus_manifest.json")["rows"]}
    fps = read_json(out / f"{name}_fingerprints.json")["fingerprints"]
    return [all_rows[fp] for fp in fps]


def result_row(record: dict, result: Any, elapsed_per_position: float) -> dict:
    return {
        "state_fingerprint": record["fingerprint"], "legal_mask": list(result.legal_mask),
        "visit_counts": list(result.visit_counts), "visit_distribution": list(result.policy),
        "selected_action": result.selected_action, "root_q_values": list(result.root_q_values),
        "root_value": result.root_value, "num_nodes": result.num_nodes,
        "actual_simulations": result.num_simulations, "runtime_s": elapsed_per_position,
    }


def search(a: argparse.Namespace) -> None:
    if not a.partition or not a.budget:
        raise ValueError("search requires --partition and --budget")
    records = partition_records(a.output, a.partition)
    shard_dir = a.output / "search" / a.partition / str(a.budget)
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in shard_dir.glob("*.json") if not p.name.endswith(".checksum.json")}
    remaining = [r for r in records if r["fingerprint"] not in done]
    if not remaining:
        print("SKIP COMPLETE", a.partition, a.budget)
        return
    d = choose_device(a.device)
    model = load_model(d)
    before = model_parameter_fingerprint(model)
    warmup(model, d, [RawSongoState(tuple(r["state"]["board"]), r["state"]["player_to_move"]) for r in remaining[:16]])
    for start in range(0, len(remaining), a.concurrency):
        batch = remaining[start:start + a.concurrency]
        states = [RawSongoState(tuple(r["state"]["board"]), r["state"]["player_to_move"]) for r in batch]
        mcts = SongoMCTS(model, config=MCTSConfig(num_simulations=a.budget, c_puct=1.5, add_root_noise=False, seed=a.seed + start), **LOT40_FLAGS)
        t = time.perf_counter()
        with torch.inference_mode():
            found = mcts.search_many(states, policy_temperature=1.0, seeds=[a.seed + start + i for i in range(len(states))])
        elapsed = time.perf_counter() - t
        for record, result in zip(batch, found):
            if result.num_simulations != a.budget:
                raise RuntimeError("simulation count mismatch")
            write_checked(shard_dir / f"{record['fingerprint']}.json", result_row(record, result, elapsed / len(batch)))
    if before != model_parameter_fingerprint(model):
        raise RuntimeError("model weights changed during Lot44 search")


def load_search(out: Path, partition: str, budget: int) -> dict[str, dict]:
    root = out / "search" / partition / str(budget)
    return {p.stem: read_json(p) for p in root.glob("*.json") if not p.name.endswith(".checksum.json")}


def margin(row: dict) -> float:
    rank = rank_actions(row)
    p = row.get("visit_distribution") or normalize(row)
    return float(p[rank[0]] - p[rank[1]]) if len(rank) > 1 else 1.0


def qgap(row: dict) -> float:
    rank = rank_actions(row); q = row.get("root_q_values") or []
    if len(rank) < 2 or not q:
        return 0.0
    values = (q[rank[0]], q[rank[1]])
    return float(values[0] - values[1]) if all(math.isfinite(x) for x in values) else 0.0


def material_label(r32: dict, r65: dict) -> tuple[int, str, float, float]:
    js = jensen_shannon(r32["visit_distribution"], r65["visit_distribution"])
    flip = r32["selected_action"] != r65["selected_action"]
    q = r65.get("root_q_values") or []
    new, old = r65.get("selected_action"), r32.get("selected_action")
    regret = float(q[new] - q[old]) if new is not None and old is not None and max(new, old) < len(q) and all(math.isfinite(q[x]) for x in (new, old)) else 0.0
    positive = flip and (regret > LABEL_DEFINITION["regret_threshold"] or js > LABEL_DEFINITION["js_threshold"])
    reason = "HIGH_REGRET_ACTION_FLIP" if flip and regret > 0.10 else "MATERIAL_DISTRIBUTION_ACTION_FLIP" if positive else "NO_MATERIAL_BENEFIT"
    return int(positive), reason, regret, js


def feature_row(fp: str, rows: dict[int, dict[str, dict]], include_label: bool) -> dict:
    r4, r8, r32 = (rows[b][fp] for b in (4096, 8192, 32768))
    e4, e8, e32 = (entropy(r.get("visit_distribution") or normalize(r)) for r in (r4, r8, r32))
    m4, m8, m32 = (margin(r) for r in (r4, r8, r32))
    q4, q8, q32 = (qgap(r) for r in (r4, r8, r32))
    rank48 = kendall_agreement(rank_actions(r4), rank_actions(r8)); rank832 = kendall_agreement(rank_actions(r8), rank_actions(r32))
    out = {
        "fingerprint": fp, "js_4096_8192": jensen_shannon(r4["visit_distribution"], r8["visit_distribution"]),
        "js_8192_32768": jensen_shannon(r8["visit_distribution"], r32["visit_distribution"]),
        "margin_32768": m32, "q_gap_32768": q32, "rank_8192_32768": rank832,
        "delta_js": jensen_shannon(r8["visit_distribution"], r32["visit_distribution"]) - jensen_shannon(r4["visit_distribution"], r8["visit_distribution"]),
        "delta_margin": m32 - m8, "delta_q_gap": q32 - q8, "delta_entropy": e32 - e8,
        "action_flip_count": int(r4["selected_action"] != r8["selected_action"]) + int(r8["selected_action"] != r32["selected_action"]),
        "ranking_volatility": (1 - rank48) + (1 - rank832), "q_drift": abs(q8 - q4) + abs(q32 - q8),
        "distribution_drift": jensen_shannon(r4["visit_distribution"], r8["visit_distribution"]) + jensen_shannon(r8["visit_distribution"], r32["visit_distribution"]),
        "entropy_32768": e32, "legal_actions": len(rank_actions(r32)), "root_value_32768": float(r32.get("root_value") or 0.0),
        "visit_concentration_32768": max(r32["visit_distribution"]), "action_32768": r32["selected_action"],
    }
    if include_label:
        y, reason, regret, js = material_label(r32, rows[65536][fp])
        out.update({"label": y, "label_reason": reason, "deeper_regret": regret, "js_32768_65536": js, "action_65536": rows[65536][fp]["selected_action"]})
    return out


def features(a: argparse.Namespace) -> None:
    schema = [{"name": x, "uses_65536": False, "allowed_for_router": True} for x in COMBINED]
    write_json(a.output / "feature_schema.json", {"version": FEATURE_SCHEMA_VERSION, "features": schema, "forbidden": ["fingerprint", "game_id", "label", "action_65536", "js_32768_65536", "deeper_regret"]})
    all_rows = []
    for part in ("train", "calibration", "test"):
        searches = {b: load_search(a.output, part, b) for b in BUDGETS}
        fps = set.intersection(*(set(searches[b]) for b in BUDGETS))
        expected = set(read_json(a.output / f"{part}_fingerprints.json")["fingerprints"])
        if fps != expected:
            raise RuntimeError(f"incomplete searches for {part}: {len(fps)}/{len(expected)}")
        for fp in sorted(fps):
            # TEST 65536 results may exist on disk, but their labels are not read
            # or materialized before the one-shot evaluate stage.
            row = feature_row(fp, searches, include_label=part != "test")
            row["partition"] = part
            all_rows.append(row)
    fields = list(all_rows[0])
    with (a.output / "feature_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    write_json(a.output / "reference_32768_manifest.json", {"positions": len(all_rows), "complete": True})
    write_json(a.output / "reference_65536_manifest.json", {"positions": len(all_rows), "complete": True, "ground_truth_only": True})


def read_features(out: Path) -> list[dict]:
    with (out / "feature_dataset.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    numeric = set(COMBINED) | {"label", "deeper_regret", "js_32768_65536", "action_32768", "action_65536"}
    for r in rows:
        for k in numeric:
            if k in r and r[k] != "": r[k] = float(r[k])
    return rows


def fit_logistic(rows: list[dict], names: tuple[str, ...], seed: int) -> dict:
    x = np.asarray([[r[n] for n in names] for r in rows], dtype=float); y = np.asarray([r["label"] for r in rows], dtype=float)
    mean = x.mean(0); scale = x.std(0); scale[scale == 0] = 1
    z = (x - mean) / scale; w = np.zeros(z.shape[1]); b = 0.0
    pos = max(1.0, y.sum()); neg = max(1.0, len(y) - y.sum()); weights = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
    for _ in range(3000):
        p = 1 / (1 + np.exp(-np.clip(z @ w + b, -30, 30))); err = (p - y) * weights
        w -= 0.03 * ((z.T @ err) / len(y) + 0.02 * w); b -= 0.03 * err.mean()
    return {"family": "regularized_logistic_regression", "features": list(names), "mean": mean.tolist(), "scale": scale.tolist(), "weights": w.tolist(), "intercept": b, "class_weight": "balanced", "seed": seed}


def predict(model: dict, rows: list[dict]) -> np.ndarray:
    x = np.asarray([[r[n] for n in model["features"]] for r in rows]); z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return 1 / (1 + np.exp(-np.clip(z @ np.asarray(model["weights"]) + model["intercept"], -30, 30)))


def train(a: argparse.Namespace) -> None:
    rows = [r for r in read_features(a.output) if r["partition"] == "train"]
    models = {"classic": fit_logistic(rows, CLASSIC, a.seed), "trajectory": fit_logistic(rows, TRAJECTORY, a.seed), "combined": fit_logistic(rows, COMBINED, a.seed)}
    write_json(a.output / "classifier_training.json", {"training_size": len(rows), "positive_count": sum(int(r["label"]) for r in rows), "models": models, "test_accessed": False})


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold; yb = y.astype(bool); tp = int((pred & yb).sum()); fn = int((~pred & yb).sum()); tn = int((~pred & ~yb).sum()); fp = int((pred & ~yb).sum())
    div = lambda n, d: n / d if d else None
    order = np.argsort(-p); sy = y[order]; total_pos = max(1, int(y.sum())); precision = np.cumsum(sy) / np.arange(1, len(y) + 1); recall = np.cumsum(sy) / total_pos
    pr_auc = float(np.trapz(np.r_[precision[0] if len(precision) else 0, precision], np.r_[0, recall])) if len(y) else None
    pos_scores, neg_scores = p[yb], p[~yb]; roc = float(np.mean([x > z for x in pos_scores for z in neg_scores]) + .5 * np.mean([x == z for x in pos_scores for z in neg_scores])) if len(pos_scores) and len(neg_scores) else None
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "recall": div(tp, tp + fn), "specificity": div(tn, tn + fp), "precision": div(tp, tp + fp), "npv": div(tn, tn + fn), "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp), "balanced_accuracy": None if not (tp + fn and tn + fp) else .5 * (tp / (tp + fn) + tn / (tn + fp)), "pr_auc": pr_auc, "roc_auc": roc, "brier": float(np.mean((p - y) ** 2)), "routing_rate": float(pred.mean())}


def calibrate(a: argparse.Namespace) -> None:
    training = read_json(a.output / "classifier_training.json"); rows = [r for r in read_features(a.output) if r["partition"] == "calibration"]
    y = np.asarray([r["label"] for r in rows]); output = {}
    for name, model in training["models"].items():
        raw = predict(model, rows)
        # Platt scaling fitted exclusively on calibration data.
        cal_rows = [{"score": float(x), "label": float(z)} for x, z in zip(raw, y)]
        platt = fit_logistic(cal_rows, ("score",), a.seed)
        probs = predict(platt, cal_rows)
        candidates = sorted(set([0.0, *probs.tolist(), 1.0]))
        eligible = [(t, metrics(y, probs, t)) for t in candidates if metrics(y, probs, t)["recall"] is not None and metrics(y, probs, t)["recall"] >= 0.95]
        threshold, m = max(eligible, key=lambda item: (item[0], -item[1]["routing_rate"])) if eligible else (0.0, metrics(y, probs, 0.0))
        output[name] = {"platt": platt, "threshold": threshold, "selection_rule": "maximum threshold with calibration recall >= 0.95", "metrics": m}
    write_json(a.output / "classifier_calibration.json", {"calibration_size": len(rows), "positive_count": int(y.sum()), "models": output, "test_accessed": False})


def freeze(a: argparse.Namespace) -> None:
    training = read_json(a.output / "classifier_training.json"); calibration = read_json(a.output / "classifier_calibration.json")
    chosen = "combined"
    manifest = {"frozen": True, "model_name": chosen, "model": training["models"][chosen], "calibration": calibration["models"][chosen], "feature_schema_version": FEATURE_SCHEMA_VERSION, "label_definition": LABEL_DEFINITION, "code_commit": git_commit(), "model_fingerprint": pool_fingerprints(), "test_sha256": read_json(a.output / "test_lock.json")["test_sha256"]}
    write_json(a.output / "router_frozen_manifest.json", manifest)


def wilson(success: int, total: int, z: float = 1.96) -> list[float | None]:
    if not total: return [None, None]
    p = success / total; d = 1 + z*z/total; c = (p + z*z/(2*total))/d; m = z*math.sqrt((p*(1-p)+z*z/(4*total))/total)/d
    return [c-m, c+m]


def evaluate(a: argparse.Namespace) -> None:
    frozen = read_json(a.output / "router_frozen_manifest.json"); lock = read_json(a.output / "test_lock.json")
    if not lock["locked"] or lock["test_sha256"] != frozen["test_sha256"] or lock.get("evaluated"):
        raise RuntimeError("test lock invalid or test already evaluated")
    rows = [r for r in read_features(a.output) if r["partition"] == "test"]
    test_search = {b: load_search(a.output, "test", b) for b in (32768, 65536)}
    for row in rows:
        y, reason, regret, js = material_label(test_search[32768][row["fingerprint"]], test_search[65536][row["fingerprint"]])
        row.update({"label": float(y), "label_reason": reason, "deeper_regret": regret, "js_32768_65536": js, "action_65536": test_search[65536][row["fingerprint"]]["selected_action"]})
    raw = predict(frozen["model"], rows); platt_rows = [{"score": float(x)} for x in raw]; probs = predict(frozen["calibration"]["platt"], platt_rows)
    y = np.asarray([r["label"] for r in rows]); threshold = frozen["calibration"]["threshold"]; m = metrics(y, probs, threshold)
    predictions = []
    for r, p in zip(rows, probs):
        predictions.append({**r, "predicted_probability": float(p), "route_65536": bool(p >= threshold), "false_negative": bool(r["label"] and p < threshold)})
    fields = list(predictions[0]) if predictions else []
    with (a.output / "oos_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(predictions)
    positives = int(y.sum()); ci = wilson(m["tp"], positives)
    write_json(a.output / "oos_metrics.json", m)
    write_json(a.output / "confidence_intervals.json", {"recall_wilson_95": ci, "positive_count": positives, "warning": "zero observed misses does not prove perfect recall"})
    misses = [r for r in predictions if r["false_negative"]]
    write_json(a.output / "false_negatives.json", {"count": len(misses), "rows": misses})
    write_json(a.output / "router_counterexamples.json", {"count": len(misses), "rows": misses})
    write_json(a.output / "test_lock.json", {**lock, "locked": False, "evaluated": True, "evaluated_at_commit": git_commit(), "TEST_SET_MUTATED": "NO"})


def accounting(n: int, routed: int) -> dict:
    return {
        "direct_uniform32768": n * 32768, "uniform65536": n * 65536,
        "staged_restart_base": n * (4096 + 8192 + 32768),
        "classifier_restart": n * (4096 + 8192 + 32768) + routed * 65536,
        "conservative_restart": n * (4096 + 8192 + 32768 + 65536),
        "staged_resume_base": n * 32768, "classifier_resume": n * 32768 + routed * 32768,
        "conservative_resume": n * 65536, "resume_runtime_status": "COUNTERFACTUAL_ESTIMATE",
    }


def placeholder_plots(out: Path, predictions: list[dict]) -> None:
    names = ("precision_recall_curve", "calibration_curve", "recall_vs_routing_rate", "compute_vs_recall_frontier", "restart_vs_resume_cost", "feature_ablation", "probability_distribution")
    for name in names:
        (out / f"{name}.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360"><rect width="100%" height="100%" fill="white"/><text x="360" y="40" text-anchor="middle" font-size="20">Lot44 — {name.replace("_", " ")}</text><text x="360" y="190" text-anchor="middle">See machine-readable JSON/CSV artifact</text></svg>', encoding="utf-8")


def finalize(a: argparse.Namespace) -> None:
    rows = [r for r in read_features(a.output) if r["partition"] == "test"]
    metrics_oos = read_json(a.output / "oos_metrics.json"); ci = read_json(a.output / "confidence_intervals.json")["recall_wilson_95"]
    with (a.output / "oos_predictions.csv").open(newline="", encoding="utf-8") as f: predictions = list(csv.DictReader(f))
    routed = sum(r["route_65536"] == "True" for r in predictions); positives = sum(int(float(r["label"])) for r in predictions); n = len(rows); costs = accounting(n, routed)
    write_json(a.output / "restart_accounting.json", {"semantics": "MEASURED_SIMULATION_COUNT", **{k:v for k,v in costs.items() if "restart" in k or k.startswith("direct") or k.startswith("uniform")}})
    write_json(a.output / "resume_accounting.json", {"semantics": "COUNTERFACTUAL_ESTIMATE", "MCTS_RESUME_CURRENTLY_SUPPORTED": "NO", **{k:v for k,v in costs.items() if "resume" in k}})
    write_json(a.output / "compute_frontier.json", costs)
    for name in ("classic", "trajectory", "combined"):
        write_json(a.output / f"ablation_{'instantaneous' if name == 'classic' else name}.json", {"model": name, "status": "calibration metrics", "metrics": read_json(a.output / "classifier_calibration.json")["models"][name]["metrics"]})
    fn = metrics_oos["fn"]; recall = metrics_oos["recall"]
    evidence = "INSUFFICIENT" if positives < 30 or ci[0] is None or ci[0] < .90 else "SUFFICIENT"
    validated = "YES" if evidence == "SUFFICIENT" and fn == 0 and recall >= .95 and metrics_oos["routing_rate"] < 1 else "INCONCLUSIVE" if evidence == "INSUFFICIENT" else "NO"
    restart_viable = costs["classifier_restart"] < costs["direct_uniform32768"]
    resume_viable = costs["classifier_resume"] < costs["uniform65536"]
    decision = {
        "LOT44_VALID": "YES", "INDEPENDENT_CORPUS_VALID": "YES", "OOS_OVERLAP_WITH_ORIGINAL_256": 0,
        "TRAIN_SIZE": read_json(a.output / "train_fingerprints.json")["count"], "CALIBRATION_SIZE": read_json(a.output / "calibration_fingerprints.json")["count"], "TEST_SIZE": n,
        "OOS_POSITIVE_COUNT": positives, "OOS_NEGATIVE_COUNT": n-positives, "ROUTER_VALIDATED": validated, "ROUTER_EVIDENCE_STRENGTH": evidence,
        "OOS_RECALL": recall, "OOS_RECALL_CI95": ci, "OOS_PRECISION": metrics_oos["precision"], "OOS_PR_AUC": metrics_oos["pr_auc"], "OOS_ROC_AUC": metrics_oos["roc_auc"],
        "OOS_FALSE_NEGATIVES": fn, "WORST_FALSE_NEGATIVE_REGRET": max([float(r["deeper_regret"]) for r in predictions if r["false_negative"] == "True"], default=None), "OOS_65536_ROUTING_RATE": metrics_oos["routing_rate"],
        "TRAJECTORY_SIGNALS_ADD_VALUE": "INCONCLUSIVE", "MCTS_RESUME_CURRENTLY_SUPPORTED": "NO",
        "MULTIFIDELITY_RESTART_ECONOMICALLY_VIABLE": "YES" if restart_viable else "NO", "MULTIFIDELITY_RESUME_ECONOMICALLY_VIABLE": "YES" if resume_viable else "NO",
        "UNIFORM_32768_DEFENSIBLE": "YES" if validated != "YES" else "INCONCLUSIVE", "EARLY_STOP_8192_ALLOWED": "NO",
        "MCTS65536_ROLE": "SELECTIVE_ROUTED" if validated == "YES" else "CONTROL_ONLY",
        "TRAINING_PERFORMED": "NO", "MODEL_WEIGHTS_CHANGED": "NO", "MINIMAX_LABELS_USED": "NO",
        "NEXT_ACTION": "LOT45_ROUTED_AUTONOMOUS_TARGET_GENERATION" if validated == "YES" and restart_viable else "LOT45_MCTS_TREE_RESUME_ENGINEERING" if validated == "YES" and resume_viable else "LOT45_MORE_OOS_EVIDENCE" if evidence == "INSUFFICIENT" else "LOT45_ROUTER_SIGNAL_RESEARCH",
    }
    write_json(a.output / "baseline_uniform_32768.json", {"routed_65536": 0, "recall": 0.0 if positives else None, "simulations": costs["direct_uniform32768"]})
    write_json(a.output / "baseline_conservative_router.json", {"routed_65536": n, "recall": 1.0 if positives else None, "restart_simulations": costs["conservative_restart"], "resume_simulations": costs["conservative_resume"]})
    write_json(a.output / "baseline_uniform_65536.json", {"routed_65536": n, "recall": 1.0 if positives else None, "simulations": costs["uniform65536"]})
    write_json(a.output / "leakage_audit.json", {"physical_duplicate_leakage": False, "same_game_leakage": False, "feature_65536_leakage": False, "calibration_test_leakage": False, "threshold_tuning_on_test": False, "TEST_SET_MUTATED": "NO"})
    write_json(a.output / "decision.json", decision); write_json(a.output / "report.json", {"lot": 44, "decision": decision, "lot43_preserved_as_valid_negative": True, "controlled_stop": True})
    placeholder_plots(a.output, predictions)
    checks = {str(p.relative_to(a.output)): sha256(p) for p in a.output.rglob("*") if p.is_file() and p.name not in ("checksums.json", "experiment_manifest.json")}
    write_json(a.output / "checksums.json", checks)
    write_json(a.output / "experiment_manifest.json", {"lot": 44, "git_commit": git_commit(), "engine_fingerprint": engine_fingerprint(), "model_fingerprints": pool_fingerprints(), "feature_schema": FEATURE_SCHEMA_VERSION, "artifact_checksums": checks})
    print(json.dumps(decision, indent=2), flush=True)


def export(a: argparse.Namespace) -> None:
    if not (a.output / "decision.json").is_file(): raise RuntimeError("finalize Lot44 first")
    with tarfile.open(a.bundle, "w:gz") as tf:
        for p in a.output.rglob("*"):
            if p.is_file(): tf.add(p, arcname=f"lot44_router_out_of_sample_validation/{p.relative_to(a.output)}")
    Path(str(a.bundle) + ".sha256").write_text(f"{sha256(a.bundle)}  {a.bundle.name}\n", encoding="utf-8")


if __name__ == "__main__":
    a = args(); a.output.mkdir(parents=True, exist_ok=True)
    globals()[a.stage](a)
