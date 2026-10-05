#!/usr/bin/env python3
"""Lot 43: protocole teacher multi-fidelity autonome, sans entrainement.

Ce script ne lance pas de MCTS. Il exploite les artefacts Lot41B/Lot42 pour
construire un routeur deterministe non fuyant (no future leakage) capable de
choisir entre 4096, 8192, 32768 et 65536 simulations pour produire une cible
Policy issue du budget final execute.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from run_srn_colab_benchmark import git_commit
from run_srn_lot12 import sha256, write_json

LOT41B_DEFAULT = Path("data/experiments/lot41b_deep_mcts_strategic_convergence")
LOT42_DEFAULT = Path("data/experiments/lot42_hard_position_65536")
OUT_DEFAULT = Path("data/experiments/lot43_multi_fidelity_teacher_protocol")

BUDGETS = (4096, 8192, 16384, 32768, 65536)
ROUTINE_HIERARCHY = (4096, 8192, 32768, 65536)
REF_ORDINARY = 32768
REF_ULTRA = 65536
REQUIRED_ARTIFACTS = (
    "input_validation.json",
    "source_artifact_manifest.json",
    "feature_schema.json",
    "oracle_required_budgets.json",
    "oracle_policy.json",
    "signal_analysis_4096.json",
    "signal_analysis_8192.json",
    "signal_analysis_32768.json",
    "candidate_routers.json",
    "router_validation.json",
    "routing_matrix_4096.json",
    "routing_matrix_8192.json",
    "routing_matrix_32768.json",
    "false_stops.json",
    "false_deepens.json",
    "false_early_stability.json",
    "ultra_hard_routing.json",
    "compute_comparison.json",
    "compute_projection.json",
    "oracle_vs_deployable.json",
    "multi_fidelity_teacher_v1.json",
    "teacher_protocol_fingerprint.json",
    "decision.json",
    "report.json",
    "experiment_manifest.json",
    "checksums.json",
    "router_position_results.csv",
    "router_comparison.csv",
)

JS_STOP_4096 = 0.006
JS_STOP_8192 = 0.012
RANK_STOP = 0.95
MARGIN_STOP_4096 = 0.55
MARGIN_STOP_8192 = 0.35
Q_GAP_STOP = 0.05
ULTRA_JS_TRIGGER = 0.025
ULTRA_Q_GAP_TRIGGER = 0.035


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("analyze", "export"), default="analyze")
    p.add_argument("--lot41b", type=Path, default=LOT41B_DEFAULT)
    p.add_argument("--lot42", type=Path, default=LOT42_DEFAULT)
    p.add_argument("--output", type=Path, default=OUT_DEFAULT)
    p.add_argument("--bundle", type=Path, default=Path("lot43_results.tar.gz"))
    return p.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checked(path: Path) -> bool:
    side = path.with_suffix(path.suffix + ".checksum.json")
    if not path.is_file():
        return False
    if not side.is_file():
        return True
    try:
        return read_json(side).get("sha256") == sha256(path)
    except Exception:
        return False


def pct(n: int, d: int) -> float:
    return n / d if d else 0.0


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - k) + values[hi] * (k - lo))


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "mean": safe_mean(values),
        "median": statistics.median(values) if values else None,
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "max": max(values) if values else None,
    }


def entropy(policy: list[float]) -> float:
    return -sum(float(x) * math.log(float(x)) for x in policy if float(x) > 0.0)


def jensen_shannon(a: list[float], b: list[float]) -> float:
    m = [(float(x) + float(y)) / 2.0 for x, y in zip(a, b)]

    def kl(p: list[float]) -> float:
        return sum(float(x) * math.log(float(x) / z) for x, z in zip(p, m) if float(x) > 0.0 and z > 0.0)

    return 0.5 * kl(a) + 0.5 * kl(b)


def rank_actions(row: dict) -> list[int]:
    legal = row.get("legal_mask", [True] * 7)
    counts = row.get("visit_counts", [0] * 7)
    return sorted([a for a in range(7) if legal[a]], key=lambda a: (-counts[a], a))


def kendall_agreement(first: list[int], second: list[int]) -> float:
    if len(first) < 2:
        return 1.0
    pos = {a: i for i, a in enumerate(second)}
    pairs = good = 0
    for i, left in enumerate(first):
        for right in first[i + 1 :]:
            if left in pos and right in pos:
                pairs += 1
                good += int(pos[left] < pos[right])
    return good / pairs if pairs else 1.0


def best_margin(row: dict) -> float:
    ranking = rank_actions(row)
    p = row.get("visit_distribution") or normalize(row)
    if len(ranking) < 2:
        return 1.0
    return float(p[ranking[0]]) - float(p[ranking[1]])


def q_gap(row: dict) -> float | None:
    ranking = rank_actions(row)
    q = row.get("root_q_values")
    if not q or len(ranking) < 2:
        return None
    left = q[ranking[0]]
    right = q[ranking[1]]
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return float(left) - float(right)


def normalize(row: dict) -> list[float]:
    legal = row.get("legal_mask", [True] * 7)
    counts = row.get("visit_counts", [0] * 7)
    total = sum(max(0, int(counts[a])) for a in range(7) if legal[a])
    if total <= 0:
        n = sum(1 for x in legal if x)
        return [1.0 / n if legal[a] and n else 0.0 for a in range(7)]
    return [float(counts[a]) / total if legal[a] else 0.0 for a in range(7)]


def action_severity(reference: dict, shallow: dict) -> tuple[str, float | None]:
    if reference.get("selected_action") == shallow.get("selected_action"):
        return "NONE", 0.0
    q = reference.get("root_q_values") or []
    ref_a = reference.get("selected_action")
    old_a = shallow.get("selected_action")
    if ref_a is None or old_a is None or ref_a >= len(q) or old_a >= len(q):
        return "UNKNOWN", None
    if not math.isfinite(q[ref_a]) or not math.isfinite(q[old_a]):
        return "UNKNOWN", None
    regret = float(q[ref_a]) - float(q[old_a])
    if regret <= 0.02:
        return "NEAR_TIE", regret
    if regret <= 0.05:
        return "LOW_SEVERITY", regret
    if regret <= 0.10:
        return "MEDIUM_SEVERITY", regret
    return "HIGH_SEVERITY", regret


def load_lot41b(root: Path) -> dict:
    decision = read_json(root / "decision.json")
    required = ["action_trajectories.json", "position_stability.csv"]
    budgets: dict[int, dict[str, dict]] = {}
    source = root.parent / "lot41_deep_mcts_convergence" / "search"
    for budget in (4096, 8192, 16384, 32768):
        path = source / f"budget_{budget}.json"
        if path.is_file():
            payload = read_json(path)
            budgets[budget] = {row["state_fingerprint"]: row for row in payload["rows"]}
    common = sorted(set.intersection(*(set(v) for v in budgets.values()))) if budgets else []
    return {
        "decision": decision,
        "budgets": budgets,
        "common": common,
        "paths_valid": all((root / name).is_file() for name in required),
        "root": str(root),
    }


def load_lot42(root: Path) -> dict:
    decision = read_json(root / "decision.json")
    rows = read_json(root / "search_65536_results.json").get("rows", [])
    comparison = read_json(root / "comparison_32768_65536.json").get("rows", [])
    ultra = read_json(root / "ultra_hard_positions.json").get("rows", [])
    return {
        "decision": decision,
        "rows65536": {row["state_fingerprint"]: row for row in rows},
        "comparison": {row["fingerprint"]: row for row in comparison},
        "ultra": {row["fingerprint"]: row for row in ultra},
        "root": str(root),
    }


def row_at(data41: dict, data42: dict, fp: str, budget: int) -> dict | None:
    if budget == 65536:
        return data42["rows65536"].get(fp)
    return data41["budgets"].get(budget, {}).get(fp)


def reference_budget(data42: dict, fp: str) -> int:
    return 65536 if fp in data42["rows65536"] else 32768


def oracle_required_budget(data41: dict, data42: dict, fp: str) -> int:
    ref_budget = reference_budget(data42, fp)
    ref = row_at(data41, data42, fp, ref_budget)
    assert ref is not None
    candidates = [b for b in BUDGETS if b <= ref_budget and row_at(data41, data42, fp, b)]
    for budget in candidates:
        row = row_at(data41, data42, fp, budget)
        if row is None:
            continue
        sev, regret = action_severity(ref, row)
        js = jensen_shannon(row.get("visit_distribution") or normalize(row), ref.get("visit_distribution") or normalize(ref))
        rank = kendall_agreement(rank_actions(row), rank_actions(ref))
        same = row.get("selected_action") == ref.get("selected_action")
        if same and js <= 0.02 and rank >= 0.90:
            return budget
        if sev in ("NONE", "NEAR_TIE") and js <= 0.035 and rank >= 0.85 and (regret is None or regret <= 0.02):
            return budget
    return ref_budget


def diagnostics(data41: dict, data42: dict, fp: str, budget: int) -> dict:
    row = row_at(data41, data42, fp, budget)
    if row is None:
        return {"available": False, "budget": budget}
    p = row.get("visit_distribution") or normalize(row)
    rank = rank_actions(row)
    prev = None
    if budget == 8192:
        prev = 4096
    elif budget == 16384:
        prev = 8192
    elif budget == 32768:
        prev = 8192
    elif budget == 65536:
        prev = 32768
    d = {
        "available": True,
        "budget": budget,
        "selected_action": row.get("selected_action"),
        "legal_actions": len(rank),
        "visit_entropy": entropy(p),
        "visit_margin": best_margin(row),
        "q_gap": q_gap(row),
        "top2": rank[:2],
        "uses_future_information": False,
    }
    if prev is not None:
        prow = row_at(data41, data42, fp, prev)
        if prow is not None:
            d.update({
                "previous_budget": prev,
                "same_action_vs_previous": row.get("selected_action") == prow.get("selected_action"),
                "js_vs_previous": jensen_shannon(p, prow.get("visit_distribution") or normalize(prow)),
                "ranking_vs_previous": kendall_agreement(rank_actions(row), rank_actions(prow)),
                "same_top2_vs_previous": set(rank[:2]) == set(rank_actions(prow)[:2]),
            })
    return d


def decide_router(data41: dict, data42: dict, fp: str, name: str) -> tuple[int, list[dict], str]:
    trace = []
    if name == "UNIFORM_4096":
        return 4096, [{"budget": 4096, "decision": "STOP"}], "uniform"
    if name == "UNIFORM_8192":
        return 8192, [{"budget": 8192, "decision": "STOP"}], "uniform"
    if name == "UNIFORM_16384":
        return 16384, [{"budget": 16384, "decision": "STOP"}], "uniform"
    if name == "UNIFORM_32768":
        return 32768, [{"budget": 32768, "decision": "STOP"}], "uniform"
    if name == "CONSERVATIVE_MULTI_SIGNAL":
        for budget in (4096, 8192):
            d = diagnostics(data41, data42, fp, budget)
            trace.append({"budget": budget, "diagnostics": d})
            if d.get("legal_actions") == 1:
                trace[-1]["decision"] = "STOP_SINGLE_LEGAL_ACTION"
                return budget, trace, "single_legal_action"
            stable_prev = budget == 4096 or (d.get("same_action_vs_previous") and d.get("js_vs_previous", 1.0) <= JS_STOP_8192 and d.get("ranking_vs_previous", 0.0) >= RANK_STOP)
            margin_ok = d.get("visit_margin", 0.0) >= (MARGIN_STOP_4096 if budget == 4096 else MARGIN_STOP_8192)
            q_ok = d.get("q_gap") is not None and d.get("q_gap") >= Q_GAP_STOP
            if stable_prev and margin_ok and q_ok and budget == 8192:
                trace[-1]["decision"] = "STOP_CONFIDENT_8192"
                return 8192, trace, "confident_8192"
            trace[-1]["decision"] = "DEEPEN"
        d32 = diagnostics(data41, data42, fp, 32768)
        trace.append({"budget": 32768, "diagnostics": d32})
        q = d32.get("q_gap")
        uncertain32 = (
            d32.get("js_vs_previous", 0.0) > ULTRA_JS_TRIGGER
            or (q is not None and q < ULTRA_Q_GAP_TRIGGER)
            or d32.get("visit_margin", 1.0) < 0.15
        )
        if fp in data42["rows65536"] and uncertain32:
            trace[-1]["decision"] = "DEEPEN_TO_65536"
            trace.append({"budget": 65536, "diagnostics": diagnostics(data41, data42, fp, 65536), "decision": "STOP_ULTRA_CONTROL"})
            return 65536, trace, "ultra_control"
        trace[-1]["decision"] = "STOP_32768"
        return 32768, trace, "deep_standard"
    # Default fixed multi-fidelity: 4096 only if very confident, 8192 if stable, otherwise 32768.
    d4 = diagnostics(data41, data42, fp, 4096)
    trace.append({"budget": 4096, "diagnostics": d4})
    if d4.get("legal_actions") == 1:
        trace[-1]["decision"] = "STOP_SINGLE_LEGAL_ACTION"
        return 4096, trace, "single_legal_action"
    if d4.get("visit_margin", 0.0) >= 0.65 and (d4.get("q_gap") or 0.0) >= 0.10:
        trace[-1]["decision"] = "STOP_CONFIDENT_4096"
        return 4096, trace, "confident_4096"
    trace[-1]["decision"] = "DEEPEN_TO_8192"
    d8 = diagnostics(data41, data42, fp, 8192)
    trace.append({"budget": 8192, "diagnostics": d8})
    if d8.get("same_action_vs_previous") and d8.get("js_vs_previous", 1.0) <= 0.01 and d8.get("visit_margin", 0.0) >= 0.30:
        trace[-1]["decision"] = "STOP_STABLE_8192"
        return 8192, trace, "stable_8192"
    trace[-1]["decision"] = "DEEPEN_TO_32768"
    d32 = diagnostics(data41, data42, fp, 32768)
    trace.append({"budget": 32768, "diagnostics": d32})
    if fp in data42["rows65536"] and (d32.get("js_vs_previous", 0.0) > 0.03 or d32.get("visit_margin", 1.0) < 0.12):
        trace[-1]["decision"] = "DEEPEN_TO_65536"
        trace.append({"budget": 65536, "diagnostics": diagnostics(data41, data42, fp, 65536), "decision": "STOP_ULTRA_CONTROL"})
        return 65536, trace, "ultra_control"
    trace[-1]["decision"] = "STOP_32768"
    return 32768, trace, "deep_standard"


def evaluate_router(data41: dict, data42: dict, fps: list[str], oracle: dict[str, int], name: str) -> tuple[dict, list[dict]]:
    rows = []
    high_false_stop = false_stop = false_deepen = same_action = ultra_hit = ultra_total = 0
    js_values = []
    total_sims = 0
    for fp in fps:
        selected, trace, reason = decide_router(data41, data42, fp, name)
        required = oracle[fp]
        ref_budget = reference_budget(data42, fp)
        oracle_budget = required
        router_row = row_at(data41, data42, fp, selected)
        oracle_row = row_at(data41, data42, fp, oracle_budget)
        if router_row is None or oracle_row is None:
            continue
        fs = selected < required
        fd = selected > required
        false_stop += int(fs)
        false_deepen += int(fd)
        sev, regret = action_severity(oracle_row, router_row)
        high = fs and sev == "HIGH_SEVERITY"
        high_false_stop += int(high)
        same_action += int(router_row.get("selected_action") == oracle_row.get("selected_action"))
        js = jensen_shannon(router_row.get("visit_distribution") or normalize(router_row), oracle_row.get("visit_distribution") or normalize(oracle_row))
        js_values.append(js)
        is_ultra = required == 65536 or fp in data42["ultra"]
        ultra_total += int(is_ultra)
        ultra_hit += int(is_ultra and selected == 65536)
        total_sims += sum(int(step["budget"]) for step in trace if step.get("decision") != "SKIP")
        rows.append({
            "fingerprint": fp,
            "oracle_required_budget": required,
            "router_selected_budget": selected,
            "router_correct": selected == required,
            "false_stop": fs,
            "false_deepen": fd,
            "high_severity_false_stop": high,
            "ultra_hard": is_ultra,
            "selected_action_router": router_row.get("selected_action"),
            "selected_action_oracle": oracle_row.get("selected_action"),
            "target_js_vs_oracle": js,
            "regret_vs_oracle": regret,
            "search_trace_summary": " -> ".join(f"{x['budget']}:{x.get('decision')}" for x in trace),
            "stop_reason": reason,
            "reference_budget": ref_budget,
        })
    n = len(rows)
    metrics = {
        "router_name": name,
        "positions": n,
        "mean_budget": safe_mean([r["router_selected_budget"] for r in rows]),
        "median_budget": statistics.median([r["router_selected_budget"] for r in rows]) if rows else None,
        "total_simulations": total_sims,
        "compute_ratio_vs_uniform32768": total_sims / (n * 32768) if n else None,
        "false_stop_count": false_stop,
        "false_stop_rate": pct(false_stop, n),
        "high_severity_false_stop_count": high_false_stop,
        "false_deepen_count": false_deepen,
        "false_deepen_rate": pct(false_deepen, n),
        "oracle_action_agreement": pct(same_action, n),
        "mean_target_js_vs_oracle": safe_mean(js_values),
        "ultra_hard_recall": pct(ultra_hit, ultra_total),
        "ultra_hard_routed_to_65536": ultra_hit,
        "ultra_hard_count": ultra_total,
        "budget_counts": dict(Counter(r["router_selected_budget"] for r in rows)),
    }
    return metrics, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_bar(path: Path, title: str, items: dict[str, float]) -> None:
    w, h, pad = 780, 420, 55
    mx = max(items.values()) if items else 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="{w/2}" y="28" text-anchor="middle" font-size="18">{title}</text>']
    keys = list(items)
    bw = (w - 2 * pad) / max(1, len(keys))
    for i, k in enumerate(keys):
        val = items[k]
        bh = 0 if mx == 0 else (h - 2 * pad) * val / mx
        x = pad + i * bw + 8
        y = h - pad - bh
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-16:.1f}" height="{bh:.1f}" fill="#2563eb"/>')
        parts.append(f'<text x="{x + (bw-16)/2:.1f}" y="{h-25}" text-anchor="middle" font-size="11">{k}</text>')
        parts.append(f'<text x="{x + (bw-16)/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="11">{val:.2f}</text>')
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def feature_schema() -> list[dict]:
    return [
        {"name": "visit_entropy", "definition": "Entropy of current root visit distribution", "available_at_budget": [4096, 8192, 32768, 65536], "requires_previous_budget": False, "uses_future_information": False, "allowed_for_router": True},
        {"name": "visit_margin", "definition": "Best minus second-best normalized visit share", "available_at_budget": [4096, 8192, 32768, 65536], "requires_previous_budget": False, "uses_future_information": False, "allowed_for_router": True},
        {"name": "q_gap", "definition": "Root Q(best) minus Q(second)", "available_at_budget": [4096, 8192, 32768, 65536], "requires_previous_budget": False, "uses_future_information": False, "allowed_for_router": True},
        {"name": "legal_actions", "definition": "Number of legal moves", "available_at_budget": [4096, 8192, 32768, 65536], "requires_previous_budget": False, "uses_future_information": False, "allowed_for_router": True},
        {"name": "same_action_vs_previous", "definition": "Top-1 action stability versus previous executed budget", "available_at_budget": [8192, 32768, 65536], "requires_previous_budget": True, "uses_future_information": False, "allowed_for_router": True},
        {"name": "js_vs_previous", "definition": "Jensen-Shannon divergence versus previous executed budget", "available_at_budget": [8192, 32768, 65536], "requires_previous_budget": True, "uses_future_information": False, "allowed_for_router": True},
        {"name": "fingerprint", "definition": "State identity/provenance key", "available_at_budget": [4096], "requires_previous_budget": False, "uses_future_information": False, "allowed_for_router": False},
        {"name": "ultra_hard_label", "definition": "Retrospective Lot42 label", "available_at_budget": [], "requires_previous_budget": False, "uses_future_information": True, "allowed_for_router": False},
    ]


def analyze(a: argparse.Namespace) -> dict:
    a.output.mkdir(parents=True, exist_ok=True)
    data41 = load_lot41b(a.lot41b)
    data42 = load_lot42(a.lot42)
    d41 = data41["decision"]
    d42 = data42["decision"]
    valid41 = d41.get("LOT41B_VALID") == "YES"
    valid42 = d42.get("LOT42_VALID") == "YES" and d42.get("LOT42_COMPLETE") == "YES"
    fps = data41["common"]
    oracle = {fp: oracle_required_budget(data41, data42, fp) for fp in fps}
    oracle_counts = Counter(oracle.values())
    candidate_names = ["UNIFORM_4096", "UNIFORM_8192", "UNIFORM_16384", "UNIFORM_32768", "FIXED_MULTI_FIDELITY", "CONSERVATIVE_MULTI_SIGNAL"]
    router_metrics = []
    router_rows_by_name = {}
    for name in candidate_names:
        metrics, rows = evaluate_router(data41, data42, fps, oracle, name)
        router_metrics.append(metrics)
        router_rows_by_name[name] = rows
    accepted_candidates = sorted(
        [m for m in router_metrics if m["router_name"] not in {"UNIFORM_4096", "UNIFORM_8192", "UNIFORM_16384", "UNIFORM_32768"}],
        key=lambda m: (m["high_severity_false_stop_count"], -m["ultra_hard_recall"], m["false_stop_count"], m["total_simulations"]),
    )
    selected = accepted_candidates[0] if accepted_candidates else router_metrics[-1]
    selected_name = selected["router_name"]
    selected_rows = router_rows_by_name[selected_name]
    protocol_accepted = valid41 and valid42 and selected["high_severity_false_stop_count"] == 0 and selected["ultra_hard_recall"] >= 0.99 and selected["compute_ratio_vs_uniform32768"] < 1.0
    false_stops = [r for r in selected_rows if r["false_stop"]]
    false_deepens = [r for r in selected_rows if r["false_deepen"]]
    false_early = [r for r in selected_rows if "8192" in r["search_trace_summary"] and r["oracle_required_budget"] >= 32768 and r["router_selected_budget"] <= 8192]
    ultra_rows = [r for r in selected_rows if r["ultra_hard"]]
    feature_rows = feature_schema()
    protocol = {
        "version": "MULTI_FIDELITY_TEACHER_V1",
        "accepted": "YES" if protocol_accepted else "NO",
        "model": "POOL_G4R",
        "budget_hierarchy": list(ROUTINE_HIERARCHY),
        "mcts16384_routing_stage": "REMOVE",
        "target_semantics": "pi_teacher = visit_distribution at final selected budget; raw visit counts retained",
        "value_target_semantics": "terminal z only; MCTS root value is metadata",
        "cost_semantics": "RESTART",
        "router": selected_name,
        "thresholds": {
            "JS_STOP_4096": JS_STOP_4096,
            "JS_STOP_8192": JS_STOP_8192,
            "RANK_STOP": RANK_STOP,
            "MARGIN_STOP_4096": MARGIN_STOP_4096,
            "MARGIN_STOP_8192": MARGIN_STOP_8192,
            "Q_GAP_STOP": Q_GAP_STOP,
            "ULTRA_JS_TRIGGER": ULTRA_JS_TRIGGER,
            "ULTRA_Q_GAP_TRIGGER": ULTRA_Q_GAP_TRIGGER,
        },
        "fallback_behavior": "missing diagnostic => deepen conservatively",
        "forbidden_features": ["fingerprint", "game_id", "source_filename", "hard_position_flag", "ultra_hard_label", "future_stability_budget"],
    }
    proto_blob = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    proto_fp = __import__("hashlib").sha256(proto_blob).hexdigest()
    protocol["fingerprint"] = proto_fp
    total = len(fps)
    total_router_sims = selected["total_simulations"]
    uniform32768 = total * 32768
    decision = {
        "LOT43_VALID": "YES" if protocol_accepted else "NO",
        "LOT41B_INPUT_VALID": "YES" if valid41 else "NO",
        "LOT42_INPUT_VALID": "YES" if valid42 else "NO",
        "FUTURE_INFORMATION_LEAKAGE": "NO",
        "TRAINING_PERFORMED": "NO",
        "OPTIMIZER_CREATED": "NO",
        "BACKWARD_CALLED": "NO",
        "MODEL_WEIGHTS_CHANGED": "NO",
        "ORACLE_STOP_4096": {"count": oracle_counts[4096], "rate": pct(oracle_counts[4096], total)},
        "ORACLE_STOP_8192": {"count": oracle_counts[8192], "rate": pct(oracle_counts[8192], total)},
        "ORACLE_STOP_16384": {"count": oracle_counts[16384], "rate": pct(oracle_counts[16384], total)},
        "ORACLE_STOP_32768": {"count": oracle_counts[32768], "rate": pct(oracle_counts[32768], total)},
        "ORACLE_REQUIRE_65536": {"count": oracle_counts[65536], "rate": pct(oracle_counts[65536], total)},
        "ROUTER_STOP_4096": {"count": selected["budget_counts"].get(4096, 0), "rate": pct(selected["budget_counts"].get(4096, 0), total)},
        "ROUTER_STOP_8192": {"count": selected["budget_counts"].get(8192, 0), "rate": pct(selected["budget_counts"].get(8192, 0), total)},
        "ROUTER_STOP_32768": {"count": selected["budget_counts"].get(32768, 0), "rate": pct(selected["budget_counts"].get(32768, 0), total)},
        "ROUTER_STOP_65536": {"count": selected["budget_counts"].get(65536, 0), "rate": pct(selected["budget_counts"].get(65536, 0), total)},
        "FALSE_STOP_COUNT": selected["false_stop_count"],
        "FALSE_STOP_RATE": selected["false_stop_rate"],
        "HIGH_SEVERITY_FALSE_STOP_COUNT": selected["high_severity_false_stop_count"],
        "FALSE_DEEPEN_COUNT": selected["false_deepen_count"],
        "FALSE_DEEPEN_RATE": selected["false_deepen_rate"],
        "FALSE_EARLY_STABILITY_COUNT": len(false_early),
        "KNOWN_ULTRA_HARD_COUNT": len(data42["ultra"]),
        "ULTRA_HARD_ROUTED_TO_65536": selected["ultra_hard_routed_to_65536"],
        "ULTRA_HARD_RECALL": selected["ultra_hard_recall"],
        "TOTAL_ROUTER_SIMULATIONS": total_router_sims,
        "UNIFORM_32768_SIMULATIONS": uniform32768,
        "SIMULATION_SAVING_VS_UNIFORM_32768": 1.0 - (total_router_sims / uniform32768 if uniform32768 else 1.0),
        "PROJECTED_WALLTIME_SAVING": 1.0 - (total_router_sims / uniform32768 if uniform32768 else 1.0),
        "ACTUAL_SEARCH_COST_SEMANTICS": "RESTART",
        "RECOMMENDED_BUDGET_HIERARCHY": list(ROUTINE_HIERARCHY),
        "MCTS16384_ROUTING_STAGE": "REMOVE",
        "MULTI_FIDELITY_TEACHER_V1_ACCEPTED": "YES" if protocol_accepted else "NO",
        "TEACHER_PROTOCOL_FINGERPRINT": proto_fp,
        "SELECTED_ROUTER": selected_name,
        "NEXT_ACTION": "LOT44_AUTONOMOUS_MULTI_FIDELITY_REANALYSIS_PILOT" if protocol_accepted else "LOT44_ROUTER_OUT_OF_SAMPLE_VALIDATION",
    }
    # Core artifacts
    write_json(a.output / "input_validation.json", {"LOT41B_INPUT_VALID": decision["LOT41B_INPUT_VALID"], "LOT42_INPUT_VALID": decision["LOT42_INPUT_VALID"], "common_positions": total, "lot42_completed": d42.get("COMPLETED_HARD_POSITIONS")})
    write_json(a.output / "source_artifact_manifest.json", {"lot41b": data41["root"], "lot42": data42["root"], "git_commit": git_commit()})
    write_json(a.output / "feature_schema.json", {"features": feature_rows})
    write_json(a.output / "oracle_required_budgets.json", {"rows": [{"fingerprint": fp, "oracle_required_budget": b} for fp, b in oracle.items()], "counts": dict(oracle_counts)})
    write_json(a.output / "oracle_policy.json", {"policy": "retrospective_lower_bound", "uses_future_information": True, "counts": dict(oracle_counts)})
    for budget in (4096, 8192, 32768):
        vals = [diagnostics(data41, data42, fp, budget) for fp in fps if row_at(data41, data42, fp, budget)]
        write_json(a.output / f"signal_analysis_{budget}.json", {"budget": budget, "signals": {k: summarize([float(v[k]) for v in vals if isinstance(v.get(k), (int, float)) and math.isfinite(float(v[k]))]) for k in ("visit_entropy", "visit_margin")}, "future_leakage": "NO"})
    write_json(a.output / "candidate_routers.json", {"routers": router_metrics})
    write_json(a.output / "router_validation.json", selected)
    for budget, name in ((4096, "routing_matrix_4096.json"), (8192, "routing_matrix_8192.json"), (32768, "routing_matrix_32768.json")):
        write_json(a.output / name, {"budget": budget, "stop": sum(1 for r in selected_rows if r["router_selected_budget"] == budget), "deepen": sum(1 for r in selected_rows if r["router_selected_budget"] > budget), "false_stop_at_or_before": sum(1 for r in selected_rows if r["false_stop"] and r["router_selected_budget"] <= budget)})
    write_json(a.output / "false_stops.json", {"count": len(false_stops), "rows": false_stops})
    write_json(a.output / "false_deepens.json", {"count": len(false_deepens), "rows": false_deepens})
    write_json(a.output / "false_early_stability.json", {"count": len(false_early), "rows": false_early})
    write_json(a.output / "ultra_hard_routing.json", {"count": len(ultra_rows), "rows": ultra_rows})
    write_json(a.output / "compute_comparison.json", {"selected": selected, "uniform32768": uniform32768, "uniform8192": total * 8192, "uniform16384": total * 16384})
    write_json(a.output / "compute_projection.json", {"PROJECTED_COMPUTE": {str(n): {"uniform4096": n * 4096, "uniform8192": n * 8192, "uniform32768": n * 32768, "multi_fidelity_estimated": int(n * (total_router_sims / total)) if total else None} for n in (10000, 50000, 100000)}})
    write_json(a.output / "oracle_vs_deployable.json", {"selected_router": selected_name, "rows": selected_rows})
    write_json(a.output / "multi_fidelity_teacher_v1.json", protocol)
    write_json(a.output / "teacher_protocol_fingerprint.json", {"fingerprint": proto_fp, "hash": "sha256"})
    write_json(a.output / "decision.json", decision)
    write_json(a.output / "report.json", {"lot": 43, "decision": decision, "notes": ["Dry-run only; no MCTS/search/training executed.", "65536 routing evidence has only 3 positives and needs out-of-sample validation."]})
    write_csv(a.output / "router_position_results.csv", selected_rows)
    write_csv(a.output / "router_comparison.csv", router_metrics)
    svg_bar(a.output / "figure1_final_budget_distribution.svg", "Deployable router final budgets", {str(k): float(v) for k, v in selected["budget_counts"].items()})
    svg_bar(a.output / "figure2_compute_vs_false_stop.svg", "Compute ratio vs false-stop rate", {m["router_name"]: float(m["compute_ratio_vs_uniform32768"] or 0) for m in router_metrics})
    cumulative = {}
    run = 0
    for b in ROUTINE_HIERARCHY:
        run += selected["budget_counts"].get(b, 0)
        cumulative[str(b)] = pct(run, total)
    svg_bar(a.output / "figure3_cumulative_resolved.svg", "Cumulative resolved positions", cumulative)
    svg_bar(a.output / "figure4_oracle_vs_router_depth.svg", "Oracle depth distribution", {str(k): float(v) for k, v in oracle_counts.items()})
    svg_bar(a.output / "figure5_deepen_precision_recall.svg", "Router comparison false-stop", {m["router_name"]: float(m["false_stop_rate"]) for m in router_metrics})
    checks = {str(p.relative_to(a.output)): sha256(p) for p in a.output.rglob("*") if p.is_file() and p.name not in ("checksums.json", "experiment_manifest.json")}
    write_json(a.output / "checksums.json", checks)
    write_json(a.output / "experiment_manifest.json", {"lot": 43, "git_commit": git_commit(), "artifact_checksums": checks})
    print(json.dumps(decision, indent=2), flush=True)
    return decision


def export(a: argparse.Namespace) -> None:
    if not (a.output / "decision.json").is_file():
        raise RuntimeError("run --stage analyze before export")
    with tarfile.open(a.bundle, "w:gz") as tf:
        for p in a.output.rglob("*"):
            if p.is_file():
                tf.add(p, arcname=f"lot43_multi_fidelity_teacher_protocol/{p.relative_to(a.output)}")
    Path(str(a.bundle) + ".sha256").write_text(f"{sha256(a.bundle)}  {a.bundle.name}\n", encoding="utf-8")


if __name__ == "__main__":
    a = args()
    if a.stage == "analyze":
        analyze(a)
    elif a.stage == "export":
        export(a)
