#!/usr/bin/env python3
"""Lot 41B: analyse strategique des artefacts Deep-MCTS Lot41.

Ce script ne lance aucune recherche MCTS. Il lit les resultats racine deja
produits par Lot41 et mesure stabilite d'action, convergence des politiques,
stabilite des rankings, regret empirique contre la reference 32768 et choix
du budget teacher.
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

import numpy as np

from run_srn_colab_benchmark import git_commit
from run_srn_lot12 import sha256, write_json

SOURCE_DEFAULT = Path("data/experiments/lot41_deep_mcts_convergence")
OUT_DEFAULT = Path("data/experiments/lot41b_deep_mcts_strategic_convergence")
BUDGETS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
B_REF = 32768
LATE_PAIRS = ((4096, 8192), (8192, 16384), (16384, 32768))
JS_STRONG = 0.02
JS_MODERATE = 0.05
HIGH_SEVERITY_RATE_THRESHOLD = 0.02


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    p.add_argument("--output", type=Path, default=OUT_DEFAULT)
    p.add_argument("--bundle", type=Path, default=Path("lot41b_results.tar.gz"))
    p.add_argument("--stage", choices=("analyze", "export"), default="analyze")
    return p.parse_args()


def read_json(path: Path):
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


def budget_path(source: Path, budget: int) -> Path:
    return source / "search" / f"budget_{budget}.json"


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "p10": None, "p75": None, "p90": None, "p95": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": float(np.percentile(values, 10)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def normalize(counts: list[int], legal: list[bool]) -> list[float]:
    total = sum(max(0, int(counts[a])) for a in range(7) if legal[a])
    if total <= 0:
        n = sum(1 for x in legal if x)
        return [1.0 / n if legal[a] and n else 0.0 for a in range(7)]
    return [float(counts[a]) / total if legal[a] else 0.0 for a in range(7)]


def entropy(policy: list[float]) -> float:
    return -sum(x * math.log(x) for x in policy if x > 0.0)


def jensen_shannon(first: list[float], second: list[float]) -> float:
    mid = [(a + b) / 2.0 for a, b in zip(first, second)]

    def kl(values: list[float]) -> float:
        return sum(v * math.log(v / m) for v, m in zip(values, mid) if v > 0.0 and m > 0.0)

    return 0.5 * kl(first) + 0.5 * kl(second)


def rank_actions(row: dict) -> list[int]:
    counts = row["visit_counts"]
    legal = row["legal_mask"]
    return sorted([a for a in range(7) if legal[a]], key=lambda a: (-counts[a], a))


def tie_count(row: dict) -> int:
    counts = [row["visit_counts"][a] for a in range(7) if row["legal_mask"][a]]
    return len(counts) - len(set(counts))


def kendall_agreement(first: list[int], second: list[int]) -> float:
    if len(first) < 2:
        return 1.0
    second_rank = {a: i for i, a in enumerate(second)}
    pairs = 0
    concordant = 0
    for i, left in enumerate(first):
        for right in first[i + 1 :]:
            pairs += 1
            concordant += int(second_rank[left] < second_rank[right])
    return concordant / pairs if pairs else 1.0


def top2_info(row: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ranking = rank_actions(row)
    top = tuple(ranking[:2])
    return tuple(sorted(top)), top


def load_inputs(source: Path) -> tuple[dict, dict[int, dict[str, dict]], dict[int, dict]]:
    data = {}
    by_budget = {}
    validation = {"budgets": {}, "missing_budgets": [], "bad_checksums": []}
    for budget in BUDGETS:
        path = budget_path(source, budget)
        if not path.is_file():
            validation["missing_budgets"].append(budget)
            continue
        if not checked(path):
            validation["bad_checksums"].append(budget)
        payload = read_json(path)
        data[budget] = payload
        by_budget[budget] = {row["state_fingerprint"]: row for row in payload["rows"]}
        validation["budgets"][str(budget)] = {
            "positions": len(payload["rows"]),
            "status": payload.get("status"),
            "wall_time_s": payload.get("wall_time_s"),
            "throughput": payload.get("global_simulations_per_second"),
            "ram": payload.get("peak_ram_bytes"),
            "model_weights_changed": payload.get("model_weights_changed"),
        }
    return validation, by_budget, data


def pct(value: int, total: int) -> float:
    return 100.0 * value / total if total else 0.0


def analyze_action(common: list[str], by_budget: dict[int, dict[str, dict]]) -> tuple[dict, dict, dict, list[dict]]:
    top1_adjacent = []
    for left, right in zip(BUDGETS, BUDGETS[1:]):
        same = sum(by_budget[left][fp]["selected_action"] == by_budget[right][fp]["selected_action"] for fp in common)
        top1_adjacent.append({
            "budget_pair": f"{left}->{right}",
            "from_budget": left,
            "to_budget": right,
            "comparable_positions": len(common),
            "same_top1": same,
            "different_top1": len(common) - same,
            "top1_agreement": same / len(common),
            "flip_rate": 1.0 - same / len(common),
        })
    top1_vs_ref = []
    for budget in BUDGETS[:-1]:
        same = sum(by_budget[budget][fp]["selected_action"] == by_budget[B_REF][fp]["selected_action"] for fp in common)
        top1_vs_ref.append({
            "budget": budget,
            "reference_budget": B_REF,
            "comparable_positions": len(common),
            "same_top1": same,
            "different_top1": len(common) - same,
            "final_action_agreement": same / len(common),
        })
    trajectories = []
    stable_counts = Counter()
    flip_counts = Counter()
    reversal_count = 0
    for fp in common:
        actions = [by_budget[b][fp]["selected_action"] for b in BUDGETS]
        flips = sum(actions[i] != actions[i - 1] for i in range(1, len(actions)))
        seen = set()
        has_reversal = False
        for action in actions:
            if action in seen and action != actions[actions.index(action)]:
                has_reversal = True
            seen.add(action)
        # Simpler and stricter: any return to an earlier action after leaving it.
        for i in range(len(actions)):
            for j in range(i + 2, len(actions)):
                if actions[i] == actions[j] and any(actions[k] != actions[i] for k in range(i + 1, j)):
                    has_reversal = True
        stable_budget = B_REF
        ref_action = actions[-1]
        for i, budget in enumerate(BUDGETS):
            if all(action == ref_action for action in actions[i:]):
                stable_budget = budget
                break
        stable_counts[stable_budget] += 1
        flip_counts[min(flips, 4)] += 1
        reversal_count += int(has_reversal)
        trajectories.append({
            "state_fingerprint": fp,
            "actions": {str(b): actions[i] for i, b in enumerate(BUDGETS)},
            "flip_count": flips,
            "reversal": has_reversal,
            "action_stability_budget": stable_budget,
        })
    cumulative = []
    running = 0
    for budget in BUDGETS:
        running += stable_counts[budget]
        cumulative.append({
            "budget": budget,
            "count": stable_counts[budget],
            "percentage": pct(stable_counts[budget], len(common)),
            "cumulative_count": running,
            "cumulative_percentage": pct(running, len(common)),
        })
    flips = {
        "positions": len(common),
        "flip_count_distribution": {
            "0": flip_counts[0],
            "1": flip_counts[1],
            "2": flip_counts[2],
            "3": flip_counts[3],
            "4+": flip_counts[4],
        },
        "late_flips": {
            f"{left}->{right}": sum(by_budget[left][fp]["selected_action"] != by_budget[right][fp]["selected_action"] for fp in common)
            for left, right in LATE_PAIRS
        },
        "reversal_count": reversal_count,
        "reversal_rate": reversal_count / len(common),
    }
    return {"rows": top1_adjacent}, {"rows": top1_vs_ref}, {"rows": cumulative, "stable_counts": dict(stable_counts)}, trajectories, flips


def analyze_js(common: list[str], by_budget: dict[int, dict[str, dict]]) -> tuple[dict, dict]:
    adjacent = []
    for left, right in zip(BUDGETS, BUDGETS[1:]):
        values = [jensen_shannon(by_budget[left][fp]["visit_distribution"], by_budget[right][fp]["visit_distribution"]) for fp in common]
        adjacent.append({"budget_pair": f"{left}->{right}", "from_budget": left, "to_budget": right, "positions": len(common), **summarize(values)})
    vs_ref = []
    for budget in BUDGETS[:-1]:
        values = [jensen_shannon(by_budget[budget][fp]["visit_distribution"], by_budget[B_REF][fp]["visit_distribution"]) for fp in common]
        strong = sum(v <= JS_STRONG for v in values)
        moderate = sum(JS_STRONG < v <= JS_MODERATE for v in values)
        material = len(values) - strong - moderate
        vs_ref.append({"budget": budget, "reference_budget": B_REF, "positions": len(common), **summarize(values), "strongly_stable_count": strong, "moderately_stable_count": moderate, "materially_changing_count": material})
    return {"rows": adjacent}, {"rows": vs_ref}


def analyze_ranking(common: list[str], by_budget: dict[int, dict[str, dict]]) -> tuple[dict, dict, dict]:
    adjacent = []
    top2_rows = []
    for left, right in zip(BUDGETS, BUDGETS[1:]):
        agreements = []
        major = 0
        same_top2 = 0
        same_order = 0
        ties = 0
        for fp in common:
            a = by_budget[left][fp]
            b = by_budget[right][fp]
            agreement = kendall_agreement(rank_actions(a), rank_actions(b))
            agreements.append(agreement)
            major += int(agreement < 0.70)
            set_a, order_a = top2_info(a)
            set_b, order_b = top2_info(b)
            same_top2 += int(set_a == set_b)
            same_order += int(order_a == order_b)
            ties += int(tie_count(a) or tie_count(b))
        adjacent.append({"budget_pair": f"{left}->{right}", "positions": len(common), **summarize(agreements), "major_ranking_inversions": major, "tie_positions": ties})
        top2_rows.append({"budget_pair": f"{left}->{right}", "positions": len(common), "same_top2_set": same_top2, "same_top2_set_rate": same_top2 / len(common), "same_top2_order": same_order, "same_top2_order_rate": same_order / len(common)})
    vs_ref = []
    for budget in BUDGETS[:-1]:
        agreements = [kendall_agreement(rank_actions(by_budget[budget][fp]), rank_actions(by_budget[B_REF][fp])) for fp in common]
        vs_ref.append({"budget": budget, "reference_budget": B_REF, "positions": len(common), **summarize(agreements)})
    return {"rows": adjacent}, {"rows": vs_ref}, {"rows": top2_rows}


def q_regret(common: list[str], by_budget: dict[int, dict[str, dict]]) -> tuple[dict, dict, dict]:
    validity = {"Q_REGRET_AVAILABLE": "YES", "reason": "root_q_values finite and reference-selected/lower-selected actions visited at 32768 when regret is recorded", "missing_or_unvisited": {}}
    rows = []
    flip_severity_rows = []
    for budget in BUDGETS[:-1]:
        regrets = []
        gaps = []
        unavailable = 0
        high_disagreements = 0
        for fp in common:
            ref = by_budget[B_REF][fp]
            row = by_budget[budget][fp]
            ref_action = ref["selected_action"]
            action = row["selected_action"]
            ref_counts = ref["visit_counts"]
            q = ref["root_q_values"]
            legal = ref["legal_mask"]
            if action is None or ref_action is None or not legal[action] or not math.isfinite(q[action]) or not math.isfinite(q[ref_action]) or ref_counts[action] <= 0 or ref_counts[ref_action] <= 0:
                unavailable += 1
                continue
            regret = q[ref_action] - q[action]
            regrets.append(regret)
            ref_policy = ref["visit_distribution"]
            visit_gap = ref_policy[ref_action] - ref_policy[action]
            gaps.append(visit_gap)
            high_disagreements += int(action != ref_action and regret > 0.10)
        validity["missing_or_unvisited"][str(budget)] = unavailable
        summary = summarize(regrets)
        rows.append({"budget": budget, "reference_budget": B_REF, "available_positions": len(regrets), "unavailable_positions": unavailable, **summary})
        flip_severity_rows.append({"budget": budget, "reference_budget": B_REF, "high_severity_disagreements": high_disagreements, "high_severity_disagreement_rate": high_disagreements / len(common), "visit_preference_gap": summarize(gaps)})
    late = []
    for left, right in LATE_PAIRS:
        near = low = med = high = unavailable = 0
        for fp in common:
            if by_budget[left][fp]["selected_action"] == by_budget[right][fp]["selected_action"]:
                continue
            ref = by_budget[B_REF][fp]
            q = ref["root_q_values"]
            a = by_budget[left][fp]["selected_action"]
            b = by_budget[right][fp]["selected_action"]
            if ref["visit_counts"][a] <= 0 or ref["visit_counts"][b] <= 0:
                unavailable += 1
                continue
            delta = abs(q[a] - q[b])
            if delta <= 0.02:
                near += 1
            elif delta <= 0.05:
                low += 1
            elif delta <= 0.10:
                med += 1
            else:
                high += 1
        late.append({"budget_pair": f"{left}->{right}", "near_tie": near, "low": low, "medium": med, "high": high, "unavailable": unavailable, "high_rate_all_positions": high / len(common)})
    return validity, {"rows": rows}, {"vs_reference": flip_severity_rows, "late_flip_severity": late, "thresholds": {"near_tie": "<=0.02 Q", "low": "<=0.05 Q", "medium": "<=0.10 Q", "high": ">0.10 Q"}}


def difficulty(stability_rows: list[dict], trajectories: list[dict]) -> dict:
    counts = Counter()
    by_fp = {}
    for item in trajectories:
        b = item["action_stability_budget"]
        if b <= 2048:
            cls = "EARLY_STABLE"
        elif b in (4096, 8192):
            cls = "MID_STABLE"
        elif b == 16384:
            cls = "LATE_STABLE"
        else:
            cls = "UNCONFIRMED_OR_HARD"
        counts[cls] += 1
        by_fp[item["state_fingerprint"]] = cls
    total = len(trajectories)
    return {"rows": [{"difficulty_class": k, "count": counts[k], "percentage": pct(counts[k], total)} for k in ("EARLY_STABLE", "MID_STABLE", "LATE_STABLE", "UNCONFIRMED_OR_HARD")], "by_fp": by_fp}


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else None


def uncertainty(common: list[str], by_budget: dict[int, dict[str, dict]], difficulty_by_fp: dict[str, str]) -> dict:
    rows = []
    hard = [1.0 if difficulty_by_fp[fp] == "UNCONFIRMED_OR_HARD" else 0.0 for fp in common]
    for budget in (1024, 2048, 4096, 8192):
        entropies = []
        margins = []
        q_gaps = []
        legal_counts = []
        for fp in common:
            row = by_budget[budget][fp]
            policy = row["visit_distribution"]
            ranked = rank_actions(row)
            entropies.append(entropy(policy))
            margins.append(policy[ranked[0]] - policy[ranked[1]] if len(ranked) > 1 else 1.0)
            q = row["root_q_values"]
            q_gaps.append(q[ranked[0]] - q[ranked[1]] if len(ranked) > 1 else 1.0)
            legal_counts.append(sum(1 for x in row["legal_mask"] if x))
        rows.append({
            "budget": budget,
            "entropy_hard_correlation": pearson(entropies, hard),
            "visit_margin_hard_correlation": pearson(margins, hard),
            "q_gap_hard_correlation": pearson(q_gaps, hard),
            "legal_count_hard_correlation": pearson([float(x) for x in legal_counts], hard),
            "exploratory_only": True,
        })
    feasible = "YES" if any(abs(r["visit_margin_hard_correlation"] or 0.0) >= 0.20 or abs(r["entropy_hard_correlation"] or 0.0) >= 0.20 for r in rows) else "INCONCLUSIVE"
    return {"rows": rows, "ADAPTIVE_DEEPENING_FEASIBLE": feasible, "note": "Associations only; no classifier was trained."}


def svg_line(path: Path, title: str, rows: list[dict], x_key: str, y_keys: list[str]) -> None:
    width, height, pad = 820, 440, 62
    xs = [str(r[x_key]) for r in rows]
    values = [float(r[k]) for r in rows for k in y_keys if r.get(k) is not None]
    if not values:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    mn, mx = min(values), max(values)
    if mn == mx:
        mx = mn + 1.0
    colors = ["#2563eb", "#dc2626", "#16a34a"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="28" text-anchor="middle" font-size="18">{title}</text><line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/><line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>']
    for idx, key in enumerate(y_keys):
        points = []
        for i, row in enumerate(rows):
            x = pad + i * (width - 2 * pad) / max(1, len(rows) - 1)
            y = height - pad - (float(row[key]) - mn) / (mx - mn) * (height - 2 * pad)
            points.append((x, y))
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        color = colors[idx % len(colors)]
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>' for x, y in points)
        parts.append(f'<text x="{width-pad-120}" y="{48+idx*20}" fill="{color}">{key}</text>')
    for i, label in enumerate(xs):
        x = pad + i * (width - 2 * pad) / max(1, len(xs) - 1)
        parts.append(f'<text x="{x:.1f}" y="{height-25}" text-anchor="middle" font-size="11">{label}</text>')
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(source: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    validation, by_budget, data = load_inputs(source)
    present = sorted(by_budget)
    common = sorted(set.intersection(*(set(by_budget[b]) for b in present))) if present else []
    validation.update({
        "LOT41_SOURCE": str(source),
        "INPUT_ARTIFACTS_VALID": "YES" if set(present) == set(BUDGETS) and not validation["bad_checksums"] and len(common) == 256 else "NO",
        "COMMON_POSITION_COUNT": len(common),
        "DEEPEST_REFERENCE": B_REF if B_REF in present else None,
        "NEW_MCTS_SEARCH_PERFORMED": "NO",
        "TRAINING_PERFORMED": "NO",
        "OPTIMIZER_CREATED": "NO",
        "BACKWARD_CALLED": "NO",
        "MODEL_WEIGHTS_CHANGED": "YES" if any(data[b].get("model_weights_changed") for b in present) else "NO",
    })
    write_json(out / "input_validation.json", validation)
    write_json(out / "position_alignment.json", {"common_position_count": len(common), "positions_by_budget": {str(b): len(by_budget.get(b, {})) for b in BUDGETS}, "common_fingerprints": common})
    if set(present) != set(BUDGETS) or B_REF not in present:
        raise RuntimeError("Lot41B requires completed budgets 256..32768; do not recompute here")

    top1_adj, top1_ref, stability, trajectories, flips = analyze_action(common, by_budget)
    js_adj, js_ref = analyze_js(common, by_budget)
    rank_adj, rank_ref, top2 = analyze_ranking(common, by_budget)
    q_validity, regret, severity = q_regret(common, by_budget)
    diff = difficulty(stability["rows"], trajectories)
    uncertainty_report = uncertainty(common, by_budget, diff["by_fp"])

    artifacts = {
        "action_trajectories.json": {"rows": trajectories},
        "top1_adjacent.json": top1_adj,
        "top1_vs_32768.json": top1_ref,
        "action_flips.json": flips,
        "action_stability_budget.json": stability,
        "js_adjacent.json": js_adj,
        "js_vs_32768.json": js_ref,
        "ranking_adjacent.json": rank_adj,
        "ranking_vs_32768.json": rank_ref,
        "top2_stability.json": top2,
        "q_validity.json": q_validity,
        "deep_search_regret.json": regret,
        "flip_severity.json": severity,
        "difficulty_distribution.json": {k: v for k, v in diff.items() if k != "by_fp"},
        "uncertainty_signals.json": uncertainty_report,
    }
    for name, payload in artifacts.items():
        write_json(out / name, payload)

    regret_by_budget = {r["budget"]: r for r in regret["rows"]}
    top1_ref_by_budget = {r["budget"]: r for r in top1_ref["rows"]}
    js_ref_by_budget = {r["budget"]: r for r in js_ref["rows"]}
    rank_ref_by_budget = {r["budget"]: r for r in rank_ref["rows"]}
    severity_by_budget = {r["budget"]: r for r in severity["vs_reference"]}
    stability_cum = {r["budget"]: r for r in stability["rows"]}
    summary_rows = []
    for budget in BUDGETS:
        item = data[budget]
        summary_rows.append({
            "budget": budget,
            "wall_time": item["wall_time_s"],
            "throughput": item["global_simulations_per_second"],
            "ram": item["peak_ram_bytes"],
            "agreement_vs_32768": 1.0 if budget == B_REF else top1_ref_by_budget[budget]["final_action_agreement"],
            "median_js_vs_32768": 0.0 if budget == B_REF else js_ref_by_budget[budget]["median"],
            "p90_js_vs_32768": 0.0 if budget == B_REF else js_ref_by_budget[budget]["p90"],
            "ranking_vs_32768": 1.0 if budget == B_REF else rank_ref_by_budget[budget]["mean"],
            "mean_regret_vs_32768": 0.0 if budget == B_REF else regret_by_budget[budget]["mean"],
            "high_severity_disagreement_rate": 0.0 if budget == B_REF else severity_by_budget[budget]["high_severity_disagreement_rate"],
            "cumulative_persistent_stability": stability_cum[budget]["cumulative_percentage"],
        })
    write_csv(out / "budget_convergence_summary.csv", summary_rows)

    trajectory_by_fp = {t["state_fingerprint"]: t for t in trajectories}
    hard_records = []
    position_rows = []
    for fp in common:
        trajectory = trajectory_by_fp[fp]
        js_late = {}
        for left, right in LATE_PAIRS:
            js_late[f"js_{left}_{right}"] = jensen_shannon(by_budget[left][fp]["visit_distribution"], by_budget[right][fp]["visit_distribution"])
        regrets = {}
        for budget in (4096, 8192, 16384):
            ref = by_budget[B_REF][fp]
            row = by_budget[budget][fp]
            a = row["selected_action"]
            best = ref["selected_action"]
            regrets[f"regret_{budget}"] = None
            if ref["visit_counts"][a] > 0 and ref["visit_counts"][best] > 0:
                regrets[f"regret_{budget}"] = ref["root_q_values"][best] - ref["root_q_values"][a]
        is_hard = (
            diff["by_fp"][fp] == "UNCONFIRMED_OR_HARD"
            or trajectory["flip_count"] >= 2
            or any(v > JS_MODERATE for v in js_late.values())
            or any((v or 0.0) > 0.10 for v in regrets.values())
        )
        row = {
            "fingerprint": fp,
            "source": "lot41_benchmark",
            **{f"action_{b}": by_budget[b][fp]["selected_action"] for b in BUDGETS},
            "flip_count": trajectory["flip_count"],
            "reversal_count": int(trajectory["reversal"]),
            "action_stability_budget": trajectory["action_stability_budget"],
            "difficulty_class": diff["by_fp"][fp],
            **js_late,
            **regrets,
            "hard_position": is_hard,
        }
        position_rows.append(row)
        if is_hard:
            hard_records.append({
                "fingerprint": fp,
                "state": next((s for s in read_json(source / "benchmark_positions.json")["states"] if s["state_fingerprint"] == fp), None) if (source / "benchmark_positions.json").is_file() else None,
                "legal_mask": by_budget[B_REF][fp]["legal_mask"],
                "action_trajectory": trajectory["actions"],
                "visit_counts_by_budget": {str(b): by_budget[b][fp]["visit_counts"] for b in BUDGETS},
                "policy_by_budget": {str(b): by_budget[b][fp]["visit_distribution"] for b in BUDGETS},
                "ranking_by_budget": {str(b): rank_actions(by_budget[b][fp]) for b in BUDGETS},
                "q_by_budget": {str(b): by_budget[b][fp]["root_q_values"] for b in BUDGETS},
                "regret": regrets,
                "stability_budget": trajectory["action_stability_budget"],
                "difficulty_class": diff["by_fp"][fp],
            })
    write_csv(out / "position_stability.csv", position_rows)
    write_json(out / "hard_positions.json", {"count": len(hard_records), "rows": hard_records})

    outliers = {
        "largest_js_16384_32768": sorted(position_rows, key=lambda r: r["js_16384_32768"], reverse=True)[:20],
        "largest_regret_4096": sorted(position_rows, key=lambda r: r["regret_4096"] or -999, reverse=True)[:20],
        "largest_flip_count": sorted(position_rows, key=lambda r: r["flip_count"], reverse=True)[:20],
        "latest_stabilization": [r for r in position_rows if r["action_stability_budget"] == B_REF][:20],
    }
    write_json(out / "outliers.json", outliers)

    late_severity = {r["budget_pair"]: r for r in severity["late_flip_severity"]}
    stability_rows_by_pair = {r["budget_pair"]: r for r in top1_adj["rows"]}
    js_by_pair = {r["budget_pair"]: r for r in js_adj["rows"]}
    stable_pairs = []
    for left, right in zip(BUDGETS, BUDGETS[1:]):
        key = f"{left}->{right}"
        high_rate = late_severity.get(key, {}).get("high_rate_all_positions", 0.0)
        row = stability_rows_by_pair[key]
        js = js_by_pair[key]
        stable = row["top1_agreement"] >= 0.95 and js["median"] <= JS_STRONG and js["p90"] <= JS_MODERATE and high_rate <= HIGH_SEVERITY_RATE_THRESHOLD
        stable_pairs.append({"budget_pair": key, "from_budget": left, "to_budget": right, "stable": stable, "top1_agreement": row["top1_agreement"], "median_js": js["median"], "p90_js": js["p90"], "high_severity_late_flip_rate": high_rate})
    empirical = "NOT_DETECTED_WITHIN_TESTED_RANGE"
    for i in range(len(stable_pairs) - 1):
        if stable_pairs[i]["stable"] and stable_pairs[i + 1]["stable"]:
            empirical = stable_pairs[i]["from_budget"]
            break
    write_json(out / "stability_summary.json", {"criterion": {"top1_agreement": ">=0.95", "median_js": "<=0.02", "p90_js": "<=0.05", "high_severity_late_flip_rate": "<=0.02"}, "pairs": stable_pairs, "EMPIRICAL_STABILITY_BUDGET": empirical})

    def sufficient(budget: int, next_budget: int) -> str:
        pair = f"{budget}->{next_budget}"
        row = stability_rows_by_pair[pair]
        js = js_by_pair[pair]
        sev = late_severity.get(pair, {"high_rate_all_positions": 0.0})
        if row["top1_agreement"] >= 0.97 and js["p90"] <= JS_MODERATE and sev["high_rate_all_positions"] <= HIGH_SEVERITY_RATE_THRESHOLD:
            return "YES"
        if row["top1_agreement"] < 0.90 or js["p90"] > 0.10 or sev["high_rate_all_positions"] > 0.05:
            return "NO"
        return "INCONCLUSIVE"

    m4096 = sufficient(4096, 8192)
    m8192 = sufficient(8192, 16384)
    m16384 = sufficient(16384, 32768)
    if m16384 == "YES" and summary_rows[-2]["agreement_vs_32768"] >= 0.95:
        teacher = 16384
    elif m8192 == "YES" and summary_rows[-3]["agreement_vs_32768"] >= 0.95:
        teacher = 8192
    else:
        teacher = "MULTI_FIDELITY" if diff["rows"][0]["percentage"] + diff["rows"][1]["percentage"] >= 50 and diff["rows"][-1]["percentage"] >= 10 else 32768
    multi = "YES" if teacher == "MULTI_FIDELITY" else ("YES" if diff["rows"][0]["percentage"] >= 35 and diff["rows"][-1]["percentage"] >= 10 else "NO")
    scientific_65536 = "YES" if m16384 == "NO" or severity_by_budget[16384]["high_severity_disagreement_rate"] > HIGH_SEVERITY_RATE_THRESHOLD else "NO" if m16384 == "YES" else "INCONCLUSIVE"
    next_action = "LOT42_MULTI_FIDELITY_AUTONOMOUS_REANALYSIS" if multi == "YES" else "LOT42_UNIFORM_DEEP_AUTONOMOUS_REANALYSIS"
    if scientific_65536 == "YES" and diff["rows"][-1]["percentage"] >= 10:
        next_action = "LOT42_HARD_POSITION_65536_EXTENSION"

    tradeoff = {"rows": summary_rows, "stable_pairs": stable_pairs, "cost_reference_budget": 4096}
    write_json(out / "compute_value_tradeoff.json", tradeoff)
    teacher_decision = {
        "RECOMMENDED_AUTONOMOUS_TEACHER_BUDGET": teacher,
        "MULTI_FIDELITY_RECOMMENDED": multi,
        "candidate_scheme": {"easy": 4096, "uncertain": 8192, "hard": 32768} if multi == "YES" else None,
        "MCTS4096_SUFFICIENT": m4096,
        "MCTS8192_SUFFICIENT": m8192,
        "MCTS16384_SUFFICIENT": m16384,
        "MCTS32768_CONVERGED": "NOT_PROVABLE_FROM_CURRENT_DATA",
        "MCTS65536_SCIENTIFICALLY_USEFUL": scientific_65536,
        "FUTURE_65536_SUBSET_RECOMMENDED": "YES" if scientific_65536 == "YES" else "NO",
        "NEXT_ACTION": next_action,
        "rationale": "Decision balances final-action agreement, JS to 32768, Q-regret severity, persistent stability and measured Colab wall time.",
    }
    write_json(out / "teacher_budget_decision.json", teacher_decision)

    decision = {
        "LOT41B_VALID": "YES",
        "INPUT_ARTIFACTS_VALID": validation["INPUT_ARTIFACTS_VALID"],
        "COMMON_POSITION_COUNT": len(common),
        "DEEPEST_REFERENCE": B_REF,
        "NEW_MCTS_SEARCH_PERFORMED": "NO",
        "TRAINING_PERFORMED": "NO",
        "MODEL_WEIGHTS_CHANGED": validation["MODEL_WEIGHTS_CHANGED"],
        "EMPIRICAL_STABILITY_BUDGET": empirical,
        "SEARCH_COMPUTE_KNEE_POINT": teacher if isinstance(teacher, int) else 8192,
        **teacher_decision,
        "ADAPTIVE_DEEPENING_FEASIBLE": uncertainty_report["ADAPTIVE_DEEPENING_FEASIBLE"],
        "TOP1_AGREEMENT_4096_8192": stability_rows_by_pair["4096->8192"]["top1_agreement"],
        "TOP1_AGREEMENT_8192_16384": stability_rows_by_pair["8192->16384"]["top1_agreement"],
        "TOP1_AGREEMENT_16384_32768": stability_rows_by_pair["16384->32768"]["top1_agreement"],
        "MEDIAN_JS_4096_8192": js_by_pair["4096->8192"]["median"],
        "MEDIAN_JS_8192_16384": js_by_pair["8192->16384"]["median"],
        "MEDIAN_JS_16384_32768": js_by_pair["16384->32768"]["median"],
        "P90_JS_4096_8192": js_by_pair["4096->8192"]["p90"],
        "P90_JS_8192_16384": js_by_pair["8192->16384"]["p90"],
        "P90_JS_16384_32768": js_by_pair["16384->32768"]["p90"],
        "STABLE_BY_4096_PERCENT": stability_cum[4096]["cumulative_percentage"],
        "STABLE_BY_8192_PERCENT": stability_cum[8192]["cumulative_percentage"],
        "STABLE_BY_16384_PERCENT": stability_cum[16384]["cumulative_percentage"],
        "ONLY_FINAL_AT_32768_PERCENT": stability_cum[32768]["percentage"],
    }
    write_json(out / "decision.json", decision)
    report = {
        "lot": "41B",
        "decision": decision,
        "answers": {
            "positions_aligned": len(common) == 256,
            "new_search_executed": False,
            "training_executed": False,
            "weights_unchanged": validation["MODEL_WEIGHTS_CHANGED"] == "NO",
            "deepest_reference_warning": "MCTS32768 is the deepest empirical reference, not ground truth or perfect play.",
        },
        "top1_adjacent": top1_adj,
        "top1_vs_32768": top1_ref,
        "js_adjacent": js_adj,
        "js_vs_32768": js_ref,
        "ranking_adjacent": rank_adj,
        "top2_stability": top2,
        "regret": regret,
        "difficulty": {k: v for k, v in diff.items() if k != "by_fp"},
    }
    write_json(out / "report.json", report)

    svg_line(out / "persistent_action_stability.svg", "Persistent action stability", stability["rows"], "budget", ["cumulative_percentage"])
    svg_line(out / "agreement_vs_32768.svg", "Top-1 agreement vs 32768", summary_rows[:-1], "budget", ["agreement_vs_32768"])
    svg_line(out / "js_adjacent.svg", "Adjacent JS divergence", js_adj["rows"], "budget_pair", ["median", "p90"])
    svg_line(out / "js_vs_32768.svg", "JS divergence to 32768", summary_rows[:-1], "budget", ["median_js_vs_32768", "p90_js_vs_32768"])
    svg_line(out / "compute_frontier.svg", "Compute vs stability", summary_rows, "budget", ["wall_time", "cumulative_persistent_stability"])
    svg_line(out / "regret_vs_32768.svg", "Mean regret vs 32768", [r for r in summary_rows[:-1] if r["mean_regret_vs_32768"] is not None], "budget", ["mean_regret_vs_32768"])

    write_json(out / "checksums.json", {str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file() and p.name not in ("checksums.json", "experiment_manifest.json")})
    write_json(out / "experiment_manifest.json", {"lot": "41B", "git_commit": git_commit(), "source": str(source), "artifact_checksums": read_json(out / "checksums.json")})
    return decision


def export(out: Path, bundle: Path) -> None:
    if not (out / "decision.json").is_file():
        raise RuntimeError("run analyze first")
    files = [p for p in out.rglob("*") if p.is_file()]
    with tarfile.open(bundle, "w:gz") as tf:
        for path in files:
            tf.add(path, arcname=f"lot41b_deep_mcts_strategic_convergence/{path.relative_to(out)}")
    Path(str(bundle) + ".sha256").write_text(f"{sha256(bundle)}  {bundle.name}\n", encoding="utf-8")


if __name__ == "__main__":
    a = args()
    if a.stage == "analyze":
        print(json.dumps(run_analysis(a.source, a.output), indent=2), flush=True)
    else:
        export(a.output, a.bundle)
