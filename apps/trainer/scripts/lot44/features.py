"""Features du routeur : uniquement l1/l2/l3, jamais la reference profonde.

``compute_features`` ne recoit physiquement aucune donnee ``ref`` : la fuite
65536 est impossible par construction (verifie par ``leakage_audit``).
"""
from __future__ import annotations

import inspect
import math

from run_srn_lot43 import entropy, jensen_shannon, kendall_agreement, normalize, rank_actions

from .artifacts import Lot44FatalError

CLASSIC = (
    "js_l2_l3",
    "margin_l3",
    "q_gap_l3",
    "rank_agreement_l2_l3",
    "entropy_l3",
    "visit_concentration_l3",
)
TRAJECTORY = (
    "js_l1_l2",
    "js_l1_l3",
    "delta_js",
    "action_changes",
    "action_changed_l2_l3",
    "ranking_volatility",
    "q_drift",
    "root_value_drift",
    "entropy_drift_l1_l2",
    "entropy_drift_l2_l3",
    "margin_drift_l1_l2",
    "margin_drift_l2_l3",
    "q_gap_drift_l2_l3",
)
STATE = ("legal_actions", "board_seeds_remaining", "root_value_l3")
COMBINED = CLASSIC + TRAJECTORY + STATE
FEATURE_SETS = {"classic": CLASSIC, "trajectory": TRAJECTORY, "combined": COMBINED}

FEATURE_DESCRIPTIONS = {
    "js_l2_l3": "Jensen-Shannon(pi_l2, pi_l3)",
    "margin_l3": "visit share gap top1-top2 at l3",
    "q_gap_l3": "Q[top1]-Q[top2] at l3",
    "rank_agreement_l2_l3": "Kendall pair agreement of rankings l2 vs l3",
    "entropy_l3": "visit distribution entropy at l3",
    "visit_concentration_l3": "max visit share at l3",
    "js_l1_l2": "Jensen-Shannon(pi_l1, pi_l2)",
    "js_l1_l3": "Jensen-Shannon(pi_l1, pi_l3) (total distribution drift)",
    "delta_js": "JS(l2,l3) - JS(l1,l2)",
    "action_changes": "number of top1 changes along l1->l2->l3",
    "action_changed_l2_l3": "top1(l2) != top1(l3)",
    "ranking_volatility": "(1-agree(l1,l2)) + (1-agree(l2,l3))",
    "q_drift": "|Q_l2[top1_l2]-Q_l1[top1_l1]| + |Q_l3[top1_l3]-Q_l2[top1_l2]|",
    "root_value_drift": "|V_l3-V_l2| + |V_l2-V_l1|",
    "entropy_drift_l1_l2": "entropy_l2 - entropy_l1",
    "entropy_drift_l2_l3": "entropy_l3 - entropy_l2",
    "margin_drift_l1_l2": "margin_l2 - margin_l1",
    "margin_drift_l2_l3": "margin_l3 - margin_l2",
    "q_gap_drift_l2_l3": "q_gap_l3 - q_gap_l2",
    "legal_actions": "number of legal actions",
    "board_seeds_remaining": "seeds left in pits (70 - stores), game phase",
    "root_value_l3": "MCTS root value at l3",
}

FORBIDDEN_COLUMNS = ("label", "label_reason", "deeper_regret", "js_l3_ref", "action_ref", "top1_flip", "flip_severity", "high_severity")


def _pi(row: dict) -> list[float]:
    return row.get("visit_distribution") or normalize(row)


def margin(row: dict) -> float:
    ranking = rank_actions(row)
    p = _pi(row)
    return float(p[ranking[0]] - p[ranking[1]]) if len(ranking) > 1 else 1.0


def q_gap(row: dict) -> float:
    ranking = rank_actions(row)
    if len(ranking) < 2:
        return 0.0
    q = row["root_q_values"]
    return float(q[ranking[0]] - q[ranking[1]])


def top1_q(row: dict) -> float:
    return float(row["root_q_values"][row["selected_action"]])


def compute_features(l1: dict, l2: dict, l3: dict, state: dict) -> dict[str, float]:
    p1, p2, p3 = _pi(l1), _pi(l2), _pi(l3)
    e1, e2, e3 = entropy(p1), entropy(p2), entropy(p3)
    m1, m2, m3 = margin(l1), margin(l2), margin(l3)
    r1, r2, r3 = rank_actions(l1), rank_actions(l2), rank_actions(l3)
    a12, a23 = kendall_agreement(r1, r2), kendall_agreement(r2, r3)
    js12, js23 = jensen_shannon(p1, p2), jensen_shannon(p2, p3)
    values = {
        "js_l2_l3": js23,
        "margin_l3": m3,
        "q_gap_l3": q_gap(l3),
        "rank_agreement_l2_l3": a23,
        "entropy_l3": e3,
        "visit_concentration_l3": max(p3),
        "js_l1_l2": js12,
        "js_l1_l3": jensen_shannon(p1, p3),
        "delta_js": js23 - js12,
        "action_changes": float((l1["selected_action"] != l2["selected_action"]) + (l2["selected_action"] != l3["selected_action"])),
        "action_changed_l2_l3": float(l2["selected_action"] != l3["selected_action"]),
        "ranking_volatility": (1.0 - a12) + (1.0 - a23),
        "q_drift": abs(top1_q(l2) - top1_q(l1)) + abs(top1_q(l3) - top1_q(l2)),
        "root_value_drift": abs(l3["root_value"] - l2["root_value"]) + abs(l2["root_value"] - l1["root_value"]),
        "entropy_drift_l1_l2": e2 - e1,
        "entropy_drift_l2_l3": e3 - e2,
        "margin_drift_l1_l2": m2 - m1,
        "margin_drift_l2_l3": m3 - m2,
        "q_gap_drift_l2_l3": q_gap(l3) - q_gap(l2),
        "legal_actions": float(len(r3)),
        "board_seeds_remaining": float(sum(state["board"][:14])),
        "root_value_l3": float(l3["root_value"]),
    }
    bad = [k for k, v in values.items() if not math.isfinite(v)]
    if bad:
        raise Lot44FatalError("NAN_CRITICAL", f"non-finite features {bad}")
    if set(values) != set(COMBINED):
        raise Lot44FatalError("FEATURE_SCHEMA", "feature set drifted from the frozen schema")
    return values


def feature_schema(budgets: dict[str, int]) -> dict:
    return {
        "version": "LOT44_PRE_REF_FEATURES_V1",
        "roles": {role: budgets[role] for role in ("l1", "l2", "l3")},
        "reference_role_budget": budgets["ref"],
        "sets": {k: list(v) for k, v in FEATURE_SETS.items()},
        "features": [{"name": n, "description": FEATURE_DESCRIPTIONS[n], "inputs": "l1/l2/l3 search rows + physical state", "uses_reference_search": False, "allowed_for_router": True} for n in COMBINED],
        "forbidden_columns": list(FORBIDDEN_COLUMNS),
        "compute_features_parameters": list(inspect.signature(compute_features).parameters),
    }
