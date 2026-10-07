"""Validation par position, construction du dataset G5, statistiques et audits.

Cible Policy : pi(a|S) = N(S,a) / sum_legal N (temperature 1, aucune
transformation irreversible) ; les visites brutes N(a) sont conservees.
Root Q / root value : diagnostics uniquement, jamais cible Value. La cible
Value de reference reste z (resultat terminal des parties autonomes), du
point de vue du joueur au trait, ou VALUE_TARGET_AVAILABLE = NO.
"""
from __future__ import annotations

import gzip
import io
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from songo_ai.songo.rules import SongoLegacyGame
from run_srn_lot39 import fingerprint_state
from run_srn_lot43 import entropy, jensen_shannon, kendall_agreement, rank_actions

from lot44.artifacts import atomic_write_bytes, canonical_hash
from lot44.corpus import state_from_dict

from .config import DATASET_NAME, DATASET_SCHEMA_VERSION, HIGH_CONFIDENCE_OLD_SHARE, ONE_HOT_LIKE_SHARE, POLICY_SUM_TOLERANCE


def engine_legal_mask(state: dict) -> list[bool]:
    game = SongoLegacyGame.from_state(state_from_dict(state).to_engine_state())
    game.normalize_terminal()
    return [False] * 7 if game.finished else list(game.legal_mask())


def policy_from_visits(visits: list[int], legal: list[bool]) -> list[float]:
    total = sum(v for v, ok in zip(visits, legal) if ok)
    return [v / total if ok else 0.0 for v, ok in zip(visits, legal)]


def validate_row(row: dict, selected: dict, budget: int) -> list[str]:
    """Liste des violations de la section 33 (vide = position acceptee)."""

    problems = []
    fp = selected["fingerprint"]
    if row["state_fingerprint"] != fp or fingerprint_state(state_from_dict(selected["state"])) != fp:
        problems.append("fingerprint_not_reproducible")
    legal = engine_legal_mask(selected["state"])
    if list(row["legal_mask"]) != legal:
        problems.append("legal_mask_differs_from_engine")
    visits = row["visit_counts"]
    if row["actual_simulations"] != budget:
        problems.append(f"simulation_count_{row['actual_simulations']}")
    if sum(visits) != budget:
        problems.append("visit_sum_mismatch")
    if any((not isinstance(v, int)) or v < 0 for v in visits):
        problems.append("negative_or_non_integer_visits")
    if any(v and not ok for v, ok in zip(visits, legal)):
        problems.append("visits_on_illegal_action")
    policy = row["visit_distribution"]
    if not all(math.isfinite(p) for p in policy) or abs(sum(policy) - 1.0) > POLICY_SUM_TOLERANCE:
        problems.append("policy_not_finite_or_not_normalized")
    elif any(abs(a - b) > 1e-9 for a, b in zip(policy, policy_from_visits(visits, legal))):
        problems.append("policy_differs_from_normalized_visits")
    action = row["selected_action"]
    if action is None or not legal[action] or visits[action] != max(v for v, ok in zip(visits, legal) if ok):
        problems.append("top_action_illegal_or_not_argmax")
    if not all(math.isfinite(q) for q in row["root_q_values"]) or not math.isfinite(row["root_value"]):
        problems.append("root_q_not_finite")
    return problems


def dataset_row(row: dict, selected: dict, shard: dict, *, teacher: dict, identity_fp: str) -> dict:
    return {
        "dataset": DATASET_NAME, "schema_version": DATASET_SCHEMA_VERSION,
        "fingerprint": selected["fingerprint"], "state": selected["state"], "legal_mask": row["legal_mask"],
        "visit_counts": row["visit_counts"], "policy_target": policy_from_visits(row["visit_counts"], row["legal_mask"]),
        "policy_target_definition": "N_32768(a)/sum_legal N_32768, temperature 1", "selected_action": row["selected_action"],
        "root_q_values_diagnostic": row["root_q_values"], "root_value_diagnostic": row["root_value"], "root_priors": row["root_priors"],
        "simulations": row["actual_simulations"], "nodes": row["nodes"], "network_evaluations": row["network_evaluations"],
        "runtime_s_estimate": shard["wall_time_s"] / shard["completed_positions"], "device": shard["device"], "shard": shard["partition"],
        "value_target_available": selected["value_target_available"], "z_counts": selected["z_counts"], "z_mean": selected["z_mean"], "z_perspective": selected["z_perspective"],
        "family": selected["family"], "primary_source": selected["primary_source"], "sources": selected["sources"], "source_occurrence_count": selected["source_occurrence_count"],
        "split_group": selected["split_group"], "holdout": selected["holdout"], "games_sample": selected["games_sample"], "min_ply": selected["min_ply"],
        "legal_count": selected["legal_count"], "seeds_in_play": selected["seeds_in_play"], "phase": selected["phase"], "order": selected["order"],
        "teacher_model": teacher, "mcts_identity_fingerprint": identity_fp, "root_noise": False,
    }


def write_dataset(path: Path, rows: list[dict]) -> str:
    """JSONL gzip deterministe (mtime=0) : meme contenu => meme sha256."""

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    atomic_write_bytes(path, buffer.getvalue())
    return canonical_hash([r["fingerprint"] for r in rows])


def read_dataset(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]  # noqa: E731
    return {"count": len(values), "mean": statistics.fmean(values), "p10": pick(0.10), "median": pick(0.50), "p90": pick(0.90), "min": ordered[0], "max": ordered[-1]}


def target_statistics(rows: list[dict]) -> dict:
    def stats(subset: list[dict]) -> dict:
        shares = [max(r["policy_target"]) for r in subset]
        margins = []
        for r in subset:
            ranked = sorted((p for p, ok in zip(r["policy_target"], r["legal_mask"]) if ok), reverse=True)
            margins.append(ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0))
        return {
            "positions": len(subset), "policy_entropy": _quantiles([entropy(r["policy_target"]) for r in subset]),
            "one_hot_like_rate": sum(s >= ONE_HOT_LIKE_SHARE for s in shares) / len(subset) if subset else None,
            "top1_concentration": _quantiles(shares), "top1_top2_margin": _quantiles(margins),
            "legal_count": dict(sorted(Counter(r["legal_count"] for r in subset).items())),
        }

    by_family = {}
    for family in sorted({r["family"] for r in rows}):
        by_family[family] = stats([r for r in rows if r["family"] == family])
    return {"overall": stats(rows), "by_family": by_family, "one_hot_like_threshold": ONE_HOT_LIKE_SHARE, "q_values": "kept as root_q_values_diagnostic only (not a Value target)"}


def old_vs_deep(rows: list[dict], selection: dict[str, dict]) -> dict:
    groups: dict[str, list[tuple[dict, dict]]] = {}
    for r in rows:
        for old in selection[r["fingerprint"]]["old_targets"]:
            key = f"{old['kind']}_MCTS{old['budget']}_noise_{old['root_noise']}"
            groups.setdefault(key, []).append((r, old))
    out = {}
    for key, pairs in sorted(groups.items()):
        agree, js, rank, e_old, e_new, high = 0, [], [], [], [], 0
        for r, old in pairs:
            legal = r["legal_mask"]
            old_counts = [int(v) if ok else 0 for v, ok in zip(old["visit_counts"], legal)]
            if sum(old_counts) == 0:
                continue
            old_pi = policy_from_visits(old_counts, legal)
            old_row = {"legal_mask": legal, "visit_counts": old_counts}
            old_top = rank_actions(old_row)[0]
            agree += old_top == r["selected_action"]
            js.append(jensen_shannon(old_pi, r["policy_target"]))
            rank.append(kendall_agreement(rank_actions(old_row), rank_actions({"legal_mask": legal, "visit_counts": r["visit_counts"]})))
            e_old.append(entropy(old_pi))
            e_new.append(entropy(r["policy_target"]))
            high += old_top != r["selected_action"] and max(old_pi) >= HIGH_CONFIDENCE_OLD_SHARE
        n = len(js)
        out[key] = {"positions": n, "top1_agreement": agree / n if n else None, "js": _quantiles(js), "ranking_agreement": _quantiles(rank), "entropy_old": _quantiles(e_old), "entropy_deep": _quantiles(e_new), "high_confidence_disagreements": high, "high_confidence_disagreement_rate": high / n if n else None}
    return {"comparisons": out, "high_confidence_old_share": HIGH_CONFIDENCE_OLD_SHARE, "nature": "diagnostic only (section 52); old self-play targets were searched with root noise and a play temperature"}


def target_audit(rows: list[dict], *, seed: int, size: int = 64) -> dict:
    sample = sorted(rows, key=lambda r: canonical_hash({"seed": seed, "audit": r["fingerprint"]}))[:size]
    entries = []
    for r in sample:
        legal = engine_legal_mask(r["state"])
        checks = {
            "fingerprint_reproducible": fingerprint_state(state_from_dict(r["state"])) == r["fingerprint"],
            "legal_mask_matches_engine": legal == r["legal_mask"],
            "visits_sum_equals_budget": sum(r["visit_counts"]) == r["simulations"],
            "no_illegal_visits": not any(v and not ok for v, ok in zip(r["visit_counts"], legal)),
            "policy_equals_normalized_visits": all(abs(a - b) <= 1e-9 for a, b in zip(r["policy_target"], policy_from_visits(r["visit_counts"], legal))),
            "top_action_legal_argmax": legal[r["selected_action"]] and r["visit_counts"][r["selected_action"]] == max(r["visit_counts"]),
        }
        entries.append({"fingerprint": r["fingerprint"], "state": r["state"], "legal_actions": [a for a in range(7) if legal[a]], "visits": r["visit_counts"], "policy_target": [round(p, 6) for p in r["policy_target"]], "top_action": r["selected_action"], "checks": checks})
    return {"sample_size": len(entries), "selection": "seeded hash order", "ALL_CHECKS_PASS": all(all(e["checks"].values()) for e in entries), "entries": entries}
