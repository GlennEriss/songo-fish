"""Comptabilite RESTART (mesuree) / RESUME (contrefactuelle) - sections 72-79.

RESTART : chaque budget est une recherche independante depuis la racine ; le
cout est la somme des simulations reellement executees et enregistrees.
RESUME : l'arbre l3 serait prolonge jusqu'a ref. ``SongoMCTS.search`` et
``search_many`` reconstruisent toujours la racine (``_root_node``) et ne
renvoient pas l'arbre : la reprise n'existe pas, le cout est contrefactuel.
"""
from __future__ import annotations

import inspect

import numpy as np

from songo_ai.search import SongoMCTS

from .config import PROTOCOL


def mcts_resume_audit() -> dict:
    """Audit du code reel : aucun parametre d'arbre/racine en entree ni en sortie."""

    search_params = list(inspect.signature(SongoMCTS.search).parameters)
    many_params = list(inspect.signature(SongoMCTS.search_many).parameters)
    many_source = inspect.getsource(SongoMCTS.search_many)
    accepts_tree = any(p in ("root", "tree", "roots", "trees", "resume_from") for p in search_params + many_params)
    rebuilds_root = "self._root_node(state)" in many_source
    return {
        "search_parameters": search_params,
        "search_many_parameters": many_params,
        "accepts_existing_tree": accepts_tree,
        "search_many_rebuilds_root_each_call": rebuilds_root,
        "MCTS_RESUME_CURRENTLY_SUPPORTED": "YES" if accepts_tree and not rebuilds_root else "PARTIAL" if accepts_tree else "NO",
        "lot44_modified_mcts": False,
    }


def strategy_costs(sims: dict[str, np.ndarray], routed: np.ndarray, *, base: str) -> dict:
    """Couts totaux. ``base`` : ``l3`` (direct), ``ladder`` (l1->l2->l3) ou ``ref`` (direct)."""

    l3, ref = sims["l3"], sims["ref"]
    if base == "ref":
        return {"restart": float(ref.sum()), "resume": float(ref.sum())}
    base_restart = (sims["l1"] + sims["l2"] + l3) if base == "ladder" else l3
    restart = float(base_restart.sum() + ref[routed].sum())
    resume = float(l3.sum() + (ref[routed] - l3[routed]).sum())
    return {"restart": restart, "resume": resume}


def evaluate_strategy(name: str, routed: np.ndarray, sims: dict[str, np.ndarray], labels: list[dict], *, base: str, resume_semantics: str) -> dict:
    y = np.asarray([r["label"] for r in labels], dtype=bool)
    costs = strategy_costs(sims, routed, base=base)
    misses = [r for r, hit, pos in zip(labels, routed, y) if pos and not hit]
    regrets = [r["deeper_regret"] for r in misses if r["deeper_regret"] is not None]
    n = len(labels)
    return {
        "strategy": name,
        "positions": n,
        "routed_to_ref": int(routed.sum()),
        "routing_rate": float(routed.mean()) if n else None,
        "positives": int(y.sum()),
        "positives_resolved": int((routed & y).sum()),
        "recall": float((routed & y).sum() / y.sum()) if y.sum() else None,
        "false_negatives": len(misses),
        "high_severity_false_negatives": sum(bool(r["high_severity"]) for r in misses),
        "worst_false_negative_regret": max(regrets) if regrets else None,
        "restart_total_simulations": costs["restart"],
        "restart_semantics": "MEASURED",
        "resume_total_simulations": costs["resume"],
        "resume_semantics": resume_semantics,
        "restart_per_position": costs["restart"] / n if n else None,
        "resume_per_position": costs["resume"] / n if n else None,
    }


def compute_accounting(sims: dict[str, np.ndarray], labels: list[dict], routes: dict[str, np.ndarray]) -> dict:
    n = len(labels)
    none = np.zeros(n, dtype=bool)
    every = np.ones(n, dtype=bool)
    oracle = np.asarray([bool(r["label"]) for r in labels], dtype=bool)
    strategies = {
        "UNIFORM_32768": evaluate_strategy("UNIFORM_32768", none, sims, labels, base="l3", resume_semantics="MEASURED"),
        "LADDER_4096_8192_32768": evaluate_strategy("LADDER_4096_8192_32768", none, sims, labels, base="ladder", resume_semantics="COUNTERFACTUAL_ESTIMATE"),
        "CONSERVATIVE_ROUTER": evaluate_strategy("CONSERVATIVE_ROUTER", routes["conservative"], sims, labels, base="ladder", resume_semantics="COUNTERFACTUAL_ESTIMATE"),
        "CALIBRATED_ROUTER_32768_TO_65536": evaluate_strategy("CALIBRATED_ROUTER_32768_TO_65536", routes["classifier"], sims, labels, base="ladder", resume_semantics="COUNTERFACTUAL_ESTIMATE"),
        "UNIFORM_65536": evaluate_strategy("UNIFORM_65536", every, sims, labels, base="ref", resume_semantics="MEASURED"),
        "ORACLE_ROUTER_UNATTAINABLE": evaluate_strategy("ORACLE_ROUTER_UNATTAINABLE", oracle, sims, labels, base="ladder", resume_semantics="COUNTERFACTUAL_ESTIMATE"),
    }
    uniform_ref = strategies["UNIFORM_65536"]["restart_total_simulations"]
    uniform_l3 = strategies["UNIFORM_32768"]["restart_total_simulations"]
    for item in strategies.values():
        item["restart_vs_uniform_32768"] = item["restart_total_simulations"] / uniform_l3 if uniform_l3 else None
        item["restart_vs_uniform_65536"] = item["restart_total_simulations"] / uniform_ref if uniform_ref else None
        item["resume_vs_uniform_65536"] = item["resume_total_simulations"] / uniform_ref if uniform_ref else None
    return {"strategies": strategies, "viability_cost_ratio": PROTOCOL["viability_cost_ratio"]}
