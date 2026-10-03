#!/usr/bin/env python3
"""Lot 36: qualification externe figée des deux G4 contre Minimax-Bidoua."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from songo_ai.evaluation import (
    ArenaConfig, HybridPolicyValueEvaluator, MinimaxBidouaReferenceAgent,
    MinimaxBidouaReferenceConfig, SRNMCTSAgent, game_result_to_dict,
    generate_unique_deterministic_openings, model_parameter_fingerprint,
    opening_to_dict, play_arena_game, summarize_arena,
)
from songo_ai.model import load_srn_checkpoint
from songo_ai.songo.rules import SongoLegacyGame
from run_srn_lot12 import sha256, write_json

OUT = Path("data/experiments/lot36_g4_vs_minimax")
IDENTITY = Path("data/experiments/lot35_generator_pool/g4_champion_identity.json")
EXPECTED_MINIMAX = "70f3b5632da10cfa9df4c6f28c19f8c8b8927d514b515d1c31984243d61e334c"
BUDGETS = (64, 128, 256)
GAMES = 512
OPENINGS = 256
WORKERS = 8
BASE_SEED = 20263600

_MODEL = _MINIMAX = _CONFIG = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "arena", "finalize"), required=True)
    p.add_argument("--model", choices=("control", "pool"))
    p.add_argument("--budget", type=int, choices=BUDGETS)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--workers", type=int, default=WORKERS)
    return p.parse_args()


def identities():
    return json.loads(IDENTITY.read_text())["candidates"]


def _architecture_fingerprint(model) -> str:
    import hashlib
    config = model.config
    payload = json.dumps(
        config.__dict__ if hasattr(config, "__dict__") else str(config),
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def exact_fingerprints():
    result = {}
    for key, item in identities().items():
        policy = load_srn_checkpoint(item["policy_checkpoint"]).model
        value = load_srn_checkpoint(item["value_checkpoint"]).model
        result[key] = {
            "model_id": item["model_id"],
            "policy_checkpoint": item["policy_checkpoint"],
            "value_checkpoint": item["value_checkpoint"],
            "policy_fingerprint": sha256(Path(item["policy_checkpoint"])),
            "value_fingerprint": sha256(Path(item["value_checkpoint"])),
            "architecture_fingerprint": _architecture_fingerprint(policy),
            "policy_parameter_fingerprint": model_parameter_fingerprint(policy),
            "value_parameter_fingerprint": model_parameter_fingerprint(value),
        }
    return result


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    reference = MinimaxBidouaReferenceConfig()
    if reference.fingerprint != EXPECTED_MINIMAX:
        raise RuntimeError("Minimax reference fingerprint mismatch")
    fps = exact_fingerprints()
    expected = identities()
    for key in ("CONTROL", "POOL"):
        if fps[key]["policy_fingerprint"] != expected[key]["policy_fingerprint"]:
            raise RuntimeError(f"{key} policy checkpoint changed")
        if fps[key]["value_fingerprint"] != expected[key]["value_fingerprint"]:
            raise RuntimeError(f"{key} value checkpoint changed")
        if fps[key]["architecture_fingerprint"] != expected[key]["architecture_fingerprint"]:
            raise RuntimeError(f"{key} architecture changed")
    config = {
        "lot": 36, "nature": "FINAL_EXTERNAL_QUALIFICATION", "models": ["CONTROL_G4R", "POOL_G4R"],
        "budgets": list(BUDGETS), "games_per_confrontation": GAMES, "opening_pairs": OPENINGS,
        "total_planned_games": 6 * GAMES, "paired": True, "side_swapped": True,
        "max_plies": 400, "repetition_limit": 3, "c_puct": 1.5,
        "workers_default": WORKERS, "no_budget_reduction": True, "no_posthoc_mcts512": True,
        "training_performed": False, "teacher_use": False, "minimax_labels": False,
        "g5_training_performed": False, "g5_massive_generation_performed": False,
        "registry_tiebreak": False, "created_before_results": True,
    }
    manifests = {}
    for budget in BUDGETS:
        seed = BASE_SEED + budget
        openings = generate_unique_deterministic_openings(count=OPENINGS, seed=seed, max_prefix_length=40)
        manifests[str(budget)] = {
            "openings_seed": seed, "arena_seed": seed + 10_000, "games": GAMES,
            "opening_pairs": OPENINGS, "shared_between_control_and_pool": True,
            "openings": [opening_to_dict(x) for x in openings],
        }
    write_json(args.output / "configuration.json", config)
    write_json(args.output / "model_fingerprints.json", {"before": fps, "after": None, "frozen": True})
    write_json(args.output / "minimax_reference.json", {"config": asdict(reference), "fingerprint": reference.fingerprint, "expected_fingerprint": EXPECTED_MINIMAX, "configuration_unchanged": True})
    write_json(args.output / "seed_manifests.json", manifests)
    print(json.dumps({"prepared": True, "planned_games": 3072, "minimax_fingerprint": reference.fingerprint}, indent=2))


def _worker_init(policy_path, value_path, model_name, budget, reference_data, arena_data):
    global _MODEL, _MINIMAX, _CONFIG
    torch.set_num_threads(1)
    evaluator = HybridPolicyValueEvaluator(load_srn_checkpoint(policy_path).model, load_srn_checkpoint(value_path).model, name=model_name)
    _MODEL = SRNMCTSAgent(model_name, evaluator, budget, c_puct=1.5)
    _MINIMAX = MinimaxBidouaReferenceAgent(MinimaxBidouaReferenceConfig(**reference_data))
    _CONFIG = ArenaConfig(**arena_data)


def _pair(opening):
    return (
        play_arena_game(opening, p1_agent=_MODEL, p2_agent=_MINIMAX, agent_a_name=_MODEL.name, agent_b_name=_MINIMAX.name, a_player=1, config=_CONFIG),
        play_arena_game(opening, p1_agent=_MINIMAX, p2_agent=_MODEL, agent_a_name=_MODEL.name, agent_b_name=_MINIMAX.name, a_player=2, config=_CONFIG),
    )


def _terminal_details(results, openings):
    opening_by_id = {x.opening_id: x for x in openings}
    buckets = defaultdict(list)
    causes = Counter()
    for result in results:
        game = SongoLegacyGame.from_state(opening_by_id[result.opening_id].state.to_engine_state())
        for action in result.action_sequence:
            game.play_local(action)
        outcome = result.a_outcome
        label = "win" if outcome == 1 else "draw" if outcome == .5 else "loss" if outcome == 0 else "truncated"
        cause = result.status.value
        causes[cause] += 1
        stores = list(game.board[14:16])
        board = list(game.board[:14])
        a_store = stores[result.a_player - 1]
        b_store = stores[1 if result.a_player == 1 else 0]
        buckets[label].append({"length": result.continuation_plies, "store_margin": a_store - b_store, "stores_final": stores, "territory_final": [sum(board[:7]), sum(board[7:])], "terminal_cause": cause})
    summary = {}
    for label, rows in buckets.items():
        summary[label] = {"games": len(rows), "mean_length": statistics.fmean(x["length"] for x in rows), "median_length": statistics.median(x["length"] for x in rows), "mean_store_margin": statistics.fmean(x["store_margin"] for x in rows), "terminal_causes": dict(Counter(x["terminal_cause"] for x in rows))}
    return {"by_outcome": summary, "terminal_cause_distribution": dict(causes)}


def arena(args):
    if args.model is None or args.budget is None:
        raise SystemExit("arena requires --model and --budget")
    key = args.model.upper(); item = identities()[key]
    manifests = json.loads((args.output / "seed_manifests.json").read_text())
    manifest = manifests[str(args.budget)]
    openings = generate_unique_deterministic_openings(count=OPENINGS, seed=manifest["openings_seed"], max_prefix_length=40)
    reference = MinimaxBidouaReferenceConfig()
    cfg = ArenaConfig(max_plies=400, repetition_limit=3, seed=manifest["arena_seed"], bootstrap_samples=20_000)
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=_worker_init, initargs=(item["policy_checkpoint"], item["value_checkpoint"], item["model_id"], args.budget, asdict(reference), asdict(cfg))) as executor:
        pairs = list(executor.map(_pair, openings))
    results = tuple(x for pair in pairs for x in pair)
    summary = asdict(summarize_arena(results, config=cfg))
    payload = {"model": item["model_id"], "budget": args.budget, "summary": summary, "terminal_analysis": _terminal_details(results, openings), "wall_time_s": time.perf_counter() - started, "games": [game_result_to_dict(x) for x in results]}
    write_json(args.output / f"{args.model}_mcts{args.budget}.json", payload)
    print(json.dumps({"artifact": f"{args.model}_mcts{args.budget}.json", "games": len(results), "score": summary["score_rate_a_terminal"], "ci95": summary["paired_bootstrap_ci"], "wall_time_s": payload["wall_time_s"]}, indent=2))


def _beats(payload):
    score = payload["summary"]["score_rate_a_terminal"]
    ci = payload["summary"]["paired_bootstrap_ci"]
    return bool(score is not None and score > .5 and ci and ci[0] > .5)


def finalize(args):
    data = {(m, b): json.loads((args.output / f"{m}_mcts{b}.json").read_text()) for m in ("control", "pool") for b in BUDGETS}
    after = exact_fingerprints(); fpdoc = json.loads((args.output / "model_fingerprints.json").read_text())
    fpdoc["after"] = after; fpdoc["unchanged"] = fpdoc["before"] == after
    if not fpdoc["unchanged"]: raise RuntimeError("G4 checkpoint or architecture mutated")
    write_json(args.output / "model_fingerprints.json", fpdoc)
    side = {}; scaling = {}; terminal = {}; decisions = {}
    for m in ("control", "pool"):
        scores=[]; beats=[]
        for b in BUDGETS:
            p=data[(m,b)]; s=p["summary"]; scores.append(s["score_rate_a_terminal"]); beats.append(_beats(p))
            side[f"{m}_mcts{b}"]={"P1":asdict_obj(s["by_a_side"]["P1"]),"P2":asdict_obj(s["by_a_side"]["P2"]),"score_P1":side_score(s["by_a_side"]["P1"]),"score_P2":side_score(s["by_a_side"]["P2"])}
            terminal[f"{m}_mcts{b}"]=p["terminal_analysis"]
        robust=(beats[0] and beats[1]) or (beats[1] and beats[2])
        scaling[m]={"scores":dict(zip(map(str,BUDGETS),scores)),"deltas":{"64_to_128":scores[1]-scores[0],"128_to_256":scores[2]-scores[1]},"strict_monotonicity_not_required":True}
        decisions[m]={"beats_by_budget":dict(zip(map(str,BUDGETS),beats)),"robustly_beats":robust}
    write_json(args.output/"side_analysis.json",side);write_json(args.output/"search_scaling.json",scaling);write_json(args.output/"terminal_analysis.json",terminal)
    best_key=max(data,key=lambda k:data[k]["summary"]["score_rate_a_terminal"]);best=data[best_key];best_score=best["summary"]["score_rate_a_terminal"]
    historical={"G2_score":.015625,"protocol":"Lot14: 128 games, MCTS64, same Minimax fingerprint","comparison":"indicative_historical_comparison_due_to_arena_size_difference","absolute_gain_percentage_points":100*(best_score-.015625)}
    write_json(args.output/"historical_comparison_g2.json",historical)
    robust=[decisions[m]["robustly_beats"] for m in ("control","pool")]
    generation="YES" if all(robust) else "PARTIAL" if any(robust) else "NO"
    control_scores=[data[("control",b)]["summary"]["score_rate_a_terminal"] for b in BUDGETS];pool_scores=[data[("pool",b)]["summary"]["score_rate_a_terminal"] for b in BUDGETS]
    differentiation="YES" if any(_beats(data[("control",b)]) != _beats(data[("pool",b)]) for b in BUDGETS) else "INCONCLUSIVE"
    allscores=control_scores+pool_scores
    scaling_status="HEALTHY" if all(x[-1]>=x[0]-.03 for x in (control_scores,pool_scores)) else "UNHEALTHY" if all(x[-1]<x[0]-.03 for x in (control_scores,pool_scores)) else "MIXED"
    qualification="STRONG" if generation=="YES" else "COMPETITIVE" if best["summary"]["paired_bootstrap_ci"][1]>=.5 or best_score>=.45 else "STILL_BEHIND"
    next_action="G4_PRODUCTION_AND_HUMAN_EXPERT_QUALIFICATION" if generation=="YES" else "G5_TARGETED_EVOLUTION_DESIGN" if qualification=="COMPETITIVE" else "MINIMAX_GAP_DIAGNOSIS_BEFORE_G5"
    verdict={"CONTROL":decisions["control"],"POOL":decisions["pool"],"G4_GENERATION_BEATS_MINIMAX":generation,"EXTERNAL_BENCHMARK_DIFFERENTIATION":differentiation,"SEARCH_SCALING_VS_MINIMAX":scaling_status,"BEST_G4_SCORE_VS_MINIMAX":best_score,"BEST_G4_CONFIGURATION":f"{best_key[0].upper()}_MCTS{best_key[1]}","BEST_G4_CI95":best["summary"]["paired_bootstrap_ci"],"BEST_G4_GAP_TO_50_PERCENTAGE_POINTS":100*(best_score-.5),"G4_EXTERNAL_QUALIFICATION":qualification,"OFFICIAL_G4_REPRESENTATION":"COCHAMPIONS_CONTROL_AND_POOL","G5_TRAINING_PERFORMED":"NO","G5_MASSIVE_DATA_GENERATION_PERFORMED":"NO","training_performed":False,"minimax_labels_used":False,"NEXT_ACTION":next_action}
    write_json(args.output/"external_benchmark_comparison.json",{"control":control_scores,"pool":pool_scores});write_json(args.output/"qualification_decision.json",verdict);write_json(args.output/"next_action.json",{"NEXT_ACTION":next_action});write_json(args.output/"report.json",{"lot":36,"status":"COMPLETE","total_games":3072,"model_fingerprints_unchanged":True,"minimax_fingerprint":EXPECTED_MINIMAX,"decisions":verdict,"historical":historical,"arenas":{f"{m}_mcts{b}":data[(m,b)]["summary"] for m in ("control","pool") for b in BUDGETS}})
    print(json.dumps(verdict,indent=2))


def asdict_obj(x): return x
def side_score(x):
    n=x["terminal_games"]
    return (x["wins"]+.5*x["draws"])/n if n else None


if __name__ == "__main__":
    args=parse_args()
    {"prepare":prepare,"arena":arena,"finalize":finalize}[args.stage](args)
