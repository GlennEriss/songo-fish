#!/usr/bin/env python3
"""Lot 41: etude de convergence Deep-MCTS, latence et cout compute.

Ce lot ne modifie ni le modele, ni le moteur, ni la semantique MCTS. Il
reutilise le pipeline optimise du Lot40 pour mesurer comment les conclusions
racine evoluent quand seul le budget MCTS augmente.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import statistics
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from songo_ai.dataset import RawSongoState
from songo_ai.evaluation import model_parameter_fingerprint
from songo_ai.evaluation.policy_target_diagnostics import jensen_shannon
from songo_ai.search import MCTSConfig, SongoMCTS, visit_counts_to_policy
from songo_ai.songo.rules import NUM_ACTIONS, SongoLegacyGame
from run_srn_colab_benchmark import env, git_commit, pool_fingerprints
from run_srn_lot12 import sha256, write_json
from run_srn_lot39 import SEED, fingerprint_state, load_model, load_positions, warmup

OUT = Path("data/experiments/lot41_deep_mcts_convergence")
DEFAULT_BUDGETS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
STAGE_BUDGETS = {
    "stage_a": (256, 512, 1024, 2048, 4096),
    "stage_b": (8192,),
    "stage_c": (16384,),
    "stage_d": (32768,),
    "stage_e": (65536,),
}
TARGET_POSITIONS = 256
CONCURRENCY = 256
LATENCY_POSITIONS = 32
ANALYSIS_TEMPERATURE = 1.0
LOT40_FLAGS = {
    "profile_runtime": False,
    "compact_tree_ops": True,
    "fast_engine_rebuild": True,
    "vectorized_graph": True,
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=("prepare", "pilot", *STAGE_BUDGETS, "latency", "finalize", "export"),
        required=True,
    )
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--positions-file", type=Path)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--bundle", type=Path, default=Path("lot41_results.tar.gz"))
    p.add_argument("--budgets", default=",".join(str(x) for x in DEFAULT_BUDGETS))
    p.add_argument("--positions-limit", type=int, default=TARGET_POSITIONS)
    p.add_argument("--latency-positions", type=int, default=LATENCY_POSITIONS)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if name == "cuda" or name == "auto" and torch.cuda.is_available() else "cpu")


def parse_budgets(raw: str) -> tuple[int, ...]:
    budgets = tuple(int(x) for x in raw.split(",") if x.strip())
    if not budgets or any(x <= 0 for x in budgets):
        raise ValueError("budgets must be positive")
    if tuple(sorted(budgets)) != budgets:
        raise ValueError("budgets must be sorted")
    return budgets


def positions_fingerprint(states: list[RawSongoState]) -> str:
    return hashlib.sha256("".join(fingerprint_state(s) for s in states).encode()).hexdigest()


def stage_path(out: Path, budget: int) -> Path:
    return out / "search" / f"budget_{budget}.json"


def checksum_payload(path: Path) -> dict:
    return {"sha256": sha256(path), "size": path.stat().st_size}


def write_checked(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    write_json(path.with_suffix(path.suffix + ".checksum.json"), checksum_payload(path))


def checked(path: Path) -> bool:
    side = path.with_suffix(path.suffix + ".checksum.json")
    if not path.is_file() or not side.is_file():
        return False
    try:
        return json.load(side.open()).get("sha256") == sha256(path)
    except Exception:
        return False


def load_benchmark(out: Path, limit: int | None = None) -> tuple[dict, list[RawSongoState]]:
    meta = json.load((out / "benchmark_positions.json").open())
    states = [
        RawSongoState(tuple(x["board"]), int(x["player_to_move"]))
        for x in meta["states"]
    ]
    if limit is not None:
        states = states[:limit]
    return meta, states


def prepare(a: argparse.Namespace) -> None:
    if not a.positions_file:
        raise ValueError("prepare requires --positions-file")
    source, states = load_positions(a.positions_file)
    states = states[: a.positions_limit]
    if len(states) != a.positions_limit:
        raise RuntimeError("not enough benchmark positions")
    if len({fingerprint_state(s) for s in states}) != len(states):
        raise RuntimeError("Lot41 requires unique physical positions")
    for state in states:
        game = SongoLegacyGame.from_state(state.to_engine_state())
        game.normalize_terminal()
        if game.finished or not game.legal_local_actions():
            raise RuntimeError("invalid benchmark position")
    a.output.mkdir(parents=True, exist_ok=True)
    positions = {
        "count": len(states),
        "seed": SEED,
        "fingerprint": positions_fingerprint(states),
        "provenance": source.get("provenance", {}),
        "labels_used": False,
        "selection": "deterministic prefix of validated Lot39/Lot40 benchmark positions",
        "states": [
            {
                "index": i,
                "board": list(s.board),
                "player_to_move": s.player_to_move,
                "state_fingerprint": fingerprint_state(s),
            }
            for i, s in enumerate(states)
        ],
    }
    write_json(a.output / "benchmark_positions.json", positions)
    d = choose_device(a.device)
    write_json(a.output / "environment.json", env(d))
    write_json(
        a.output / "configuration.json",
        {
            "lot": 41,
            "model": "POOL_G4R",
            "budgets": list(parse_budgets(a.budgets)),
            "positions": len(states),
            "parallel_searches": CONCURRENCY,
            "analysis_temperature": ANALYSIS_TEMPERATURE,
            "dirichlet_for_convergence": "OFF",
            "root_noise": False,
            "training_performed": False,
            "optimizer_created": False,
            "backward_called": False,
            "teacher_labels": False,
            "minimax_labels": False,
            "lot40_flags": LOT40_FLAGS,
        },
    )
    write_json(a.output / "fingerprints.json", {"before": {"model": pool_fingerprints()}, "after": None})
    write_json(a.output / "stage_state.json", {"stages": {s: "PENDING" for s in STAGE_BUDGETS}})


def result_row(index: int, state: RawSongoState, budget: int, result) -> dict:
    return {
        "index": index,
        "state_fingerprint": fingerprint_state(state),
        "budget": budget,
        "legal_mask": list(result.legal_mask),
        "visit_counts": list(result.visit_counts),
        "visit_distribution": list(result.policy),
        "selected_action": result.selected_action,
        "root_q_values": list(result.root_q_values),
        "root_value": result.root_value,
        "policy_prior": list(result.root_priors),
        "search_runtime_s": result.elapsed_s,
        "nodes": result.num_nodes,
        "network_evaluations": result.network_evaluations,
        "network_calls": None,
        "effective_batch_mean": None,
        "termination_status": "COMPLETE",
        "seed": SEED + index,
        "target_temperature": ANALYSIS_TEMPERATURE,
    }


def run_budget(a: argparse.Namespace, budget: int, states: list[RawSongoState], d: torch.device) -> dict:
    path = stage_path(a.output, budget)
    if checked(path):
        return json.load(path.open())
    model = load_model(d)
    before = model_parameter_fingerprint(model)
    warmup(model, d, states[: min(len(states), CONCURRENCY)])
    if d.type == "cuda":
        torch.cuda.reset_peak_memory_stats(d)
    search = SongoMCTS(
        model,
        config=MCTSConfig(num_simulations=budget, c_puct=1.5, add_root_noise=False, seed=SEED),
        **LOT40_FLAGS,
    )
    started = time.perf_counter()
    cpu_started = time.process_time()
    with torch.inference_mode():
        results = search.search_many(states, policy_temperature=ANALYSIS_TEMPERATURE, seeds=[SEED + i for i in range(len(states))])
    wall = time.perf_counter() - started
    cpu_time = time.process_time() - cpu_started
    after = model_parameter_fingerprint(model)
    rows = [result_row(i, s, budget, r) for i, (s, r) in enumerate(zip(states, results))]
    batches = [int(x) for x in search.last_profile.get("effective_batch_sizes", [])]
    for row in rows:
        row["network_calls"] = len(batches)
        row["effective_batch_mean"] = statistics.fmean(batches) if batches else 0.0
    payload = {
        "status": "COMPLETE",
        "budget": budget,
        "git_commit": git_commit(),
        "device": str(d),
        "positions": len(states),
        "parallel_searches": len(states),
        "total_simulations": sum(r.num_simulations for r in results),
        "wall_time_s": wall,
        "global_simulations_per_second": sum(r.num_simulations for r in results) / wall,
        "positions_per_second": len(states) / wall,
        "network_calls": len(batches),
        "network_evaluations": sum(r.network_evaluations for r in results),
        "mean_batch": statistics.fmean(batches) if batches else 0.0,
        "median_batch": statistics.median(batches) if batches else 0.0,
        "nodes": sum(r.num_nodes for r in results),
        "mean_nodes_per_tree": statistics.fmean(r.num_nodes for r in results),
        "max_nodes_per_tree": max(r.num_nodes for r in results),
        "nodes_per_simulation": sum(r.num_nodes for r in results) / max(1, sum(r.num_simulations for r in results)),
        "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_gpu_memory": torch.cuda.max_memory_allocated(d) if d.type == "cuda" else None,
        "process_cpu_time_s": cpu_time,
        "model_weights_changed": before != after,
        "model_fingerprints": pool_fingerprints(),
        "rows": rows,
    }
    write_checked(path, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2), flush=True)
    return payload


def run_search_stage(a: argparse.Namespace, stage: str) -> None:
    wanted = set(parse_budgets(a.budgets))
    if stage == "stage_a":
        budgets = tuple(b for b in sorted(wanted) if b <= 4096)
    else:
        budgets = tuple(b for b in STAGE_BUDGETS[stage] if b in wanted)
    if not budgets:
        print(f"SKIP {stage}: no requested budget")
        return
    d = choose_device(a.device)
    _, states = load_benchmark(a.output, a.positions_limit)
    for budget in budgets:
        run_budget(a, budget, states, d)
    state = json.load((a.output / "stage_state.json").open()) if (a.output / "stage_state.json").is_file() else {"stages": {}}
    state["stages"][stage] = "COMPLETE"
    write_json(a.output / "stage_state.json", state)


def pilot(a: argparse.Namespace) -> None:
    d = choose_device(a.device)
    _, states = load_benchmark(a.output, min(16, a.positions_limit))
    budgets = tuple(b for b in parse_budgets(a.budgets) if b in (4096, 8192, 16384))
    if not budgets:
        budgets = (parse_budgets(a.budgets)[-1],)
    estimates = []
    for budget in budgets:
        model = load_model(d)
        search = SongoMCTS(model, config=MCTSConfig(num_simulations=budget, c_puct=1.5, add_root_noise=False, seed=SEED), **LOT40_FLAGS)
        t = time.perf_counter()
        with torch.inference_mode():
            result = search.search_many(states, policy_temperature=ANALYSIS_TEMPERATURE, seeds=[SEED + i for i in range(len(states))])
        elapsed = time.perf_counter() - t
        sims = sum(x.num_simulations for x in result)
        estimates.append({"budget": budget, "sample_positions": len(states), "elapsed_s": elapsed, "simulations_per_second": sims / elapsed, "estimated_256_positions_s": elapsed * TARGET_POSITIONS / len(states)})
    write_json(a.output / "estimated_total_compute.json", {"status": "COMPLETE", "estimates": estimates})
    print(json.dumps(estimates, indent=2), flush=True)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "p75": None, "p90": None, "p95": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def ranking(policy: list[float], legal: list[bool]) -> list[int]:
    return sorted([a for a in range(NUM_ACTIONS) if legal[a]], key=lambda a: (-policy[a], a))


def kendall_distance(first: list[int], second: list[int]) -> float:
    pairs = 0
    discordant = 0
    rank_second = {a: i for i, a in enumerate(second)}
    for i, left in enumerate(first):
        for right in first[i + 1 :]:
            pairs += 1
            discordant += int(rank_second[left] > rank_second[right])
    return discordant / pairs if pairs else 0.0


def margin(policy: list[float], legal: list[bool]) -> float:
    values = sorted((policy[a] for a in range(NUM_ACTIONS) if legal[a]), reverse=True)
    return values[0] - values[1] if len(values) > 1 else 1.0


def convergence(rows_by_budget: dict[int, list[dict]]) -> dict:
    budgets = sorted(rows_by_budget)
    by_fp = {b: {r["state_fingerprint"]: r for r in rows} for b, rows in rows_by_budget.items()}
    adjacent = []
    for left, right in zip(budgets, budgets[1:]):
        common = sorted(set(by_fp[left]) & set(by_fp[right]))
        js_values = []
        agreements = []
        ranks = []
        q_deltas = []
        margin_deltas = []
        flips = []
        for fp in common:
            a = by_fp[left][fp]
            b = by_fp[right][fp]
            js_values.append(jensen_shannon(a["visit_distribution"], b["visit_distribution"]))
            agreements.append(a["selected_action"] == b["selected_action"])
            ranks.append(kendall_distance(ranking(a["visit_distribution"], a["legal_mask"]), ranking(b["visit_distribution"], b["legal_mask"])))
            legal = [bool(x) for x in a["legal_mask"]]
            q_deltas.extend(abs(a["root_q_values"][i] - b["root_q_values"][i]) for i in range(NUM_ACTIONS) if legal[i])
            margin_deltas.append(abs(margin(a["visit_distribution"], legal) - margin(b["visit_distribution"], legal)))
            flips.append(a["selected_action"] != b["selected_action"])
        stable = statistics.fmean(agreements) >= 0.95 and summarize(js_values)["median"] <= 0.02 and summarize(js_values)["p90"] <= 0.05
        adjacent.append({"from_budget": left, "to_budget": right, "positions": len(common), "top1_agreement": statistics.fmean(agreements) if agreements else None, "js": summarize(js_values), "kendall_distance": summarize(ranks), "root_q_abs_delta": summarize(q_deltas), "best_action_margin_delta": summarize(margin_deltas), "action_flips": sum(flips), "practically_stable": stable})
    action_paths = []
    for fp in sorted(set.intersection(*(set(x) for x in by_fp.values())) if by_fp else set()):
        actions = [by_fp[b][fp]["selected_action"] for b in budgets]
        flips = sum(actions[i] != actions[i - 1] for i in range(1, len(actions)))
        stable_budget = None
        for i, budget in enumerate(budgets):
            if all(action == actions[i] for action in actions[i:]):
                stable_budget = budget
                break
        action_paths.append({"state_fingerprint": fp, "actions": dict(zip(map(str, budgets), actions)), "flips": flips, "action_stability_budget": stable_budget, "has_reversal": any(actions[i] == actions[i + 2] and actions[i] != actions[i + 1] for i in range(len(actions) - 2))})
    first_robust = "NOT_DETECTED"
    for i in range(len(adjacent) - 1):
        if adjacent[i]["practically_stable"] and adjacent[i + 1]["practically_stable"]:
            first_robust = adjacent[i]["from_budget"]
            break
    if budgets and adjacent and not any(x["practically_stable"] for x in adjacent):
        first_robust = "ABOVE_65536" if budgets[-1] >= 65536 else "NOT_DETECTED"
    return {"budgets": budgets, "adjacent": adjacent, "action_paths": action_paths, "empirical_convergence_budget": first_robust}


def latency(a: argparse.Namespace) -> None:
    budgets = parse_budgets(a.budgets)
    _, states = load_benchmark(a.output, min(a.latency_positions, a.positions_limit))
    rows = []
    for requested in ("cpu", "cuda"):
        if requested == "cuda" and not torch.cuda.is_available():
            continue
        d = torch.device(requested)
        for budget in budgets:
            latencies = []
            actions = []
            model = load_model(d)
            for i, state in enumerate(states):
                search = SongoMCTS(model, config=MCTSConfig(num_simulations=budget, c_puct=1.5, add_root_noise=False, seed=SEED + i), **LOT40_FLAGS)
                t = time.perf_counter()
                with torch.inference_mode():
                    result = search.search_many([state], policy_temperature=ANALYSIS_TEMPERATURE, seeds=[SEED + i])[0]
                latencies.append(time.perf_counter() - t)
                actions.append(result.selected_action)
            rows.append({"device": requested, "budget": budget, "positions": len(states), "mean_latency_s": statistics.fmean(latencies), "median_latency_s": statistics.median(latencies), "p95_latency_s": float(np.percentile(latencies, 95)), "selected_actions": actions})
            print(json.dumps({k: v for k, v in rows[-1].items() if k != "selected_actions"}, indent=2), flush=True)
    write_checked(a.output / "latency.json", {"status": "COMPLETE", "rows": rows})


def finalize(a: argparse.Namespace) -> None:
    budgets = [b for b in parse_budgets(a.budgets) if checked(stage_path(a.output, b))]
    if not budgets:
        raise RuntimeError("no completed budget")
    data = {b: json.load(stage_path(a.output, b).open()) for b in budgets}
    rows_by_budget = {b: data[b]["rows"] for b in budgets}
    conv = convergence(rows_by_budget)
    write_json(a.output / "convergence_metrics.json", conv)
    perf_rows = []
    for b in budgets:
        item = data[b]
        perf_rows.append({k: item[k] for k in ("budget", "positions", "total_simulations", "wall_time_s", "global_simulations_per_second", "mean_nodes_per_tree", "max_nodes_per_tree", "nodes_per_simulation", "peak_ram_bytes", "peak_gpu_memory", "model_weights_changed")})
    with (a.output / "search_efficiency_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=perf_rows[0].keys())
        writer.writeheader()
        writer.writerows(perf_rows)
    model_changed = any(data[b]["model_weights_changed"] for b in budgets)
    max_budget = max(budgets)
    recommended_teacher = conv["empirical_convergence_budget"]
    if recommended_teacher in ("NOT_DETECTED", "ABOVE_65536"):
        recommended_teacher = max_budget
    decision = {
        "LOT41_VALID": "YES",
        "TRAINING_PERFORMED": "NO",
        "OPTIMIZER_CREATED": "NO",
        "BACKWARD_CALLED": "NO",
        "MODEL_WEIGHTS_CHANGED": "YES" if model_changed else "NO",
        "DIRICHLET_FOR_CONVERGENCE": "OFF",
        "ANALYSIS_TEMPERATURE": ANALYSIS_TEMPERATURE,
        "MAX_COMPLETED_BUDGET": max_budget,
        "EMPIRICAL_CONVERGENCE_BUDGET": conv["empirical_convergence_budget"],
        "RECOMMENDED_TEACHER_MCTS_BUDGET": recommended_teacher,
        "COLAB_INTERACTIVE_BUDGET": min((b for b in budgets if data[b]["wall_time_s"] <= 60), default=min(budgets)),
        "COLAB_INTERACTIVE_BUDGET_MOBILE_VALIDATED": "NO",
        "MCTS131072_RECOMMENDED": "YES" if max_budget >= 65536 and conv["empirical_convergence_budget"] == "ABOVE_65536" else "NO" if max_budget >= 65536 else "INCONCLUSIVE",
        "NEXT_ACTION": "DESIGN_G5_DEEP_TARGET_GENERATION" if max_budget >= 16384 else "CONTINUE_LOT41_DEEP_STAGES",
    }
    write_json(a.output / "decision.json", decision)
    write_json(a.output / "report.json", {"lot": 41, "decision": decision, "performance": perf_rows, "convergence": conv})
    fp = json.load((a.output / "fingerprints.json").open()) if (a.output / "fingerprints.json").is_file() else {"before": None}
    fp["after"] = {"model": pool_fingerprints()}
    fp["unchanged"] = fp["before"] == fp["after"] if fp.get("before") else None
    write_json(a.output / "fingerprints.json", fp)
    print(json.dumps(decision, indent=2), flush=True)


def export(a: argparse.Namespace) -> None:
    if not (a.output / "decision.json").is_file():
        raise RuntimeError("finalize Lot41 first")
    files = [p for p in a.output.rglob("*") if p.is_file() and p.name not in ("experiment_manifest.json", "checksums.json")]
    checks = {str(p.relative_to(a.output)): sha256(p) for p in files}
    write_json(a.output / "checksums.json", checks)
    files.append(a.output / "checksums.json")
    write_json(a.output / "experiment_manifest.json", {"lot": 41, "git_commit": git_commit(), "model_fingerprints": pool_fingerprints(), "artifact_checksums": {str(p.relative_to(a.output)): sha256(p) for p in files}})
    files.append(a.output / "experiment_manifest.json")
    with tarfile.open(a.bundle, "w:gz") as tf:
        for p in files:
            tf.add(p, arcname=f"lot41_deep_mcts_convergence/{p.relative_to(a.output)}")
    Path(str(a.bundle) + ".sha256").write_text(f"{sha256(a.bundle)}  {a.bundle.name}\n")


if __name__ == "__main__":
    a = args()
    if a.stage == "prepare":
        prepare(a)
    elif a.stage == "pilot":
        pilot(a)
    elif a.stage in STAGE_BUDGETS:
        run_search_stage(a, a.stage)
    elif a.stage == "latency":
        latency(a)
    elif a.stage == "finalize":
        finalize(a)
    elif a.stage == "export":
        export(a)
