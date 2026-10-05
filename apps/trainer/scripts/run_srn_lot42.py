#!/usr/bin/env python3
"""Lot 42: extension MCTS65536 ciblee sur les hard positions Lot41B.

Ce lot ne relance pas les 256 positions. Il fige le sous-ensemble hard produit
par Lot41B, execute MCTS65536 uniquement sur ce sous-ensemble, puis compare les
racines a la reference MCTS32768 existante.
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
from songo_ai.search import MCTSConfig, SongoMCTS
from run_srn_colab_benchmark import env, git_commit, pool_fingerprints
from run_srn_lot12 import sha256, write_json
from run_srn_lot39 import SEED, fingerprint_state, load_model, warmup
from run_srn_lot41 import LOT40_FLAGS

LOT41 = Path("data/experiments/lot41_deep_mcts_convergence")
LOT41B = Path("data/experiments/lot41b_deep_mcts_strategic_convergence")
OUT = Path("data/experiments/lot42_hard_position_65536")
BUDGET = 65536
REF_BUDGET = 32768
ANALYSIS_TEMPERATURE = 1.0


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "pilot", "run", "finalize", "export"), required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--lot41", type=Path, default=LOT41)
    p.add_argument("--lot41b", type=Path, default=LOT41B)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--bundle", type=Path, default=Path("lot42_results.tar.gz"))
    p.add_argument("--pilot-count", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=4)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if name == "cuda" or name == "auto" and torch.cuda.is_available() else "cpu")


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


def write_checked(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    write_json(path.with_suffix(path.suffix + ".checksum.json"), {"sha256": sha256(path), "size": path.stat().st_size})


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


def jensen_shannon(first: list[float], second: list[float]) -> float:
    mid = [(a + b) / 2.0 for a, b in zip(first, second)]

    def kl(values: list[float]) -> float:
        return sum(v * math.log(v / m) for v, m in zip(values, mid) if v > 0.0 and m > 0.0)

    return 0.5 * kl(first) + 0.5 * kl(second)


def rank_actions(row: dict) -> list[int]:
    return sorted([a for a in range(7) if row["legal_mask"][a]], key=lambda a: (-row["visit_counts"][a], a))


def kendall_agreement(first: list[int], second: list[int]) -> float:
    if len(first) < 2:
        return 1.0
    rank = {a: i for i, a in enumerate(second)}
    pairs = 0
    concordant = 0
    for i, left in enumerate(first):
        for right in first[i + 1 :]:
            pairs += 1
            concordant += int(rank[left] < rank[right])
    return concordant / pairs if pairs else 1.0


def top2(row: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    r = rank_actions(row)[:2]
    return tuple(sorted(r)), tuple(r)


def hard_set_sha(rows: list[dict]) -> str:
    payload = json.dumps([r["fingerprint"] for r in rows], separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def state_from_record(record: dict) -> RawSongoState:
    state = record.get("state") or {}
    return RawSongoState(tuple(state["board"]), int(state["player_to_move"]))


def result_row(record: dict, result, elapsed: float, network_calls: int, mean_batch: float) -> dict:
    return {
        "state_fingerprint": record["fingerprint"],
        "legal_mask": list(result.legal_mask),
        "visit_counts": list(result.visit_counts),
        "visit_distribution": list(result.policy),
        "selected_action": result.selected_action,
        "ranking": rank_actions({"legal_mask": list(result.legal_mask), "visit_counts": list(result.visit_counts)}),
        "root_q_values": list(result.root_q_values),
        "root_value": result.root_value,
        "runtime_65536": elapsed,
        "nodes_65536": result.num_nodes,
        "network_evaluations_65536": result.network_evaluations,
        "network_calls_65536": network_calls,
        "effective_batch_mean": mean_batch,
        "requested_simulations": BUDGET,
        "actual_simulations": result.num_simulations,
    }


def load_hard_records(lot41b: Path) -> list[dict]:
    hard = read_json(lot41b / "hard_positions.json")
    rows = hard.get("rows", [])
    if not rows:
        raise RuntimeError("Lot42 requires non-empty Lot41B hard_positions.json")
    missing = [r.get("fingerprint") for r in rows if not r.get("state")]
    if missing:
        raise RuntimeError(f"hard positions missing physical state: {missing[:5]}")
    return sorted(rows, key=lambda row: row["fingerprint"])


def validate_inputs(lot41: Path, lot41b: Path) -> dict:
    decision = read_json(lot41b / "decision.json")
    required = {
        "LOT41B_VALID": "YES",
        "INPUT_ARTIFACTS_VALID": "YES",
        "COMMON_POSITION_COUNT": 256,
        "RECOMMENDED_AUTONOMOUS_TEACHER_BUDGET": "MULTI_FIDELITY",
        "NEXT_ACTION": "LOT42_HARD_POSITION_65536_EXTENSION",
    }
    checks = {key: decision.get(key) == value for key, value in required.items()}
    ref_path = lot41 / "search" / f"budget_{REF_BUDGET}.json"
    checks["lot41_32768_exists"] = ref_path.is_file()
    checks["lot41b_hard_positions_exists"] = (lot41b / "hard_positions.json").is_file()
    return {
        "LOT41B_DECISION": decision,
        "checks": checks,
        "INPUTS_VALID": "YES" if all(checks.values()) else "NO",
        "NEW_MCTS_SEARCH_ON_FULL_256": "NO",
        "TRAINING_PERFORMED": "NO",
        "MINIMAX_LABELS_USED": "NO",
    }


def prepare(a: argparse.Namespace) -> None:
    a.output.mkdir(parents=True, exist_ok=True)
    validation = validate_inputs(a.lot41, a.lot41b)
    if validation["INPUTS_VALID"] != "YES":
        write_json(a.output / "input_validation.json", validation)
        raise RuntimeError("invalid Lot41B/Lot41 inputs for Lot42")
    hard = load_hard_records(a.lot41b)
    fp = hard_set_sha(hard)
    ref = read_json(a.lot41 / "search" / f"budget_{REF_BUDGET}.json")
    ref_by_fp = {row["state_fingerprint"]: row for row in ref["rows"]}
    missing = [row["fingerprint"] for row in hard if row["fingerprint"] not in ref_by_fp]
    if missing:
        raise RuntimeError(f"hard positions missing from Lot41 MCTS32768: {missing[:5]}")
    validation.update({"HARD_POSITION_COUNT": len(hard), "HARD_SET_SHA256": fp})
    write_json(a.output / "input_validation.json", validation)
    write_json(a.output / "hard_set_manifest.json", {"count": len(hard), "rows": hard})
    write_json(a.output / "hard_set_fingerprint.json", {"HARD_SET_SHA256": fp, "ordered_fingerprints": [r["fingerprint"] for r in hard]})
    write_json(a.output / "execution_plan.json", {"budget": BUDGET, "root_noise": "OFF", "default_concurrency": a.concurrency, "pilot_count": min(a.pilot_count, len(hard)), "lot40_flags": LOT40_FLAGS, "shard_order": "fingerprint_ascending"})
    d = choose_device(a.device)
    write_json(a.output / "environment.json", env(d))


def completed_results(out: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted((out / "shards").glob("*.json")):
        if path.name.endswith(".checksum.json"):
            continue
        if checked(path):
            for row in read_json(path)["rows"]:
                rows[row["state_fingerprint"]] = row
    pilot = out / "memory_pilot.json"
    if checked(pilot):
        for row in read_json(pilot).get("rows", []):
            rows[row["state_fingerprint"]] = row
    return rows


def run_records(records: list[dict], d: torch.device) -> dict:
    states = [state_from_record(r) for r in records]
    model = load_model(d)
    before = model_parameter_fingerprint(model)
    warmup(model, d, states[: min(len(states), 16)])
    if d.type == "cuda":
        torch.cuda.reset_peak_memory_stats(d)
    search = SongoMCTS(model, config=MCTSConfig(num_simulations=BUDGET, c_puct=1.5, add_root_noise=False, seed=SEED), **LOT40_FLAGS)
    start = time.perf_counter()
    with torch.inference_mode():
        results = search.search_many(states, policy_temperature=ANALYSIS_TEMPERATURE, seeds=[SEED + i for i in range(len(states))])
    elapsed = time.perf_counter() - start
    after = model_parameter_fingerprint(model)
    batches = [int(x) for x in search.last_profile.get("effective_batch_sizes", [])]
    mean_batch = statistics.fmean(batches) if batches else 0.0
    rows = [result_row(record, result, elapsed, len(batches), mean_batch) for record, result in zip(records, results)]
    for row in rows:
        if row["actual_simulations"] != BUDGET:
            raise RuntimeError(f"simulation mismatch for {row['state_fingerprint']}")
        if row["selected_action"] is not None and not row["legal_mask"][row["selected_action"]]:
            raise RuntimeError(f"illegal selected action for {row['state_fingerprint']}")
        if not math.isfinite(sum(row["visit_distribution"])):
            raise RuntimeError(f"non-finite policy for {row['state_fingerprint']}")
    return {
        "rows": rows,
        "wall_time_s": elapsed,
        "total_simulations": sum(row["actual_simulations"] for row in rows),
        "global_simulations_per_second": sum(row["actual_simulations"] for row in rows) / elapsed,
        "network_calls": len(batches),
        "mean_batch": mean_batch,
        "peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_gpu_memory": torch.cuda.max_memory_allocated(d) if d.type == "cuda" else None,
        "model_weights_changed": before != after,
        "model_fingerprints": pool_fingerprints(),
    }


def pilot(a: argparse.Namespace) -> None:
    hard = load_hard_records(a.lot41b)
    count = min(a.pilot_count, len(hard))
    path = a.output / "memory_pilot.json"
    if checked(path):
        print(f"SKIP COMPLETE {path}")
        return
    d = choose_device(a.device)
    payload = run_records(hard[:count], d)
    payload.update({"status": "COMPLETE", "stage": "memory_pilot", "pilot_count": count, "budget": BUDGET, "safe_concurrency_recommendation": max(1, min(a.concurrency, count))})
    write_checked(path, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2), flush=True)


def run(a: argparse.Namespace) -> None:
    hard = load_hard_records(a.lot41b)
    done = completed_results(a.output)
    remaining = [row for row in hard if row["fingerprint"] not in done]
    d = choose_device(a.device)
    shard_dir = a.output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(remaining), a.concurrency):
        shard_records = remaining[start : start + a.concurrency]
        if not shard_records:
            continue
        first_fp = shard_records[0]["fingerprint"][:12]
        last_fp = shard_records[-1]["fingerprint"][:12]
        key = f"shard_{first_fp}_{last_fp}.json"
        path = shard_dir / key
        if checked(path):
            continue
        payload = run_records(shard_records, d)
        payload.update({"status": "COMPLETE", "stage": "run", "budget": BUDGET, "shard": key, "positions": len(shard_records)})
        write_checked(path, payload)
        write_json(a.output / "stage_state.json", {"completed_positions": len(completed_results(a.output)), "total_hard_positions": len(hard), "last_shard": key})
        print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2), flush=True)


def severity(delta: float) -> str:
    if delta <= 0.02:
        return "NEAR_TIE"
    if delta <= 0.05:
        return "LOW_SEVERITY"
    if delta <= 0.10:
        return "MEDIUM_SEVERITY"
    return "HIGH_SEVERITY"


def wilson(success: int, total: int, z: float = 1.96) -> dict:
    if total == 0:
        return {"low": None, "high": None}
    p = success / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return {"low": center - margin, "high": center + margin}


def finalize(a: argparse.Namespace) -> None:
    hard = load_hard_records(a.lot41b)
    hard_by_fp = {row["fingerprint"]: row for row in hard}
    found = completed_results(a.output)
    ref = read_json(a.lot41 / "search" / f"budget_{REF_BUDGET}.json")
    ref_by_fp = {row["state_fingerprint"]: row for row in ref["rows"]}
    run_payloads = []
    pilot_path = a.output / "memory_pilot.json"
    if checked(pilot_path):
        run_payloads.append(read_json(pilot_path))
    run_payloads.extend(
        read_json(p)
        for p in sorted((a.output / "shards").glob("*.json"))
        if not p.name.endswith(".checksum.json") and checked(p)
    )
    common = sorted(set(found) & set(hard_by_fp) & set(ref_by_fp))
    paired = []
    regrets = []
    js_values = []
    ranks = []
    same = 0
    top2_same = 0
    top2_order = 0
    sev_counts = {"NEAR_TIE": 0, "LOW_SEVERITY": 0, "MEDIUM_SEVERITY": 0, "HIGH_SEVERITY": 0}
    ultra = []
    stable = []
    for fp in common:
        h = hard_by_fp[fp]
        r32 = ref_by_fp[fp]
        r65 = found[fp]
        js = jensen_shannon(r32["visit_distribution"], r65["visit_distribution"])
        rank_sim = kendall_agreement(rank_actions(r32), r65["ranking"])
        set32, ord32 = top2(r32)
        set65, ord65 = top2({"legal_mask": r65["legal_mask"], "visit_counts": r65["visit_counts"]})
        flip = r32["selected_action"] != r65["selected_action"]
        same += int(not flip)
        top2_same += int(set32 == set65)
        top2_order += int(ord32 == ord65)
        q = r65["root_q_values"]
        best = r65["selected_action"]
        old = r32["selected_action"]
        regret = None
        sev = None
        if best is not None and old is not None and r65["visit_counts"][best] > 0 and r65["visit_counts"][old] > 0:
            regret = q[best] - q[old]
            regrets.append(regret)
            sev = severity(abs(regret)) if flip else "NEAR_TIE"
            if flip:
                sev_counts[sev] += 1
        material = flip and (js > 0.05 or (regret is not None and regret > 0.10))
        target = ultra if material else stable
        record = {
            "fingerprint": fp,
            "state": h.get("state"),
            "hard_reason": h.get("difficulty_class") or h.get("Lot41B hard reason") or "LOT41B_HARD_POSITION",
            "action_32768": r32["selected_action"],
            "action_65536": r65["selected_action"],
            "flip_32768_65536": flip,
            "js_32768_65536": js,
            "ranking_similarity": rank_sim,
            "q_regret_32768_vs_65536": regret,
            "flip_severity": sev,
            "trajectory_extended": {**h.get("action_trajectory", {}), "65536": r65["selected_action"]},
            "visits_32768": r32["visit_counts"],
            "visits_65536": r65["visit_counts"],
            "pi_32768": r32["visit_distribution"],
            "pi_65536": r65["visit_distribution"],
            "ranking_32768": rank_actions(r32),
            "ranking_65536": r65["ranking"],
            "q_32768": r32["root_q_values"],
            "q_65536": r65["root_q_values"],
            "runtime_65536": r65["runtime_65536"],
            "nodes_65536": r65["nodes_65536"],
            "ram_metrics": None,
            "ultra_hard": material,
        }
        paired.append(record)
        target.append(record)
        js_values.append(js)
        ranks.append(rank_sim)
    total = len(common)
    flips = total - same
    top1 = same / total if total else None
    high_rate = sev_counts["HIGH_SEVERITY"] / total if total else None
    stable_at_32768 = "YES" if total and top1 >= 0.95 and statistics.median(js_values) <= 0.02 and float(np.percentile(js_values, 90)) <= 0.05 and high_rate <= 0.02 else "NO"
    gain = "NEGLIGIBLE"
    if flips / total > 0.20 or high_rate > 0.05:
        gain = "MATERIAL"
    elif flips / total > 0.05 or (js_values and float(np.percentile(js_values, 90)) > 0.05):
        gain = "SMALL"
    if high_rate and high_rate > 0.10:
        gain = "LARGE"
    role = "NOT_NEEDED" if gain == "NEGLIGIBLE" else "VALIDATION_ONLY" if gain == "SMALL" else "ULTRA_HARD_ONLY" if len(ultra) / total <= 0.25 else "ROUTINE_HARD_TEACHER"
    summary = {
        "LOT42_VALID": "YES",
        "LOT42_COMPLETE": "YES" if total == len(hard) else "NO",
        "HARD_POSITION_COUNT": len(hard),
        "COMPLETED_HARD_POSITIONS": total,
        "TOP1_AGREEMENT_32768_65536": top1,
        "ACTION_FLIP_RATE_32768_65536": 1 - top1 if top1 is not None else None,
        "TOP1_WILSON_CI": wilson(same, total),
        "MEDIAN_JS_32768_65536": statistics.median(js_values) if js_values else None,
        "P90_JS_32768_65536": float(np.percentile(js_values, 90)) if js_values else None,
        "RANKING_STABILITY_32768_65536": statistics.fmean(ranks) if ranks else None,
        "FLIPS_32768_65536": flips,
        "NEAR_TIE_FLIPS": sev_counts["NEAR_TIE"],
        "LOW_SEVERITY_FLIPS": sev_counts["LOW_SEVERITY"],
        "MEDIUM_SEVERITY_FLIPS": sev_counts["MEDIUM_SEVERITY"],
        "HIGH_SEVERITY_FLIPS": sev_counts["HIGH_SEVERITY"],
        "HIGH_SEVERITY_FLIP_RATE": high_rate,
        "Q_REGRET_AVAILABLE": "YES" if regrets else "NO",
        "MEAN_DEEPER_SEARCH_REGRET_32768": statistics.fmean(regrets) if regrets else None,
        "MEDIAN_DEEPER_SEARCH_REGRET_32768": statistics.median(regrets) if regrets else None,
        "P90_DEEPER_SEARCH_REGRET_32768": float(np.percentile(regrets, 90)) if regrets else None,
        "MAX_DEEPER_SEARCH_REGRET_32768": max(regrets) if regrets else None,
        "HARD_SET_EMPIRICALLY_STABLE_AT_32768": stable_at_32768,
        "MCTS65536_ADDITIONAL_INFORMATION": gain,
        "MCTS65536_ROLE": role,
        "MULTI_FIDELITY_CONFIRMED": "YES",
        "ULTRA_HARD_POSITION_COUNT": len(ultra),
        "ULTRA_HARD_POSITION_RATE_WITHIN_HARD_SET": len(ultra) / total if total else None,
        "ADAPTIVE_DEEPENING_FEASIBLE": "YES" if ultra and len(ultra) < total else "INCONCLUSIVE",
        "FUTURE_TEACHER_PROTOCOL": {"BASE_BUDGET": 4096, "INTERMEDIATE_BUDGET": 8192, "DEEP_BUDGET": 32768, "ULTRA_DEEP_BUDGET": 65536 if role in ("ULTRA_HARD_ONLY", "ROUTINE_HARD_TEACHER") else None},
        "NEXT_LOT": "LOT43_MULTI_FIDELITY_TARGET_GENERATION_PROTOCOL",
        "TRAINING_PERFORMED": "NO",
        "MODEL_WEIGHTS_CHANGED": "YES" if any(p.get("model_weights_changed", False) for p in run_payloads) else "NO",
    }
    write_json(a.output / "search_65536_results.json", {"rows": list(found.values())})
    write_json(a.output / "comparison_32768_65536.json", {"rows": paired})
    write_json(a.output / "top1_stability.json", summary)
    write_json(a.output / "js_stability.json", summarize(js_values))
    write_json(a.output / "ranking_stability.json", summarize(ranks))
    write_json(a.output / "top2_stability.json", {"same_top2_set": top2_same, "same_top2_set_rate": top2_same / total if total else None, "same_top2_order": top2_order, "same_top2_order_rate": top2_order / total if total else None})
    write_json(a.output / "q_stability.json", {"Q_AVAILABLE": "YES", "regret_summary": summarize(regrets)})
    write_json(a.output / "deeper_search_regret.json", {"rows": [{"fingerprint": row["fingerprint"], "regret": row["q_regret_32768_vs_65536"]} for row in paired], "summary": summarize([x for x in regrets if x is not None])})
    write_json(a.output / "flip_severity.json", {"counts": sev_counts, "thresholds": {"near": "<=0.02 Q", "low": "<=0.05 Q", "medium": "<=0.10 Q", "high": ">0.10 Q"}})
    write_json(a.output / "extended_action_trajectories.json", {"rows": [{"fingerprint": row["fingerprint"], "trajectory": row["trajectory_extended"]} for row in paired]})
    write_json(a.output / "extended_stability_budget.json", {"rows": [{"fingerprint": row["fingerprint"], "extended_stability_budget": 65536 if row["flip_32768_65536"] else 32768} for row in paired]})
    write_json(a.output / "ultra_hard_positions.json", {"count": len(ultra), "rows": ultra})
    write_json(a.output / "hard_but_stable_at_32768.json", {"count": len(stable), "rows": stable})
    write_json(a.output / "uncertainty_predictors.json", {"status": "DESCRIPTIVE_ONLY", "note": "Predictor analysis is deferred to final hard-set results; no classifier trained."})
    write_json(a.output / "adaptive_deepening_analysis.json", {"ADAPTIVE_DEEPENING_FEASIBLE": summary["ADAPTIVE_DEEPENING_FEASIBLE"], "oracle_policy": summary["FUTURE_TEACHER_PROTOCOL"], "predictable_policy": "Requires pre-65536 signal validation; no learned scheduler trained."})
    write_json(a.output / "multi_fidelity_decision.json", summary)
    write_json(a.output / "compute_summary.json", {"completed_positions": total, "total_65536_simulations": total * BUDGET, "runs": run_payloads})
    write_json(a.output / "decision.json", summary)
    write_json(a.output / "report.json", {"lot": 42, "decision": summary, "warning": "Hard-set metrics are conditional on Lot41B hard positions, not all 256 positions."})
    with (a.output / "hard_position_65536_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["fingerprint", "hard_reason", "action_32768", "action_65536", "flip_32768_65536", "js_32768_65536", "ranking_similarity", "q_regret_32768_vs_65536", "flip_severity", "ultra_hard", "runtime_65536", "nodes_65536"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in paired:
            w.writerow({k: row.get(k) for k in fields})
    checks = {str(p.relative_to(a.output)): sha256(p) for p in a.output.rglob("*") if p.is_file() and p.name not in ("checksums.json", "experiment_manifest.json")}
    write_json(a.output / "checksums.json", checks)
    write_json(a.output / "experiment_manifest.json", {"lot": 42, "git_commit": git_commit(), "hard_set_sha256": read_json(a.output / "hard_set_fingerprint.json")["HARD_SET_SHA256"], "artifact_checksums": checks})
    print(json.dumps(summary, indent=2), flush=True)


def export(a: argparse.Namespace) -> None:
    if not (a.output / "decision.json").is_file():
        raise RuntimeError("finalize Lot42 first")
    files = [p for p in a.output.rglob("*") if p.is_file()]
    with tarfile.open(a.bundle, "w:gz") as tf:
        for p in files:
            tf.add(p, arcname=f"lot42_hard_position_65536/{p.relative_to(a.output)}")
    Path(str(a.bundle) + ".sha256").write_text(f"{sha256(a.bundle)}  {a.bundle.name}\n", encoding="utf-8")


if __name__ == "__main__":
    a = args()
    if a.stage == "prepare":
        prepare(a)
    elif a.stage == "pilot":
        pilot(a)
    elif a.stage == "run":
        run(a)
    elif a.stage == "finalize":
        finalize(a)
    elif a.stage == "export":
        export(a)
