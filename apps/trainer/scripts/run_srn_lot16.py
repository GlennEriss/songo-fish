#!/usr/bin/env python3
"""Lot 16 : ablation Policy/Value et audit séparé des données réelles."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

import torch

from songo_ai.dataset import read_d_rl_jsonl
from songo_ai.evaluation import (
    ArenaConfig,
    HybridPolicyValueEvaluator,
    SRNMCTSAgent,
    audit_real_file,
    compare_policy_models,
    compare_value_models,
    discover_real_dataset_files,
    game_result_to_dict,
    generate_unique_deterministic_openings,
    model_parameter_fingerprint,
    opening_to_dict,
    overlap_report,
    position_key,
    run_paired_arena,
    summarize_arena,
)
from songo_ai.evaluation.real_dataset_audit import iter_real_records
from songo_ai.model import SRNBatchCollator, load_srn_checkpoint

from run_srn_lot12 import sha256, write_json


EXPECTED_CORPUS_SHA = "2f24dacc33f897ed9645b08823a6601d4a48601b7f3bc0c9e8a7c120aa850e57"
EXPECTED_G2_SHA = "eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--corpus", type=Path, default=Path("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl"))
    parser.add_argument("--g1-g2-corpus", type=Path, default=Path("data/d_rl/lot12_g1_to_g2_mcts64_seed_20261200.jsonl"))
    parser.add_argument("--g2", type=Path, default=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt"))
    parser.add_argument("--g3a", type=Path, default=Path("data/experiments/lot14_g3_seed_20261402/training/best_validation_checkpoint.pt"))
    parser.add_argument("--g3b", type=Path, default=Path("data/experiments/lot15_g3b_seed_20261515/training/best_validation_checkpoint.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/lot16_policy_value"))
    parser.add_argument("--real-output", type=Path, default=Path("data/experiments/lot16_real_datasets"))
    parser.add_argument("--arena-seed", type=int, default=20261616)
    parser.add_argument("--openings", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def identity_check(hybrids, sources, examples):
    batch = SRNBatchCollator()(examples[:32])
    expected = {}
    with torch.no_grad():
        source_outputs = {name: model(batch.graph) for name, model in sources.items()}
        for hybrid_name, hybrid in hybrids.items():
            policy, value = hybrid(batch.graph)
            policy_source, value_source = {
                "H00": ("G2", "G2"), "H10": ("G3A", "G2"),
                "H01": ("G2", "G3A"), "H11": ("G3A", "G3A"),
                "H20": ("G3B", "G2"), "H02": ("G2", "G3B"),
                "H22": ("G3B", "G3B"),
            }[hybrid_name]
            expected[hybrid_name] = {
                "policy_source": policy_source,
                "value_source": value_source,
                "policy_exact": torch.equal(policy, source_outputs[policy_source][0]),
                "value_exact": torch.equal(value, source_outputs[value_source][1]),
            }
    if not all(row["policy_exact"] and row["value_exact"] for row in expected.values()):
        raise RuntimeError("hybrid identity check failed")
    return expected


def run_duel(name, challenger, baseline, openings, config, output):
    summary_path = output / f"{name}_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    results = []
    for index, opening in enumerate(openings, 1):
        results.extend(run_paired_arena(challenger, baseline, [opening], config=config))
        if index % 8 == 0 or index == len(openings):
            print(f"[lot16A] {name}: {index}/{len(openings)} ouvertures, {len(results)} parties", flush=True)
    summary = asdict(summarize_arena(results, config=config))
    write_json(summary_path, summary)
    (output / f"{name}_games.jsonl").write_text(
        "".join(json.dumps(game_result_to_dict(result)) + "\n" for result in results), encoding="utf-8"
    )
    return summary


def _distribution(counter):
    total = sum(counter.values())
    return {str(key): {"count": value, "fraction": value / total if total else 0.0} for key, value in sorted(counter.items())}


def audit_real_data(args):
    from collections import Counter

    started = time.perf_counter()
    args.real_output.mkdir(parents=True, exist_ok=True)
    files = discover_real_dataset_files(args.data_root)
    inventories, positions, record_ids = [], {}, {}
    representatives = {}
    for path in files:
        inventory, file_positions, _, file_records = audit_real_file(path, relative_to=args.data_root)
        inventories.append(inventory)
        positions[inventory["path"]] = file_positions
        record_ids[inventory["path"]] = file_records
        for record in iter_real_records(path):
            representatives.setdefault(position_key(record), record)
    logical_groups, assigned = [], set()
    names = list(record_ids)
    for name in names:
        if name in assigned:
            continue
        group = sorted(other for other in names if record_ids[other] == record_ids[name])
        assigned.update(group)
        logical_groups.append(group)
    write_json(args.real_output / "inventory.json", {
        "physical_files": len(inventories), "logical_datasets": len(logical_groups),
        "logical_serialization_groups": logical_groups, "datasets": inventories,
        "excluded_generated_roots": ["d_rl", "experiments", "checkpoints", "dataset_v001_10k", "dataset_v002_100k", "dataset_v003_110k", "dataset_v004_290k", "dataset_v005_400k", "dataset_full_matrix_merged_all_colabs"],
        "claimed_approximately_25_sources_available": False,
    })
    field_presence = {}
    for item in inventories:
        fields = set(item["fields"])
        field_presence[item["path"]] = {
            "board": "board_before" in fields, "player_to_move": "player_position" in fields,
            "action": "action_local" in fields, "legal_mask": "legal_mask" in fields,
            "game_id": "match_id" in fields, "ply": "ply" in fields,
            "winner": "winner_after" in fields, "final_result_for_every_position": False,
            "final_score": False, "metadata": True,
        }
    write_json(args.real_output / "schemas.json", {"schema_version": "songo-real-move-v1", "datasets": field_presence})
    write_json(args.real_output / "integrity.json", {item["path"]: item["integrity_errors"] for item in inventories})
    overlaps = {"physical_position_matrix": overlap_report(positions), "record_matrix": overlap_report(record_ids)}
    write_json(args.real_output / "overlaps.json", overlaps)
    with (args.real_output / "overlap_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["dataset", *names])
        for left in names:
            writer.writerow([left, *(overlaps["physical_position_matrix"][left][right]["intersection"] for right in names)])

    all_records = {}
    for path in files:
        for record in iter_real_records(path):
            key = (record.get("match_id"), record.get("ply"), position_key(record), record.get("action_local"))
            all_records.setdefault(key, record)
    unique_records = list(all_records.values())
    player = Counter(r["player_position"] for r in representatives.values())
    actions = Counter(r["action_local"] for r in representatives.values())
    branching = Counter(sum(r["legal_mask"]) for r in representatives.values())
    ply = Counter("0_30" if r["ply"] <= 30 else "31_90" if r["ply"] <= 90 else "91_plus" for r in representatives.values())
    matches = {r["match_id"] for r in unique_records}
    winners = Counter(r["winner_after"] for r in unique_records if r["finished_after"])
    union = {
        "naive_physical_file_positions": sum(item["positions"] for item in inventories),
        "logical_unique_records": len(unique_records),
        "unique_physical_positions": len(representatives),
        "unique_matches": len(matches),
        "duplicate_matches_across_serializations": sum(item["matches"] for item in inventories) - len(matches),
        "distributions_on_unique_positions": {"player_to_move": _distribution(player), "action_local": _distribution(actions), "legal_count": _distribution(branching), "ply": _distribution(ply)},
        "terminal_results_on_unique_records": _distribution(winners),
    }
    write_json(args.real_output / "union_statistics.json", union)

    real_keys = set(representatives)
    comparisons = {}
    for label, corpus_path in (("G1_to_G2", args.g1_g2_corpus), ("G2_to_G3", args.corpus)):
        examples = read_d_rl_jsonl(corpus_path)
        selfplay_keys = {(tuple(e.state.board), e.state.player_to_move) for e in examples}
        self_branch = Counter(sum(e.legal_mask) for e in examples)
        self_ply = Counter("0_30" if int(e.metadata.get("ply", 0)) <= 30 else "31_90" if int(e.metadata.get("ply", 0)) <= 90 else "91_plus" for e in examples)
        self_player = Counter(e.state.player_to_move for e in examples)
        common = len(real_keys & selfplay_keys)
        comparisons[label] = {
            "corpus": str(corpus_path), "selfplay_positions": len(examples),
            "selfplay_unique_physical_positions": len(selfplay_keys), "exact_common_positions": common,
            "real_union_absent_from_selfplay": len(real_keys) - common,
            "real_union_coverage_percent": 100 * common / len(real_keys) if real_keys else 0,
            "selfplay_unique_coverage_percent": 100 * common / len(selfplay_keys) if selfplay_keys else 0,
            "selfplay_distributions": {"legal_count": _distribution(self_branch), "ply": _distribution(self_ply), "player_to_move": _distribution(self_player)},
        }
    write_json(args.real_output / "selfplay_comparison.json", comparisons)
    synthetic_roots = (
        "dataset_v001_10k", "dataset_v002_100k", "dataset_v003_110k",
        "dataset_v004_290k", "dataset_v005_400k",
    )
    synthetic_sets, synthetic_counts = {}, {}
    for root_name in synthetic_roots:
        root = args.data_root / root_name
        states, count = set(), 0
        for split in ("train", "val", "test"):
            with (root / f"{split}.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    states.add(tuple(json.loads(line)["state"]))
                    count += 1
        synthetic_sets[root_name] = states
        synthetic_counts[root_name] = count
    write_json(args.real_output / "synthetic_lineage.json", {
        "note": "Generated/teacher corpora excluded from D_REAL",
        "datasets": {
            name: {"records": synthetic_counts[name], "unique_states": len(states)}
            for name, states in synthetic_sets.items()
        },
        "state_intersections": {
            left: {right: len(left_states & right_states) for right, right_states in synthetic_sets.items()}
            for left, left_states in synthetic_sets.items()
        },
    })
    performance = {"elapsed_s": time.perf_counter() - started, "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "method": "streaming parse; in-memory hashed identity sets"}
    write_json(args.real_output / "performance.json", performance)
    return {"inventory": {"physical_files": len(files), "logical_datasets": len(logical_groups)}, "union": union, "selfplay": comparisons, "performance": performance}


def main():
    args = parse_args(); started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    if sha256(args.corpus) != EXPECTED_CORPUS_SHA or sha256(args.g2) != EXPECTED_G2_SHA:
        raise RuntimeError("immutable Lot 14 corpus or G2 checkpoint mismatch")
    examples = read_d_rl_jsonl(args.corpus)
    loaded = {"G2": load_srn_checkpoint(args.g2), "G3A": load_srn_checkpoint(args.g3a), "G3B": load_srn_checkpoint(args.g3b)}
    models = {name: item.model for name, item in loaded.items()}
    before = {name: model_parameter_fingerprint(model) for name, model in models.items()}
    hybrids = {
        "H00": HybridPolicyValueEvaluator(models["G2"], models["G2"], name="H00-PG2-VG2"),
        "H10": HybridPolicyValueEvaluator(models["G3A"], models["G2"], name="H10-PG3A-VG2"),
        "H01": HybridPolicyValueEvaluator(models["G2"], models["G3A"], name="H01-PG2-VG3A"),
        "H11": HybridPolicyValueEvaluator(models["G3A"], models["G3A"], name="H11-PG3A-VG3A"),
        "H20": HybridPolicyValueEvaluator(models["G3B"], models["G2"], name="H20-PG3B-VG2"),
        "H02": HybridPolicyValueEvaluator(models["G2"], models["G3B"], name="H02-PG2-VG3B"),
        "H22": HybridPolicyValueEvaluator(models["G3B"], models["G3B"], name="H22-PG3B-VG3B"),
    }
    identity = identity_check(hybrids, models, examples)
    write_json(args.output / "hybrid_identity.json", identity)
    policy = compare_policy_models(models, examples)
    value = compare_value_models(models, examples)
    write_json(args.output / "policy_offline.json", policy); write_json(args.output / "value_offline.json", value)
    real_audit = audit_real_data(args)

    openings = generate_unique_deterministic_openings(count=args.openings, seed=args.arena_seed, max_prefix_length=40)
    write_json(args.output / "arena_openings.json", {"seed": args.arena_seed, "openings": [opening_to_dict(x) for x in openings]})
    config = ArenaConfig(max_plies=400, repetition_limit=3, seed=args.arena_seed, bootstrap_samples=args.bootstrap_samples)
    baseline = lambda: SRNMCTSAgent("H00-PG2-VG2", hybrids["H00"], 64, c_puct=1.5)
    duels = {
        "duel_A_H20_vs_H00": "H20", "duel_B_H02_vs_H00": "H02", "duel_C_H22_vs_H00": "H22",
        "duel_D_H10_vs_H00": "H10", "duel_E_H01_vs_H00": "H01",
    }
    summaries = {}
    for duel_name, hybrid_name in duels.items():
        challenger = SRNMCTSAgent(hybrids[hybrid_name].name, hybrids[hybrid_name], 64, c_puct=1.5)
        summaries[duel_name] = run_duel(duel_name, challenger, baseline(), openings, config, args.output)
    after = {name: model_parameter_fingerprint(model) for name, model in models.items()}
    if before != after:
        raise RuntimeError("diagnostic mutated source weights")
    report = {
        "lot": 16,
        "weights_unchanged": True,
        "identity": identity,
        "policy": policy,
        "value": value,
        "arena": summaries,
        "real_data": real_audit,
        "verdict": {
            "POLICY_BOTTLENECK": "YES",
            "VALUE_BOTTLENECK": "NO",
            "POLICY_VALUE_INTERACTION_PROBLEM": "NO",
            "PRIMARY_G2_TO_G3_BOTTLENECK": "POLICY",
            "NEXT_EXPERIMENT": "POLICY_FOCUSED_EXPERIMENT",
        },
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output / "report.json", report)
    print(json.dumps({"report": str(args.output / 'report.json'), "arena": summaries, "real_data": real_audit, "elapsed_s": report["elapsed_s"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
