#!/usr/bin/env python3
"""Lot 15 : réplication G3-B à split/hyperparamètres/corpus fixes."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from songo_ai.dataset import read_d_rl_jsonl
from songo_ai.evaluation import (
    ArenaConfig,
    SRNMCTSAgent,
    analyze_policy_mcts_signal,
    assert_no_minimax_labels,
    game_result_to_dict,
    generate_unique_deterministic_openings,
    model_parameter_fingerprint,
    opening_to_dict,
    run_paired_arena,
    summarize_arena,
)
from songo_ai.model import (
    SRNConfig,
    SRNTrainingConfig,
    load_srn_checkpoint,
    train_srn_from_d_rl,
)

from run_srn_lot12 import fixed_batch_outputs, sanity_report, sha256, write_json, write_metrics_csv


EXPECTED_CORPUS_SHA = "2f24dacc33f897ed9645b08823a6601d4a48601b7f3bc0c9e8a7c120aa850e57"
EXPECTED_G2_SHA = "eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753"
EXPECTED_MINIMAX_FINGERPRINT = "70f3b5632da10cfa9df4c6f28c19f8c8b8927d514b515d1c31984243d61e334c"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl"),
    )
    parser.add_argument(
        "--g2-checkpoint", type=Path,
        default=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt"),
    )
    parser.add_argument(
        "--g3a-checkpoint", type=Path,
        default=Path("data/experiments/lot14_g3_seed_20261402/training/best_validation_checkpoint.pt"),
    )
    parser.add_argument(
        "--lot14-report", type=Path,
        default=Path("data/experiments/lot14_g3_seed_20261402/report.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/experiments/lot15_g3b_seed_20261515"),
    )
    parser.add_argument("--optimization-seed", type=int, default=20261515)
    parser.add_argument("--arena-seed", type=int, default=20261516)
    parser.add_argument("--arena-openings", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--max-plies", type=int, default=400)
    return parser.parse_args()


def metrics_delta(first, second):
    return {
        key: getattr(second, key) - getattr(first, key)
        for key in (
            "total_loss", "policy_cross_entropy", "policy_kl",
            "policy_top1_accuracy", "value_loss", "value_mae",
            "value_sign_accuracy",
        )
    }


def run_arena(g2, g3b, args):
    openings = generate_unique_deterministic_openings(
        count=args.arena_openings, seed=args.arena_seed, max_prefix_length=40
    )
    write_json(args.output / "arena_openings.json", {
        "seed": args.arena_seed,
        "openings": [opening_to_dict(opening) for opening in openings],
    })
    config = ArenaConfig(
        max_plies=args.max_plies,
        repetition_limit=3,
        seed=args.arena_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    summaries, games = {}, []
    for budget in (32, 64):
        results = run_paired_arena(
            SRNMCTSAgent("G3-B-best-validation", g3b, budget, c_puct=1.5),
            SRNMCTSAgent("G2-best-validation", g2, budget, c_puct=1.5),
            openings,
            config=config,
        )
        summaries[str(budget)] = asdict(summarize_arena(results, config=config))
        games.extend({**game_result_to_dict(result), "budget": budget} for result in results)
        print(f"[lot15] arène budget={budget}: {len(results)} parties", flush=True)
    write_json(args.output / "arena_summaries.json", summaries)
    (args.output / "arena_games.jsonl").write_text(
        "".join(json.dumps(game) + "\n" for game in games), encoding="utf-8"
    )
    return summaries, games


def verdict(summaries):
    principal = summaries["64"]
    score = principal["score_rate_a_terminal"]
    interval = principal["paired_bootstrap_ci"]
    clear_progress = score > 0.5 and interval[0] > 0.5
    clear_regression = score < 0.5 and interval[1] < 0.5
    if clear_progress:
        scenario = 1
        variance = "YES"
    elif clear_regression:
        scenario = 3
        variance = "NO"
    else:
        scenario = 2
        variance = "INCONCLUSIVE"
    return {
        "scenario": scenario,
        "OPTIMIZATION_VARIANCE_EXPLAINS_G3A": variance,
        "G3B_PROGRESS_OVER_G2": "YES" if clear_progress else "NO",
        "G3B_GENERATOR_CANDIDATE": "YES" if clear_progress else "NO",
        "NEXT_ACTION": (
            "Run the fixed Minimax benchmark for G3-B, then require independent confirmation before promotion."
            if clear_progress else
            "STOP seed replications; diagnose how G2 Policy is transformed by MCTS64 and why the Value signal does not transfer."
            if clear_regression else
            "Do not promote; treat G2→G3 as a weak/plateau signal and design one mechanism diagnostic before any new generation."
        ),
    }


def main():
    args = parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    corpus_hash_before = sha256(args.corpus)
    if corpus_hash_before != EXPECTED_CORPUS_SHA:
        raise RuntimeError("Lot 14 immutable corpus hash mismatch")
    if sha256(args.g2_checkpoint) != EXPECTED_G2_SHA:
        raise RuntimeError("G2 checkpoint hash mismatch")
    lot14 = json.loads(args.lot14_report.read_text(encoding="utf-8"))
    if lot14["minimax_reference"]["fingerprint"] != EXPECTED_MINIMAX_FINGERPRINT:
        raise RuntimeError("Minimax reference fingerprint mismatch")
    g2_loaded = load_srn_checkpoint(args.g2_checkpoint)
    g3a_loaded = load_srn_checkpoint(args.g3a_checkpoint)
    if int(g3a_loaded.payload["training_config"]["seed"]) == args.optimization_seed:
        raise RuntimeError("G3-B optimization seed must differ from G3-A")
    if g3a_loaded.payload["lineage"]["d_rl_sha256"] != EXPECTED_CORPUS_SHA:
        raise RuntimeError("G3-A did not use the expected corpus")
    examples = read_d_rl_jsonl(args.corpus)
    assert_no_minimax_labels(examples)
    g2 = g2_loaded.model
    parent_before = model_parameter_fingerprint(g2)

    signal_started = time.perf_counter()
    signal = analyze_policy_mcts_signal(g2, examples, batch_size=512)
    signal["elapsed_s"] = time.perf_counter() - signal_started
    signal["selection"] = "all immutable Lot 14 corpus positions"
    write_json(args.output / "policy_mcts_signal.json", signal)

    training_data = dict(g3a_loaded.payload["training_config"])
    seed_g3a = int(training_data["seed"])
    training_data["seed"] = args.optimization_seed
    training_data["device"] = "cpu"
    training_config = SRNTrainingConfig(**training_data)
    srn_config = SRNConfig(**g3a_loaded.payload["srn_config"])
    fixed_examples, fixed_batch, g2_policy, g2_value = fixed_batch_outputs(g2, examples)
    clone = load_srn_checkpoint(args.g2_checkpoint).model
    clone.eval()
    with torch.no_grad():
        clone_policy, clone_value = clone(fixed_batch.graph)
    initialization = {
        "policy_exact": torch.equal(g2_policy, clone_policy),
        "value_exact": torch.equal(g2_value, clone_value),
        "parent_fingerprint": parent_before,
    }
    if not initialization["policy_exact"] or not initialization["value_exact"]:
        raise RuntimeError("G3-B epoch 0 is not exactly G2")

    training_started = time.perf_counter()
    lineage = {
        "generation": "G3-B-replication",
        "same_generation_as": "G3-A",
        "parent": "G2-best-validation",
        "parent_checkpoint": str(args.g2_checkpoint),
        "parent_checkpoint_sha256": EXPECTED_G2_SHA,
        "generator": "G2-best-validation",
        "d_rl_sha256": EXPECTED_CORPUS_SHA,
        "optimization_seed": args.optimization_seed,
    }
    training = train_srn_from_d_rl(
        args.corpus,
        args.output / "training",
        srn_config=srn_config,
        training_config=training_config,
        initial_checkpoint=args.g2_checkpoint,
        lineage=lineage,
        fixed_train_game_ids=g3a_loaded.payload["train_game_ids"],
        fixed_validation_game_ids=g3a_loaded.payload["validation_game_ids"],
    )
    training_elapsed = time.perf_counter() - training_started
    split_check = {
        "train_exact": list(training.train_game_ids) == g3a_loaded.payload["train_game_ids"],
        "validation_exact": list(training.validation_game_ids) == g3a_loaded.payload["validation_game_ids"],
        "no_leakage": not bool(set(training.train_game_ids) & set(training.validation_game_ids)),
    }
    if not all(split_check.values()):
        raise RuntimeError("G3-B split differs from G3-A")
    epoch0_a = g3a_loaded.payload["history"][0]
    epoch0_b = asdict(training.history[0])
    epoch0_check = {
        "exact": epoch0_a == epoch0_b,
        "G3A": epoch0_a,
        "G3B": epoch0_b,
    }
    if not epoch0_check["exact"]:
        raise RuntimeError("G3-A/G3-B epoch 0 differs on fixed split")
    write_metrics_csv(args.output / "training" / "metrics.csv", training.history)
    g3b_loaded = load_srn_checkpoint(training.best_validation_checkpoint)
    last_loaded = load_srn_checkpoint(training.last_checkpoint)
    if g3b_loaded.payload["lineage"] != lineage or last_loaded.payload["lineage"] != lineage:
        raise RuntimeError("G3-B checkpoint lineage mismatch")
    write_json(args.output / "sanity_fixed_positions.json", sanity_report(g2, g3b_loaded.model, examples))
    if model_parameter_fingerprint(g2) != parent_before:
        raise RuntimeError("training mutated G2")

    arena_started = time.perf_counter()
    arena_summaries, arena_games = run_arena(g2, g3b_loaded.model, args)
    arena_elapsed = time.perf_counter() - arena_started
    result_verdict = verdict(arena_summaries)
    if model_parameter_fingerprint(g2) != parent_before:
        raise RuntimeError("arena mutated G2")
    corpus_hash_after = sha256(args.corpus)
    if corpus_hash_after != corpus_hash_before:
        raise RuntimeError("immutable corpus changed")

    best_b = next(record for record in training.history if record.epoch == training.best_epoch)
    best_a_record = g3a_loaded.payload["history"][g3a_loaded.payload["best_epoch"]]
    comparison = {
        "G3-A": {
            "seed": seed_g3a,
            "best_epoch": g3a_loaded.payload["best_epoch"],
            "best_validation": best_a_record["validation"],
            "arena": lot14["G3_vs_G2"]["summaries"],
        },
        "G3-B": {
            "seed": args.optimization_seed,
            "best_epoch": training.best_epoch,
            "best_validation": asdict(best_b.validation),
            "arena": arena_summaries,
        },
    }
    report = {
        "lot": 15,
        "integrity": {
            "corpus_path": str(args.corpus),
            "corpus_sha256_before": corpus_hash_before,
            "corpus_sha256_after": corpus_hash_after,
            "corpus_unchanged": corpus_hash_before == corpus_hash_after,
            "G2_checkpoint": str(args.g2_checkpoint),
            "G2_sha256": sha256(args.g2_checkpoint),
            "G3A_seed": seed_g3a,
            "G3B_seed": args.optimization_seed,
            "split": split_check,
            "epoch0": epoch0_check,
            "initialization": initialization,
            "minimax_labels_absent": True,
            "datasets_10k_100k_used": False,
        },
        "training": {
            "config": asdict(training_config),
            "best_epoch": training.best_epoch,
            "best_validation": asdict(best_b),
            "last": asdict(training.history[-1]),
            "stopped_early": training.stopped_early,
            "stop_epoch": training.stop_epoch,
            "elapsed_s": training_elapsed,
            "best_checkpoint": str(training.best_validation_checkpoint),
            "last_checkpoint": str(training.last_checkpoint),
            "best_sha256": sha256(training.best_validation_checkpoint),
            "last_sha256": sha256(training.last_checkpoint),
        },
        "policy_mcts_signal": signal,
        "arena": {
            "seed": args.arena_seed,
            "openings": args.arena_openings,
            "games": len(arena_games),
            "summaries": arena_summaries,
            "elapsed_s": arena_elapsed,
        },
        "comparison_G3A_G3B": comparison,
        "verdict": result_verdict,
        "minimax_benchmark_rerun": False,
        "historical_minimax": {
            "fingerprint": EXPECTED_MINIMAX_FINGERPRINT,
            "G2_score": 0.015625,
            "G3A_score": 0.01953125,
        },
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output / "report.json", report)
    print(json.dumps({
        "report": str(args.output / "report.json"),
        "training_best_epoch": training.best_epoch,
        "signal": signal["global"],
        "arena": arena_summaries,
        "verdict": result_verdict,
        "elapsed_s": report["elapsed_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
