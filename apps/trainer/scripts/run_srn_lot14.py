#!/usr/bin/env python3
"""Lot 14 : génération G2→G3 et benchmark externe Minimax-Bidoua V1."""

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

from songo_ai.dataset import read_d_rl_jsonl, write_d_rl_jsonl
from songo_ai.evaluation import (
    ArenaConfig,
    MinimaxBidouaReferenceAgent,
    MinimaxBidouaReferenceConfig,
    SRNMCTSAgent,
    assert_no_minimax_labels,
    game_result_to_dict,
    generate_unique_deterministic_openings,
    model_parameter_fingerprint,
    opening_to_dict,
    play_arena_game,
    policy_corpus_metrics,
    repeated_state_metrics,
    run_paired_arena,
    summarize_arena,
)
from songo_ai.generation import SelfPlayConfig, SelfPlayRunner
from songo_ai.model import (
    SRNConfig,
    SRNTrainingConfig,
    load_srn_checkpoint,
    train_srn_from_d_rl,
)
from songo_ai.search import MCTSConfig

# Helpers mécaniques déjà validés au Lot 12 ; aucune politique scientifique
# n'est importée implicitement.
from run_srn_lot12 import (
    corpus_game_statistics,
    fixed_batch_outputs,
    quality_gate,
    sanity_report,
    sha256,
    validate_roundtrip,
    write_json,
    write_metrics_csv,
)


_BENCHMARK_MODEL_AGENT = None
_BENCHMARK_REFERENCE_AGENT = None
_BENCHMARK_ARENA_CONFIG = None


def _benchmark_worker_init(model_path, model_name, reference_data, arena_data):
    global _BENCHMARK_MODEL_AGENT, _BENCHMARK_REFERENCE_AGENT, _BENCHMARK_ARENA_CONFIG
    torch.set_num_threads(1)
    model = load_srn_checkpoint(model_path).model
    _BENCHMARK_MODEL_AGENT = SRNMCTSAgent(model_name, model, 64, c_puct=1.5)
    _BENCHMARK_REFERENCE_AGENT = MinimaxBidouaReferenceAgent(
        MinimaxBidouaReferenceConfig(**reference_data)
    )
    _BENCHMARK_ARENA_CONFIG = ArenaConfig(**arena_data)


def _benchmark_opening_pair(opening):
    model_agent = _BENCHMARK_MODEL_AGENT
    reference = _BENCHMARK_REFERENCE_AGENT
    config = _BENCHMARK_ARENA_CONFIG
    return (
        play_arena_game(
            opening,
            p1_agent=model_agent,
            p2_agent=reference,
            agent_a_name=model_agent.name,
            agent_b_name=reference.name,
            a_player=1,
            config=config,
        ),
        play_arena_game(
            opening,
            p1_agent=reference,
            p2_agent=model_agent,
            agent_a_name=model_agent.name,
            agent_b_name=reference.name,
            a_player=2,
            config=config,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lot12-report", type=Path,
        default=Path("data/experiments/lot12_g2_seed_20261200/report.json"),
    )
    parser.add_argument(
        "--lot12-corpus", type=Path,
        default=Path("data/d_rl/lot12_g1_to_g2_mcts64_seed_20261200.jsonl"),
    )
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/experiments/lot14_g3_seed_20261402"),
    )
    parser.add_argument("--generation-seed", type=int, default=20261402)
    parser.add_argument("--benchmark-seed", type=int, default=20261401)
    parser.add_argument("--arena-seed", type=int, default=20261403)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--arena-openings", type=int, default=128)
    parser.add_argument("--benchmark-openings", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--benchmark-workers", type=int, default=8)
    parser.add_argument("--reuse-corpus", action="store_true")
    parser.add_argument("--reuse-g2-benchmark", action="store_true")
    return parser.parse_args()


def selfplay_config(args, generator_sha):
    return SelfPlayConfig(
        games=args.games,
        max_game_plies=args.max_plies,
        repetition_limit=3,
        target_temperature=1.0,
        action_temperature=1.0,
        temperature_drop_ply=30,
        late_action_temperature=0.0,
        seed=args.generation_seed,
        generation=3,
        checkpoint_id=f"G2-best-{generator_sha[:12]}",
        provenance={
            "generation_lineage": "G2_to_G3",
            "generator": "G2-best-validation",
            "generator_checkpoint_sha256": generator_sha,
        },
        include_truncated_examples=True,
        mcts=MCTSConfig(
            num_simulations=args.simulations,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            add_root_noise=True,
        ),
    )


def validate_generation_contract(examples, args, generator_sha):
    failures = Counter()
    assert_no_minimax_labels(examples)
    for example in examples:
        meta = example.metadata
        checks = {
            "lineage": meta.get("generation_lineage") == "G2_to_G3",
            "generator": meta.get("generator") == "G2-best-validation",
            "generator_hash": meta.get("generator_checkpoint_sha256") == generator_sha,
            "seed": meta.get("selfplay_seed") == args.generation_seed,
            "simulations": meta.get("mcts_simulations") == 64,
            "c_puct": meta.get("c_puct") == 1.5,
            "visits": sum(example.visit_counts) == 64,
            "illegal_visits": not any(c for c, legal in zip(example.visit_counts, example.legal_mask) if not legal),
        }
        for name, passed in checks.items():
            if not passed:
                failures[name] += 1
    return {"valid": not failures, "failures": dict(failures), "minimax_labels_absent": True}


def search_cost(results):
    nodes = Counter()
    seconds = Counter()
    decisions = Counter()
    for result in results:
        for agent, value in (result.search_nodes_by_agent or {}).items():
            nodes[agent] += value
        for agent, value in (result.search_time_s_by_agent or {}).items():
            seconds[agent] += value
        # Une décision par action de continuation ; attribution physique
        # reconstruite depuis a_player et l'alternance n'est pas fiable si le
        # moteur conserve le trait. Le coût moyen est donc par nœud/partie.
        for agent in (result.agent_a, result.agent_b):
            decisions[agent] += 1
    return {
        agent: {
            "total_nodes": nodes[agent],
            "total_search_s": seconds[agent],
            "nodes_per_game": nodes[agent] / decisions[agent],
            "search_s_per_game": seconds[agent] / decisions[agent],
        }
        for agent in sorted(decisions)
    }


def run_benchmark(model_name, model_path, reference, openings, args, path_prefix):
    config = ArenaConfig(
        max_plies=args.max_plies,
        repetition_limit=3,
        seed=args.benchmark_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.benchmark_workers,
        mp_context=context,
        initializer=_benchmark_worker_init,
        initargs=(
            str(model_path), model_name, asdict(reference.config), asdict(config)
        ),
    ) as executor:
        pairs = list(executor.map(_benchmark_opening_pair, openings))
    results = tuple(result for pair in pairs for result in pair)
    summary = asdict(summarize_arena(results, config=config))
    payload = {
        "summary": summary,
        "search_cost": search_cost(results),
        "games": [game_result_to_dict(result) for result in results],
    }
    write_json(args.output / f"{path_prefix}.json", payload)
    print(
        f"[lot14] {model_name} vs Minimax: {len(results)} parties, "
        f"score={summary['score_rate_a_terminal']:.4f}", flush=True,
    )
    return payload


def run_relative_arena(g2, g3, args):
    openings = generate_unique_deterministic_openings(
        count=args.arena_openings, seed=args.arena_seed, max_prefix_length=40
    )
    write_json(args.output / "g3_vs_g2_openings.json", {
        "seed": args.arena_seed,
        "openings": [opening_to_dict(o) for o in openings],
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
            SRNMCTSAgent("G3-best-validation", g3, budget, c_puct=1.5),
            SRNMCTSAgent("G2-best-validation", g2, budget, c_puct=1.5),
            openings,
            config=config,
        )
        summaries[str(budget)] = asdict(summarize_arena(results, config=config))
        games.extend({**game_result_to_dict(r), "budget": budget} for r in results)
        print(f"[lot14] G3 vs G2 budget={budget}: {len(results)} parties", flush=True)
    write_json(args.output / "g3_vs_g2_summaries.json", summaries)
    (args.output / "g3_vs_g2_games.jsonl").write_text(
        "".join(json.dumps(game) + "\n" for game in games), encoding="utf-8"
    )
    return summaries, games


def generation_verdict(summaries):
    scores = [summaries[str(b)]["score_rate_a_terminal"] for b in (32, 64)]
    intervals = [summaries[str(b)]["paired_bootstrap_ci"] for b in (32, 64)]
    if all(score is not None and score > 0.5 for score in scores) and all(ci and ci[0] > 0.5 for ci in intervals):
        category = "C"
    elif any(score is not None and score > 0.5 for score in scores):
        category = "B"
    else:
        category = "A"
    return {
        "category": category,
        "G3_GENERATOR_CANDIDATE": category == "C",
        "automatic_authorization": False,
    }


def main():
    args = parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    lot12 = json.loads(args.lot12_report.read_text(encoding="utf-8"))
    g2_path = Path(lot12["checkpoints"]["G2_best_validation"])
    expected_g2_sha = lot12["checkpoints"]["best_sha256"]
    if sha256(g2_path) != expected_g2_sha:
        raise RuntimeError("G2 checkpoint hash mismatch")
    g2_loaded = load_srn_checkpoint(g2_path)
    if g2_loaded.payload["epoch"] != 4:
        raise RuntimeError("G2 checkpoint is not epoch 4")
    g2 = g2_loaded.model
    g2_before = model_parameter_fingerprint(g2)

    reference_config = MinimaxBidouaReferenceConfig()
    reference = MinimaxBidouaReferenceAgent(reference_config)
    benchmark_openings = generate_unique_deterministic_openings(
        count=args.benchmark_openings,
        seed=args.benchmark_seed,
        max_prefix_length=40,
    )
    benchmark_manifest = {
        "config": asdict(reference_config),
        "fingerprint": reference_config.fingerprint,
        "selection_reason": "strong historical table configuration minimax:14:5:bidoua",
        "openings_seed": args.benchmark_seed,
        "openings": [opening_to_dict(o) for o in benchmark_openings],
    }
    write_json(args.output / "minimax_bidoua_reference_v1.json", benchmark_manifest)
    g2_benchmark_path = args.output / "g2_vs_minimax.json"
    if args.reuse_g2_benchmark:
        g2_benchmark = json.loads(g2_benchmark_path.read_text(encoding="utf-8"))
    else:
        g2_benchmark = run_benchmark(
            "G2-best-validation", g2_path, reference, benchmark_openings, args,
            "g2_vs_minimax",
        )

    config = selfplay_config(args, expected_g2_sha)
    generation_started = time.perf_counter()
    if args.reuse_corpus:
        examples = read_d_rl_jsonl(args.corpus)
        generation = {"reused": True, "elapsed_s": 0.0}
        roundtrip = {"valid": True, "examples": len(examples), "sha256": sha256(args.corpus)}
    else:
        if args.corpus.exists():
            raise FileExistsError("immutable corpus exists; use --reuse-corpus")
        run = SelfPlayRunner(g2, config).generate()
        write_d_rl_jsonl(args.corpus, run.examples)
        examples = read_d_rl_jsonl(args.corpus)
        roundtrip = validate_roundtrip(args.corpus, run.examples)
        generation = {
            "reused": False,
            "elapsed_s": time.perf_counter() - generation_started,
            "statistics": asdict(run.statistics),
        }
    if model_parameter_fingerprint(g2) != g2_before:
        raise RuntimeError("generation mutated G2")
    contract = validate_generation_contract(examples, args, expected_g2_sha)
    stats = corpus_game_statistics(examples)
    policy = policy_corpus_metrics(examples)
    repeated = repeated_state_metrics(examples)
    baseline_examples = read_d_rl_jsonl(args.lot12_corpus)
    baseline = {
        "policy": policy_corpus_metrics(baseline_examples),
        "repeated": repeated_state_metrics(baseline_examples),
    }
    candidate = {"policy": policy, "repeated": repeated, "game_statistics": stats}
    gate = quality_gate(
        baseline, candidate,
        serialization_valid=roundtrip["valid"], contract_valid=contract["valid"],
    )
    gate["TRAINING_G3_AUTHORIZED"] = gate.pop("TRAINING_AUTHORIZED")
    corpus_sha = sha256(args.corpus)
    manifest = {
        "generation_lineage": "G2_to_G3",
        "path": str(args.corpus), "sha256": corpus_sha,
        "generator_checkpoint": str(g2_path), "generator_sha256": expected_g2_sha,
        "selfplay_config": asdict(config), "statistics": stats,
        "contract": contract, "roundtrip": roundtrip,
        "minimax_or_teacher_data_used": False,
    }
    write_json(args.output / "corpus_manifest.json", manifest)
    write_json(args.output / "quality_gate.json", {"baseline_G1_to_G2": baseline, "G2_to_G3": candidate, "gate": gate})
    if not gate["TRAINING_G3_AUTHORIZED"]:
        raise SystemExit("Lot 14 stopped before training: G2→G3 gate failed")
    print(f"[lot14] corpus validé: {len(examples)} positions", flush=True)

    srn_config = SRNConfig(**g2_loaded.payload["srn_config"])
    training_config = SRNTrainingConfig(
        epochs=args.epochs, batch_size=256, learning_rate=0.003,
        weight_decay=0.0001, lambda_policy=1.0, lambda_value=1.0,
        gradient_clip_norm=1.0, validation_fraction=0.2,
        early_stopping_patience=args.patience, early_stopping_min_delta=1e-4,
        seed=args.generation_seed,
    )
    fixed_examples, fixed_batch, g2_policy, g2_value = fixed_batch_outputs(g2, examples)
    clone = load_srn_checkpoint(g2_path).model
    clone.eval()
    with torch.no_grad():
        initial_policy, initial_value = clone(fixed_batch.graph)
    init_check = {
        "policy_exact": torch.equal(g2_policy, initial_policy),
        "value_exact": torch.equal(g2_value, initial_value),
    }
    if not all(init_check.values()):
        raise RuntimeError("G3 epoch 0 differs from G2")
    training_started = time.perf_counter()
    lineage = {
        "generation": "G3", "parent": "G2-best-validation",
        "parent_checkpoint": str(g2_path), "parent_checkpoint_sha256": expected_g2_sha,
        "generator": "G2-best-validation", "d_rl_sha256": corpus_sha,
    }
    training = train_srn_from_d_rl(
        args.corpus, args.output / "training",
        srn_config=srn_config, training_config=training_config,
        initial_checkpoint=g2_path, lineage=lineage,
    )
    training_elapsed = time.perf_counter() - training_started
    if set(training.train_game_ids) & set(training.validation_game_ids):
        raise RuntimeError("train/validation leakage")
    write_metrics_csv(args.output / "training" / "metrics.csv", training.history)
    g3_loaded = load_srn_checkpoint(training.best_validation_checkpoint)
    last_loaded = load_srn_checkpoint(training.last_checkpoint)
    if g3_loaded.payload["lineage"] != lineage or last_loaded.payload["lineage"] != lineage:
        raise RuntimeError("G3 lineage missing")
    write_json(args.output / "sanity_fixed_positions.json", sanity_report(g2, g3_loaded.model, examples))
    if model_parameter_fingerprint(g2) != g2_before:
        raise RuntimeError("training mutated parent G2")

    relative_summaries, relative_games = run_relative_arena(g2, g3_loaded.model, args)
    verdict = generation_verdict(relative_summaries)
    g3_benchmark = run_benchmark(
        "G3-best-validation", training.best_validation_checkpoint, reference,
        benchmark_openings, args, "g3_vs_minimax",
    )
    if model_parameter_fingerprint(g2) != g2_before:
        raise RuntimeError("evaluation mutated G2")
    best_record = next(r for r in training.history if r.epoch == training.best_epoch)
    report = {
        "lot": 14,
        "minimax_reference": benchmark_manifest,
        "baseline_G2_vs_minimax": g2_benchmark,
        "generation": {**generation, "config": asdict(config), "statistics": stats},
        "corpus": {"manifest": manifest, "quality": candidate, "gate": gate},
        "training": {
            "config": asdict(training_config), "initialization": init_check,
            "train_games": len(training.train_game_ids), "validation_games": len(training.validation_game_ids),
            "train_positions": training.train_positions, "validation_positions": training.validation_positions,
            "epoch_0": asdict(training.history[0]), "best_epoch": training.best_epoch,
            "best_validation": asdict(best_record), "last": asdict(training.history[-1]),
            "stopped_early": training.stopped_early, "stop_epoch": training.stop_epoch,
            "elapsed_s": training_elapsed,
        },
        "checkpoints": {
            "G3_best_validation": str(training.best_validation_checkpoint),
            "G3_last": str(training.last_checkpoint),
            "best_sha256": sha256(training.best_validation_checkpoint),
            "last_sha256": sha256(training.last_checkpoint), "lineage": lineage,
        },
        "G3_vs_G2": {"summaries": relative_summaries, "games": len(relative_games), "verdict": verdict},
        "G3_vs_minimax": g3_benchmark,
        "benchmark_comparison": {
            "G2_score": g2_benchmark["summary"]["score_rate_a_terminal"],
            "G3_score": g3_benchmark["summary"]["score_rate_a_terminal"],
        },
        "constraints": {
            "minimax_used_as_teacher": False, "old_corpora_mixed": False,
            "canonicalization": False, "mirror_augmentation": False,
            "G4_trained": False, "automatic_promotion": False,
        },
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output / "report.json", report)
    print(json.dumps({
        "report": str(args.output / "report.json"),
        "G2_vs_minimax": g2_benchmark["summary"],
        "G3_vs_G2": relative_summaries,
        "G3_vs_minimax": g3_benchmark["summary"],
        "verdict": verdict, "elapsed_s": report["elapsed_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
