#!/usr/bin/env python3
"""Lot 12 : corpus autonome G1/MCTS64, entraînement G2 et arène G2/G1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch

from songo_ai.dataset import read_d_rl_jsonl, write_d_rl_jsonl
from songo_ai.evaluation import (
    ArenaConfig,
    SRNMCTSAgent,
    game_result_to_dict,
    generate_deterministic_openings,
    model_parameter_fingerprint,
    opening_to_dict,
    policy_corpus_metrics,
    repeated_state_metrics,
    run_paired_arena,
    summarize_arena,
)
from songo_ai.generation import SelfPlayConfig, SelfPlayRunner
from songo_ai.model import (
    SRNBatchCollator,
    SRNConfig,
    SRNTrainingConfig,
    load_srn_checkpoint,
    mask_policy_logits,
    policy_probabilities,
    train_srn_from_d_rl,
)
from songo_ai.search import MCTSConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g1-checkpoint",
        type=Path,
        default=Path(
            "data/experiments/lot6_srn_seed_20260924/checkpoints/"
            "best_validation_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--pilot-d-rl",
        type=Path,
        default=Path("data/d_rl/lot11_g1_selected_seed_20260924.jsonl"),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/d_rl/lot12_g1_to_g2_mcts64_seed_20261200.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/lot12_g2_seed_20261200"),
    )
    parser.add_argument("--seed", type=int, default=20261200)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--arena-openings", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--reuse-corpus", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selfplay_config(args, checkpoint_sha: str) -> SelfPlayConfig:
    return SelfPlayConfig(
        games=args.games,
        max_game_plies=args.max_plies,
        repetition_limit=3,
        target_temperature=1.0,
        action_temperature=1.0,
        temperature_drop_ply=30,
        late_action_temperature=0.0,
        seed=args.seed,
        generation=2,
        checkpoint_id=f"G1-best-{checkpoint_sha[:12]}",
        provenance={
            "generation_lineage": "G1_to_G2",
            "generator": "G1-best",
            "generator_checkpoint_sha256": checkpoint_sha,
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


def validate_roundtrip(path, original) -> dict:
    loaded = read_d_rl_jsonl(path)
    valid = len(loaded) == len(original) and all(left == right for left, right in zip(original, loaded))
    return {"valid": valid, "examples": len(loaded), "sha256": sha256(path)}


def corpus_game_statistics(examples) -> dict:
    games = defaultdict(list)
    for example in examples:
        games[str(example.metadata["game_id"])].append(example)
    rows = []
    for game_id, values in games.items():
        values.sort(key=lambda value: int(value.metadata["ply"]))
        rows.append(
            {
                "game_id": game_id,
                "status": values[0].metadata["status"],
                "winner": values[0].metadata["winner"],
                "length": len(values),
            }
        )
    lengths = [row["length"] for row in rows]
    wins = Counter(row["winner"] for row in rows if row["winner"] is not None)
    statuses = Counter(row["status"] for row in rows)
    return {
        "games": len(rows),
        "positions": len(examples),
        "wins": {"P1": wins[1], "P2": wins[2], "draws": wins[0]},
        "statuses": dict(statuses),
        "truncations": sum(status.startswith("TRUNCATED") for status in statuses.elements()),
        "length": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "minimum": min(lengths),
            "maximum": max(lengths),
        },
        "terminal_labeled": sum(example.value_target is not None for example in examples),
        "unlabeled_truncated": sum(example.value_target is None for example in examples),
    }


def validate_corpus_contract(examples, args, checkpoint_sha) -> dict:
    failures = Counter()
    for example in examples:
        metadata = example.metadata
        if metadata.get("generation_lineage") != "G1_to_G2":
            failures["generation_lineage"] += 1
        if metadata.get("generator_checkpoint_sha256") != checkpoint_sha:
            failures["generator_hash"] += 1
        if metadata.get("selfplay_seed") != args.seed:
            failures["seed"] += 1
        if metadata.get("mcts_simulations") != args.simulations:
            failures["mcts_simulations"] += 1
        if metadata.get("c_puct") != 1.5:
            failures["c_puct"] += 1
        if sum(example.visit_counts) != args.simulations:
            failures["visit_total"] += 1
        if any(count for count, legal in zip(example.visit_counts, example.legal_mask) if not legal):
            failures["illegal_visits"] += 1
    return {"valid": not failures, "failures": dict(failures)}


def quality_gate(pilot, candidate, *, serialization_valid, contract_valid) -> dict:
    checks = {
        "one_hot_no_major_regression": candidate["policy"]["one_hot_fraction"] <= pilot["policy"]["one_hot_fraction"] + 0.05,
        "support_no_major_regression": candidate["policy"]["support_mean"] >= pilot["policy"]["support_mean"] - 0.5,
        "entropy_no_major_regression": candidate["policy"]["entropy_mean"] >= pilot["policy"]["entropy_mean"] - 0.15,
        "S0_stability_no_major_regression": candidate["repeated"]["initial_state"]["pairwise_jensen_shannon_mean"] <= pilot["repeated"]["initial_state"]["pairwise_jensen_shannon_mean"] + 0.05,
        "serialization_valid": serialization_valid,
        "contract_valid": contract_valid,
        "terminal_signal_present": candidate["game_statistics"]["terminal_labeled"] > 0,
    }
    return {
        "TRAINING_AUTHORIZED": all(checks.values()),
        "checks": checks,
        "rationale": "Bornes de non-régression majeure, absolues autour du pilote Lot 11; elles tolèrent la variation statistique normale.",
    }


def fixed_batch_outputs(model, examples, count=16):
    ranked = sorted(
        examples,
        key=lambda e: hashlib.sha256(repr((e.state.board, e.state.player_to_move)).encode()).hexdigest(),
    )[:count]
    batch = SRNBatchCollator()(ranked)
    model.eval()
    with torch.no_grad():
        policy, value = model(batch.graph)
    return ranked, batch, policy.detach().clone(), value.detach().clone()


def write_metrics_csv(path: Path, history) -> None:
    fields = [
        "epoch", "global_step", "split", "total_loss", "policy_loss",
        "policy_cross_entropy", "policy_kl", "policy_top1_accuracy", "value_loss",
        "value_mae", "value_sign_accuracy", "examples", "labeled_values",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in history:
            for split in ("train", "validation"):
                writer.writerow({"epoch": record.epoch, "global_step": record.global_step, "split": split, **asdict(getattr(record, split))})


def sanity_report(g1, g2, examples, count=12) -> list[dict]:
    selected, batch, _, _ = fixed_batch_outputs(g1, examples, count)
    with torch.no_grad():
        g1_logits, g1_value = g1(batch.graph)
        g2_logits, g2_value = g2(batch.graph)
        g1_policy = policy_probabilities(g1_logits, batch.legal_mask)
        g2_policy = policy_probabilities(g2_logits, batch.legal_mask)
    return [
        {
            "state": {"board": list(example.state.board), "player_to_move": example.state.player_to_move},
            "legal_actions": [i for i, legal in enumerate(example.legal_mask) if legal],
            "pi_target": list(example.policy_target),
            "policy_G1": g1_policy[i].tolist(),
            "policy_G2": g2_policy[i].tolist(),
            "z": example.value_target,
            "value_G1": float(g1_value[i]),
            "value_G2": float(g2_value[i]),
        }
        for i, example in enumerate(selected)
    ]


def arena(args, g1, g2):
    prefix_lengths = [0] + [1 + (index % 30) for index in range(1, args.arena_openings)]
    openings = generate_deterministic_openings(prefix_lengths=prefix_lengths, seed=args.seed + 1)
    write_json(args.output / "arena_openings.json", {"seed": args.seed + 1, "openings": [opening_to_dict(o) for o in openings]})
    config = ArenaConfig(
        max_plies=args.max_plies,
        repetition_limit=3,
        seed=args.seed + 2,
        bootstrap_samples=args.bootstrap_samples,
    )
    summaries = {}
    games = []
    for budget in (32, 64):
        results = run_paired_arena(
            SRNMCTSAgent("G2-best-validation", g2, budget),
            SRNMCTSAgent("G1-best", g1, budget),
            openings,
            config=config,
        )
        summaries[str(budget)] = asdict(summarize_arena(results, config=config))
        games.extend({**game_result_to_dict(result), "budget": budget} for result in results)
        print(f"[lot12] arène budget={budget}: {len(results)} parties", flush=True)
    (args.output / "arena_games.jsonl").write_text("".join(json.dumps(game) + "\n" for game in games), encoding="utf-8")
    write_json(args.output / "arena_summaries.json", summaries)
    return openings, summaries, games


def progression_verdict(summaries) -> dict:
    rates = [summaries[str(budget)]["score_rate_a_terminal"] for budget in (32, 64)]
    cis = [summaries[str(budget)]["paired_bootstrap_ci"] for budget in (32, 64)]
    if all(rate is not None and rate > 0.5 for rate in rates) and all(ci and ci[0] > 0.5 for ci in cis):
        category = "C"
        label = "progression cohérente dans les conditions expérimentales testées"
    elif any(rate is not None and rate > 0.5 for rate in rates):
        category = "B"
        label = "signal de progression mais incertitude encore importante"
    else:
        category = "A"
        label = "pas de progression stratégique détectable"
    return {"category": category, "label": label, "rule": "C: scores > 0.5 et bornes basses IC95 > 0.5 aux deux budgets; B: au moins un score > 0.5; sinon A."}


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    loaded_g1 = load_srn_checkpoint(args.g1_checkpoint, device="cpu")
    g1 = loaded_g1.model
    g1_sha = sha256(args.g1_checkpoint)
    parent_before = model_parameter_fingerprint(g1)
    config = selfplay_config(args, g1_sha)

    generation_started = time.perf_counter()
    if args.reuse_corpus:
        if not args.corpus.exists():
            raise FileNotFoundError("--reuse-corpus requested but corpus does not exist")
        examples = read_d_rl_jsonl(args.corpus)
        generation = {"reused": True, "elapsed_s": 0.0}
        roundtrip = {"valid": True, "examples": len(examples), "sha256": sha256(args.corpus)}
    else:
        if args.corpus.exists():
            raise FileExistsError("immutable Lot 12 corpus already exists; use --reuse-corpus")
        run = SelfPlayRunner(g1, config).generate()
        write_d_rl_jsonl(args.corpus, run.examples)
        examples = read_d_rl_jsonl(args.corpus)
        roundtrip = validate_roundtrip(args.corpus, run.examples)
        generation = {"reused": False, "elapsed_s": time.perf_counter() - generation_started, "runner_statistics": asdict(run.statistics)}
    if model_parameter_fingerprint(g1) != parent_before:
        raise RuntimeError("self-play mutated G1")

    game_stats = corpus_game_statistics(examples)
    contract = validate_corpus_contract(examples, args, g1_sha)
    corpus_sha = sha256(args.corpus)
    manifest = {
        "dataset_family": "D_RL",
        "generation_lineage": "G1_to_G2",
        "path": str(args.corpus),
        "sha256": corpus_sha,
        "immutable_after_validation": True,
        "generator": {"name": "G1-best", "checkpoint": str(args.g1_checkpoint), "checkpoint_sha256": g1_sha, "source_epoch": loaded_g1.payload["epoch"]},
        "selfplay_config": asdict(config),
        "statistics": game_stats,
        "contract_validation": contract,
        "roundtrip": roundtrip,
        "generation_run": generation,
    }
    write_json(args.output / "corpus_manifest.json", manifest)

    pilot_examples = read_d_rl_jsonl(args.pilot_d_rl)
    pilot_quality = {"policy": policy_corpus_metrics(pilot_examples), "repeated": repeated_state_metrics(pilot_examples)}
    corpus_quality = {
        "policy": policy_corpus_metrics(examples),
        "repeated": repeated_state_metrics(examples),
        "game_statistics": game_stats,
    }
    gate = quality_gate(pilot_quality, corpus_quality, serialization_valid=roundtrip["valid"], contract_valid=contract["valid"])
    write_json(args.output / "quality_gate.json", {"pilot_lot11": pilot_quality, "lot12": corpus_quality, "gate": gate})
    if not gate["TRAINING_AUTHORIZED"]:
        write_json(args.output / "report.json", {"TRAINING_AUTHORIZED": False, "manifest": manifest, "quality": corpus_quality, "gate": gate})
        raise SystemExit("Lot 12 stopped before optimizer.step: corpus quality gate failed")
    print(f"[lot12] corpus validé: {len(examples)} positions, entraînement autorisé", flush=True)

    srn_config = SRNConfig(**loaded_g1.payload["srn_config"])
    training_config = SRNTrainingConfig(
        epochs=args.epochs,
        batch_size=256,
        learning_rate=0.003,
        weight_decay=0.0001,
        lambda_policy=1.0,
        lambda_value=1.0,
        gradient_clip_norm=1.0,
        validation_fraction=0.2,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=1e-4,
        seed=args.seed,
    )
    fixed_examples, fixed_batch, parent_policy, parent_value = fixed_batch_outputs(g1, examples)
    clone = load_srn_checkpoint(args.g1_checkpoint).model
    clone.eval()
    with torch.no_grad():
        clone_policy, clone_value = clone(fixed_batch.graph)
    initialization_check = {
        "policy_exact": torch.equal(parent_policy, clone_policy),
        "value_exact": torch.equal(parent_value, clone_value),
        "batch_examples": len(fixed_examples),
    }
    if not all((initialization_check["policy_exact"], initialization_check["value_exact"])):
        raise RuntimeError("G2 initialization is not exactly G1-best")

    training_started = time.perf_counter()
    lineage = {
        "generation": "G2",
        "parent": "G1-best",
        "parent_checkpoint": str(args.g1_checkpoint),
        "parent_checkpoint_sha256": g1_sha,
        "generator": "G1-best",
        "d_rl_sha256": corpus_sha,
    }
    training = train_srn_from_d_rl(
        args.corpus,
        args.output / "training",
        srn_config=srn_config,
        training_config=training_config,
        initial_checkpoint=args.g1_checkpoint,
        lineage=lineage,
    )
    training_elapsed = time.perf_counter() - training_started
    if set(training.train_game_ids) & set(training.validation_game_ids):
        raise RuntimeError("train/validation game leakage")
    write_metrics_csv(args.output / "training" / "metrics.csv", training.history)
    best_loaded = load_srn_checkpoint(training.best_validation_checkpoint)
    last_loaded = load_srn_checkpoint(training.last_checkpoint)
    training.model.eval()
    last_loaded.model.eval()
    with torch.no_grad():
        expected_last = training.model(fixed_batch.graph)
        actual_last = last_loaded.model(fixed_batch.graph)
    reload_check = {
        "last_policy_exact": torch.equal(expected_last[0], actual_last[0]),
        "last_value_exact": torch.equal(expected_last[1], actual_last[1]),
        "best_lineage_exact": best_loaded.payload["lineage"] == lineage,
        "last_lineage_exact": last_loaded.payload["lineage"] == lineage,
    }
    if not all(reload_check.values()):
        raise RuntimeError("G2 checkpoint reload/provenance failed")
    sanity = sanity_report(g1, best_loaded.model, examples)
    write_json(args.output / "sanity_fixed_positions.json", sanity)
    if model_parameter_fingerprint(g1) != parent_before:
        raise RuntimeError("training mutated the parent G1 object")

    arena_started = time.perf_counter()
    fingerprints_before_arena = {"G1": model_parameter_fingerprint(g1), "G2": model_parameter_fingerprint(best_loaded.model)}
    openings, arena_summaries, arena_games = arena(args, g1, best_loaded.model)
    fingerprints_after_arena = {"G1": model_parameter_fingerprint(g1), "G2": model_parameter_fingerprint(best_loaded.model)}
    if fingerprints_before_arena != fingerprints_after_arena:
        raise RuntimeError("arena mutated model parameters")
    verdict = progression_verdict(arena_summaries)
    best_record = next(record for record in training.history if record.epoch == training.best_epoch)
    report = {
        "lot": 12,
        "question": "En partant de G1-best et de 500 parties autonomes G1/MCTS64, G2 apprend-il et progresse-t-il face à G1 ?",
        "generation": {**generation, "generator_checkpoint": str(args.g1_checkpoint), "generator_sha256": g1_sha, "selfplay_config": asdict(config), "game_statistics": game_stats},
        "corpus": {"manifest": str(args.output / "corpus_manifest.json"), "sha256": corpus_sha, "quality": corpus_quality, "pilot_lot11": pilot_quality},
        "TRAINING_AUTHORIZED": True,
        "gate": gate,
        "training": {
            "configuration": asdict(training_config),
            "initialization_check": initialization_check,
            "train_games": len(training.train_game_ids),
            "validation_games": len(training.validation_game_ids),
            "train_positions": training.train_positions,
            "validation_positions": training.validation_positions,
            "no_game_leakage": not bool(set(training.train_game_ids) & set(training.validation_game_ids)),
            "train_game_ids": list(training.train_game_ids),
            "validation_game_ids": list(training.validation_game_ids),
            "epoch_0": asdict(training.history[0]),
            "best_epoch": training.best_epoch,
            "best_validation": asdict(best_record),
            "last": asdict(training.history[-1]),
            "stopped_early": training.stopped_early,
            "stop_epoch": training.stop_epoch,
            "global_step": training.global_step,
            "metrics_json": str(training.metrics_path),
            "metrics_csv": str(args.output / "training" / "metrics.csv"),
            "elapsed_s": training_elapsed,
        },
        "checkpoints": {
            "G2_best_validation": str(training.best_validation_checkpoint),
            "G2_last": str(training.last_checkpoint),
            "best_sha256": sha256(training.best_validation_checkpoint),
            "last_sha256": sha256(training.last_checkpoint),
            "reload": reload_check,
            "lineage": lineage,
        },
        "arena": {
            "openings": len(openings),
            "games": len(arena_games),
            "budgets": [32, 64],
            "dirichlet": False,
            "temperature": 0.0,
            "summaries": arena_summaries,
            "elapsed_s": time.perf_counter() - arena_started,
            "parameter_integrity": {"before": fingerprints_before_arena, "after": fingerprints_after_arena, "unchanged": True},
        },
        "conclusions": {
            "level_1_learning": "Comparer epoch 0, best-validation et courbes; une baisse ne prouve pas la force de jeu.",
            "level_2_behavioral_change": "Les sorties fixes G1/G2 sont enregistrées séparément.",
            "level_3_strategic_progress": verdict,
        },
        "constraints": {"engine_modified": False, "canonicalization": False, "mirror_augmentation": False, "teacher": False, "G3_trained": False, "automatic_promotion": False},
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output / "report.json", report)
    print(json.dumps({"report": str(args.output / "report.json"), "best_epoch": training.best_epoch, "arena": arena_summaries, "verdict": verdict, "elapsed_s": report["elapsed_s"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
