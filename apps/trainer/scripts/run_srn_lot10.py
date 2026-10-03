#!/usr/bin/env python3
"""Lot 10 : génération D_RL 8 sims, gate, entraînements séparés et arène."""

from __future__ import annotations

import argparse
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
    audit_state_collection,
    audit_transition_equivariance,
    evaluate_corpus_quality_gate,
    game_result_to_dict,
    generate_deterministic_openings,
    jensen_shannon,
    model_parameter_fingerprint,
    opening_to_dict,
    policy_corpus_metrics,
    policy_entropy,
    repeated_state_metrics,
    run_paired_arena,
    summarize_arena,
    SWAP_KEEP_LOCAL,
)
from songo_ai.generation import SelfPlayConfig, SelfPlayRunner
from songo_ai.model import (
    SRNConfig,
    SRNTrainingConfig,
    SongoRelationalNetwork,
    load_srn_checkpoint,
    set_training_seed,
    train_srn_from_d_rl,
)
from songo_ai.search import MCTSConfig, SongoMCTS


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
        "--old-d-rl", type=Path, default=Path("data/d_rl/pilot_lot5_seed_20260924.jsonl")
    )
    parser.add_argument(
        "--g0-d-rl", type=Path, default=Path("data/d_rl/lot10_g0_8_seed_20260924.jsonl")
    )
    parser.add_argument(
        "--g1-d-rl", type=Path, default=Path("data/d_rl/lot10_g1_8_seed_20260924.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/lot10_drl8_seed_20260924"),
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--selfplay-seed", type=int, default=20261008)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--control-positions", type=int, default=24)
    parser.add_argument("--control-repetitions", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--arena-openings", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--reuse-generated", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_seed(base: int, *parts: object) -> int:
    payload = ":".join(str(part) for part in (base,) + parts)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def load_models(args):
    loaded = load_srn_checkpoint(args.g1_checkpoint, device="cpu")
    if loaded.payload["seed"] != args.seed:
        raise RuntimeError("G1 checkpoint seed does not match Lot 10 seed")
    config = SRNConfig(**loaded.payload["srn_config"])
    set_training_seed(args.seed)
    g0 = SongoRelationalNetwork(config)
    return config, g0, loaded.model, loaded


def generation_config(args, checkpoint_id: str) -> SelfPlayConfig:
    return SelfPlayConfig(
        games=args.games,
        max_game_plies=args.max_plies,
        repetition_limit=3,
        target_temperature=1.0,
        action_temperature=1.0,
        temperature_drop_ply=30,
        late_action_temperature=0.0,
        seed=args.selfplay_seed,
        generation=10,
        checkpoint_id=checkpoint_id,
        include_truncated_examples=True,
        mcts=MCTSConfig(
            num_simulations=args.simulations,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            add_root_noise=True,
        ),
    )


def validate_roundtrip(path: Path, original) -> dict:
    loaded = read_d_rl_jsonl(path)
    valid = len(loaded) == len(original) and all(
        left.state == right.state
        and left.legal_mask == right.legal_mask
        and left.visit_counts == right.visit_counts
        and left.policy_target == right.policy_target
        and left.value_target == right.value_target
        and dict(left.metadata) == dict(right.metadata)
        for left, right in zip(original, loaded)
    )
    return {"valid": valid, "examples": len(loaded), "sha256": sha256(path)}


def generate_corpus(name, model, path, config, *, reuse=False):
    before = model_parameter_fingerprint(model)
    if reuse and path.exists():
        examples = read_d_rl_jsonl(path)
        return None, examples, {
            "reused": True,
            "path": str(path),
            "sha256": sha256(path),
            "parameter_fingerprint": before,
        }
    started = time.perf_counter()
    run = SelfPlayRunner(model, config).generate()
    write_d_rl_jsonl(path, run.examples)
    roundtrip = validate_roundtrip(path, run.examples)
    after = model_parameter_fingerprint(model)
    if before != after:
        raise RuntimeError(f"self-play modified {name} parameters")
    return run, read_d_rl_jsonl(path), {
        "reused": False,
        "path": str(path),
        "sha256": sha256(path),
        "elapsed_s": time.perf_counter() - started,
        "statistics": asdict(run.statistics),
        "roundtrip": roundtrip,
        "parameter_fingerprint_before": before,
        "parameter_fingerprint_after": after,
    }


def game_statistics(examples) -> dict:
    games = defaultdict(list)
    for example in examples:
        games[str(example.metadata["game_id"])].append(example)
    rows = []
    for game_id, values in sorted(games.items()):
        values.sort(key=lambda item: int(item.metadata["ply"]))
        rows.append(
            {
                "game_id": game_id,
                "status": values[0].metadata["status"],
                "winner": values[0].metadata["winner"],
                "length": len(values),
                "p1_positions": sum(value.state.player_to_move == 1 for value in values),
                "p2_positions": sum(value.state.player_to_move == 2 for value in values),
            }
        )
    wins = Counter(row["winner"] for row in rows if row["winner"] is not None)
    statuses = Counter(row["status"] for row in rows)
    by_player = {
        player: [
            float(example.value_target)
            for example in examples
            if example.state.player_to_move == player and example.value_target is not None
        ]
        for player in (1, 2)
    }
    lengths = {
        label: [row["length"] for row in rows if row["winner"] == winner]
        for label, winner in (("P1", 1), ("P2", 2), ("DRAW", 0))
    }
    return {
        "games": len(rows),
        "wins": {"P1": wins[1], "P2": wins[2], "draws": wins[0]},
        "statuses": dict(sorted(statuses.items())),
        "length_by_winner": {
            label: {
                "count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for label, values in lengths.items()
        },
        "positions": {
            "P1": sum(example.state.player_to_move == 1 for example in examples),
            "P2": sum(example.state.player_to_move == 2 for example in examples),
        },
        "mean_z": {
            "P1": statistics.fmean(by_player[1]) if by_player[1] else None,
            "P2": statistics.fmean(by_player[2]) if by_player[2] else None,
        },
        "terminal_labeled_examples": sum(example.value_target is not None for example in examples),
        "unlabeled_truncated_examples": sum(example.value_target is None for example in examples),
    }


def asymmetry_statistics(examples) -> dict:
    states = [(str(example.metadata["game_id"]), example.state) for example in examples]
    potential = audit_state_collection(states, SWAP_KEEP_LOCAL, max_examples=10)
    played_failures = []
    played_failure_count = 0
    affected_games = set()
    for example in examples:
        record = audit_transition_equivariance(
            example.state,
            int(example.metadata["action_played"]),
            SWAP_KEEP_LOCAL,
        )
        if not record["success"]:
            played_failure_count += 1
            affected_games.add(str(example.metadata["game_id"]))
            if len(played_failures) < 20:
                played_failures.append(
                    {
                        "game_id": example.metadata["game_id"],
                        "ply": example.metadata["ply"],
                        "action": example.metadata["action_played"],
                        "state": list(example.state.board),
                        "failures": record["failures"],
                    }
                )
    return {
        "potential": potential,
        "played_non_equivariant_actions": played_failure_count,
        "games_with_played_non_equivariant_action": len(affected_games),
        "played_examples": played_failures,
    }


def select_control_examples(examples, count, seed):
    unique = {}
    for example in examples:
        key = (example.state.board, example.state.player_to_move)
        unique.setdefault(key, example)
    ranked = sorted(
        unique.values(),
        key=lambda example: hashlib.sha256(
            (f"{seed}:" + repr((example.state.board, example.state.player_to_move))).encode()
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError("not enough unique states for 8 vs 32 control")
    return ranked[:count]


def compare_8_vs_32(model, examples, repetitions, seed):
    rows = []
    for index, example in enumerate(examples):
        for repetition in range(repetitions):
            run_seed = derived_seed(seed, "8v32", index, repetition)
            results = {}
            for budget in (8, 32):
                results[budget] = SongoMCTS(
                    model,
                    config=MCTSConfig(
                        num_simulations=budget,
                        c_puct=1.5,
                        add_root_noise=True,
                        seed=run_seed,
                    ),
                ).search(example.state, policy_temperature=1.0)
            p8, p32 = results[8].policy, results[32].policy
            rows.append(
                {
                    "js": jensen_shannon(p8, p32),
                    "argmax_agreement": max(range(7), key=p8.__getitem__)
                    == max(range(7), key=p32.__getitem__),
                    "support_8": sum(value > 0 for value in p8),
                    "support_32": sum(value > 0 for value in p32),
                    "entropy_8": policy_entropy(p8),
                    "entropy_32": policy_entropy(p32),
                }
            )
    return {
        "states": len(examples),
        "repetitions": repetitions,
        "comparisons": len(rows),
        "jensen_shannon_mean": statistics.fmean(row["js"] for row in rows),
        "jensen_shannon_median": statistics.median(row["js"] for row in rows),
        "argmax_agreement": statistics.fmean(row["argmax_agreement"] for row in rows),
        "support_mean_8": statistics.fmean(row["support_8"] for row in rows),
        "support_mean_32": statistics.fmean(row["support_32"] for row in rows),
        "entropy_mean_8": statistics.fmean(row["entropy_8"] for row in rows),
        "entropy_mean_32": statistics.fmean(row["entropy_32"] for row in rows),
    }


def training_report(result) -> dict:
    best = next(record for record in result.history if record.epoch == result.best_epoch)
    return {
        "configuration": asdict(
            SRNTrainingConfig(**load_srn_checkpoint(result.last_checkpoint).payload["training_config"])
        ),
        "initialization": result.initialization,
        "train_games": len(result.train_game_ids),
        "validation_games": len(result.validation_game_ids),
        "train_positions": result.train_positions,
        "validation_positions": result.validation_positions,
        "epoch_0": asdict(result.history[0]),
        "best_validation": asdict(best),
        "last_epoch": asdict(result.history[-1]),
        "best_epoch": result.best_epoch,
        "stopped_early": result.stopped_early,
        "stop_epoch": result.stop_epoch,
        "last_checkpoint": str(result.last_checkpoint),
        "best_validation_checkpoint": str(result.best_validation_checkpoint),
        "train_game_ids": list(result.train_game_ids),
        "validation_game_ids": list(result.validation_game_ids),
    }


def run_arena(models, args):
    openings = generate_deterministic_openings(
        prefix_lengths=list(range(args.arena_openings)), seed=args.seed + 10
    )
    write_json(
        args.output / "arena_openings.json",
        {
            "seed": args.seed + 10,
            "openings": [opening_to_dict(opening) for opening in openings],
        },
    )
    config = ArenaConfig(
        max_plies=args.max_plies,
        repetition_limit=3,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    pairs = (
        ("G1-best", "G2-from-G0-data"),
        ("G1-best", "G2-from-G1-data"),
        ("G2-from-G0-data", "G2-from-G1-data"),
    )
    summaries = {}
    games = []
    for budget in (8, 32):
        for left, right in pairs:
            results = run_paired_arena(
                SRNMCTSAgent(left, models[left], budget),
                SRNMCTSAgent(right, models[right], budget),
                openings,
                config=config,
            )
            key = f"{left}__vs__{right}__budget-{budget}"
            summaries[key] = {**asdict(summarize_arena(results, config=config)), "budget": budget}
            games.extend(
                {**game_result_to_dict(result), "budget": budget, "comparison": key}
                for result in results
            )
            print(f"[lot10] arène {key}: {len(results)} parties", flush=True)
    write_json(args.output / "arena_summaries.json", summaries)
    (args.output / "arena_games.jsonl").write_text(
        "".join(json.dumps(game) + "\n" for game in games), encoding="utf-8"
    )
    return summaries, games


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    srn_config, g0, g1, g1_loaded = load_models(args)
    g1_sha = sha256(args.g1_checkpoint)
    configs = {
        "G0": generation_config(args, f"G0-seed-{args.seed}"),
        "G1-best": generation_config(args, f"G1-best-{g1_sha[:12]}"),
    }
    runs = {}
    corpora = {}
    generation = {}
    for name, model, path in (
        ("G0", g0, args.g0_d_rl),
        ("G1-best", g1, args.g1_d_rl),
    ):
        run, examples, report = generate_corpus(
            name,
            model,
            path,
            configs[name],
            reuse=args.reuse_generated,
        )
        runs[name] = run
        corpora[name] = examples
        generation[name] = {
            **report,
            "selfplay_config": asdict(configs[name]),
            "game_statistics": game_statistics(examples),
            "checkpoint_provenance": {
                "kind": "seeded_random" if name == "G0" else "checkpoint",
                "checkpoint_path": None if name == "G0" else str(args.g1_checkpoint),
                "checkpoint_sha256": None if name == "G0" else g1_sha,
                "model_fingerprint": model_parameter_fingerprint(model),
            },
        }
        print(
            f"[lot10] {name}: {len(examples)} positions dans {len(set(e.metadata['game_id'] for e in examples))} parties",
            flush=True,
        )

    old_examples = read_d_rl_jsonl(args.old_d_rl)
    baseline_policy = policy_corpus_metrics(old_examples)
    baseline_repeated = repeated_state_metrics(old_examples)
    quality = {"Lot5": {"policy": baseline_policy, "repeated": baseline_repeated}}
    gates = {}
    controls = {}
    asymmetry = {}
    for name, model in (("G0", g0), ("G1-best", g1)):
        examples = corpora[name]
        policy = policy_corpus_metrics(examples)
        repeated = repeated_state_metrics(examples)
        stats = generation[name]["game_statistics"]
        serialization_valid = bool(
            generation[name].get("roundtrip", {}).get("valid", args.reuse_generated)
        )
        gate = evaluate_corpus_quality_gate(
            baseline_policy,
            baseline_repeated,
            policy,
            repeated,
            serialization_valid=serialization_valid,
            terminal_labeled_examples=stats["terminal_labeled_examples"],
        )
        quality[name] = {"policy": policy, "repeated": repeated}
        gates[name] = gate
        fixed = select_control_examples(
            examples, args.control_positions, derived_seed(args.seed, name)
        )
        controls[name] = compare_8_vs_32(
            model, fixed, args.control_repetitions, derived_seed(args.seed, "control", name)
        )
        asymmetry[name] = asymmetry_statistics(examples)
    quality_report = {
        "generation": generation,
        "quality": quality,
        "gates": gates,
        "control_8_vs_32": controls,
        "asymmetry": asymmetry,
        "paired_seed_contract": {
            "base_seed": args.selfplay_seed,
            "derivation": "sha256(base_seed:game_index:ply:purpose), independent of model id",
            "same_parameters": True,
        },
        "no_canonicalization": True,
        "no_mirror_augmentation": True,
        "datasets_merged": False,
    }
    write_json(args.output / "quality_report.json", quality_report)
    if not all(gate["passed"] for gate in gates.values()):
        write_json(
            args.output / "report.json",
            {
                **quality_report,
                "training_started": False,
                "stop_reason": "quality gate failed",
                "elapsed_s": time.perf_counter() - started,
            },
        )
        raise SystemExit("Lot 10 stopped before training: quality gate failed")
    print("[lot10] gate qualité satisfait pour les deux corpus", flush=True)

    training_config = SRNTrainingConfig(
        epochs=args.epochs,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        validation_fraction=0.2,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=1e-4,
        seed=args.seed,
    )
    training_results = {}
    training_reports = {}
    for name, dataset_path, directory in (
        ("G2-from-G0-data", args.g0_d_rl, "g2_from_g0_data"),
        ("G2-from-G1-data", args.g1_d_rl, "g2_from_g1_data"),
    ):
        train_started = time.perf_counter()
        result = train_srn_from_d_rl(
            dataset_path,
            args.output / "training" / directory,
            srn_config=srn_config,
            training_config=training_config,
            initial_checkpoint=args.g1_checkpoint,
        )
        training_results[name] = result
        training_reports[name] = {
            **training_report(result),
            "elapsed_s": time.perf_counter() - train_started,
        }
        print(
            f"[lot10] {name}: best epoch {result.best_epoch}, stop epoch {result.stop_epoch}",
            flush=True,
        )
    initializations = {
        json.dumps(result.initialization, sort_keys=True)
        for result in training_results.values()
    }
    if len(initializations) != 1:
        raise RuntimeError("G2 experiments did not use the same initialization")

    arena_models = {"G1-best": g1_loaded.model}
    for name, result in training_results.items():
        arena_models[name] = load_srn_checkpoint(
            result.best_validation_checkpoint, device="cpu"
        ).model
    fingerprints_before = {
        name: model_parameter_fingerprint(model) for name, model in arena_models.items()
    }
    arena_summaries, arena_games = run_arena(arena_models, args)
    fingerprints_after = {
        name: model_parameter_fingerprint(model) for name, model in arena_models.items()
    }
    if fingerprints_before != fingerprints_after:
        raise RuntimeError("arena modified model parameters")

    report = {
        **quality_report,
        "training_started": True,
        "training": training_reports,
        "arena": {
            "summaries": arena_summaries,
            "games": len(arena_games),
            "budgets": [8, 32],
            "openings": args.arena_openings,
            "dirichlet": False,
            "temperature": 0,
            "parameter_integrity": {
                "before": fingerprints_before,
                "after": fingerprints_after,
                "unchanged": True,
            },
        },
        "elapsed_s": time.perf_counter() - started,
        "no_promotion": True,
        "no_multi_generation_loop": True,
    }
    write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                "gates": gates,
                "training": {
                    name: {
                        "best_epoch": result.best_epoch,
                        "stop_epoch": result.stop_epoch,
                        "stopped_early": result.stopped_early,
                    }
                    for name, result in training_results.items()
                },
                "arena_games": len(arena_games),
                "elapsed_s": report["elapsed_s"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
