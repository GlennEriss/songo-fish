#!/usr/bin/env python3
"""Lot 8 : ablation Policy/Value et calibration de Value."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from songo_ai.evaluation import (
    ArenaConfig,
    HybridPolicyValueEvaluator,
    SRNMCTSAgent,
    calibration_report,
    evaluate_raw_network_outputs,
    game_result_to_dict,
    generate_deterministic_openings,
    model_parameter_fingerprint,
    opening_to_dict,
    run_paired_arena,
    select_extended_d_lab_benchmark,
    summarize_arena,
    value_distribution_statistics,
)
from songo_ai.generation import SelfPlayConfig, SelfPlayRunner
from songo_ai.model import (
    SRNBatchCollator,
    SRNConfig,
    SongoRelationalNetwork,
    load_srn_checkpoint,
    set_training_seed,
)
from songo_ai.search import MCTSConfig, SongoMCTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_dir = Path("data/experiments/lot6_srn_seed_20260924/checkpoints")
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=checkpoint_dir / "best_validation_checkpoint.pt",
    )
    parser.add_argument(
        "--last-checkpoint", type=Path, default=checkpoint_dir / "last_checkpoint.pt"
    )
    parser.add_argument(
        "--d-lab", type=Path, default=Path("data/dataset_v001_10k/test.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/lot8_ablation_seed_20260924"),
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--calibration-seed", type=int, default=2026092408)
    parser.add_argument("--lab-positions", type=int, default=350)
    parser.add_argument("--calibration-games", type=int, default=32)
    parser.add_argument("--openings", type=int, default=32)
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--neutral-budget", type=int, default=8)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--mcts-analysis-positions", type=int, default=5)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join(str(part) for part in (base_seed,) + parts)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def load_models(args):
    best = load_srn_checkpoint(args.best_checkpoint, device="cpu")
    last = load_srn_checkpoint(args.last_checkpoint, device="cpu")
    if best.payload["srn_config"] != last.payload["srn_config"]:
        raise RuntimeError("G1 checkpoints do not share the same SRNConfig")
    if best.payload["seed"] != last.payload["seed"] or best.payload["seed"] != args.seed:
        raise RuntimeError("checkpoint/G0 seeds do not match")
    config = SRNConfig(**best.payload["srn_config"])
    set_training_seed(args.seed)
    g0 = SongoRelationalNetwork(config)
    return config, {"G0": g0, "G1-best": best.model, "G1-last": last.model}, best, last


def make_evaluator(name: str, models: dict[str, torch.nn.Module]):
    definitions = {
        "P0V0": ("G0", "G0", False),
        "P1V1": ("G1-best", "G1-best", False),
        "P0V1": ("G0", "G1-best", False),
        "P1V0": ("G1-best", "G0", False),
        "P0Vneutral": ("G0", None, True),
        "P1Vneutral": ("G1-best", None, True),
    }
    policy_name, value_name, neutral = definitions[name]
    return HybridPolicyValueEvaluator(
        models[policy_name],
        models[value_name] if value_name else None,
        neutral_value=neutral,
        name=name,
    )


def predict_values(models, examples, batch_size: int = 512) -> dict[str, list[float]]:
    predictions = {name: [] for name in models}
    collator = SRNBatchCollator()
    for start in range(0, len(examples), batch_size):
        batch = collator(examples[start : start + batch_size])
        for name, model in models.items():
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    _, values = model(batch.graph)
            finally:
                model.train(was_training)
            predictions[name].extend(float(value) for value in values.tolist())
    return predictions


def build_calibration_set(g0, args):
    config = SelfPlayConfig(
        games=args.calibration_games,
        max_game_plies=args.max_plies,
        repetition_limit=3,
        target_temperature=1.0,
        action_temperature=1.0,
        temperature_drop_ply=30,
        late_action_temperature=0.0,
        seed=args.calibration_seed,
        generation=0,
        checkpoint_id="G0-lot8-calibration-only",
        include_truncated_examples=False,
        mcts=MCTSConfig(
            num_simulations=2,
            c_puct=args.c_puct,
            add_root_noise=True,
        ),
    )
    run = SelfPlayRunner(g0, config).generate()
    if run.statistics.games_truncated:
        raise RuntimeError("calibration generation contains truncated games")
    examples = []
    records = []
    for game in run.games:
        for example in game.examples:
            if example.value_target is None:
                raise RuntimeError("terminal calibration example has no z target")
            ply = int(example.metadata["ply"])
            distance = game.num_plies - ply
            examples.append(example)
            records.append(
                {
                    "game_id": game.game_id,
                    "ply": ply,
                    "distance_to_observed_terminal": distance,
                    "state": {
                        "board": list(example.state.board),
                        "player_to_move": example.state.player_to_move,
                    },
                    "value_target": example.value_target,
                    "winner": example.metadata["winner"],
                }
            )
    return run, tuple(examples), records


def raw_diagnostics(models, positions):
    report = evaluate_raw_network_outputs(models, positions)
    for name, model_report in report["models"].items():
        values = [record["value"] for record in model_report["positions"]]
        model_report["value_distribution"] = value_distribution_statistics(values)
    return report


def run_comparison(
    left_name,
    right_name,
    *,
    budget,
    models,
    openings,
    arena_config,
    c_puct,
):
    left_evaluator = make_evaluator(left_name, models)
    right_evaluator = make_evaluator(right_name, models)
    results = run_paired_arena(
        SRNMCTSAgent(left_name, left_evaluator, budget, c_puct),
        SRNMCTSAgent(right_name, right_evaluator, budget, c_puct),
        openings,
        config=arena_config,
    )
    summary = {**asdict(summarize_arena(results, config=arena_config)), "budget": budget}
    return results, summary


def analyze_mcts_propagation(models, positions, raw_report, args):
    pair = raw_report["pairwise"]["G0__vs__G1-best"]
    low_js_disagreements = sorted(
        (record for record in pair["per_position"] if record["argmax_disagreement"]),
        key=lambda record: (record["jensen_shannon"], record["position_id"]),
    )
    selected_ids = {
        record["position_id"] for record in low_js_disagreements[: args.mcts_analysis_positions]
    }
    selected = [position for position in positions if position.position_id in selected_ids]
    evaluator_names = ("P0V0", "P1V1", "P0V1", "P1V0", "P0Vneutral", "P1Vneutral")
    records = []
    for position in selected:
        raw_pair = next(
            record for record in low_js_disagreements if record["position_id"] == position.position_id
        )
        entry = {
            "position_id": position.position_id,
            "state": {
                "board": list(position.state.board),
                "player_to_move": position.state.player_to_move,
            },
            "legal_mask": list(position.legal_mask),
            "raw_policy_jensen_shannon": raw_pair["jensen_shannon"],
            "raw_argmax": {"G0": raw_pair["left_argmax"], "G1-best": raw_pair["right_argmax"]},
            "budgets": {},
        }
        for budget in args.budgets:
            budget_results = {}
            for evaluator_name in evaluator_names:
                evaluator = make_evaluator(evaluator_name, models)
                before = evaluator.counters
                result = SongoMCTS(
                    evaluator,
                    config=MCTSConfig(
                        num_simulations=budget,
                        c_puct=args.c_puct,
                        add_root_noise=False,
                        seed=derived_seed(args.seed, "mcts-propagation", position.position_id, budget),
                    ),
                ).search(position.state, policy_temperature=0.0)
                after = evaluator.counters
                budget_results[evaluator_name] = {
                    "root_priors": list(result.root_priors),
                    "visit_counts": list(result.visit_counts),
                    "root_q": list(result.root_q_values),
                    "selected_action": result.selected_action,
                    "model_evaluations": after.model_evaluations - before.model_evaluations,
                    "network_forward_calls": after.network_forward_calls - before.network_forward_calls,
                }
            entry["budgets"][str(budget)] = budget_results
        records.append(entry)
    return records


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    srn_config, models, best, last = load_models(args)
    initial_fingerprints = {
        name: model_parameter_fingerprint(model) for name, model in models.items()
    }
    print("[lot8] modeles charges, poids figes", flush=True)

    positions = select_extended_d_lab_benchmark(
        args.d_lab, seed=args.seed, target_count=args.lab_positions
    )
    raw_report = raw_diagnostics(models, positions)
    raw_report["source"] = {
        "path": str(args.d_lab),
        "sha256": sha256(args.d_lab),
        "dataset_family": "D_LAB",
        "used_for_training": False,
        "selection": "structural-bin round-robin; no teacher labels or model outputs",
    }
    write_json(args.output / "expanded_raw_diagnostics.json", raw_report)
    print(f"[lot8] batterie D_LAB analysee: {len(positions)} positions", flush=True)

    calibration_run, calibration_examples, calibration_records = build_calibration_set(
        models["G0"], args
    )
    calibration_predictions = predict_values(models, calibration_examples)
    targets = [float(example.value_target) for example in calibration_examples]
    distances = [record["distance_to_observed_terminal"] for record in calibration_records]
    players = [example.state.player_to_move for example in calibration_examples]
    calibration = calibration_report(
        calibration_predictions, targets, distances, players
    )
    calibration_payload = {
        "generation": {
            "purpose": "D_EVAL_VALUE only; never added to D_RL",
            "seed": args.calibration_seed,
            "games": asdict(calibration_run.statistics),
            "mcts_simulations_per_move": 2,
            "root_noise": True,
        },
        "positions": len(calibration_examples),
        "metrics": calibration,
    }
    write_json(args.output / "value_calibration.json", calibration_payload)
    calibration_path = args.output / "calibration_trajectories.jsonl"
    calibration_path.write_text(
        "".join(json.dumps(record) + "\n" for record in calibration_records),
        encoding="utf-8",
    )
    print(
        f"[lot8] calibration: {len(calibration_run.games)} parties, "
        f"{len(calibration_examples)} positions terminalement etiquetees",
        flush=True,
    )

    openings = generate_deterministic_openings(
        prefix_lengths=list(range(args.openings)), seed=args.seed + 8
    )
    write_json(
        args.output / "openings.json",
        {
            "seed": args.seed + 8,
            "prefix_lengths": list(range(args.openings)),
            "uses_model_outputs": False,
            "openings": [opening_to_dict(opening) for opening in openings],
        },
    )
    arena_config = ArenaConfig(
        max_plies=args.max_plies,
        repetition_limit=3,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    priority_pairs = (
        ("P0V0", "P0V1", "value_effect_with_P0"),
        ("P0V0", "P1V0", "policy_effect_with_V0"),
        ("P1V1", "P0V1", "policy_effect_with_V1"),
        ("P1V1", "P1V0", "value_effect_with_P1"),
    )
    neutral_pairs = (
        ("P0V0", "P0Vneutral", "V0_vs_neutral"),
        ("P1V1", "P1Vneutral", "V1_vs_neutral"),
        ("P0Vneutral", "P1Vneutral", "policy_effect_with_neutral_value"),
    )
    summaries = {}
    games = []
    for budget in args.budgets:
        for left, right, effect in priority_pairs:
            results, summary = run_comparison(
                left,
                right,
                budget=budget,
                models=models,
                openings=openings,
                arena_config=arena_config,
                c_puct=args.c_puct,
            )
            key = f"{left}__vs__{right}__budget-{budget}"
            summaries[key] = {**summary, "effect": effect, "kind": "priority"}
            games.extend(
                {**game_result_to_dict(result), "budget": budget, "effect": effect}
                for result in results
            )
            print(f"[lot8] {key}: {len(results)} parties", flush=True)
    for left, right, effect in neutral_pairs:
        results, summary = run_comparison(
            left,
            right,
            budget=args.neutral_budget,
            models=models,
            openings=openings,
            arena_config=arena_config,
            c_puct=args.c_puct,
        )
        key = f"{left}__vs__{right}__budget-{args.neutral_budget}"
        summaries[key] = {**summary, "effect": effect, "kind": "neutral_control"}
        games.extend(
            {
                **game_result_to_dict(result),
                "budget": args.neutral_budget,
                "effect": effect,
            }
            for result in results
        )
        print(f"[lot8] {key}: {len(results)} parties", flush=True)

    propagation = analyze_mcts_propagation(models, positions, raw_report, args)
    write_json(args.output / "mcts_propagation.json", propagation)
    write_json(args.output / "arena_summaries.json", summaries)
    (args.output / "arena_games.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in games), encoding="utf-8"
    )

    final_fingerprints = {
        name: model_parameter_fingerprint(model) for name, model in models.items()
    }
    if final_fingerprints != initial_fingerprints:
        raise RuntimeError("Lot 8 modified source model parameters")
    elapsed = time.perf_counter() - started_at
    report = {
        "lot": 8,
        "question": "Which learned head explains the strategic change, and is Value reliable?",
        "seed": args.seed,
        "srn_config": asdict(srn_config),
        "checkpoints": {
            "G1-best": {"path": str(args.best_checkpoint), "epoch": best.payload["epoch"]},
            "G1-last": {"path": str(args.last_checkpoint), "epoch": last.payload["epoch"]},
        },
        "expanded_battery": {
            "positions": len(positions),
            "source": raw_report["source"],
            "models": {
                name: {
                    "policy_entropy": data["policy_entropy"],
                    "policy_top1_margin": data["policy_top1_margin"],
                    "value_distribution": data["value_distribution"],
                }
                for name, data in raw_report["models"].items()
            },
            "pairwise": {
                name: {key: value for key, value in data.items() if key != "per_position"}
                for name, data in raw_report["pairwise"].items()
            },
        },
        "calibration": calibration_payload,
        "arena_protocol": {
            "openings": len(openings),
            "paired_games_per_opening": 2,
            "priority_budgets": args.budgets,
            "neutral_control_budget": args.neutral_budget,
            "c_puct": args.c_puct,
            "dirichlet": False,
            "temperature": 0,
            "bootstrap_unit": "opening_pair",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "arena": summaries,
        "mcts_propagation_positions": len(propagation),
        "parameter_integrity": {
            "before": initial_fingerprints,
            "after": final_fingerprints,
            "unchanged": True,
        },
        "totals": {
            "arena_games": len(games),
            "arena_mcts_simulations": sum(game["total_mcts_simulations"] for game in games),
            "arena_model_evaluations": sum(game["total_network_evaluations"] for game in games),
            "arena_network_forward_calls": sum(
                game["total_network_forward_calls"] for game in games
            ),
            "calibration_games": len(calibration_run.games),
            "calibration_positions": len(calibration_examples),
            "elapsed_s": elapsed,
        },
        "no_training": True,
        "no_checkpoint_promoted": True,
    }
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
