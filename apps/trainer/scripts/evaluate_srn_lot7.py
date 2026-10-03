#!/usr/bin/env python3
"""Lot 7 : sorties brutes et arene controlee G0 / G1 du SRN."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from songo_ai.evaluation import (
    ArenaConfig,
    RandomLegalAgent,
    SRNMCTSAgent,
    evaluate_raw_network_outputs,
    game_result_to_dict,
    generate_deterministic_openings,
    model_parameter_fingerprint,
    opening_to_dict,
    run_paired_arena,
    select_d_lab_benchmark,
    summarize_arena,
)
from songo_ai.model import (
    DRLDataset,
    SRNConfig,
    SRNTrainingConfig,
    SongoRelationalNetwork,
    evaluate_srn,
    load_srn_checkpoint,
    make_srn_loader,
    set_training_seed,
    split_examples_by_game_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=Path(
            "data/experiments/lot6_srn_seed_20260924/checkpoints/"
            "best_validation_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--last-checkpoint",
        type=Path,
        default=Path(
            "data/experiments/lot6_srn_seed_20260924/checkpoints/last_checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--d-rl",
        type=Path,
        default=Path("data/d_rl/pilot_lot5_seed_20260924.jsonl"),
    )
    parser.add_argument(
        "--d-lab",
        type=Path,
        default=Path("data/dataset_v001_10k/test.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/lot7_g0_g1_seed_20260924"),
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 8, 32])
    parser.add_argument(
        "--prefix-lengths", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8, 13, 21]
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--repetition-limit", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-baseline-budget", type=int, default=8)
    parser.add_argument("--skip-random-baseline", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _assert_same_checkpoint_contract(best, last, expected_seed: int) -> SRNConfig:
    if best.payload["srn_config"] != last.payload["srn_config"]:
        raise RuntimeError("G1-best and G1-last use different SRNConfig values")
    if best.payload["training_config"] != last.payload["training_config"]:
        raise RuntimeError("G1-best and G1-last use different training contracts")
    if best.payload["seed"] != last.payload["seed"]:
        raise RuntimeError("checkpoint seeds do not match")
    if best.payload["seed"] != expected_seed:
        raise RuntimeError("requested G0 seed does not match Lot 6 checkpoints")
    if best.payload["epoch"] != 8 or last.payload["epoch"] != 12:
        raise RuntimeError("unexpected Lot 6 checkpoint epochs")
    return SRNConfig(**best.payload["srn_config"])


def _verify_g0_reconstruction(g0, checkpoint_payload: dict, d_rl_path: Path) -> dict:
    training_data = dict(checkpoint_payload["training_config"])
    training_data["device"] = "cpu"
    training_config = SRNTrainingConfig(**training_data)
    dataset = DRLDataset(d_rl_path)
    split = split_examples_by_game_id(
        dataset.examples,
        validation_fraction=training_config.validation_fraction,
        seed=training_config.seed,
    )
    loader = make_srn_loader(
        split.validation_examples,
        batch_size=training_config.batch_size,
        shuffle=False,
        seed=training_config.seed,
    )
    observed = asdict(evaluate_srn(g0, loader, training_config))
    expected = checkpoint_payload["history"][0]["validation"]
    numeric_deltas = {
        key: abs(float(observed[key]) - float(expected[key]))
        for key in observed
        if isinstance(observed[key], (int, float))
    }
    maximum_delta = max(numeric_deltas.values(), default=0.0)
    return {
        "method": "seeded re-instantiation, then exact epoch-0 validation metric replay",
        "expected_epoch_0_validation": expected,
        "observed_epoch_0_validation": observed,
        "maximum_absolute_metric_delta": maximum_delta,
        "verified": maximum_delta <= 1e-12,
    }


def _conclusion(main_summaries: dict) -> dict:
    records = [
        value
        for key, value in main_summaries.items()
        if key.startswith("G0__vs__G1-best-validation__budget-")
    ]
    if not records:
        return {"category": "B", "text": "experience principale absente"}
    scores = [record["score_rate_a_terminal"] for record in records]
    intervals = [record["paired_bootstrap_ci"] for record in records]
    observed_scores = [score for score in scores if score is not None]
    if observed_scores and len(observed_scores) == len(scores) and all(
        score == 0.5 for score in observed_scores
    ):
        return {
            "category": "A",
            "text": "aucune difference strategique detectee dans ces conditions",
        }
    all_below = all(interval is not None and interval[1] < 0.5 for interval in intervals)
    all_above = all(interval is not None and interval[0] > 0.5 for interval in intervals)
    if all_below or all_above:
        direction = "G1-best" if all_below else "G0"
        return {
            "category": "C",
            "text": f"difference coherente sur tous les budgets testes, favorable a {direction}",
        }
    return {
        "category": "B",
        "text": "difference observee mais incertitude ou incoherence entre budgets trop grande pour conclure",
    }


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    best = load_srn_checkpoint(args.best_checkpoint, device="cpu")
    last = load_srn_checkpoint(args.last_checkpoint, device="cpu")
    srn_config = _assert_same_checkpoint_contract(best, last, args.seed)

    set_training_seed(args.seed)
    g0 = SongoRelationalNetwork(srn_config)
    g0_verification = _verify_g0_reconstruction(g0, best.payload, args.d_rl)
    if not g0_verification["verified"]:
        raise RuntimeError("G0 reconstruction does not reproduce Lot 6 epoch-0 metrics")

    models = {
        "G0": g0,
        "G1-best-validation": best.model,
        "G1-last": last.model,
    }
    initial_fingerprints = {
        name: model_parameter_fingerprint(model) for name, model in models.items()
    }
    print("[lot7] G0 reconstruit et contrat SRN verifie", flush=True)

    lab_positions = select_d_lab_benchmark(args.d_lab, seed=args.seed)
    raw_report = evaluate_raw_network_outputs(models, lab_positions)
    raw_report["source"] = {
        "path": str(args.d_lab),
        "sha256": _sha256(args.d_lab),
        "dataset_family": "D_LAB",
        "used_for_training": False,
    }
    raw_report["seed"] = args.seed
    _write_json(args.output / "raw_network_outputs.json", raw_report)
    print(f"[lot7] analyse brute terminee sur {len(lab_positions)} positions D_LAB", flush=True)

    openings = generate_deterministic_openings(
        prefix_lengths=args.prefix_lengths,
        seed=args.seed,
    )
    _write_json(
        args.output / "openings.json",
        {
            "seed": args.seed,
            "generation": "uniform legal actions from the engine; one independent RNG per prefix",
            "uses_model_outputs": False,
            "openings": [opening_to_dict(opening) for opening in openings],
        },
    )

    arena_config = ArenaConfig(
        max_plies=args.max_plies,
        repetition_limit=args.repetition_limit,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    pair_names = [
        ("G0", "G1-best-validation"),
        ("G0", "G1-last"),
        ("G1-best-validation", "G1-last"),
    ]
    all_games: list[dict] = []
    summaries: dict[str, dict] = {}
    for budget in args.budgets:
        for left_name, right_name in pair_names:
            left = SRNMCTSAgent(left_name, models[left_name], budget, args.c_puct)
            right = SRNMCTSAgent(right_name, models[right_name], budget, args.c_puct)
            results = run_paired_arena(left, right, openings, config=arena_config)
            key = f"{left_name}__vs__{right_name}__budget-{budget}"
            summaries[key] = {**asdict(summarize_arena(results, config=arena_config)), "budget": budget}
            all_games.extend({**game_result_to_dict(result), "budget": budget, "kind": "main"} for result in results)
            print(f"[lot7] {key}: {len(results)} parties terminees", flush=True)

    random_summaries: dict[str, dict] = {}
    if not args.skip_random_baseline:
        random_agent = RandomLegalAgent()
        for model_name in ("G0", "G1-best-validation"):
            srn_agent = SRNMCTSAgent(
                model_name,
                models[model_name],
                args.random_baseline_budget,
                args.c_puct,
            )
            results = run_paired_arena(srn_agent, random_agent, openings, config=arena_config)
            key = f"{model_name}__vs__random-legal__budget-{args.random_baseline_budget}"
            random_summaries[key] = {
                **asdict(summarize_arena(results, config=arena_config)),
                "budget": args.random_baseline_budget,
            }
            all_games.extend(
                {
                    **game_result_to_dict(result),
                    "budget": args.random_baseline_budget,
                    "kind": "random_baseline",
                }
                for result in results
            )
            print(f"[lot7] {key}: {len(results)} parties terminees", flush=True)

    games_path = args.output / "arena_games.jsonl"
    games_path.write_text(
        "".join(json.dumps(record) + "\n" for record in all_games), encoding="utf-8"
    )
    _write_json(
        args.output / "arena_summaries.json",
        {"main": summaries, "random_baseline": random_summaries},
    )

    final_fingerprints = {
        name: model_parameter_fingerprint(model) for name, model in models.items()
    }
    unchanged = initial_fingerprints == final_fingerprints
    if not unchanged:
        raise RuntimeError("arena modified at least one network parameter")
    total_elapsed = time.perf_counter() - started_at
    report = {
        "lot": 7,
        "question": "Does the first SRN training have a measurable strategic effect inside MCTS?",
        "seed": args.seed,
        "device": "cpu",
        "srn_config": asdict(srn_config),
        "checkpoint_contract": {
            "G0": "reconstructed from Lot 6 seed before any optimizer step",
            "G1-best-validation": {
                "path": str(args.best_checkpoint),
                "sha256": _sha256(args.best_checkpoint),
                "epoch": best.payload["epoch"],
            },
            "G1-last": {
                "path": str(args.last_checkpoint),
                "sha256": _sha256(args.last_checkpoint),
                "epoch": last.payload["epoch"],
            },
            "g0_reconstruction": g0_verification,
        },
        "raw_network_evaluation": {
            "positions": len(lab_positions),
            "source": raw_report["source"],
            "pairwise": raw_report["pairwise"],
            "model_summaries": {
                name: {
                    "policy_entropy": data["policy_entropy"],
                    "value": data["value"],
                }
                for name, data in raw_report["models"].items()
            },
        },
        "arena_protocol": {
            "openings": len(openings),
            "prefix_lengths": args.prefix_lengths,
            "opening_generation_independent_of_models": True,
            "paired_games_per_opening": 2,
            "budgets": args.budgets,
            "c_puct": args.c_puct,
            "dirichlet_noise": False,
            "action_selection": "deterministic argmax visit_counts; smallest action index breaks final ties",
            "mcts_seed_meaning": "controls only exact PUCT tie-breaking; derived from experiment seed, opening and continuation ply; identical for both side assignments of a pair",
            "max_plies": args.max_plies,
            "repetition_limit": args.repetition_limit,
            "truncations_are_draws": False,
            "bootstrap": {
                "method": "percentile cluster bootstrap",
                "unit": "opening pair (both side assignments kept together)",
                "samples": args.bootstrap_samples,
                "confidence_level": arena_config.confidence_level,
            },
        },
        "main": summaries,
        "random_baseline": random_summaries,
        "negamax": {"executed": False, "reason": "optional; main paired SRN experiment prioritized"},
        "parameter_integrity": {
            "before": initial_fingerprints,
            "after": final_fingerprints,
            "unchanged": unchanged,
        },
        "totals": {
            "games": len(all_games),
            "main_games": sum(value["games"] for value in summaries.values()),
            "random_baseline_games": sum(value["games"] for value in random_summaries.values()),
            "mcts_simulations": sum(record["total_mcts_simulations"] for record in all_games),
            "network_evaluations": sum(record["total_network_evaluations"] for record in all_games),
            "elapsed_s": total_elapsed,
        },
        "conclusion": _conclusion(summaries),
        "no_checkpoint_promoted": True,
    }
    _write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
