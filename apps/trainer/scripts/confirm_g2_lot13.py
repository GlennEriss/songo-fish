#!/usr/bin/env python3
"""Lot 13 : confirmation indépendante de G2 face à G1, sans entraînement."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from songo_ai.evaluation import (
    ArenaConfig,
    SRNMCTSAgent,
    checkpoint_identity,
    confirmation_verdict,
    game_result_to_dict,
    generate_unique_deterministic_openings,
    model_parameter_fingerprint,
    opening_to_dict,
    run_paired_arena,
    summarize_arena,
)
from songo_ai.model import load_srn_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lot12-report",
        type=Path,
        default=Path("data/experiments/lot12_g2_seed_20261200/report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/lot13_confirm_g2_seed_20261313"),
    )
    parser.add_argument("--seed", type=int, default=20261313)
    parser.add_argument("--openings", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--max-plies", type=int, default=400)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def paired_results(results) -> list[dict]:
    groups = defaultdict(list)
    for result in results:
        groups[result.opening_id].append(result)
    rows = []
    for opening_id, games in sorted(groups.items()):
        games.sort(key=lambda game: game.a_player)
        outcomes = [game.a_outcome for game in games]
        terminal = [outcome for outcome in outcomes if outcome is not None]
        rows.append(
            {
                "opening_id": opening_id,
                "games": len(games),
                "G2_as_P1_score": next(
                    (game.a_outcome for game in games if game.a_player == 1), None
                ),
                "G2_as_P2_score": next(
                    (game.a_outcome for game in games if game.a_player == 2), None
                ),
                "pair_mean_terminal": (
                    statistics.fmean(terminal) if terminal else None
                ),
                "statuses": [game.status.value for game in games],
            }
        )
    return rows


def length_by_outcome(results) -> dict:
    groups = defaultdict(list)
    for result in results:
        outcome = result.a_outcome
        label = (
            "G2_win" if outcome == 1.0 else
            "draw" if outcome == 0.5 else
            "G1_win" if outcome == 0.0 else
            "truncated"
        )
        groups[label].append(result.continuation_plies)
    return {
        label: {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for label, values in sorted(groups.items())
    }


def main() -> None:
    args = parse_args()
    if args.openings < 128:
        raise ValueError("Lot 13 requires at least 128 openings")
    if args.bootstrap_samples < 10_000:
        raise ValueError("Lot 13 requires at least 10,000 bootstrap samples")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    lot12 = json.loads(args.lot12_report.read_text(encoding="utf-8"))
    g1_path = Path(lot12["checkpoints"]["lineage"]["parent_checkpoint"])
    g2_path = Path(lot12["checkpoints"]["G2_best_validation"])
    identities = {
        "G1-best": checkpoint_identity(
            g1_path,
            expected_sha256=lot12["checkpoints"]["lineage"]["parent_checkpoint_sha256"],
        ),
        "G2-best-validation": checkpoint_identity(
            g2_path,
            expected_sha256=lot12["checkpoints"]["best_sha256"],
        ),
    }
    if not all(identity["matches"] for identity in identities.values()):
        raise RuntimeError("checkpoint identity mismatch")
    g1_loaded = load_srn_checkpoint(g1_path)
    g2_loaded = load_srn_checkpoint(g2_path)
    identities["G1-best"]["epoch"] = g1_loaded.payload["epoch"]
    identities["G2-best-validation"]["epoch"] = g2_loaded.payload["epoch"]
    if identities["G1-best"]["epoch"] != 8 or identities["G2-best-validation"]["epoch"] != 4:
        raise RuntimeError("checkpoint epoch mismatch")
    g1, g2 = g1_loaded.model, g2_loaded.model
    fingerprints_before = {
        "G1-best": model_parameter_fingerprint(g1),
        "G2-best-validation": model_parameter_fingerprint(g2),
    }

    # La longueur demandée dépend seulement de (seed, index), jamais des modèles.
    openings = generate_unique_deterministic_openings(
        count=args.openings,
        seed=args.seed,
        max_prefix_length=40,
    )
    opening_payload = {
        "seed": args.seed,
        "selection": "uniform random legal prefixes; independent of model outputs",
        "openings": [opening_to_dict(opening) for opening in openings],
    }
    write_json(args.output / "openings.json", opening_payload)

    arena_config = ArenaConfig(
        max_plies=args.max_plies,
        repetition_limit=3,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=0.95,
    )
    summaries = {}
    paired = {}
    all_games = []
    for budget in (32, 64):
        results = run_paired_arena(
            SRNMCTSAgent("G2-best-validation", g2, budget, c_puct=1.5),
            SRNMCTSAgent("G1-best", g1, budget, c_puct=1.5),
            openings,
            config=arena_config,
        )
        summary = asdict(summarize_arena(results, config=arena_config))
        summary["length_by_outcome"] = length_by_outcome(results)
        summaries[str(budget)] = summary
        paired[str(budget)] = paired_results(results)
        all_games.extend(
            {**game_result_to_dict(result), "budget": budget} for result in results
        )
        print(
            f"[lot13] budget={budget}: {len(results)} parties, "
            f"score G2={summary['score_rate_a_terminal']:.4f}",
            flush=True,
        )

    fingerprints_after = {
        "G1-best": model_parameter_fingerprint(g1),
        "G2-best-validation": model_parameter_fingerprint(g2),
    }
    anomalies = []
    for budget in (32, 64):
        summary = summaries[str(budget)]
        if summary["games"] != 2 * args.openings:
            anomalies.append(f"budget {budget}: incomplete pairing")
        truncated = (
            summary["aggregate"]["truncated_repetition"]
            + summary["aggregate"]["truncated_max_plies"]
        )
        if truncated / summary["games"] > 0.10:
            anomalies.append(f"budget {budget}: truncation rate above 10%")
        if any(row["games"] != 2 for row in paired[str(budget)]):
            anomalies.append(f"budget {budget}: opening without exactly two games")
    if fingerprints_before != fingerprints_after:
        anomalies.append("model parameters changed during arena")

    verdict = confirmation_verdict(summaries["64"], anomalies=anomalies)
    configuration = {
        "experimental_seed": args.seed,
        "openings": args.openings,
        "games_per_budget": 2 * args.openings,
        "budgets": [32, 64],
        "principal_budget": 64,
        "c_puct": 1.5,
        "dirichlet": False,
        "temperature": 0.0,
        "action_selection": "argmax visit_counts",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "opening_pair",
        "confidence_level": 0.95,
        "max_plies": args.max_plies,
        "repetition_limit": 3,
    }
    write_json(args.output / "configuration.json", configuration)
    write_json(args.output / "paired_results.json", paired)
    write_json(args.output / "summaries.json", summaries)
    (args.output / "games.jsonl").write_text(
        "".join(json.dumps(game) + "\n" for game in all_games),
        encoding="utf-8",
    )
    report = {
        "lot": 13,
        "question": "La progression G2 > G1 est-elle reproductible sur 128 nouvelles ouvertures appariées ?",
        "checkpoint_identities": identities,
        "configuration": configuration,
        "summaries": summaries,
        "paired_results_path": str(args.output / "paired_results.json"),
        "comparison_lot12": {
            "lot12_seed": lot12["generation"]["selfplay_config"]["seed"],
            "lot12_arena": lot12["arena"]["summaries"],
            "lot13_seed_is_distinct": args.seed != lot12["generation"]["selfplay_config"]["seed"],
        },
        "verdict": verdict,
        "parameter_integrity": {
            "before": fingerprints_before,
            "after": fingerprints_after,
            "unchanged": fingerprints_before == fingerprints_after,
        },
        "training_or_generation": {
            "d_rl_created": False,
            "G2_retrained": False,
            "G3_created": False,
        },
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                "MCTS64": summaries["64"],
                "verdict": verdict,
                "elapsed_s": report["elapsed_s"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
