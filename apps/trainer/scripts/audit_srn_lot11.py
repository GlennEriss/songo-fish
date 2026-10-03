#!/usr/bin/env python3
"""Lot 11 : diagnostic MCTS final de G1-best avant toute autorisation G2."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import torch

from songo_ai.dataset import RawSongoState, read_d_rl_jsonl, write_d_rl_jsonl
from songo_ai.evaluation import (
    HybridPolicyValueEvaluator,
    LabBenchmarkPosition,
    ScaledValueEvaluator,
    choose_smallest_mcts_budget,
    final_g2_readiness_gate,
    interseed_stability_report,
    jensen_shannon,
    model_parameter_fingerprint,
    policy_corpus_metrics,
    repeated_state_metrics,
    select_extended_d_lab_benchmark,
    value_distribution_statistics,
)
from songo_ai.generation import SelfPlayConfig, SelfPlayRunner
from songo_ai.model import (
    SRNConfig,
    SongoGraphBuilder,
    SongoRelationalNetwork,
    load_srn_checkpoint,
    set_training_seed,
)
from songo_ai.search import MCTSConfig, SongoMCTS
from songo_ai.songo.rules import SongoLegacyGame


BUDGETS = (8, 16, 32, 64)


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
        "--baseline-d-rl",
        type=Path,
        default=Path("data/d_rl/lot10_g1_8_seed_20260924.jsonl"),
    )
    parser.add_argument(
        "--d-lab", type=Path, default=Path("data/dataset_v001_10k/test.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/lot11_g1_budget_seed_20260924"),
    )
    parser.add_argument(
        "--pilot-d-rl",
        type=Path,
        default=Path("data/d_rl/lot11_g1_selected_seed_20260924.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--pilot-games", type=int, default=50)
    parser.add_argument("--pilot-seed", type=int, default=20261150)
    parser.add_argument("--max-plies", type=int, default=400)
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
    if int(loaded.payload["seed"]) != args.seed:
        raise RuntimeError("G1 checkpoint seed does not match Lot 11 seed")
    config = SRNConfig(**loaded.payload["srn_config"])
    set_training_seed(args.seed)
    g0 = SongoRelationalNetwork(config)
    return config, g0, loaded.model, loaded


def initial_position() -> LabBenchmarkPosition:
    state = RawSongoState((5,) * 14 + (0, 0), 1)
    game = SongoLegacyGame.from_state(state.to_engine_state())
    return LabBenchmarkPosition(
        position_id="S0-initial",
        state=state,
        legal_mask=tuple(game.legal_mask()),
        source_trajectory_id="engine-initial",
        move_number=0,
        phase_proxy="opening",
    )


def build_battery(args) -> tuple[LabBenchmarkPosition, ...]:
    if args.positions < 4:
        raise ValueError("positions must be at least 4")
    candidates = select_extended_d_lab_benchmark(
        args.d_lab, seed=args.seed, target_count=args.positions
    )
    s0 = initial_position()
    return (s0,) + tuple(
        position for position in candidates if position.state != s0.state
    )[: args.positions - 1]


def run_search(model, state, *, budget, seed, noise, c_puct=1.5, trace=False):
    return SongoMCTS(
        model,
        config=MCTSConfig(
            num_simulations=budget,
            c_puct=c_puct,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            add_root_noise=noise,
            collect_simulation_trace=trace,
            seed=seed,
        ),
    ).search(state, policy_temperature=1.0)


def stability_study(
    model, positions, args, *, noise, label, budgets=BUDGETS, c_puct=1.5
):
    started = time.perf_counter()
    per_budget = {}
    raw = {}
    for budget in budgets:
        policies = []
        detail = []
        elapsed = 0.0
        simulations = 0
        for position_index, position in enumerate(positions):
            runs = []
            records = []
            for repetition in range(args.seeds):
                seed = derived_seed(args.seed, label, position_index, repetition)
                result = run_search(
                    model,
                    position.state,
                    budget=budget,
                    seed=seed,
                    noise=noise,
                    c_puct=c_puct,
                )
                runs.append(result.policy)
                elapsed += result.elapsed_s
                simulations += result.num_simulations
                records.append(
                    {
                        "seed": seed,
                        "policy": list(result.policy),
                        "visits": list(result.visit_counts),
                        "root_priors": list(result.root_priors),
                        "root_q": list(result.root_q_values),
                        "root_value": result.root_value,
                        "argmax": result.selected_action,
                    }
                )
            policies.append(runs)
            detail.append({"position_id": position.position_id, "runs": records})
        report = interseed_stability_report(
            policies, [position.legal_count for position in positions]
        )
        report["cost"] = {
            "elapsed_search_s": elapsed,
            "simulations": simulations,
            "simulations_per_second": simulations / elapsed,
            "elapsed_wall_s": time.perf_counter() - started,
        }
        per_budget[budget] = report
        raw[budget] = {"policies": policies, "positions": detail}
        print(
            f"[lot11] {label} budget={budget}: JS={report['pairwise_js']['mean']:.4f}, "
            f"support={report['support']['mean']:.2f}",
            flush=True,
        )
    return per_budget, raw


def per_state_js(raw_budget: dict, positions) -> list[dict]:
    rows = []
    for position, runs in zip(positions, raw_budget["policies"]):
        values = [jensen_shannon(first, second) for first, second in combinations(runs, 2)]
        rows.append(
            {
                "position_id": position.position_id,
                "mean_js": statistics.fmean(values) if values else 0.0,
                "legal_count": position.legal_count,
            }
        )
    return sorted(rows, key=lambda row: row["mean_js"], reverse=True)


def evaluator_ablation(evaluators, positions, args, *, budget, label, c_puct=1.5):
    reports = {}
    for name, evaluator in evaluators.items():
        curve, _ = stability_study(
            evaluator,
            positions,
            args,
            noise=True,
            # Même famille de seeds entre variantes : seule la source de
            # Policy/Value (ou c_puct) change dans une comparaison donnée.
            label=label,
            budgets=(budget,),
            c_puct=c_puct,
        )
        reports[name] = curve[budget]
    return reports


def trace_diagnostics(g0, g1, positions, problematic, args, *, budget):
    by_id = {position.position_id: position for position in positions}
    selected = [positions[0]] + [by_id[row["position_id"]] for row in problematic[:3]]
    traces = []
    leaf_states = []
    g1_values = []
    terminal_count = 0
    for position_index, position in enumerate(selected):
        for repetition in range(min(args.seeds, 4)):
            seed = derived_seed(args.seed, "trace", position.position_id, repetition)
            result = run_search(
                g1,
                position.state,
                budget=budget,
                seed=seed,
                noise=True,
                trace=True,
            )
            record = {
                "position_id": position.position_id,
                "seed": seed,
                "root_policy": list(result.policy),
                "root_visits": list(result.visit_counts),
                "simulations": list(result.simulation_trace),
            }
            traces.append(record)
            for row in result.simulation_trace:
                if row["leaf_terminal"]:
                    terminal_count += 1
                else:
                    state = row["leaf_state"]
                    leaf_states.append(
                        RawSongoState(tuple(state["board"]), int(state["player_to_move"]))
                    )
                    g1_values.append(float(row["leaf_value"]))
    builder = SongoGraphBuilder()
    g0_values = []
    with torch.no_grad():
        for start in range(0, len(leaf_states), 256):
            _, values = g0(builder.build_batch(leaf_states[start : start + 256]))
            g0_values.extend(float(value) for value in values.reshape(-1).tolist())
    early_actions = [
        row["root_action"]
        for trace in traces
        for row in trace["simulations"][:8]
        if row["root_action"] is not None
    ]
    return {
        "selected_positions": [position.position_id for position in selected],
        "searches": len(traces),
        "nonterminal_leaf_evaluations": len(leaf_states),
        "terminal_leaf_evaluations": terminal_count,
        "leaf_value_distributions": {
            "G0_same_leaves": value_distribution_statistics(g0_values),
            "G1_best_same_leaves": value_distribution_statistics(g1_values),
        },
        "first_8_root_action_histogram": dict(sorted(Counter(early_actions).items())),
        "trace_semantics": {
            "leaf_value": "perspective du joueur au trait dans la feuille",
            "root_q": "perspective du joueur au trait à la racine après backup",
        },
        "traces": traces,
    }


def validate_roundtrip(path, original) -> dict:
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


def game_statistics(examples) -> dict:
    games = defaultdict(list)
    for example in examples:
        games[str(example.metadata["game_id"])].append(example)
    statuses = Counter(rows[0].metadata["status"] for rows in games.values())
    winners = Counter(rows[0].metadata["winner"] for rows in games.values())
    lengths = [len(rows) for rows in games.values()]
    return {
        "games": len(games),
        "positions": len(examples),
        "statuses": dict(statuses),
        "wins": {"P1": winners[1], "P2": winners[2], "draws": winners[0]},
        "length_mean": statistics.fmean(lengths),
        "length_median": statistics.median(lengths),
        "terminal_labeled_examples": sum(e.value_target is not None for e in examples),
        "unlabeled_examples": sum(e.value_target is None for e in examples),
    }


def generate_pilot(g1, args, *, budget, checkpoint_id):
    config = SelfPlayConfig(
        games=args.pilot_games,
        max_game_plies=args.max_plies,
        repetition_limit=3,
        target_temperature=1.0,
        action_temperature=1.0,
        temperature_drop_ply=30,
        late_action_temperature=0.0,
        seed=args.pilot_seed,
        generation=11,
        checkpoint_id=checkpoint_id,
        include_truncated_examples=True,
        mcts=MCTSConfig(
            num_simulations=budget,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            add_root_noise=True,
        ),
    )
    before = model_parameter_fingerprint(g1)
    started = time.perf_counter()
    result = SelfPlayRunner(g1, config).generate()
    elapsed = time.perf_counter() - started
    after = model_parameter_fingerprint(g1)
    if before != after:
        raise RuntimeError("pilot self-play modified G1 parameters")
    write_d_rl_jsonl(args.pilot_d_rl, result.examples)
    return result, {
        "path": str(args.pilot_d_rl),
        "selfplay_config": asdict(config),
        "statistics": asdict(result.statistics),
        "games": game_statistics(result.examples),
        "roundtrip": validate_roundtrip(args.pilot_d_rl, result.examples),
        "elapsed_wall_s": elapsed,
        "parameter_fingerprint_before": before,
        "parameter_fingerprint_after": after,
    }


def main() -> None:
    args = parse_args()
    if args.seeds < 8:
        raise ValueError("Lot 11 requires at least 8 seeds")
    if args.positions < 128:
        raise ValueError("Lot 11 requires at least 128 positions")
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    config, g0, g1, loaded = load_models(args)
    fingerprints_before = {
        "G0": model_parameter_fingerprint(g0),
        "G1-best": model_parameter_fingerprint(g1),
    }
    positions = build_battery(args)
    battery = {
        "requested": args.positions,
        "actual": len(positions),
        "seeds": args.seeds,
        "minimum_choice_rationale": (
            "128 positions and 8 seeds are the lower preregistered bounds; this preserves "
            "structural coverage while bounding the CPU cost before G2."
        ),
        "selection_independent_of_model_outputs": True,
        "contains_S0": positions[0].position_id == "S0-initial",
        "positions": [
            {
                "position_id": p.position_id,
                "state": list(p.state.board),
                "player_to_move": p.state.player_to_move,
                "legal_mask": list(p.legal_mask),
                "phase": p.phase_proxy,
            }
            for p in positions
        ],
    }
    write_json(args.output / "battery.json", battery)

    main_curve, main_raw = stability_study(
        g1, positions, args, noise=True, label="P1V1-noise-on"
    )
    problematic = per_state_js(main_raw[8], positions)
    selection = choose_smallest_mcts_budget(main_curve)
    selected_budget = int(selection["selected_budget"])
    write_json(
        args.output / "budget_curve.json",
        {
            "curve": main_curve,
            "selection": selection,
            "per_state_budget8": problematic,
            "S0_detailed_by_budget": {
                str(budget): main_raw[budget]["positions"][0]
                for budget in BUDGETS
            },
        },
    )

    control_positions = positions[:32]
    off_curve, _ = stability_study(
        g1, control_positions, args, noise=False, label="P1V1-noise-off"
    )
    hybrids = {
        "P1V1": HybridPolicyValueEvaluator(g1, g1, name="P1V1"),
        "P1V0": HybridPolicyValueEvaluator(g1, g0, name="P1V0"),
        "P1Vneutral": HybridPolicyValueEvaluator(g1, neutral_value=True, name="P1Vneutral"),
    }
    hybrid_report = evaluator_ablation(
        hybrids, positions, args, budget=selected_budget, label="hybrid"
    )
    alpha_positions = positions[:64]
    alpha_report = evaluator_ablation(
        {
            "alpha_0.0": ScaledValueEvaluator(g1, 0.0, name="alpha-0"),
            "alpha_0.5": ScaledValueEvaluator(g1, 0.5, name="alpha-05"),
            "alpha_1.0": ScaledValueEvaluator(g1, 1.0, name="alpha-1"),
        },
        alpha_positions,
        args,
        budget=selected_budget,
        label="alpha",
    )
    cpuct_report = {}
    for c_puct in (0.75, 1.5, 3.0):
        cpuct_report[str(c_puct)] = evaluator_ablation(
            {"P1V1": HybridPolicyValueEvaluator(g1, g1, name=f"cpuct-{c_puct}")},
            control_positions,
            args,
            budget=selected_budget,
            label="cpuct",
            c_puct=c_puct,
        )["P1V1"]
    ablations = {
        "diagnostic_budget": selected_budget,
        "dirichlet_off_control": off_curve,
        "policy_value_source": hybrid_report,
        "value_scaling": alpha_report,
        "c_puct": cpuct_report,
        "no_training": True,
        "no_checkpoint_created": True,
    }
    write_json(args.output / "ablations.json", ablations)

    trace_report = trace_diagnostics(
        g0, g1, positions, problematic, args, budget=selected_budget
    )
    write_json(args.output / "simulation_traces.json", trace_report)

    checkpoint_id = f"G1-best-{sha256(args.g1_checkpoint)[:12]}"
    pilot_run, pilot_generation = generate_pilot(
        g1, args, budget=selected_budget, checkpoint_id=checkpoint_id
    )
    baseline_examples = read_d_rl_jsonl(args.baseline_d_rl)
    baseline = {
        "policy": policy_corpus_metrics(baseline_examples),
        "repeated": repeated_state_metrics(baseline_examples),
    }
    pilot = {
        "policy": policy_corpus_metrics(pilot_run.examples),
        "repeated": repeated_state_metrics(pilot_run.examples),
    }
    valid = (
        pilot_generation["roundtrip"]["valid"]
        and pilot_generation["games"]["terminal_labeled_examples"] > 0
    )
    gate = final_g2_readiness_gate(baseline, pilot, valid=valid)
    per_game_s = pilot_generation["elapsed_wall_s"] / args.pilot_games
    recommended_games = 500
    complete_corpus = {
        "recommended_games": recommended_games,
        "selected_budget": selected_budget,
        "estimated_elapsed_s_linear": per_game_s * recommended_games,
        "estimate_basis": f"{args.pilot_games}-game independent Lot 11 pilot",
        "estimated_positions": round(
            len(pilot_run.examples) / args.pilot_games * recommended_games
        ),
    }
    fingerprints_after = {
        "G0": model_parameter_fingerprint(g0),
        "G1-best": model_parameter_fingerprint(g1),
    }
    if fingerprints_before != fingerprints_after:
        raise RuntimeError("Lot 11 diagnostics modified model parameters")
    report = {
        "lot": 11,
        "objective": "final MCTS diagnostic before any G2 authorization",
        "provenance": {
            "g1_checkpoint": str(args.g1_checkpoint),
            "g1_checkpoint_sha256": sha256(args.g1_checkpoint),
            "g1_best_epoch": loaded.payload.get("best_epoch", 8),
            "srn_config": asdict(config),
            "baseline_d_rl": str(args.baseline_d_rl),
            "baseline_sha256": sha256(args.baseline_d_rl),
            "d_lab": str(args.d_lab),
            "d_lab_sha256": sha256(args.d_lab),
        },
        "battery": {key: value for key, value in battery.items() if key != "positions"},
        "main_budget_curve": main_curve,
        "budget_selection": selection,
        "ablations": ablations,
        "trace_summary": {
            key: value for key, value in trace_report.items() if key != "traces"
        },
        "pilot_generation": pilot_generation,
        "comparison_to_lot10_g1_8": {"baseline": baseline, "pilot": pilot},
        "g2_readiness_gate": gate,
        "READY_FOR_G2": "YES" if gate["ready_for_g2"] else "NO",
        "next_lot_configuration": (
            {
                "initial_checkpoint": str(args.g1_checkpoint),
                "mcts_simulations": selected_budget,
                "selfplay_games": recommended_games,
                "c_puct": 1.5,
                "dirichlet_alpha": 0.3,
                "dirichlet_epsilon": 0.25,
                "repetition_limit": 3,
                "max_game_plies": args.max_plies,
                "target_temperature": 1.0,
                "action_temperature": 1.0,
                "temperature_drop_ply": 30,
                "late_action_temperature": 0.0,
            }
            if gate["ready_for_g2"]
            else None
        ),
        "blocking_diagnostic": (
            None
            if gate["ready_for_g2"]
            else {
                "failed_checks": [
                    name for name, passed in gate["checks"].items() if not passed
                ],
                "minimal_next_experiment": (
                    "Do not train G2. Test one additional bounded budget (128 simulations) "
                    "on S0 plus the 31 most unstable states, with the same 8 seeds."
                ),
            }
        ),
        "complete_corpus_cost": complete_corpus,
        "parameter_integrity": {
            "before": fingerprints_before,
            "after": fingerprints_after,
            "unchanged": fingerprints_before == fingerprints_after,
        },
        "constraints": {
            "g2_training_started": False,
            "engine_modified": False,
            "srn_architecture_modified": False,
            "canonicalization_added": False,
            "mirror_augmentation_added": False,
        },
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                "selected_budget": selected_budget,
                "READY_FOR_G2": report["READY_FOR_G2"],
                "failed_checks": report["blocking_diagnostic"],
                "elapsed_s": report["elapsed_s"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
