#!/usr/bin/env python3
"""Lot 9B : symétrie réelle du moteur et qualité des cibles MCTS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from diagnose_lot1a_engine import (
    _generated_canonical_cases,
    _manual_canonical_cases,
    canonicalization_diagnostics,
)
from songo_ai.dataset import RawSongoState, read_d_rl_jsonl
from songo_ai.evaluation import (
    PLAYER_SWAP_CANDIDATES,
    SWAP_KEEP_LOCAL,
    audit_state_collection,
    audit_transition_equivariance,
    jensen_shannon,
    model_parameter_fingerprint,
    select_extended_d_lab_benchmark,
    summarize_policy_repetitions,
)
from songo_ai.generation import select_action_from_policy
from songo_ai.model import SRNConfig, SongoGraphBuilder, SongoRelationalNetwork
from songo_ai.model.srn_network import policy_probabilities
from songo_ai.search import MCTSConfig, SongoMCTS
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO, SongoLegacyGame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d-lab-dir", type=Path, default=Path("data/dataset_v001_10k")
    )
    parser.add_argument(
        "--d-rl", type=Path, default=Path("data/d_rl/pilot_lot5_seed_20260924.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/lot9b_symmetry_engine_seed_20260924"),
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--random-games", type=int, default=100)
    parser.add_argument("--random-max-plies", type=int, default=300)
    parser.add_argument("--mcts-positions", type=int, default=16)
    parser.add_argument("--mcts-repetitions", type=int, default=8)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 8, 32, 64])
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_seed(base: int, *parts: object) -> int:
    payload = ":".join(str(part) for part in (base,) + parts)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def load_d_lab_states(directory: Path):
    states = []
    paths = []
    for split in ("train", "val", "test"):
        path = directory / f"{split}.jsonl"
        paths.append(path)
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                states.append(
                    (
                        str(record["trajectory_id"]),
                        RawSongoState(tuple(record["state"]), PLAYER_ONE),
                    )
                )
    return states, paths


def generate_random_diagnostic_states(seed: int, games: int, max_plies: int):
    rng = random.Random(seed)
    states = []
    trajectory_lengths = []
    for game_index in range(games):
        game = SongoLegacyGame()
        length = 0
        for _ in range(max_plies):
            game.normalize_terminal()
            if game.finished:
                break
            states.append((f"random-{game_index}", RawSongoState.from_game(game)))
            legal = game.legal_local_actions()
            game.play_local(rng.choice(legal))
            length += 1
        trajectory_lengths.append(length)
    return states, trajectory_lengths


def audit_lot1a_cases() -> dict:
    cases = _manual_canonical_cases() + _generated_canonical_cases()
    report = {}
    for transform in PLAYER_SWAP_CANDIDATES:
        records = []
        for index, (board, turn, action, tags) in enumerate(cases):
            record = audit_transition_equivariance(
                RawSongoState(tuple(board), turn), action, transform
            )
            record["index"] = index
            record["tags"] = tags
            records.append(record)
        failures = [record for record in records if not record["success"]]
        report[transform.name] = {
            "transitions_tested": len(records),
            "successes": len(records) - len(failures),
            "failures": len(failures),
            "failure_rate": len(failures) / len(records),
            "seed_count_histogram": dict(
                sorted(
                    Counter(
                        "unavailable"
                        if record.get("seed_count") is None
                        else str(record["seed_count"])
                        for record in failures
                    ).items()
                )
            ),
            "all_failures_final_store_deposit": all(
                record.get("original_final_store_deposit")
                or record.get("transformed_final_store_deposit")
                for record in failures
            ),
            "all_failures_capture_difference": all(
                record.get("original_capture") != record.get("transformed_capture")
                for record in failures
            ),
            "failure_kind_histogram": dict(
                sorted(Counter(kind for record in failures for kind in record["failures"]).items())
            ),
        }
    return report


def summarize_values(values):
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def audit_d_rl_bias(examples) -> dict:
    games = defaultdict(list)
    for example in examples:
        games[str(example.metadata["game_id"])].append(example)
    game_rows = []
    for game_id, rows in sorted(games.items()):
        rows.sort(key=lambda example: int(example.metadata["ply"]))
        winner = int(rows[0].metadata["winner"])
        game_rows.append(
            {
                "game_id": game_id,
                "winner": winner,
                "length": len(rows),
                "positions_p1": sum(row.state.player_to_move == PLAYER_ONE for row in rows),
                "positions_p2": sum(row.state.player_to_move == PLAYER_TWO for row in rows),
            }
        )

    by_player = defaultdict(list)
    by_parity = defaultdict(list)
    for example in examples:
        by_player[example.state.player_to_move].append(float(example.value_target))
        by_parity[int(example.metadata["ply"]) % 2].append(float(example.value_target))
    wins = Counter(row["winner"] for row in game_rows)
    lengths_by_winner = {
        f"P{player}": summarize_values(
            [row["length"] for row in game_rows if row["winner"] == player]
        )
        for player in (PLAYER_ONE, PLAYER_TWO)
    }
    game_balanced_p1 = (wins[PLAYER_ONE] - wins[PLAYER_TWO]) / len(game_rows)
    game_balanced_p2 = -game_balanced_p1
    observed_p1 = statistics.fmean(by_player[PLAYER_ONE])
    observed_p2 = statistics.fmean(by_player[PLAYER_TWO])
    return {
        "games": len(game_rows),
        "wins": {"P1": wins[PLAYER_ONE], "P2": wins[PLAYER_TWO]},
        "length_by_winner": lengths_by_winner,
        "positions": {
            "P1": len(by_player[PLAYER_ONE]),
            "P2": len(by_player[PLAYER_TWO]),
        },
        "z_by_player_to_move": {
            "P1": summarize_values(by_player[PLAYER_ONE]),
            "P2": summarize_values(by_player[PLAYER_TWO]),
        },
        "z_by_ply_parity": {
            "even": summarize_values(by_parity[0]),
            "odd": summarize_values(by_parity[1]),
        },
        "game_balanced_z_expected_from_12_8_only": {
            "P1": game_balanced_p1,
            "P2": game_balanced_p2,
        },
        "additional_position_weighting_effect": {
            "P1": observed_p1 - game_balanced_p1,
            "P2": observed_p2 - game_balanced_p2,
        },
        "per_game": game_rows,
    }


def make_pilot_g0(seed: int):
    torch.manual_seed(seed)
    return SongoRelationalNetwork(
        SRNConfig(hidden_dim=16, num_relational_blocks=2)
    )


def raw_policy(model, state):
    graph = SongoGraphBuilder().build(state).as_batch()
    game = SongoLegacyGame.from_state(state.to_engine_state())
    mask = torch.tensor([game.legal_mask()], dtype=torch.bool)
    with torch.no_grad():
        logits, _ = model(graph)
    return tuple(float(value) for value in policy_probabilities(logits, mask)[0].tolist())


def search_policy(model, state, budget, noise, seed):
    result = SongoMCTS(
        model,
        config=MCTSConfig(
            num_simulations=budget,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            add_root_noise=noise,
            seed=seed,
        ),
    ).search(state, policy_temperature=1.0)
    return result


def selfplay_search_seed(base_seed: int, game_index: int, ply: int) -> int:
    payload = f"{base_seed}:{game_index}:{ply}:mcts"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def selfplay_action_seed(base_seed: int, game_index: int) -> int:
    payload = f"{base_seed}:{game_index}:0:actions"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def audit_initial_position(examples, model, seed: int) -> dict:
    initial = [
        example
        for example in examples
        if int(example.metadata["ply"]) == 0
        and example.state.board == tuple([5] * 14 + [0, 0])
    ]
    initial.sort(key=lambda example: int(str(example.metadata["game_id"]).split("-")[2]))
    state = initial[0].state
    base_priors = raw_policy(model, state)
    records = []
    no_noise_targets = []
    for example in initial:
        game_index = int(str(example.metadata["game_id"]).split("-")[2])
        search_seed = selfplay_search_seed(seed, game_index, 0)
        result = search_policy(model, state, 2, True, search_seed)
        no_noise = search_policy(model, state, 2, False, search_seed)
        no_noise_targets.append(no_noise.policy)
        action_rng = random.Random(selfplay_action_seed(seed, game_index))
        replayed_action = select_action_from_policy(
            example.metadata["play_policy"], example.legal_mask, action_rng
        )
        noise = tuple(
            (prior - 0.75 * base) / 0.25
            for prior, base in zip(result.root_priors, base_priors)
        )
        records.append(
            {
                "game_index": game_index,
                "game_id": example.metadata["game_id"],
                "search_seed_reconstructed": search_seed,
                "visit_counts_stored": list(example.visit_counts),
                "visit_counts_recomputed": list(result.visit_counts),
                "policy_target": list(example.policy_target),
                "root_priors_before_noise": list(base_priors),
                "root_priors_after_noise_recomputed": list(result.root_priors),
                "dirichlet_vector_reconstructed": list(noise),
                "action_stored": example.metadata["action_played"],
                "action_replayed": replayed_action,
                "no_noise_visit_counts": list(no_noise.visit_counts),
                "winner": example.metadata["winner"],
            }
        )
    fixed_seed_runs = [
        search_policy(model, state, 2, True, records[0]["search_seed_reconstructed"]).policy
        for _ in range(5)
    ]
    return {
        "occurrences": len(initial),
        "unique_stored_targets": len({tuple(example.policy_target) for example in initial}),
        "unique_stored_visit_counts": len({tuple(example.visit_counts) for example in initial}),
        "raw_network_priors": list(base_priors),
        "raw_network_prior_vectors": 1,
        "recomputed_visit_counts_exact": sum(
            record["visit_counts_stored"] == record["visit_counts_recomputed"]
            for record in records
        ),
        "replayed_actions_exact": sum(
            record["action_stored"] == record["action_replayed"] for record in records
        ),
        "unique_targets_same_seeds_without_dirichlet": len(set(no_noise_targets)),
        "fixed_seed_repetitions_identical": len(set(fixed_seed_runs)) == 1,
        "interpretation": (
            "The network prior is identical. Variability is introduced by root Dirichlet "
            "noise and amplified by the two-visit discretization; residual no-noise "
            "variability can only come from seeded exact PUCT tie-breaks."
        ),
        "records": records,
    }


def mcts_stability_battery(model, positions, budgets, repetitions, seed):
    report = {}
    for budget in budgets:
        modes = {}
        raw = {}
        for noise in (False, True):
            per_state_policies = []
            per_state_visits = []
            for position in positions:
                policies = []
                visits = []
                for repetition in range(repetitions):
                    run_seed = derived_seed(
                        seed, "lot9b", position.position_id, budget, noise, repetition
                    )
                    result = search_policy(
                        model, position.state, budget, noise, run_seed
                    )
                    policies.append(result.policy)
                    visits.append(result.visit_counts)
                per_state_policies.append(policies)
                per_state_visits.append(visits)
            key = "dirichlet_on" if noise else "dirichlet_off"
            non_forced = [
                policies
                for position, policies in zip(positions, per_state_policies)
                if position.legal_count > 1
            ]
            modes[key] = {
                "all_states": summarize_policy_repetitions(per_state_policies),
                "non_forced_states": summarize_policy_repetitions(non_forced),
            }
            raw[key] = per_state_policies
        noise_vs_off_js = []
        noise_vs_off_argmax = []
        for state_index in range(len(positions)):
            for repetition in range(repetitions):
                off = raw["dirichlet_off"][state_index][repetition]
                on = raw["dirichlet_on"][state_index][repetition]
                noise_vs_off_js.append(jensen_shannon(off, on))
                noise_vs_off_argmax.append(
                    max(range(7), key=off.__getitem__)
                    == max(range(7), key=on.__getitem__)
                )
        modes["dirichlet_effect"] = {
            "paired_policy_jensen_shannon_mean": statistics.fmean(noise_vs_off_js),
            "paired_argmax_agreement": statistics.fmean(noise_vs_off_argmax),
        }
        report[str(budget)] = modes
    return report


def played_trajectory_equivariance(examples) -> dict:
    by_game = defaultdict(list)
    for example in examples:
        by_game[str(example.metadata["game_id"])].append(example)
    failed_games = set()
    failed_transitions = 0
    tested = 0
    for game_id, rows in by_game.items():
        for example in rows:
            tested += 1
            record = audit_transition_equivariance(
                example.state,
                int(example.metadata["action_played"]),
                SWAP_KEEP_LOCAL,
            )
            if not record["success"]:
                failed_transitions += 1
                failed_games.add(game_id)
    return {
        "transitions_tested": tested,
        "transition_failures": failed_transitions,
        "transition_failure_rate": failed_transitions / tested,
        "trajectories_tested": len(by_game),
        "trajectories_with_failure": len(failed_games),
        "trajectory_failure_rate": len(failed_games) / len(by_game),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    lot1a = audit_lot1a_cases()
    historical = canonicalization_diagnostics()["SongoLegacyGame"]
    print("[lot9b] Lot 1A reproduit", flush=True)

    d_lab_states, d_lab_paths = load_d_lab_states(args.d_lab_dir)
    d_rl_examples = read_d_rl_jsonl(args.d_rl)
    d_rl_states = [
        (str(example.metadata["game_id"]), example.state) for example in d_rl_examples
    ]
    random_states, random_lengths = generate_random_diagnostic_states(
        args.seed + 9, args.random_games, args.random_max_plies
    )
    source_states = {
        "D_LAB": d_lab_states,
        "D_RL_Lot5": d_rl_states,
        "random_diagnostic": random_states,
    }
    engine_frequency = {
        transform.name: {
            source: audit_state_collection(states, transform)
            for source, states in source_states.items()
        }
        for transform in PLAYER_SWAP_CANDIDATES
    }
    print("[lot9b] fréquences moteur mesurées", flush=True)

    model = make_pilot_g0(args.seed)
    fingerprint_before = model_parameter_fingerprint(model)
    initial_position = audit_initial_position(d_rl_examples, model, args.seed)
    positions = select_extended_d_lab_benchmark(
        args.d_lab_dir / "test.jsonl",
        seed=args.seed + 9,
        target_count=args.mcts_positions,
    )
    stability = mcts_stability_battery(
        model, positions, args.budgets, args.mcts_repetitions, args.seed
    )
    fingerprint_after = model_parameter_fingerprint(model)
    if fingerprint_before != fingerprint_after:
        raise RuntimeError("Lot 9B modified pilot G0 parameters")
    print("[lot9b] stabilité MCTS mesurée", flush=True)

    d_rl_bias = audit_d_rl_bias(d_rl_examples)
    played_equivariance = played_trajectory_equivariance(d_rl_examples)
    report = {
        "lot": "9B",
        "seed": args.seed,
        "no_training": True,
        "no_engine_change": True,
        "no_srn_change": True,
        "transformations": {
            "swap_keep_local": {
                "pits": "new[0:7]=old[7:14], new[7:14]=old[0:7]",
                "stores": "new[14]=old[15], new[15]=old[14]",
                "player": "P1<->P2",
                "action": "a->a",
                "winner": "P1<->P2, DRAW unchanged",
            },
            "swap_reverse_local": {
                "pits": "new[0:7]=reverse(old[7:14]), new[7:14]=reverse(old[0:7])",
                "stores": "new[14]=old[15], new[15]=old[14]",
                "player": "P1<->P2",
                "action": "a->6-a",
                "winner": "P1<->P2, DRAW unchanged",
            },
        },
        "lot1a_reproduction": {
            "new_transform_audit": lot1a,
            "historical_audit": {
                "tested": historical["tested"],
                "successes": historical["successes"],
                "failures": historical["failures"],
                "failure_kinds": historical["failure_kinds"],
            },
        },
        "engine_frequency": engine_frequency,
        "d_rl_played_trajectory_equivariance": played_equivariance,
        "terminal_value_derivation": {
            "definition": "z(S)=+1 iff winner(S)=player_to_move(S), -1 otherwise, 0 draw",
            "under_valid_swap": (
                "winner and player_to_move are both swapped; their equality is preserved; z_T=z"
            ),
        },
        "policy_mapping": {
            "keep_local": "M_T[a]=M[a], pi_T[a]=pi[a]",
            "reverse_local": "M_T[6-a]=M[a], pi_T[6-a]=pi[a]",
            "warning": (
                "This structural mapping does not assert equality of noisy two-visit MCTS targets."
            ),
        },
        "d_rl_bias": d_rl_bias,
        "initial_position": initial_position,
        "mcts_stability": {
            "model": "exact random SRN used by Lot 5 (hidden=16, blocks=2, seed=20260924)",
            "positions": len(positions),
            "repetitions_per_state": args.mcts_repetitions,
            "budgets": args.budgets,
            "target_temperature": 1.0,
            "metrics": stability,
        },
        "decision": {
            "category": "B",
            "canonicalization": (
                "Not valid globally for the current engine. Conditional transition-level use is "
                "possible only with an explicit exact equivariance certificate."
            ),
            "blanket_mirror_augmentation": False,
            "eligibility_predicates": {
                "transition": "T(F(S,a)) == F(T(S),T_action(a)) including terminal/winner",
                "observed_trajectory": "every played transition passes and final winner is swapped",
                "policy_label": (
                    "every transition visited by the producing MCTS tree must pass; the current "
                    "D_RL format does not store enough tree edges to certify this"
                ),
            },
            "symmetry_regularization": (
                "Not mathematically justified globally before the engine asymmetry is resolved; "
                "it could only be masked to certified eligible samples."
            ),
        },
        "next_d_rl_recommendation": {
            "do_not_train_G2_yet": True,
            "actions": [
                "arbitrate or preserve explicitly the full-turn final-store behavior",
                "record root priors, search seed and root visit information in future diagnostics",
                "use an experimentally selected budget above two; retain 8/32/64 comparison",
                "balance physical winners and game lengths, not only position counts",
                "run the symmetry audit as a generation gate without augmenting uncertified labels",
            ],
        },
        "sources": {
            "D_LAB": [
                {"path": str(path), "sha256": sha256(path)} for path in d_lab_paths
            ],
            "D_RL": {"path": str(args.d_rl), "sha256": sha256(args.d_rl)},
            "random_diagnostic": {
                "games": args.random_games,
                "states": len(random_states),
                "length_mean": statistics.fmean(random_lengths),
                "length_min": min(random_lengths),
                "length_max": max(random_lengths),
            },
        },
        "parameter_integrity": {
            "before": fingerprint_before,
            "after": fingerprint_after,
            "unchanged": True,
        },
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "lot": report["lot"],
                "report": str(args.output / "report.json"),
                "lot1a_keep_local": lot1a["swap_keep_local"],
                "played_d_rl": played_equivariance,
                "decision": report["decision"],
                "elapsed_s": report["elapsed_s"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
