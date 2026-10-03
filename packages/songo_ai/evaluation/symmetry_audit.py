"""Audit reproductible de symetrie P1/P2 et d'information du pilote D_RL."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Mapping, Sequence

import torch

from songo_ai.dataset.selfplay_schema import RawSongoState, RLTrainingExample
from songo_ai.model.srn_graph import SongoGraphBuilder
from songo_ai.model.srn_network import policy_probabilities
from songo_ai.songo.rules import DRAW, PLAYER_ONE, PLAYER_TWO, SongoLegacyGame

from .srn_benchmark import LabBenchmarkPosition


def mirror_raw_state(state: RawSongoState) -> RawSongoState:
    """Echange les identites physiques P1/P2 sans changer le repere local."""

    board = state.board
    player = PLAYER_TWO if state.player_to_move == PLAYER_ONE else PLAYER_ONE
    return RawSongoState(
        board[7:14] + board[0:7] + (board[15], board[14]),
        player,
    )


def validate_mirrored_legality(state: RawSongoState) -> tuple[bool, ...]:
    """Valide le statut terminal et la legalite sous renommage P1/P2."""

    original = SongoLegacyGame.from_state(state.to_engine_state())
    mirrored = SongoLegacyGame.from_state(mirror_raw_state(state).to_engine_state())
    original.normalize_terminal()
    mirrored.normalize_terminal()
    if original.finished != mirrored.finished:
        raise ValueError("mirroring changed terminal status")
    expected_winner = original.winner
    if expected_winner not in (None, DRAW):
        expected_winner = PLAYER_TWO if expected_winner == PLAYER_ONE else PLAYER_ONE
    if mirrored.winner != expected_winner:
        raise ValueError("mirroring did not exchange the physical winner")
    original_mask = tuple(original.legal_mask()) if not original.finished else (False,) * 7
    mirrored_mask = tuple(mirrored.legal_mask()) if not mirrored.finished else (False,) * 7
    if original_mask != mirrored_mask:
        raise ValueError("mirroring changed local-action legality")
    return original_mask


def _jensen_shannon(first: Sequence[float], second: Sequence[float]) -> float:
    middle = [(left + right) / 2.0 for left, right in zip(first, second)]

    def kl(values):
        return sum(
            value * math.log(value / mid)
            for value, mid in zip(values, middle)
            if value > 0.0
        )

    return 0.5 * kl(first) + 0.5 * kl(second)


def evaluate_model_symmetry(
    models: Mapping[str, torch.nn.Module],
    positions: Sequence[LabBenchmarkPosition],
    *,
    graph_builder: SongoGraphBuilder | None = None,
) -> dict:
    if not models or not positions:
        raise ValueError("models and positions must not be empty")
    graph_builder = graph_builder or SongoGraphBuilder()
    states = [position.state for position in positions]
    mirrored_states = [mirror_raw_state(state) for state in states]
    masks = torch.tensor([position.legal_mask for position in positions], dtype=torch.bool)
    results = {}
    for name, model in models.items():
        device = next(model.parameters(), torch.empty(0)).device
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits, values = model(graph_builder.build_batch(states).to(device))
                mirrored_logits, mirrored_values = model(
                    graph_builder.build_batch(mirrored_states).to(device)
                )
                policies = policy_probabilities(logits, masks.to(device)).cpu().tolist()
                mirrored_policies = policy_probabilities(
                    mirrored_logits, masks.to(device)
                ).cpu().tolist()
        finally:
            model.train(was_training)
        values = [float(value) for value in values.cpu().tolist()]
        mirrored_values = [float(value) for value in mirrored_values.cpu().tolist()]
        value_differences = [
            abs(left - right) for left, right in zip(values, mirrored_values)
        ]
        policy_js = [
            _jensen_shannon(left, right)
            for left, right in zip(policies, mirrored_policies)
        ]
        agreements = [
            max(range(7), key=left.__getitem__)
            == max(range(7), key=right.__getitem__)
            for left, right in zip(policies, mirrored_policies)
        ]
        by_player = {}
        for player in (PLAYER_ONE, PLAYER_TWO):
            indices = [
                index
                for index, position in enumerate(positions)
                if position.state.player_to_move == player
            ]
            if not indices:
                by_player[f"P{player}"] = {"positions": 0}
                continue
            by_player[f"P{player}"] = {
                "positions": len(indices),
                "value_absolute_difference_mean": statistics.fmean(
                    value_differences[index] for index in indices
                ),
                "value_original_mean": statistics.fmean(values[index] for index in indices),
                "value_mirrored_mean": statistics.fmean(
                    mirrored_values[index] for index in indices
                ),
                "policy_jensen_shannon_mean": statistics.fmean(
                    policy_js[index] for index in indices
                ),
                "policy_argmax_agreement": statistics.fmean(
                    agreements[index] for index in indices
                ),
            }
        worst_indices = sorted(
            range(len(positions)), key=value_differences.__getitem__, reverse=True
        )[:10]
        results[name] = {
            "positions": len(positions),
            "value_absolute_difference_mean": statistics.fmean(value_differences),
            "value_absolute_difference_median": statistics.median(value_differences),
            "value_absolute_difference_max": max(value_differences),
            "value_original_mean": statistics.fmean(values),
            "value_mirrored_mean": statistics.fmean(mirrored_values),
            "policy_jensen_shannon_mean": statistics.fmean(policy_js),
            "policy_jensen_shannon_max": max(policy_js),
            "policy_argmax_agreement": statistics.fmean(agreements),
            "by_original_physical_player": by_player,
            "worst_value_symmetry_positions": [
                {
                    "position_id": positions[index].position_id,
                    "original_player": positions[index].state.player_to_move,
                    "value_original": values[index],
                    "value_mirrored": mirrored_values[index],
                    "absolute_difference": value_differences[index],
                    "policy_jensen_shannon": policy_js[index],
                    "policy_argmax_original": max(
                        range(7), key=policies[index].__getitem__
                    ),
                    "policy_argmax_mirrored": max(
                        range(7), key=mirrored_policies[index].__getitem__
                    ),
                }
                for index in worst_indices
            ],
        }
    return results


def audit_d_rl_information(examples: Sequence[RLTrainingExample]) -> dict:
    if not examples:
        raise ValueError("examples must not be empty")
    if any(example.policy_target is None for example in examples):
        raise ValueError("all examples must contain policy_target")
    if any(example.visit_counts is None for example in examples):
        raise ValueError("all examples must contain visit_counts")
    policy_targets = [example.policy_target for example in examples]
    visit_counts = [example.visit_counts for example in examples]
    support_sizes = [
        sum(probability > 0.0 for probability in target)
        for target in policy_targets
    ]
    entropies = [
        -sum(
            probability * math.log(probability)
            for probability in target
            if probability > 0.0
        )
        for target in policy_targets
    ]
    visit_totals = [sum(counts) for counts in visit_counts]
    players = Counter(example.state.player_to_move for example in examples)
    targets_by_player = defaultdict(list)
    for example in examples:
        if example.value_target is not None:
            targets_by_player[example.state.player_to_move].append(example.value_target)

    by_state: dict[tuple, list[tuple[float, ...]]] = defaultdict(list)
    for example in examples:
        key = (example.state.board, example.state.player_to_move)
        by_state[key].append(example.policy_target)
    repeated = {key: values for key, values in by_state.items() if len(values) > 1}
    conflicting = {
        key: values for key, values in repeated.items() if len(set(values)) > 1
    }
    initial_key = (tuple([5] * 14 + [0, 0]), PLAYER_ONE)
    initial_targets = by_state.get(initial_key, [])

    return {
        "examples": len(examples),
        "players": {f"P{player}": count for player, count in sorted(players.items())},
        "value_target_by_player": {
            f"P{player}": {
                "count": len(values),
                "mean": statistics.fmean(values),
                "win_fraction": sum(value == 1.0 for value in values) / len(values),
                "loss_fraction": sum(value == -1.0 for value in values) / len(values),
            }
            for player, values in sorted(targets_by_player.items())
        },
        "mcts_visit_total_histogram": dict(sorted(Counter(visit_totals).items())),
        "policy_support_histogram": dict(sorted(Counter(support_sizes).items())),
        "policy_one_hot_fraction": sum(size == 1 for size in support_sizes) / len(support_sizes),
        "policy_entropy_mean": statistics.fmean(entropies),
        "policy_entropy_median": statistics.median(entropies),
        "unique_states": len(by_state),
        "repeated_states": len(repeated),
        "repeated_states_with_conflicting_policy_targets": len(conflicting),
        "initial_state_occurrences": len(initial_targets),
        "initial_state_unique_policy_targets": len(set(initial_targets)),
    }
