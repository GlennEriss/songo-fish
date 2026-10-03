"""Tests du diagnostic de symetrie et d'information D_RL du Lot 9A."""

from __future__ import annotations

import torch
from torch import nn

from songo_ai.dataset.selfplay_schema import RLTrainingExample, RawSongoState
from songo_ai.evaluation import (
    LabBenchmarkPosition,
    audit_d_rl_information,
    evaluate_model_symmetry,
    mirror_raw_state,
    validate_mirrored_legality,
)
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO, SongoLegacyGame


class _ConstantPolicyValue(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, graph):
        logits = torch.zeros(
            (graph.batch_size, 7), device=graph.node_features.device
        ) + self.anchor
        values = torch.zeros(graph.batch_size, device=graph.node_features.device)
        return logits, values


def _position() -> LabBenchmarkPosition:
    game = SongoLegacyGame()
    for action in (2, 4, 1):
        game.play_local(action)
    state = RawSongoState.from_game(game)
    return LabBenchmarkPosition(
        position_id="test-position",
        state=state,
        legal_mask=tuple(game.legal_mask()),
        source_trajectory_id="test",
        move_number=3,
        phase_proxy="opening",
    )


def test_mirror_is_an_involution_and_exchanges_player_and_stores() -> None:
    state = _position().state
    mirrored = mirror_raw_state(state)

    assert mirror_raw_state(mirrored) == state
    assert mirrored.player_to_move == (
        PLAYER_TWO if state.player_to_move == PLAYER_ONE else PLAYER_ONE
    )
    assert mirrored.board[:7] == state.board[7:14]
    assert mirrored.board[7:14] == state.board[:7]
    assert mirrored.board[14:] == (state.board[15], state.board[14])


def test_mirror_preserves_local_legality_and_successor_dynamics() -> None:
    game = SongoLegacyGame()
    for action in (0, 3, 1, 5, 2, 4):
        state = RawSongoState.from_game(game)
        legal_mask = validate_mirrored_legality(state)
        if not legal_mask[action]:
            action = next(index for index, legal in enumerate(legal_mask) if legal)

        original_next = game.clone_for_search()
        original_next.play_local(action)
        mirrored_next = SongoLegacyGame.from_state(
            mirror_raw_state(state).to_engine_state()
        )
        mirrored_next.play_local(action)

        assert mirror_raw_state(RawSongoState.from_game(original_next)) == (
            RawSongoState.from_game(mirrored_next)
        )
        game = original_next


def test_constant_model_is_exactly_invariant_under_physical_player_swap() -> None:
    report = evaluate_model_symmetry(
        {"constant": _ConstantPolicyValue()}, [_position()]
    )["constant"]

    assert report["value_absolute_difference_mean"] == 0.0
    assert report["policy_jensen_shannon_mean"] == 0.0
    assert report["policy_argmax_agreement"] == 1.0


def test_d_rl_audit_reports_low_visit_and_conflicting_targets() -> None:
    position = _position()
    state = position.state
    legal = position.legal_mask
    legal_actions = [index for index, is_legal in enumerate(legal) if is_legal]
    first, second = legal_actions[:2]

    def example(action: int, game_id: str, value: float) -> RLTrainingExample:
        visits = [0] * 7
        visits[action] = 2
        policy = [0.0] * 7
        policy[action] = 1.0
        return RLTrainingExample(
            state=state,
            legal_mask=legal,
            visit_counts=tuple(visits),
            policy_target=tuple(policy),
            value_target=value,
            metadata={"game_id": game_id},
        )

    report = audit_d_rl_information(
        [example(first, "g1", 1.0), example(second, "g2", -1.0)]
    )

    assert report["examples"] == 2
    assert report["mcts_visit_total_histogram"] == {2: 2}
    assert report["policy_support_histogram"] == {1: 2}
    assert report["policy_one_hot_fraction"] == 1.0
    assert report["repeated_states"] == 1
    assert report["repeated_states_with_conflicting_policy_targets"] == 1
