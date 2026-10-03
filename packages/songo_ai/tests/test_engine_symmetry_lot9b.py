"""Tests de caractérisation des transformations P1/P2 du Lot 9B."""

from __future__ import annotations

import pytest

from songo_ai.dataset import RawSongoState
from songo_ai.evaluation import (
    SWAP_KEEP_LOCAL,
    SWAP_REVERSE_LOCAL,
    audit_legal_equivariance,
    audit_state_collection,
    audit_transition_equivariance,
    summarize_policy_repetitions,
)
from songo_ai.search import terminal_value
from songo_ai.songo.rules import DRAW, PLAYER_ONE, PLAYER_TWO, SongoLegacyGame


FULL_LAP_P2 = (0, 0, 0, 0, 0, 4, 1, 14, 0, 0, 0, 0, 51, 0, 0, 0)


@pytest.mark.parametrize("transform", [SWAP_KEEP_LOCAL, SWAP_REVERSE_LOCAL])
def test_player_swap_transform_is_an_involution(transform) -> None:
    state = RawSongoState((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5), 2)
    assert transform.transform_state(transform.transform_state(state)) == state
    assert all(
        transform.transform_action(transform.transform_action(action)) == action
        for action in range(7)
    )


def test_action_mask_and_policy_mapping_are_explicit() -> None:
    mask = (True, False, True, False, False, True, False)
    policy = (0.2, 0.0, 0.3, 0.0, 0.0, 0.5, 0.0)

    assert SWAP_KEEP_LOCAL.transform_mask(mask) == mask
    assert SWAP_KEEP_LOCAL.transform_policy(policy) == policy
    assert SWAP_REVERSE_LOCAL.transform_mask(mask) == tuple(reversed(mask))
    assert SWAP_REVERSE_LOCAL.transform_policy(policy) == tuple(reversed(policy))


def test_winner_mapping_and_local_value_target_derivation() -> None:
    assert SWAP_KEEP_LOCAL.transform_winner(None) is None
    assert SWAP_KEEP_LOCAL.transform_winner(DRAW) == DRAW
    assert SWAP_KEEP_LOCAL.transform_winner(PLAYER_ONE) == PLAYER_TWO
    assert SWAP_KEEP_LOCAL.transform_winner(PLAYER_TWO) == PLAYER_ONE
    for winner in (DRAW, PLAYER_ONE, PLAYER_TWO):
        assert terminal_value(winner, PLAYER_ONE) == terminal_value(
            SWAP_KEEP_LOCAL.transform_winner(winner), PLAYER_TWO
        )


def test_keep_local_is_equivariant_on_initial_transitions() -> None:
    state = RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE)
    assert audit_legal_equivariance(state, SWAP_KEEP_LOCAL)["success"]
    assert all(
        audit_transition_equivariance(state, action, SWAP_KEEP_LOCAL)["success"]
        for action in range(7)
    )


def test_reverse_local_is_not_a_transition_symmetry_even_initially() -> None:
    state = RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE)
    assert audit_legal_equivariance(state, SWAP_REVERSE_LOCAL)["success"]
    assert all(
        not audit_transition_equivariance(state, action, SWAP_REVERSE_LOCAL)["success"]
        for action in range(7)
    )


def test_known_full_lap_counterexample_is_preserved_without_engine_change() -> None:
    state = RawSongoState(FULL_LAP_P2, PLAYER_TWO)
    report = audit_transition_equivariance(state, 0, SWAP_KEEP_LOCAL)

    assert not report["success"]
    assert report["failures"] == ["pits", "stores"]
    assert report["seed_count"] == 14
    assert report["original_final_store_deposit"]
    assert report["transformed_final_store_deposit"]
    assert report["original_capture"] == 0
    assert report["transformed_capture"] == 2


def test_collection_audit_is_deterministic() -> None:
    initial = RawSongoState(tuple([5] * 14 + [0, 0]), PLAYER_ONE)
    full_lap = RawSongoState(FULL_LAP_P2, PLAYER_TWO)
    states = [("initial", initial), ("full-lap", full_lap)]

    first = audit_state_collection(states, SWAP_KEEP_LOCAL)
    second = audit_state_collection(states, SWAP_KEEP_LOCAL)

    assert first == second
    assert first["states_tested"] == 2
    assert first["transition_failures"] == 1
    assert first["failure_families"]["final_store_deposit"] == 1


def test_audit_does_not_modify_input_game_behavior() -> None:
    game = SongoLegacyGame()
    state = RawSongoState.from_game(game)
    before = (tuple(game.board), game.turn, game.finished, game.winner)
    audit_state_collection([("one", state)], SWAP_KEEP_LOCAL)
    assert before == (tuple(game.board), game.turn, game.finished, game.winner)


def test_policy_stability_metrics_separate_states_and_are_deterministic() -> None:
    per_state = [
        [
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ],
        [
            (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        ],
    ]
    first = summarize_policy_repetitions(per_state)
    second = summarize_policy_repetitions(per_state)

    assert first == second
    assert first["states"] == 2
    assert first["one_hot_fraction"] == 0.5
    assert first["visited_actions_mean"] == 1.5
    assert first["states_with_stable_target_fraction"] == 0.5
    assert first["pairwise_argmax_agreement"] == 0.5
