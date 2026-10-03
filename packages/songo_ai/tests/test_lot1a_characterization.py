"""Lot 1A: tests de caracterisation du moteur, sans correction metier.

Ces tests figent volontairement le comportement observe au 24 septembre
2026, y compris les asymetries suspectes. Ils ne declarent pas que ce
comportement est la bonne regle du Songo. Une correction validee en Lot 1B
devra remplacer les assertions de caracterisation concernees.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from songo_ai.dataset.schema import canonicalize_board
from songo_ai.songo.fast_rules import FastSongoGame, capture as fast_capture, sow as fast_sow
from songo_ai.songo.rules import (
    DRAW,
    P1_STORE,
    P2_STORE,
    PLAYER_ONE,
    PLAYER_TWO,
    SongoLegacyGame,
    assert_invariants,
    local_action_to_pit,
)


# Plus petit nombre de graines permettant au code actuel d'atteindre le
# traitement "dernier pion apres un tour complet": 14. Les autres graines
# sont placees de facon a conserver les 70 graines et a isoler une case a 2
# graines sur le bord adverse apres le semis.
FULL_LAP_P1 = [14, 0, 0, 0, 0, 51, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0]
FULL_LAP_P2 = [0, 0, 0, 0, 0, 4, 1, 14, 0, 0, 0, 0, 51, 0, 0, 0]


def _swap_player(player: int) -> int:
    return PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE


def _relative_player(player: int, perspective: int) -> int:
    return player if perspective == PLAYER_ONE else _swap_player(player)


def _play_in_original_and_canonical_frames(board, turn, local_action):
    original = SongoLegacyGame.from_board(board, turn)
    canonical = SongoLegacyGame.from_board(canonicalize_board(tuple(board), turn), PLAYER_ONE)
    assert local_action in original.legal_local_actions()
    assert local_action in canonical.legal_local_actions()

    original_result = original.play_local(local_action)
    canonical_result = canonical.play_local(local_action)
    expected_board = canonicalize_board(tuple(original.board), turn)
    expected_turn = _relative_player(original.turn, turn)
    expected_winner = None if original.winner is None else (
        DRAW if original.winner == DRAW else _relative_player(original.winner, turn)
    )
    return original, canonical, original_result, canonical_result, expected_board, expected_turn, expected_winner


def test_reference_full_lap_store_arrival_characterizes_current_j1_j2_asymmetry() -> None:
    j1 = SongoLegacyGame.from_board(FULL_LAP_P1, PLAYER_ONE)
    j1_store_before = j1.board[P1_STORE]
    j1_arrival = j1._sow(local_action_to_pit(PLAYER_ONE, 0))
    j1_capture_input = j1_arrival
    j1_captured = j1._capture(0, j1_capture_input)

    assert j1.last_sow_trace == list(range(1, 14)) + [P1_STORE]
    assert j1.last_sow_trace[-1] == P1_STORE  # arrivee physique reelle
    assert j1_arrival == 13  # valeur actuellement retournee par _sow
    assert j1_capture_input == 13
    assert j1.last_capture_trace == [13]
    assert j1_captured == 2
    assert (j1_store_before, j1.board[P1_STORE]) == (0, 3)
    assert sum(j1.board) == 70

    j2 = SongoLegacyGame.from_board(FULL_LAP_P2, PLAYER_TWO)
    j2_store_before = j2.board[P2_STORE]
    j2_arrival = j2._sow(local_action_to_pit(PLAYER_TWO, 0))
    j2_capture_input = j2_arrival
    j2_captured = j2._capture(7, j2_capture_input)

    assert j2.last_sow_trace == list(range(8, 14)) + list(range(0, 7)) + [P2_STORE]
    assert j2.last_sow_trace[-1] == P2_STORE  # arrivee physique reelle
    assert j2_arrival == P1_STORE  # valeur actuellement retournee par _sow
    assert j2_capture_input == P1_STORE
    assert j2.last_capture_trace == []
    assert j2_captured == 0
    assert (j2_store_before, j2.board[P2_STORE]) == (0, 1)
    assert sum(j2.board) == 70


def test_fast_rules_full_lap_matches_the_same_current_asymmetry() -> None:
    j1_board = np.array(FULL_LAP_P1, dtype=np.int64)
    j1_arrival = int(fast_sow(j1_board, 0))
    j1_captured = int(fast_capture(j1_board, 0, j1_arrival))
    assert j1_arrival == 13
    assert j1_captured == 2
    assert int(j1_board[P1_STORE]) == 3
    assert int(j1_board.sum()) == 70

    j2_board = np.array(FULL_LAP_P2, dtype=np.int64)
    j2_arrival = int(fast_sow(j2_board, 7))
    j2_captured = int(fast_capture(j2_board, 7, j2_arrival))
    assert j2_arrival == P1_STORE
    assert j2_captured == 0
    assert int(j2_board[P2_STORE]) == 1
    assert int(j2_board.sum()) == 70


@pytest.mark.parametrize(
    "board,turn",
    [(FULL_LAP_P1, PLAYER_ONE), (FULL_LAP_P2, PLAYER_TWO)],
)
def test_reference_and_fast_rules_are_equivalent_on_full_lap_adversarial_cases(board, turn) -> None:
    reference = SongoLegacyGame.from_board(board, turn)
    fast = FastSongoGame.from_board(board, turn)
    r1 = reference.play_local(0)
    r2 = fast.play_local(0)
    assert r1.captured == r2.captured
    assert r1.reason == r2.reason
    assert r1.finished == r2.finished
    assert r1.winner == r2.winner
    assert list(reference.board) == fast.board.tolist()


def test_canonicalization_succeeds_on_simple_non_full_lap_transition() -> None:
    board = [5] * 14 + [0, 0]
    original, canonical, r1, r2, expected_board, expected_turn, expected_winner = (
        _play_in_original_and_canonical_frames(board, PLAYER_TWO, 0)
    )
    assert tuple(canonical.board) == expected_board
    assert canonical.turn == expected_turn
    assert canonical.winner == expected_winner
    assert r1.captured == r2.captured
    assert canonical.legal_mask() == original.legal_mask()


def test_canonicalization_currently_fails_on_minimal_full_lap_case() -> None:
    original, canonical, r1, r2, expected_board, expected_turn, expected_winner = (
        _play_in_original_and_canonical_frames(FULL_LAP_P2, PLAYER_TWO, 0)
    )
    # Caracterisation, pas validation metier: la position canonique J1
    # capture 2 graines alors que la position originale J2 n'en capture pas.
    assert r1.captured == 0
    assert r2.captured == 2
    assert tuple(canonical.board) != expected_board
    assert canonical.turn == expected_turn
    assert canonical.winner == expected_winner


@pytest.mark.parametrize("game_cls", [SongoLegacyGame, FastSongoGame])
def test_legal_moves_player_parameter_characterizes_current_turn_dependency(game_cls) -> None:
    game = game_cls.initial() if game_cls is FastSongoGame else game_cls()
    assert game.legal_moves() == list(range(7))
    assert game.legal_moves(PLAYER_ONE) == list(range(7))
    assert game.legal_moves(PLAYER_TWO) == []
    assert game.legal_mask() == (True,) * 7

    game.play_local(0)
    assert game.turn == PLAYER_TWO
    assert game.legal_moves(PLAYER_ONE) == []
    assert game.legal_moves(PLAYER_TWO) == game.legal_moves()


TERMINAL_CASES = {
    # Le magasin depasse deja 35, mais normalize_terminal ne regarde pas ce
    # critere dans le comportement actuel.
    "imported_store_over_35": ([14, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 36, 0], PLAYER_ONE),
    "stores_35_35": ([0] * 14 + [35, 35], PLAYER_ONE),
    # Somme maximale sans transmission pour P1: 6,5,4,3,2,1,1.
    "famine_no_transmit": ([6, 5, 4, 3, 2, 1, 1] + [0] * 7 + [24, 24], PLAYER_ONE),
    "current_camp_empty": ([0] * 7 + [20, 0, 0, 0, 0, 0, 0] + [25, 25], PLAYER_ONE),
}


@pytest.mark.parametrize("game_cls", [SongoLegacyGame, FastSongoGame])
@pytest.mark.parametrize("case_name", sorted(TERMINAL_CASES))
def test_terminal_normalization_characterizes_current_behavior(game_cls, case_name) -> None:
    board, turn = TERMINAL_CASES[case_name]
    game = game_cls.from_board(board, turn)
    before = (game.finished, game.winner, game.legal_mask(), tuple(game.legal_moves()))
    game.normalize_terminal()
    after_once = (game.finished, game.winner, game.legal_mask(), tuple(game.legal_moves()))
    game.normalize_terminal()
    after_twice = (game.finished, game.winner, game.legal_mask(), tuple(game.legal_moves()))

    assert after_once == after_twice  # normalisation idempotente
    assert sum(int(v) for v in game.board) == 70

    if case_name == "imported_store_over_35":
        assert before[0:2] == (False, None)
        assert after_once[0:2] == (False, None)
        assert any(after_once[2])
    elif case_name == "stores_35_35":
        assert before == (False, None, (False,) * 7, ())
        assert after_once[0:2] == (True, DRAW)
    elif case_name == "famine_no_transmit":
        assert before == (False, None, (False,) * 7, ())
        assert after_once[0:2] == (True, PLAYER_ONE)
    else:
        assert before == (False, None, (False,) * 7, ())
        assert after_once[0:2] == (True, PLAYER_TWO)


@pytest.mark.parametrize("game_cls", [SongoLegacyGame, FastSongoGame])
@pytest.mark.parametrize(
    "board,turn,local_action",
    [
        ([5] * 14 + [0, 0], PLAYER_ONE, 0),
        (FULL_LAP_P1, PLAYER_ONE, 0),
        (FULL_LAP_P2, PLAYER_TWO, 0),
        ([0, 0, 0, 0, 4, 0, 0, 1, 1, 0, 0, 0, 0, 10, 24, 30], PLAYER_ONE, 4),
        ([0, 0, 0, 0, 0, 0, 2] + [0] * 7 + [34, 34], PLAYER_ONE, 6),
    ],
)
def test_seed_conservation_on_adversarial_transitions(game_cls, board, turn, local_action) -> None:
    assert sum(board) == 70
    game = game_cls.from_board(board, turn)
    assert local_action in game.legal_local_actions()
    game.play_local(local_action)
    assert sum(int(v) for v in game.board) == 70


@pytest.mark.parametrize("game_cls", [SongoLegacyGame, FastSongoGame])
def test_seed_conservation_on_long_random_trajectories(game_cls) -> None:
    for seed in range(10):
        rng = random.Random(seed)
        game = game_cls.initial() if game_cls is FastSongoGame else game_cls()
        for _ in range(500):
            assert_invariants(int(v) for v in game.board)
            if game.finished:
                break
            legal = game.legal_local_actions()
            if not legal:
                game.normalize_terminal()
                break
            game.play_local(rng.choice(legal))
        assert_invariants(int(v) for v in game.board)

