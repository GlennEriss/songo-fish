"""Test differentiel obligatoire (section 12.1 : une seule implementation
des regles doit faire foi). FastSongoGame (Numba) n'est autorise a remplacer
SongoLegacyGame (reference Python pure) dans la recherche/le professeur que
tant que ce test reste vert : chaque coup, sur des milliers de trajectoires
aleatoires, doit produire un etat rigoureusement identique sur les deux
moteurs.
"""

from __future__ import annotations

import random

import pytest

from songo_ai.songo.fast_rules import FastSongoGame
from songo_ai.songo.rules import SongoLegacyGame, assert_invariants


def _play_lockstep(seed: int, max_moves: int = 400) -> None:
    rng = random.Random(seed)
    reference = SongoLegacyGame()
    fast = FastSongoGame.initial()

    moves_played = 0
    while moves_played < max_moves:
        assert reference.finished == fast.finished, f"finished mismatch at move {moves_played}"
        assert reference.turn == fast.turn, f"turn mismatch at move {moves_played}"
        assert list(reference.board) == [int(x) for x in fast.board], f"board mismatch at move {moves_played}"

        if reference.finished:
            assert reference.winner == fast.winner
            break

        ref_legal = reference.legal_moves()
        fast_legal = fast.legal_moves()
        assert ref_legal == fast_legal, f"legal_moves mismatch at move {moves_played}: {ref_legal} vs {fast_legal}"
        assert reference.legal_mask() == fast.legal_mask()

        if not ref_legal:
            reference.normalize_terminal()
            fast.normalize_terminal()
            assert reference.finished == fast.finished
            assert reference.winner == fast.winner
            break

        move = rng.choice(ref_legal)
        r1 = reference.play(move)
        r2 = fast.play(move)

        assert r1.captured == r2.captured, f"captured mismatch at move {moves_played}"
        assert r1.finished == r2.finished
        assert r1.winner == r2.winner
        assert r1.reason == r2.reason, f"reason mismatch at move {moves_played}: {r1.reason} vs {r2.reason}"
        assert r1.board == [int(x) for x in fast.board]
        assert_invariants(reference.board)
        assert_invariants([int(x) for x in fast.board])

        moves_played += 1


@pytest.mark.parametrize("seed", range(60))
def test_fast_matches_reference_on_random_trajectories(seed: int) -> None:
    _play_lockstep(seed)


def test_fast_matches_reference_on_hand_crafted_capture_cascade() -> None:
    board = [0, 0, 0, 0, 4, 0, 0, 1, 1, 0, 0, 0, 0, 10, 0, 0]
    reference = SongoLegacyGame.from_board(board, 1)
    fast = FastSongoGame.from_board(board, 1)
    r1 = reference.play(4)
    r2 = fast.play(4)
    assert r1.captured == r2.captured == 4
    assert list(reference.board) == [int(x) for x in fast.board]


def test_fast_matches_reference_on_single_seed_edge_case() -> None:
    board = [0, 0, 0, 0, 0, 0, 1, 5, 5, 5, 5, 5, 5, 5, 0, 0]
    reference = SongoLegacyGame.from_board(board, 1)
    fast = FastSongoGame.from_board(board, 1)
    r1 = reference.play(6)
    r2 = fast.play(6)
    assert r1.reason == r2.reason == "next_turn"
    assert list(reference.board) == [int(x) for x in fast.board]


def test_fast_illegal_move_raises_like_reference() -> None:
    from songo_ai.songo.rules import IllegalMove

    reference = SongoLegacyGame()
    fast = FastSongoGame.initial()
    illegal_move = 7  # appartient au joueur 2, mais joueur 1 au trait
    with pytest.raises(IllegalMove):
        reference.validate_move(illegal_move)
    with pytest.raises(IllegalMove):
        fast.validate_move(illegal_move)
