"""Tests du moteur de regles (etape 1 : porte de passage = proprietes validees).

Scenarios construits et verifies a la main a partir de la logique de
songo_ai/songo/rules.py (portage de songo_legacy_single.py, seule source de
verite retenue pour ce projet). Les anciens tests C# (SongoRulesEngineTests)
ne sont pas repris a l'identique : une divergence a ete constatee sur la
regle "1 graine depuis la case bord" (voir note en bas de fichier) et le C#
n'est plus la reference.
"""

from __future__ import annotations

import random

import pytest

from songo_ai.songo.rules import (
    NUM_ACTIONS,
    P1_STORE,
    P2_STORE,
    PLAYER_ONE,
    PLAYER_TWO,
    DRAW,
    IllegalMove,
    RepetitionTracker,
    SongoLegacyGame,
    State,
    assert_invariants,
    zobrist_hash,
)


# ---------------------------------------------------------------------------
# Invariants et proprietes (fuzzing de parties completes depuis l'etat initial)
# ---------------------------------------------------------------------------


def _random_playout(seed: int, max_moves: int = 400) -> SongoLegacyGame:
    rng = random.Random(seed)
    game = SongoLegacyGame()
    moves_played = 0
    while not game.finished and moves_played < max_moves:
        legal = game.legal_moves()
        assert_invariants(game.board)
        if not legal:
            # Aucun coup legal alors que la partie n'est pas terminee ne
            # devrait pas arriver : normalize_terminal() doit l'avoir capte.
            game.normalize_terminal()
            assert game.finished, "no legal move but game not marked finished"
            break
        move = rng.choice(legal)
        game.play(move)
        moves_played += 1
    return game


@pytest.mark.parametrize("seed", range(25))
def test_random_playouts_preserve_seed_conservation(seed: int) -> None:
    game = _random_playout(seed)
    assert_invariants(game.board)


@pytest.mark.parametrize("seed", range(10))
def test_random_playouts_eventually_finish(seed: int) -> None:
    game = _random_playout(seed, max_moves=1000)
    assert game.finished, "game did not terminate within move budget"


def test_no_move_applies_on_terminal_state() -> None:
    game = SongoLegacyGame.from_board([0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 10, 15], turn=PLAYER_ONE)
    result = game.play(0)
    assert result.reason == "finished_before_move"
    assert game.finished
    assert game.winner == PLAYER_TWO  # score_2 = 15 + 5 = 20 > score_1 = 10
    with pytest.raises(IllegalMove):
        game.validate_move(0)


def test_legal_mask_matches_legal_moves() -> None:
    game = SongoLegacyGame()
    mask = game.legal_mask()
    assert len(mask) == NUM_ACTIONS
    expected = set(game.legal_moves())
    from songo_ai.songo.rules import local_action_to_pit

    actual = {local_action_to_pit(game.turn, a) for a in range(NUM_ACTIONS) if mask[a]}
    assert actual == expected


def test_determinism_same_state_same_move_same_result() -> None:
    board = [4, 0, 3, 5, 0, 2, 1, 5, 5, 5, 5, 5, 5, 5, 0, 0]
    g1 = SongoLegacyGame.from_board(board, PLAYER_ONE)
    g2 = SongoLegacyGame.from_board(board, PLAYER_ONE)
    move = g1.legal_moves()[0]
    r1 = g1.play(move)
    r2 = g2.play(move)
    assert r1.board == r2.board
    assert r1.captured == r2.captured
    assert r1.reason == r2.reason


def test_clone_for_search_has_no_history_growth() -> None:
    game = SongoLegacyGame()
    search_game = game.clone_for_search()
    for move in list(search_game.legal_moves())[:1]:
        search_game.play(move)
    assert search_game.history == []
    assert game.history == []  # l'original n'est pas affecte


# ---------------------------------------------------------------------------
# Mecaniques precises (calculees a la main a partir de _sow/_capture)
# ---------------------------------------------------------------------------


def test_single_seed_from_edge_pit_goes_to_own_store_without_capture() -> None:
    # move=6 (case bord P1), 1 graine -> va directement au magasin P1 (14),
    # jamais capturee (arrival est un magasin).
    board = [0, 0, 0, 0, 0, 0, 1, 5, 5, 5, 5, 5, 5, 5, 0, 0]
    game = SongoLegacyGame.from_board(board, PLAYER_ONE)
    result = game.play(6)
    assert result.reason == "next_turn"
    assert result.captured == 0
    assert game.board[P1_STORE] == 1
    assert game.board[6] == 0
    assert game.turn == PLAYER_TWO
    assert game.last_sow_trace == [P1_STORE]
    assert game.last_capture_trace == []


def test_capture_cascade_backward_stops_at_boundary() -> None:
    # move=4, 4 graines: atterrit en case8 (2), la case7 devient aussi 2 ->
    # cascade arriere 8 puis 7, s'arrete avant case6 (frontiere de camp).
    board = [0, 0, 0, 0, 4, 0, 0, 1, 1, 0, 0, 0, 0, 10, 0, 0]
    game = SongoLegacyGame.from_board(board, PLAYER_ONE)
    result = game.play(4)
    assert result.captured == 4
    assert game.board[7] == 0
    assert game.board[8] == 0
    assert game.board[5] == 1
    assert game.board[6] == 1
    assert game.board[P1_STORE] == 4
    assert game.board[13] == 10  # hors de portee de la cascade
    assert game.last_sow_trace == [5, 6, 7, 8]  # une graine deposee a chaque case, dans l'ordre
    assert game.last_capture_trace == [8, 7]  # capture en arriere, de l'arrivee vers la frontiere


def test_no_capture_when_arrival_is_first_opponent_pit() -> None:
    # move=5, 2 graines, atterrit exactement en case7 (premiere case
    # adverse) : jamais de capture meme si le compte est dans [2,4].
    board = [0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 10, 0, 0]
    game = SongoLegacyGame.from_board(board, PLAYER_ONE)
    result = game.play(5)
    assert result.captured == 0


def test_capture_cascade_never_empties_opponent_entirely() -> None:
    # move=5, 8 graines : atterrit en case13 (derniere case adverse), et les
    # 7 cases adverses (7-13) valent toutes 2 apres semis -- un videment
    # total serait possible sans la protection. La cascade doit capturer
    # les 6 dernieres (13..8) et EPARGNER la toute premiere (case7), pas
    # bloquer toute la capture.
    board = [0, 0, 0, 0, 0, 8, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    game = SongoLegacyGame.from_board(board, PLAYER_ONE)
    result = game.play(5)
    assert result.captured == 12  # 6 cases x 2 graines, pas 0
    assert game.board[7] == 2  # case protegee, jamais capturee
    assert game.board[8] == 0 and game.board[13] == 0  # le reste de la cascade est bien capture
    assert game.board[P1_STORE] == 12
    assert game.last_capture_trace == [13, 12, 11, 10, 9, 8]


def test_capture_cascade_never_empties_opponent_entirely_mirror_side() -> None:
    # Meme situation, cote miroir : P2 joue, cascade vers le camp P1,
    # case0 doit etre epargnee.
    board = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 8, 0, 0, 0]
    game = SongoLegacyGame.from_board(board, PLAYER_TWO)
    result = game.play(12)
    assert result.captured == 12
    assert game.board[0] == 2  # case protegee
    assert game.board[1] == 0 and game.board[6] == 0
    assert game.board[P2_STORE] == 12


def test_finish_when_current_player_has_no_seeds_on_own_side() -> None:
    board = [0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 10, 15]
    game = SongoLegacyGame.from_board(board, PLAYER_ONE)
    game.play(0)
    assert game.finished
    assert game.winner == PLAYER_TWO
    assert game.final_score_with_territory() == (10, 20)


# ---------------------------------------------------------------------------
# Zobrist hash / repetition tracker
# ---------------------------------------------------------------------------


def test_zobrist_hash_is_deterministic_and_order_sensitive() -> None:
    board = [5] * 14 + [0, 0]
    h1 = zobrist_hash(State(tuple(board), PLAYER_ONE))
    h2 = zobrist_hash(State(tuple(board), PLAYER_ONE))
    h3 = zobrist_hash(State(tuple(board), PLAYER_TWO))
    assert h1 == h2
    assert h1 != h3


def test_repetition_tracker_counts_identical_states() -> None:
    tracker = RepetitionTracker()
    state = State.initial()
    assert tracker.record(state) == 1
    assert tracker.record(state) == 2
    assert tracker.count(state) == 2
    other = State(tuple([0] * 16), PLAYER_ONE)
    assert tracker.count(other) == 0


# NOTE (divergence C# vs reference Python retenue) :
# Le test C# `PlayTurn_SingleSeedFromCaseSixUsesLegacyArrivalAndCanCaptureCaseThirteen`
# (Songo_v1/Assets/.../SongoRulesEngineTests.cs) attend qu'un seul pion joue
# depuis la case bord (case 6) atterrisse chez l'adversaire (case 13) et
# capture, alors que songo_legacy_single.py (retenu ici comme reference)
# envoie ce pion unique directement au magasin du joueur, sans capture
# possible. Les deux fichiers ne decrivent pas la meme regle sur ce cas
# precis. A trancher si besoin, mais hors perimetre de ce projet Python.
