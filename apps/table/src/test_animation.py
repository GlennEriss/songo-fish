"""Tests de la machine a etats d'animation (logique pure, sans pygame)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from animation import CAPTURE_STEP_SECONDS, SOW_STEP_SECONDS, MoveAnimation


def test_source_pit_empties_immediately() -> None:
    board_before = [5] * 14 + [0, 0]
    anim = MoveAnimation.start(board_before, source_pit=1, sow_trace=[2, 3, 4, 5, 6], capture_trace=[], final_board=[5, 0, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 0, 0])
    assert anim.board[1] == 0
    assert anim.board[2] == 5  # pas encore anime


def test_sowing_advances_one_pit_at_a_time() -> None:
    board_before = [5] * 14 + [0, 0]
    final = [5, 0, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 0, 0]
    anim = MoveAnimation.start(board_before, source_pit=1, sow_trace=[2, 3, 4, 5, 6], capture_trace=[], final_board=final)

    anim.update(SOW_STEP_SECONDS - 0.01)
    assert anim.board[2] == 5  # pas encore atteint le seuil du premier pas
    assert anim.receiving_pit is None

    anim.update(0.02)
    assert anim.board[2] == 6  # premier pas franchi
    assert anim.receiving_pit == 2
    assert anim.board[3] == 5  # le reste n'a pas encore bouge


def test_animation_converges_exactly_to_final_board() -> None:
    board_before = [5] * 14 + [0, 0]
    final = [5, 0, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 0, 0]
    anim = MoveAnimation.start(board_before, source_pit=1, sow_trace=[2, 3, 4, 5, 6], capture_trace=[], final_board=final)

    for _ in range(1000):
        if anim.finished:
            break
        anim.update(1 / 60)

    assert anim.finished
    assert anim.board == final


def test_capture_phase_moves_seeds_to_correct_store() -> None:
    # capture cote joueur 1 (pits < 7 captures vers le magasin P2 = index 15,
    # cf. songo_ai.songo.rules._capture) -- ici on simule l'inverse : joueur1
    # a capture les pits 8 et 7 (cote adverse), donc le butin va au magasin P1 (14).
    board_before = [0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 10, 0, 0]
    final_board = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 10, 4, 0]
    anim = MoveAnimation.start(
        board_before, source_pit=4, sow_trace=[5, 6, 7, 8], capture_trace=[8, 7], final_board=final_board
    )

    for _ in range(1000):
        if anim.finished:
            break
        anim.update(1 / 60)

    assert anim.finished
    assert anim.board == final_board
    assert anim.board[14] == 4  # magasin P1 a bien recu le butin


def test_no_sow_trace_but_capture_trace_starts_in_capturing_phase() -> None:
    anim = MoveAnimation.start([5] * 16, source_pit=6, sow_trace=[], capture_trace=[7], final_board=[5] * 16)
    assert anim.phase == "capturing"


def test_empty_traces_finish_immediately() -> None:
    board = [5] * 16
    anim = MoveAnimation.start(board, source_pit=0, sow_trace=[], capture_trace=[], final_board=board)
    assert anim.finished
