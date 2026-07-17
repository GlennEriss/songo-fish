"""Verrouille le sens de semis horaire (droite->gauche en bas,
gauche->droite en haut), verifie contre l'exemple du livre de Mbarga Owona
(p.20) et contre le retour utilisateur : "la case du joueur sud est vers
la 1ere a droite, la case 1 du joueur nord vers la 1ere a gauche"."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from board_view import pit_position


def test_bottom_row_pit0_is_rightmost() -> None:
    x0, _ = pit_position(0)
    x6, _ = pit_position(6)
    assert x0 > x6  # pit 0 = case "1" du joueur du bas, la plus a droite


def test_bottom_row_index_decreases_left_to_right() -> None:
    xs = [pit_position(i)[0] for i in range(7)]
    assert xs == sorted(xs, reverse=True)  # index croissant -> x decroissant (droite->gauche)


def test_top_row_pit7_is_leftmost() -> None:
    x7, _ = pit_position(7)
    x13, _ = pit_position(13)
    assert x7 < x13  # pit 7 = case "1" du joueur du haut, la plus a gauche


def test_top_row_index_increases_left_to_right() -> None:
    xs = [pit_position(i)[0] for i in range(7, 14)]
    assert xs == sorted(xs)  # index croissant -> x croissant (gauche->droite)


def test_crossing_pits_are_adjacent_at_the_left_edge() -> None:
    # pit 6 (dernier du bas) -> pit 7 (premier du haut) : la traversee se
    # fait a gauche, donc les deux cases doivent etre proches en x.
    x6, _ = pit_position(6)
    x7, _ = pit_position(7)
    assert abs(x6 - x7) < 5


def test_wraparound_pits_are_adjacent_at_the_right_edge() -> None:
    # pit 13 (dernier du haut) -> pit 0 (premier du bas, retour de boucle) :
    # la traversee se fait a droite.
    x13, _ = pit_position(13)
    x0, _ = pit_position(0)
    assert abs(x13 - x0) < 5
