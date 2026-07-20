"""Tests de la conversion du format externe songo-model-stockfish (etape
5+, section "vraies parties" -- voir external_import.py pour le detail de
l'encodage et comment il a ete determine)."""

from __future__ import annotations

from songo_ai.dataset.external_import import convert_row_to_board14, resolve_turn
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO, SongoLegacyGame


def test_convert_row_to_board14_reverses_south_only() -> None:
    # south (7 premiers) inverse, north (7 derniers) inchange.
    row = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
    assert convert_row_to_board14(row) == [7, 6, 5, 4, 3, 2, 1, 10, 11, 12, 13, 14, 15, 16]


def test_convert_row_to_board14_initial_position_is_symmetric() -> None:
    # Le plateau initial (tout a 5) est invariant par cette conversion :
    # aucune ambiguite possible sur ce cas au moins.
    row = [5] * 14
    assert convert_row_to_board14(row) == [5] * 14


def test_resolve_turn_finds_the_matching_player() -> None:
    board14 = convert_row_to_board14([5, 6, 6, 6, 6, 6, 0, 5, 5, 5, 5, 5, 5, 5])
    game_p2 = SongoLegacyGame.from_board(board14 + [0, 0], PLAYER_TWO)
    theirs_mask = list(game_p2.legal_mask())
    assert resolve_turn(board14, theirs_mask) in (PLAYER_ONE, PLAYER_TWO)
    # doit specifiquement retrouver P2 ici (P1 aurait un masque different
    # puisque son cote -- converti depuis 'south' -- a une case vide)
    assert resolve_turn(board14, theirs_mask) == PLAYER_TWO


def test_resolve_turn_returns_none_when_neither_player_matches() -> None:
    board14 = [5] * 14
    impossible_mask = [False] * 7  # aucun coup legal ne matche un plateau plein
    assert resolve_turn(board14, impossible_mask) is None
