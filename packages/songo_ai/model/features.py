"""Extraction de features pour le reseau (section 7.1).

Entrees explicitement listees par le plan : compteurs canoniques normalises,
masque des 7 coups legaux, difference des magasins et graines par
territoire, mobilite propre/adverse, indicateurs de transmission/famine et
phase de partie. Aucune metadonnee du professeur (profondeur, nombre de
noeuds...) n'entre dans le vecteur : elle ne serait pas disponible en
production (section 7.1, dernier point)."""

from __future__ import annotations

from typing import List, Sequence

from songo_ai.songo.rules import BOARD_SIZE, PLAYER_ONE, PLAYER_TWO, TOTAL_SEEDS, SongoLegacyGame

FEATURE_SIZE = 33
NUM_ACTIONS = 7


def observation_features(state: Sequence[int], legal_mask: Sequence[bool]) -> List[float]:
    """`state` est deja canonicalise (index 0-6/14 = "moi", 7-13/15 =
    "adversaire", cf. songo_ai.dataset.schema.canonicalize_board) : on
    reconstruit un jeu temporaire avec turn=PLAYER_ONE pour calculer les
    grandeurs derivees (mobilite adverse, transmission) sans dupliquer la
    logique des regles."""
    assert len(state) == BOARD_SIZE

    counters = [v / TOTAL_SEEDS for v in state]
    mask = [1.0 if m else 0.0 for m in legal_mask]

    store_diff = (state[14] - state[15]) / TOTAL_SEEDS
    my_territory = sum(state[0:7]) / TOTAL_SEEDS
    opp_territory = sum(state[7:14]) / TOTAL_SEEDS

    mine_game = SongoLegacyGame.from_board(list(state), PLAYER_ONE)
    opponent_game = SongoLegacyGame.from_board(list(state), PLAYER_TWO)

    my_mobility = sum(mask) / 7.0
    opp_mobility = len(opponent_game.legal_local_actions()) / 7.0 if not opponent_game.finished else 0.0

    can_transmit_mine = 1.0 if mine_game.can_transmit_for_player(PLAYER_ONE) else 0.0
    can_transmit_opp = 1.0 if opponent_game.can_transmit_for_player(PLAYER_TWO) else 0.0

    opp_side_empty = 1.0 if sum(state[7:14]) == 0 else 0.0
    my_side_empty = 1.0 if sum(state[0:7]) == 0 else 0.0

    seeds_in_play = sum(state[0:14]) / TOTAL_SEEDS  # proxy de phase : baisse au fil de la partie

    extra = [
        store_diff,
        my_territory,
        opp_territory,
        my_mobility,
        opp_mobility,
        can_transmit_mine,
        can_transmit_opp,
        opp_side_empty,
        my_side_empty,
        seeds_in_play,
    ]

    features = counters + mask + extra
    assert len(features) == FEATURE_SIZE, f"expected {FEATURE_SIZE} features, got {len(features)}"
    return features
