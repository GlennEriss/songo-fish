"""Tests de l'extraction de features (section 7.1)."""

from __future__ import annotations

from songo_ai.model.features import FEATURE_SIZE, observation_features
from songo_ai.songo.rules import PLAYER_ONE
from songo_ai.dataset import canonicalize_board


def test_observation_features_has_expected_size() -> None:
    board = [5] * 14 + [0, 0]
    legal_mask = [True] * 7
    features = observation_features(board, legal_mask)
    assert len(features) == FEATURE_SIZE


def test_observation_features_initial_state_is_symmetric() -> None:
    board = canonicalize_board(tuple([5] * 14 + [0, 0]), PLAYER_ONE)
    features = observation_features(board, [True] * 7)
    # Position initiale : magasins vides (store_diff=0), mobilite identique
    # des deux cotes (7 coups chacun), aucun cote vide, aucun camp adverse a
    # zero graine.
    store_diff = features[16 + 7]
    my_mobility = features[16 + 7 + 3]
    opp_mobility = features[16 + 7 + 4]
    opp_side_empty = features[16 + 7 + 7]
    assert store_diff == 0.0
    assert my_mobility == 1.0
    assert opp_mobility == 1.0
    assert opp_side_empty == 0.0


def test_observation_features_detects_opponent_side_empty() -> None:
    board = [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    features = observation_features(board, [False, False, False, False, False, False, True])
    opp_side_empty = features[16 + 7 + 7]
    assert opp_side_empty == 1.0
