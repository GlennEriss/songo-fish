"""Tests du moteur SongoFish (etape 8) : reseau (evaluation + ordonnancement)
branche sur la recherche existante, sans reseau entraine reel -- un
SongoNet non entraine suffit pour verifier le branchement (formes, plages
de valeurs, absence de crash), la qualite de jeu se verifie a part
(tournoi, table de jeu)."""

from __future__ import annotations

from songo_ai.hybrid import SongoFishConfig, make_network_evaluate, make_network_priority, make_songofish_agent
from songo_ai.hybrid.network_eval import EVAL_SCALE
from songo_ai.model.network import SongoNet
from songo_ai.search.negamax import default_evaluate
from songo_ai.songo.rules import PLAYER_ONE, SongoLegacyGame


def _fresh_model() -> SongoNet:
    model = SongoNet(width=32, num_blocks=1, dropout=0.0)
    model.eval()
    return model


def test_network_evaluate_matches_default_on_terminal_position() -> None:
    board = [0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 40, 20]
    game = SongoLegacyGame.from_board(board, PLAYER_ONE)
    game.normalize_terminal()
    assert game.finished

    evaluate = make_network_evaluate(_fresh_model())
    assert evaluate(game, PLAYER_ONE) == default_evaluate(game, PLAYER_ONE)


def test_network_evaluate_is_bounded_on_non_terminal_position() -> None:
    game = SongoLegacyGame()
    evaluate = make_network_evaluate(_fresh_model())
    value = evaluate(game, game.turn)
    assert abs(value) <= EVAL_SCALE + 1e-6


def test_network_priority_covers_exactly_the_legal_actions() -> None:
    game = SongoLegacyGame()
    legal = game.legal_local_actions()
    priority = make_network_priority(_fresh_model())
    priorities = priority(game, legal)
    assert set(priorities.keys()) == set(legal)
    assert all(0.0 <= v <= 1.0 for v in priorities.values())


def test_songofish_agent_returns_a_legal_move() -> None:
    game = SongoLegacyGame()
    agent = make_songofish_agent(_fresh_model(), SongoFishConfig(max_depth=3, max_nodes=20_000, max_time_s=2.0))
    action = agent(game, None)
    assert action in game.legal_local_actions()


def test_songofish_agent_handles_near_terminal_position() -> None:
    # Territoire adverse presque asseche : verifie que le branchement reseau
    # ne casse pas sur les cas limites (famine/normalize_terminal) deja geres
    # par le moteur de recherche.
    board = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 30, 30]
    game = SongoLegacyGame.from_board(board, PLAYER_ONE)
    agent = make_songofish_agent(_fresh_model(), SongoFishConfig(max_depth=4, max_nodes=20_000, max_time_s=2.0))
    if not game.finished and game.legal_local_actions():
        action = agent(game, None)
        assert action in game.legal_local_actions()
