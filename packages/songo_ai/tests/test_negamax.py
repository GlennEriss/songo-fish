"""Tests du moteur de recherche (etape 3 : porte de passage = TT, ID,
ordonnancement et PV fonctionnels)."""

from __future__ import annotations

from songo_ai.search.negamax import EXACT, SearchLimits, TTEntry, _expand_children, iterative_deepening
from songo_ai.songo.rules import PLAYER_ONE, SongoLegacyGame, local_action_to_pit


def test_search_returns_a_legal_move_from_initial_position() -> None:
    game = SongoLegacyGame()
    result = iterative_deepening(game, SearchLimits(max_depth=4, max_nodes=50_000, max_time_s=2.0))
    assert result.local_action in game.legal_local_actions()
    assert result.depth_reached >= 1
    assert result.nodes > 0


def test_search_prefers_immediate_large_capture() -> None:
    # move local 4 (case4, 4 graines) capture 4 graines en cascade (cf.
    # test_capture_cascade_backward_stops_at_boundary) ; les autres coups
    # legaux ne capturent rien. A depth=1 la recherche doit choisir move 4.
    board = [0, 0, 0, 0, 4, 1, 1, 1, 1, 0, 0, 0, 0, 10, 0, 0]
    game = SongoLegacyGame.from_board(board, PLAYER_ONE)
    result = iterative_deepening(game, SearchLimits(max_depth=1, max_nodes=10_000, max_time_s=2.0))
    assert local_action_to_pit(PLAYER_ONE, result.local_action) == 4


def test_deeper_search_does_not_crash_and_improves_depth() -> None:
    game = SongoLegacyGame()
    shallow = iterative_deepening(game.clone_for_search(), SearchLimits(max_depth=2, max_nodes=500_000, max_time_s=5.0))
    deeper = iterative_deepening(game.clone_for_search(), SearchLimits(max_depth=5, max_nodes=500_000, max_time_s=5.0))
    assert deeper.depth_reached >= shallow.depth_reached


def test_node_budget_is_respected() -> None:
    game = SongoLegacyGame()
    result = iterative_deepening(game, SearchLimits(max_depth=20, max_nodes=500, max_time_s=5.0))
    # La verification du budget se fait a l'entree de chaque noeud : un leger
    # depassement (quelques noeuds) est possible avant l'abandon de la
    # profondeur en cours, mais pas un facteur x10.
    assert result.nodes <= 500 + 200
    assert result.depth_reached >= 1


def test_search_is_deterministic() -> None:
    game = SongoLegacyGame()
    limits = SearchLimits(max_depth=4, max_nodes=50_000, max_time_s=2.0)
    r1 = iterative_deepening(game.clone_for_search(), limits)
    r2 = iterative_deepening(game.clone_for_search(), limits)
    assert r1.local_action == r2.local_action
    assert r1.score == r2.score
    assert r1.pv == r2.pv


def test_principal_variation_is_legal_move_sequence() -> None:
    game = SongoLegacyGame()
    result = iterative_deepening(game, SearchLimits(max_depth=4, max_nodes=100_000, max_time_s=2.0))
    replay = game.clone_for_search()
    for local_action in result.pv:
        assert not replay.finished
        assert local_action in replay.legal_local_actions()
        replay.play_local(local_action)


# ---- evaluate_fn/priority_fn (section 8.1 : brancher songo_ai.hybrid sans
# dupliquer la recherche) ----------------------------------------------


def test_custom_evaluate_fn_is_used_instead_of_default() -> None:
    game = SongoLegacyGame()
    target_local_action = 3

    expected_child = game.clone_for_search()
    expected_child.play_local(target_local_action)
    expected_board = tuple(expected_child.board)

    def evaluate_fn(g, perspective):
        # negamax negocie (negation) le score de l'enfant avant de le
        # comparer au parent : on renvoie donc une valeur tres NEGATIVE
        # pour que, une fois negatee par l'appelant, l'action ciblee
        # ressorte comme la meilleure du point de vue de la racine.
        return -99999.0 if tuple(g.board) == expected_board else 0.0

    result = iterative_deepening(
        game, SearchLimits(max_depth=1, max_nodes=50_000, max_time_s=2.0), evaluate_fn=evaluate_fn
    )
    assert result.local_action == target_local_action


def test_priority_fn_orders_children_by_given_score() -> None:
    game = SongoLegacyGame()
    legal = game.legal_local_actions()
    priorities = {a: float(a) for a in legal}
    ordered = _expand_children(game, legal, tt_entry=None, priority_fn=lambda g, actions: priorities)
    ordered_actions = [a for a, _ in ordered]
    assert ordered_actions == sorted(legal, key=lambda a: priorities[a], reverse=True)


def test_tt_move_still_takes_priority_over_priority_fn() -> None:
    game = SongoLegacyGame()
    legal = game.legal_local_actions()
    priorities = {a: 0.0 for a in legal}
    priorities[2] = 100.0
    tt_entry = TTEntry(depth=1, score=0.0, flag=EXACT, best_local_action=5)
    ordered = _expand_children(game, legal, tt_entry, priority_fn=lambda g, actions: priorities)
    assert ordered[0][0] == 5
