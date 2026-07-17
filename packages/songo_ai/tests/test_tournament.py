"""Tests du module de tournoi (section 10.2)."""

from __future__ import annotations

from songo_ai.evaluation import play_match
from songo_ai.evaluation.tournament import _wilson_interval
from songo_ai.generation import make_shallow_search_agent, random_agent


def test_wilson_interval_contains_the_point_estimate() -> None:
    low, high = _wilson_interval(50, 100)
    assert low < 0.5 < high


def test_wilson_interval_narrows_with_more_games() -> None:
    low_small, high_small = _wilson_interval(50, 100)
    low_big, high_big = _wilson_interval(500, 1000)
    assert (high_big - low_big) < (high_small - low_small)


def test_play_match_returns_consistent_totals() -> None:
    result = play_match(random_agent, random_agent, num_games=20, seed=0)
    assert result.games == 20
    assert result.wins_a + result.wins_b + result.draws == 20
    assert 0.0 <= result.win_rate_a <= 1.0
    assert result.ci_low <= result.win_rate_a <= result.ci_high


def test_shallow_search_beats_random_agent() -> None:
    strong = make_shallow_search_agent(max_depth=4, max_nodes=20_000, max_time_s=1.0)
    result = play_match(strong, random_agent, num_games=20, seed=1)
    # Un minimax meme peu profond doit dominer un agent aleatoire.
    assert result.win_rate_a > 0.7


def test_two_deterministic_agents_produce_degenerate_diversity_without_opening_randomization() -> None:
    # Deux agents deterministes (minimax sans aleatoire) ne rejouent que 2
    # parties distinctes (une par cote de depart), peu importe num_games :
    # c'est precisement le piege documente en tete de module.
    agent_x = make_shallow_search_agent(max_depth=2, max_nodes=5_000, max_time_s=0.5)
    agent_y = make_shallow_search_agent(max_depth=1, max_nodes=5_000, max_time_s=0.5)
    result = play_match(agent_x, agent_y, num_games=10, seed=7, opening_random_plies=0)
    assert result.distinct_games == 2


def test_opening_random_plies_restores_game_diversity() -> None:
    agent_x = make_shallow_search_agent(max_depth=2, max_nodes=5_000, max_time_s=0.5)
    agent_y = make_shallow_search_agent(max_depth=1, max_nodes=5_000, max_time_s=0.5)
    result = play_match(agent_x, agent_y, num_games=10, seed=7, opening_random_plies=4)
    assert result.distinct_games > 2
