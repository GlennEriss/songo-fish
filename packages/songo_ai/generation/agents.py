"""Sources de trajectoires (section 6.2) : aleatoire legal et minimax a
profondeur variable. MCTS et corpus humain sont hors perimetre de ce
premier lot de 10k positions (section 6.2, lignes "MCTS"/"Corpus humain
futur") : ils s'ajouteront comme sources supplementaires plus tard sans
changer le schema de sortie.
"""

from __future__ import annotations

import random
from typing import Protocol

from songo_ai.search.negamax import EvaluateFn, SearchLimits, default_evaluate, iterative_deepening
from songo_ai.songo.rules import SongoLegacyGame


class Agent(Protocol):
    def __call__(self, game: SongoLegacyGame, rng: random.Random) -> int: ...


def random_agent(game: SongoLegacyGame, rng: random.Random) -> int:
    return rng.choice(game.legal_local_actions())


def make_shallow_search_agent(
    max_depth: int, max_nodes: int = 20_000, max_time_s: float = 0.5, evaluate_fn: EvaluateFn = default_evaluate
) -> Agent:
    """Fabrique un agent "minimax de niveaux varies" (section 6.2) : plus
    `max_depth` est petit, plus les trajectoires produites ressemblent a un
    joueur faible/tactique court ; plus il est grand, a un joueur fort.

    `evaluate_fn` (defaut `default_evaluate`) : parametre expose surtout
    pour comparer AVEC/SANS le bonus territoire bidoua/Yinda a recherche
    identique (cf. apps/table/src/controllers.py `minimax:...:baseline`) --
    la generation de dataset (usage principal de cette fonction) n'a pas
    besoin d'y toucher, le defaut est le bon reglage."""

    limits = SearchLimits(max_depth=max_depth, max_nodes=max_nodes, max_time_s=max_time_s)

    def agent(game: SongoLegacyGame, rng: random.Random) -> int:
        result = iterative_deepening(game.clone_for_search(), limits, evaluate_fn)
        return result.local_action

    return agent


def make_mixed_agent(rng: random.Random, random_weight: float = 0.4) -> Agent:
    """Melange aleatoire legal / minimax a profondeur tiree au sort a chaque
    coup : source unique couvrant a la fois "Aleatoire legal" et "Minimax de
    niveaux varies" (section 6.2), utile pour diversifier une trajectoire
    unique plutot que de figer un seul style par partie."""

    depth_choices = [1, 2, 3, 4, 6]
    search_agents = {d: make_shallow_search_agent(d) for d in depth_choices}

    def agent(game: SongoLegacyGame, agent_rng: random.Random) -> int:
        if agent_rng.random() < random_weight:
            return random_agent(game, agent_rng)
        depth = agent_rng.choice(depth_choices)
        return search_agents[depth](game, agent_rng)

    return agent
