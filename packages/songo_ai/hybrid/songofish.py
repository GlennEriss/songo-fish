"""Moteur SongoFish (etape 8) : reseau (ordonnancement + evaluation de
feuille, cf. network_eval.py) + recherche Alpha-Beta bornee (temps/noeuds/
profondeur, cf. songo_ai.search.negamax.SearchLimits). Reutilise le meme
Agent Protocol que le reste du projet (songo_ai.generation.agents.Agent),
pour brancher directement sur la table de jeu ou en tournoi sans code
supplementaire."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from songo_ai.hybrid.network_eval import make_network_evaluate, make_network_priority
from songo_ai.model.network import SongoNet
from songo_ai.search.negamax import SearchLimits, iterative_deepening
from songo_ai.songo.rules import SongoLegacyGame


@dataclass
class SongoFishConfig:
    max_depth: int = 10
    max_nodes: int = 300_000
    max_time_s: float = 2.0


def make_songofish_agent(model: SongoNet, config: Optional[SongoFishConfig] = None):
    """Fabrique un Agent qui recherche avec le reseau plutot que de se fier
    a lui seul (contrairement a songo_ai.model.inference.make_network_agent,
    qui joue le coup argmax d'un seul passage avant, sans recherche)."""
    cfg = config or SongoFishConfig()
    evaluate_fn = make_network_evaluate(model)
    priority_fn = make_network_priority(model)
    limits = SearchLimits(max_depth=cfg.max_depth, max_nodes=cfg.max_nodes, max_time_s=cfg.max_time_s)

    def agent(game: SongoLegacyGame, rng: Optional[random.Random] = None) -> int:
        result = iterative_deepening(game.clone_for_search(), limits, evaluate_fn, priority_fn)
        return result.local_action

    return agent
