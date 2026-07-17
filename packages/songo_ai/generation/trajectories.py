"""Generation de trajectoires legales completes depuis l'etat initial
(section 6.1 : "les positions sont extraites de trajectoires legales
demarrant de l'etat initial", pas de graines arbitraires)."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import List

from .agents import Agent
from songo_ai.songo.fast_rules import FastSongoGame
from songo_ai.songo.rules import State


@dataclass(frozen=True)
class TrajectoryPosition:
    state: State
    trajectory_id: str
    move_number: int
    total_moves: int  # rempli apres la fin de la partie (longueur totale)


def generate_trajectory(agent: Agent, rng: random.Random, max_moves: int = 400, game_factory=FastSongoGame.initial) -> List[TrajectoryPosition]:
    """Joue une partie complete et renvoie chaque position AVANT chaque coup
    (jamais la position terminale : section 6.6, "aucune position
    terminale"). `total_moves` permet de classer la phase de partie
    (section 6.3) sans dependre d'une estimation a priori.

    `game_factory` cree le moteur de jeu : FastSongoGame (Numba) par
    defaut pour la production (section 12.1 : implementation interchangeable
    validee par test differentiel), SongoLegacyGame reste utilisable pour
    deboguer/comparer."""

    trajectory_id = uuid.uuid4().hex[:12]
    game = game_factory()
    raw_positions: List[State] = []

    moves_played = 0
    while not game.finished and moves_played < max_moves:
        legal = game.legal_local_actions()
        if not legal:
            game.normalize_terminal()
            break
        raw_positions.append(game.to_state())
        local_action = agent(game, rng)
        if local_action not in legal:
            local_action = legal[0]
        game.play_local(local_action)
        moves_played += 1

    total_moves = len(raw_positions)
    return [
        TrajectoryPosition(state=state, trajectory_id=trajectory_id, move_number=i, total_moves=total_moves)
        for i, state in enumerate(raw_positions)
    ]


def generate_trajectories(
    agent_factory, num_trajectories: int, seed: int = 0, max_moves: int = 400, game_factory=FastSongoGame.initial
) -> List[TrajectoryPosition]:
    """`agent_factory(rng) -> Agent` : permet de tirer un agent (donc un
    style/une profondeur) different par partie, tout en restant
    deterministe pour une seed donnee."""
    rng = random.Random(seed)
    positions: List[TrajectoryPosition] = []
    for _ in range(num_trajectories):
        agent = agent_factory(rng)
        positions.extend(generate_trajectory(agent, rng, max_moves=max_moves, game_factory=game_factory))
    return positions
