"""Tournois (section 10.2) : faire jouer deux agents l'un contre l'autre,
en alternant strictement qui commence, et publier un intervalle de
confiance plutot qu'un pourcentage isole (section 10.2, dernier point)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from songo_ai.songo.fast_rules import FastSongoGame
from songo_ai.songo.rules import PLAYER_ONE


@dataclass
class MatchResult:
    games: int
    wins_a: int
    wins_b: int
    draws: int
    score_a: float  # wins_a + 0.5*draws, sur `games`
    win_rate_a: float
    ci_low: float
    ci_high: float


def _wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple:
    """Intervalle de confiance de Wilson (95% par defaut) pour une
    proportion, plus fiable qu'une approximation normale sur des petits
    echantillons ou des taux proches de 0/1."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    return max(0.0, center - margin), min(1.0, center + margin)


def play_match(agent_a, agent_b, num_games: int, seed: int = 0, max_moves: int = 400) -> MatchResult:
    """Alterne strictement le joueur qui commence (section 10.2). Chaque
    partie est jouee jusqu'a la fin ou `max_moves`, sur le moteur rapide
    (Numba), sans historique."""
    rng = random.Random(seed)
    wins_a = wins_b = draws = 0

    for game_index in range(num_games):
        a_starts = game_index % 2 == 0
        game = FastSongoGame.initial()
        moves_played = 0

        while not game.finished and moves_played < max_moves:
            current_is_a = (game.turn == PLAYER_ONE) == a_starts
            agent = agent_a if current_is_a else agent_b
            legal = game.legal_local_actions()
            if not legal:
                game.normalize_terminal()
                break
            action = agent(game, rng)
            if action not in legal:
                action = legal[0]
            game.play_local(action)
            moves_played += 1

        if not game.finished:
            game.normalize_terminal()

        if not game.finished or game.winner == 0:
            # Partie tronquee par max_moves sans etre reellement terminee
            # (aucun cote n'a "gagne" cette situation) ou nulle reelle :
            # comptee comme nulle, jamais attribuee arbitrairement a un cote.
            draws += 1
        else:
            winner_is_a = (game.winner == PLAYER_ONE) == a_starts
            if winner_is_a:
                wins_a += 1
            else:
                wins_b += 1

    score_a = wins_a + 0.5 * draws
    win_rate_a = score_a / num_games if num_games else 0.0
    ci_low, ci_high = _wilson_interval(score_a, num_games)

    return MatchResult(
        games=num_games,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        score_a=score_a,
        win_rate_a=win_rate_a,
        ci_low=ci_low,
        ci_high=ci_high,
    )
