#!/usr/bin/env python3
"""Benchmark de l'alpha-beta reel (annexe C : Alpha-Beta).

Mesure noeuds/seconde et profondeur atteinte a budget de temps fixe, sur la
position initiale et sur quelques positions de milieu de partie generees
aleatoirement. Sert a decider si le moteur de regles doit etre accelere
(Numba) avant de construire le professeur (etape 4).
"""

from __future__ import annotations

import random
import time

from songo_ai.search.negamax import SearchLimits, iterative_deepening
from songo_ai.songo.rules import SongoLegacyGame


def random_midgame(seed: int, num_moves: int = 20) -> SongoLegacyGame:
    rng = random.Random(seed)
    game = SongoLegacyGame()
    played = 0
    while not game.finished and played < num_moves:
        legal = game.legal_moves()
        if not legal:
            game.normalize_terminal()
            break
        game.play(rng.choice(legal))
        played += 1
    return game


def main() -> None:
    positions = [("initiale", SongoLegacyGame())]
    positions += [(f"milieu-{seed}", random_midgame(seed)) for seed in range(5)]

    for label, game in positions:
        if game.finished:
            print(f"{label}: partie deja terminee, ignoree")
            continue
        limits = SearchLimits(max_depth=12, max_nodes=2_000_000, max_time_s=3.0)
        start = time.perf_counter()
        result = iterative_deepening(game.clone_for_search(), limits)
        elapsed = time.perf_counter() - start
        rate = result.nodes / elapsed if elapsed > 0 else float("inf")
        print(
            f"{label}: profondeur={result.depth_reached} noeuds={result.nodes:,} "
            f"temps={elapsed:.3f}s -> {rate:,.0f} noeuds/s (timed_out={result.timed_out})"
        )


if __name__ == "__main__":
    main()
