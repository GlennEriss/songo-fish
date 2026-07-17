#!/usr/bin/env python3
"""Compare le moteur de reference (Python pur) et le moteur Numba
(FastSongoGame) : debit brut (clone+play) et noeuds/s en alpha-beta reel.
"""

from __future__ import annotations

import random
import time

from songo_ai.search.negamax import SearchLimits, iterative_deepening
from songo_ai.songo.fast_rules import FastSongoGame, warmup
from songo_ai.songo.rules import SongoLegacyGame


def bench_transitions_per_second(game_factory, duration_s: float = 2.0, seed: int = 0):
    rng = random.Random(seed)
    game = game_factory()
    moves_done = 0
    start = time.perf_counter()
    deadline = start + duration_s
    while time.perf_counter() < deadline:
        if game.finished:
            game = game_factory()
        legal = game.legal_moves()
        if not legal:
            game.normalize_terminal()
            if game.finished:
                continue
        move = rng.choice(legal)
        child = game.clone_for_search()
        child.play(move)
        game = child
        moves_done += 1
    elapsed = time.perf_counter() - start
    return moves_done, elapsed


def bench_alpha_beta(game_factory, max_time_s: float = 3.0):
    limits = SearchLimits(max_depth=12, max_nodes=5_000_000, max_time_s=max_time_s)
    game = game_factory()
    start = time.perf_counter()
    result = iterative_deepening(game.clone_for_search(), limits)
    elapsed = time.perf_counter() - start
    return result, elapsed


def main() -> None:
    print("Chauffe JIT Numba (compilation, ne pas chronometrer)...")
    warmup()

    print("\n=== Debit brut (clone + play, 2s) ===")
    for label, factory in [("reference (Python pur)", SongoLegacyGame), ("fast (Numba)", FastSongoGame.initial)]:
        moves, elapsed = bench_transitions_per_second(factory)
        print(f"{label:28s}: {moves:>8,} coups en {elapsed:.3f}s -> {moves / elapsed:>10,.0f} coups/s")

    print("\n=== Alpha-beta reel (position initiale, budget 3s) ===")
    for label, factory in [("reference (Python pur)", SongoLegacyGame), ("fast (Numba)", FastSongoGame.initial)]:
        result, elapsed = bench_alpha_beta(factory)
        rate = result.nodes / elapsed if elapsed > 0 else float("inf")
        print(
            f"{label:28s}: profondeur={result.depth_reached:>2} noeuds={result.nodes:>8,} "
            f"-> {rate:>10,.0f} noeuds/s"
        )


if __name__ == "__main__":
    main()
