#!/usr/bin/env python3
"""Benchmark obligatoire du moteur de regles (Annexe C : Regles).

Mesure, sur le moteur Python pur actuel (avant toute decision Numba) :
- transitions/seconde (clone + play, l'operation atomique d'une recherche) ;
- cout moyen d'un coup complet (sowing + capture) ;
- memoire approximative allouee par coup (tracemalloc).

Objectif : decider, a partir de chiffres reels, si le moteur doit etre
optimise avant l'etape 3 (Alpha-Beta), conformement a la section 4.3 du plan
("Profilage : optimiser uniquement les fonctions reellement dominantes
apres mesure").
"""

from __future__ import annotations

import random
import time
import tracemalloc

from songo_ai.songo.rules import SongoLegacyGame


def bench_transitions_per_second(duration_s: float = 2.0, seed: int = 0) -> tuple[int, float]:
    """Clone + play en boucle, comme le ferait un noeud de recherche."""
    rng = random.Random(seed)
    game = SongoLegacyGame()
    moves_done = 0
    start = time.perf_counter()
    deadline = start + duration_s
    while time.perf_counter() < deadline:
        if game.finished:
            game = SongoLegacyGame()
        legal = game.legal_moves()
        if not legal:
            game.normalize_terminal()
            if game.finished:
                continue
        move = rng.choice(legal)
        # Operation atomique d'un noeud de recherche : cloner puis jouer,
        # sans conserver l'historique (clone_for_search).
        child = game.clone_for_search()
        child.play(move)
        game = child
        moves_done += 1
    elapsed = time.perf_counter() - start
    return moves_done, elapsed


def bench_full_random_game_cost(num_games: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Cout moyen (ms) et nombre moyen de coups d'une partie aleatoire complete."""
    rng = random.Random(seed)
    start = time.perf_counter()
    total_moves = 0
    for _ in range(num_games):
        game = SongoLegacyGame()
        moves = 0
        while not game.finished and moves < 1000:
            legal = game.legal_moves()
            if not legal:
                game.normalize_terminal()
                break
            game.play(rng.choice(legal))
            moves += 1
        total_moves += moves
    elapsed = time.perf_counter() - start
    return (elapsed / num_games) * 1000.0, total_moves / num_games


def bench_memory_per_move(num_moves: int = 5000, seed: int = 0) -> float:
    # Les clones sont conserves en vie (kept) pour empecher le GC de les
    # recycler avant la mesure : sinon la memoire "courante" retombe a
    # zero entre deux coups et fausse totalement le resultat.
    rng = random.Random(seed)
    game = SongoLegacyGame()
    kept = []
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    done = 0
    while done < num_moves:
        if game.finished:
            game = SongoLegacyGame()
        legal = game.legal_moves()
        if not legal:
            game.normalize_terminal()
            continue
        child = game.clone_for_search()
        child.play(rng.choice(legal))
        kept.append(child)
        game = child
        done += 1
    after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (after - before) / done  # octets/coup


def main() -> None:
    moves_done, elapsed = bench_transitions_per_second()
    rate = moves_done / elapsed
    print(f"[transitions] {moves_done} coups en {elapsed:.3f}s -> {rate:,.0f} coups/s (clone+play)")

    avg_ms, avg_len = bench_full_random_game_cost()
    print(f"[partie complete] {avg_ms:.3f} ms/partie en moyenne ({avg_len:.1f} coups/partie)")

    bytes_per_move = bench_memory_per_move()
    print(f"[memoire] ~{bytes_per_move:,.0f} octets (pic cumule) par coup (clone_for_search + play)")

    print()
    print("Reperes annexe C : professeur profond = potentiellement plusieurs")
    print("millions de noeuds par position. A ce debit, 1 million de noeuds")
    print(f"prendrait environ {1_000_000 / rate:.1f} s en Python pur (a comparer")
    print("au budget cible du professeur avant de decider Numba/multiprocessing).")


if __name__ == "__main__":
    main()
