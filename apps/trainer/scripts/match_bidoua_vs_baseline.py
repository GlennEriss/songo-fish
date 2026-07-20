#!/usr/bin/env python3
"""Tournoi : SongoFish (reseau + recherche) AVEC le bonus territoire
(bidoua/Yinda, cf. docs/trainer/README.md) contre la meme configuration
SANS ce bonus -- meme reseau, meme recherche, seule la formule
d'evaluation de feuille differe. Isole l'effet du bonus a l'echelle
"modele + recherche", contrairement au tournoi initial (recherche pure
sans reseau, cf. commit precedent) qui isolait l'effet a l'echelle
heuristique seule.

Usage :
    .venv/bin/python apps/trainer/scripts/match_bidoua_vs_baseline.py \\
        --games 20 --depth 14 --time-s 5.0

Attention au budget : chaque coup joue jusqu'a `--time-s` secondes (les
DEUX agents), donc une partie de ~30-40 coups a 5s/coup peut prendre
plusieurs minutes -- une vingtaine de parties a 5s/14 est un ordre de
grandeur de 1h+. Commencer petit (--games 6) pour un premier signal avant
de lancer un lot complet en tache de fond.
"""

from __future__ import annotations

import argparse
import random
import time

from songo_ai.evaluation import play_match
from songo_ai.hybrid.network_eval import EVAL_SCALE, NetworkCache, make_network_evaluate, make_network_priority
from songo_ai.model import get_champion, load_model, load_registry
from songo_ai.search.negamax import SearchLimits, default_evaluate, iterative_deepening
from songo_ai.songo.fast_rules import FastSongoGame, warmup
from songo_ai.songo.rules import SongoLegacyGame


def _load(version: str):
    entry = get_champion() if version == "champion" else load_registry()["versions"].get(version)
    if entry is None:
        raise ValueError(f"version inconnue: {version} (registre: data/checkpoints/registry.json)")
    return load_model(
        entry["checkpoint_path"],
        dropout=entry["architecture"]["dropout"],
        width=entry["architecture"].get("width", 128),
        num_blocks=entry["architecture"].get("num_blocks", 3),
    )


def _make_baseline_evaluate(cache: NetworkCache):
    """Reseau seul, sans bonus territoire -- comportement de
    make_network_evaluate() AVANT le correctif bidoua (juillet 2026)."""

    def evaluate(game, perspective: int) -> float:
        if game.finished:
            return default_evaluate(game, perspective)
        _, lean = cache.outputs(game)
        return lean * EVAL_SCALE

    return evaluate


def _make_agent(model, evaluate_fn, max_depth: int, max_time_s: float, priority_fn):
    limits = SearchLimits(max_depth=max_depth, max_nodes=2_000_000, max_time_s=max_time_s, quiescence_depth=4)

    def agent(game: SongoLegacyGame, rng: random.Random) -> int:
        fast_game = FastSongoGame.from_board(game.board, game.turn)
        result = iterative_deepening(fast_game.clone_for_search(), limits, evaluate_fn, priority_fn)
        return result.local_action

    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--time-s", type=float, default=5.0)
    parser.add_argument("--version", default="champion")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--opening-plies", type=int, default=4)
    args = parser.parse_args()

    warmup()
    model = _load(args.version)

    cache_bidoua = NetworkCache(model)
    agent_bidoua = _make_agent(
        model,
        make_network_evaluate(model, cache_bidoua),
        args.depth,
        args.time_s,
        make_network_priority(model, cache_bidoua),
    )

    cache_baseline = NetworkCache(model)
    agent_baseline = _make_agent(
        model,
        _make_baseline_evaluate(cache_baseline),
        args.depth,
        args.time_s,
        make_network_priority(model, cache_baseline),
    )

    print(f"model={args.version} depth<={args.depth} time_s={args.time_s} games={args.games}")
    start = time.perf_counter()
    result = play_match(
        agent_bidoua, agent_baseline, num_games=args.games, seed=args.seed, opening_random_plies=args.opening_plies
    )
    elapsed = time.perf_counter() - start

    print(f"parties: {result.games} (dont {result.distinct_games} distinctes)")
    print(f"AVEC bidoua    : {result.wins_a} victoires")
    print(f"SANS (baseline): {result.wins_b} victoires")
    print(f"nulles: {result.draws}")
    print(
        f"score AVEC bidoua: {result.win_rate_a:.1%}  "
        f"[IC95%: {result.ci_low:.1%} - {result.ci_high:.1%}]"
    )
    print(f"temps total: {elapsed:.1f}s ({elapsed/result.games:.1f}s/partie)")


if __name__ == "__main__":
    main()
