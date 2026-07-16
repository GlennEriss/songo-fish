#!/usr/bin/env python3
"""Entrainement reel (train/val, dropout, early stopping) sur le corpus
combine 110k, puis evaluation en tournoi contre un minimax tres faible
(profondeur 1, peu de noeuds) -- section 10.2 : "Aleatoire / Minimax
niveaux varies", ici le niveau le plus bas comme premiere reference."""

from __future__ import annotations

import time
from pathlib import Path

from songo_ai.evaluation import play_match
from songo_ai.generation import make_shallow_search_agent, random_agent
from songo_ai.model import load_model, make_network_agent, train_model

DATA_DIR = Path("data/dataset_v003_110k")
CHECKPOINT = Path("data/checkpoints/model_110k.pt")


def main() -> None:
    print("=== Entrainement sur le corpus 110k (train/val reels) ===")
    start = time.perf_counter()
    history = train_model(
        train_shard=DATA_DIR / "train.jsonl",
        val_shard=DATA_DIR / "val.jsonl",
        epochs=60,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.1,
        checkpoint_path=CHECKPOINT,
        early_stopping_patience=10,
    )
    elapsed = time.perf_counter() - start

    best = min(history, key=lambda m: m.val_loss)
    print(f"Entrainement termine en {elapsed/60:.1f} min, {len(history)} epoques (early stopping eventuel)")
    print(f"Meilleure epoque: {best.epoch} val_loss={best.val_loss:.4f} val_top1={best.val_policy_top1:.3f}")
    print(f"Derniere epoque: train_loss={history[-1].train_loss:.4f} train_top1={history[-1].train_policy_top1:.3f} "
          f"val_loss={history[-1].val_loss:.4f} val_top1={history[-1].val_policy_top1:.3f}")

    print()
    print("=== Tournoi : reseau (inference seule) vs agents de reference ===")
    model = load_model(CHECKPOINT, dropout=0.1)
    network_agent = make_network_agent(model)

    print("--- vs aleatoire (100 parties) ---")
    result_random = play_match(network_agent, random_agent, num_games=100, seed=42)
    print(f"score reseau: {result_random.score_a}/{result_random.games} "
          f"({result_random.win_rate_a*100:.1f}%, IC95 [{result_random.ci_low*100:.1f}%, {result_random.ci_high*100:.1f}%])")
    print(f"victoires={result_random.wins_a} defaites={result_random.wins_b} nulles={result_random.draws}")

    print("--- vs minimax tres faible, profondeur 1 (100 parties) ---")
    weak_minimax = make_shallow_search_agent(max_depth=1, max_nodes=5_000, max_time_s=0.5)
    result_minimax = play_match(network_agent, weak_minimax, num_games=100, seed=43)
    print(f"score reseau: {result_minimax.score_a}/{result_minimax.games} "
          f"({result_minimax.win_rate_a*100:.1f}%, IC95 [{result_minimax.ci_low*100:.1f}%, {result_minimax.ci_high*100:.1f}%])")
    print(f"victoires={result_minimax.wins_a} defaites={result_minimax.wins_b} nulles={result_minimax.draws}")


if __name__ == "__main__":
    main()
