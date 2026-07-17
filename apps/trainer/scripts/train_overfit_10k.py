#!/usr/bin/env python3
"""Surapprentissage volontaire sur le corpus de 10k (etape 6, section 7.4) :
le but n'est pas d'obtenir un bon joueur, mais de prouver que le format de
donnees et la boucle d'entrainement fonctionnent de bout en bout (policy
top-1 sur le train doit approcher 100%)."""

from __future__ import annotations

from pathlib import Path

from songo_ai.model import train_overfit

DATA_DIR = Path("data/dataset_v001_10k")
CHECKPOINT = Path("data/checkpoints/overfit_10k.pt")


def main() -> None:
    history = train_overfit(
        train_shard=DATA_DIR / "train.jsonl",
        val_shard=DATA_DIR / "val.jsonl",
        epochs=200,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-5,
        checkpoint_path=CHECKPOINT,
    )

    for m in history:
        if m.epoch == 1 or m.epoch % 20 == 0 or m.epoch == len(history):
            val_part = f" val_loss={m.val_loss:.4f} val_top1={m.val_policy_top1:.3f}" if m.val_loss is not None else ""
            print(f"epoch {m.epoch:3d} train_loss={m.train_loss:.4f} train_top1={m.train_policy_top1:.3f}{val_part}")

    first, last = history[0], history[-1]
    print()
    print(f"Resume : top1 train {first.train_policy_top1:.3f} -> {last.train_policy_top1:.3f}")
    print(f"         loss  train {first.train_loss:.4f} -> {last.train_loss:.4f}")
    if last.val_policy_top1 is not None:
        print(f"         top1  val   {history[0].val_policy_top1:.3f} -> {last.val_policy_top1:.3f} (generalisation, pas l'objectif ici)")
    print(f"Checkpoint ecrit : {CHECKPOINT}")


if __name__ == "__main__":
    main()
