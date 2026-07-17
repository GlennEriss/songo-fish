#!/usr/bin/env python3
"""Enregistrement retroactif, une seule fois, des deux modeles entraines
avant la mise en place du registre : overfit_10k (etape 6, validation
pipeline) et model_110k (etape 7, premier entrainement reel). A ne pas
relancer une fois fait (le registre serait ecrase avec des donnees
partielles, cf. notes ci-dessous sur ce qui est reconstruit de memoire vs
mesure precisement)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from songo_ai.model import ModelManifest, register_model

CHECKPOINTS = Path("data/checkpoints")


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def register_overfit_10k() -> None:
    # train_overfit() sauvegarde TOUJOURS la derniere epoque (pas la
    # meilleure sur le val, contrairement a train_model) : le manifeste doit
    # refleter l'epoque 200, pas l'epoque au meilleur val_loss observe en
    # cours de route (epoque ~40), qui ne correspond pas aux poids sauves.
    manifest = ModelManifest(
        version="0.0.1",
        created_at_unix=1784156700.0,  # 15 juillet 2026, run original (approx.)
        git_commit=_git_commit(),
        architecture={"width": 128, "num_blocks": 3, "dropout": 0.0, "feature_size": 33},
        dataset={
            "manifest_path": "data/dataset_v001_10k/manifest.json",
            "total_positions": 10000,
        },
        training={
            "function": "train_overfit",
            "epochs": 200,
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "checkpoint_policy": "derniere epoque (pas la meilleure sur le val)",
        },
        metrics={
            "epochs_run": 200,
            "best_epoch": 200,
            "train_loss": 1.0209,
            "train_policy_top1": 0.807,
            "val_loss": 3.5583,
            "val_policy_top1": 0.359,
        },
        notes=(
            "Surapprentissage volontaire (etape 6) : objectif = valider le pipeline "
            "de bout en bout, pas d'obtenir un bon joueur. Ecart train/val enorme "
            "(80.7% vs 35.9%) attendu et recherche. Ne pas utiliser en tournoi."
        ),
    )
    register_model(manifest, CHECKPOINTS / "overfit_10k.pt", promote=False)
    print(f"Enregistre {manifest.version} (overfit_10k, non-champion)")


def register_model_110k() -> None:
    manifest = ModelManifest(
        version="0.1.0",
        created_at_unix=1784240100.0,  # 16 juillet 2026, run original (approx.)
        git_commit=_git_commit(),
        architecture={"width": 128, "num_blocks": 3, "dropout": 0.1, "feature_size": 33},
        dataset={
            "manifest_path": "data/dataset_v003_110k/manifest.json",
            "total_positions": 110000,
        },
        training={
            "function": "train_model",
            "epochs_max": 60,
            "epochs_run": 54,
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.1,
            "early_stopping_patience": 10,
            "checkpoint_policy": "meilleure epoque sur le val (epoque 44)",
        },
        metrics={
            "epochs_run": 54,
            "best_epoch": 44,
            # train_loss/train_policy_top1 mesures a la derniere epoque (54), pas
            # exactement a l'epoque 44 : l'historique detaille n'a pas ete
            # sauvegarde sur disque lors de ce run, seul un resume a ete imprime.
            "train_loss": 1.6215,
            "train_policy_top1": 0.529,
            "val_loss": 1.7203,
            "val_policy_top1": 0.484,
        },
        notes=(
            "Premier entrainement reel (etape 7) sur le corpus combine 110k "
            "(10k standard + 100k profond GCP). Ecart train/val faible (~4.5 pts) "
            "-> vraie generalisation. Tournoi : 85% vs aleatoire (IC95 [76.7,90.7], "
            "100 parties dont seulement 2 parties distinctes -- agents deterministes, "
            "IC a interpreter avec prudence) ; 100% vs minimax profondeur 1 (idem, "
            "2 parties distinctes)."
        ),
    )
    register_model(manifest, CHECKPOINTS / "model_110k.pt", promote=True)
    print(f"Enregistre {manifest.version} (model_110k, promu champion)")


if __name__ == "__main__":
    register_overfit_10k()
    register_model_110k()
