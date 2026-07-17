"""Registre de versions de modeles (section 10.2/10.3 : champion/challenger).

Chaque version correspond a un entrainement COMPLET FROM SCRATCH sur un
dataset donne -- jamais un fine-tuning des poids d'une version precedente.
Raisons (voir apps/trainer/README.md) : le plan directeur organise deja
l'amelioration du modele autour d'un cycle champion/challenger avec porte
de promotion statistique (section 10.2), pas d'une evolution continue des
memes poids ; et le reseau est assez petit (quelques centaines de milliers
de parametres) pour que reentrainer from scratch coute des minutes, pas des
heures -- l'argument "economiser du calcul en repartant de l'existant" ne
s'applique pas a cette echelle.

Le registre trace la lignee complete de chaque version (dataset utilise,
hyperparametres, metriques, commit git) pour que deux versions soient
comparables et qu'une promotion soit une decision explicite, jamais
automatique a la fin d'un entrainement.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .features import FEATURE_SIZE
from .train import EpochMetrics, train_model

DEFAULT_REGISTRY_PATH = Path("data/checkpoints/registry.json")
DEFAULT_CHECKPOINT_DIR = Path("data/checkpoints")


@dataclass
class ModelManifest:
    version: str
    created_at_unix: float
    git_commit: Optional[str]
    architecture: Dict
    dataset: Dict
    training: Dict
    metrics: Dict
    notes: str = ""


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def build_manifest(
    version: str,
    architecture: Dict,
    dataset_manifest_path: Path,
    training_config: Dict,
    history: List[EpochMetrics],
    notes: str = "",
) -> ModelManifest:
    dataset_manifest_path = Path(dataset_manifest_path)
    dataset_manifest = json.loads(dataset_manifest_path.read_text()) if dataset_manifest_path.exists() else {}

    scored = [m for m in history if m.val_loss is not None]
    best = min(scored, key=lambda m: m.val_loss) if scored else history[-1]

    return ModelManifest(
        version=version,
        created_at_unix=time.time(),
        git_commit=_git_commit(),
        architecture=architecture,
        dataset={
            "manifest_path": str(dataset_manifest_path),
            "total_positions": dataset_manifest.get("total_positions"),
            "checksums_sha256": dataset_manifest.get("checksums_sha256"),
        },
        training=training_config,
        metrics={
            "epochs_run": len(history),
            "best_epoch": best.epoch,
            "train_loss": best.train_loss,
            "train_policy_top1": best.train_policy_top1,
            "val_loss": best.val_loss,
            "val_policy_top1": best.val_policy_top1,
        },
        notes=notes,
    )


def load_registry(registry_path: Path = DEFAULT_REGISTRY_PATH) -> Dict:
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return {"champion": None, "versions": {}}
    return json.loads(registry_path.read_text())


def save_registry(registry: Dict, registry_path: Path = DEFAULT_REGISTRY_PATH) -> None:
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2))


def register_model(
    manifest: ModelManifest,
    checkpoint_path: Path,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    promote: bool = False,
) -> Dict:
    """Ajoute une version au registre. `promote=True` en fait explicitement
    le champion courant (a decider apres revue des metriques/tournoi, jamais
    automatiquement) ; la toute premiere version enregistree devient
    champion par defaut (rien d'autre a comparer)."""
    registry = load_registry(registry_path)
    registry["versions"][manifest.version] = {
        **asdict(manifest),
        "checkpoint_path": str(checkpoint_path),
    }
    if promote or registry["champion"] is None:
        registry["champion"] = manifest.version
    save_registry(registry, registry_path)
    return registry


def promote_version(version: str, registry_path: Path = DEFAULT_REGISTRY_PATH) -> Dict:
    registry = load_registry(registry_path)
    if version not in registry["versions"]:
        raise ValueError(f"version inconnue du registre: {version}")
    registry["champion"] = version
    save_registry(registry, registry_path)
    return registry


def get_champion(registry_path: Path = DEFAULT_REGISTRY_PATH) -> Optional[Dict]:
    registry = load_registry(registry_path)
    champion_version = registry.get("champion")
    if champion_version is None:
        return None
    return registry["versions"].get(champion_version)


def list_versions(registry_path: Path = DEFAULT_REGISTRY_PATH) -> List[str]:
    registry = load_registry(registry_path)
    return sorted(registry["versions"].keys())


def train_and_register(
    version: str,
    train_shard: Path,
    val_shard: Path,
    dataset_manifest_path: Path,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    promote: bool = False,
    notes: str = "",
    width: int = 128,
    num_blocks: int = 3,
    dropout: float = 0.1,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: Optional[int] = 10,
    device: str = "cpu",
) -> ModelManifest:
    """Entrainement from scratch + enregistrement dans le registre, en un
    seul appel : le point d'entree recommande pour tout entrainement a
    partir de maintenant (au lieu d'appeler train_model puis a construire
    le manifeste a la main, comme pour les deux premieres versions)."""
    checkpoint_path = Path(checkpoint_dir) / f"model_v{version}.pt"

    history = train_model(
        train_shard=train_shard,
        val_shard=val_shard,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        width=width,
        num_blocks=num_blocks,
        device=device,
        checkpoint_path=checkpoint_path,
        early_stopping_patience=early_stopping_patience,
    )

    manifest = build_manifest(
        version=version,
        architecture={
            "width": width,
            "num_blocks": num_blocks,
            "dropout": dropout,
            "feature_size": FEATURE_SIZE,
        },
        dataset_manifest_path=dataset_manifest_path,
        training_config={
            "function": "train_model",
            "epochs_max": epochs,
            "epochs_run": len(history),
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "early_stopping_patience": early_stopping_patience,
            "checkpoint_policy": "meilleure epoque sur le val",
        },
        history=history,
        notes=notes,
    )
    register_model(manifest, checkpoint_path, registry_path=registry_path, promote=promote)
    return manifest
