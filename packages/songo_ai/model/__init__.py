"""Reseau multi-tetes, pertes, entrainement, inference (etape 6-7 du plan directeur)."""

from .dataset import ObservationDataset
from .features import FEATURE_SIZE, observation_features
from .losses import LossBreakdown, LossWeights, compute_loss, masked_q_loss, soft_cross_entropy
from .network import SongoNet
from .inference import load_model, make_network_agent
from .registry import (
    ModelManifest,
    build_manifest,
    get_champion,
    list_versions,
    load_registry,
    promote_version,
    register_model,
    save_registry,
    train_and_register,
)
from .train import EpochMetrics, train_model, train_overfit

__all__ = [
    "ObservationDataset",
    "FEATURE_SIZE",
    "observation_features",
    "LossBreakdown",
    "LossWeights",
    "compute_loss",
    "masked_q_loss",
    "soft_cross_entropy",
    "SongoNet",
    "EpochMetrics",
    "train_model",
    "train_overfit",
    "load_model",
    "make_network_agent",
    "ModelManifest",
    "build_manifest",
    "get_champion",
    "list_versions",
    "load_registry",
    "promote_version",
    "register_model",
    "save_registry",
    "train_and_register",
]
