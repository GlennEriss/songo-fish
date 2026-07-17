"""Boucle d'entrainement (section 7 + curriculum section 7.4 : premier
palier "10k positions" -> valider le format et surapprendre volontairement
un petit corpus). L'objectif ici n'est PAS d'obtenir un bon joueur : c'est
de prouver que le format de donnees et la boucle d'entrainement
fonctionnent de bout en bout, en verifiant que le reseau peut memoriser
(top-1 policy proche de 100%) le corpus d'entrainement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from .dataset import ObservationDataset
from .losses import LossWeights, compute_loss
from .network import SongoNet


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_policy_top1: float
    val_loss: Optional[float] = None
    val_policy_top1: Optional[float] = None


def _policy_top1_accuracy(policy_logits: torch.Tensor, policy_target: torch.Tensor, legal_mask: torch.Tensor) -> float:
    masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
    predicted = masked_logits.argmax(dim=-1)
    target_best = policy_target.argmax(dim=-1)
    return (predicted == target_best).float().mean().item()


def _run_epoch(model: SongoNet, loader: DataLoader, optimizer, weights: LossWeights, device: str) -> tuple:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            policy_logits, wdl_logits, q_pred = model(batch["features"])
            breakdown = compute_loss(policy_logits, wdl_logits, q_pred, batch, weights)

            if is_train:
                optimizer.zero_grad()
                breakdown.total.backward()
                optimizer.step()

            batch_size = batch["features"].shape[0]
            total_loss += breakdown.total.item() * batch_size
            total_correct += _policy_top1_accuracy(policy_logits, batch["policy_target"], batch["legal_mask"]) * batch_size
            total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def train_overfit(
    train_shard: Path,
    val_shard: Optional[Path] = None,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    checkpoint_path: Optional[Path] = None,
) -> List[EpochMetrics]:
    train_dataset = ObservationDataset(train_shard)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_shard is not None:
        val_loader = DataLoader(ObservationDataset(val_shard), batch_size=batch_size, shuffle=False)

    model = SongoNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = LossWeights()

    history: List[EpochMetrics] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_top1 = _run_epoch(model, train_loader, optimizer, weights, device)

        val_loss = val_top1 = None
        if val_loader is not None:
            val_loss, val_top1 = _run_epoch(model, val_loader, None, weights, device)

        history.append(EpochMetrics(epoch, train_loss, train_top1, val_loss, val_top1))

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    return history


def train_model(
    train_shard: Path,
    val_shard: Path,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    width: int = 128,
    num_blocks: int = 3,
    device: str = "cpu",
    checkpoint_path: Optional[Path] = None,
    early_stopping_patience: Optional[int] = 10,
) -> List[EpochMetrics]:
    """Entrainement "reel" (etape 7, curriculum 100k : "verifier la
    generalisation") : dropout + weight decay plus fermes que
    `train_overfit`, et le checkpoint sauvegarde est celui de la MEILLEURE
    epoque sur le val (pas la derniere), avec arret anticipe si le val ne
    s'ameliore plus pendant `early_stopping_patience` epoques consecutives."""

    train_dataset = ObservationDataset(train_shard)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ObservationDataset(val_shard), batch_size=batch_size, shuffle=False)

    model = SongoNet(width=width, num_blocks=num_blocks, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = LossWeights()

    history: List[EpochMetrics] = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_top1 = _run_epoch(model, train_loader, optimizer, weights, device)
        val_loss, val_top1 = _run_epoch(model, val_loader, None, weights, device)
        history.append(EpochMetrics(epoch, train_loss, train_top1, val_loss, val_top1))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is not None and checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)

    return history
