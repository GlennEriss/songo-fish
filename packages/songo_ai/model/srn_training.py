"""Premier entrainement controle du SRN sur le contrat ``D_RL``."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .srn_dataset import (
    DRLDataset,
    SRNBatchCollator,
    SRNTrainingBatch,
    split_examples_by_fixed_game_ids,
    split_examples_by_game_id,
)
from .srn_network import (
    SRNConfig,
    SongoRelationalNetwork,
    mask_policy_logits,
    masked_policy_cross_entropy,
)


SRN_TRAINING_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class SRNTrainingConfig:
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_policy: float = 1.0
    lambda_value: float = 1.0
    gradient_clip_norm: Optional[float] = 1.0
    validation_fraction: float = 0.2
    value_sign_epsilon: float = 0.1
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.lambda_policy < 0.0 or self.lambda_value < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive or None")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.value_sign_epsilon < 0.0:
            raise ValueError("value_sign_epsilon must be non-negative")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or None")
        if not math.isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be finite and non-negative")


@dataclass(frozen=True)
class SRNLossBreakdown:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    labeled_values: int


@dataclass(frozen=True)
class SRNEpochMetrics:
    total_loss: float
    policy_loss: float
    policy_cross_entropy: float
    policy_kl: float
    policy_top1_accuracy: float
    value_loss: float
    value_mae: float
    value_sign_accuracy: float
    examples: int
    labeled_values: int


@dataclass(frozen=True)
class SRNEpochRecord:
    epoch: int
    global_step: int
    train: SRNEpochMetrics
    validation: SRNEpochMetrics


@dataclass(frozen=True)
class SRNTrainingResult:
    model: SongoRelationalNetwork
    optimizer: torch.optim.Optimizer
    history: tuple[SRNEpochRecord, ...]
    train_game_ids: tuple[str, ...]
    validation_game_ids: tuple[str, ...]
    train_positions: int
    validation_positions: int
    last_checkpoint: Path
    best_validation_checkpoint: Path
    metrics_path: Path
    best_epoch: int
    global_step: int
    stopped_early: bool
    stop_epoch: int
    initialization: dict[str, Any]


@dataclass(frozen=True)
class LoadedSRNCheckpoint:
    model: SongoRelationalNetwork
    optimizer: torch.optim.Optimizer
    payload: dict[str, Any]


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_srn_loss(
    policy_logits: torch.Tensor,
    value_prediction: torch.Tensor,
    batch: SRNTrainingBatch,
    config: SRNTrainingConfig,
) -> SRNLossBreakdown:
    policy_loss = masked_policy_cross_entropy(
        policy_logits,
        batch.legal_mask,
        batch.policy_target,
    )
    value_prediction = value_prediction.reshape(-1)
    if value_prediction.shape != batch.value_target.shape:
        raise ValueError("value prediction and target shapes do not match")
    labeled_values = int(batch.value_mask.sum().item())
    if labeled_values:
        difference = value_prediction[batch.value_mask] - batch.value_target[batch.value_mask]
        value_loss = difference.square().mean()
    else:
        # Zero differentiable : aucune cible absente n'est transformee en nul.
        value_loss = value_prediction.sum() * 0.0
    total = config.lambda_policy * policy_loss + config.lambda_value * value_loss
    return SRNLossBreakdown(total, policy_loss, value_loss, labeled_values)


def _prediction_sign(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.where(
        values > epsilon,
        torch.ones_like(values),
        torch.where(values < -epsilon, -torch.ones_like(values), torch.zeros_like(values)),
    )


def evaluate_srn(
    model: SongoRelationalNetwork,
    loader: DataLoader,
    config: SRNTrainingConfig,
) -> SRNEpochMetrics:
    return _run_srn_epoch(model, loader, config, optimizer=None)[0]


def train_srn_epoch(
    model: SongoRelationalNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: SRNTrainingConfig,
) -> tuple[SRNEpochMetrics, int, float]:
    return _run_srn_epoch(model, loader, config, optimizer=optimizer)


def _run_srn_epoch(
    model: SongoRelationalNetwork,
    loader: DataLoader,
    config: SRNTrainingConfig,
    optimizer: Optional[torch.optim.Optimizer],
) -> tuple[SRNEpochMetrics, int, float]:
    is_train = optimizer is not None
    model.train(is_train)
    policy_loss_sum = 0.0
    policy_kl_sum = 0.0
    policy_correct = 0
    value_squared_sum = 0.0
    value_absolute_sum = 0.0
    value_sign_correct = 0
    total_examples = 0
    total_labeled = 0
    steps = 0
    last_gradient_norm = 0.0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for cpu_batch in loader:
            batch = cpu_batch.to(config.device)
            policy_logits, value_prediction = model(batch.graph)
            breakdown = compute_srn_loss(policy_logits, value_prediction, batch, config)
            if is_train:
                optimizer.zero_grad()
                breakdown.total.backward()
                if config.gradient_clip_norm is not None:
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )
                    last_gradient_norm = float(norm.item())
                optimizer.step()
                steps += 1

            batch_size = batch.batch_size
            total_examples += batch_size
            policy_loss_sum += float(breakdown.policy.item()) * batch_size
            masked_logits = mask_policy_logits(policy_logits, batch.legal_mask)
            log_policy = torch.log_softmax(masked_logits, dim=-1)
            target_log = torch.where(
                batch.policy_target > 0.0,
                torch.log(batch.policy_target.clamp_min(torch.finfo(batch.policy_target.dtype).tiny)),
                torch.zeros_like(batch.policy_target),
            )
            kl = (
                batch.policy_target * (target_log - log_policy)
            ).sum(dim=-1)
            policy_kl_sum += float(kl.sum().item())
            prediction = masked_logits.argmax(dim=-1)
            target = batch.policy_target.argmax(dim=-1)
            policy_correct += int((prediction == target).sum().item())

            if breakdown.labeled_values:
                selected_prediction = value_prediction.reshape(-1)[batch.value_mask]
                selected_target = batch.value_target[batch.value_mask]
                difference = selected_prediction - selected_target
                value_squared_sum += float(difference.square().sum().item())
                value_absolute_sum += float(difference.abs().sum().item())
                predicted_sign = _prediction_sign(
                    selected_prediction, config.value_sign_epsilon
                )
                target_sign = torch.sign(selected_target)
                value_sign_correct += int((predicted_sign == target_sign).sum().item())
                total_labeled += breakdown.labeled_values

    if total_examples == 0:
        raise ValueError("cannot evaluate an empty loader")
    policy_loss = policy_loss_sum / total_examples
    value_loss = value_squared_sum / total_labeled if total_labeled else 0.0
    metrics = SRNEpochMetrics(
        total_loss=config.lambda_policy * policy_loss + config.lambda_value * value_loss,
        policy_loss=policy_loss,
        policy_cross_entropy=policy_loss,
        policy_kl=policy_kl_sum / total_examples,
        policy_top1_accuracy=policy_correct / total_examples,
        value_loss=value_loss,
        value_mae=value_absolute_sum / total_labeled if total_labeled else 0.0,
        value_sign_accuracy=value_sign_correct / total_labeled if total_labeled else 0.0,
        examples=total_examples,
        labeled_values=total_labeled,
    )
    return metrics, steps, last_gradient_norm


def make_srn_loader(
    examples: Sequence,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        DRLDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=SRNBatchCollator(),
        num_workers=0,
    )


def _dataset_manifest(path: Path, dataset: DRLDataset) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "dataset_id": f"{path.name}:{digest[:12]}",
        "path": str(path),
        "sha256": digest,
        "examples": len(dataset),
        "games": len(dataset.game_ids),
        "dataset_family": "D_RL",
        "format_version": 1,
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_payload(
    *,
    model: SongoRelationalNetwork,
    optimizer: torch.optim.Optimizer,
    srn_config: SRNConfig,
    training_config: SRNTrainingConfig,
    epoch: int,
    global_step: int,
    dataset_manifest: dict[str, Any],
    train_game_ids: Sequence[str],
    validation_game_ids: Sequence[str],
    history: Sequence[SRNEpochRecord],
    best_epoch: int,
    best_validation_loss: float,
    initialization: dict[str, Any],
    stopped_early: bool,
    lineage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "checkpoint_type": "songo_srn_d_rl_training",
        "checkpoint_version": SRN_TRAINING_CONTRACT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "srn_config": asdict(srn_config),
        "training_config": asdict(training_config),
        "epoch": epoch,
        "global_step": global_step,
        "seed": training_config.seed,
        "dataset_manifest": dataset_manifest,
        "train_game_ids": list(train_game_ids),
        "validation_game_ids": list(validation_game_ids),
        "history": [asdict(record) for record in history],
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "initialization": initialization,
        "stopped_early": stopped_early,
        "lineage": dict(lineage or {}),
    }


def load_srn_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> LoadedSRNCheckpoint:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("checkpoint_type") != "songo_srn_d_rl_training":
        raise ValueError("not an SRN D_RL training checkpoint")
    if payload.get("checkpoint_version") != SRN_TRAINING_CONTRACT_VERSION:
        raise ValueError("unsupported SRN training checkpoint version")
    srn_config = SRNConfig(**payload["srn_config"])
    training_data = dict(payload["training_config"])
    training_data["device"] = device
    training_config = SRNTrainingConfig(**training_data)
    model = SongoRelationalNetwork(srn_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return LoadedSRNCheckpoint(model=model, optimizer=optimizer, payload=payload)


def train_srn_from_d_rl(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    srn_config: Optional[SRNConfig] = None,
    training_config: Optional[SRNTrainingConfig] = None,
    resume_checkpoint: Optional[str | Path] = None,
    initial_checkpoint: Optional[str | Path] = None,
    lineage: Optional[dict[str, Any]] = None,
    fixed_train_game_ids: Optional[Sequence[str]] = None,
    fixed_validation_game_ids: Optional[Sequence[str]] = None,
) -> SRNTrainingResult:
    """Entraine le SRN et conserve ``last`` et ``best_validation``.

    ``best_validation`` signifie uniquement la plus petite loss de validation,
    jamais le meilleur joueur.
    """

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = training_config or SRNTrainingConfig()
    srn_config = srn_config or SRNConfig()
    if resume_checkpoint is not None and initial_checkpoint is not None:
        raise ValueError("resume_checkpoint and initial_checkpoint are mutually exclusive")
    set_training_seed(training_config.seed)
    dataset = DRLDataset(dataset_path)
    if (fixed_train_game_ids is None) != (fixed_validation_game_ids is None):
        raise ValueError("both fixed train and validation ids must be provided together")
    split = (
        split_examples_by_game_id(
            dataset.examples,
            validation_fraction=training_config.validation_fraction,
            seed=training_config.seed,
        )
        if fixed_train_game_ids is None
        else split_examples_by_fixed_game_ids(
            dataset.examples,
            train_game_ids=fixed_train_game_ids,
            validation_game_ids=fixed_validation_game_ids,
        )
    )
    manifest = _dataset_manifest(dataset_path, dataset)
    last_path = output_dir / "last_checkpoint.pt"
    best_path = output_dir / "best_validation_checkpoint.pt"
    metrics_path = output_dir / "metrics.json"

    if resume_checkpoint is not None:
        loaded = load_srn_checkpoint(resume_checkpoint, device=training_config.device)
        if loaded.payload["srn_config"] != asdict(srn_config):
            raise ValueError("resume checkpoint SRNConfig does not match")
        if loaded.payload["train_game_ids"] != list(split.train_game_ids):
            raise ValueError("resume checkpoint train split does not match")
        if loaded.payload["validation_game_ids"] != list(split.validation_game_ids):
            raise ValueError("resume checkpoint validation split does not match")
        model = loaded.model
        optimizer = loaded.optimizer
        history = tuple(
            SRNEpochRecord(
                epoch=record["epoch"],
                global_step=record["global_step"],
                train=SRNEpochMetrics(**record["train"]),
                validation=SRNEpochMetrics(**record["validation"]),
            )
            for record in loaded.payload["history"]
        )
        start_epoch = loaded.payload["epoch"] + 1
        global_step = loaded.payload["global_step"]
        best_epoch = loaded.payload["best_epoch"]
        best_validation_loss = loaded.payload["best_validation_loss"]
        initialization = dict(
            loaded.payload.get("initialization", {"kind": "legacy_or_random"})
        )
        epochs_without_improvement = max(0, loaded.payload["epoch"] - best_epoch)
    else:
        if initial_checkpoint is None:
            model = SongoRelationalNetwork(srn_config).to(training_config.device)
            initialization = {"kind": "seeded_random", "seed": training_config.seed}
        else:
            initial_path = Path(initial_checkpoint)
            loaded_initial = load_srn_checkpoint(
                initial_path, device=training_config.device
            )
            if loaded_initial.payload["srn_config"] != asdict(srn_config):
                raise ValueError("initial checkpoint SRNConfig does not match")
            model = loaded_initial.model
            initialization = {
                "kind": "checkpoint_weights_fresh_optimizer",
                "path": str(initial_path),
                "sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
                "source_epoch": loaded_initial.payload["epoch"],
            }
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        train_eval_loader = make_srn_loader(
            split.train_examples,
            batch_size=training_config.batch_size,
            shuffle=False,
            seed=training_config.seed,
        )
        validation_loader = make_srn_loader(
            split.validation_examples,
            batch_size=training_config.batch_size,
            shuffle=False,
            seed=training_config.seed,
        )
        baseline_train = evaluate_srn(model, train_eval_loader, training_config)
        baseline_validation = evaluate_srn(model, validation_loader, training_config)
        history = (
            SRNEpochRecord(
                epoch=0,
                global_step=0,
                train=baseline_train,
                validation=baseline_validation,
            ),
        )
        start_epoch = 1
        global_step = 0
        best_epoch = 0
        best_validation_loss = baseline_validation.total_loss
        epochs_without_improvement = 0
        baseline_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            srn_config=srn_config,
            training_config=training_config,
            epoch=0,
            global_step=0,
            dataset_manifest=manifest,
            train_game_ids=split.train_game_ids,
            validation_game_ids=split.validation_game_ids,
            history=history,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            initialization=initialization,
            stopped_early=False,
            lineage=lineage,
        )
        _atomic_torch_save(baseline_payload, best_path)

    stopped_early = False
    for epoch in range(start_epoch, training_config.epochs + 1):
        train_loader = make_srn_loader(
            split.train_examples,
            batch_size=training_config.batch_size,
            shuffle=True,
            seed=training_config.seed + epoch,
        )
        _, steps, _ = train_srn_epoch(model, train_loader, optimizer, training_config)
        global_step += steps
        train_metrics = evaluate_srn(
            model,
            make_srn_loader(
                split.train_examples,
                batch_size=training_config.batch_size,
                shuffle=False,
                seed=training_config.seed,
            ),
            training_config,
        )
        validation_metrics = evaluate_srn(
            model,
            make_srn_loader(
                split.validation_examples,
                batch_size=training_config.batch_size,
                shuffle=False,
                seed=training_config.seed,
            ),
            training_config,
        )
        history = history + (
            SRNEpochRecord(epoch, global_step, train_metrics, validation_metrics),
        )
        improved = (
            validation_metrics.total_loss
            < best_validation_loss - training_config.early_stopping_min_delta
        )
        if improved:
            best_validation_loss = validation_metrics.total_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        should_stop = (
            training_config.early_stopping_patience is not None
            and epochs_without_improvement >= training_config.early_stopping_patience
        )
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            srn_config=srn_config,
            training_config=training_config,
            epoch=epoch,
            global_step=global_step,
            dataset_manifest=manifest,
            train_game_ids=split.train_game_ids,
            validation_game_ids=split.validation_game_ids,
            history=history,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            initialization=initialization,
            stopped_early=should_stop,
            lineage=lineage,
        )
        _atomic_torch_save(payload, last_path)
        if improved:
            _atomic_torch_save(payload, best_path)
        metrics_path.write_text(
            json.dumps([asdict(record) for record in history], indent=2),
            encoding="utf-8",
        )
        if should_stop:
            stopped_early = True
            break

    if not last_path.exists():
        # Reprise deja arrivee a l'epoque cible : materialise tout de meme last.
        _atomic_torch_save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                srn_config=srn_config,
                training_config=training_config,
                epoch=history[-1].epoch,
                global_step=global_step,
                dataset_manifest=manifest,
                train_game_ids=split.train_game_ids,
                validation_game_ids=split.validation_game_ids,
                history=history,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                initialization=initialization,
                stopped_early=stopped_early,
                lineage=lineage,
            ),
            last_path,
        )
    if not metrics_path.exists():
        metrics_path.write_text(
            json.dumps([asdict(record) for record in history], indent=2),
            encoding="utf-8",
        )
    return SRNTrainingResult(
        model=model,
        optimizer=optimizer,
        history=history,
        train_game_ids=split.train_game_ids,
        validation_game_ids=split.validation_game_ids,
        train_positions=len(split.train_examples),
        validation_positions=len(split.validation_examples),
        last_checkpoint=last_path,
        best_validation_checkpoint=best_path,
        metrics_path=metrics_path,
        best_epoch=best_epoch,
        global_step=global_step,
        stopped_early=stopped_early,
        stop_epoch=history[-1].epoch,
        initialization=initialization,
    )
