#!/usr/bin/env python3
"""Micro-overfit puis premier entrainement SRN controle sur le pilote D_RL."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from songo_ai.model import (
    DRLDataset,
    SRNBatchCollator,
    SRNConfig,
    SRNTrainingConfig,
    SongoRelationalNetwork,
    evaluate_srn,
    make_srn_loader,
    policy_probabilities,
    set_training_seed,
    split_examples_by_game_id,
    train_srn_epoch,
    train_srn_from_d_rl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/d_rl/pilot_lot5_seed_20260924.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/lot6_srn_seed_20260924"),
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--micro-examples", type=int, default=16)
    parser.add_argument("--micro-epochs", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def metrics_dict(metrics) -> dict:
    return asdict(metrics)


def predict_examples(model, examples, device="cpu") -> list[dict]:
    batch = SRNBatchCollator()(examples).to(device)
    model.eval()
    with torch.no_grad():
        logits, values = model(batch.graph)
        probabilities = policy_probabilities(logits, batch.legal_mask)
    records = []
    for index, example in enumerate(examples):
        records.append(
            {
                "game_id": example.metadata["game_id"],
                "ply": example.metadata["ply"],
                "policy_target": list(example.policy_target),
                "policy_prediction": probabilities[index].cpu().tolist(),
                "value_target": example.value_target,
                "value_prediction": float(values[index].cpu().item()),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = DRLDataset(args.dataset)
    srn_config = SRNConfig(hidden_dim=32, num_relational_blocks=2)

    # Micro-overfit : positions uniques d'une seule partie pour eviter des
    # etats initiaux identiques associes a des cibles MCTS contradictoires.
    first_game = dataset.game_ids[0]
    game_examples = [
        example for example in dataset.examples if example.metadata["game_id"] == first_game
    ]
    if len(game_examples) < args.micro_examples:
        raise RuntimeError("not enough positions in the selected game for micro-overfit")
    stride = max(1, len(game_examples) // args.micro_examples)
    micro_examples = tuple(game_examples[::stride][: args.micro_examples])
    micro_config = SRNTrainingConfig(
        epochs=args.micro_epochs,
        batch_size=args.micro_examples,
        learning_rate=1e-2,
        weight_decay=0.0,
        gradient_clip_norm=5.0,
        validation_fraction=0.2,
        seed=args.seed,
    )
    set_training_seed(args.seed)
    micro_model = SongoRelationalNetwork(srn_config)
    micro_loader = make_srn_loader(
        micro_examples,
        batch_size=args.micro_examples,
        shuffle=False,
        seed=args.seed,
    )
    initial_micro = evaluate_srn(micro_model, micro_loader, micro_config)
    optimizer = torch.optim.AdamW(
        micro_model.parameters(),
        lr=micro_config.learning_rate,
        weight_decay=micro_config.weight_decay,
    )
    micro_started = time.perf_counter()
    for _ in range(args.micro_epochs):
        train_srn_epoch(micro_model, micro_loader, optimizer, micro_config)
    micro_elapsed = time.perf_counter() - micro_started
    final_micro = evaluate_srn(micro_model, micro_loader, micro_config)
    micro_success = (
        final_micro.total_loss < initial_micro.total_loss * 0.35
        and final_micro.policy_loss < initial_micro.policy_loss * 0.8
        and final_micro.value_mae < initial_micro.value_mae * 0.25
    )
    micro_report = {
        "examples": len(micro_examples),
        "game_id": first_game,
        "epochs": args.micro_epochs,
        "steps": args.micro_epochs,
        "learning_rate": micro_config.learning_rate,
        "elapsed_s": micro_elapsed,
        "initial": metrics_dict(initial_micro),
        "final": metrics_dict(final_micro),
        "success": micro_success,
    }
    (args.output / "micro_overfit.json").write_text(
        json.dumps(micro_report, indent=2), encoding="utf-8"
    )
    if not micro_success:
        raise RuntimeError("micro-overfit failed; full pilot training was not started")

    full_config = SRNTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=3e-3,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        validation_fraction=0.2,
        seed=args.seed,
    )
    split = split_examples_by_game_id(
        dataset.examples,
        validation_fraction=full_config.validation_fraction,
        seed=full_config.seed,
    )
    sanity_examples = split.validation_examples[:3]
    set_training_seed(args.seed)
    initial_model = SongoRelationalNetwork(srn_config)
    sanity_before = predict_examples(initial_model, sanity_examples)

    full_started = time.perf_counter()
    training = train_srn_from_d_rl(
        args.dataset,
        args.output / "checkpoints",
        srn_config=srn_config,
        training_config=full_config,
    )
    full_elapsed = time.perf_counter() - full_started
    sanity_after = predict_examples(training.model, sanity_examples)
    best_record = min(training.history, key=lambda record: record.validation.total_loss)
    full_report = {
        "configuration": asdict(full_config),
        "srn_config": asdict(srn_config),
        "train_games": len(training.train_game_ids),
        "validation_games": len(training.validation_game_ids),
        "train_game_ids": list(training.train_game_ids),
        "validation_game_ids": list(training.validation_game_ids),
        "train_positions": training.train_positions,
        "validation_positions": training.validation_positions,
        "elapsed_s": full_elapsed,
        "epoch_0": asdict(training.history[0]),
        "best_validation": asdict(best_record),
        "last_epoch": asdict(training.history[-1]),
        "best_epoch": training.best_epoch,
        "last_checkpoint": str(training.last_checkpoint),
        "best_validation_checkpoint": str(training.best_validation_checkpoint),
        "metrics_path": str(training.metrics_path),
        "sanity_check": {
            "before": sanity_before,
            "after": sanity_after,
        },
    }
    report_path = args.output / "training_report.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(json.dumps({"micro_overfit": micro_report, "training": full_report}, indent=2))


if __name__ == "__main__":
    main()
