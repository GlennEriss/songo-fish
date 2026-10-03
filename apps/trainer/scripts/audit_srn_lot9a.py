#!/usr/bin/env python3
"""Lot 9A : audit de symetrie P1/P2 et d'information du pilote D_RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from songo_ai.evaluation import (
    audit_d_rl_information,
    evaluate_model_symmetry,
    model_parameter_fingerprint,
    select_extended_d_lab_benchmark,
    validate_mirrored_legality,
)
from songo_ai.model import (
    DRLDataset,
    SRNConfig,
    SongoRelationalNetwork,
    load_srn_checkpoint,
    set_training_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_dir = Path("data/experiments/lot6_srn_seed_20260924/checkpoints")
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=checkpoint_dir / "best_validation_checkpoint.pt",
    )
    parser.add_argument(
        "--last-checkpoint", type=Path, default=checkpoint_dir / "last_checkpoint.pt"
    )
    parser.add_argument(
        "--d-lab", type=Path, default=Path("data/dataset_v001_10k/test.jsonl")
    )
    parser.add_argument(
        "--d-rl", type=Path, default=Path("data/d_rl/pilot_lot5_seed_20260924.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/lot9a_symmetry_audit_seed_20260924"),
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--lab-positions", type=int, default=350)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_models(args: argparse.Namespace):
    best = load_srn_checkpoint(args.best_checkpoint, device="cpu")
    last = load_srn_checkpoint(args.last_checkpoint, device="cpu")
    if best.payload["srn_config"] != last.payload["srn_config"]:
        raise RuntimeError("G1 checkpoints do not share the same SRNConfig")
    if best.payload["seed"] != last.payload["seed"] or best.payload["seed"] != args.seed:
        raise RuntimeError("checkpoint/G0 seeds do not match")
    config = SRNConfig(**best.payload["srn_config"])
    set_training_seed(args.seed)
    g0 = SongoRelationalNetwork(config)
    return config, {"G0": g0, "G1-best": best.model, "G1-last": last.model}, best, last


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    config, models, best, last = load_models(args)
    before = {name: model_parameter_fingerprint(model) for name, model in models.items()}
    positions = select_extended_d_lab_benchmark(
        args.d_lab, seed=args.seed, target_count=args.lab_positions
    )
    for position in positions:
        validated_mask = validate_mirrored_legality(position.state)
        if validated_mask != position.legal_mask:
            raise RuntimeError(f"D_LAB mask mismatch for {position.position_id}")
    symmetry = evaluate_model_symmetry(models, positions)

    dataset = DRLDataset(args.d_rl)
    d_rl_information = audit_d_rl_information(dataset.examples)
    d_rl_information["games"] = len(dataset.game_ids)

    after = {name: model_parameter_fingerprint(model) for name, model in models.items()}
    if before != after:
        raise RuntimeError("Lot 9A modified source model parameters")

    report = {
        "lot": "9A",
        "question": (
            "Are the learned outputs invariant to physical P1/P2 identity, and does "
            "the pilot D_RL contain enough Policy information to justify G2?"
        ),
        "seed": args.seed,
        "srn_config": asdict(config),
        "symmetry_contract": {
            "operation": (
                "swap pits 0..6 with 7..13, swap stores 14/15, toggle physical "
                "player, preserve local action indices 0..6"
            ),
            "expected_policy": "P(a|S) = P(a|mirror(S))",
            "expected_value": "V(S) = V(mirror(S))",
            "engine_legality_positions_validated": len(positions),
        },
        "sources": {
            "D_LAB": {
                "path": str(args.d_lab),
                "sha256": sha256(args.d_lab),
                "positions": len(positions),
                "used_for_training": False,
            },
            "D_RL": {
                "path": str(args.d_rl),
                "sha256": sha256(args.d_rl),
                "examples": len(dataset),
                "games": len(dataset.game_ids),
            },
            "checkpoints": {
                "G1-best": {
                    "path": str(args.best_checkpoint),
                    "sha256": sha256(args.best_checkpoint),
                    "epoch": best.payload["epoch"],
                },
                "G1-last": {
                    "path": str(args.last_checkpoint),
                    "sha256": sha256(args.last_checkpoint),
                    "epoch": last.payload["epoch"],
                },
            },
        },
        "model_symmetry": symmetry,
        "d_rl_information": d_rl_information,
        "decision": {
            "train_G2_now": False,
            "reason": (
                "G1 learned a strong physical-player asymmetry while the two-visit "
                "Policy targets are frequently one-hot and contradictory for repeated states."
            ),
            "required_before_G2": [
                "freeze an explicit canonical/symmetry contract for state, Policy and Value",
                "remove or neutralize physical-player shortcuts in model inputs",
                "augment or regenerate D_RL with mirrored examples and balanced outcomes",
                "experimentally increase MCTS information beyond the two-visit pilot",
                "make symmetry metrics a checkpoint acceptance gate",
            ],
        },
        "parameter_integrity": {"before": before, "after": after, "unchanged": True},
        "elapsed_s": time.perf_counter() - started_at,
        "no_training": True,
        "no_checkpoint_promoted": True,
    }
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
