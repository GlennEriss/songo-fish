"""Primitives verifiables du Lot46 (G5 deep-target training).

Ce module ne lance jamais un entrainement implicitement. Il impose les gates
Lot45, reconstruit les cibles depuis les visites brutes, et garde les etats
terminaux/sans action legale hors de la loss Policy.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.model import SongoGraphBuilder, mask_policy_logits
from songo_ai.model.correct_preserve import correction_loss, preservation_loss


class Lot46Error(RuntimeError):
    """Violation d'un contrat scientifique du Lot46."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reconstruct_policy_target(visits: Sequence[int], legal_mask: Sequence[bool],
                              temperature: float = 1.0) -> list[float] | None:
    """Reconstruit pi depuis N; ``None`` est le contrat d'un etat terminal."""
    if len(visits) != 7 or len(legal_mask) != 7:
        raise Lot46Error("visits and legal_mask must contain 7 actions")
    if not math.isfinite(temperature) or temperature <= 0:
        raise Lot46Error("training target temperature must be finite and positive")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in visits):
        raise Lot46Error("visits must be non-negative integers")
    if not any(legal_mask):
        if any(visits):
            raise Lot46Error("terminal/all-masked state has visits")
        return None
    if any(v and not ok for v, ok in zip(visits, legal_mask)):
        raise Lot46Error("illegal action received visits")
    weights = [(float(v) ** (1.0 / temperature)) if ok else 0.0
               for v, ok in zip(visits, legal_mask)]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise Lot46Error("legal visits must have a finite positive sum")
    target = [w / total for w in weights]
    if not all(math.isfinite(x) for x in target) or abs(sum(target) - 1.0) > 1e-6:
        raise Lot46Error("invalid normalized policy target")
    return target


def _stable_bucket(value: str, seed: int) -> int:
    return int(hashlib.sha256(f"lot46:{seed}:{value}".encode()).hexdigest()[:16], 16) % 10000


def group_aware_split(rows: Sequence[dict], seed: int = 20264601,
                      train: float = .80, validation: float = .10) -> dict[str, list[dict]]:
    """Split deterministe; holdouts reserves et etats physiques ne fuient jamais."""
    if not (0 < train < 1 and 0 <= validation < 1 and train + validation < 1):
        raise ValueError("invalid split fractions")
    fingerprints: dict[str, str] = {}
    groups: dict[str, set[str]] = {}
    for row in rows:
        fp, group = str(row["fingerprint"]), str(row.get("split_group") or f"state:{fp}")
        previous = fingerprints.setdefault(fp, group)
        if previous != group:
            raise Lot46Error(f"duplicate state crosses groups: {fp}")
        groups.setdefault(group, set()).add(fp)
    output = {"train": [], "validation": [], "strategic_holdout": []}
    cut_train, cut_validation = int(train * 10000), int((train + validation) * 10000)
    assignment = {}
    for group in groups:
        bucket = _stable_bucket(group, seed)
        assignment[group] = "train" if bucket < cut_train else "validation" if bucket < cut_validation else "strategic_holdout"
    for row in rows:
        group = str(row.get("split_group") or f"state:{row['fingerprint']}")
        part = "strategic_holdout" if row.get("holdout") else assignment[group]
        # A group containing a protected row is entirely protected.
        if any(r.get("holdout") and str(r.get("split_group") or f"state:{r['fingerprint']}") == group for r in rows):
            part = "strategic_holdout"
        output[part].append(row)
    seen = {}
    for part, values in output.items():
        for row in values:
            fp = row["fingerprint"]
            if fp in seen and seen[fp] != part:
                raise Lot46Error(f"physical state leakage: {fp}")
            seen[fp] = part
    return output


@dataclass
class Lot46Dataset:
    rows: list[dict]
    manifest: dict
    decision: dict
    dataset_sha256: str

    @classmethod
    def load(cls, dataset: Path, manifest: Path, decision: Path) -> "Lot46Dataset":
        if not all(p.is_file() for p in (dataset, manifest, decision)):
            raise Lot46Error("Lot45 dataset, manifest, or decision is missing")
        m, d = json.loads(manifest.read_text()), json.loads(decision.read_text())
        if d.get("LOT45_VALID") != "YES" or d.get("DATASET_READY_FOR_G5") != "YES":
            raise Lot46Error("Lot45 hard gate is closed")
        opener = gzip.open if dataset.suffix == ".gz" else open
        with opener(dataset, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        obj = cls(rows, m, d, sha256(dataset))
        obj.validate()
        return obj

    def validate(self) -> None:
        expected = int(self.manifest.get("unique_states", -1))
        if expected != len(self.rows) or len({r.get("fingerprint") for r in self.rows}) != len(self.rows):
            raise Lot46Error("dataset cardinality or uniqueness mismatch")
        forbidden = {"minimax_label", "minimax_value", "historical_teacher_label"}
        for index, row in enumerate(self.rows):
            if forbidden.intersection(row):
                raise Lot46Error(f"forbidden labels at row {index}")
            if row.get("simulations") != 32768 or row.get("root_noise") is not False:
                raise Lot46Error(f"invalid MCTS identity at row {index}")
            target = reconstruct_policy_target(row["visit_counts"], row["legal_mask"])
            if target is None:
                raise Lot46Error(f"training dataset contains terminal row {index}")
            if any(abs(a - b) > 1e-7 for a, b in zip(target, row["policy_target"])):
                raise Lot46Error(f"policy target mismatch at row {index}")
            available = row.get("value_target_available")
            z = row.get("z_mean")
            if available and (z is None or not math.isfinite(float(z)) or not -1 <= float(z) <= 1):
                raise Lot46Error(f"invalid terminal z at row {index}")
            if not available and z is not None:
                raise Lot46Error(f"missing-z row silently contains a value at row {index}")


def collate(rows: Sequence[dict], device: torch.device | str = "cpu") -> dict:
    states = [RawSongoState(tuple(r["state"]["board"]), int(r["state"]["player_to_move"])) for r in rows]
    graph = SongoGraphBuilder().build_batch_vectorized(states).to(device)
    masks = torch.tensor([r["legal_mask"] for r in rows], dtype=torch.bool, device=device)
    targets = torch.tensor([reconstruct_policy_target(r["visit_counts"], r["legal_mask"]) for r in rows], dtype=torch.float32, device=device)
    values = torch.tensor([float(r["z_mean"]) if r.get("value_target_available") else 0.0 for r in rows], dtype=torch.float32, device=device)
    value_mask = torch.tensor([bool(r.get("value_target_available")) for r in rows], dtype=torch.bool, device=device)
    return {"graph": graph, "legal_mask": masks, "policy_target": targets,
            "value_target": values, "value_target_available": value_mask}


def lot46_loss(logits: torch.Tensor, values: torch.Tensor, batch: dict, *,
               correction_pairs=(), preservation_pairs=(), lambda_correction: float = .1,
               lambda_preservation: float = 1.0, rho: float = .5,
               train_policy: bool = True, train_value: bool = True) -> tuple[torch.Tensor, dict]:
    masked = mask_policy_logits(logits, batch["legal_mask"])
    policy = -(batch["policy_target"] * F.log_softmax(masked, -1)).sum(-1).mean()
    vmask = batch["value_target_available"]
    value = (values.reshape(-1)[vmask] - batch["value_target"][vmask]).square().mean() if vmask.any() else values.sum() * 0
    corr = correction_loss(logits, correction_pairs) if correction_pairs else logits.sum() * 0
    preserve = preservation_loss(logits, preservation_pairs, rho=rho) if preservation_pairs else logits.sum() * 0
    total = (policy + lambda_correction * corr + lambda_preservation * preserve if train_policy else logits.sum() * 0) + (value if train_value else values.sum() * 0)
    if not torch.isfinite(total):
        raise Lot46Error("non-finite training loss")
    return total, {"policy_loss": float(policy.detach()), "value_loss": float(value.detach()),
                   "correction_loss": float(corr.detach()), "preservation_loss": float(preserve.detach())}


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        check = torch.load(temporary, map_location="cpu", weights_only=False)
        required = {"model_state_dict", "optimizer_state_dict", "global_step", "rng_state"}
        if not required.issubset(check):
            raise Lot46Error(f"checkpoint missing {sorted(required - set(check))}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rng_state() -> dict:
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
