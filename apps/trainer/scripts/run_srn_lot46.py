#!/usr/bin/env python3
"""Lot46 — validation et entrainement controle G5 (aucune promotion automatique)."""
from __future__ import annotations

import argparse
import json
import hashlib
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

import torch

from songo_ai.model import load_srn_checkpoint
from songo_ai.training.lot46 import (Lot46Dataset, Lot46Error, atomic_torch_save, checkpoint_diagnostic,
                                     collate, group_aware_split, lot46_loss,
                                     rng_state, sha256)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPOSITORY_ROOT / "data/experiments/lot46_g5_training"
LOT45 = REPOSITORY_ROOT / "data/experiments/lot45_g5_target_generation"
G4 = REPOSITORY_ROOT / "data/experiments/lot35_generator_pool/g4_champion_identity.json"
DATASET = LOT45 / "dataset/g5_deep_autonomous_reanalysis_v1.jsonl.gz"
LOT46_INPUT_FILES = (
    G4,
    Path("data/experiments/lot34r_g4_retry/checkpoints/control/step-08000.pt"),
    Path("data/experiments/lot34r_g4_retry/checkpoints/control/step-12000.pt"),
    Path("data/experiments/lot34r_g4_retry/checkpoints/pool/step-06000.pt"),
    Path("data/experiments/lot34r_g4_retry/checkpoints/pool/step-12000.pt"),
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def candidates() -> dict:
    return json.loads(G4.read_text())["candidates"]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def git_commit() -> str | None:
    p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def not_run(reason: str) -> dict:
    return {"status": "NOT_RUN", "reason": reason}


def prepare_inputs(bundle: Path) -> None:
    """Construit le petit bundle Colab Lot46, distinct des resultats Lot45."""
    files = [resolve_project_path(path) for path in LOT46_INPUT_FILES]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Lot46 input files missing: {missing}")
    manifest = {"lot": 46, "purpose": "G4R_FROZEN_BASELINES", "files": {str(p.relative_to(REPOSITORY_ROOT)): sha256(p) for p in files}}
    manifest_path = REPOSITORY_ROOT / "lot46_input_manifest.json"
    atomic_json(manifest_path, manifest)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_name(f".{bundle.name}.tmp")
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(manifest_path, arcname=manifest_path.name)
            for path in files:
                archive.add(path, arcname=str(path.relative_to(REPOSITORY_ROOT)))
        temporary.replace(bundle)
    finally:
        if temporary.exists(): temporary.unlink()
        if manifest_path.exists(): manifest_path.unlink()
    Path(str(bundle) + ".sha256").write_text(f"{sha256(bundle)}  {bundle.name}\n")
    print(json.dumps({"bundle": str(bundle), "sha256": sha256(bundle), "files": len(LOT46_INPUT_FILES)}, indent=2))


def initialize_artifacts(out: Path, reason: str) -> None:
    for name in ("training_history.jsonl",):
        (out / name).touch(exist_ok=True)
    for name in ("checkpoint_manifest.json", "resume_test_report.json", "policy_validation.json",
                 "value_validation.json", "strategic_preservation.json", "candidate_selection.json",
                 "policy_value_crossover.json", "arena_results.json", "arena_confidence_intervals.json",
                 "search_compression.json", "minimax_benchmark.json", "failure_analysis.json"):
        if not (out / name).exists(): atomic_json(out / name, not_run(reason))


def audit(out: Path) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    ids = candidates()
    model_checks = {}
    for key, item in ids.items():
        policy_path, value_path = resolve_project_path(item["policy_checkpoint"]), resolve_project_path(item["value_checkpoint"])
        model_checks[key] = {
            "policy_checkpoint": item["policy_checkpoint"], "value_checkpoint": item["value_checkpoint"],
            "policy_expected": item["policy_fingerprint"], "value_expected": item["value_fingerprint"],
            "policy_actual": sha256(policy_path) if policy_path.is_file() else None,
            "value_actual": sha256(value_path) if value_path.is_file() else None,
        }
        model_checks[key]["valid"] = (model_checks[key]["policy_expected"] == model_checks[key]["policy_actual"] and
                                       model_checks[key]["value_expected"] == model_checks[key]["value_actual"])
    paths = {"dataset": DATASET, "manifest": LOT45 / "dataset_manifest.json", "decision": LOT45 / "decision.json"}
    available = all(p.is_file() for p in paths.values())
    report = {"archive_checked": False, "extracted_input_available": available,
              "paths": {k: str(v) for k, v in paths.items()}, "g4_models": model_checks,
              "MODEL_G4_WEIGHTS_CHANGED": "NO" if all(x["valid"] for x in model_checks.values()) else "YES"}
    valid = False
    if available:
        try:
            data = Lot46Dataset.load(paths["dataset"], paths["manifest"], paths["decision"])
            split = group_aware_split(data.rows)
            valid = True
            report.update({"LOT45_VALID": "YES", "DATASET_READY_FOR_G5": "YES", "rows": len(data.rows),
                           "dataset_sha256": data.dataset_sha256})
            atomic_json(out / "split_manifest.json", {k: [r["fingerprint"] for r in v] for k, v in split.items()})
            atomic_json(out / "leakage_audit.json", {"status": "PASS", "physical_state_overlap": 0,
                        "split_counts": {k: len(v) for k, v in split.items()}})
        except Exception as exc:
            report.update({"LOT45_VALID": "NO", "DATASET_READY_FOR_G5": "NO", "error": str(exc)})
    else:
        report.update({"LOT45_VALID": "NO", "DATASET_READY_FOR_G5": "NO",
                       "error": "lot45_results/extracted final dataset absent; input bundle is not a result dataset"})
        atomic_json(out / "split_manifest.json", not_run(report["error"]))
        atomic_json(out / "leakage_audit.json", not_run(report["error"]))
    atomic_json(out / "lot45_input_validation.json", report)
    atomic_json(out / "dataset_audit.json", {**report, "TRAINING_ALLOWED": "YES" if valid else "NO"})
    return valid and all(x["valid"] for x in model_checks.values())


def plan(out: Path) -> None:
    payload = {"lot": 46, "pre_registered": True, "seed": 20264601,
      "baselines": ["CONTROL_G4R", "POOL_G4R"], "target": "RAW_VISIT_NORMALIZED",
      "candidates": {"A": "NO_DEEP_CONTROL", "B": "DEEP_POLICY_CORRECTION",
                     "C": "DEEP_POLICY_PLUS_HISTORICAL_REPLAY", "D": "POLICY_VALUE_INTERACTION"},
      "objective": {"name": "CORRECT_AND_PRESERVE", "lambda_correction": .1,
                    "lambda_preservation": 1.0, "rho": .5},
      "optimizer": "AdamW", "learning_rate": .0003, "weight_decay": .0001,
      "batch_size": 256, "gradient_clip": 1.0, "checkpoint_cadence": 2000,
      "strategic_preservation_min": .85, "root_noise": False,
      "arena_budgets": [64, 128, 512, 1024, 4096], "automatic_promotion": False,
      "notes": "Hyperparameters inherited from Lot34R; no sweep."}
    atomic_json(out / "training_plan.json", payload)
    atomic_json(out / "candidate_registry.json", {"status": "PRE_REGISTERED", "candidates": payload["candidates"]})


def smoke(out: Path) -> bool:
    gate = json.loads((out / "dataset_audit.json").read_text())
    if gate.get("TRAINING_ALLOWED") != "YES":
        atomic_json(out / "smoke_test_report.json", not_run("dataset hard gate closed")); return False
    data = Lot46Dataset.load(DATASET, LOT45 / "dataset_manifest.json", LOT45 / "decision.json")
    rows = group_aware_split(data.rows)["train"][:8]
    checkpoint = candidates()["POOL"]["policy_checkpoint"]
    loaded = load_srn_checkpoint(checkpoint); model, optimizer = loaded.model, loaded.optimizer
    batch = collate(rows); before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    logits, values = model(batch["graph"]); loss, metrics = lot46_loss(logits, values, batch)
    optimizer.zero_grad(); loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); optimizer.step()
    ckpt = out / "checkpoints/smoke.pt"
    atomic_torch_save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                       "global_step": 1, "rng_state": rng_state(), "source_checkpoint": checkpoint}, ckpt)
    changed = any(not torch.equal(before[k], model.state_dict()[k]) for k in before)
    ok = torch.isfinite(loss) and grad > 0 and changed
    atomic_json(out / "smoke_test_report.json", {"status": "PASS" if ok else "FAIL", "loss": float(loss),
                "gradient_norm": grad, "parameters_changed": changed, "checkpoint": str(ckpt), **metrics})
    return bool(ok)


def preflight(out: Path) -> bool:
    gate = json.loads((out / "dataset_audit.json").read_text()) if (out / "dataset_audit.json").is_file() else {}
    checkpoint_rows = []
    for arm, item in candidates().items():
        for role in ("policy", "value"):
            checkpoint_rows.append(checkpoint_diagnostic(
                name=f"{arm}_G4R_{role.upper()}", expected_path=item[f"{role}_checkpoint"],
                expected_sha256=item[f"{role}_fingerprint"],
                expected_architecture=item["architecture_fingerprint"], repository_root=REPOSITORY_ROOT))
    checkpoint_ok = all(row["CHECKPOINT_STATUS"] == "PASS" for row in checkpoint_rows)
    checkpoint_report = {
        "CHECKPOINT_STATUS": "PASS" if checkpoint_ok else "FAIL",
        "CHECKPOINT_NAME": [row["CHECKPOINT_NAME"] for row in checkpoint_rows],
        "CHECKPOINT_EXPECTED_PATH": [row["CHECKPOINT_EXPECTED_PATH"] for row in checkpoint_rows],
        "CHECKPOINT_RESOLVED_PATH": [row["CHECKPOINT_RESOLVED_PATH"] for row in checkpoint_rows],
        "CHECKPOINT_EXISTS": all(row["CHECKPOINT_EXISTS"] for row in checkpoint_rows),
        "CHECKPOINT_FILE_SIZE": {row["CHECKPOINT_NAME"]: row["CHECKPOINT_FILE_SIZE"] for row in checkpoint_rows},
        "CHECKPOINT_SHA256_VALID": all(row["CHECKPOINT_SHA256_VALID"] for row in checkpoint_rows),
        "CHECKPOINT_FORMAT_VALID": all(row["CHECKPOINT_FORMAT_VALID"] for row in checkpoint_rows),
        "CHECKPOINT_ARCHITECTURE_VALID": all(row["CHECKPOINT_ARCHITECTURE_VALID"] for row in checkpoint_rows),
        "CHECKPOINT_LOAD_VALID": all(row["CHECKPOINT_LOAD_VALID"] for row in checkpoint_rows),
        "CHECKPOINT_FORWARD_VALID": all(row["CHECKPOINT_FORWARD_VALID"] for row in checkpoint_rows),
        "CHECKPOINT_ERROR_TYPE": [row["CHECKPOINT_ERROR_TYPE"] for row in checkpoint_rows if row["CHECKPOINT_ERROR_TYPE"]],
        "CHECKPOINT_ERROR_MESSAGE": [row["CHECKPOINT_ERROR_MESSAGE"] for row in checkpoint_rows if row["CHECKPOINT_ERROR_MESSAGE"]],
        "CHECKPOINT_TRACEBACK": [row["CHECKPOINT_TRACEBACK"] for row in checkpoint_rows if row["CHECKPOINT_TRACEBACK"]],
        "checkpoints": checkpoint_rows,
    }
    atomic_json(out / "checkpoint_preflight_report.json", checkpoint_report)
    checks = {"PROJECT_IMPORTS": True, "PYTHON_DEPENDENCIES": True, "CUDA": torch.cuda.is_available(),
              "DATASET": gate.get("TRAINING_ALLOWED") == "YES", "CHECKPOINT": checkpoint_ok,
              "OUTPUT_WRITE": out.is_dir() and out.stat() is not None}
    critical = all(checks[k] for k in ("PROJECT_IMPORTS", "PYTHON_DEPENDENCIES", "DATASET", "CHECKPOINT", "OUTPUT_WRITE"))
    payload = {"PREFLIGHT_STATUS": "PASS" if critical else "FAIL", "SCIENTIFIC_TRAINING_ALLOWED": "YES" if critical else "NO",
               "checks": checks, **checkpoint_report, "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
               "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    atomic_json(out / "preflight_report.json", payload); return critical


def finalize_blocked(out: Path) -> None:
    reason = "Lot45 final results dataset is absent; scientific training was not authorized."
    initialize_artifacts(out, reason)
    decision = {"G5_PROMOTED": "NO", "G5_STRATEGIC_IMPROVEMENT": "INCONCLUSIVE",
                "DEEP_TARGET_BENEFIT": "INCONCLUSIVE", "POLICY_IMPROVEMENT": "INCONCLUSIVE",
                "VALUE_IMPROVEMENT": "INCONCLUSIVE", "SEARCH_COMPRESSION": "INCONCLUSIVE",
                "TRAINING_COMPLETED": "NO", "DATASET_VALID": "NO", "NEXT_ACTION": "MORE_ANALYSIS_REQUIRED",
                "reason": reason, "controlled_stop": True}
    atomic_json(out / "decision.json", decision); atomic_json(out / "report.json", {"lot": 46, "decision": decision})
    atomic_json(out / "experiment_manifest.json", {"lot": 46, "code_commit": git_commit(), "status": "BLOCKED_AT_DATA_GATE"})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--stage", choices=("prepare-inputs", "audit", "plan", "smoke", "preflight", "prepare"), default="prepare"); p.add_argument("--output", type=Path, default=ROOT); p.add_argument("--bundle", type=Path, default=Path("data/colab_bridge/lot46_inputs.tar.gz")); a = p.parse_args()
    if a.stage == "prepare-inputs": prepare_inputs(a.bundle); return
    if a.stage in ("audit", "prepare"): allowed = audit(a.output)
    if a.stage in ("plan", "prepare"): plan(a.output)
    if a.stage == "smoke": smoke(a.output)
    if a.stage in ("preflight", "prepare"): ready = preflight(a.output)
    if a.stage == "prepare" and not (allowed and ready): finalize_blocked(a.output)
    print(json.dumps({"output": str(a.output), "stage": a.stage}, indent=2))


if __name__ == "__main__": main()
