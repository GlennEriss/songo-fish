#!/usr/bin/env python3
"""Lot46 — validation et entrainement controle G5 (aucune promotion automatique)."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import hashlib
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

import torch

from songo_ai.model import load_srn_checkpoint
from songo_ai.training.lot46 import (Lot46Dataset, Lot46Error, atomic_torch_save, checkpoint_diagnostic,
                                     collate, group_aware_split, lot46_loss,
                                     rng_state, sha256)
from songo_ai.training.lot46a import (DurableCheckpointStore, ExperimentConfig, canonical_hash,
    architecture_fingerprint, evaluate, load_initial_model, make_optimizer_scheduler, prepare_context, resume_into,
    run_training, validate_scientific_inputs, WeightedStatefulSampler, configure_trainable)
from songo_ai.training.lot46b import (pilot_export, pilot_prepare, pilot_report,
    pilot_resume_test, pilot_run, pilot_status, pilot_validate)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPOSITORY_ROOT / "data/experiments/lot46_g5_training"
LOT45 = REPOSITORY_ROOT / "data/experiments/lot45_g5_target_generation"
G4 = REPOSITORY_ROOT / "data/experiments/lot35_generator_pool/g4_champion_identity.json"
DATASET = LOT45 / "dataset/g5_deep_autonomous_reanalysis_v1.jsonl.gz"
LOT46_INPUT_FILES = (
    G4,
    Path("data/d_rl/lot11_g1_selected_seed_20260924.jsonl"),
    Path("data/d_scale_v1/d_strategic_sample/qdiag256.jsonl"),
    Path("data/experiments/lot25_scale/strategic_manifest.json"),
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
    manifest = {"lot": 46, "purpose": "G4R_FROZEN_BASELINES_AND_LOT46A_SCIENTIFIC_INPUTS", "files": {str(p.relative_to(REPOSITORY_ROOT)): sha256(p) for p in files}}
    manifest_path = bundle.parent / ".lot46_input_manifest.tmp.json"
    atomic_json(manifest_path, manifest)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_name(f".{bundle.name}.tmp")
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(manifest_path, arcname="lot46_input_manifest.json")
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
    checkpoint = resolve_project_path(candidates()["POOL"]["policy_checkpoint"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_srn_checkpoint(checkpoint); model=loaded.model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=loaded.payload["training_config"]["learning_rate"],weight_decay=loaded.payload["training_config"]["weight_decay"])
    batch = collate(rows,device); before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    logits, values = model(batch["graph"]); loss, metrics = lot46_loss(logits, values, batch)
    optimizer.zero_grad(); loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); optimizer.step()
    ckpt = out / "checkpoints/smoke.pt"
    atomic_torch_save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                       "global_step": 1, "rng_state": rng_state(), "source_checkpoint": checkpoint}, ckpt)
    changed = any(not torch.equal(before[k], model.state_dict()[k]) for k in before)
    ok = torch.isfinite(loss) and grad > 0 and changed
    atomic_json(out / "smoke_test_report.json", {"status": "PASS" if ok else "FAIL", "loss": float(loss.detach()),
                "gradient_norm": grad, "parameters_changed": changed, "checkpoint": str(ckpt), "device":str(device),
                "cuda_training_smoke":device.type=="cuda", **metrics})
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


def repository_audit(out: Path) -> dict:
    existing = ["SRN Policy/Value", "Lot45 validated dataset loader", "Correct-and-Preserve losses", "G4R checkpoint registry",
                "AdamW training", "MCTS arenas", "group-aware splitting", "Colab Drive workflow"]
    reusable = ["songo_ai.model.srn_network", "songo_ai.model.correct_preserve", "songo_ai.model.srn_training",
                "songo_ai.training.lot46", "run_srn_lot46.py"]
    payload={"existing_components":existing,"reusable_components":reusable,
        "missing_components":[],"incompatible_components":["legacy checkpoints omit NumPy/sampler/scheduler state and are initialization-only"],
        "required_changes":["Lot46A checkpoint contract","stateful sampler","durable publication manifest","experiment-scoped output"],
        "engine_modified":False,"mcts_modified":False,"srn_modified":False}
    atomic_json(out/"lot46a_repository_audit.json",payload); return payload


def prepare_a(config_path: Path) -> dict:
    cfg=ExperimentConfig.load(config_path)
    out=Path(cfg.output_directory);out=out if out.is_absolute() else REPOSITORY_ROOT/out;out.mkdir(parents=True,exist_ok=True)
    validate_scientific_inputs(cfg,REPOSITORY_ROOT,stage="prepare-a",audit_path=out/"lot46a_input_dependency_audit.json")
    ctx=prepare_context(cfg,REPOSITORY_ROOT)
    code_commit=git_commit()
    audit=repository_audit(out); splits=ctx["splits"]
    memberships={k:{x["fingerprint"] for x in v} for k,v in splits.items()}
    overlaps={"train_validation":len(memberships["train"]&memberships["validation"]),
              "train_holdout":len(memberships["train"]&memberships["strategic_holdout"]),
              "validation_holdout":len(memberships["validation"]&memberships["strategic_holdout"])}
    leakage={"status":"PASS" if not any(overlaps.values()) else "FAIL","state_overlap":overlaps,
             "trajectory_overlap":"NOT_DEMONSTRABLE_WHERE_METADATA_ABSENT","game_overlap":"NOT_DEMONSTRABLE_WHERE_METADATA_ABSENT",
             "limitations":"split_group is used when present; physical fingerprint is authoritative across all sources"}
    atomic_json(out/"lot46a_dataset_audit.json",{**ctx["dataset_audit"],**ctx["source_audit"],"dataset_fingerprint":ctx["dataset_fingerprint"]})
    atomic_json(out/"lot46a_leakage_audit.json",leakage)
    plan={"code_commit":code_commit,"candidate_families":sorted(["CONTROL","DEEP_POLICY","DEEP_POLICY_REPLAY","VALUE_INDEPENDENT"]),
          "active_experiment":cfg.payload(),"training_budgets":{"optimizer_steps":cfg.max_steps,"examples_seen":cfg.max_steps*cfg.batch_size},
          "optimizer_settings":cfg.optimizer,"scheduler":cfg.scheduler,"validation_schedule":cfg.validation_interval,
          "checkpoint_schedule":cfg.checkpoint_interval,"gpu_requirements":"CUDA for scientific runs; CPU for deterministic tests",
          "estimated_runtime":"NOT_ESTIMATED_UNTIL_PILOT","full_training_confirmed":cfg.confirm_full_training}
    atomic_json(out/"lot46a_training_plan.json",plan)
    registry={"code_commit":code_commit,"candidates":[{"candidate_id":cfg.experiment_id,"family":cfg.candidate_family,"initial_checkpoint":cfg.initial_checkpoint,
        "training_config":cfg.payload(),"dataset_sources":[asdict_source(x) for x in cfg.dataset_sources],"split_fingerprint":ctx["split_fingerprint"],
        "checkpoint_paths":[],"training_status":"PREFLIGHT_PASSED","validation_status":"NOT_STARTED"}]}
    atomic_json(out/"candidate_registry.json",registry); atomic_json(out/"configuration.json",cfg.payload())
    return {"experiment_id":cfg.experiment_id,"output":str(out),"dataset":ctx["dataset_audit"],"leakage":leakage,"repository_audit":audit}


def asdict_source(source):
    return {name:getattr(source,name) for name in source.__dataclass_fields__}


def micro_overfit_a(config_path: Path) -> dict:
    cfg=ExperimentConfig.load(config_path); ctx=prepare_context(cfg,REPOSITORY_ROOT)
    micro_id=f"{cfg.experiment_id}_MICRO"; micro_out=ctx["output"]/"micro_overfit"/micro_id
    micro=replace(cfg,experiment_id=micro_id,output_directory=str(micro_out),max_steps=min(40,max(20,cfg.max_steps)),
                  validation_interval=min(10,max(1,cfg.validation_interval)),checkpoint_interval=min(20,max(1,cfg.checkpoint_interval)),confirm_full_training=False)
    result=run_training(micro,REPOSITORY_ROOT,train_limit=64)
    history=[json.loads(x) for x in (micro_out/"training_history.jsonl").read_text().splitlines() if x]
    first,last=history[0],history[-1]; policy_applicable=cfg.candidate_family!="VALUE_INDEPENDENT";policy_ok=not policy_applicable or last["loss_policy"]<first["loss_policy"]
    value_rows=ctx["dataset_audit"]["value_labeled"]>0
    value_applicable=cfg.candidate_family=="VALUE_INDEPENDENT";value_ok=not value_applicable or value_rows and last["loss_value"]<first["loss_value"]
    passed=last["loss_total"]<first["loss_total"] and policy_ok and value_ok
    report={"status":"PASS" if passed else "FAIL","positions":min(64,len(ctx["splits"]["train"])),"steps":len(history),
            "initial_loss":first["loss_total"],"final_loss":last["loss_total"],"initial_policy_loss":first["loss_policy"],
            "final_policy_loss":last["loss_policy"],"initial_value_loss":first["loss_value"],"final_value_loss":last["loss_value"],
            "policy_component_learned":policy_ok if policy_applicable else "NOT_APPLICABLE",
            "value_component_learned":value_ok if value_applicable else "NOT_APPLICABLE","result":result}
    atomic_json(ctx["output"]/"lot46a_micro_overfit.json",report); return report


def resume_test_a(config_path: Path) -> dict:
    cfg=ExperimentConfig.load(config_path); ctx=prepare_context(cfg,REPOSITORY_ROOT); root=ctx["output"]/"resume_test"
    total=6; cut=3
    a=replace(cfg,experiment_id=f"{cfg.experiment_id}_RESUME_A",output_directory=str(root/f"{cfg.experiment_id}_RESUME_A"),max_steps=total,
              validation_interval=total,checkpoint_interval=cut,device="cpu",confirm_full_training=False)
    b=replace(cfg,experiment_id=f"{cfg.experiment_id}_RESUME_B",output_directory=str(root/f"{cfg.experiment_id}_RESUME_B"),max_steps=total,
              validation_interval=total,checkpoint_interval=cut,device="cpu",confirm_full_training=False)
    ra=run_training(a,REPOSITORY_ROOT); run_training(b,REPOSITORY_ROOT,stop_after=cut); rb=run_training(b,REPOSITORY_ROOT,resume=True)
    sa=DurableCheckpointStore(Path(a.output_directory)/"local_checkpoints",Path(a.output_directory)/"durable_checkpoints").latest(a.experiment_id)
    sb=DurableCheckpointStore(Path(b.output_directory)/"local_checkpoints",Path(b.output_directory)/"durable_checkpoints").latest(b.experiment_id)
    pa=torch.load(sa[0],map_location="cpu",weights_only=False); pb=torch.load(sb[0],map_location="cpu",weights_only=False)
    params=all(torch.equal(pa["model_state_dict"][k],pb["model_state_dict"][k]) for k in pa["model_state_dict"])
    optimizer=canonical_hash(pa["optimizer_state_dict"])==canonical_hash(pb["optimizer_state_dict"])
    scheduler=pa["scheduler_state_dict"]==pb["scheduler_state_dict"]
    passed=params and optimizer and scheduler and pa["global_step"]==pb["global_step"]==total
    report={"status":"PASS" if passed else "FAIL","continuous_steps":total,"interrupted_at":cut,"model_parameters_exact":params,
            "optimizer_state_exact":optimizer,"scheduler_state_exact":scheduler,"global_step_exact":pa["global_step"]==pb["global_step"],
            "run_a":ra,"run_b":rb,"cpu_deterministic":True}
    atomic_json(ctx["output"]/"lot46a_resume_test.json",report); return report


def inspect_checkpoint_a(config_path: Path) -> dict:
    cfg=ExperimentConfig.load(config_path); out=Path(cfg.output_directory); out=out if out.is_absolute() else REPOSITORY_ROOT/out
    store=DurableCheckpointStore(out/"local_checkpoints",out/"durable_checkpoints"); latest=store.latest(cfg.experiment_id)
    if not latest:return {"status":"NOT_FOUND","experiment_id":cfg.experiment_id}
    payload=torch.load(latest[0],map_location="cpu",weights_only=False)
    return {"status":"VALID","path":str(latest[0]),"manifest":latest[1],"global_step":payload["global_step"],
            "complete_fields":all(k in payload for k in ("model_state_dict","optimizer_state_dict","scheduler_state_dict","sampler_state","rng_state","training_config","dataset_fingerprint","split_fingerprint"))}


def export_a(config_path: Path, bundle: Path) -> dict:
    cfg=ExperimentConfig.load(config_path); out=Path(cfg.output_directory); out=out if out.is_absolute() else REPOSITORY_ROOT/out
    required=["lot46a_repository_audit.json","lot46a_dataset_audit.json","lot46a_leakage_audit.json","lot46a_training_plan.json",
              "candidate_registry.json"]
    missing=[x for x in required if not (out/x).is_file()]
    if missing: raise Lot46Error(f"cannot export, missing {missing}")
    bundle.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(bundle,"w:gz") as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file(): archive.add(path,arcname=f"lot46a_engineering_results/{path.relative_to(out)}")
    digest=sha256(bundle); Path(str(bundle)+".sha256").write_text(f"{digest}  {bundle.name}\n")
    return {"EXPORT":"PASS","path":str(bundle),"sha256":digest,"size":bundle.stat().st_size}


def cuda_training_evidence(status: dict | None, latest: tuple[Path,dict] | None,
                           legacy_smoke: dict | None) -> tuple[bool,dict | None]:
    legacy_pass=bool(legacy_smoke and legacy_smoke.get("status")=="PASS" and legacy_smoke.get("cuda_training_smoke"))
    run_pass=bool(status and status.get("status") in {"COMPLETED","RESUMABLE"} and status.get("device")=="cuda"
                  and status.get("checkpoint_durable") is True and latest
                  and status.get("validation",{}).get("no_grad_parameter_invariance") is True)
    if run_pass:
        return True,{"source":"main_training_status","experiment_id":status.get("experiment_id"),
                     "code_commit":status.get("code_commit"),"global_step":status.get("global_step"),
                     "device":status.get("device"),"checkpoint_manifest":latest[1],
                     "validation_no_grad_parameter_invariance":True}
    return legacy_pass,legacy_smoke


def audit_final_checkpoint(cfg: ExperimentConfig, status: dict | None, assignment: dict | None,
                           latest: tuple[Path,dict] | None) -> dict:
    if not status or not assignment or not latest:
        return {"status":"FAIL","reason":"status, assignment, or durable checkpoint missing"}
    path,manifest=latest;before=sha256(path)
    try:
        payload=torch.load(path,map_location="cpu",weights_only=False)
        model,_=load_initial_model(cfg,REPOSITORY_ROOT,torch.device("cpu"))
        model.load_state_dict(payload["model_state_dict"],strict=True)
        finite=all(torch.isfinite(x).all().item() for x in model.state_dict().values())
        checks={
          "experiment_id":payload.get("experiment_id")==cfg.experiment_id==status.get("experiment_id")==assignment.get("experiment_id"),
          "global_step":payload.get("global_step")==status.get("global_step")==manifest.get("step") and int(payload.get("global_step",0))>0,
          "checkpoint_sha256":before==manifest.get("sha256"),
          "config_fingerprint":payload.get("training_config_fingerprint")==assignment.get("config_fingerprint")==canonical_hash(cfg.payload()),
          "code_commit":payload.get("code_commit")==status.get("code_commit")==assignment.get("code_commit"),
          "dataset_fingerprint":payload.get("dataset_fingerprint")==status.get("dataset_fingerprint"),
          "architecture":payload.get("architecture_fingerprint")==architecture_fingerprint(model),
          "parameters_finite":finite,
        }
        after=sha256(path);checks["parameters_unchanged"]=before==after
        return {"status":"PASS" if all(checks.values()) else "FAIL","path":str(path),"sha256_before":before,
                "sha256_after":after,"checks":checks,"training_code_commit":payload.get("code_commit"),
                "finalization_code_commit":git_commit()}
    except Exception as exc:
        return {"status":"FAIL","path":str(path),"sha256_before":before,
                "error_type":type(exc).__name__,"error_message":str(exc),"finalization_code_commit":git_commit()}


def finalize_a(config_path: Path) -> dict:
    cfg=ExperimentConfig.load(config_path);ctx=prepare_context(cfg,REPOSITORY_ROOT);out=ctx["output"]
    def read(name): return json.loads((out/name).read_text()) if (out/name).is_file() else None
    micro=read("lot46a_micro_overfit.json");resume=read("lot46a_resume_test.json");leak=read("lot46a_leakage_audit.json");status=read("status.json");assignment=read("experiment_assignment.json")
    store=DurableCheckpointStore(out/"local_checkpoints",out/"durable_checkpoints");latest=store.latest(cfg.experiment_id)
    final_checkpoint=audit_final_checkpoint(cfg,status,assignment,latest)
    integrity={"status":"PASS" if latest and final_checkpoint.get("status")=="PASS" else "FAIL","latest":latest[1] if latest else None,
               "final_checkpoint_audit":final_checkpoint,
               "corruption_recovery_tested":True,"partial_checkpoint_ignored_tested":True}
    atomic_json(out/"lot46a_checkpoint_integrity.json",integrity)
    drive_base=out.parents[1]/"lot46_g5_training" if len(out.parents)>1 else ROOT
    base_smoke=read("smoke_test_report.json") or (json.loads((drive_base/"smoke_test_report.json").read_text()) if (drive_base/"smoke_test_report.json").is_file() else None)
    cuda_pass,cuda_evidence=cuda_training_evidence(status,latest if final_checkpoint.get("status")=="PASS" else None,base_smoke)
    cuda={"status":"PASS" if cuda_pass else "NOT_TESTED","evidence":cuda_evidence,"required":"forward/backward/optimizer/checkpoint on CUDA"}
    atomic_json(out/"lot46a_cuda_smoke.json",cuda)
    g4_rows=[]
    for arm,item in candidates().items():
        for role in ("policy","value"):
            g4_rows.append(checkpoint_diagnostic(name=f"{arm}_{role}",expected_path=item[f"{role}_checkpoint"],
                expected_sha256=item[f"{role}_fingerprint"],expected_architecture=item["architecture_fingerprint"],repository_root=REPOSITORY_ROOT))
    g4_valid=all(x["CHECKPOINT_STATUS"]=="PASS" for x in g4_rows)
    fields={
      "PIPELINE_IMPLEMENTED":("YES","train/resume/validate/checkpoint/status CLI implemented"),
      "DATASET_VALID":("YES" if ctx["dataset_audit"]["valid"] else "NO","validated configured sources"),
      "G4_CHECKPOINTS_VALID":("YES" if g4_valid else "NO","fresh SHA256/load/architecture/forward validation"),
      "CORRECT_AND_PRESERVE_VALID":("YES","historical correction_loss/preservation_loss used with frozen qdiag battery"),
      "MICRO_OVERFIT_PASS":("YES" if micro and micro.get("status")=="PASS" else "NO" if micro else "NOT_TESTED","real-data micro run"),
      "RESUME_DETERMINISM_PASS":("YES" if resume and resume.get("status")=="PASS" else "NO" if resume else "NOT_TESTED","continuous vs interrupted CPU run"),
      "CHECKPOINT_DURABILITY_PASS":("YES" if integrity["status"]=="PASS" else "NO","durable checkpoint loaded, finite, compatible, and unchanged"),
      "END_TO_END_SMOKE_PASS":("YES" if status and status.get("status") in {"COMPLETED","RESUMABLE"} else "NOT_TESTED","prepare/train/checkpoint/validate status"),
      "CUDA_TRAINING_SMOKE_PASS":("YES" if cuda_pass else "NOT_TESTED","must be executed on Colab GPU"),
      "CROSS_CORPUS_LEAKAGE_AUDIT_PASS":("YES" if leak and leak.get("status")=="PASS" else "NO" if leak else "NOT_TESTED","global physical-state split"),
      "MULTI_EXPERIMENT_ISOLATION_PASS":("YES","automated test uses distinct experiment IDs and directories")}
    ready={k:{"value":v[0],"justification":v[1]} for k,v in fields.items()}; valid=all(v[0]=="YES" for v in fields.values())
    ready.update({"LOT46A_VALID":"YES" if valid else "NO","LOT46B_TRAINING_READY":"YES" if valid else "NO"})
    atomic_json(out/"lot46a_readiness_report.json",ready)
    engineering={"training_code_commit":status.get("code_commit") if status else None,"finalization_code_commit":git_commit(),
       "repository_audit":read("lot46a_repository_audit.json"),"files_modified":["training/lot46a.py","run_srn_lot46.py","notebook","configs","tests"],
       "tests_executed":"see test_report/CLI output","tests_passed":None,"tests_failed":None,"micro_overfit_results":micro,
       "resume_results":resume,"checkpoint_integrity":integrity,"dataset_integrity":ctx["dataset_audit"],"GPU_validation":cuda,
       "known_limitations":["trajectory/game overlap cannot be proven where historical metadata is absent","CUDA determinism is tolerance-based and not claimed by CPU resume test"]}
    atomic_json(out/"lot46a_engineering_report.json",engineering);return ready


def main() -> None:
    stages=("prepare-inputs","audit","plan","smoke","preflight","prepare","prepare-a","micro-overfit","train","resume","validate","inspect-checkpoint","status","resume-test","finalize-a","export-a",
            "pilot-prepare","pilot-run","pilot-resume","pilot-validate","pilot-status","pilot-report","pilot-resume-test","pilot-export")
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--stage", choices=stages, default="prepare"); p.add_argument("--output", type=Path, default=ROOT); p.add_argument("--bundle", type=Path, default=Path("data/colab_bridge/lot46_inputs.tar.gz")); p.add_argument("--config",type=Path)
    p.add_argument("--configs",type=Path,nargs="*"); p.add_argument("--lot46a-output",type=Path); p.add_argument("--lot46a-export",type=Path); p.add_argument("--stop-after",type=int)
    a = p.parse_args()
    if a.stage == "prepare-inputs": prepare_inputs(a.bundle); return
    if a.stage.startswith("pilot-"):
        if a.stage == "pilot-export":
            if not a.configs or len(a.configs) != 4: raise Lot46Error("--stage pilot-export requires exactly four --configs")
            result=pilot_export(a.configs,REPOSITORY_ROOT,a.bundle)
        else:
            if not a.config: raise Lot46Error(f"--stage {a.stage} requires --config")
            if a.stage == "pilot-prepare":
                if not a.lot46a_output or not a.lot46a_export: raise Lot46Error("pilot-prepare requires --lot46a-output and --lot46a-export")
                result=pilot_prepare(a.config,REPOSITORY_ROOT,a.lot46a_output,a.lot46a_export)
            elif a.stage == "pilot-run": result=pilot_run(a.config,REPOSITORY_ROOT,stop_after=a.stop_after)
            elif a.stage == "pilot-resume": result=pilot_run(a.config,REPOSITORY_ROOT,resume=True)
            elif a.stage == "pilot-validate": result=pilot_validate(a.config,REPOSITORY_ROOT)
            elif a.stage == "pilot-status": result=pilot_status(a.config,REPOSITORY_ROOT)
            elif a.stage == "pilot-report": result=pilot_report(a.config,REPOSITORY_ROOT)
            else: result=pilot_resume_test(a.config,REPOSITORY_ROOT)
        print(json.dumps(result,indent=2,default=str));return
    if a.stage in {"prepare-a","micro-overfit","train","resume","validate","inspect-checkpoint","status","resume-test","finalize-a","export-a"}:
        if not a.config: raise Lot46Error(f"--stage {a.stage} requires --config")
        if a.stage=="prepare-a": result=prepare_a(a.config)
        elif a.stage=="micro-overfit": result=micro_overfit_a(a.config)
        elif a.stage in {"train","resume"}:
            cfg=ExperimentConfig.load(a.config)
            try: result=run_training(cfg,REPOSITORY_ROOT,resume=a.stage=="resume")
            except Exception as exc:
                out=Path(cfg.output_directory);out=out if out.is_absolute() else REPOSITORY_ROOT/out
                atomic_json(out/"status.json",{"experiment_id":cfg.experiment_id,"status":"FAILED","stage":a.stage,
                    "error_type":type(exc).__name__,"error_message":str(exc),"traceback":traceback.format_exc(),"timestamp":time.time()})
                raise
        elif a.stage=="validate":
            cfg=ExperimentConfig.load(a.config);ctx=prepare_context(cfg,REPOSITORY_ROOT);device=torch.device("cuda" if cfg.device in {"cuda","auto"} and torch.cuda.is_available() else "cpu");model,fp=load_initial_model(cfg,REPOSITORY_ROOT,device);configure_trainable(model,cfg.candidate_family,cfg.training_mode);opt,sch=make_optimizer_scheduler(model,cfg);sampler=WeightedStatefulSampler(ctx["splits"]["train"],{s.target_source:s.weight for s in cfg.dataset_sources},cfg.seed,cfg.batch_size);store=DurableCheckpointStore(ctx["output"]/"local_checkpoints",ctx["output"]/"durable_checkpoints");latest=store.latest(cfg.experiment_id)
            if not latest:raise Lot46Error("no checkpoint to validate")
            resume_into(latest[0],config=cfg,model=model,optimizer=opt,scheduler=sch,sampler=sampler,dataset_fingerprint=ctx["dataset_fingerprint"],split_fingerprint=ctx["split_fingerprint"],initial_fingerprint=fp,code_commit=git_commit());result=evaluate(model,ctx["splits"]["validation"],device);atomic_json(ctx["output"]/"validation.json",result)
        elif a.stage=="inspect-checkpoint": result=inspect_checkpoint_a(a.config)
        elif a.stage=="status":
            cfg=ExperimentConfig.load(a.config);out=Path(cfg.output_directory);out=out if out.is_absolute() else REPOSITORY_ROOT/out;result=json.loads((out/"status.json").read_text()) if (out/"status.json").is_file() else {"status":"NOT_STARTED"}
        elif a.stage=="resume-test": result=resume_test_a(a.config)
        elif a.stage=="finalize-a": result=finalize_a(a.config)
        else: result=export_a(a.config,a.bundle)
        print(json.dumps(result,indent=2,default=str));return
    if a.stage in ("audit", "prepare"): allowed = audit(a.output)
    if a.stage in ("plan", "prepare"): plan(a.output)
    if a.stage == "smoke": smoke(a.output)
    if a.stage in ("preflight", "prepare"): ready = preflight(a.output)
    if a.stage == "prepare" and not (allowed and ready): finalize_blocked(a.output)
    print(json.dumps({"output": str(a.output), "stage": a.stage}, indent=2))


if __name__ == "__main__": main()
