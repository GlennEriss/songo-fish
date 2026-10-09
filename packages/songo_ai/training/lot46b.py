"""Lot46B-B1 scientific pilots built on the validated Lot46A pipeline.

This module deliberately contains no arena, promotion, or main-training entrypoint.
"""
from __future__ import annotations

import json
import math
import tarfile
import time
import traceback
from pathlib import Path

import torch

from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.model import SongoGraphBuilder, mask_policy_logits
from songo_ai.training.lot46 import Lot46Error, collate, sha256
from songo_ai.training.lot46a import (
    DurableCheckpointStore,
    ExperimentConfig,
    WeightedStatefulSampler,
    architecture_fingerprint,
    canonical_hash,
    configure_trainable,
    evaluate,
    git_commit,
    load_initial_model,
    make_optimizer_scheduler,
    model_fingerprint,
    prepare_context,
    resume_into,
    run_training,
    validate_scientific_inputs,
)

PILOT_FAMILIES = {
    "CONTROL": "lot46b_pilot_control_report.json",
    "DEEP_POLICY": "lot46b_pilot_deep_policy_report.json",
    "DEEP_POLICY_REPLAY": "lot46b_pilot_deep_replay_report.json",
    "VALUE_INDEPENDENT": "lot46b_pilot_value_report.json",
}
PILOT_IDS = {
    "CONTROL": "LOT46B_PILOT_CONTROL_001",
    "DEEP_POLICY": "LOT46B_PILOT_DEEP_POLICY_001",
    "DEEP_POLICY_REPLAY": "LOT46B_PILOT_DEEP_REPLAY_001",
    "VALUE_INDEPENDENT": "LOT46B_PILOT_VALUE_001",
}
LOT46A_EXPORT_SHA256 = "01b9e1f2e0c0b7a559fceaec74bdd30347786b4fcb5860bae5b0acbae6c96083"
LOT46A_TRAINING_COMMIT = "b9417555e555ac936580f46c33e640cdcc60d51d"
LOT46A_FINALIZATION_COMMIT = "35d12ae5f14517f5b472513c0929fd09ebafa9ca"


def _json(path: Path) -> dict:
    if not path.is_file():
        raise Lot46Error(f"required artifact missing: {path}")
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def audit_lot46a_prerequisites(lot46a_output: Path, export_bundle: Path) -> dict:
    """Validate authoritative Lot46A evidence instead of trusting prompt text."""
    readiness = _json(lot46a_output / "lot46a_readiness_report.json")
    integrity = _json(lot46a_output / "lot46a_checkpoint_integrity.json")
    cuda = _json(lot46a_output / "lot46a_cuda_smoke.json")
    engineering = _json(lot46a_output / "lot46a_engineering_report.json")
    actual_sha = sha256(export_bundle) if export_bundle.is_file() else None
    checks = {
        "LOT46A_VALID": readiness.get("LOT46A_VALID") == "YES",
        "LOT46B_TRAINING_READY": readiness.get("LOT46B_TRAINING_READY") == "YES",
        "CUDA_TRAINING_SMOKE_PASS": readiness.get("CUDA_TRAINING_SMOKE_PASS", {}).get("value") == "YES"
            and cuda.get("status") == "PASS",
        "CHECKPOINT_INTEGRITY": integrity.get("status") == "PASS",
        "EXPORT_SHA256": actual_sha == LOT46A_EXPORT_SHA256,
        "TRAINING_COMMIT": engineering.get("training_code_commit") == LOT46A_TRAINING_COMMIT,
        "FINALIZATION_COMMIT": engineering.get("finalization_code_commit") == LOT46A_FINALIZATION_COMMIT,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "lot46a_output": str(lot46a_output),
        "export_bundle": str(export_bundle),
        "export_sha256_expected": LOT46A_EXPORT_SHA256,
        "export_sha256_actual": actual_sha,
        "checked_at": time.time(),
    }
    if report["status"] != "PASS":
        raise Lot46Error(f"LOT46A_PREREQUISITE_FAILURE: {checks}")
    return report


def _assert_pilot_contract(config: ExperimentConfig) -> None:
    expected = PILOT_IDS.get(config.candidate_family)
    if config.experiment_id != expected:
        raise Lot46Error(f"pilot id mismatch: expected {expected}")
    if config.max_steps < 100 or config.max_steps > 300:
        raise Lot46Error("Lot46B-B1 pilot budget must stay within 100..300 steps")
    if config.confirm_full_training:
        raise Lot46Error("Lot46B-B1 must not enable full training")
    if config.device != "cuda":
        raise Lot46Error("Lot46B-B1 scientific pilots require explicit CUDA")
    if config.initial_checkpoint != "POOL_G4R" or "/pool/" not in config.policy_checkpoint or "/pool/" not in config.value_checkpoint:
        raise Lot46Error("all Lot46B-B1 pilots must initialize from POOL_G4R")
    if config.seed != 20264621:
        raise Lot46Error("comparative pilot seed is frozen at 20264621")


def pilot_prepare(config_path: Path, root: Path, lot46a_output: Path, export_bundle: Path) -> dict:
    config = ExperimentConfig.load(config_path)
    _assert_pilot_contract(config)
    prerequisites = audit_lot46a_prerequisites(lot46a_output, export_bundle)
    dependency = validate_scientific_inputs(config, root, stage="train")
    ctx = prepare_context(config, root)
    train = ctx["splits"]["train"]
    validation = ctx["splits"]["validation"]
    holdout = ctx["splits"]["strategic_holdout"]
    sets = [{x["fingerprint"] for x in rows} for rows in (train, validation, holdout)]
    overlaps = {
        "train_validation": len(sets[0] & sets[1]),
        "train_holdout": len(sets[0] & sets[2]),
        "validation_holdout": len(sets[1] & sets[2]),
    }
    if any(overlaps.values()):
        raise Lot46Error(f"physical-state leakage detected: {overlaps}")
    inventory = []
    for source in config.dataset_sources:
        meta = ctx["source_audit"]["sources"][source.path]
        inventory.append({"name": source.target_source, "source": source.kind, "path": source.path,
            "sha256": meta["sha256"], "number_of_rows": meta["rows"],
            "number_of_unique_states": sum(x["target_source"] == source.target_source for x in ctx["rows"]),
            "target_type": "TERMINAL_Z_ONLY" if config.candidate_family == "VALUE_INDEPENDENT" else "MCTS_VISIT_DISTRIBUTION",
            "allowed_usage": "TRAIN_VALIDATION_ONLY", "weight": source.weight})
    report = {
        "status": "PASS", "experiment_id": config.experiment_id, "family": config.candidate_family,
        "configuration": config.payload(), "config_fingerprint": canonical_hash(config.payload()),
        "dataset_fingerprint": ctx["dataset_fingerprint"], "split_fingerprint": ctx["split_fingerprint"],
        "dataset": ctx["dataset_audit"], "inventory": inventory, "physical_state_overlap": overlaps,
        "prerequisites": prerequisites, "dependencies": dependency, "code_commit": git_commit(root),
        "holdout_policy": "strategic_holdout excluded from optimization and pilot selection",
    }
    _write_json(ctx["output"] / "lot46b_pilot_preflight.json", report)
    return report


def _latest_model(config: ExperimentConfig, root: Path, ctx: dict, device: torch.device):
    model, initial_fp = load_initial_model(config, root, device)
    configure_trainable(model, config.candidate_family, config.training_mode)
    optimizer, scheduler = make_optimizer_scheduler(model, config)
    sampler = WeightedStatefulSampler(ctx["splits"]["train"],
        {s.target_source: s.weight for s in config.dataset_sources}, config.seed, config.batch_size)
    store = DurableCheckpointStore(ctx["output"] / "local_checkpoints", ctx["output"] / "durable_checkpoints")
    latest = store.latest(config.experiment_id)
    if latest is None:
        raise Lot46Error("pilot durable checkpoint not found")
    payload = resume_into(latest[0], config=config, model=model, optimizer=optimizer, scheduler=scheduler,
        sampler=sampler, dataset_fingerprint=ctx["dataset_fingerprint"], split_fingerprint=ctx["split_fingerprint"],
        initial_fingerprint=initial_fp, code_commit=git_commit(root))
    return model, payload, latest


def _initial_comparison(config: ExperimentConfig, root: Path, rows: list[dict], trained, device: torch.device) -> dict:
    initial, _ = load_initial_model(config, root, device)
    initial.eval(); trained.eval(); n = agree = legal = 0; kl = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            batch = collate(rows[start:start + 256], device)
            a, _ = initial(batch["graph"]); b, _ = trained(batch["graph"])
            pa = torch.softmax(mask_policy_logits(a, batch["legal_mask"]), -1)
            pb = torch.softmax(mask_policy_logits(b, batch["legal_mask"]), -1)
            agree += int((pa.argmax(-1) == pb.argmax(-1)).sum())
            legal += int(batch["legal_mask"].gather(1, pb.argmax(-1, keepdim=True)).sum())
            kl += float((pa * (torch.log(pa.clamp_min(1e-12)) - torch.log(pb.clamp_min(1e-12)))).sum())
            n += len(pa)
    return {"positions": n, "initial_policy_top1_agreement": agree / n,
        "initial_to_pilot_policy_kl": kl / n, "legal_action_consistency": legal / n}


def _strategic_preservation(config: ExperimentConfig, root: Path, trained, device: torch.device) -> dict:
    if config.candidate_family == "VALUE_INDEPENDENT":
        return {"status": "NOT_APPLICABLE", "reason": "Policy is frozen in Value-only pilot"}
    path = Path(config.objective["strategic_battery"])
    path = path if path.is_absolute() else root / path
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    initial, _ = load_initial_model(config, root, device); initial.eval(); trained.eval(); builder = SongoGraphBuilder()
    n = agree = 0; kl = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            chunk = rows[start:start + 256]
            states = [RawSongoState(tuple(x["state"]["board"]), int(x["state"]["player_to_move"])) for x in chunk]
            graph = builder.build_batch_vectorized(states).to(device)
            masks = torch.tensor([x["legal_mask"] for x in chunk], dtype=torch.bool, device=device)
            a, _ = initial(graph); b, _ = trained(graph)
            pa = torch.softmax(mask_policy_logits(a, masks), -1); pb = torch.softmax(mask_policy_logits(b, masks), -1)
            agree += int((pa.argmax(-1) == pb.argmax(-1)).sum())
            kl += float((pa * (torch.log(pa.clamp_min(1e-12)) - torch.log(pb.clamp_min(1e-12)))).sum())
            n += len(chunk)
    threshold = float(config.objective.get("preservation_agreement_min", 0.85))
    agreement = agree / n
    return {"status": "PASS" if agreement >= threshold else "FAIL", "positions": n,
        "initial_policy_top1_agreement": agreement, "initial_to_pilot_policy_kl": kl / n,
        "agreement_threshold_preregistered": threshold, "threshold_modified_after_observation": False}


def pilot_validate(config_path: Path, root: Path) -> dict:
    config = ExperimentConfig.load(config_path); _assert_pilot_contract(config)
    ctx = prepare_context(config, root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload, latest = _latest_model(config, root, ctx, device)
    before = model_fingerprint(model)
    metrics = evaluate(model, ctx["splits"]["validation"], device)
    comparison = _initial_comparison(config, root, ctx["splits"]["validation"], model, device)
    preservation = _strategic_preservation(config, root, model, device)
    forward_ok = metrics["positions"] > 0 and all(math.isfinite(x) for x in (
        metrics["policy_ce"], metrics["policy_kl"], metrics["top1_agreement"]))
    finite = all(torch.isfinite(x).all().item() for x in model.state_dict().values())
    manifest = latest[1]
    checks = {"global_step": payload.get("global_step") == config.max_steps == manifest.get("step"),
        "sha256": sha256(latest[0]) == manifest.get("sha256"), "experiment_id": payload.get("experiment_id") == config.experiment_id,
        "config": payload.get("training_config_fingerprint") == canonical_hash(config.payload()),
        "dataset": payload.get("dataset_fingerprint") == ctx["dataset_fingerprint"],
        "split": payload.get("split_fingerprint") == ctx["split_fingerprint"],
        "architecture": payload.get("architecture_fingerprint") == architecture_fingerprint(model),
        "parameters_finite": finite, "forward": forward_ok, "validation_no_update": before == model_fingerprint(model)}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "validation_metrics": metrics, "initial_comparison": comparison, "preservation_metrics": preservation,
        "checkpoint": {"path": str(latest[0]), "manifest": manifest}}
    _write_json(ctx["output"] / "lot46b_pilot_validation.json", result)
    return result


def pilot_report(config_path: Path, root: Path) -> dict:
    config = ExperimentConfig.load(config_path); _assert_pilot_contract(config); ctx = prepare_context(config, root)
    output = ctx["output"]
    status = _json(output / "status.json")
    validation = _json(output / "lot46b_pilot_validation.json")
    history = [json.loads(x) for x in (output / "training_history.jsonl").read_text().splitlines() if x.strip()]
    sampled = [fingerprint for row in history for fingerprint in row.get("batch_fingerprints", [])]
    unique_states = len(set(sampled))
    source_counts = {s.target_source: 0 for s in config.dataset_sources}
    for row in history:
        for source, count in row.get("batch_source_counts", {}).items(): source_counts[source] = source_counts.get(source, 0) + count
    report = {
        "configuration": config.payload(), "dataset": {**ctx["dataset_audit"], "sources": ctx["source_audit"],
            "fingerprint": ctx["dataset_fingerprint"], "split_fingerprint": ctx["split_fingerprint"]},
        "initial_checkpoint": config.initial_checkpoint, "steps": status["global_step"],
        "training_metrics": {"last": history[-1] if history else None, "samples_seen": status["global_step"] * config.batch_size,
            "unique_states_seen": unique_states, "repeated_states": max(0, len(sampled) - unique_states),
            "effective_epochs": status["global_step"] * config.batch_size / len(ctx["splits"]["train"]),
            "samples_per_second": status.get("samples_per_second"), "steps_per_second": status.get("steps_per_second")},
        "validation_metrics": validation["validation_metrics"], "preservation_metrics": validation["preservation_metrics"],
        "GPU_metrics": status.get("gpu_metrics", {"status": "MISSING"}),
        "checkpoint_integrity": validation, "resume_status": status.get("resume_validation", "PENDING_EXPLICIT_RESUME_TEST"),
        "limitations": ["offline pilot metrics do not demonstrate strategic strength", "no arena was run",
            "pilot budgets are too short for a strategic-strength conclusion"], "source_sample_counts": source_counts,
    }
    _write_json(output / PILOT_FAMILIES[config.candidate_family], report)
    return report


def pilot_run(config_path: Path, root: Path, *, resume: bool = False, stop_after: int | None = None) -> dict:
    config = ExperimentConfig.load(config_path); _assert_pilot_contract(config)
    if not torch.cuda.is_available():
        raise Lot46Error("CUDA is mandatory for Lot46B-B1 scientific pilots")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = run_training(config, root, resume=resume, stop_after=stop_after)
    except Exception as exc:
        output = Path(config.output_directory); output = output if output.is_absolute() else root / output
        _write_json(output / "status.json", {"experiment_id": config.experiment_id, "status": "FAILED",
            "stage": "pilot-resume" if resume else "pilot-run", "error_type": type(exc).__name__,
            "error_message": str(exc), "traceback": traceback.format_exc(), "timestamp": time.time()})
        raise
    elapsed = time.perf_counter() - started
    executed = result.get("steps_executed_this_invocation", result["global_step"])
    result["gpu_metrics"] = {"gpu_model": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__, "vram_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "vram_peak_bytes": torch.cuda.max_memory_allocated(), "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(1, executed),
        "samples_per_second": executed * config.batch_size / max(elapsed, 1e-9)}
    _write_json(Path(config.output_directory) / "status.json", result)
    return result


def pilot_resume_test(config_path: Path, root: Path) -> dict:
    config = ExperimentConfig.load(config_path); _assert_pilot_contract(config)
    ctx = prepare_context(config, root); status = _json(ctx["output"] / "status.json")
    evidence = {"status": "PASS" if status.get("resumed") and status.get("global_step") == config.max_steps else "FAIL",
        "resumed": status.get("resumed"), "global_step": status.get("global_step"),
        "optimizer_scheduler_sampler_rng_restored": bool(status.get("resumed")),
        "note": "resume_into validates and restores optimizer, scheduler, sampler and RNG before continuing"}
    status["resume_validation"] = evidence
    _write_json(ctx["output"] / "status.json", status)
    _write_json(ctx["output"] / "lot46b_resume_report.json", evidence)
    return evidence


def pilot_status(config_path: Path, root: Path) -> dict:
    config = ExperimentConfig.load(config_path); ctx = prepare_context(config, root)
    return _json(ctx["output"] / "status.json") if (ctx["output"] / "status.json").is_file() else {"status": "NOT_STARTED"}


def pilot_export(config_paths: list[Path], root: Path, bundle: Path) -> dict:
    configs = [ExperimentConfig.load(x) for x in config_paths]
    reports = []
    completed = []; failed = []
    for config in configs:
        _assert_pilot_contract(config); out = Path(config.output_directory)
        report = out / PILOT_FAMILIES[config.candidate_family]
        validation = out / "lot46b_pilot_validation.json"
        resume_report = out / "lot46b_resume_report.json"
        if (report.is_file() and validation.is_file() and resume_report.is_file()
                and _json(validation).get("status") == "PASS" and _json(resume_report).get("status") == "PASS"):
            completed.append(config.experiment_id)
        else: failed.append(config.experiment_id)
        reports.append((config, out, report))
    all_valid = len(completed) == 4 and not failed
    global_report = {"experiment_design": {"families": list(PILOT_FAMILIES), "seed": 20264621, "steps": 100},
        "completed_pilots": completed, "failed_pilots": failed, "training_stability": "PASS" if all_valid else "INCONCLUSIVE",
        "GPU_throughput": "see per-pilot reports", "VRAM_usage": "see per-pilot reports",
        "checkpoint_integrity": "PASS" if all_valid else "INCONCLUSIVE", "resume_validation": "PASS" if all_valid else "INCONCLUSIVE",
        "offline_metrics": "AVAILABLE" if all_valid else "PARTIAL", "scientific_limitations": ["no arenas", "no strategic-strength claim"],
        "LOT46B_PILOT_VALID": "YES" if all_valid else "INCONCLUSIVE", "MAIN_TRAINING_READY": "YES" if all_valid else "NO"}
    bundle.parent.mkdir(parents=True, exist_ok=True)
    global_path = bundle.parent / "lot46b_pilot_report.json"; _write_json(global_path, global_report)
    plan = {"status": "PLANNED_NOT_LAUNCHED", "candidate_families": list(PILOT_FAMILIES), "seeds": [20264621],
        "budgets": "TO_BE_SELECTED_AFTER_PILOT_MEASUREMENTS", "Colab_assignments": {"COLAB_1": "CONTROL", "COLAB_2": "DEEP_POLICY",
        "COLAB_3": "DEEP_POLICY_REPLAY", "FIRST_AVAILABLE": "VALUE_INDEPENDENT"}}
    plan_path = bundle.parent / "lot46b_main_training_plan.json"; _write_json(plan_path, plan)
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(global_path, arcname=global_path.name); archive.add(plan_path, arcname=plan_path.name)
        for config, out, _ in reports:
            for path in sorted(out.rglob("*")):
                if path.is_file() and "checkpoint-" not in path.name:
                    archive.add(path, arcname=f"{config.experiment_id}/{path.relative_to(out)}")
            for manifest in sorted((out / "durable_checkpoints").glob("*.manifest.json")):
                archive.add(manifest, arcname=f"{config.experiment_id}/checkpoint_references/{manifest.name}")
    digest = sha256(bundle); Path(str(bundle) + ".sha256").write_text(f"{digest}  {bundle.name}\n")
    return {"EXPORT": "PASS", "path": str(bundle), "sha256": digest, "size": bundle.stat().st_size,
        "weights_included": False, "global_report": global_report,
        "MAIN_TRAINING_COMPLETED": "NO", "STRATEGIC_VALIDATION": "NOT_RUN", "ARENAS": "NOT_RUN",
        "SEARCH_COMPRESSION": "NOT_RUN", "G5_PROMOTED": "NO"}
