"""Finalize : dataset G5 valide, audits, decision ; export verifie."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from lot44.artifacts import Lot44FatalError, build_checksums, canonical_hash, make_bundle, read_json, read_jsonl, update_stage_state, utc_now, verify_checksums, write_json, write_jsonl
from lot44.search import load_completed, shard_dir

from . import dataset as dataset_module, generation, selection as selection_module, sources as sources_module
from .config import BUNDLE_NAME, DATASET_NAME, DATASET_SCHEMA_VERSION, EXPERIMENT_DIR_NAME, FORBIDDEN_TARGET_FIELDS, TEACHER_MODEL
from .dataset import dataset_row, old_vs_deep, read_dataset, target_audit, target_statistics, validate_row, write_dataset
from .generation import Context, shard_records

TECHNICAL = ("preflight_report.json", "test_report.json", "smoke_test_report.json", "pilot_report.json", "run_lock_history.json", "errors.jsonl")
INPUT_REPORTS = ("candidate_source_audit.json", "source_selection.json", "source_manifest.json", "deduplication_report.json", "coverage_report.json")
DATASET_FILE = "dataset/g5_deep_autonomous_reanalysis_v1.jsonl.gz"


def minimax_audit() -> dict:
    """Aucun module Lot45 n'importe la recherche Minimax/negamax ou un teacher."""

    forbidden_modules = ("songo_ai.search.negamax", "songo_ai.teachers", "songo_ai.evaluation.minimax_reference")
    imports = {}
    for module in (dataset_module, generation, selection_module, sources_module):
        tree = ast.parse(inspect.getsource(module))
        names = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imports[module.__name__] = sorted(n for n in names if any(n.startswith(f) for f in forbidden_modules) or "negamax" in n or "minimax" in n)
    return {"forbidden_imports": imports, "MINIMAX_LABELS_USED": "NO" if not any(imports.values()) else "YES"}


def collect(ctx: Context) -> tuple[list[dict], list[dict], list[dict], dict]:
    manifest = read_json(ctx.out / "shard_manifest.json")
    identity_fp = ctx.identity().fingerprint()
    if manifest["search_identity_fingerprint"] != identity_fp:
        raise Lot44FatalError("MCTS_CONFIG_INCOMPATIBLE", "search identity differs from shard_manifest.json")
    selection = {r["fingerprint"]: r for r in ctx.selection()}
    teacher = {"model_id": TEACHER_MODEL, **ctx.fingerprints()["model"]}
    rows, invalid, statuses = [], [], []
    for shard in manifest["shards"]:
        records = shard_records(ctx, shard)
        expected = {r["fingerprint"] for r in records}
        directory = shard_dir(ctx.out, shard["shard"], ctx.budget)
        done, files = load_completed(directory, identity_fp, expected, remove_incomplete=False)
        status = {**shard, "complete": len(done) == len(expected), "completed_positions": len(done), "files": files, "devices": []}
        for name in files:
            payload = read_json(directory / name)
            status["devices"].append(payload["device"])
            status.setdefault("wall_time_s", 0.0)
            status["wall_time_s"] += payload["wall_time_s"]
            for row in payload["rows"]:
                problems = validate_row(row, selection[row["state_fingerprint"]], ctx.budget)
                if problems:
                    invalid.append({"fingerprint": row["state_fingerprint"], "reason": problems, "source": selection[row["state_fingerprint"]]["primary_source"], "shard": shard["shard"], "file": name})
                else:
                    rows.append(dataset_row(row, selection[row["state_fingerprint"]], payload, teacher=teacher, identity_fp=identity_fp))
        statuses.append(status)
    rows.sort(key=lambda r: r["order"])
    return rows, invalid, statuses, manifest


def finalize(ctx: Context) -> dict:
    required = INPUT_REPORTS + (() if ctx.smoke else TECHNICAL + ("lot44_input_validation.json",))
    missing = [n for n in required if not (ctx.out / n).is_file()]
    if missing:
        raise Lot44FatalError("ARTIFACT_MISSING", f"cannot finalize, missing {missing}")
    if not ctx.smoke:
        for name, key, value in (("preflight_report.json", "PREFLIGHT_STATUS", "PASS"), ("smoke_test_report.json", "SMOKE_TEST", "PASS"), ("pilot_report.json", "PILOT", "PASS")):
            if read_json(ctx.out / name).get(key) != value:
                raise Lot44FatalError("GATE_FAILED", f"{name}: {key} != {value}")
        if read_json(ctx.out / "test_report.json").get("failed", 1) != 0:
            raise Lot44FatalError("TESTS_FAILED", "test_report.json reports failures")
    rows, invalid, statuses, manifest = collect(ctx)
    incomplete = [s["shard"] for s in statuses if not s["complete"]]
    write_jsonl(ctx.out / "invalid_positions.jsonl", invalid)
    errors = read_jsonl(ctx.out / "errors.jsonl") if (ctx.out / "errors.jsonl").stat().st_size else []
    failed = [e for e in errors if str(e.get("stage", "")).startswith("search/") and e.get("position_fingerprint")]
    write_jsonl(ctx.out / "failed_positions.jsonl", [{"fingerprints": e["position_fingerprint"], "reason": e["message"], "shard": e.get("shard"), "timestamp": e["timestamp"], "fatal": e.get("fatal")} for e in failed])
    write_json(ctx.out / "shard_manifest.json", {**manifest, "status": statuses, "finalized_utc": utc_now()})
    if incomplete:
        raise Lot44FatalError("INCOMPLETE_SEARCH", f"{len(incomplete)} planned shards incomplete (first: {incomplete[:3]}); refusing to present a partial dataset as complete")
    fps = [r["fingerprint"] for r in rows]
    duplicate_conflicts = len(fps) - len(set(fps))
    leaked = sorted({k for r in rows for k in r if k in FORBIDDEN_TARGET_FIELDS})
    selection_rows = {r["fingerprint"]: r for r in ctx.selection()}
    dataset_sha = write_dataset(ctx.out / DATASET_FILE, rows)
    reread = read_dataset(ctx.out / DATASET_FILE)
    if [r["fingerprint"] for r in reread] != fps:
        raise Lot44FatalError("ARTIFACT_INCONSISTENT", "dataset round-trip mismatch")
    stats = target_statistics(rows)
    audit = target_audit(rows, seed=ctx.seed)
    comparison = old_vs_deep(rows, selection_rows)
    write_json(ctx.out / "target_statistics.json", stats)
    write_json(ctx.out / "target_audit.json", audit)
    write_json(ctx.out / "deep_vs_old_target_comparison.json", comparison)
    minimax = minimax_audit()
    with_z = sum(r["value_target_available"] for r in rows)
    total_wall = sum(s.get("wall_time_s", 0.0) for s in statuses)
    devices = sorted({d for s in statuses for d in s["devices"]})
    gpu_wall = sum(s.get("wall_time_s", 0.0) for s in statuses if "cuda" in s["devices"])
    checks = {
        "ALL_TARGETS_FINITE": not invalid, "ALL_TOP_ACTIONS_LEGAL": not any("top_action_illegal_or_not_argmax" in i["reason"] for i in invalid),
        "ALL_BUDGETS_EXACT_32768" if ctx.budget == 32768 else f"ALL_BUDGETS_EXACT_{ctx.budget}": all(r["simulations"] == ctx.budget for r in rows) and not invalid,
        "PHYSICAL_DUPLICATE_CONFLICTS": duplicate_conflicts, "FORBIDDEN_FIELDS_IN_DATASET": leaked,
        "TARGET_AUDIT_PASS": audit["ALL_CHECKS_PASS"], "MINIMAX_IMPORTS": minimax["MINIMAX_LABELS_USED"] == "NO",
    }
    valid = not invalid and duplicate_conflicts == 0 and not leaked and audit["ALL_CHECKS_PASS"] and minimax["MINIMAX_LABELS_USED"] == "NO" and len(rows) == manifest["target_positions"]
    coverage = read_json(ctx.out / "coverage_report.json")
    dataset_manifest = {
        "dataset_name": DATASET_NAME, "schema_version": DATASET_SCHEMA_VERSION, "creation_utc": utc_now(), "file": DATASET_FILE,
        "fingerprints_sha256": dataset_sha, "unique_states": len(rows), "logical_occurrences": sum(r["source_occurrence_count"] for r in rows),
        "source_composition": dict(sorted({f: sum(r["family"] == f for r in rows) for f in {r["family"] for r in rows}}.items())),
        "positions_with_z": with_z, "positions_without_z": len(rows) - with_z, "holdout_positions": sum(r["holdout"] for r in rows),
        "mcts_budget": ctx.budget, "root_noise": "OFF", "policy_target": "normalized raw visits, temperature 1; raw visits kept",
        "value_target": "terminal z (player_to_move perspective) when available; MCTS root Q is diagnostic only",
        "teacher_checkpoint": {"model_id": TEACHER_MODEL, **ctx.fingerprints()["model"]}, "engine_fingerprint": ctx.fingerprints()["engine"],
        "mcts_identity": ctx.identity().payload(), "mcts_identity_fingerprint": ctx.identity().fingerprint(), "code_commit": ctx.fingerprints()["code_commit"],
        "devices": devices, "selection_sha256": manifest["selection_sha256"], "split_metadata": "split_group (game id, or state:<fp>) + holdout flag for group-aware splits in Lot46",
        "smoke": ctx.smoke,
    }
    write_json(ctx.out / "dataset_manifest.json", dataset_manifest)
    decision = {
        "SMOKE_RUN_NOT_SCIENTIFIC": ctx.smoke,
        "LOT45_VALID": "YES" if valid else "NO", "DATASET_READY_FOR_G5": "YES" if valid and not ctx.smoke else "NO",
        "DATASET_NAME": DATASET_NAME, "UNIQUE_PHYSICAL_STATES": len(rows), "LOGICAL_OCCURRENCES": dataset_manifest["logical_occurrences"],
        "POSITIONS_WITH_TERMINAL_Z": with_z, "POSITIONS_WITHOUT_TERMINAL_Z": len(rows) - with_z,
        "MCTS_BUDGET": ctx.budget, "ROOT_NOISE": "OFF", "TEACHER_MODEL": TEACHER_MODEL,
        "MINIMAX_LABELS_USED": minimax["MINIMAX_LABELS_USED"], "HISTORICAL_TEACHER_LABELS_USED": "NO" if not leaked else "YES",
        "MODEL_WEIGHTS_CHANGED": "NO", "TRAINING_PERFORMED": "NO",
        "TOTAL_POSITIONS_REANALYZED": len(rows) + len(invalid), "TOTAL_SIMULATIONS": sum(r["simulations"] for r in rows),
        "TOTAL_GPU_HOURS": gpu_wall / 3600, "TOTAL_DEVICE_HOURS": total_wall / 3600, "DEVICES": devices,
        "MEAN_SIMULATIONS_PER_SECOND": sum(r["simulations"] for r in rows) / total_wall if total_wall else None,
        "FAILED_POSITION_COUNT": len(failed) + len(invalid), "RETRY_COUNT": sum(int(e.get("retry_count") or 0) for e in errors),
        "SHARDS_COMPLETED": sum(s["complete"] for s in statuses), "SHARDS_TOTAL": len(statuses),
        **checks, "SOURCE_COVERAGE": dataset_manifest["source_composition"], "NEW_STATE_RATE": coverage["NEW_STATE_RATE_VS_G2_G3_G4_TRAINING"],
        "NEXT_ACTION": "LOT46_G5_TRAINING" if valid and not ctx.smoke else "LOT45_DATA_REPAIR",
    }
    write_json(ctx.out / "decision.json", decision)
    write_json(ctx.out / "report.json", {"lot": 45, "decision": decision, "dataset_manifest": dataset_manifest, "target_statistics": stats["overall"], "deep_vs_old": comparison["comparisons"], "minimax_audit": minimax, "controlled_stop": True, "finalized_utc": utc_now()})
    update_stage_state(ctx.out, "finalize", {"status": "COMPLETE", "valid": valid})
    checksums = build_checksums(ctx.out)
    write_json(ctx.out / "checksums.json", checksums)
    if verify_checksums(ctx.out, checksums):
        raise Lot44FatalError("CHECKSUM_MISMATCH", "checksums changed right after writing")
    write_json(ctx.out / "experiment_manifest.json", {"lot": 45, "smoke": ctx.smoke, "code_commit": ctx.fingerprints()["code_commit"], "engine_fingerprint": ctx.fingerprints()["engine"], "model_fingerprints": ctx.fingerprints()["model"], "mcts_identity_fingerprint": ctx.identity().fingerprint(), "seed": ctx.seed, "dataset_sha256": canonical_hash(fps), "checksums_sha256": canonical_hash(checksums), "artifacts": len(checksums), "finalized_utc": utc_now()})
    return decision


def export(ctx: Context, bundle: Path | None) -> dict:
    checksums = ctx.out / "checksums.json"
    if not checksums.is_file() or not (ctx.out / "decision.json").is_file():
        raise Lot44FatalError("ARTIFACT_MISSING", "finalize Lot45 before export")
    bad = verify_checksums(ctx.out, read_json(checksums))
    if bad:
        raise Lot44FatalError("CHECKSUM_MISMATCH", f"artifacts changed after finalize: {bad[:5]}")
    return {"EXPORT": "PASS", **make_bundle(ctx.out, bundle or ctx.out.parent.parent / "exports" / BUNDLE_NAME, EXPERIMENT_DIR_NAME)}
