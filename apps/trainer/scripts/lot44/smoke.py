"""Smoke test end-to-end miniature (section 35) avec les vraies APIs.

Utilise des positions des 256 d'origine (exclues du corpus OOS par
construction) et des budgets 16/32/64/128 : aucune donnee du TEST Lot44 n'est
touchee. Exerce prepare (dont l'exclusion d'overlap), interruption/reprise,
develop/freeze, evaluate, finalize, export, verification du bundle et
idempotence d'une relance complete.
"""
from __future__ import annotations

import dataclasses
import shutil
import time
from pathlib import Path

from run_srn_lot39 import fingerprint_state

from .artifacts import Lot44FatalError, read_json, utc_now, verify_bundle, write_json
from .config import SMOKE_BUDGETS, SMOKE_DIR_NAME, SMOKE_POSITIONS
from .corpus import state_from_dict
from .finalize import export, finalize
from .pipeline import Context, prepare, run

SMOKE_OVERLAP_INJECTED = 2


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Lot44FatalError("SMOKE_FAILED", message)


def smoke_inputs(original_positions: Path, count: int) -> tuple[list[dict], dict]:
    states = read_json(original_positions)["states"]
    rows = []
    for s in states:
        state = state_from_dict(s)
        rows.append({"fingerprint": fingerprint_state(state), "state": {"board": list(state.board), "player_to_move": state.player_to_move}, "game_id": str(s["source_game_id"]), "ply": s.get("source_ply"), "source_shard": s.get("source_shard"), "source_line": s.get("source_line"), "generator": {}})
    candidates = sorted(rows[:count], key=lambda r: r["fingerprint"])
    reference_fps = {r["fingerprint"] for r in rows[count:]} | {r["fingerprint"] for r in candidates[:SMOKE_OVERLAP_INJECTED]}
    reference = {"positions_file": "SMOKE_SYNTHETIC_REFERENCE", "fingerprints": reference_fps, "games": set(), "lot41_consistent": None, "lot42_fingerprints": 0}
    return candidates, reference


def run_smoke(base: Context, smoke_dir: Path | None = None, *, count: int = SMOKE_POSITIONS) -> dict:
    out = smoke_dir or base.out.parent / SMOKE_DIR_NAME
    if out.name != SMOKE_DIR_NAME:
        raise ValueError(f"smoke output directory must be named {SMOKE_DIR_NAME}: {out}")
    if out.exists():
        shutil.rmtree(out)
    bundle = out.parent / f"{SMOKE_DIR_NAME}_results.tar.gz"
    ctx = dataclasses.replace(base, out=out, budgets=dict(SMOKE_BUDGETS), smoke=True, concurrency_override=8, max_positions=count, max_shards=None, _fingerprints={})
    started = time.perf_counter()
    candidates, reference = smoke_inputs(base.original_positions, count)
    report: dict = {"SMOKE_TEST": "FAIL", "started_utc": utc_now(), "output": str(out), "budgets": SMOKE_BUDGETS, "positions_requested": count, "steps": {}}
    prep = prepare(ctx, candidates=candidates, reference=reference)
    overlap = read_json(out / "overlap_audit.json")
    report["steps"]["prepare"] = {**prep, "candidate_state_overlap_removed": overlap["candidate_state_overlap_removed"]}
    require(overlap["candidate_state_overlap_removed"] == SMOKE_OVERLAP_INJECTED, "injected overlap was not removed")
    require(overlap["OOS_OVERLAP_WITH_ORIGINAL_256"] == 0, "overlap with reference remains")
    interrupted = run(dataclasses.replace(ctx, max_shards=1))
    report["steps"]["interrupted_run"] = {"status": interrupted["status"], "steps": interrupted["steps"]}
    require(interrupted["status"] == "PARTIAL", "max_shards=1 should leave the run partial")
    full = run(ctx)
    report["steps"]["resumed_run"] = {"status": full["status"], "develop": full.get("develop"), "evaluate": full.get("evaluate"), "steps": full["steps"]}
    require(full["status"] == "COMPLETE", f"resumed run status {full['status']}")
    first = full["steps"][0]
    require(first["already_complete"] > 0 and first["computed"] < report["steps"]["prepare"]["sizes"]["train"], "resume recomputed completed shards")
    training = read_json(out / "classifier_training.json")
    calibration = read_json(out / "classifier_calibration.json")
    fitted = [k for k, v in training["models"].items() if v["model"]["status"] == "FITTED"]
    report["steps"]["classifier"] = {"train_positives": training["positives"], "calibration_positives": calibration["positives"], "fitted_models": fitted, "calibration_methods": {k: v["calibrator"]["method"] for k, v in calibration["models"].items()}}
    require(training["positives"] >= 2 and len(fitted) == 3, "smoke must exercise a non-degenerate classifier fit")
    decision = finalize(ctx)
    report["steps"]["finalize"] = {k: decision[k] for k in ("LOT44_VALID", "ROUTER_VALIDATED", "OOS_POSITIVE_COUNT", "TEST_SIZE", "NEXT_ACTION")}
    exported = export(ctx, bundle)
    report["steps"]["export"] = exported
    report["steps"]["bundle_reverified"] = verify_bundle(bundle, "lot44_router_out_of_sample_validation")
    rerun = run(ctx)
    recomputed = sum(s["computed"] for s in rerun["steps"])
    report["steps"]["idempotent_rerun"] = {"status": rerun["status"], "recomputed_positions": recomputed, "develop": rerun["develop"], "evaluate": rerun["evaluate"]}
    require(recomputed == 0 and rerun["develop"]["status"] == "ALREADY_FROZEN" and rerun["evaluate"]["status"] == "ALREADY_EVALUATED", "idempotent rerun recomputed or re-evaluated")
    report.update({"SMOKE_TEST": "PASS", "seconds": round(time.perf_counter() - started, 2), "finished_utc": utc_now(), "scientific_validity": "NONE (pipeline check only)"})
    write_json(out / "smoke_test_report.json", report)
    write_json(base.out / "smoke_test_report.json", report)
    return report
