"""Smoke Lot45 : pipeline complet a budget reduit, au FORMAT final du dataset.

Positions de sonde (les 256 du Lot39, deja hors corpus) ; budget 64 ; shards de
8. Exerce : prepare, contention du verrou, pilote, interruption/reprise,
finalize (dataset final), export verifie et relance idempotente.
"""
from __future__ import annotations

import dataclasses
import shutil
import time
from pathlib import Path

from run_srn_lot39 import fingerprint_state

from lot44.artifacts import Lot44FatalError, read_json, utc_now, verify_bundle, write_json
from lot44.corpus import legal_action_count, state_from_dict

from .config import EXPERIMENT_DIR_NAME, SMOKE_BUDGET, SMOKE_DIR_NAME, SMOKE_SHARD_SIZE
from .dataset import read_dataset
from .finalize import DATASET_FILE, export, finalize
from .generation import Context, generate, pilot, prepare
from .run_lock import RunLock
from .selection import composition, coverage_report, phase_bin, write_selection

SMOKE_POSITIONS = 24
REQUIRED_DATASET_KEYS = {"fingerprint", "state", "legal_mask", "visit_counts", "policy_target", "selected_action", "root_q_values_diagnostic", "root_priors", "simulations", "nodes", "network_evaluations", "runtime_s_estimate", "device", "value_target_available", "z_counts", "family", "sources", "source_occurrence_count", "split_group", "holdout", "teacher_model", "mcts_identity_fingerprint"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Lot44FatalError("SMOKE_FAILED", message)


def smoke_selection(probe: Path, count: int) -> list[dict]:
    rows = []
    for s in read_json(probe)["states"]:
        state = {"board": s["board"], "player_to_move": s["player_to_move"]}
        legal = legal_action_count(state_from_dict(state))
        if legal is None or legal < 2:
            continue
        seeds = sum(s["board"][:14])
        z = (1, 0, 0, 0) if len(rows) % 3 == 0 else (0, 0, 0, 1)
        rows.append({
            "order": len(rows), "fingerprint": fingerprint_state(state_from_dict(state)), "state": state, "family": "SMOKE_PROBE", "primary_source": "SMOKE_LOT39_PROBE",
            "source_occurrence_count": 1 + len(rows) % 2, "sources": {"SMOKE_LOT39_PROBE": 1 + len(rows) % 2},
            "z_counts": dict(zip(("win", "draw", "loss", "unknown"), z)), "z_perspective": "player_to_move", "value_target_available": z[3] == 0, "z_mean": 1.0 if z[3] == 0 else None,
            "split_group": str(s["source_game_id"]), "games_sample": [str(s["source_game_id"])], "min_ply": s.get("source_ply"), "legal_count": legal, "seeds_in_play": seeds,
            "phase": phase_bin(seeds), "stores": s["board"][14:16], "holdout": len(rows) % 10 == 0, "old_targets": [],
        })
        if len(rows) == count:
            break
    return rows


def run_smoke(base: Context, smoke_dir: Path | None = None) -> dict:
    out = smoke_dir or base.out.parent / SMOKE_DIR_NAME
    if out.name != SMOKE_DIR_NAME:
        raise ValueError(f"smoke directory must be named {SMOKE_DIR_NAME}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = smoke_selection(base.original_positions, SMOKE_POSITIONS)
    selection = out / "inputs" / "selection.jsonl.gz"
    write_selection(selection, rows)
    for name, payload in (("candidate_source_audit.json", {"smoke": True}), ("source_selection.json", {"smoke": True, "composition_full": composition(rows)}), ("source_manifest.json", {"smoke": True}), ("deduplication_report.json", {"smoke": True}), ("coverage_report.json", coverage_report(rows))):
        write_json(out / name, payload)
    ctx = dataclasses.replace(base, out=out, selection_file=selection, budget=SMOKE_BUDGET, shard_size=SMOKE_SHARD_SIZE, concurrency=SMOKE_SHARD_SIZE, target_positions=SMOKE_POSITIONS, smoke=True, lock_settle_s=0.0, max_shards=None, _fingerprints={}, _selection=None)
    started = time.perf_counter()
    report: dict = {"SMOKE_TEST": "FAIL", "started_utc": utc_now(), "output": str(out), "budget": SMOKE_BUDGET, "positions": SMOKE_POSITIONS, "steps": {}}
    report["steps"]["prepare"] = prepare(ctx)
    foreign = RunLock(out, owner="foreign-writer", settle_s=0)
    foreign.acquire()
    try:
        generate(ctx, owner="smoke")
        refused = False
    except Lot44FatalError as exc:
        refused = exc.code == "RUN_LOCKED"
    foreign.release()
    require(refused, "a second writer was not refused while the lock was held")
    report["steps"]["second_writer_refused"] = True
    report["steps"]["pilot"] = pilot(ctx, shards=1)
    require(report["steps"]["pilot"]["PILOT"] == "PASS", "pilot failed")
    interrupted = generate(ctx, owner="smoke", max_shards=1)
    require(interrupted["status"] == "PARTIAL" and interrupted["skipped_shards"] == 1, "interrupted generation did not resume after the pilot shard")
    full = generate(ctx, owner="smoke")
    require(full["status"] == "COMPLETE" and len(full["computed_shards"]) == 1, "resume recomputed completed shards")
    decision = finalize(ctx)
    require(decision["LOT45_VALID"] == "YES" and decision["DATASET_READY_FOR_G5"] == "NO" and decision["SMOKE_RUN_NOT_SCIENTIFIC"], f"unexpected smoke decision {decision}")
    dataset = read_dataset(out / DATASET_FILE)
    missing = REQUIRED_DATASET_KEYS - set(dataset[0])
    require(len(dataset) == SMOKE_POSITIONS and not missing, f"dataset format: {len(dataset)} rows, missing keys {missing}")
    require(all(abs(sum(r["policy_target"]) - 1) < 1e-9 and sum(r["visit_counts"]) == SMOKE_BUDGET for r in dataset), "policy/visits inconsistent")
    report["steps"]["finalize"] = {k: decision[k] for k in ("LOT45_VALID", "UNIQUE_PHYSICAL_STATES", "POSITIONS_WITH_TERMINAL_Z", "ALL_TARGETS_FINITE", "ALL_TOP_ACTIONS_LEGAL", "PHYSICAL_DUPLICATE_CONFLICTS", "MINIMAX_LABELS_USED", "NEXT_ACTION")}
    bundle = out.parent / f"{SMOKE_DIR_NAME}_results.tar.gz"
    report["steps"]["export"] = export(ctx, bundle)
    report["steps"]["bundle_reverified"] = verify_bundle(bundle, EXPERIMENT_DIR_NAME)
    rerun = generate(ctx, owner="smoke")
    require(rerun["positions_computed"] == 0, "idempotent rerun recomputed positions")
    history = [e["event"] for e in read_json(out / "run_lock_history.json")["events"]]
    report["steps"]["lock_history"] = history
    report.update({"SMOKE_TEST": "PASS", "seconds": round(time.perf_counter() - started, 2), "finished_utc": utc_now(), "scientific_validity": "NONE (pipeline/format check only)"})
    write_json(out / "smoke_test_report.json", report)
    write_json(base.out / "smoke_test_report.json", report)
    return report
