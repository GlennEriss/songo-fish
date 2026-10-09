"""Plan de shards, generation MCTS32768 uniforme (lot44.search.run_search),
pilote et plan de calcul. Toute ecriture passe par le verrou mono-ecrivain."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from run_srn_colab_benchmark import engine_fingerprint, git_commit, pool_fingerprints
from run_srn_lot39 import load_model

from colab_drive import sha256
from lot44.artifacts import ErrorLog, Lot44FatalError, canonical_hash, checked_status, read_json, update_stage_state, utc_now, write_json
from lot44.pipeline import choose_device
from lot44.search import SearchIdentity, estimated_peak_bytes, load_completed, run_search, shard_dir

from .config import BUDGET, CANDIDATE_TIERS, DISTRIBUTED_MARKER, LOCK_SETTLE_S, SHARD_SIZE
from .run_lock import RunLock, machine_identity
from .dataset import validate_row
from .selection import read_selection

# Debit mesure Lot44 (T4, MCTS32768, concurrence 128) : 3771-4058 sims/s.
MEASURED_T4_SIMS_PER_S_32768 = 3900.0
BYTES_PER_POSITION_ESTIMATE = 1400


@dataclass
class Context:
    out: Path
    seed: int
    device_name: str
    selection_file: Path
    original_positions: Path
    lot44: Path | None
    budget: int = BUDGET
    shard_size: int = SHARD_SIZE
    concurrency: int = 128
    target_positions: int | None = None
    max_shards: int | None = None
    smoke: bool = False
    takeover_stale_lock: bool = False
    lock_settle_s: float = LOCK_SETTLE_S
    log: Callable[[str], None] = print
    _fingerprints: dict = field(default_factory=dict)
    _selection: list | None = None

    @property
    def errors(self) -> ErrorLog:
        return ErrorLog(self.out / "errors.jsonl")

    def fingerprints(self) -> dict:
        if not self._fingerprints:
            self._fingerprints = {"model": pool_fingerprints(), "engine": engine_fingerprint(), "code_commit": git_commit()}
        return self._fingerprints

    def identity(self) -> SearchIdentity:
        fps = self.fingerprints()
        return SearchIdentity(budget=self.budget, seed=self.seed, model_fingerprints=fps["model"], engine_fingerprint=fps["engine"])

    def selection(self) -> list[dict]:
        if self._selection is None:
            self._selection = read_selection(self.selection_file)
        return self._selection


def partition_name(index: int) -> str:
    return f"shard_{index:05d}"


def validate_lot44(lot44: Path) -> dict:
    decision = read_json(lot44 / "decision.json")
    expected = {"LOT44_VALID": "YES", "ROUTER_VALIDATED": "INCONCLUSIVE", "UNIFORM_32768_DEFENSIBLE": "YES", "EARLY_STOP_8192_ALLOWED": "NO", "MCTS_RESUME_CURRENTLY_SUPPORTED": "NO", "NEXT_ACTION": "LOT45_G5_UNIFORM_32768_TARGET_GENERATION"}
    checks = {k: decision.get(k) == v for k, v in expected.items()}
    return {"checks": checks, "VALID": "YES" if all(checks.values()) else "NO", "router_used_in_lot45": False, "positions_routed_to_65536": 0, "early_stop_8192": False, "lot44_decision_excerpt": {k: decision.get(k) for k in (*expected, "OOS_POSITIVE_COUNT", "TEST_SIZE", "OOS_HIGH_SEVERITY_LATE_BIFURCATIONS")}}


def shard_plan(rows: list[dict], target: int, shard_size: int) -> list[dict]:
    plan = []
    for index in range(math.ceil(target / shard_size)):
        chunk = rows[index * shard_size: min(target, (index + 1) * shard_size)]
        plan.append({"shard": partition_name(index), "start": chunk[0]["order"], "end": chunk[-1]["order"], "count": len(chunk), "input_sha256": canonical_hash([r["fingerprint"] for r in chunk])})
    return plan


def measured_t4_rate(lot44: Path | None) -> tuple[float, str]:
    if lot44 is not None:
        rates = []
        for part in ("calibration", "test"):
            for p in (lot44 / "search" / part / "32768").glob("shard_*.json"):
                if p.name.endswith(".sha256.json"):
                    continue
                payload = read_json(p)
                if payload.get("device") == "cuda" and payload["concurrency"] >= 64:
                    rates.append(payload["simulations_per_second"])
        if rates:
            return sum(rates) / len(rates), f"Lot44 T4 MCTS32768 shards (n={len(rates)})"
    return MEASURED_T4_SIMS_PER_S_32768, "Lot44 T4 MCTS32768 log (3771-4058 sims/s)"


def compute_plan(ctx: Context, available: int) -> dict:
    t4_rate, t4_source = measured_t4_rate(ctx.lot44)
    pilot = read_json(ctx.out / "pilot_report.json") if (ctx.out / "pilot_report.json").is_file() else None
    pilot_rate = pilot["simulations_per_second"] if pilot else None
    bytes_per_position = pilot["artifact_bytes_per_position"] if pilot else BYTES_PER_POSITION_ESTIMATE
    tiers = sorted({*CANDIDATE_TIERS, *([ctx.target_positions] if ctx.target_positions else [])})
    rows = []
    for n in tiers:
        sims = n * ctx.budget
        rows.append({
            "positions": n, "available": n <= available, "simulations": sims,
            "estimated_t4_hours": sims / t4_rate / 3600,
            "estimated_pilot_device_hours": sims / pilot_rate / 3600 if pilot_rate else None,
            "estimated_storage_mb": n * bytes_per_position / 1e6, "shards": math.ceil(n / ctx.shard_size),
        })
    return {
        "budget": ctx.budget, "shard_size": ctx.shard_size, "selection_available": available,
        "t4_simulations_per_second": t4_rate, "t4_rate_source": t4_source,
        "pilot_device": pilot["device"] if pilot else None, "pilot_simulations_per_second": pilot_rate,
        "bytes_per_position": bytes_per_position, "bytes_source": "pilot artifacts" if pilot else "estimate before pilot",
        "candidates": rows, "chosen_target_positions": ctx.target_positions,
        "peak_ram_estimate_gb": estimated_peak_bytes(ctx.budget, ctx.concurrency) / 1e9,
    }


def refuse_if_distributed(ctx: Context, stage: str) -> None:
    """Apres migration, seul le coordinateur distribue attribue les shards."""

    if (ctx.out / DISTRIBUTED_MARKER).is_file():
        raise Lot44FatalError("MIGRATED_TO_DISTRIBUTED", f"{ctx.out.name} is coordinated by the distributed runner ({DISTRIBUTED_MARKER}); the single-writer '{stage}' stage is disabled — use --stage worker")


def prepare(ctx: Context) -> dict:
    refuse_if_distributed(ctx, "prepare")
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "errors.jsonl").touch()
    if not ctx.smoke:
        validation = validate_lot44(ctx.lot44)
        write_json(ctx.out / "lot44_input_validation.json", validation)
        if validation["VALID"] != "YES":
            raise Lot44FatalError("LOT44_INPUT_INVALID", f"unexpected Lot44 decision: {validation['checks']}")
    rows = ctx.selection()
    target = ctx.target_positions or len(rows)
    if target > len(rows):
        raise Lot44FatalError("PLAN_INVALID", f"target {target} exceeds the selection ({len(rows)})")
    plan = shard_plan(rows, target, ctx.shard_size)
    path = ctx.out / "shard_manifest.json"
    if path.is_file():
        previous = read_json(path)
        if previous["shard_size"] != ctx.shard_size or previous["selection_sha256"] != canonical_hash([r["fingerprint"] for r in rows]) or previous["search_identity_fingerprint"] != ctx.identity().fingerprint():
            raise Lot44FatalError("SPLIT_MODIFIED", "shard plan incompatible with existing shard_manifest.json (selection, shard size or search identity changed)")
        common = min(len(previous["shards"]), len(plan))
        if previous["shards"][:common] != plan[:common]:
            raise Lot44FatalError("SPLIT_MODIFIED", "shard prefix changed")
    write_json(path, {"selection_sha256": canonical_hash([r["fingerprint"] for r in rows]), "target_positions": target, "shard_size": ctx.shard_size, "budget": ctx.budget, "search_identity": ctx.identity().payload(), "search_identity_fingerprint": ctx.identity().fingerprint(), "shards": plan, "updated_utc": utc_now()})
    write_json(ctx.out / "compute_plan.json", compute_plan(ctx, len(rows)))
    update_stage_state(ctx.out, "prepare", {"status": "COMPLETE", "target_positions": target, "shards": len(plan)})
    return {"target_positions": target, "shards": len(plan), "selection": len(rows)}


def shard_records(ctx: Context, shard: dict) -> list[dict]:
    rows = ctx.selection()[shard["start"]: shard["end"] + 1]
    if canonical_hash([r["fingerprint"] for r in rows]) != shard["input_sha256"]:
        raise Lot44FatalError("SPLIT_MODIFIED", f"{shard['shard']} inputs changed")
    return [{"fingerprint": r["fingerprint"], "state": r["state"], "game_id": r["split_group"]} for r in rows]


def shard_complete(ctx: Context, shard: dict) -> bool:
    records = shard_records(ctx, shard)
    done, _ = load_completed(shard_dir(ctx.out, shard["shard"], ctx.budget), ctx.identity().fingerprint(), {r["fingerprint"] for r in records}, remove_incomplete=False)
    return len(done) == len(records)


def generate(ctx: Context, *, owner: str, max_shards: int | None = None, first_shards_only: int | None = None) -> dict:
    """Calcule les shards manquants du plan sous verrou mono-ecrivain.

    ``max_shards`` borne le nombre de shards NOUVELLEMENT calcules (session
    decoupee) ; ``first_shards_only`` restreint le parcours aux N premiers.
    """

    refuse_if_distributed(ctx, "generate")
    manifest = read_json(ctx.out / "shard_manifest.json")
    if manifest["search_identity_fingerprint"] != ctx.identity().fingerprint():
        raise Lot44FatalError("MCTS_CONFIG_INCOMPATIBLE", "search identity differs from shard_manifest.json")
    shards = manifest["shards"][:first_shards_only]
    if all(shard_complete(ctx, s) for s in shards):
        ctx.log("[Lot45][generate] SKIP: all planned shards already complete")
        done = sum(shard_complete(ctx, s) for s in manifest["shards"])
        return {"device": None, "computed_shards": [], "skipped_shards": len(shards), "positions_computed": 0, "shards_complete": done, "shards_total": len(manifest["shards"]), "status": "COMPLETE" if done == len(manifest["shards"]) else "PARTIAL"}
    device = choose_device(ctx.device_name)
    lock = RunLock(ctx.out, owner=owner, settle_s=ctx.lock_settle_s)
    lock.acquire(takeover_stale=ctx.takeover_stale_lock, reason=f"stage {owner}")
    lock.start_heartbeat()
    model_cache: dict = {}

    def loader() -> torch.nn.Module:
        if "model" not in model_cache:
            model_cache["model"] = load_model(device)
        return model_cache["model"]

    summary = {"device": str(device), "lock_run_id": lock.run_id, "machine": machine_identity()["machine_id"], "computed_shards": [], "skipped_shards": 0, "positions_computed": 0}
    started = time.time()
    try:
        for shard in shards:
            if max_shards is not None and len(summary["computed_shards"]) >= max_shards:
                break
            records = shard_records(ctx, shard)
            result = run_search(ctx.out, partition=shard["shard"], records=records, identity=ctx.identity(), concurrency=min(ctx.concurrency, len(records)), model_loader=loader, device=device, errors=ctx.errors, log=ctx.log, before_write=lock.assert_owner)
            if result["computed"]:
                summary["computed_shards"].append(shard["shard"])
                summary["positions_computed"] += result["computed"]
            else:
                summary["skipped_shards"] += 1
    finally:
        lock.release()
    done = sum(shard_complete(ctx, s) for s in manifest["shards"])
    summary.update({"elapsed_s": time.time() - started, "shards_complete": done, "shards_total": len(manifest["shards"]), "status": "COMPLETE" if done == len(manifest["shards"]) else "PARTIAL"})
    update_stage_state(ctx.out, "generate", {"status": summary["status"], "shards_complete": done, "shards_total": len(manifest["shards"])})
    return summary


def pilot(ctx: Context, *, shards: int) -> dict:
    """Premiers shards du plan, sous le protocole final (reutilisables, section 38)."""

    first = generate(ctx, owner="pilot", first_shards_only=shards)
    rerun = generate(ctx, owner="pilot-resume-check", first_shards_only=shards)
    manifest = read_json(ctx.out / "shard_manifest.json")
    files, payloads = [], []
    for shard in manifest["shards"][:shards]:
        for p in sorted(shard_dir(ctx.out, shard["shard"], ctx.budget).glob("shard_*.json")):
            if p.name.endswith(".sha256.json"):
                continue
            if checked_status(p) != "VALID":
                raise Lot44FatalError("CHECKSUM_MISMATCH", f"pilot shard {p} invalid")
            files.append(p)
            payloads.append(read_json(p))
    rows = [r for p in payloads for r in p["rows"]]
    selection = {r["fingerprint"]: r for r in ctx.selection()}
    problems = [problem for r in rows for problem in validate_row(r, selection[r["state_fingerprint"]], ctx.budget)]
    wall = sum(p["wall_time_s"] for p in payloads)
    report = {
        "PILOT": "PASS" if not problems and rows else "FAIL", "shards": shards, "positions": len(rows), "device": first["device"],
        "wall_time_s": wall, "seconds_per_position": wall / len(rows) if rows else None,
        "simulations_per_second": len(rows) * ctx.budget / wall if wall else None,
        "peak_rss_bytes": max(p["memory"]["peak_rss_bytes"] for p in payloads) if payloads else None,
        "gpu_peak_bytes": max(p["memory"].get("gpu_peak_bytes", 0) for p in payloads) if payloads else None,
        "artifact_bytes_per_position": sum(f.stat().st_size for f in files) / len(rows) if rows else None,
        "artifact_sha256": {f.name: sha256(f) for f in files},
        "validation_problems": problems[:20], "resume_rerun_computed_positions": rerun["positions_computed"],
        "reusable_as_training_data": not problems, "reuse_rule": "pilot shards are the first shards of the final plan, computed with the final search identity",
        "concurrency": ctx.concurrency, "timestamp_utc": utc_now(),
    }
    if rerun["positions_computed"] != 0:
        report["PILOT"] = "FAIL"
    write_json(ctx.out / "pilot_report.json", report)
    write_json(ctx.out / "compute_plan.json", compute_plan(ctx, len(ctx.selection())))
    return report
