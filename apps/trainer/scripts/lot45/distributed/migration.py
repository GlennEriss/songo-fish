"""Audit des shards existants et migration vers le coordinateur.

Classes d'audit : COMPLETED_VALID, PENDING, RUNNING, FAILED, INVALID.  Seuls
les shards COMPLETED_VALID sont importes COMPLETED, avec la reference
immuable de leurs fichiers actuels (chemins + sha256) ; aucun fichier n'est
deplace, reecrit ni supprime.  Les autres sont importes PENDING et seront
recalcules dans ``attempts/`` (jamais par-dessus l'ancien fichier).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from colab_drive import sha256
from lot44.artifacts import Lot44FatalError, canonical_hash, checked_status, read_json, read_jsonl, utc_now, write_json
from lot44.search import shard_dir

from ..dataset import validate_row
from ..generation import Context, shard_records
from ..run_lock import RunLock
from .config import DISTRIBUTED_DIR, MIGRATION_TTL_S, RUN_MARKER
from .coordinator import COMPLETED, PENDING, ShardCoordinator
from .storage import file_ref, shard_files

BACKUP_FILES = ("shard_manifest.json", "stage_state.json", "run_lock.json", "run_lock_history.json", "compute_plan.json", "pilot_report.json", "execution_manifest.json", "errors.jsonl", "heartbeat.json")


def run_key(manifest: dict) -> str:
    return "lot45-" + canonical_hash({k: manifest[k] for k in ("selection_sha256", "target_positions", "shard_size", "budget", "search_identity_fingerprint")})[:16]


def run_meta(ctx: Context, manifest: dict) -> dict:
    fps = ctx.fingerprints()
    return {
        "lot": 45, "run_key": run_key(manifest), "search_identity_fingerprint": manifest["search_identity_fingerprint"],
        "model_fingerprints": fps["model"], "engine_fingerprint": fps["engine"], "seed": ctx.seed, "budget": ctx.budget,
        "selection_sha256": manifest["selection_sha256"], "target_positions": manifest["target_positions"], "shard_size": manifest["shard_size"],
    }


def check_compatible(ctx: Context, manifest: dict, info: dict) -> None:
    """Le protocole scientifique d'un worker doit etre identique a celui du run."""

    mine = run_meta(ctx, manifest)
    differences = {k: (info.get(k), v) for k, v in mine.items() if info.get(k) != v}
    if ctx.identity().fingerprint() != manifest["search_identity_fingerprint"]:
        differences["local_search_identity"] = (manifest["search_identity_fingerprint"], ctx.identity().fingerprint())
    if differences:
        raise Lot44FatalError("MCTS_CONFIG_INCOMPATIBLE", f"worker protocol differs from the coordinator run: {differences}")


def _legacy_failures(out: Path) -> set[str]:
    path = out / "errors.jsonl"
    if not path.is_file() or not path.stat().st_size:
        return set()
    return {str(e["stage"]).split("/")[1] for e in read_jsonl(path) if str(e.get("stage", "")).startswith("search/shard_")}


def classify_shards(ctx: Context, *, legacy_writer_live: bool) -> list[dict]:
    """Classe chaque shard du plan d'apres les fichiers du runner historique."""

    manifest = read_json(ctx.out / "shard_manifest.json")
    identity_fp = manifest["search_identity_fingerprint"]
    selection = {r["fingerprint"]: r for r in ctx.selection()}
    failures = _legacy_failures(ctx.out)
    rows_out, first_open = [], None
    for index, shard in enumerate(manifest["shards"]):
        records = shard_records(ctx, shard)
        expected = {r["fingerprint"] for r in records}
        directory = shard_dir(ctx.out, shard["shard"], ctx.budget)
        entry = {"shard_id": shard["shard"], "index": index, "input_sha256": shard["input_sha256"], "count": shard["count"], "files": [], "detail": None}
        statuses = {p.name: checked_status(p) for p in shard_files(directory)}
        covered: dict[str, dict] = {}
        problem = None
        for name, status in statuses.items():
            if status == "CORRUPT":
                problem = f"{name}: checksum mismatch"
                break
            if status != "VALID":
                continue  # INCOMPLETE : ecriture interrompue avant le sidecar ; jamais referencee
            payload = read_json(directory / name)
            if payload.get("search_identity_fingerprint") != identity_fp or payload.get("status") != "COMPLETE":
                problem = f"{name}: identity/status mismatch"
                break
            for row in payload["rows"]:
                fp = row["state_fingerprint"]
                if fp not in expected or fp in covered:
                    problem = f"{name}: position {fp[:12]} outside the shard or duplicated"
                    break
                bad = validate_row(row, selection[fp], ctx.budget)
                if bad:
                    problem = f"{name}: {fp[:12]} {bad}"
                    break
                covered[fp] = row
            if problem:
                break
            entry["files"].append(file_ref(ctx.out, directory / name))
        if problem:
            entry.update(status="INVALID", detail=problem, files=[])
        elif set(covered) == expected:
            entry["status"] = "COMPLETED_VALID"
        else:
            if first_open is None:
                first_open = index
            entry["detail"] = f"{len(covered)}/{len(expected)} positions; incomplete files {[n for n, s in statuses.items() if s == 'INCOMPLETE']}"
            entry["files"] = []
            entry["status"] = "RUNNING" if legacy_writer_live and index == first_open else ("FAILED" if shard["shard"] in failures else "PENDING")
        rows_out.append(entry)
    return rows_out


def audit(ctx: Context) -> dict:
    """Audit en lecture seule (aucune ecriture)."""

    reader = RunLock(ctx.out, owner="audit", settle_s=0)
    lock = reader.current()
    live = bool(lock and not lock.get("released") and reader.age_s(lock) < reader.stale_after_s)
    shards = classify_shards(ctx, legacy_writer_live=live)
    counts = {s: sum(r["status"] == s for r in shards) for s in ("COMPLETED_VALID", "PENDING", "RUNNING", "FAILED", "INVALID")}
    return {"timestamp_utc": utc_now(), "legacy_lock": lock, "legacy_writer_live": live, "counts": counts, "total": len(shards), "shards": shards}


def backup(out: Path) -> dict:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = out / DISTRIBUTED_DIR / "migration" / f"backup_{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name in BACKUP_FILES:
        source = out / name
        if source.is_file():
            shutil.copy2(source, target / name)
            if sha256(target / name) != sha256(source):
                raise Lot44FatalError("BACKUP_FAILED", f"backup copy of {name} differs from the source")
            copied[name] = sha256(target / name)
    write_json(target / "backup_manifest.json", {"created_utc": utc_now(), "files": copied})
    return {"directory": str(target.relative_to(out)), "files": copied}


def migrate(ctx: Context, coordinator: ShardCoordinator, *, holder: str, takeover_stale_legacy_lock: bool = False) -> dict:
    """Migration exclusive et idempotente vers le coordinateur.

    1. verrou historique (refuse si un ancien runner ecrit encore) ;
    2. verrou exclusif ``migration`` (transaction Firestore) ;
    3. sauvegarde des manifests ; audit ; import (sans jamais ecraser) ;
    4. marqueur ``distributed/run.json`` : le runner historique refuse ensuite d'ecrire.
    """

    manifest = read_json(ctx.out / "shard_manifest.json")
    if manifest["search_identity_fingerprint"] != ctx.identity().fingerprint():
        raise Lot44FatalError("MCTS_CONFIG_INCOMPATIBLE", "local search identity differs from shard_manifest.json")
    if coordinator.run_key != run_key(manifest):
        raise Lot44FatalError("COORDINATOR_RUN_MISMATCH", f"coordinator run {coordinator.run_key} != {run_key(manifest)}")
    marker = ctx.out / RUN_MARKER
    info = coordinator.run_info()
    if marker.is_file() and info is not None and info.get("migrated_utc"):
        check_compatible(ctx, manifest, info)
        return {"status": "ALREADY_MIGRATED", "run_key": coordinator.run_key, "migrated_utc": info["migrated_utc"], "progress": coordinator.progress()}
    legacy = RunLock(ctx.out, owner="distributed-migration", settle_s=ctx.lock_settle_s)
    legacy.acquire(takeover_stale=takeover_stale_legacy_lock, reason="migration to the distributed coordinator")
    granted = None
    try:
        granted = coordinator.acquire_exclusive("migration", holder, MIGRATION_TTL_S)
        saved = backup(ctx.out)
        shards = classify_shards(ctx, legacy_writer_live=False)
        counts = {s: sum(r["status"] == s for r in shards) for s in ("COMPLETED_VALID", "PENDING", "RUNNING", "FAILED", "INVALID")}
        report = {"timestamp_utc": utc_now(), "run_key": coordinator.run_key, "holder": holder, "backup": saved, "counts": counts, "shards": shards, "coordinator": coordinator.config.public()}
        write_json(ctx.out / DISTRIBUTED_DIR / "migration" / f"migration_report_{saved['directory'].rsplit('_', 1)[-1]}.json", report)
        imported = [{
            "shard_id": s["shard_id"], "index": s["index"], "input_sha256": s["input_sha256"], "count": s["count"],
            "status": COMPLETED if s["status"] == "COMPLETED_VALID" else PENDING, "legacy_classification": s["status"],
            "artifact": {"files": s["files"], "rows": s["count"], "search_identity_fingerprint": manifest["search_identity_fingerprint"], "attempt_dir": None, "legacy": True, "lease_id": None, "fencing_token": 0, "worker_id": "legacy-runner", "attempt_number": 0} if s["status"] == "COMPLETED_VALID" else None,
        } for s in shards]
        meta = {**run_meta(ctx, manifest), "migrated_utc": utc_now(), "migration_holder": holder, "scratch": False}
        result = coordinator.import_shards(meta, imported)
        write_json(marker, {**meta, "coordinator": coordinator.config.public(), "imported": len(result["created"]), "kept": len(result["kept"]), "counts": counts})
        return {"status": "MIGRATED", "run_key": coordinator.run_key, "counts": counts, "created": len(result["created"]), "kept": len(result["kept"]), "backup": saved["directory"], "progress": coordinator.progress()}
    finally:
        if granted is not None:
            coordinator.release_exclusive("migration", granted)
        legacy.release()
