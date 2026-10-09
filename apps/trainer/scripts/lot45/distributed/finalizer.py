"""FINALIZER : finalisation globale exclusive a partir des enregistrements COMPLETED.

Autorisee uniquement si : tous les shards du plan sont COMPLETED, aucun manquant
ni inattendu, une seule tentative acceptee par shard, empreintes scientifiques
identiques, checksums des fichiers references valides, et aucun autre
finalizer actif (verrou exclusif transactionnel).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from colab_drive import sha256
from lot44.artifacts import Lot44FatalError, checked_status, read_json, utc_now, write_json

from ..finalize import finalize
from ..generation import Context
from .config import DISTRIBUTED_DIR, FINALIZER_TTL_S
from .coordinator import COMPLETED, ShardCoordinator
from .migration import check_compatible
from .storage import unreferenced_attempts

REPORT = f"{DISTRIBUTED_DIR}/distributed_report.json"


def checksum_errors(out: Path, refs: list[dict]) -> list[tuple[str, str]]:
    """(chemin, MISSING|INCOMPLETE|CORRUPT|SHA_MISMATCH) des fichiers references invalides."""

    errors = []
    for ref in refs:
        path = out / ref["path"]
        status = checked_status(path)
        if status != "VALID":
            errors.append((ref["path"], status))
        elif sha256(path) != ref["sha256"]:
            errors.append((ref["path"], "SHA_MISMATCH"))
    return errors


def preconditions(ctx: Context, coordinator: ShardCoordinator) -> dict:
    manifest = read_json(ctx.out / "shard_manifest.json")
    info = coordinator.run_info()
    if info is None:
        raise Lot44FatalError("COORDINATOR_RUN_MISSING", f"run {coordinator.run_key} absent from the coordinator")
    check_compatible(ctx, manifest, info)
    records = {r["shard_id"]: r for r in coordinator.shard_records()}
    planned = [s["shard"] for s in manifest["shards"]]
    missing = [s for s in planned if s not in records]
    unexpected = sorted(set(records) - set(planned))
    not_completed = {s: records[s]["status"] for s in planned if s in records and records[s]["status"] != COMPLETED}
    if missing or unexpected or not_completed:
        raise Lot44FatalError("FINALIZE_TOO_EARLY", f"{len(not_completed)} shards not COMPLETED (first {dict(list(not_completed.items())[:5])}), missing {missing[:5]}, unexpected {unexpected[:5]}")
    identity_fp = manifest["search_identity_fingerprint"]
    incompatible = [s for s in planned if records[s]["artifact"]["search_identity_fingerprint"] != identity_fp]
    if incompatible:
        raise Lot44FatalError("MCTS_CONFIG_INCOMPATIBLE", f"accepted artifacts with another search identity: {incompatible[:5]}")
    counts = Counter(f["path"] for s in planned for f in records[s]["artifact"]["files"])
    duplicated = sorted(p for p, n in counts.items() if n > 1)
    if duplicated:
        raise Lot44FatalError("DUPLICATE_ACCEPTED", f"files referenced by several shards: {duplicated[:5]}")
    bad = [e for s in planned for e in checksum_errors(ctx.out, records[s]["artifact"]["files"])]
    if bad and all(kind in ("MISSING", "INCOMPLETE") for _, kind in bad):
        # Fichiers publies par un autre Colab pas encore visibles sur ce montage Drive.
        raise Lot44FatalError("DRIVE_SYNC_PENDING", f"{len(bad)} referenced files not visible yet on this Drive mount (first {bad[:3]}); wait for Drive to sync and rerun finalize")
    if bad:
        raise Lot44FatalError("CHECKSUM_MISMATCH", f"{len(bad)} referenced files fail their checksum: {bad[:5]}")
    return {"manifest": manifest, "records": records, "planned": planned}


def distributed_report(ctx: Context, coordinator: ShardCoordinator, checked: dict) -> dict:
    records, planned = checked["records"], checked["planned"]
    accepted = {s: records[s]["artifact"].get("lease_id") for s in planned}
    events = coordinator.event_records()
    kinds = Counter(e["event"] for e in events)
    return {
        "run_key": coordinator.run_key, "generated_utc": utc_now(), "coordinator": coordinator.config.public(),
        "COMPLETED": sum(records[s]["status"] == COMPLETED for s in planned), "SHARDS_TOTAL": len(planned),
        "DUPLICATE_ACCEPTED": 0, "MISSING": 0, "CHECKSUM_ERRORS": 0,
        "legacy_shards": sum(bool(records[s]["artifact"].get("legacy")) for s in planned),
        "distributed_shards": sum(not records[s]["artifact"].get("legacy") for s in planned),
        "workers": sorted({records[s]["artifact"]["worker_id"] for s in planned}),
        "attempts_per_shard": {s: records[s]["attempt_number"] for s in planned},
        "accepted": {s: {"lease_id": accepted[s], "fencing_token": records[s]["artifact"].get("fencing_token"), "worker_id": records[s]["artifact"]["worker_id"], "files": [f["path"] for f in records[s]["artifact"]["files"]]} for s in planned},
        "unreferenced_attempts": unreferenced_attempts(ctx.out, accepted),
        "events": dict(kinds), "takeovers": kinds.get("TAKEOVER", 0), "rejected_publications": kinds.get("COMPLETE_REJECTED", 0),
    }


def finalize_distributed(ctx: Context, coordinator: ShardCoordinator, *, holder: str) -> dict:
    info = coordinator.run_info()
    if info is None or not info.get("migrated_utc"):
        raise Lot44FatalError("COORDINATOR_RUN_MISSING", f"run {coordinator.run_key} not initialised in the coordinator (run the migrate stage)")
    granted = coordinator.acquire_exclusive("finalizer", holder, FINALIZER_TTL_S)
    try:
        checked = preconditions(ctx, coordinator)
        report = distributed_report(ctx, coordinator, checked)
        write_json(ctx.out / REPORT, report)
        artifacts = {s: checked["records"][s]["artifact"]["files"] for s in checked["planned"]}
        decision = finalize(ctx, artifacts=artifacts)
        return {**decision, "DISTRIBUTED": {k: report[k] for k in ("COMPLETED", "SHARDS_TOTAL", "DUPLICATE_ACCEPTED", "MISSING", "CHECKSUM_ERRORS", "legacy_shards", "distributed_shards", "workers", "takeovers", "rejected_publications")}}
    finally:
        coordinator.release_exclusive("finalizer", granted)
