"""WORKER_EXECUTOR : reserver -> calculer -> valider -> publier -> COMPLETED.

Le calcul est exactement celui du runner historique (``lot44.search.run_search``,
MCTS32768, bruit de racine OFF, graine par position) ; seul le dossier de
sortie change : ``attempts/{shard}/{lease}/``.  Pas de reprise intra-shard :
un shard interrompu est recalcule entierement par la tentative suivante.
"""
from __future__ import annotations

import os
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from run_srn_lot39 import load_model

from lot44.artifacts import ErrorLog, Lot44FatalError, read_json, total_ram_bytes, utc_now, write_checked_json
from lot44.pipeline import choose_device
from lot44.search import estimated_peak_bytes, run_search

from ..generation import Context, shard_records
from ..run_lock import machine_identity
from .config import DISTRIBUTED_DIR, HEARTBEAT_INTERVAL_S, IDLE_POLL_S, MAX_CONSECUTIVE_RECOVERABLE, RUN_MARKER
from .coordinator import FAILED_FATAL, FAILED_RETRYABLE, CoordinatorUnavailable, Lease, ShardCoordinator, TransactionContention
from .heartbeat import LeaseHeartbeat
from .migration import check_compatible
from .storage import attempt_root, file_ref, validate_attempt

# Erreurs scientifiques/deterministes : le shard passe FAILED_FATAL et le worker s'arrete.
FATAL_CODES = frozenset({"NAN_CRITICAL", "ILLEGAL_ACTION", "SIMULATION_COUNT", "MODEL_WEIGHTS_CHANGED", "MCTS_CONFIG_INCOMPATIBLE", "SPLIT_MODIFIED", "TARGET_INVALID"})
# Le bail n'appartient plus a ce worker : rien a publier, on passe au shard suivant.
LOST_CODES = frozenset({"LEASE_LOST", "FENCING_REJECTED"})
RECOVERABLE = (OSError, CoordinatorUnavailable, TransactionContention, torch.cuda.OutOfMemoryError)
CONCURRENCY_LADDER = (128, 64, 32, 16, 8, 4, 2, 1)


def default_worker_id() -> str:
    identity = machine_identity()
    return f"{'colab' if identity['colab'] else 'local'}-{socket.gethostname()[:12]}-{uuid.uuid4().hex[:8]}"


def worker_dir(out: Path, worker_id: str) -> Path:
    return out / DISTRIBUTED_DIR / "workers" / worker_id


def auto_concurrency(budget: int, shard_size: int, ram_bytes: int | None, *, ram_fraction: float = 0.8) -> int:
    """Plus grande concurrence locale dont le pic RAM estime tient dans ``ram_fraction`` de la RAM."""

    if ram_bytes is None:
        raise Lot44FatalError("RAM_UNKNOWN", "cannot size the MCTS concurrency: total RAM unavailable; pass --concurrency")
    for value in CONCURRENCY_LADDER:
        if value <= shard_size and estimated_peak_bytes(budget, value) <= ram_fraction * ram_bytes:
            return value
    raise Lot44FatalError("RAM_INSUFFICIENT", f"{ram_bytes / 1e9:.1f} GB RAM cannot hold one MCTS{budget} tree")


@dataclass
class WorkerSettings:
    worker_id: str
    concurrency: int
    heartbeat_s: float = HEARTBEAT_INTERVAL_S
    idle_poll_s: float = IDLE_POLL_S
    max_idle_polls: int | None = None
    max_shards: int | None = None
    on_claimed: Callable[[Lease], None] | None = None


class Worker:
    def __init__(self, ctx: Context, coordinator: ShardCoordinator, settings: WorkerSettings, *, log: Callable[[str], None] = print) -> None:
        self.ctx = ctx
        self.coordinator = coordinator
        self.settings = settings
        self.log = log
        self.run_id = uuid.uuid4().hex
        self.dir = worker_dir(ctx.out, settings.worker_id)
        self.errors = ErrorLog(self.dir / "errors.jsonl")
        self.manifest = read_json(ctx.out / "shard_manifest.json")
        self.plan = {s["shard"]: s for s in self.manifest["shards"]}
        self.selection = {r["fingerprint"]: r for r in ctx.selection()}
        self.device: torch.device | None = None
        self._model = None

    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = load_model(self.device)
        return self._model

    def startup(self) -> dict:
        if not (self.ctx.out / RUN_MARKER).is_file():
            raise Lot44FatalError("NOT_MIGRATED", f"{RUN_MARKER} absent: run the migrate stage first")
        info = self.coordinator.run_info()
        if info is None or not info.get("migrated_utc"):
            raise Lot44FatalError("COORDINATOR_RUN_MISSING", f"run {self.coordinator.run_key} not initialised in the coordinator")
        check_compatible(self.ctx, self.manifest, info)
        self.device = choose_device(self.ctx.device_name)
        registration = {**machine_identity(), "device": str(self.device), "concurrency": self.settings.concurrency, "run_id": self.run_id, "started_utc": utc_now(), "code_commit": self.ctx.fingerprints()["code_commit"]}
        self.coordinator.register_worker(self.settings.worker_id, registration)
        return registration

    def execute(self, lease: Lease, heartbeat: LeaseHeartbeat) -> dict:
        if lease.shard_id not in self.plan:
            raise Lot44FatalError("SPLIT_MODIFIED", f"{lease.shard_id} absent from shard_manifest.json")
        shard = self.plan[lease.shard_id]
        records = shard_records(self.ctx, shard)
        root = attempt_root(self.ctx.out, lease.shard_id, lease.lease_id)
        identity = self.ctx.identity()
        result = run_search(root, partition=lease.shard_id, records=records, identity=identity, concurrency=min(self.settings.concurrency, len(records)), model_loader=self.model, device=self.device, errors=self.errors, log=self.log, before_write=heartbeat.assert_valid)
        if result["status"] != "COMPLETE":
            raise Lot44FatalError("ARTIFACT_INCOMPLETE", f"{lease.shard_id}: search status {result['status']}")
        heartbeat.assert_valid()
        artifact = validate_attempt(self.ctx.out, root, lease.shard_id, self.ctx.budget, identity.fingerprint(), records, self.selection)
        fps = self.ctx.fingerprints()
        attempt_manifest = {
            "shard_id": lease.shard_id, "worker_id": lease.worker_id, "run_id": lease.run_id, "lease_id": lease.lease_id, "fencing_token": lease.fencing_token,
            "attempt_number": lease.attempt_number, "device": str(self.device), "concurrency": min(self.settings.concurrency, len(records)),
            "checkpoint_fingerprints": fps["model"], "mcts_identity_fingerprint": identity.fingerprint(), "mcts_identity": identity.payload(),
            "engine_fingerprint": fps["engine"], "dataset_fingerprint": {"selection_sha256": self.manifest["selection_sha256"], "shard_input_sha256": shard["input_sha256"]},
            "seed": self.ctx.seed, "code_commit": fps["code_commit"], "files": artifact["files"], "written_utc": utc_now(),
        }
        write_checked_json(root / "attempt_manifest.json", attempt_manifest)
        heartbeat.assert_valid()
        return {**artifact, "attempt_manifest": file_ref(self.ctx.out, root / "attempt_manifest.json"), "device": str(self.device), "engine_fingerprint": fps["engine"], "checkpoint_fingerprints": fps["model"], "seed": self.ctx.seed}

    def _release(self, lease: Lease, status: str, reason: str, *, retry_after_s: float) -> None:
        try:
            released = self.coordinator.release(lease, status, reason, retry_after_s=retry_after_s)
            self.log(f"[Lot45][worker] {lease.shard_id} -> {status if released else 'not owner anymore'} ({reason[:120]})")
        except (CoordinatorUnavailable, TransactionContention) as exc:
            self.errors.record(stage=f"release/{lease.shard_id}", exc=exc, shard=lease.shard_id, fatal=False)
            self.log(f"[Lot45][worker] {lease.shard_id}: release impossible ({exc}); the lease will expire after its TTL")

    def run(self) -> dict:
        registration = self.startup()
        summary = {"worker_id": self.settings.worker_id, "run_id": self.run_id, "device": registration["device"], "completed": [], "lost": [], "released_retryable": [], "failed_fatal": [], "stop_reason": None}
        consecutive, idle = 0, 0
        while True:
            if self.settings.max_shards is not None and len(summary["completed"]) >= self.settings.max_shards:
                summary["stop_reason"] = "MAX_SHARDS"
                break
            try:
                lease = self.coordinator.claim_next_shard(self.settings.worker_id, self.run_id)
            except (CoordinatorUnavailable, TransactionContention) as exc:
                consecutive += 1
                self.errors.record(stage="claim", exc=exc, fatal=False, retry_count=consecutive)
                if consecutive > MAX_CONSECUTIVE_RECOVERABLE:
                    summary["stop_reason"] = "COORDINATOR_UNAVAILABLE"
                    break
                time.sleep(self.settings.idle_poll_s)
                continue
            if lease is None:
                progress = self.coordinator.progress()
                if progress["all_settled"]:
                    summary["stop_reason"] = "ALL_SHARDS_SETTLED"
                    summary["progress"] = progress
                    break
                idle += 1
                if self.settings.max_idle_polls is not None and idle > self.settings.max_idle_polls:
                    summary["stop_reason"] = "IDLE_LIMIT"
                    break
                self.log(f"[Lot45][worker] aucun shard disponible ({progress['remaining']} en cours ailleurs) ; nouvelle tentative dans {self.settings.idle_poll_s:.0f}s")
                time.sleep(self.settings.idle_poll_s)
                continue
            idle = 0
            self.log(f"[Lot45][worker] {self.settings.worker_id} CLAIM {lease.shard_id} token {lease.fencing_token} attempt {lease.attempt_number} lease {lease.lease_id[:8]}")
            if self.settings.on_claimed is not None:
                self.settings.on_claimed(lease)
            heartbeat = LeaseHeartbeat(self.coordinator, lease, interval_s=self.settings.heartbeat_s, log=self.log).start()
            started = time.time()
            try:
                artifact = self.execute(lease, heartbeat)
                heartbeat.assert_valid()
                outcome = self.coordinator.complete(heartbeat.lease, artifact)
                summary["completed"].append(lease.shard_id)
                consecutive = 0
                self.log(f"[Lot45][worker] {lease.shard_id} {outcome['status']} en {time.time() - started:.0f}s (token {lease.fencing_token})")
            except Lot44FatalError as exc:
                self.errors.record(stage=f"shard/{lease.shard_id}", exc=exc, shard=lease.shard_id, fatal=exc.code not in LOST_CODES)
                if exc.code in LOST_CODES:
                    summary["lost"].append(lease.shard_id)
                    self.log(f"[Lot45][worker] {lease.shard_id}: {exc.code}, resultat non publie")
                    continue
                if exc.code in FATAL_CODES:
                    self._release(heartbeat.lease, FAILED_FATAL, str(exc), retry_after_s=0.0)
                    summary["failed_fatal"].append(lease.shard_id)
                    summary["stop_reason"] = f"FATAL:{exc.code}"
                    break
                self._release(heartbeat.lease, FAILED_RETRYABLE, str(exc), retry_after_s=0.0)
                summary["released_retryable"].append(lease.shard_id)
                summary["stop_reason"] = f"UNEXPECTED:{exc.code}"
                break
            except RECOVERABLE as exc:
                consecutive += 1
                self.errors.record(stage=f"shard/{lease.shard_id}", exc=exc, shard=lease.shard_id, fatal=False, retry_count=consecutive)
                self._release(heartbeat.lease, FAILED_RETRYABLE, f"{type(exc).__name__}: {exc}", retry_after_s=60.0)
                summary["released_retryable"].append(lease.shard_id)
                if consecutive > MAX_CONSECUTIVE_RECOVERABLE:
                    summary["stop_reason"] = "TOO_MANY_RECOVERABLE_ERRORS"
                    break
            except BaseException as exc:
                # Interruption (Colab, Ctrl-C) ou erreur inattendue : le shard est rendu
                # immediatement disponible puis l'exception est propagee.
                self.errors.record(stage=f"shard/{lease.shard_id}", exc=exc, shard=lease.shard_id)
                self._release(heartbeat.lease, FAILED_RETRYABLE, f"{type(exc).__name__}: {exc}", retry_after_s=0.0)
                raise
            finally:
                heartbeat.stop()
        summary["coordinator_ops"] = self.coordinator.ops
        self.coordinator.update_worker(self.settings.worker_id, {"stopped_utc": utc_now(), "stop_reason": summary["stop_reason"], "completed": summary["completed"]})
        return summary


def run_worker(ctx: Context, coordinator: ShardCoordinator, settings: WorkerSettings, *, log: Callable[[str], None] = print) -> dict:
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    return Worker(ctx, coordinator, settings, log=log).run()


def resolve_concurrency(ctx: Context, requested: int | None) -> int:
    return requested if requested else auto_concurrency(ctx.budget, ctx.shard_size, total_ram_bytes())
