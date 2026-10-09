"""Preflight distribue (un rapport par worker) : verifications reelles du
coordinateur dans un run jetable ``preflight-{worker_id}`` supprime ensuite.

Ordre : PROJECT_IMPORTS, CHECKPOINT, CUDA, DRIVE, COORDINATOR_CONNECTION,
COORDINATOR_ATOMICITY_TEST, WORKER_REGISTRATION, SHARD_CLAIM, HEARTBEAT,
ARTIFACT_WRITE, FENCING, RESUME.  Un preflight echoue interdit le calcul reel.
"""
from __future__ import annotations

import dataclasses
import importlib
import shutil
import threading
import time
from typing import Callable

import torch

from colab_drive import sha256
from lot44.artifacts import Lot44FatalError, checked_status, read_json, utc_now, write_checked_json, write_json

from ..generation import Context
from ..preflight import Lot45Preflight
from .config import MAX_CLOCK_SKEW_S, CoordinatorConfig
from .coordinator import COMPLETED, PENDING, Lease, ShardCoordinator, make_client
from .heartbeat import LeaseHeartbeat
from .migration import check_compatible, run_key
from .worker import worker_dir

ORDER = ("PROJECT_IMPORTS", "CHECKPOINT", "CUDA", "DRIVE", "COORDINATOR_CONNECTION", "COORDINATOR_ATOMICITY_TEST", "WORKER_REGISTRATION", "SHARD_CLAIM", "HEARTBEAT", "ARTIFACT_WRITE", "FENCING", "RESUME")
RACERS = 8


def scratch_shard(index: int) -> dict:
    return {"shard_id": f"shard_{index:05d}", "index": index, "input_sha256": f"preflight-{index}", "count": 1, "status": PENDING}


class DistributedPreflight(Lot45Preflight):
    def __init__(self, ctx: Context, config: CoordinatorConfig, worker_id: str, *, coordinator_factory: Callable[..., ShardCoordinator] | None = None) -> None:
        self.base_out = ctx.out
        self.worker_id = worker_id
        self.config = config
        self.factory = coordinator_factory or (lambda run, **kw: ShardCoordinator(make_client(config), config, run, **kw))
        super().__init__(dataclasses.replace(ctx, out=worker_dir(ctx.out, worker_id)), require_inputs=True)
        self.scratch_key = f"preflight-{worker_id}"
        self.coordinator: ShardCoordinator | None = None

    def scratch(self, shards: int, **kwargs) -> ShardCoordinator:
        coordinator = self.factory(self.scratch_key, **kwargs)
        if coordinator.run_info() is not None:
            coordinator.delete_run()
        coordinator.import_shards({"scratch": True, "lot": 45, "worker_id": self.worker_id}, [scratch_shard(i) for i in range(shards)])
        return coordinator

    # -- checks ----------------------------------------------------------
    def distributed_imports(self) -> dict:
        modules = ["google.cloud.firestore", "lot45.distributed.coordinator", "lot45.distributed.worker", "lot45.distributed.finalizer", "lot45.distributed.migration", "songo_ai.search"]
        resolved = {m: importlib.import_module(m).__name__ for m in modules}
        return {**self.project_imports(), "modules": resolved}

    def checkpoint_matches_run(self) -> dict:
        details = self.checkpoint()
        manifest = read_json(self.base_out / "shard_manifest.json")
        if self.ctx.identity().fingerprint() != manifest["search_identity_fingerprint"]:
            raise Lot44FatalError("MCTS_CONFIG_INCOMPATIBLE", "checkpoint/engine/MCTS identity differs from shard_manifest.json")
        details["mcts_identity_fingerprint"] = manifest["search_identity_fingerprint"]
        return details

    def connection(self) -> dict:
        self.coordinator = self.factory(self.scratch_key)
        skew = self.coordinator.server_clock_skew(f"{self.worker_id}-{int(time.time())}")
        if abs(skew) > MAX_CLOCK_SKEW_S:
            raise Lot44FatalError("CLOCK_SKEW", f"local clock differs from Firestore by {skew:.1f}s (> {MAX_CLOCK_SKEW_S}s)")
        manifest = read_json(self.base_out / "shard_manifest.json")
        real = self.factory(run_key(manifest))
        info = real.run_info()
        if info is not None and info.get("migrated_utc"):
            check_compatible(self.ctx, manifest, info)
        return {"coordinator": self.config.public(), "clock_skew_s": round(skew, 3), "run_key": real.run_key, "run_initialised": bool(info and info.get("migrated_utc")), "progress": real.progress() if info and "total_shards" in info else None}

    def atomicity(self) -> dict:
        """Vraie course : RACERS clients independants reservent 1 shard, puis 3 shards."""

        def race(shards: int) -> list:
            self.scratch(shards)
            barrier = threading.Barrier(RACERS)
            results: list = [None] * RACERS
            failures: list = []

            def contender(i: int) -> None:
                coordinator = self.factory(self.scratch_key)
                barrier.wait()
                try:
                    results[i] = coordinator.claim_next_shard(f"racer-{i}", f"run-{i}")
                except BaseException as exc:  # remonte apres la course
                    failures.append(exc)

            threads = [threading.Thread(target=contender, args=(i,)) for i in range(RACERS)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if failures:
                raise failures[0]
            return [r for r in results if r is not None]

        one = race(1)
        if len(one) != 1:
            raise RuntimeError(f"{len(one)} leases granted on a single shard")
        three = race(3)
        if len(three) != 3 or len({lease.shard_id for lease in three}) != 3:
            raise RuntimeError(f"3 shards / {RACERS} racers gave {[(lease.shard_id, lease.worker_id) for lease in three]}")
        return {"racers": RACERS, "single_shard_winners": 1, "three_shards_distinct": sorted(lease.shard_id for lease in three)}

    def registration(self) -> dict:
        coordinator = self.scratch(1)
        coordinator.register_worker(self.worker_id, {"device": str(self.device), "preflight": True})
        info = coordinator.worker_info(self.worker_id)
        if info is None or info["worker_id"] != self.worker_id:
            raise RuntimeError("worker registration not readable")
        return {"worker_id": self.worker_id}

    def claim(self) -> dict:
        coordinator = self.scratch(1)
        lease = coordinator.claim_next_shard(self.worker_id, "preflight")
        if not isinstance(lease, Lease) or lease.fencing_token != 1 or lease.attempt_number != 1:
            raise RuntimeError(f"unexpected lease {lease}")
        if coordinator.claim_next_shard("other", "preflight") is not None:
            raise RuntimeError("a held lease was granted twice")
        return {"lease": {k: v for k, v in lease.as_dict().items() if k in ("shard_id", "fencing_token", "attempt_number", "status")}}

    def heartbeat(self) -> dict:
        coordinator = self.scratch(1, lease_ttl_s=3.0)
        lease = coordinator.claim_next_shard(self.worker_id, "preflight")
        beat = LeaseHeartbeat(coordinator, lease, interval_s=0.5).start()
        time.sleep(4.5)
        beat.stop()
        beat.assert_valid()
        if beat.renewals < 2 or beat.lease.lease_expires_at <= lease.lease_expires_at:
            raise RuntimeError(f"lease not renewed ({beat.renewals} renewals)")
        if coordinator.claim_next_shard("intruder", "preflight") is not None:
            raise RuntimeError("renewed lease was stolen")
        return {"renewals": beat.renewals, "extended_by_s": round(beat.lease.lease_expires_at - lease.lease_expires_at, 2)}

    def artifact_write(self) -> dict:
        target = self.tmp / "attempt_probe" / "probe.json"
        digest = write_checked_json(target, {"probe": utc_now(), "worker_id": self.worker_id})
        if checked_status(target) != "VALID" or sha256(target) != digest:
            raise OSError("checked artifact probe failed on Drive")
        shutil.rmtree(target.parent)
        return {"sha256": digest, "path": str(target)}

    def fencing(self) -> dict:
        coordinator = self.scratch(1, lease_ttl_s=1.0)
        first = coordinator.claim_next_shard("worker-A", "preflight")
        time.sleep(1.5)
        second = coordinator.claim_next_shard("worker-B", "preflight")
        if second is None or second.fencing_token != first.fencing_token + 1:
            raise RuntimeError(f"expired lease not reclaimed with a higher token ({first}, {second})")
        try:
            coordinator.complete(first, {"files": [], "search_identity_fingerprint": "preflight"})
            stale_published = True
        except Lot44FatalError as exc:
            stale_published = exc.code != "FENCING_REJECTED"
        if stale_published:
            raise RuntimeError("stale lease was allowed to publish")
        coordinator.complete(second, {"files": [], "search_identity_fingerprint": "preflight"})
        record = coordinator.shard_records()[0]
        if record["status"] != COMPLETED or record["artifact"]["lease_id"] != second.lease_id:
            raise RuntimeError(f"coordinator record {record['status']} {record['artifact']}")
        self._completed = (coordinator, second)
        return {"stale_token": first.fencing_token, "winner_token": second.fencing_token}

    def resume(self) -> dict:
        coordinator, winner = self._completed
        again = coordinator.complete(winner, {"files": [], "search_identity_fingerprint": "preflight"})
        if again["status"] != "ALREADY_COMPLETED":
            raise RuntimeError(f"completion not idempotent: {again['status']}")
        if coordinator.claim_next_shard("restarted", "preflight") is not None:
            raise RuntimeError("a COMPLETED shard was claimable again")
        return {"completed_shard_not_reclaimed": True, "idempotent_completion": True}

    def run(self) -> dict:
        self.ctx.out.mkdir(parents=True, exist_ok=True)
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self._completed = None
        self.check("PROJECT_IMPORTS", self.distributed_imports)
        self.check("CHECKPOINT", self.checkpoint_matches_run)
        self.check("CUDA", self.cuda, critical=self.ctx.device_name == "cuda")
        self.check("DRIVE", self.drive)
        self.check("COORDINATOR_CONNECTION", self.connection)
        connected = self.checks["COORDINATOR_CONNECTION"]["status"] == "PASS"
        for name, fn in (("COORDINATOR_ATOMICITY_TEST", self.atomicity), ("WORKER_REGISTRATION", self.registration), ("SHARD_CLAIM", self.claim), ("HEARTBEAT", self.heartbeat)):
            if connected:
                self.check(name, fn)
            else:
                self.checks[name] = {"status": "FAIL", "critical": True, "error": "coordinator unreachable"}
        self.check("ARTIFACT_WRITE", self.artifact_write)
        for name, fn in (("FENCING", self.fencing), ("RESUME", self.resume)):
            if connected and (name == "FENCING" or self._completed is not None):
                self.check(name, fn)
            else:
                self.checks[name] = {"status": "FAIL", "critical": True, "error": "coordinator unreachable or FENCING failed"}
        cleanup = None
        if connected:
            cleanup = self.factory(self.scratch_key).delete_run()
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        failed = [n for n in ORDER if self.checks[n]["critical"] and self.checks[n]["status"] != "PASS"]
        report = {
            "lot": 45, "mode": "distributed", "worker_id": self.worker_id, "timestamp_utc": utc_now(),
            "code_commit": self.ctx.fingerprints()["code_commit"], "device": str(self.device) if self.device else self.ctx.device_name,
            "CUDA_AVAILABLE": "YES" if torch.cuda.is_available() else "NO", "coordinator": self.config.public(), "scratch_docs_deleted": cleanup,
            "checks": {n: self.checks[n] for n in ORDER}, "summary": {n: self.checks[n]["status"] for n in ORDER},
            "critical_failures": failed, "PREFLIGHT_STATUS": "PASS" if not failed else "FAIL", "SCIENTIFIC_RUN_ALLOWED": "YES" if not failed else "NO",
        }
        write_json(self.ctx.out / "preflight_report.json", report)
        return report
