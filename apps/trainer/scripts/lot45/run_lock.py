"""Verrou mono-ecrivain (sections 9-11).

``run_lock.json`` designe le seul processus autorise a ecrire dans le dossier
d'experience. Un thread rafraichit ``heartbeat_utc`` ; avant chaque ecriture de
shard, le proprietaire verifie qu'il possede toujours le verrou. Un verrou ne
peut etre repris que s'il est perime ET que la reprise est demandee
explicitement ; chaque acquisition/reprise est journalisee dans
``run_lock_history.json``.

Limite connue : Drive synchronise avec une latence. ``settle_s`` attend puis
relit le verrou pour detecter une acquisition concurrente.
"""
from __future__ import annotations

import calendar
import os
import platform
import socket
import threading
import time
import uuid
from pathlib import Path

from lot44.artifacts import Lot44FatalError, read_json, write_json

from .config import LOCK_HEARTBEAT_S, LOCK_SETTLE_S, LOCK_STALE_AFTER_S


def machine_identity() -> dict:
    colab = Path("/content").is_dir() and "COLAB_RELEASE_TAG" in os.environ
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "pid": os.getpid(), "colab": colab, "machine_id": f"{'colab' if colab else 'local'}:{socket.gethostname()}"}


def _parse(ts: str) -> float:
    return float(calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")))


class RunLock:
    def __init__(self, out: Path, *, owner: str, stale_after_s: float = LOCK_STALE_AFTER_S, heartbeat_s: float = LOCK_HEARTBEAT_S, settle_s: float = LOCK_SETTLE_S, clock=time.time) -> None:
        self.path = out / "run_lock.json"
        self.history = out / "run_lock_history.json"
        self.owner = owner
        self.stale_after_s = stale_after_s
        self.heartbeat_s = heartbeat_s
        self.settle_s = settle_s
        self.clock = clock
        self.run_id = uuid.uuid4().hex
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _now(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock()))

    def _log(self, event: dict) -> None:
        history = read_json(self.history) if self.history.is_file() else {"events": []}
        history["events"].append({**event, "timestamp_utc": self._now()})
        write_json(self.history, history)

    def current(self) -> dict | None:
        return read_json(self.path) if self.path.is_file() else None

    def age_s(self, lock: dict) -> float:
        return self.clock() - _parse(lock["heartbeat_utc"])

    def payload(self, started: str) -> dict:
        return {"run_id": self.run_id, "owner": self.owner, **machine_identity(), "start_utc": started, "heartbeat_utc": self._now(), "released": False}

    def acquire(self, *, takeover_stale: bool = False, reason: str = "") -> dict:
        existing = self.current()
        previous = None
        if existing and not existing.get("released"):
            age = self.age_s(existing)
            if age < self.stale_after_s:
                raise Lot44FatalError("RUN_LOCKED", f"output owned by {existing['machine_id']} (owner={existing['owner']}, run_id={existing['run_id'][:8]}, heartbeat {age:.0f}s ago); stop that process/runtime first")
            if not takeover_stale:
                raise Lot44FatalError("STALE_LOCK", f"stale lock from {existing['machine_id']} (heartbeat {age:.0f}s ago); rerun with --takeover-stale-lock to take over explicitly")
            previous = existing
        started = self._now()
        write_json(self.path, self.payload(started))
        if self.settle_s:
            time.sleep(self.settle_s)
        if (self.current() or {}).get("run_id") != self.run_id:
            raise Lot44FatalError("RUN_LOCKED", "another writer acquired the lock concurrently")
        self._log({"event": "TAKEOVER" if previous else "ACQUIRE", "run_id": self.run_id, "new_owner": machine_identity()["machine_id"], "owner": self.owner, "previous_owner": (previous or {}).get("machine_id"), "previous_run_id": (previous or {}).get("run_id"), "previous_heartbeat_utc": (previous or {}).get("heartbeat_utc"), "reason": reason or ("stale heartbeat" if previous else "fresh start")})
        self._started = started
        return self.current()

    def assert_owner(self) -> None:
        lock = self.current()
        if not lock or lock.get("run_id") != self.run_id or lock.get("released"):
            raise Lot44FatalError("LOCK_LOST", f"run lock no longer owned by this process ({(lock or {}).get('machine_id')}); refusing to write")

    def beat(self) -> None:
        self.assert_owner()
        write_json(self.path, {**self.current(), "heartbeat_utc": self._now()})

    def start_heartbeat(self) -> None:
        def loop() -> None:
            while not self._stop.wait(self.heartbeat_s):
                self.beat()

        self._thread = threading.Thread(target=loop, name="lot45-run-lock", daemon=True)
        self._thread.start()

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        lock = self.current()
        if lock and lock.get("run_id") == self.run_id:
            write_json(self.path, {**lock, "released": True, "released_utc": self._now()})
            self._log({"event": "RELEASE", "run_id": self.run_id, "owner": self.owner})

    def __enter__(self) -> "RunLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
