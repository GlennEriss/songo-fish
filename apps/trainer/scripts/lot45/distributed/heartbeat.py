"""Renouvellement periodique d'un bail, dans un thread, via le coordinateur.

Le calcul GPU n'est jamais bloque : le thread renouvelle toutes les
``interval_s`` secondes (une transaction Firestore, aucune ecriture Drive).
Un echec transitoire est retente de facon bornee par le coordinateur ; si le
bail est supplante (``LEASE_LOST``) ou expire localement sans renouvellement,
le bail est declare perdu et ``assert_valid`` interdit toute publication.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Callable

from lot44.artifacts import Lot44FatalError

from .coordinator import CoordinatorUnavailable, Lease, ShardCoordinator, TransactionContention


class LeaseHeartbeat:
    def __init__(self, coordinator: ShardCoordinator, lease: Lease, *, interval_s: float, log: Callable[[str], None] = print, clock: Callable[[], float] = time.time) -> None:
        self.coordinator = coordinator
        self.lease = lease
        self.interval_s = interval_s
        self.log = log
        self.clock = clock
        self.lost: str | None = None
        self.renewals = 0
        self.failures = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "LeaseHeartbeat":
        self._thread = threading.Thread(target=self._loop, name=f"lease-{self.lease.shard_id}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.coordinator.op_timeout_s * (self.coordinator.max_retries + 2))

    def beat(self) -> None:
        """Un renouvellement (appele par le thread ; public pour les tests)."""

        try:
            self.lease = self.coordinator.renew(self.lease)
            self.renewals += 1
        except Lot44FatalError as exc:
            if exc.code != "LEASE_LOST":
                raise
            self.lost = str(exc)
            self.log(f"[Lot45][heartbeat] {self.lease.shard_id}: bail perdu : {exc}")
        except (CoordinatorUnavailable, TransactionContention) as exc:
            self.failures += 1
            self.log(f"[Lot45][heartbeat] {self.lease.shard_id}: renouvellement impossible ({exc}); expiration locale a {self.lease.lease_expires_at:.0f}")
            if self.clock() >= self.lease.lease_expires_at - self.interval_s:
                self.lost = f"lease could not be renewed before expiry ({exc})"

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.beat()
            except BaseException as exc:  # thread : l'erreur est rendue visible par self.lost et le journal
                self.lost = f"heartbeat thread crashed: {type(exc).__name__}: {exc}"
                self.log(f"[Lot45][heartbeat] {self.lost}\n{traceback.format_exc()}")
                return
            if self.lost:
                return

    def assert_valid(self) -> None:
        if self.lost:
            raise Lot44FatalError("LEASE_LOST", f"{self.lease.shard_id}: {self.lost}; refusing to publish")
        if self.clock() >= self.lease.lease_expires_at:
            raise Lot44FatalError("LEASE_LOST", f"{self.lease.shard_id}: lease expired locally at {self.lease.lease_expires_at:.0f}; refusing to publish")
