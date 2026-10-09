"""SHARD_COORDINATOR : baux transactionnels Firestore.

Schema (``{namespace}`` configurable, defaut ``lot45_distributed``) ::

    {namespace}/{run_key}                      document du run (compteurs, verrous exclusifs)
    {namespace}/{run_key}/shards/{shard_id}    un bail par shard
    {namespace}/{run_key}/workers/{worker_id}  enregistrement des workers
    {namespace}/{run_key}/events/{auto}        journal d'audit des transitions

Toute transition de statut passe par une transaction Firestore qui relit le
document : deux workers ne peuvent pas obtenir simultanement un bail valide sur
le meme shard, et seul le detenteur du dernier ``fencing_token`` peut publier.
Les horodatages de bail sont en secondes epoch de l'horloge du worker (derive
d'horloge verifiee par le preflight, ``MAX_CLOCK_SKEW_S``).
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

from google.api_core import exceptions as gexc
from google.auth import exceptions as auth_exceptions
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from lot44.artifacts import Lot44FatalError

from .config import CLAIM_CANDIDATES, LEASE_TTL_S, MAX_ATTEMPTS_PER_SHARD, MAX_OPS_PER_PROCESS, RETRYABLE_BACKOFF_S, CoordinatorConfig

PENDING, RUNNING, COMPLETED, FAILED_RETRYABLE, FAILED_FATAL = "PENDING", "RUNNING", "COMPLETED", "FAILED_RETRYABLE", "FAILED_FATAL"
STATUSES = (PENDING, RUNNING, COMPLETED, FAILED_RETRYABLE, FAILED_FATAL)
TERMINAL = (COMPLETED, FAILED_FATAL)
# None = document absent (import par la migration).
TRANSITIONS = frozenset({
    (None, PENDING), (None, COMPLETED),
    (PENDING, RUNNING), (FAILED_RETRYABLE, RUNNING),
    (RUNNING, RUNNING),  # renouvellement, ou reprise d'un bail expire (nouveau jeton)
    (RUNNING, COMPLETED), (RUNNING, FAILED_RETRYABLE), (RUNNING, FAILED_FATAL),
    (PENDING, FAILED_FATAL), (FAILED_RETRYABLE, FAILED_FATAL),  # tentatives epuisees
})
NEVER = 1e18  # available_at d'un shard terminal : jamais selectionne
TRANSIENT_ERRORS = (
    gexc.ServiceUnavailable, gexc.DeadlineExceeded, gexc.InternalServerError, gexc.TooManyRequests,
    gexc.ResourceExhausted, gexc.Aborted, gexc.GatewayTimeout, gexc.RetryError, gexc.Unknown,
    auth_exceptions.TransportError,
)
_TX_EXHAUSTED = "Failed to commit transaction in"


class CoordinatorUnavailable(OSError):
    """Coordinateur injoignable apres les tentatives bornees (erreur recuperable)."""


class TransactionContention(OSError):
    """Transaction abandonnee apres ``max_attempts`` conflits (erreur recuperable)."""


def check_transition(old: str | None, new: str) -> None:
    if (old, new) not in TRANSITIONS:
        raise Lot44FatalError("INVALID_TRANSITION", f"lease transition {old} -> {new} is not allowed")


@dataclass(frozen=True)
class Lease:
    shard_id: str
    worker_id: str
    run_id: str
    lease_id: str
    fencing_token: int
    lease_started_at: float
    lease_expires_at: float
    heartbeat_at: float
    attempt_number: int
    status: str = RUNNING

    def as_dict(self) -> dict:
        return asdict(self)


def make_client(config: CoordinatorConfig) -> firestore.Client:
    """Client Firestore explicite : emulateur, ADC ou cle de compte de service hors depot."""

    if config.emulator_host:
        import os

        os.environ["FIRESTORE_EMULATOR_HOST"] = config.emulator_host
        return firestore.Client(project=config.project, database=config.database)
    if config.credentials == "adc":
        import google.auth

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if hasattr(credentials, "with_quota_project"):
            credentials = credentials.with_quota_project(config.project)
    else:
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(config.credentials, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return firestore.Client(project=config.project, database=config.database, credentials=credentials)


class ShardCoordinator:
    def __init__(
        self,
        client: firestore.Client,
        config: CoordinatorConfig,
        run_key: str,
        *,
        clock: Callable[[], float] = time.time,
        lease_ttl_s: float = LEASE_TTL_S,
        max_attempts_per_shard: int = MAX_ATTEMPTS_PER_SHARD,
        max_retries: int = 4,
        backoff_s: float = 1.0,
        op_timeout_s: float = 30.0,
        tx_attempts: int = 25,
        ops_budget: int = MAX_OPS_PER_PROCESS,
        log: Callable[[str], None] = print,
    ) -> None:
        self.client = client
        self.config = config
        self.run_key = run_key
        self.clock = clock
        self.lease_ttl_s = lease_ttl_s
        self.max_attempts_per_shard = max_attempts_per_shard
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.op_timeout_s = op_timeout_s
        self.tx_attempts = tx_attempts
        self.ops_budget = ops_budget
        self.ops = 0
        self.log = log
        # retry=None : seuls les retries bornes de _call s'appliquent (les retries
        # par defaut de google-api-core ignorent ``timeout`` et durent jusqu'a 300 s).
        self.rpc = {"retry": None, "timeout": op_timeout_s}
        self.root = client.collection(config.namespace).document(run_key)
        self.shards = self.root.collection("shards")
        self.workers = self.root.collection("workers")
        self.events = self.root.collection("events")

    # -- infrastructure ---------------------------------------------------------
    def _charge(self, ops: int) -> None:
        self.ops += ops
        if self.ops > self.ops_budget:
            raise Lot44FatalError("COORDINATOR_BUDGET_EXCEEDED", f"{self.ops} Firestore operations > budget {self.ops_budget} for this process")

    def _call(self, what: str, action: Callable[[], Any], *, ops: int = 1) -> Any:
        """Retry borne des erreurs transitoires ; au-dela : CoordinatorUnavailable."""

        for attempt in range(self.max_retries + 1):
            self._charge(ops)
            try:
                return action()
            except TRANSIENT_ERRORS as exc:
                if attempt == self.max_retries:
                    raise CoordinatorUnavailable(f"{what}: {type(exc).__name__}: {exc} after {attempt + 1} attempts") from exc
                self.log(f"[Lot45][coordinator] {what}: {type(exc).__name__} (tentative {attempt + 1}/{self.max_retries + 1})")
            except TransactionContention as exc:
                if attempt == self.max_retries:
                    raise
                self.log(f"[Lot45][coordinator] {what}: {exc} (tentative {attempt + 1}/{self.max_retries + 1})")
            time.sleep(self.backoff_s * (2 ** attempt) * (0.5 + random.random()))
        raise AssertionError("unreachable")

    def _transaction(self, body: Callable[..., Any], *args: Any) -> Any:
        transaction = self.client.transaction(max_attempts=self.tx_attempts)
        try:
            return firestore.transactional(body)(transaction, *args)
        except ValueError as exc:
            if _TX_EXHAUSTED in str(exc):
                raise TransactionContention(str(exc)) from exc
            # BeginTransaction/Commit en echec transitoire : la bibliotheque tente un
            # rollback impossible et leve ValueError ; la cause reelle est __context__.
            if isinstance(exc.__context__, TRANSIENT_ERRORS):
                raise TransactionContention(f"transaction interrupted by {type(exc.__context__).__name__}: {exc.__context__}") from exc
            raise

    def _get(self, transaction, ref) -> dict | None:
        snap = ref.get(transaction=transaction, **self.rpc)
        return snap.to_dict() if snap.exists else None

    def _event(self, transaction, event: str, **fields: Any) -> None:
        transaction.set(self.events.document(), {"event": event, "t": self.clock(), "server_time": firestore.SERVER_TIMESTAMP, **fields})

    # -- run ------------------------------------------------------------------
    def run_info(self) -> dict | None:
        snap = self._call("read run", lambda: self.root.get(**self.rpc))
        return snap.to_dict() if snap.exists else None

    def progress(self) -> dict:
        info = self.run_info()
        if info is None or "total_shards" not in info:
            raise Lot44FatalError("COORDINATOR_RUN_MISSING", f"run {self.run_key} is not initialised in the coordinator (run the migrate stage)")
        done = info["completed_count"] + info["fatal_count"]
        return {"total": info["total_shards"], "completed": info["completed_count"], "failed_fatal": info["fatal_count"], "remaining": info["total_shards"] - done, "all_settled": done >= info["total_shards"]}

    def import_shards(self, meta: dict, shards: list[dict]) -> dict:
        """Cree le run et les shards absents ; ne modifie jamais un shard existant.

        ``shards`` : ``{shard_id, index, input_sha256, count, status (PENDING|COMPLETED), artifact|None, legacy_classification}``.
        Un shard deja present doit avoir le meme ``input_sha256`` (sinon SPLIT_MODIFIED).
        """

        if len(shards) > 400:
            raise Lot44FatalError("PLAN_INVALID", f"{len(shards)} shards exceed one Firestore transaction (400)")
        refs = [self.shards.document(s["shard_id"]) for s in shards]

        def body(transaction):
            run = self._get(transaction, self.root) or {}
            existing = {snap.id: snap.to_dict() for snap in transaction.get_all(refs, **self.rpc) if snap.exists}
            created, kept = [], []
            for shard, ref in zip(shards, refs):
                current = existing.get(shard["shard_id"])
                if current is not None:
                    if current["input_sha256"] != shard["input_sha256"]:
                        raise Lot44FatalError("SPLIT_MODIFIED", f"{shard['shard_id']}: coordinator input_sha256 differs from the shard plan")
                    kept.append(shard["shard_id"])
                    continue
                check_transition(None, shard["status"])
                completed = shard["status"] == COMPLETED
                transaction.set(ref, {
                    "shard_id": shard["shard_id"], "index": shard["index"], "input_sha256": shard["input_sha256"], "count": shard["count"],
                    "status": shard["status"], "available_at": NEVER if completed else 0.0, "worker_id": None, "run_id": None, "lease_id": None,
                    "fencing_token": 0, "lease_started_at": None, "lease_expires_at": None, "heartbeat_at": None, "attempt_number": 0,
                    "artifact": shard.get("artifact"), "legacy_classification": shard.get("legacy_classification"), "imported": True,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                })
                created.append(shard["shard_id"])
            already_completed = sum(1 for s in kept if existing[s]["status"] == COMPLETED)
            already_fatal = sum(1 for s in kept if existing[s]["status"] == FAILED_FATAL)
            new_completed = sum(1 for s in shards if s["shard_id"] in created and s["status"] == COMPLETED)
            transaction.set(self.root, {
                **{k: v for k, v in run.items() if k not in meta}, **meta,
                "total_shards": len(shards), "completed_count": already_completed + new_completed, "fatal_count": already_fatal,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            self._event(transaction, "IMPORT", created=len(created), kept=len(kept))
            return {"created": created, "kept": kept}

        return self._call("import shards", lambda: self._transaction(body), ops=2 * len(shards) + 4)

    # -- verrous exclusifs (migration, finalisation) ------------------------------
    def acquire_exclusive(self, name: str, holder: str, ttl_s: float) -> dict:
        def body(transaction):
            run = self._get(transaction, self.root) or {}
            lock = (run.get("locks") or {}).get(name)
            now = self.clock()
            if lock and lock["holder"] != holder and lock["expires_at"] > now:
                raise Lot44FatalError(f"{name.upper()}_ACTIVE", f"{name} lock held by {lock['holder']} until {lock['expires_at']:.0f} (now {now:.0f})")
            token = (lock or {}).get("token", 0) + 1
            granted = {"holder": holder, "token": token, "acquired_at": now, "expires_at": now + ttl_s}
            transaction.set(self.root, {"locks": {name: granted}, "updated_at": firestore.SERVER_TIMESTAMP}, merge=True)
            self._event(transaction, f"{name.upper()}_ACQUIRE", holder=holder, token=token, previous_holder=(lock or {}).get("holder"))
            return granted

        return self._call(f"acquire {name}", lambda: self._transaction(body), ops=3)

    def release_exclusive(self, name: str, granted: dict) -> bool:
        def body(transaction):
            run = self._get(transaction, self.root) or {}
            lock = (run.get("locks") or {}).get(name)
            if not lock or lock["holder"] != granted["holder"] or lock["token"] != granted["token"]:
                return False
            transaction.set(self.root, {"locks": {name: {**lock, "expires_at": 0.0, "released_at": self.clock()}}, "updated_at": firestore.SERVER_TIMESTAMP}, merge=True)
            self._event(transaction, f"{name.upper()}_RELEASE", holder=granted["holder"], token=granted["token"])
            return True

        return self._call(f"release {name}", lambda: self._transaction(body), ops=3)

    # -- workers --------------------------------------------------------------
    def register_worker(self, worker_id: str, info: dict) -> None:
        payload = {**info, "worker_id": worker_id, "registered_at": self.clock(), "last_seen": self.clock(), "server_time": firestore.SERVER_TIMESTAMP}
        self._call("register worker", lambda: self.workers.document(worker_id).set(payload, merge=True, **self.rpc))

    def worker_info(self, worker_id: str) -> dict | None:
        snap = self._call("read worker", lambda: self.workers.document(worker_id).get(**self.rpc))
        return snap.to_dict() if snap.exists else None

    def update_worker(self, worker_id: str, fields: dict) -> None:
        self._call("update worker", lambda: self.workers.document(worker_id).set({**fields, "last_seen": self.clock()}, merge=True, **self.rpc))

    # -- baux -----------------------------------------------------------------
    def _claimable(self, shard: dict, now: float) -> bool:
        if shard["status"] in (PENDING, FAILED_RETRYABLE):
            return shard["available_at"] <= now
        return shard["status"] == RUNNING and shard["lease_expires_at"] is not None and shard["lease_expires_at"] <= now

    def _try_claim(self, ref, worker_id: str, run_id: str) -> Lease | str | None:
        def body(transaction):
            shard = self._get(transaction, ref)
            now = self.clock()
            if shard is None or not self._claimable(shard, now):
                return None
            attempt = shard["attempt_number"] + 1
            if attempt > self.max_attempts_per_shard:
                check_transition(shard["status"], FAILED_FATAL)
                transaction.update(ref, {"status": FAILED_FATAL, "available_at": NEVER, "failure": {"reason": "MAX_ATTEMPTS_EXCEEDED", "attempts": shard["attempt_number"]}, "updated_at": firestore.SERVER_TIMESTAMP})
                transaction.update(self.root, {"fatal_count": firestore.Increment(1)})
                self._event(transaction, "MAX_ATTEMPTS_EXCEEDED", shard_id=ref.id, attempts=shard["attempt_number"])
                return "EXHAUSTED"
            check_transition(shard["status"], RUNNING)
            lease = Lease(shard_id=ref.id, worker_id=worker_id, run_id=run_id, lease_id=uuid.uuid4().hex, fencing_token=shard["fencing_token"] + 1, lease_started_at=now, lease_expires_at=now + self.lease_ttl_s, heartbeat_at=now, attempt_number=attempt)
            transaction.update(ref, {**lease.as_dict(), "available_at": lease.lease_expires_at, "updated_at": firestore.SERVER_TIMESTAMP})
            self._event(transaction, "TAKEOVER" if shard["status"] == RUNNING else "CLAIM", shard_id=ref.id, worker_id=worker_id, lease_id=lease.lease_id, fencing_token=lease.fencing_token, attempt_number=attempt, previous_worker=shard.get("worker_id"), previous_status=shard["status"], previous_token=shard["fencing_token"])
            return lease

        return self._call(f"claim {ref.id}", lambda: self._transaction(body), ops=3)

    def claim_next_shard(self, worker_id: str, run_id: str) -> Lease | None:
        """Reserve atomiquement un shard PENDING/FAILED_RETRYABLE ou un bail expire.

        Les candidats sont lus hors transaction (requete ``available_at <= now``),
        puis chaque reservation relit le document dans une transaction : un
        candidat pris entre-temps est simplement ignore.
        """

        rng = random.Random(f"{worker_id}:{self.clock()}")
        for _ in range(3):
            now = self.clock()
            query = self.shards.where(filter=FieldFilter("available_at", "<=", now)).order_by("available_at").limit(CLAIM_CANDIDATES)
            candidates = self._call("list candidates", lambda: list(query.stream(**self.rpc)), ops=CLAIM_CANDIDATES)
            if not candidates:
                return None
            rng.shuffle(candidates)
            for snap in candidates:
                outcome = self._try_claim(snap.reference, worker_id, run_id)
                if isinstance(outcome, Lease):
                    return outcome
        return None

    def _owned(self, shard: dict | None, lease: Lease) -> bool:
        return shard is not None and shard["status"] == RUNNING and shard["lease_id"] == lease.lease_id and shard["fencing_token"] == lease.fencing_token

    def renew(self, lease: Lease) -> Lease:
        ref = self.shards.document(lease.shard_id)

        def body(transaction):
            shard = self._get(transaction, ref)
            if not self._owned(shard, lease):
                raise Lot44FatalError("LEASE_LOST", f"{lease.shard_id}: lease {lease.lease_id[:8]} token {lease.fencing_token} superseded (now {None if shard is None else (shard['status'], shard['lease_id'] and shard['lease_id'][:8], shard['fencing_token'])})")
            now = self.clock()
            check_transition(RUNNING, RUNNING)
            transaction.update(ref, {"heartbeat_at": now, "lease_expires_at": now + self.lease_ttl_s, "available_at": now + self.lease_ttl_s, "updated_at": firestore.SERVER_TIMESTAMP})
            return replace(lease, heartbeat_at=now, lease_expires_at=now + self.lease_ttl_s)

        return self._call(f"renew {lease.shard_id}", lambda: self._transaction(body), ops=2)

    def complete(self, lease: Lease, artifact: dict) -> dict:
        """Transition RUNNING -> COMPLETED, reservee au detenteur du jeton courant.

        Idempotent : si une tentative precedente de ce meme bail a deja commite
        (reponse perdue), renvoie l'enregistrement existant.
        """

        ref = self.shards.document(lease.shard_id)

        def body(transaction):
            shard = self._get(transaction, ref)
            if shard is not None and shard["status"] == COMPLETED and (shard.get("artifact") or {}).get("lease_id") == lease.lease_id:
                return {"status": "ALREADY_COMPLETED", "artifact": shard["artifact"]}
            if not self._owned(shard, lease):
                return {"status": "REJECTED", "current": None if shard is None else {k: shard.get(k) for k in ("status", "worker_id", "lease_id", "fencing_token")}}
            check_transition(RUNNING, COMPLETED)
            now = self.clock()
            record = {**artifact, "lease_id": lease.lease_id, "fencing_token": lease.fencing_token, "worker_id": lease.worker_id, "attempt_number": lease.attempt_number, "completed_at": now}
            transaction.update(ref, {"status": COMPLETED, "available_at": NEVER, "artifact": record, "completed_at": now, "updated_at": firestore.SERVER_TIMESTAMP})
            transaction.update(self.root, {"completed_count": firestore.Increment(1)})
            self._event(transaction, "COMPLETE", shard_id=lease.shard_id, worker_id=lease.worker_id, lease_id=lease.lease_id, fencing_token=lease.fencing_token, attempt_number=lease.attempt_number)
            return {"status": "COMPLETED", "artifact": record}

        outcome = self._call(f"complete {lease.shard_id}", lambda: self._transaction(body), ops=4)
        if outcome["status"] == "REJECTED":
            self._call("record rejection", lambda: self.events.document().set({"event": "COMPLETE_REJECTED", "t": self.clock(), "shard_id": lease.shard_id, "worker_id": lease.worker_id, "lease_id": lease.lease_id, "fencing_token": lease.fencing_token, "current": outcome["current"], "server_time": firestore.SERVER_TIMESTAMP}, **self.rpc))
            raise Lot44FatalError("FENCING_REJECTED", f"{lease.shard_id}: token {lease.fencing_token} (lease {lease.lease_id[:8]}) may not publish; coordinator holds {outcome['current']}")
        return outcome

    def release(self, lease: Lease, status: str, reason: str, *, retry_after_s: float = RETRYABLE_BACKOFF_S) -> bool:
        """Rend un bail detenu (FAILED_RETRYABLE ou FAILED_FATAL). Faux si plus proprietaire."""

        if status not in (FAILED_RETRYABLE, FAILED_FATAL):
            raise ValueError(status)
        ref = self.shards.document(lease.shard_id)

        def body(transaction):
            shard = self._get(transaction, ref)
            if not self._owned(shard, lease):
                return False
            check_transition(RUNNING, status)
            now = self.clock()
            transaction.update(ref, {"status": status, "available_at": NEVER if status == FAILED_FATAL else now + retry_after_s, "lease_expires_at": now, "failure": {"reason": reason[:2000], "worker_id": lease.worker_id, "lease_id": lease.lease_id, "at": now}, "updated_at": firestore.SERVER_TIMESTAMP})
            if status == FAILED_FATAL:
                transaction.update(self.root, {"fatal_count": firestore.Increment(1)})
            self._event(transaction, f"RELEASE_{status}", shard_id=lease.shard_id, worker_id=lease.worker_id, lease_id=lease.lease_id, fencing_token=lease.fencing_token, reason=reason[:500])
            return True

        return self._call(f"release {lease.shard_id}", lambda: self._transaction(body), ops=4)

    # -- lecture pour audit / finalisation ------------------------------------------
    def shard_records(self) -> list[dict]:
        snaps = self._call("list shards", lambda: list(self.shards.stream(**self.rpc)), ops=500)
        return sorted((s.to_dict() for s in snaps), key=lambda d: d["index"])

    def event_records(self) -> list[dict]:
        snaps = self._call("list events", lambda: list(self.events.stream(**self.rpc)), ops=2000)
        return sorted((s.to_dict() for s in snaps), key=lambda d: d["t"])

    def server_clock_skew(self, probe_id: str) -> float:
        """Ecart (s) entre l'horloge locale et l'horloge serveur Firestore."""

        ref = self.root.collection("probes").document(probe_id)
        before = self.clock()
        self._call("clock probe write", lambda: ref.set({"server_time": firestore.SERVER_TIMESTAMP}, **self.rpc))
        after = self.clock()
        snap = self._call("clock probe read", lambda: ref.get(**self.rpc))
        self._call("clock probe delete", lambda: ref.delete(**self.rpc))
        server = snap.to_dict()["server_time"].timestamp()
        return (before + after) / 2 - server

    def delete_run(self) -> int:
        """Supprime un run jetable (preflight/tests). Refuse un run reel migre."""

        info = self.run_info()
        if info is not None and info.get("scratch") is not True:
            raise Lot44FatalError("REFUSED", f"run {self.run_key} is not a scratch run; refusing to delete")
        deleted = 0
        for collection in (self.shards, self.workers, self.events, self.root.collection("probes")):
            for snap in self._call("list scratch docs", lambda c=collection: list(c.stream(**self.rpc)), ops=100):
                self._call("delete scratch doc", lambda s=snap: s.reference.delete(**self.rpc))
                deleted += 1
        self._call("delete scratch run", lambda: self.root.delete(**self.rpc))
        return deleted + 1
