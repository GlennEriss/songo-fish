"""Constantes et configuration explicite du coordinateur distribue."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from lot44.artifacts import Lot44FatalError

from ..config import DISTRIBUTED_MARKER

DISTRIBUTED_DIR = "distributed"
ATTEMPTS_DIR = "attempts"
RUN_MARKER = DISTRIBUTED_MARKER
DEFAULT_NAMESPACE = "lot45_distributed"

LEASE_TTL_S = 900.0
HEARTBEAT_INTERVAL_S = 120.0
IDLE_POLL_S = 60.0
MAX_ATTEMPTS_PER_SHARD = 8
RETRYABLE_BACKOFF_S = 60.0
CLAIM_CANDIDATES = 8
FINALIZER_TTL_S = 7200.0
MIGRATION_TTL_S = 1800.0
MAX_CLOCK_SKEW_S = 60.0
MAX_CONSECUTIVE_RECOVERABLE = 3
# Garde-fou de cout : operations Firestore maximales par processus worker.
# Estimation : ~30 ops par shard + 2 ops/min au repos (palier gratuit : 50k
# lectures et 20k ecritures par jour).
MAX_OPS_PER_PROCESS = 8000

COORDINATOR_KEYS = {"backend", "project", "database", "namespace", "credentials", "emulator_host"}


@dataclass(frozen=True)
class CoordinatorConfig:
    """``credentials`` : ``"adc"`` (Application Default Credentials, ex.
    ``google.colab.auth.authenticate_user()``) ou chemin d'une cle de compte de
    service stockee HORS du depot (ex. sur Drive)."""

    project: str
    database: str = "(default)"
    namespace: str = DEFAULT_NAMESPACE
    credentials: str = "adc"
    emulator_host: str | None = None
    backend: str = "firestore"

    def public(self) -> dict:
        """Description sans secret (le chemin de cle n'est pas un secret, son contenu oui)."""

        return {"backend": self.backend, "project": self.project, "database": self.database, "namespace": self.namespace, "credentials": "adc" if self.credentials == "adc" else "service_account_file", "emulator": bool(self.emulator_host)}


def parse_coordinator_config(raw: str | dict | None) -> CoordinatorConfig:
    """Accepte un dict, une chaine JSON ou le chemin d'un fichier JSON."""

    if raw is None or raw == "":
        raise Lot44FatalError("COORDINATOR_CONFIG_MISSING", "a coordinator configuration is required (--coordinator-config)")
    if isinstance(raw, str):
        text = raw.strip()
        if not text.startswith("{"):
            path = Path(text).expanduser()
            if not path.is_file():
                raise Lot44FatalError("COORDINATOR_CONFIG_INVALID", f"coordinator config file not found: {path}")
            text = path.read_text(encoding="utf-8")
        raw = json.loads(text)
    unknown = set(raw) - COORDINATOR_KEYS
    if unknown:
        raise Lot44FatalError("COORDINATOR_CONFIG_INVALID", f"unknown coordinator keys {sorted(unknown)}")
    if raw.get("backend", "firestore") != "firestore":
        raise Lot44FatalError("COORDINATOR_CONFIG_INVALID", f"unsupported backend {raw.get('backend')!r} (only 'firestore' is transactional)")
    if not raw.get("project"):
        raise Lot44FatalError("COORDINATOR_CONFIG_INVALID", "coordinator 'project' must be set explicitly")
    emulator = raw.get("emulator_host") or os.environ.get("FIRESTORE_EMULATOR_HOST") or None
    return CoordinatorConfig(project=raw["project"], database=raw.get("database", "(default)"), namespace=raw.get("namespace", DEFAULT_NAMESPACE), credentials=raw.get("credentials", "adc"), emulator_host=emulator)
