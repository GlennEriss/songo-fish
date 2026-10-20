"""LOT46B C0.4 preexecution primitives.

This module deliberately contains no game engine or MCTS call.  It prepares a
two-game schedule, selects an inference device, records attempt provenance and
enforces a separate, fail-closed real-execution authorization gate.
"""
from __future__ import annotations

import json
import os
import platform
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .arena_runner import (
    ArenaStore,
    Candidate,
    ExperimentManifest,
    ScheduledGame,
    canonical,
    file_sha256,
    fingerprint,
    schedule,
)

REGISTRY_SHA256 = "73cf111f5b8837075916c592c7f5c8cb973740becee674f94c84c240ebebf68e"
OPENING_ID = "LOT46B-C0-OPENING-0001"
EXPERIMENT_ID = "LOT46B_STAGE_C0_4_SMOKE"
MCTS_SIMULATIONS = 512
REAL_EXECUTION_AUTHORIZED = False
REAL_AUTHORIZATION_SCOPE = "EXACTLY_TWO_REAL_GAMES_MCTS512_ONE_FROZEN_OPENING"


class DeviceSelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceProvenance:
    requested_mode: str
    selected_device: str
    device_name: str
    torch_version: str
    cuda_version: str | None
    precision_mode: str
    deterministic_settings: Mapping[str, Any]
    inference_batch_size: int
    fallback_reason: str | None
    initialization_seconds: float


def _cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"


def select_device(
    torch_module: Any,
    mode: str = "auto",
    *,
    precision: str = "fp32",
    inference_batch_size: int = 1,
    cuda_probe: Callable[[], None] | None = None,
) -> DeviceProvenance:
    """Select AUTO/CUDA/CPU without touching CUDA in forced CPU mode."""
    requested = mode.upper()
    if requested not in {"AUTO", "CUDA", "CPU"}:
        raise ValueError("device mode must be AUTO, CUDA, or CPU")
    started = time.monotonic()
    fallback = None
    selected = "cpu"
    name = platform.processor() or platform.machine() or "CPU"
    cuda_version = str(getattr(getattr(torch_module, "version", None), "cuda", None))
    if cuda_version == "None":
        cuda_version = None
    if requested != "CPU":
        available = False
        try:
            available = bool(torch_module.cuda.is_available())
            if not available:
                raise DeviceSelectionError("CUDA unavailable")
            (cuda_probe or (lambda: torch_module.empty(1, device="cuda")))()
            selected = "cuda"
            name = str(torch_module.cuda.get_device_name(0))
        except Exception as exc:
            if requested == "CUDA":
                raise DeviceSelectionError(f"forced CUDA initialization failed: {exc}") from exc
            fallback = f"{exc.__class__.__name__}: {exc}"
            selected = "cpu"
    return DeviceProvenance(
        requested, selected, name, str(torch_module.__version__), cuda_version,
        precision, {"root_noise": False, "temperature": 0.0, "explicit_seeds": True},
        inference_batch_size, fallback, time.monotonic() - started,
    )


def release_cuda(torch_module: Any) -> None:
    try:
        torch_module.cuda.empty_cache()
    except Exception:
        pass


def fallback_before_commit(
    exc: BaseException,
    *,
    result_committed: bool,
    torch_module: Any,
    attempt_recorder: "AttemptRecorder",
    game: ScheduledGame,
    attempt_id: str,
) -> DeviceProvenance:
    """Fail over only by restarting an uncommitted scheduled game from opening."""
    attempt_recorder.fail(attempt_id, exc)
    if result_committed:
        raise DeviceSelectionError("result already committed; replay forbidden") from exc
    release_cuda(torch_module)
    return select_device(torch_module, "cpu")


class AttemptRecorder:
    """Attempt provenance separate from exactly-once scientific results."""
    def __init__(self, store: ArenaStore):
        self.store = store
        self.store.db.execute(
            "CREATE TABLE IF NOT EXISTS attempts("
            "attempt_id TEXT PRIMARY KEY, game_id TEXT NOT NULL, payload BLOB NOT NULL)"
        )
        self.store.db.commit()

    def begin(self, game: ScheduledGame, provenance: DeviceProvenance, attempt_id: str) -> None:
        payload = canonical({
            "attempt_id": attempt_id, "game_id": game.game_id, "opening_id": game.opening_id,
            "status": "STARTED", "device": asdict(provenance), "started_monotonic": time.monotonic(),
        })
        try:
            with self.store.db:
                self.store.db.execute("INSERT INTO attempts VALUES(?,?,?)", (attempt_id, game.game_id, payload))
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate attempt identity") from exc

    def fail(self, attempt_id: str, exc: BaseException) -> None:
        row = self.store.db.execute("SELECT payload FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        if not row:
            raise ValueError("unknown attempt identity")
        payload = json.loads(row[0]); payload.update({"status": "FAILED", "failure": f"{exc.__class__.__name__}: {exc}"})
        with self.store.db:
            self.store.db.execute("UPDATE attempts SET payload=? WHERE attempt_id=?", (canonical(payload), attempt_id))


def require_real_execution(*, dry_run: bool, authorization: str | None) -> None:
    if dry_run:
        raise PermissionError("DRY_RUN=TRUE; real game execution forbidden")
    if not REAL_EXECUTION_AUTHORIZED or authorization != "LOT46B_STAGE_C0_4_REAL_SMOKE_APPROVED":
        raise PermissionError("REAL_GAME_EXECUTION_AUTHORIZED=NO")


def load_frozen_opening(registry_path: Path, expected_sha256: str = REGISTRY_SHA256) -> dict:
    if file_sha256(registry_path) != expected_sha256:
        raise ValueError("frozen registry SHA256 mismatch")
    payload = json.loads(registry_path.read_text())
    matches = [x for x in payload["records"] if x["opening_id"] == OPENING_ID]
    if len(matches) != 1:
        raise ValueError("frozen smoke opening missing or duplicated")
    return matches[0]


def proposed_manifest(commit: str, candidates: tuple[Candidate, Candidate], opening: dict) -> ExperimentManifest:
    if {c.name for c in candidates} != {"POOL_G4R", "DEEP_POLICY_UNCONSTRAINED"}:
        raise ValueError("unexpected smoke candidates")
    manifest = ExperimentManifest(
        EXPERIMENT_ID, "lot46b-c0.4-smoke-v1", commit, "VERIFY_EXISTING_ENGINE_AT_EXECUTION",
        candidates,
        ({"comparison_id": "C0.4-SMOKE", "candidate_a": "POOL_G4R",
          "candidate_b": "DEEP_POLICY_UNCONSTRAINED", "budget_a": MCTS_SIMULATIONS,
          "budget_b": MCTS_SIMULATIONS},),
        (opening["opening_id"],), 20264640,
        {"technical_smoke_only": True, "excluded_from_c1_c2": True,
         "c1_planned_games": 4800, "c2_non_inferiority_margin": -0.05,
         "registry_sha256": REGISTRY_SHA256},
        "AUTO", "fp32", {},
    )
    games = schedule(manifest)
    if len(games) != 2 or games[0].opening_id != games[1].opening_id:
        raise ValueError("invalid two-game frozen-opening schedule")
    if (games[0].candidate_p1, games[0].candidate_p2) != (games[1].candidate_p2, games[1].candidate_p1):
        raise ValueError("side reversal failure")
    return manifest


def validate_checkpoint(path: Path, expected_sha256: str) -> dict:
    if not path.is_file():
        return {"status": "NOT_VERIFIED", "path": str(path), "expected_sha256": expected_sha256}
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"checkpoint SHA256 mismatch: {path}")
    return {"status": "PASS", "path": str(path), "sha256": actual}


def performance_schema() -> dict:
    return {key: None for key in (
        "MODEL_LOAD_TIME_SECONDS", "DEVICE_INITIALIZATION_SECONDS", "INFERENCE_TIME_SECONDS",
        "MCTS_SEARCH_TIME_SECONDS", "MCTS_SIMULATIONS_COMPLETED", "MCTS_SIMULATIONS_PER_SECOND",
        "AVERAGE_MOVE_TIME_SECONDS", "GAME_DURATION_SECONDS", "SQLITE_COMMIT_TIME_SECONDS",
        "DRIVE_SNAPSHOT_TIME_SECONDS", "PEAK_MEMORY_USAGE", "DEVICE_TYPE",
    )}
