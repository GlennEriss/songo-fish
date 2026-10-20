"""Secure orchestration contracts for the future C0.4 real smoke run.

Importing this module never imports the Songo engine or MCTS.  Those imports
are deliberately delayed until a bound authorization and every preflight gate
have passed.
"""
from __future__ import annotations

import json, os, sqlite3, time, uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from .arena_runner import ArenaStore, ExperimentManifest, GameResult, ScheduledGame, canonical, file_sha256, schedule
from .arena_integration import SQLiteSnapshotManager
from .stage_c0_4 import EXPERIMENT_ID, MCTS_SIMULATIONS, OPENING_ID, REAL_AUTHORIZATION_SCOPE, REGISTRY_SHA256

PROTOCOL = "lot46b-c0.4-smoke-v1"
MODEL_A, MODEL_B = "POOL_G4R", "DEEP_POLICY_UNCONSTRAINED"
AUTHORIZATION_SCHEMA = 1

class RunState(str, Enum):
    CREATED="CREATED"; PREFLIGHT_VALIDATED="PREFLIGHT_VALIDATED"; AUTHORIZED="AUTHORIZED"
    RUNNING="RUNNING"; INTERRUPTED="INTERRUPTED"; RESUMABLE="RESUMABLE"
    COMPLETED="COMPLETED"; FAILED="FAILED"; BLOCKED="BLOCKED"

TRANSITIONS={
    RunState.CREATED:{RunState.PREFLIGHT_VALIDATED,RunState.BLOCKED,RunState.FAILED},
    RunState.PREFLIGHT_VALIDATED:{RunState.AUTHORIZED,RunState.BLOCKED,RunState.FAILED},
    RunState.AUTHORIZED:{RunState.RUNNING,RunState.BLOCKED,RunState.FAILED},
    RunState.RUNNING:{RunState.INTERRUPTED,RunState.COMPLETED,RunState.FAILED},
    RunState.INTERRUPTED:{RunState.RESUMABLE,RunState.FAILED},
    RunState.RESUMABLE:{RunState.RUNNING,RunState.BLOCKED,RunState.FAILED},
    RunState.COMPLETED:set(),RunState.FAILED:set(),RunState.BLOCKED:set(),
}

def transition(current:RunState,target:RunState)->RunState:
    if target not in TRANSITIONS[current]: raise ValueError(f"invalid state transition: {current}->{target}")
    return target

def schedule_contract(manifest:ExperimentManifest)->dict:
    games=schedule(manifest)
    if manifest.experiment_id!=EXPERIMENT_ID or manifest.protocol_version!=PROTOCOL: raise ValueError("wrong experiment identity")
    if len(games)!=2: raise ValueError("exactly two games required")
    if any(g.opening_id!=OPENING_ID or g.budget_p1!=512 or g.budget_p2!=512 for g in games): raise ValueError("opening or MCTS budget mismatch")
    if (games[0].candidate_p1,games[0].candidate_p2)!=(MODEL_A,MODEL_B): raise ValueError("GAME_1 side assignment mismatch")
    if (games[1].candidate_p1,games[1].candidate_p2)!=(MODEL_B,MODEL_A): raise ValueError("GAME_2 side assignment mismatch")
    return {"game_ids":[g.game_id for g in games],"side_assignment_schedule":[[g.candidate_p1,g.candidate_p2] for g in games]}

def authorization_binding(manifest:ExperimentManifest,checkpoint_identities:Mapping[str,str])->dict:
    contract=schedule_contract(manifest)
    return {"schema":AUTHORIZATION_SCHEMA,"experiment_id":EXPERIMENT_ID,"experiment_manifest_sha256":manifest.fingerprint,
        "registry_sha256":REGISTRY_SHA256,"opening_id":OPENING_ID,"model_a":MODEL_A,"model_b":MODEL_B,
        "mcts_simulations_per_decision":MCTS_SIMULATIONS,"total_scheduled_games":2,
        "side_assignment_schedule":contract["side_assignment_schedule"],"checkpoint_identities":dict(checkpoint_identities),
        "authorization_scope":REAL_AUTHORIZATION_SCOPE,"scientific_protocol_version":PROTOCOL,"approved":True}

def validate_authorization(path:Path,manifest:ExperimentManifest,checkpoint_identities:Mapping[str,str])->dict:
    if not path.is_file(): raise PermissionError("authorization record missing")
    actual=json.loads(path.read_text()); expected=authorization_binding(manifest,checkpoint_identities)
    if actual!=expected: raise PermissionError("authorization record is not exactly bound to immutable manifest")
    return actual

@dataclass(frozen=True)
class OwnershipConfirmation:
    experiment_id:str; owner_id:str; single_runner_confirmed:bool

def validate_ownership(record:OwnershipConfirmation)->None:
    # Google Drive has no atomic compare-and-swap. R1 therefore deliberately
    # uses fail-closed operator-confirmed single-runner mode.
    if record.experiment_id!=EXPERIMENT_ID or not record.owner_id or not record.single_runner_confirmed:
        raise PermissionError("fail-closed single-runner ownership confirmation required")

class ExecutionJournal:
    def __init__(self,store:ArenaStore):
        self.store=store
        store.db.execute("CREATE TABLE IF NOT EXISTS c04_attempts(attempt_id TEXT PRIMARY KEY,game_id TEXT NOT NULL,status TEXT NOT NULL,device TEXT NOT NULL,payload BLOB NOT NULL)")
        store.db.commit()
    def begin(self,game:ScheduledGame,device:str)->str:
        attempt_id=f"{game.game_id[:12]}-{uuid.uuid4().hex}"
        with self.store.db:self.store.db.execute("INSERT INTO c04_attempts VALUES(?,?,?,?,?)",(attempt_id,game.game_id,"RUNNING",device,canonical({"attempt_id":attempt_id,"game_id":game.game_id,"device":device})))
        return attempt_id
    def interrupt(self,attempt_id:str,reason:str)->None:
        with self.store.db:self.store.db.execute("UPDATE c04_attempts SET status='INTERRUPTED',payload=? WHERE attempt_id=?",(canonical({"attempt_id":attempt_id,"reason":reason}),attempt_id))
    def complete(self,attempt_id:str)->None:
        with self.store.db:self.store.db.execute("UPDATE c04_attempts SET status='COMPLETED' WHERE attempt_id=?",(attempt_id,))
    def history(self)->list[dict]:
        return [dict(zip(("attempt_id","game_id","status","device"),r)) for r in self.store.db.execute("SELECT attempt_id,game_id,status,device FROM c04_attempts ORDER BY rowid")]

def verify_resume(store:ArenaStore,manifest:ExperimentManifest)->tuple[ScheduledGame,...]:
    contract=schedule_contract(manifest); store.initialize(manifest,schedule(manifest))
    row=store.db.execute("SELECT v FROM meta WHERE k='fingerprint'").fetchone()
    if not row or row[0]!=manifest.fingerprint: raise ValueError("resume manifest mismatch")
    pending=set(store.pending())
    return tuple(g for g in schedule(manifest) if g.game_id in pending)

def publish_and_verify(store:ArenaStore,directory:Path,manifest:ExperimentManifest,generation:int)->dict:
    info=SQLiteSnapshotManager(directory,manifest.fingerprint).publish(store,generation)
    restored=directory/f"restore-{generation:08d}-{uuid.uuid4().hex}.sqlite"
    SQLiteSnapshotManager(directory,manifest.fingerprint).restore(generation,restored)
    db=sqlite3.connect(restored); integrity=db.execute("PRAGMA integrity_check").fetchone()[0]; db.close(); restored.unlink()
    if integrity!="ok": raise ValueError("Drive snapshot read-back integrity failure")
    return info|{"read_back":"PASS","integrity":integrity}

class RealAdapters:
    """Lazy boundaries; compatibility can be inspected without playing/searching."""
    @staticmethod
    def compatibility()->dict:
        from songo_ai.dataset.selfplay_schema import RawSongoState
        from songo_ai.songo.rules import SongoLegacyGame
        from songo_ai.search.mcts import SongoMCTS
        return {"engine":SongoLegacyGame.__name__,"state":RawSongoState.__name__,"mcts":SongoMCTS.__name__}
    @staticmethod
    def execute_game(*args:Any,**kwargs:Any)->Mapping[str,Any]:
        from songo_ai.evaluation.srn_arena import play_arena_game
        return asdict(play_arena_game(*args,validate_prefix=False,**kwargs))

def execute_pending(*,manifest:ExperimentManifest,store:ArenaStore,authorization_file:Path,
    checkpoint_identities:Mapping[str,str],ownership:OwnershipConfirmation,device:str,
    executor:Callable[[ScheduledGame,str],GameResult],snapshot_directory:Path,max_games:int=2)->dict:
    """Generic exactly-once loop. R1 tests inject a synthetic executor."""
    validate_authorization(authorization_file,manifest,checkpoint_identities);validate_ownership(ownership)
    pending=verify_resume(store,manifest)
    if len(schedule(manifest))!=2 or max_games!=2: raise PermissionError("authorized pair boundary violated")
    journal=ExecutionJournal(store); committed=[]
    for game in pending:
        attempt=journal.begin(game,device)
        try:
            result=executor(game,device); outcome=store.complete(result)
            if outcome not in {"COMMITTED","IDEMPOTENT"}: raise RuntimeError(outcome)
            journal.complete(attempt); committed.append(game.game_id)
            publish_and_verify(store,snapshot_directory,manifest,len(store.paired_records())+len(committed))
        except BaseException as exc:
            journal.interrupt(attempt,f"{type(exc).__name__}: {exc}"); raise
    return {"completed":len(list(store.db.execute('SELECT 1 FROM results'))),"pending":len(store.pending()),"attempt_history":journal.history()}
