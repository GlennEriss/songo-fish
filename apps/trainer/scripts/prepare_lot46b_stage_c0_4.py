#!/usr/bin/env python3
"""Prepare C0.4 artifacts. This entry point cannot execute games or MCTS."""
from __future__ import annotations
import argparse, json, subprocess
from dataclasses import asdict
from pathlib import Path
import torch
from songo_ai.evaluation.arena_runner import ArenaStore, Candidate, canonical, file_sha256, schedule
from songo_ai.evaluation.arena_integration import SQLiteSnapshotManager
from songo_ai.evaluation.stage_c0_4 import *

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--registry-path",type=Path,required=True);p.add_argument("--opening-id",default=OPENING_ID)
    p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    p.add_argument("--dry-run",action="store_true",required=True);p.add_argument("--max-games",type=int,default=2)
    p.add_argument("--unconstrained-checkpoint",type=Path,required=True)
    a=p.parse_args()
    if a.opening_id!=OPENING_ID or a.max_games!=2:raise SystemExit("C0.4 requires the frozen opening and exactly two games")
    opening=load_frozen_opening(a.registry_path); commit=subprocess.check_output(["git","-C",str(a.repo),"rev-parse","HEAD"],text=True).strip()
    g4p=a.repo/"data/experiments/lot34r_g4_retry/checkpoints/pool/step-06000.pt"
    g4v=a.repo/"data/experiments/lot34r_g4_retry/checkpoints/pool/step-12000.pt"
    checks={"pool_policy":validate_checkpoint(g4p,"fbf379dfdc34775cd6109bf6520d2694d2e94ede97ee594250ff01c40e01b05f"),"pool_value":validate_checkpoint(g4v,"9bb4ee187d62998d93d5d578c84071ee6925977338a934a969c99dfc9813cc79"),"unconstrained":validate_checkpoint(a.unconstrained_checkpoint,"83d8256ce1b01a794f1a5f0f3b5cde7a4c4d05a178cb031941455811366ce5d7")}
    digest=lambda check: check["sha256"] if check["status"]=="PASS" else check["expected_sha256"]
    candidates=(Candidate("POOL_G4R",str(g4p),digest(checks["pool_policy"]),commit,"SRN_COMPLETE_NETWORK","POOL_G4R_POLICY_VALUE_PAIR","LOT34R_G4_RETRY",12000),Candidate("DEEP_POLICY_UNCONSTRAINED",str(a.unconstrained_checkpoint),digest(checks["unconstrained"]),commit,"SRN_COMPLETE_NETWORK","DEEP_POLICY_UNCONSTRAINED","LOT46B_B2_STAGE_B_DEEP_UNCONSTRAINED_001",5000))
    manifest=proposed_manifest(commit,candidates,opening); device=select_device(torch,a.device)
    a.output.mkdir(parents=True,exist_ok=True); store=ArenaStore(a.output/"arena.sqlite");store.initialize(manifest,schedule(manifest))
    snapshot=SQLiteSnapshotManager(a.output/"sqlite_snapshots",manifest.fingerprint).publish(store,0)
    complete=all(x["status"]=="PASS" for x in checks.values())
    report={"status":"DRY_RUN_PASS" if complete else "DRY_RUN_BLOCKED_CHECKPOINT_NOT_LOCAL","commit":commit,"manifest":manifest.scientific(),"manifest_fingerprint":manifest.fingerprint,"schedule":[asdict(g) for g in schedule(manifest)],"device_provenance":asdict(device),"checkpoint_integrity":checks,"snapshot":snapshot,"performance":performance_schema(),"REAL_GPU_THROUGHPUT":"NOT_MEASURED","REAL_CPU_THROUGHPUT":"NOT_MEASURED","REAL_GAME_EXECUTION_AUTHORIZED":"NO","REAL_GAMES_EXECUTED":0,"MCTS_SEARCH":"NOT_RUN","TRAINING":"NOT_RUN","ARENAS":"NOT_RUN"}
    (a.output/"stage_c0_4_dry_run.json").write_bytes(canonical(report));print(json.dumps(report,indent=2));print("CONTROLLED STOP — no real game, no MCTS")
if __name__=="__main__":main()
