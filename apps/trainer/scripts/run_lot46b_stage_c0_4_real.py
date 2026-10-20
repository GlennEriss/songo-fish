#!/usr/bin/env python3
"""Secure C0.4 real runner. Safe default is non-executing preflight."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import torch
from songo_ai.evaluation.arena_runner import ArenaStore, Candidate, GameResult, schedule
from songo_ai.evaluation.stage_c0_4 import proposed_manifest, select_device
from songo_ai.evaluation.stage_c0_4_real import *

BASE_COMMIT="5066e19c860b75b1670bc58b4a004a1a99e1f3b3"
G4P="fbf379dfdc34775cd6109bf6520d2694d2e94ede97ee594250ff01c40e01b05f"
G4V="9bb4ee187d62998d93d5d578c84071ee6925977338a934a969c99dfc9813cc79"
DEEP="83d8256ce1b01a794f1a5f0f3b5cde7a4c4d05a178cb031941455811366ce5d7"

def build_manifest():
    candidates=(Candidate(MODEL_A,"pool-policy.pt",G4P,BASE_COMMIT,"SRN_COMPLETE_NETWORK","POOL_G4R_POLICY_VALUE_PAIR","LOT34R_G4_RETRY",12000),Candidate(MODEL_B,"deep-unconstrained.pt",DEEP,BASE_COMMIT,"SRN_COMPLETE_NETWORK","DEEP_POLICY_UNCONSTRAINED","LOT46B_B2_STAGE_B_DEEP_UNCONSTRAINED_001",5000))
    return proposed_manifest(BASE_COMMIT,candidates,{"opening_id":OPENING_ID})

def paths(a):
    return {"POOL_G4R_POLICY":(a.checkpoint_root/"data/experiments/lot34r_g4_retry/checkpoints/pool/step-06000.pt",G4P),"POOL_G4R_VALUE":(a.checkpoint_root/"data/experiments/lot34r_g4_retry/checkpoints/pool/step-12000.pt",G4V),"DEEP_POLICY_UNCONSTRAINED":(a.checkpoint_root/"experiments/lot46b/LOT46B_B2_STAGE_B_DEEP_UNCONSTRAINED_001/durable_checkpoints/checkpoint-00005000.pt",DEEP)}

def real_executor_factory(a,opening_record,selected):
    # These real imports and model loads are reached only after production
    # authorization, ownership and immutable scientific preflight pass.
    from songo_ai.dataset import RawSongoState
    from songo_ai.evaluation import HybridPolicyValueEvaluator
    from songo_ai.evaluation.srn_arena import ArenaConfig,ArenaOpening,SRNMCTSAgent,ArenaStatus
    from songo_ai.model import load_srn_checkpoint
    device=torch.device(selected);cp=paths(a);started=time.monotonic()
    policy=load_srn_checkpoint(cp["POOL_G4R_POLICY"][0],device=selected).model.eval()
    value=load_srn_checkpoint(cp["POOL_G4R_VALUE"][0],device=selected).model.eval()
    models={MODEL_A:HybridPolicyValueEvaluator(policy,value,name=MODEL_A).to(device).eval(),MODEL_B:load_srn_checkpoint(cp[MODEL_B][0],device=selected).model.eval()}
    for model in models.values():model.requires_grad_(False)
    load_seconds=time.monotonic()-started
    s=opening_record["state"];opening=ArenaOpening(OPENING_ID,RawSongoState(tuple(s["board"]),int(s["player_to_move"])),(),0)
    def execute(game,device_name):
        agents={name:SRNMCTSAgent(name,model,512) for name,model in models.items()}
        raw=RealAdapters.execute_game(opening,p1_agent=agents[game.candidate_p1],p2_agent=agents[game.candidate_p2],agent_a_name=MODEL_A,agent_b_name=MODEL_B,a_player=1 if game.candidate_p1==MODEL_A else 2,config=ArenaConfig(seed=game.seed))
        status=raw["status"]
        if status not in (ArenaStatus.TERMINAL_WIN,ArenaStatus.TERMINAL_DRAW,"TERMINAL_WIN","TERMINAL_DRAW"):raise RuntimeError(f"non-terminal smoke result: {status}")
        draw=raw["winner"] in (None,0);terminal="DRAW" if draw else "P1_WIN" if raw["winner"]==1 else "P2_WIN"
        winner=None if draw else game.candidate_p1 if terminal=="P1_WIN" else game.candidate_p2
        trace=hashlib.sha256(json.dumps(raw["action_sequence"]).encode()).hexdigest()
        meta={"device":device_name,"model_load_time_seconds":load_seconds,"game_duration_seconds":raw["elapsed_s"],"total_moves":raw["continuation_plies"],"mcts_simulations":raw["total_mcts_simulations"],"mcts_search_time_seconds":sum(raw["search_time_s_by_agent"].values()),"synthetic":False}
        return GameResult(**game.__dict__,terminal_result=terminal,winner=winner,draw=draw,game_length=raw["continuation_plies"],terminal_state_fingerprint=trace,game_trace_hash=trace,execution_metadata=meta)
    return execute

def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id",default=EXPERIMENT_ID);p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    p.add_argument("--registry-path",type=Path,required=True);p.add_argument("--opening-id",default=OPENING_ID)
    p.add_argument("--checkpoint-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--drive-output",type=Path,required=True);p.add_argument("--authorization-file",type=Path)
    p.add_argument("--resume",action="store_true");p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--execute",action="store_true",help="R2 only: requires a manifest-bound authorization record")
    p.add_argument("--single-runner-confirmed",action="store_true")
    return p

def main(argv=None)->int:
    a=parser().parse_args(argv)
    if a.experiment_id!=EXPERIMENT_ID or a.opening_id!=OPENING_ID: raise SystemExit("immutable C0.4 identity mismatch")
    manifest=build_manifest();checkpoint_paths=paths(a);identities={k:v[1] for k,v in checkpoint_paths.items()}
    amendment=Path(__file__).resolve().parents[3]/"docs/lot46b_stage_c0_4_p5/opening_protocol_amendment_v1.json"
    def preflight():return validate_preflight(manifest=manifest,registry=a.registry_path,amendment=amendment,checkpoints=checkpoint_paths,authorization_file=a.authorization_file)
    if not a.execute:
        report={"status":"SAFE_PREFLIGHT_ONLY","REAL_GAMES_EXECUTED":0,"MCTS_SEARCH":"NOT_RUN","experiment_id":a.experiment_id,"device_mode":a.device.upper(),"manifest":manifest.scientific(),"manifest_fingerprint":manifest.fingerprint}
        if a.authorization_file:report["validation"]=preflight()
        print(json.dumps(report,indent=2));return 0
    if a.preflight_only: raise SystemExit("--execute conflicts with --preflight-only")
    if not a.authorization_file: raise SystemExit("--execute requires --authorization-file")
    if not a.single_runner_confirmed: raise SystemExit("--execute requires --single-runner-confirmed (Google Drive has no atomic distributed lock)")
    opening=[x for x in json.loads(a.registry_path.read_text())["records"] if x["opening_id"]==OPENING_ID][0]
    a.output_dir.mkdir(parents=True,exist_ok=True);a.drive_output.mkdir(parents=True,exist_ok=True)
    store=ArenaStore(a.output_dir/"arena.sqlite");store.initialize(manifest,schedule(manifest))
    owner=OwnershipConfirmation(EXPERIMENT_ID,"operator-confirmed-session",True)
    def choose():return select_device(torch,a.device).selected_device
    outcome=production_control_flow(preflight=lambda:validate_preflight(manifest=manifest,registry=a.registry_path,amendment=amendment,checkpoints=checkpoint_paths,authorization_file=a.authorization_file,ownership=owner),manifest=manifest,store=store,authorization_file=a.authorization_file,checkpoint_identities=identities,ownership=owner,device_selector=choose,executor_factory=lambda device:real_executor_factory(a,opening,device),snapshot_directory=a.drive_output/"sqlite_snapshots")
    print(json.dumps(outcome,indent=2,default=str));return 0

if __name__=="__main__":sys.exit(main())
