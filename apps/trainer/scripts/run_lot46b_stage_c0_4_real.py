#!/usr/bin/env python3
"""Secure C0.4 real runner. Safe default is preflight-only; R1 forbids --execute."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from songo_ai.evaluation.stage_c0_4_real import EXPERIMENT_ID, OPENING_ID

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
    if not a.execute:
        print(json.dumps({"status":"SAFE_PREFLIGHT_ONLY","REAL_GAMES_EXECUTED":0,"MCTS_SEARCH":"NOT_RUN","experiment_id":a.experiment_id,"device_mode":a.device.upper()},indent=2));return 0
    if a.preflight_only: raise SystemExit("--execute conflicts with --preflight-only")
    if not a.authorization_file: raise SystemExit("--execute requires --authorization-file")
    if not a.single_runner_confirmed: raise SystemExit("--execute requires --single-runner-confirmed (Google Drive has no atomic distributed lock)")
    raise SystemExit("R1 controlled stop: production execution wiring is disabled until R2 authorization record is installed")

if __name__=="__main__":sys.exit(main())
