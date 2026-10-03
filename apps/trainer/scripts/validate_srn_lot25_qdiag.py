#!/usr/bin/env python3
"""Contrôle ciblé Qdiag256/512 du Lot 25."""
import json,statistics
from pathlib import Path
from songo_ai.dataset import RawSongoState
from songo_ai.evaluation import diagnostic_action_values
from songo_ai.model import load_srn_checkpoint
from run_srn_lot12 import write_json

G2=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt")
SOURCE=Path("data/d_scale_v1/d_strategic_sample/qdiag256.jsonl")
CACHE=Path("data/experiments/lot25_scale/qdiag512_stability_cache.jsonl")
OUT=Path("data/experiments/lot25_scale/strategic_stability.json")

def main():
    rows=[json.loads(x) for x in SOURCE.open() if x.strip()][:200];model=load_srn_checkpoint(G2).model;cached={}
    if CACHE.exists():
        for line in CACHE.open():r=json.loads(line);cached[r["position_hash"]]=r
    with CACHE.open("a") as out:
        for i,row in enumerate(rows,1):
            key=row["position_hash"]
            if key not in cached:
                s=row["state"];result=diagnostic_action_values(RawSongoState(tuple(s["board"]),s["player_to_move"]),model,num_simulations=512,seed=20263000+i);item={"position_hash":key,**result};out.write(json.dumps(item,sort_keys=True)+"\n");out.flush();cached[key]=item
            if i%25==0:print(f"[lot25] Qdiag512 {i}/{len(rows)}",flush=True)
    agreements=[];gaps=[]
    for row in rows:
        q256=row["q_values"];q512=cached[row["position_hash"]]["q_values"];legal=[i for i,x in enumerate(row["legal_mask"]) if x];a=max(legal,key=lambda i:q256[i]);b=max(legal,key=lambda i:q512[i]);agreements.append(a==b);gaps.append(abs(q256[a]-q512[a]))
    report={"positions":len(rows),"budgets_per_action":[256,512],"top1_agreement":statistics.fmean(agreements),"preferred_action_q_abs_delta_mean":statistics.fmean(gaps),"generator":"G2-best","dirichlet":False};write_json(OUT,report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
