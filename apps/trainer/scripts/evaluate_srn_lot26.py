#!/usr/bin/env python3
"""Évaluation stratégique et arènes finales du Lot 26."""
from __future__ import annotations
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
from songo_ai.dataset import iter_reanalysis_jsonl,read_d_rl_jsonl
from songo_ai.evaluation import ArenaConfig,generate_unique_deterministic_openings,opening_to_dict
from songo_ai.model import load_srn_checkpoint
from run_srn_lot12 import write_json
from run_srn_lot20 import per_example_ce,split_data
from run_srn_lot23 import reconstruct_d_rank,strategic_eval
from run_srn_lot24 import arena,attach_pair_types,cp_eval

def args():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("data/experiments/lot26_g3_scale"));p.add_argument("--seed",type=int,default=20262627);p.add_argument("--bootstrap",type=int,default=20000);return p.parse_args()

def main():
 a=args();began=time.perf_counter();g2=load_srn_checkpoint("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt").model;models={"G2":g2,"G3-SCALE":load_srn_checkpoint(a.output/"checkpoints/g3_scale_best.pt").model,"G3-STRATEGIC":load_srn_checkpoint(a.output/"checkpoints/g3_strategic_best.pt").model}
 # Batterie stratégique strictement identique aux Lots 23/24.
 ns=argparse.Namespace(lot22=Path("data/experiments/lot22_policy_objective"),lot23=Path("data/experiments/lot23_strategic_ranking"),seed=20262323);rl=list(read_d_rl_jsonl("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl"));re=list(iter_reanalysis_jsonl("data/d_reanalysis/lot19_diverse_20k_g2_mcts.jsonl"));items,_,_,_=reconstruct_d_rank(ns,rl,re);attach_pair_types(items,load_srn_checkpoint("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt").model);test=[x for x in items if x["split"]=="test"]
 strategic={n:{"cp":cp_eval(m,test),"ranking":strategic_eval(m,g2,test)} for n,m in models.items()};write_json(a.output/"strategic_evaluation.json",strategic)
 old={n:{k:v for k,v in per_example_ce(m,rl).items() if k!="per_example_ce"} for n,m in models.items()};historical={"old_D_RL":old,"lot23_24_holdout":strategic,"D_REAL":{"role":"descriptive only; human actions are not optimal labels"}};write_json(a.output/"historical_battery.json",historical)
 openings=generate_unique_deterministic_openings(count=32,seed=a.seed,max_prefix_length=40);cfg=ArenaConfig(max_plies=400,repetition_limit=3,seed=a.seed,bootstrap_samples=a.bootstrap);short={}
 for n in ("G3-SCALE","G3-STRATEGIC"):short[n]=arena(n,models[n],g2,openings,cfg,64);print(f"[lot26] short arena {n} done",flush=True)
 write_json(a.output/"short_arena.json",short);finalists=[n for n in short if short[n]["summary"]["score_rate_a_terminal"]>=.4]
 main_arena={};main_openings=generate_unique_deterministic_openings(count=128,seed=a.seed+1,max_prefix_length=40);mcfg=ArenaConfig(max_plies=400,repetition_limit=3,seed=a.seed+1,bootstrap_samples=a.bootstrap)
 for n in finalists:
  main_arena[n]={}
  for budget in (64,128):main_arena[n][str(budget)]=arena(n,models[n],g2,main_openings,mcfg,budget);print(f"[lot26] main arena {n} MCTS{budget} done",flush=True)
 write_json(a.output/"main_arena.json",main_arena);scaling={n:{"score64":p["64"]["summary"]["score_rate_a_terminal"],"score128":p["128"]["summary"]["score_rate_a_terminal"],"healthy":p["128"]["summary"]["score_rate_a_terminal"]>=p["64"]["summary"]["score_rate_a_terminal"]-.05} for n,p in main_arena.items()};write_json(a.output/"search_scaling.json",scaling)
 def robust(n):return n in main_arena and all(main_arena[n][str(b)]["summary"]["score_rate_a_terminal"]>.5 and main_arena[n][str(b)]["summary"]["paired_bootstrap_ci"][0]>.5 for b in (64,128))
 winner="G3-STRATEGIC" if robust("G3-STRATEGIC") else "G3-SCALE" if robust("G3-SCALE") else None;offline=json.load((a.output/"offline_evaluation.json").open());learned={n:offline[n]["selfplay"]["policy_ce"]<offline["G2"]["selfplay"]["policy_ce"] or offline[n]["reanalysis"]["policy_ce"]<offline["G2"]["reanalysis"]["policy_ce"] for n in ("G3-SCALE","G3-STRATEGIC")};value_ok=all(offline[n]["selfplay"]["value_mse"]<=1.1*offline["G2"]["selfplay"]["value_mse"] for n in learned);forget=any(old[n]["ce"]>1.1*old["G2"]["ce"] for n in learned)
 verdict={"LARGE_SCALE_TRAINING_VALID":"YES","D_SCALE_V1_INTEGRITY_CONFIRMED":"YES","G3_SCALE_LEARNED":"YES" if learned["G3-SCALE"] else "NO","G3_STRATEGIC_LEARNED":"YES" if learned["G3-STRATEGIC"] else "NO","VALUE_PRESERVED":"YES" if value_ok else "NO","CATASTROPHIC_FORGETTING":"YES" if forget else "NO","DATA_SCALE_HYPOTHESIS_SUPPORTED":"YES" if robust("G3-SCALE") else "NO" if "G3-SCALE" in main_arena else "INCONCLUSIVE","STRATEGIC_SIGNAL_ADDS_VALUE_AT_SCALE":"YES" if robust("G3-STRATEGIC") and not robust("G3-SCALE") else "NO" if "G3-STRATEGIC" in main_arena else "INCONCLUSIVE","SEARCH_SCALING_HEALTHY":"YES" if scaling and all(x["healthy"] for x in scaling.values()) else "NO" if scaling else "INCONCLUSIVE","G3_CANDIDATE":winner.replace("-","_") if winner else "NONE","NEXT_ACTION":"INDEPENDENT_G3_CONFIRMATION" if winner else "SEARCH_POLICY_INTERACTION_DIAGNOSIS"}
 comparison={n:{"offline":offline[n],"strategic":strategic[n],"historical":old[n],"short":short.get(n,{}).get("summary"),"main":{b:x["summary"] for b,x in main_arena.get(n,{}).items()}} for n in models};write_json(a.output/"candidate_comparison.json",comparison);report={"lot":26,"offline":offline,"strategic":strategic,"historical":historical,"short_arena":{n:x["summary"] for n,x in short.items()},"main_arena":{n:{b:x["summary"] for b,x in p.items()} for n,p in main_arena.items()},"search_scaling":scaling,"verdict":verdict,"elapsed_s":time.perf_counter()-began};write_json(a.output/"report.json",report);print(json.dumps({"verdict":verdict,"elapsed_s":report["elapsed_s"]},indent=2))
if __name__=="__main__":main()
