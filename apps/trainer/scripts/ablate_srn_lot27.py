#!/usr/bin/env python3
"""Ablations racine/descendants/Value sur le failure set du Lot 27."""
import json,statistics
from collections import Counter
from pathlib import Path
from songo_ai.dataset import RawSongoState
from songo_ai.evaluation.hybrid_evaluator import DepthPolicyValueEvaluator,HybridPolicyValueEvaluator
from songo_ai.model import load_srn_checkpoint
from songo_ai.search import MCTSConfig,SongoMCTS
from run_srn_lot12 import write_json
from run_srn_lot27 import battery,js_divergence
OUT=Path("data/experiments/lot27_search_policy")
def main():
 rows,_=battery(2000);failure={r["position_hash"] for r in json.load((OUT/"scaling_failure_set.json").open())["rows"]};selected=[r for r in rows if r["position_hash"] in failure][:200]+[r for r in rows if r["position_hash"] not in failure][:200];g2=load_srn_checkpoint("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt").model;g3=load_srn_checkpoint("data/experiments/lot26_g3_scale/checkpoints/g3_strategic_best.pt").model
 configs={"G2_FULL":g2,"ROOT_ONLY":DepthPolicyValueEvaluator(g3,g2,g2),"DESC_ONLY":DepthPolicyValueEvaluator(g2,g3,g2),"FULL_POLICY_G2_VALUE":DepthPolicyValueEvaluator(g3,g3,g2),"G3_FULL":g3,"G3_POLICY_G2_VALUE":HybridPolicyValueEvaluator(g3,g2)};cache=OUT/"ablation_cache.jsonl";known={}
 if cache.exists():
  for line in cache.open():r=json.loads(line);known[(r["position_hash"],r["config"])]=r
 with cache.open("a") as out:
  for i,row in enumerate(selected,1):
   state=RawSongoState(tuple(row["state"]["board"]),row["state"]["player_to_move"])
   for name,model in configs.items():
    key=(row["position_hash"],name)
    if key not in known:
     r=SongoMCTS(model,config=MCTSConfig(num_simulations=128,c_puct=1.5,add_root_noise=False,collect_simulation_trace=True,seed=20262728)).search(state);points={}
     for b in (64,128):t=r.simulation_trace[b-1];v=t["visit_counts_after_backup"];tot=sum(v);p=[x/tot for x in v];a=max((j for j,x in enumerate(row["legal_mask"]) if x),key=lambda j:(v[j],-j));points[str(b)]={"action":a,"policy":p,"visits":v};item={"position_hash":row["position_hash"],"config":name,"failure":row["position_hash"] in failure,"points":points,"nodes":r.num_nodes,"depths":[t["leaf_depth"] for t in r.simulation_trace],"root_value":r.root_value};out.write(json.dumps(item,sort_keys=True)+"\n");out.flush();known[key]=item
   if i%25==0:print(f"[lot27] ablation {i}/{len(selected)}",flush=True)
 summary={}
 for name in configs:
  rs=[known[(r["position_hash"],name)] for r in selected];summary[name]={"positions":len(rs),"action_switch_64_128":statistics.fmean(x["points"]["64"]["action"]!=x["points"]["128"]["action"] for x in rs),"nodes_mean":statistics.fmean(x["nodes"] for x in rs),"depth_mean":statistics.fmean(d for x in rs for d in x["depths"]),"root_value_mean":statistics.fmean(x["root_value"] for x in rs)}
 base=[known[(r["position_hash"],"G2_FULL")] for r in selected]
 for name in configs:
  rs=[known[(r["position_hash"],name)] for r in selected];summary[name]["agreement_with_G2"]={str(b):statistics.fmean(x["points"][str(b)]["action"]==y["points"][str(b)]["action"] for x,y in zip(rs,base)) for b in (64,128)}
 write_json(OUT/"policy_ablation.json",summary);write_json(OUT/"value_interaction.json",{"G3_FULL_vs_G3_POLICY_G2_VALUE":{str(b):{"action_agreement":statistics.fmean(known[(r["position_hash"],"G3_FULL")]["points"][str(b)]["action"]==known[(r["position_hash"],"G3_POLICY_G2_VALUE")]["points"][str(b)]["action"] for r in selected),"visit_js_mean":statistics.fmean(js_divergence(known[(r["position_hash"],"G3_FULL")]["points"][str(b)]["policy"],known[(r["position_hash"],"G3_POLICY_G2_VALUE")]["points"][str(b)]["policy"]) for r in selected)} for b in (64,128)}});write_json(OUT/"tree_distribution.json",summary);write_json(OUT/"depth_divergence.json",summary);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
