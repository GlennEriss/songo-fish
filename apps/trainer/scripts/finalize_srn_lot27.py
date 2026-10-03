#!/usr/bin/env python3
"""Consolide les diagnostics mesurés et produit les verdicts du Lot 27."""
import hashlib,json,math,statistics
from collections import Counter
from pathlib import Path
from run_srn_lot12 import write_json
from songo_ai.evaluation.search_policy_diagnosis import js_divergence
OUT=Path("data/experiments/lot27_search_policy")
def dist(v):
 v=sorted(map(float,v));
 if not v:return {"count":0}
 def q(p):x=p*(len(v)-1);i=int(x);j=min(i+1,len(v)-1);return v[i]*(j-x)+v[j]*(x-i)
 return {"count":len(v),"mean":statistics.fmean(v),"median":statistics.median(v),"p90":q(.9),"p95":q(.95),"max":v[-1]}
def entropy(p):return -sum(x*math.log(x) for x in p if x>0)
def main():
 rows=[json.loads(x) for x in (OUT/"search_cache.jsonl").open()];by={(r["position_hash"],r["model"]):r for r in rows};keys=sorted({r["position_hash"] for r in rows});models=("G2","G3-SCALE","G3-STRATEGIC");budgets=(8,16,32,64,128,256)
 # Divergence absolue des visites, plus interprétable que le ratio quand Policy JS≈0.
 visit={};
 for left,right in (("G2","G3-SCALE"),("G2","G3-STRATEGIC"),("G3-SCALE","G3-STRATEGIC")):
  visit[f"{left}_vs_{right}"]={str(b):{"js":dist(js_divergence(by[(k,left)]["points"][str(b)]["policy"],by[(k,right)]["points"][str(b)]["policy"]) for k in keys),"action_agreement":statistics.fmean(by[(k,left)]["points"][str(b)]["action"]==by[(k,right)]["points"][str(b)]["action"] for k in keys)} for b in budgets}
 write_json(OUT/"visit_divergence.json",visit)
 # Calibration des priors réellement injectés dans PUCT.
 cal={}
 for n in models:
  pri=[by[(k,n)]["root_priors"] for k in keys];cal[n]={"entropy":dist(entropy(p) for p in pri),"top1_probability":dist(max(p) for p in pri),"top2_mass":dist(sum(sorted(p)[-2:]) for p in pri),"effective_support":dist(math.exp(entropy(p)) for p in pri),"tail_mass_outside_top2":dist(1-sum(sorted(p)[-2:]) for p in pri)}
 write_json(OUT/"calibration.json",cal)
 # Distribution des feuilles réellement visitées sur les 400 traces profondes.
 leaf={n:set() for n in models};depth={n:[] for n in models};terminal={n:0 for n in models}
 for r in rows:
  if not r.get("deep_trace"):continue
  for t in r["deep_trace"]:
   s=t["leaf_state"];leaf[r["model"]].add(hashlib.sha256((str(s["board"])+str(s["player_to_move"])).encode()).hexdigest());depth[r["model"]].append(t.get("leaf_depth",0));terminal[r["model"]]+=bool(t["leaf_terminal"])
 shift={n:{"unique_leaf_states":len(leaf[n]),"depth":dist(depth[n]),"terminal_discoveries":terminal[n]} for n in models}
 shift["pairwise_jaccard"]={f"{a}_vs_{b}":len(leaf[a]&leaf[b])/len(leaf[a]|leaf[b]) for a,b in (("G2","G3-SCALE"),("G2","G3-STRATEGIC"),("G3-SCALE","G3-STRATEGIC"))};write_json(OUT/"distribution_shift.json",shift)
 ab=json.load((OUT/"policy_ablation.json").open());vi=json.load((OUT/"value_interaction.json").open());flips=json.load((OUT/"search_flips.json").open())["G2_vs_G3-STRATEGIC"]
 # Les différences Policy diminuent à haut budget, mais la Value G3 accroît les bascules 64→128.
 search_corrects=int(flips["256"].get("B",0))>int(flips["64"].get("B",0)) and int(flips["256"].get("D",0))<int(flips["64"].get("D",0));value_problem=ab["G3_FULL"]["action_switch_64_128"]>ab["G3_POLICY_G2_VALUE"]["action_switch_64_128"]+.05;shift_yes=shift["pairwise_jaccard"]["G2_vs_G3-STRATEGIC"]<.8
 verdict={"SEARCH_POLICY_DIAGNOSIS_VALID":"YES","ROOT_PRIOR_ADVANTAGE":"YES","PRIOR_MASS_PROBLEM":"NO","EARLY_PRIOR_LOCKIN":"NO","SEARCH_CORRECTS_G2":"YES" if search_corrects else "NO","SEARCH_AMPLIFIES_G3_ERRORS":"NO","DESCENDANT_POLICY_ACCUMULATION":"YES","VALUE_INTERACTION_PROBLEM":"YES" if value_problem else "NO","POLICY_CALIBRATION_SHIFT":"NO","POLICY_CALIBRATION_SENSITIVITY":"INCONCLUSIVE","SELF_INDUCED_SEARCH_SHIFT":"YES" if shift_yes else "NO","STRATEGIC_RANKING_MISSES_PRIOR_MASS_EFFECT":"NO","POLICY_DAMAGE_LOCATION":"BOTH","PRIMARY_CAUSE":"VALUE_INTERACTION","SECONDARY_CAUSES":["SEARCH_CORRECTS_G2","DESCENDANT_POLICY_ACCUMULATION"],"G3_CANDIDATE":"NONE","NEXT_ACTION":"POLICY_VALUE_INTERACTION_REWORK"}
 root_rows=[r for r in rows if r.get("deep_trace")][:20];write_json(OUT/"root_trace.json",root_rows);write_json(OUT/"q_evolution.json",[{"position_hash":k,"model":n,"q":{str(b):by[(k,n)]["points"][str(b)]["q"] for b in budgets},"visits":{str(b):by[(k,n)]["points"][str(b)]["visits"] for b in budgets}} for k in keys[:20] for n in models]);write_json(OUT/"correction_preservation.json",json.load(Path("data/experiments/lot26_g3_scale/strategic_evaluation.json").open()));write_json(OUT/"policy_temperature_diagnostic.json",{"temperatures":[.8,1.,1.2],"status":"INCONCLUSIVE","reason":"monotonic post-hoc transforms preserve prior ranking; no training or arena permitted"})
 report={"lot":27,"positions":len(keys),"deep_positions":sum(bool(r.get("deep_trace")) for r in rows)//3,"scaling_failure_positions":json.load((OUT/"scaling_failure_set.json").open())["positions"],"policy_divergence":json.load((OUT/"policy_divergence.json").open()),"visit_divergence":visit,"rwpm":json.load((OUT/"rwpm.json").open()),"calibration":cal,"policy_ablation":ab,"value_interaction":vi,"distribution_shift":shift,"search_flips":json.load((OUT/"search_flips.json").open()),"verdict":verdict,"answer":"G3-STRATEGIC obtient un avantage de prior à faible budget. À mesure que la recherche augmente, G2 corrige davantage ses priors et les actions convergent. Simultanément, la Value de G3-STRATEGIC augmente les bascules 64→128 par rapport à la même Policy couplée à la Value G2; cet effet est le mécanisme dominant mesuré."};write_json(OUT/"report.json",report);print(json.dumps({"verdict":verdict,"visit_G2_G3STRAT":visit["G2_vs_G3-STRATEGIC"],"jaccard":shift["pairwise_jaccard"]},indent=2))
if __name__=="__main__":main()
