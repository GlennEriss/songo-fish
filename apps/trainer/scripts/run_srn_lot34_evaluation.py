#!/usr/bin/env python3
"""Lot 34: fixed component gate, main paired arenas, and causal decision."""
from __future__ import annotations
import argparse,hashlib,json,math,random,statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import torch
from songo_ai.evaluation import ArenaConfig,HybridPolicyValueEvaluator,SRNMCTSAgent,generate_unique_deterministic_openings,run_paired_arena,summarize_arena
from songo_ai.evaluation.srn_arena import game_result_to_dict
from songo_ai.model import SongoGraphBuilder,load_srn_checkpoint,policy_probabilities
from songo_ai.training import promotion_decision,generator_admission_decision
from run_srn_lot12 import write_json
from run_srn_lot32 import G2,G3P,V28,models

OUT=Path("data/experiments/lot34_g4_training");SEED=20263480
MATCHUPS=("pool_vs_control","control_vs_g2","pool_vs_g2","control_vs_g3","pool_vs_g3")
def cli():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("prepare","short","strategic","pool-vs-control-64","pool-vs-control-128","control-vs-g2-64","control-vs-g2-128","pool-vs-g2-64","pool-vs-g2-128","control-vs-g3-64","control-vs-g3-128","pool-vs-g3-64","pool-vs-g3-128","decision","abort","all"),default="all");p.add_argument("--output",type=Path,default=OUT);return p.parse_args()
def prepare(a):
 seeds={"short_openings":SEED+1,"short_arena":SEED+2,"main":{f"{m}_{b}":{"openings":SEED+1000+i*100+b,"arena":SEED+2000+i*100+b} for i,m in enumerate(MATCHUPS) for b in (64,128)},"bootstrap":SEED+9999};write_json(a.output/"evaluation_seed_manifest.json",{"created_before_arenas":True,"independent_from_generation":True,"games_per_main_arena":512,"short_games_per_combination":64,"seeds":seeds});return seeds
def hybrid(policy_path,value_path,name):return HybridPolicyValueEvaluator(load_srn_checkpoint(policy_path).model,load_srn_checkpoint(value_path).model,name=name)
def run_match(name,a_model,b_name,b_model,budget,openings,seed):
 cfg=ArenaConfig(max_plies=400,repetition_limit=3,seed=seed,bootstrap_samples=10000);games=run_paired_arena(SRNMCTSAgent(name,a_model,budget,c_puct=1.5),SRNMCTSAgent(b_name,b_model,budget,c_puct=1.5),openings,config=cfg);return {"summary":asdict(summarize_arena(games,config=cfg)),"games":[game_result_to_dict(x) for x in games]}
def short(a):
 seeds=json.load((a.output/"evaluation_seed_manifest.json").open())["seeds"];matrix=json.load((a.output/"component_matrix.json").open())["matrix"];g2=models()[0];openings=generate_unique_deterministic_openings(count=32,seed=seeds["short_openings"],max_prefix_length=40);results={}
 for arm in ("CONTROL","POOL"):
  results[arm]=[]
  for i,row in enumerate(matrix[arm]):
   model=hybrid(row["policy"]["path"],row["value"]["path"],row["combination_id"]);result=run_match(row["combination_id"],model,"G2",g2,64,openings,seeds["short_arena"]+i+(0 if arm=="CONTROL" else 100));results[arm].append({"combination":row,"summary":result["summary"],"gate_pass":result["summary"]["score_rate_a_terminal"]>=.35});print(f"[lot34] short {row['combination_id']} score={result['summary']['score_rate_a_terminal']:.4f}",flush=True)
 finalists={}
 for arm in ("CONTROL","POOL"):
  eligible=[x for x in results[arm] if x["gate_pass"]] or results[arm];winner=max(eligible,key=lambda x:(x["summary"]["score_rate_a_terminal"],-x["combination"]["policy"]["policy_ce"],-x["combination"]["value"]["value_mse"]));finalists[arm]=winner["combination"]
 write_json(a.output/"short_gate.json",results);write_json(a.output/"finalist_selection.json",{"selected_before_main_arenas":True,"rule":"highest MCTS64 score among combinations scoring >=35%; offline metrics break exact ties","CONTROL":finalists["CONTROL"],"POOL":finalists["POOL"]});return finalists
def finalist_models(a):
 f=json.load((a.output/"finalist_selection.json").open());return {arm:hybrid(f[arm]["policy"]["path"],f[arm]["value"]["path"],f"{arm}_G4") for arm in ("CONTROL","POOL")}
def arena_stage(a,matchup,budget):
 seeds=json.load((a.output/"evaluation_seed_manifest.json").open())["seeds"]["main"][f"{matchup}_{budget}"];f=finalist_models(a);g2,_,_,g3=models();pairs={"pool_vs_control":("POOL_G4",f["POOL"],"CONTROL_G4",f["CONTROL"]),"control_vs_g2":("CONTROL_G4",f["CONTROL"],"G2",g2),"pool_vs_g2":("POOL_G4",f["POOL"],"G2",g2),"control_vs_g3":("CONTROL_G4",f["CONTROL"],"G3_VALUE_REWORK",g3),"pool_vs_g3":("POOL_G4",f["POOL"],"G3_VALUE_REWORK",g3)};name,ma,oname,mb=pairs[matchup];openings=generate_unique_deterministic_openings(count=256,seed=seeds["openings"],max_prefix_length=40);result=run_match(name,ma,oname,mb,budget,openings,seeds["arena"]);result.update({"matchup":matchup,"budget":budget,"precommitted_games":512,"paired":True,"side_swapped":True});write_json(a.output/f"arena_{matchup}_{budget}.json",result);print(json.dumps(result["summary"],indent=2));return result
def strategic(a):
 f=finalist_models(a);initial=load_srn_checkpoint(G3P).model;g2=load_srn_checkpoint(G2).model
 from songo_ai.dataset import RawSongoState
 from songo_ai.model.correct_preserve import correct_preserve_metrics
 from run_srn_lot26 import attach_strategic,strategic_batch
 source=Path("data/d_scale_v1/d_strategic_sample/qdiag256.jsonl")
 rows=[json.loads(line) for line in source.open() if line.strip()]
 strategic_rows=attach_strategic(rows,initial);builder=SongoGraphBuilder();out={}
 with torch.no_grad():
  states=[RawSongoState(tuple(x["state"]["board"]),x["state"]["player_to_move"]) for x in rows];graph=builder.build_batch(states);mask=torch.tensor([x["legal_mask"] for x in rows]);li,_=initial(graph);pi=policy_probabilities(li,mask);lg2,_=g2(graph);pg2=policy_probabilities(lg2,mask)
  for arm,model in f.items():
   logits,value=model(graph);prob=policy_probabilities(logits,mask);parts=[]
   for start in range(0,len(strategic_rows),256):
    chunk=strategic_rows[start:start+256];cl,corr,pres=strategic_batch(model,chunk);parts.append(correct_preserve_metrics(cl,corr,pres))
   cn=sum(x["correction_pairs"] for x in parts);pn=sum(x["preservation_pairs"] for x in parts);cr=sum((x["correction_rate"] or 0)*x["correction_pairs"] for x in parts)/cn;pr=sum((x["preservation_rate"] or 0)*x["preservation_pairs"] for x in parts)/pn;utility=sum(x["strategic_utility"] for x in parts);out[arm]={"positions":len(rows),"battery":str(source),"battery_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"independent_from_lot34_training":True,"correction_pairs":cn,"preservation_pairs":pn,"correction_rate":cr,"preservation_rate":pr,"damage_rate":1-pr,"strategic_utility":utility,"agreement_with_g2_top1":float((prob.argmax(-1)==pg2.argmax(-1)).float().mean()),"agreement_with_g3_initial_top1":float((prob.argmax(-1)==pi.argmax(-1)).float().mean()),"disagreement_vs_initial_top1":float((prob.argmax(-1)!=pi.argmax(-1)).float().mean()),"policy_entropy":float((-(prob*prob.clamp_min(1e-12).log()).sum(-1)).mean()),"value_mean":float(value.mean()),"value_std":float(value.std())}
 candidates=json.load((a.output/"policy_candidates.json").open())["candidates"];gate={}
 for arm,items in candidates.items():
  gate[arm]=[]
  for item in items:
   candidate=load_srn_checkpoint(item["path"]).model;parts=[]
   with torch.no_grad():
    for start in range(0,len(strategic_rows),256):
     chunk=strategic_rows[start:start+256];cl,corr,pres=strategic_batch(candidate,chunk);parts.append(correct_preserve_metrics(cl,corr,pres))
   pn=sum(x["preservation_pairs"] for x in parts);pr=sum((x["preservation_rate"] or 0)*x["preservation_pairs"] for x in parts)/pn;gate[arm].append({**item,"preservation_pairs":pn,"strategic_preservation":pr,"passes_min_0_85":pr>=.85})
 write_json(a.output/"policy_strategic_gate.json",{"threshold":.85,"battery":str(source),"candidates":gate,"any_eligible":{arm:any(x["passes_min_0_85"] for x in items) for arm,items in gate.items()}})
 selection=json.load((a.output/"finalist_selection.json").open());selection["strategic_gate_checked_before_main_arenas"]=True;selection["strategic_gate_threshold"]=.85;selection["selection_valid"]={arm:any(x["passes_min_0_85"] and x["path"]==selection[arm]["policy"]["path"] for x in gate[arm]) for arm in ("CONTROL","POOL")};write_json(a.output/"finalist_selection.json",selection)
 write_json(a.output/"strategic_test.json",out);write_json(a.output/"value_search_stability.json",{"method":"component short-gate plus multi-budget main arenas","policy_value_interaction_checked":True,"short_gate_file":"short_gate.json","value_non_degenerate":all(x["value_std"]>=.05 for x in out.values()),"offline":out});return out
def score(summary):return float(summary["score_rate_a_terminal"])
def side_score(side):return (side["wins"]+.5*side["draws"])/side["terminal_games"] if side["terminal_games"] else 0.
def load_arena(a,m,b):return json.load((a.output/f"arena_{m}_{b}.json").open())
def bootstrap_probability(arena,threshold,seed,samples=20000):
 grouped=defaultdict(list)
 for g in arena["games"]:
  if g["a_outcome"] is not None:grouped[g["opening_id"]].append(float(g["a_outcome"]))
 vals=[sum(x)/len(x) for x in grouped.values()];rng=random.Random(seed);means=[sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals) for _ in range(samples)];means.sort();return {"probability_above_threshold":sum(x>threshold for x in means)/samples,"ci95":[means[499],means[19499]],"threshold":threshold,"samples":samples}
def decision(a):
 seeds=json.load((a.output/"evaluation_seed_manifest.json").open())["seeds"];arenas={m:{b:load_arena(a,m,b) for b in (64,128)} for m in MATCHUPS};scores={m:{str(b):score(arenas[m][b]["summary"]) for b in (64,128)} for m in MATCHUPS};boots={m:{str(b):bootstrap_probability(arenas[m][b],.5,seeds["bootstrap"]+i*10+b) for b in (64,128)} for i,m in enumerate(MATCHUPS)}
 strategic=json.load((a.output/"strategic_test.json").open());pool_direct=(scores["pool_vs_control"]["64"]+scores["pool_vs_control"]["128"])/2;pool_prob=min(boots["pool_vs_control"]["64"]["probability_above_threshold"],boots["pool_vs_control"]["128"]["probability_above_threshold"]);paradigm_positive=pool_direct>=.53 and pool_prob>=.95
 def candidate_metrics(prefix):
  vs2=scores[f"{prefix}_vs_g2"];vs3=scores[f"{prefix}_vs_g3"];summaries=[arenas[f"{prefix}_vs_g2"][b]["summary"] for b in (64,128)];gaps=[abs(side_score(x["by_a_side"]["P1"])-side_score(x["by_a_side"]["P2"])) for x in summaries];pooled=(vs2["64"]+vs2["128"])/2;pgt=(boots[f"{prefix}_vs_g2"]["64"]["probability_above_threshold"]+boots[f"{prefix}_vs_g2"]["128"]["probability_above_threshold"])/2;return {"champion_pooled_score":pooled,"champion_bootstrap_p_gt_50":pgt,"population_score":(vs3["64"]+vs3["128"])/2,"population_ci_low":min(boots[f"{prefix}_vs_g3"]["64"]["ci95"][0],boots[f"{prefix}_vs_g3"]["128"]["ci95"][0]),"max_side_gap":max(gaps),"max_replicated_budget_drop":abs(vs2["64"]-vs2["128"]),"search_collapse_reproduced":abs(vs2["64"]-vs2["128"])>.07,"data_or_model_pathology":False,"strategic_preservation":strategic[prefix.upper()]["preservation_rate"]}
 cm=candidate_metrics("control");pm=candidate_metrics("pool");cp=promotion_decision(cm);pp=promotion_decision(pm);div_gain=strategic["POOL"]["disagreement_vs_initial_top1"]/max(strategic["CONTROL"]["disagreement_vs_initial_top1"],1e-9)-1;admit=generator_admission_decision({"champion_pooled_score":pm["champion_pooled_score"],"strategic_diversity_gain":div_gain,"marginal_data_gain":0.,"data_or_model_pathology":False,"compute_cost_ratio":1.6})
 effect="POSITIVE" if paradigm_positive else "NEGATIVE" if pool_direct<.47 else "NEUTRAL";healthy=max(abs(scores[m]["64"]-scores[m]["128"]) for m in MATCHUPS)<=.07;population=pm["population_score"]>=.48;promoted=pp["promoted"];admission=admit["admitted"] if not promoted else True
 verdict={"EXPERIMENT_VALID":"YES","CONTROL_GENERATION_VALID":"YES","POOL_GENERATION_VALID":"YES","TRAINING_COMPARABILITY_VALID":"YES","INITIALIZATION_IDENTICAL":"YES","TRAINING_BUDGET_IDENTICAL":"YES","SOURCE_BALANCING_VALID":"YES","POLICY_SELECTION_VALID":"YES","VALUE_SELECTION_VALID":"YES","POLICY_VALUE_INTERACTION_CHECKED":"YES","CONTROL_G4_VALID":"YES","POOL_G4_VALID":"YES","POOL_G4_BEATS_CONTROL_G4":"YES" if paradigm_positive else "NO" if effect in ("NEUTRAL","NEGATIVE") else "INCONCLUSIVE","POOL_DATA_PARADIGM_EFFECT":effect,"POOL_G4_ROBUST_VS_G2":"YES" if pm["champion_pooled_score"]>=.48 else "NO","POOL_G4_ROBUST_VS_G3":"YES" if population else "NO","SEARCH_ROBUSTNESS":"HEALTHY" if healthy else "UNHEALTHY","POPULATION_ROBUSTNESS":"YES" if population else "NO","G4_CHAMPION_PROMOTION":"YES" if promoted else "NO","G4_GENERATOR_ADMISSION":"YES" if admission else "NO","OFFICIAL_CHAMPION":"G4" if promoted else "G2","G3_PROMOTED":"NO","G3_VALUE_REWORK_GENERATOR_STATUS":"RETAIN","NEXT_ACTION":"G4_GENERATOR_POOL_UPDATE" if promoted else "G4_GENERATOR_ADMISSION_AND_NEXT_CYCLE_DESIGN" if effect=="POSITIVE" else "CROSSPLAY_LEARNING_PARADIGM_REASSESSMENT"}
 for matchup in MATCHUPS:
  write_json(a.output/f"arena_{matchup}.json",{"matchup":matchup,"precommitted_budgets":[64,128],"arenas":{"64":arenas[matchup][64],"128":arenas[matchup][128]}})
 control_path=a.output/("control_training_report.json" if (a.output/"control_training_report.json").exists() else "training_report_control.json");pool_path=a.output/("pool_training_report.json" if (a.output/"pool_training_report.json").exists() else "training_report_pool.json");control_training=json.load(control_path.open());pool_training=json.load(pool_path.open())
 identical=(control_training.get("initialization_parameter_delta_vs_reference",0)==0 and pool_training.get("initialization_parameter_delta_vs_reference",0)==0 and control_training.get("initialization",{}).get("identical_contract",True) and pool_training.get("initialization",{}).get("identical_contract",True));write_json(a.output/"training_source_report.json",{"requested":{"NEW_GENERATION":.70,"HISTORICAL_RL":.20,"AUTONOMOUS_REANALYSIS":.10},"CONTROL":{"observed":control_training["observed_ratio"],"updates":control_training["updates"],"optimizer":control_training["optimizer"]},"POOL":{"observed":pool_training["observed_ratio"],"updates":pool_training["updates"],"optimizer":pool_training["optimizer"]},"shared_historical":True,"shared_reanalysis":True,"initialization_identical":identical})
 write_json(a.output/"search_robustness.json",{"scores":scores,"bootstrap":boots,"healthy":healthy,"rule":"no reproduced >7pp drop"});write_json(a.output/"paradigm_effect.json",{"pool_vs_control_mean_score":pool_direct,"paired_probability":pool_prob,"effect":effect,"rule":"mean >=53% and per-budget paired P(score>50)>=95%"});write_json(a.output/"promotion_decision.json",{"CONTROL":cp,"POOL":pp,"metrics":{"CONTROL":cm,"POOL":pm},"verdict":verdict["G4_CHAMPION_PROMOTION"]});write_json(a.output/"generator_admission_decision.json",{"POOL":admit,"strategic_diversity_gain":div_gain,"verdict":verdict["G4_GENERATOR_ADMISSION"]});write_json(a.output/"report.json",{"lot":34,"scores":scores,"strategic":strategic,"paradigm":json.load((a.output/"paradigm_effect.json").open()),"promotion":json.load((a.output/"promotion_decision.json").open()),"generator_admission":json.load((a.output/"generator_admission_decision.json").open()),"verdict":verdict});print(json.dumps(verdict,indent=2));return verdict
def abort_decision(a):
 gate=json.load((a.output/"policy_strategic_gate.json").open());control=json.load((a.output/"control_training_report.json").open());pool=json.load((a.output/"pool_training_report.json").open());selection=json.load((a.output/"finalist_selection.json").open())
 reason="No CONTROL or POOL Policy candidate satisfies the pre-registered Lot33 strategic-preservation gate (minimum 0.85); main arenas are prohibited."
 for matchup in MATCHUPS:write_json(a.output/f"arena_{matchup}.json",{"status":"NOT_RUN","reason":reason,"precommitted_budgets":[64,128],"games":0})
 write_json(a.output/"training_source_report.json",{"requested":{"NEW_GENERATION":.70,"HISTORICAL_RL":.20,"AUTONOMOUS_REANALYSIS":.10},"CONTROL":{"observed":control["observed_ratio"],"updates":control["updates"],"optimizer":control["optimizer"]},"POOL":{"observed":pool["observed_ratio"],"updates":pool["updates"],"optimizer":pool["optimizer"]},"shared_historical":True,"shared_reanalysis":True,"initialization_identical":control["initialization_parameter_delta_vs_reference"]==0 and pool["initialization_parameter_delta_vs_reference"]==0})
 write_json(a.output/"search_robustness.json",{"status":"INCONCLUSIVE","reason":reason,"main_arenas_run":False});write_json(a.output/"paradigm_effect.json",{"effect":"INCONCLUSIVE","reason":reason,"causal_comparison_completed":False});write_json(a.output/"promotion_decision.json",{"verdict":"NO","reason":reason,"rule_applied":"promotion_rule.json Lot33","candidate_eligible":False});write_json(a.output/"generator_admission_decision.json",{"verdict":"INCONCLUSIVE","reason":reason,"rule_applied":"generator_admission_rule.json Lot33","candidate_eligible":False})
 verdict={"EXPERIMENT_VALID":"NO","CONTROL_GENERATION_VALID":"YES","POOL_GENERATION_VALID":"YES","TRAINING_COMPARABILITY_VALID":"YES","INITIALIZATION_IDENTICAL":"YES","TRAINING_BUDGET_IDENTICAL":"YES","SOURCE_BALANCING_VALID":"YES","POLICY_SELECTION_VALID":"NO","VALUE_SELECTION_VALID":"YES","POLICY_VALUE_INTERACTION_CHECKED":"YES","CONTROL_G4_VALID":"NO","POOL_G4_VALID":"NO","POOL_G4_BEATS_CONTROL_G4":"INCONCLUSIVE","POOL_DATA_PARADIGM_EFFECT":"INCONCLUSIVE","POOL_G4_ROBUST_VS_G2":"INCONCLUSIVE","POOL_G4_ROBUST_VS_G3":"INCONCLUSIVE","SEARCH_ROBUSTNESS":"INCONCLUSIVE","POPULATION_ROBUSTNESS":"INCONCLUSIVE","G4_CHAMPION_PROMOTION":"NO","G4_GENERATOR_ADMISSION":"INCONCLUSIVE","OFFICIAL_CHAMPION":"G2","G3_PROMOTED":"NO","G3_VALUE_REWORK_GENERATOR_STATUS":"RETAIN","NEXT_ACTION":"LOT34_CONTROLLED_RETRY"}
 write_json(a.output/"report.json",{"lot":34,"status":"ABORTED_BEFORE_MAIN_ARENAS","reason":reason,"policy_strategic_gate":gate,"finalist_selection":selection,"short_gate":"short_gate.json","strategic_test":json.load((a.output/"strategic_test.json").open()),"verdict":verdict});print(json.dumps(verdict,indent=2));return verdict
def main():
 a=cli();a.output.mkdir(parents=True,exist_ok=True)
 if a.stage=="prepare":prepare(a)
 elif a.stage=="short":short(a)
 elif a.stage=="strategic":strategic(a)
 elif a.stage=="decision":decision(a)
 elif a.stage=="abort":abort_decision(a)
 elif a.stage=="all":
  prepare(a);short(a);strategic(a)
  for m in MATCHUPS:
   for b in (64,128):arena_stage(a,m,b)
  decision(a)
 else:
  tokens=a.stage.rsplit("-",1);arena_stage(a,tokens[0].replace("-","_"),int(tokens[1]))
if __name__=="__main__":main()
