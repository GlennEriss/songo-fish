#!/usr/bin/env python3
"""Lot 35R: resolve the unique G4 champion identity with paired arenas."""
from __future__ import annotations
import argparse,json,random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from songo_ai.evaluation import ArenaConfig,HybridPolicyValueEvaluator,SRNMCTSAgent,generate_unique_deterministic_openings,run_paired_arena,summarize_arena
from songo_ai.evaluation.srn_arena import game_result_to_dict
from songo_ai.model import load_srn_checkpoint
from run_srn_lot12 import sha256,write_json

OUT=Path("data/experiments/lot35r_champion_identity");L35=Path("data/experiments/lot35_generator_pool");THRESHOLD=.95
def cli():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("prepare","phase1","decide1","phase2","finalize"),required=True);p.add_argument("--output",type=Path,default=OUT);return p.parse_args()
def candidates():
 identity=json.load((L35/"g4_champion_identity.json").open())["candidates"]
 return identity,{k:HybridPolicyValueEvaluator(load_srn_checkpoint(v["policy_checkpoint"]).model,load_srn_checkpoint(v["value_checkpoint"]).model,name=f"{k}_G4R") for k,v in identity.items()}
def prepare(a):
 a.output.mkdir(parents=True,exist_ok=True);identity,_=candidates();write_json(a.output/"configuration.json",{"lot":"35R","purpose":"CHAMPION_IDENTITY_RESOLUTION_ONLY","phase1_games":1024,"phase1_budget":128,"phase2_games":1024,"phase2_budget":256,"paired":True,"side_swapped":True,"bootstrap_probability_threshold":THRESHOLD,"no_mcts512":True,"training_performed":False,"g5_generation_performed":False,"pruning_performed":False,"engine_modified":False,"srn_modified":False,"mcts_modified":False});write_json(a.output/"candidate_identity.json",{"exactly_two":True,"candidates":identity,"fingerprints_frozen_before_arena":True});write_json(a.output/"phase1_seed_manifest.json",{"created_before_results":True,"independent_from_lot34r":True,"openings_seed":20263501,"arena_seed":20263511,"bootstrap_seed":20263521,"opening_pairs":512,"games":1024,"budget":128});write_json(a.output/"phase2_seed_manifest.json",{"created_before_phase1_results":True,"independent_from_phase1_and_lot34r":True,"openings_seed":20263502,"arena_seed":20263512,"bootstrap_seed":20263522,"opening_pairs":512,"games":1024,"budget":256,"conditional_execution":True});return identity
def arena(a,phase,budget):
 manifest=json.load((a.output/f"{phase}_seed_manifest.json").open());_,models=candidates();openings=generate_unique_deterministic_openings(count=512,seed=manifest["openings_seed"],max_prefix_length=40);cfg=ArenaConfig(max_plies=400,repetition_limit=3,seed=manifest["arena_seed"],bootstrap_samples=10000);games=run_paired_arena(SRNMCTSAgent("POOL_G4R",models["POOL"],budget,c_puct=1.5),SRNMCTSAgent("CONTROL_G4R",models["CONTROL"],budget,c_puct=1.5),openings,config=cfg);result={"phase":phase,"budget":budget,"paired":True,"side_swapped":True,"summary":asdict(summarize_arena(games,config=cfg)),"games":[game_result_to_dict(x) for x in games]};write_json(a.output/("phase1_mcts128.json" if phase=="phase1" else "phase2_mcts256.json"),result);print(json.dumps(result["summary"],indent=2));return result
def bootstrap(arena,seed,samples=50000):
 groups=defaultdict(list)
 for game in arena["games"]:
  if game["a_outcome"] is not None:groups[game["opening_id"]].append(float(game["a_outcome"]))
 values=[sum(v)/len(v) for v in groups.values()];rng=random.Random(seed);means=[sum(values[rng.randrange(len(values))] for _ in values)/len(values) for _ in range(samples)];means.sort();p_pool=sum(x>.5 for x in means)/samples;p_control=sum(x<.5 for x in means)/samples;return {"bootstrap_unit":"opening_pair","effective_pairs":len(values),"samples":samples,"P_POOL_GT_CONTROL":p_pool,"P_CONTROL_GT_POOL":p_control,"ci95":[means[int(.025*samples)],means[int(.975*samples)-1]],"threshold":THRESHOLD}
def decide(a,phase):
 arena_path=a.output/("phase1_mcts128.json" if phase=="phase1" else "phase2_mcts256.json");arena_data=json.load(arena_path.open());manifest=json.load((a.output/f"{phase}_seed_manifest.json").open());boot=bootstrap(arena_data,manifest["bootstrap_seed"]);decision="POOL" if boot["P_POOL_GT_CONTROL"]>=THRESHOLD else "CONTROL" if boot["P_CONTROL_GT_POOL"]>=THRESHOLD else "INCONCLUSIVE";summary=arena_data["summary"];valid=summary["games"]==1024 and summary["by_a_side"]["P1"]["games"]==summary["by_a_side"]["P2"]["games"]==512;payload={f"{phase.upper()}_VALID":"YES" if valid else "NO",f"{phase.upper()}_GAMES":summary["games"],f"{phase.upper()}_BUDGET":f"MCTS{arena_data['budget']}",f"{phase.upper()}_POOL_SCORE":summary["score_rate_a_terminal"],f"{phase.upper()}_CONTROL_SCORE":1-summary["score_rate_a_terminal"],**boot,f"{phase.upper()}_DECISION":decision};write_json(a.output/f"{phase}_bootstrap.json",boot);write_json(a.output/f"{phase}_decision.json",payload);print(json.dumps(payload,indent=2));return payload
def finalize(a):
 p1=json.load((a.output/"phase1_decision.json").open());p2_path=a.output/"phase2_decision.json";p2=json.load(p2_path.open()) if p2_path.exists() else None;winner=p1["PHASE1_DECISION"] if p1["PHASE1_DECISION"]!="INCONCLUSIVE" else p2["PHASE2_DECISION"] if p2 else "INCONCLUSIVE";resolved=winner in ("POOL","CONTROL");identity=json.load((a.output/"candidate_identity.json").open())["candidates"];champ=identity[winner] if resolved else None;alternate="CONTROL" if winner=="POOL" else "POOL" if winner=="CONTROL" else None
 decision={"G4_CHAMPION_IDENTITY_RESOLVED":"YES" if resolved else "NO","G4_CHAMPION_SOURCE":f"{winner}_G4R" if resolved else "UNRESOLVED","G4_ALTERNATE_SOURCE":f"{alternate}_G4R" if resolved else "UNRESOLVED","OFFICIAL_CHAMPION":f"G4_{winner}_G4R" if resolved else "G4_ALIAS_IDENTITY_UNRESOLVED","G4_CHAMPION_POLICY_CHECKPOINT":champ["policy_checkpoint"] if champ else None,"G4_CHAMPION_VALUE_CHECKPOINT":champ["value_checkpoint"] if champ else None,"G4_CHAMPION_POLICY_FINGERPRINT":champ["policy_fingerprint"] if champ else None,"G4_CHAMPION_VALUE_FINGERPRINT":champ["value_fingerprint"] if champ else None,"PHASE2_EXECUTED":"YES" if p2 else "NO","no_pruning":True,"G5_TRAINING_PERFORMED":"NO","G5_MASSIVE_DATA_GENERATION_PERFORMED":"NO"};write_json(a.output/"champion_identity_decision.json",decision)
 registry={"updated":resolved,"alias_G4":decision["OFFICIAL_CHAMPION"],"G4_CHAMPION":{**champ,"roles":["CHAMPION","ACTIVE_GENERATOR"]} if champ else None,"G4_ALTERNATE":{**identity[alternate],"roles":["PENDING_GENERATOR_EVALUATION"]} if alternate else None,"ACTIVE_GENERATOR_POOL":[decision["OFFICIAL_CHAMPION"]] if resolved else [],"HISTORICAL_OPPONENT_POOL":["G2","G3_VALUE_REWORK"],"fingerprints_unchanged":all(sha256(Path(x["policy_checkpoint"]))==x["policy_fingerprint"] and sha256(Path(x["value_checkpoint"]))==x["value_fingerprint"] for x in identity.values())};write_json(a.output/"model_registry_update.json",registry);next_action="RESUME_LOT35_GENERATOR_POOL_PRUNING" if resolved else "G4_COCHAMPION_REGISTRY_DESIGN";write_json(a.output/"next_action.json",{"NEXT_ACTION":next_action});write_json(a.output/"report.json",{"lot":"35R","status":"COMPLETE","phase1":p1,"phase2":p2,"decision":decision,"registry":registry,"NEXT_ACTION":next_action});print(json.dumps({**decision,"NEXT_ACTION":next_action},indent=2));return decision
def main():
 a=cli()
 if a.stage=="prepare":prepare(a)
 elif a.stage=="phase1":arena(a,"phase1",128)
 elif a.stage=="decide1":decide(a,"phase1")
 elif a.stage=="phase2":
  p1=json.load((a.output/"phase1_decision.json").open())
  if p1["PHASE1_DECISION"]!="INCONCLUSIVE":raise RuntimeError("Phase1 resolved identity; MCTS256 forbidden")
  arena(a,"phase2",256);decide(a,"phase2")
 elif a.stage=="finalize":finalize(a)
if __name__=="__main__":main()
