#!/usr/bin/env python3
"""Lot 31 : génération comparative G2-only versus generator pool, sans gradient."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics,time
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
import torch
from songo_ai.dataset import read_d_rl_jsonl,write_d_rl_jsonl
from songo_ai.evaluation import action_metrics,crossplay_only_states,marginal_novelty,overlap,physical_state_key,source_state_sets,state_coverage
from songo_ai.evaluation.search_policy_diagnosis import js_divergence
from songo_ai.generation import MatchupQuota,ModelRole,SelfPlayConfig,SelfPlayRunner,deterministic_schedule,source_balanced_indices
from songo_ai.model import SongoGraphBuilder,load_srn_checkpoint,policy_probabilities
from songo_ai.search import MCTSConfig
from run_srn_lot12 import sha256,write_json
from songo_ai.evaluation.hybrid_evaluator import HybridPolicyValueEvaluator

G2=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt");G3P=Path("data/experiments/lot26_g3_scale/checkpoints/g3_strategic_best.pt");V28=Path("data/experiments/lot28_value_search/checkpoints/v28_a_best.pt")
OUT=Path("data/experiments/lot31_generator_pool");SEED=20263131;SHARD_GAMES=20
THRESHOLDS={"material_normalized_gain":.02,"strategic_relative_gain":.05,"crossplay_unique_fraction":.02,"pathology_pool_minus_control_pp":.02,"max_truncation_rate":.05,"max_elapsed_per_sim_ratio":2.0,"high_policy_js":.001}

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("identity","control","pool","analyze","all"),default="all");p.add_argument("--output",type=Path,default=OUT);return p.parse_args()
def fingerprint(model,keys=None):
 h=hashlib.sha256();state=model.state_dict();keys=sorted(keys or state)
 for k in keys:h.update(k.encode());h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def load_models():
 g2=load_srn_checkpoint(G2).model;p=load_srn_checkpoint(G3P).model;v=load_srn_checkpoint(V28).model
 for m in (g2,p,v):m.eval()
 return g2,p,v,HybridPolicyValueEvaluator(p,v,name="G3_VALUE_REWORK")
def identity(a):
 g2,p,v,g3=load_models();result={"G2":{"checkpoint":str(G2),"sha256":sha256(G2),"fingerprint":fingerprint(g2),"roles":["CHAMPION","GENERATOR"]},"G3_VALUE_REWORK":{"policy_checkpoint":str(G3P),"policy_sha256":sha256(G3P),"value_checkpoint":str(V28),"value_sha256":sha256(V28),"policy_fingerprint":fingerprint(p,[k for k in p.state_dict() if not k.startswith("value_mlp.")]),"value_fingerprint":fingerprint(v,[k for k in v.state_dict() if k.startswith("value_mlp.")]),"roles":["PROBATIONARY_GENERATOR"],"G3_PROMOTED":False},"architecture_fingerprint":hashlib.sha256(Path("packages/songo_ai/model/srn_network.py").read_bytes()+Path("packages/songo_ai/model/srn_graph.py").read_bytes()).hexdigest(),"valid":True};write_json(a.output/"model_identity.json",result);return result
def quotas(arm):
 if arm=="CONTROL":return (MatchupQuota("G2","G2",ModelRole.CHAMPION,ModelRole.CHAMPION,400,"CONTROL_G2_SELFPLAY"),)
 return (MatchupQuota("G2","G2",ModelRole.CHAMPION,ModelRole.CHAMPION,100,"POOL_G2_SELFPLAY"),MatchupQuota("G3_VALUE_REWORK","G3_VALUE_REWORK",ModelRole.GENERATOR,ModelRole.GENERATOR,100,"POOL_G3_SELFPLAY"),MatchupQuota("G2","G3_VALUE_REWORK",ModelRole.CHAMPION,ModelRole.GENERATOR,100,"POOL_CROSSPLAY_G2_P1"),MatchupQuota("G3_VALUE_REWORK","G2",ModelRole.GENERATOR,ModelRole.CHAMPION,100,"POOL_CROSSPLAY_G3_P1"))
def source_type(game):
 if game.p1_model_id==game.p2_model_id=="G2":return "G2_G2"
 if game.p1_model_id==game.p2_model_id=="G3_VALUE_REWORK":return "G3_G3"
 return "CROSS_PLAY"
def generation_config(game,identity):
 return SelfPlayConfig(games=1,max_game_plies=400,repetition_limit=3,target_temperature=1.,action_temperature=1.,temperature_drop_ply=30,late_action_temperature=0.,seed=game.seed,generation=31,checkpoint_id=f"{game.p1_model_id}_vs_{game.p2_model_id}",provenance={"experiment_arm":"CONTROL" if game.provenance.startswith("CONTROL") else "POOL","generation_id":game.generation_id,"scheduled_game_id":game.game_id,"p1_model_id":game.p1_model_id,"p2_model_id":game.p2_model_id,"p1_role":game.p1_role.value,"p2_role":game.p2_role.value,"p1_fingerprint":identity[game.p1_model_id]["fingerprint"] if game.p1_model_id=="G2" else identity[game.p1_model_id]["policy_fingerprint"]+":"+identity[game.p1_model_id]["value_fingerprint"],"p2_fingerprint":identity[game.p2_model_id]["fingerprint"] if game.p2_model_id=="G2" else identity[game.p2_model_id]["policy_fingerprint"]+":"+identity[game.p2_model_id]["value_fingerprint"],"p1_role_registry":game.p1_role.value,"p2_role_registry":game.p2_role.value,"mcts_budget":64,"generator_role":game.provenance,"provenance":game.provenance,"source_type":source_type(game),"lot":31,"canonicalization":False},include_truncated_examples=True,mcts=MCTSConfig(num_simulations=64,c_puct=1.5,dirichlet_alpha=.3,dirichlet_epsilon=.25,add_root_noise=True))
def generate_arm(a,arm):
 ident=json.load((a.output/"model_identity.json").open());g2,p,v,g3=load_models();models={"G2":g2,"G3_VALUE_REWORK":g3};schedule=deterministic_schedule(f"LOT31_{arm}_V1",quotas(arm),seed=SEED+(0 if arm=="CONTROL" else 10000),mcts_budget=64);root=a.output/arm.lower();root.mkdir(exist_ok=True);manifest=[]
 for start in range(0,len(schedule),SHARD_GAMES):
  chunk=schedule[start:start+SHARD_GAMES];part=start//SHARD_GAMES;path=root/f"part-{part:04d}.jsonl";meta=root/f"part-{part:04d}.manifest.json"
  if path.exists() and meta.exists():m=json.load(meta.open());print(f"[lot31] {arm} shard {part+1} reused",flush=True)
  else:
   began=time.perf_counter();examples=[];games=[]
   for game in chunk:
    cfg=generation_config(game,ident);run=SelfPlayRunner(models[game.p1_model_id],cfg,models_by_player={1:models[game.p1_model_id],2:models[game.p2_model_id]}).play_game(0);examples.extend(run.examples);games.append({"scheduled_game_id":game.game_id,"game_id":run.game_id,"p1":game.p1_model_id,"p2":game.p2_model_id,"source_type":source_type(game),"status":run.status.value,"winner":run.winner,"plies":run.num_plies,"total_mcts_simulations":run.total_mcts_simulations,"network_evaluations":run.total_network_evaluations,"elapsed_s":run.elapsed_s,"action_sequence":list(run.action_sequence)})
   write_d_rl_jsonl(path,examples);m={"arm":arm,"part":part,"games":len(games),"positions":len(examples),"sha256":sha256(path),"elapsed_s":time.perf_counter()-began,"game_records":games};write_json(meta,m);print(f"[lot31] {arm} shard {part+1}/{math.ceil(len(schedule)/SHARD_GAMES)}: {len(games)} games, {len(examples)} positions",flush=True)
  manifest.append(m)
 out={"arm":arm,"planned_games":len(schedule),"actual_games":sum(x["games"] for x in manifest),"positions":sum(x["positions"] for x in manifest),"sources":dict(Counter(source_type(x) for x in schedule)),"shards":len(manifest),"mcts_budget":64,"configuration_equal_across_arms":True,"total_mcts_simulations":sum(g["total_mcts_simulations"] for x in manifest for g in x["game_records"]),"elapsed_s":sum(x["elapsed_s"] for x in manifest),"terminal_games":sum(g["status"] in ("TERMINAL_WIN","TERMINAL_DRAW") for x in manifest for g in x["game_records"]),"truncated_games":sum(g["status"].startswith("TRUNCATED") for x in manifest for g in x["game_records"])};write_json(a.output/("control_manifest.json" if arm=="CONTROL" else "pool_manifest.json"),out);return out
def read_arm(root):return [x for p in sorted(root.glob("part-*.jsonl")) for x in read_d_rl_jsonl(p)]
def historical_states():
 states=set()
 for base in (Path("data/d_scale_v1/d_selfplay_large"),Path("data/d_rl")):
  for p in sorted(base.glob("*.jsonl")):
   try:states.update(physical_state_key(x) for x in read_d_rl_jsonl(p))
   except ValueError:continue
 for p in sorted(Path("data/d_scale_v1/d_reanalysis_large").glob("*.jsonl")):
  for line in p.open():
   r=json.loads(line);s=r["state"];states.add((tuple(s["board"]),int(s["player_to_move"])))
 return states
def trajectory_metrics(examples):
 games=defaultdict(list)
 for x in examples:games[x.metadata["game_id"]].append(x)
 sig=[];prefix=[];length=[]
 for rows in games.values():
  rows.sort(key=lambda x:int(x.metadata["ply"]));actions=tuple(int(x.metadata["action_played"]) for x in rows);sig.append(hashlib.sha256(bytes(actions)).hexdigest());prefix.append(actions[:10]);length.append(len(rows))
 return {"games":len(games),"unique_trajectory_signatures":len(set(sig)),"unique_prefix10":len(set(prefix)),"prefix10_overlap_rate":1-len(set(prefix))/len(prefix),"length_mean":statistics.fmean(length),"length_median":statistics.median(length),"length_min":min(length),"length_max":max(length)}
def phase_coverage(examples):
 """Couverture brute/unique par phase, sans feature experte ni canonicalisation."""
 phases=defaultdict(list)
 for x in examples:
  ply=int(x.metadata["ply"]);seeds=sum(x.state.board[:14])
  phase="OPENING" if ply<20 else "ENDGAME" if seeds<=20 else "MIDGAME"
  phases[phase].append(x)
 return {phase:state_coverage(phases.get(phase,[])) for phase in ("OPENING","MIDGAME","ENDGAME")}
def terminal_metrics(examples):
 """Une observation par partie pour éviter de compter chaque position terminalisée."""
 games={}
 for x in examples:games.setdefault(x.metadata["game_id"],x)
 status=Counter(str(x.metadata["status"]) for x in games.values());winners=Counter("DRAW" if str(x.metadata["status"])=="TERMINAL_DRAW" else "TRUNCATED" if str(x.metadata["status"]).startswith("TRUNCATED") else f"P{x.metadata.get('winner')}" for x in games.values())
 scores=[x.metadata.get("final_score") for x in games.values() if x.metadata.get("final_score") is not None]
 return {"games":len(games),"status_counts":dict(status),"winner_counts":dict(winners),"truncation_rate":sum(v for k,v in status.items() if k.startswith("TRUNCATED"))/len(games),"mean_final_stores":[statistics.fmean(s[i] for s in scores) for i in (0,1)] if scores else None}
def policy_diversity(examples,g2,g3,limit=50000):
 unique={physical_state_key(x):x for x in examples};rows=sorted(unique.values(),key=lambda x:hashlib.sha256(repr(physical_state_key(x)).encode()).hexdigest())[:limit];builder=SongoGraphBuilder();js=[];dis=[];top2=[];e2=[];e3=[]
 with torch.no_grad():
  for start in range(0,len(rows),512):
   c=rows[start:start+512];graph=builder.build_batch([x.state for x in c]);mask=torch.tensor([x.legal_mask for x in c]);l2,_=g2(graph);l3,_=g3(graph);p2=policy_probabilities(l2,mask).tolist();p3=policy_probabilities(l3,mask).tolist()
   for a,b in zip(p2,p3):js.append(js_divergence(a,b));dis.append(max(range(7),key=lambda i:a[i])!=max(range(7),key=lambda i:b[i]));top2.append(set(sorted(range(7),key=lambda i:a[i])[-2:])!=set(sorted(range(7),key=lambda i:b[i])[-2:]));e2.append(-sum(x*math.log(x) for x in a if x));e3.append(-sum(x*math.log(x) for x in b if x))
 return {"sampled_unique_states":len(rows),"policy_js_mean":statistics.fmean(js),"policy_js_high_threshold":THRESHOLDS["high_policy_js"],"high_divergence_states":sum(x>=THRESHOLDS["high_policy_js"] for x in js),"high_divergence_rate":sum(x>=THRESHOLDS["high_policy_js"] for x in js)/len(js),"top1_disagreement_rate":statistics.fmean(dis),"top2_disagreement_rate":statistics.fmean(top2),"G2_entropy_mean":statistics.fmean(e2),"G3_entropy_mean":statistics.fmean(e3)}
def structural(examples):
 bins={"seeds_in_play":Counter(),"store_difference":Counter(),"ply":Counter(),"legal_moves":Counter()}
 for x in examples:
  b=x.state.board;bins["seeds_in_play"][str((sum(b[:14])//10)*10)]+=1;bins["store_difference"][str(max(-7,min(7,(b[14]-b[15])//5)))]+=1;bins["ply"][str(min(9,int(x.metadata["ply"])//20))]+=1;bins["legal_moves"][str(sum(x.legal_mask))]+=1
 return {k:{"occupied_bins":len(v),"counts":dict(v)} for k,v in bins.items()}
def analyze(a):
 control=read_arm(a.output/"control");pool=read_arm(a.output/"pool");cm=json.load((a.output/"control_manifest.json").open());pm=json.load((a.output/"pool_manifest.json").open());cs={physical_state_key(x) for x in control};ps={physical_state_key(x) for x in pool};covc=state_coverage(control);covp=state_coverage(pool);history=historical_states();sources=source_state_sets(pool);crossonly=crossplay_only_states(sources);marg=marginal_novelty(("G2_G2","G3_G3","CROSS_PLAY"),sources);g2,p,v,g3=load_models();dc=policy_diversity(control,g2,g3);dp=policy_diversity(pool,g2,g3);source_div={s:policy_diversity([x for x in pool if x.metadata["source_type"]==s],g2,g3,20000) for s in sources}
 coverage={"CONTROL":covc,"POOL":covp,"comparison":overlap(cs,ps),"UNIQUE_STATE_GAIN":covp["unique_physical_states"]-covc["unique_physical_states"],"normalized_gain":covp["unique_per_10000_raw"]/covc["unique_per_10000_raw"]-1,"POOL_ONLY_STATES":len(ps-cs),"CONTROL_ONLY_STATES":len(cs-ps),"structural":{"CONTROL":structural(control),"POOL":structural(pool)}};write_json(a.output/"state_coverage.json",coverage)
 hist={"history_unique_states":len(history),"CONTROL":{"new":len(cs-history),"new_per_10000_raw":10000*len(cs-history)/len(control),"overlap":len(cs&history)},"POOL":{"new":len(ps-history),"new_per_10000_raw":10000*len(ps-history)/len(pool),"overlap":len(ps&history)}};hist["normalized_novelty_gain"]=hist["POOL"]["new_per_10000_raw"]/hist["CONTROL"]["new_per_10000_raw"]-1 if hist["CONTROL"]["new_per_10000_raw"] else None;write_json(a.output/"historical_novelty.json",hist)
 pool_rows={s:[x for x in pool if x.metadata["source_type"]==s] for s in sources};traj={"CONTROL":trajectory_metrics(control),"POOL":trajectory_metrics(pool),"phase_coverage":{"CONTROL":phase_coverage(control),"POOL":phase_coverage(pool)},"terminality":{"CONTROL":terminal_metrics(control),"POOL":terminal_metrics(pool),"by_pool_source":{s:terminal_metrics(rows) for s,rows in pool_rows.items()}}};write_json(a.output/"trajectory_diversity.json",traj);write_json(a.output/"action_diversity.json",{"CONTROL":action_metrics(control),"POOL":action_metrics(pool),"by_pool_source":{s:action_metrics(rows) for s,rows in pool_rows.items()}});strategic={"CONTROL":dc,"POOL":dp,"by_pool_source":source_div,"relative_top1_disagreement_gain":dp["top1_disagreement_rate"]/dc["top1_disagreement_rate"]-1 if dc["top1_disagreement_rate"] else None,"definition":"G2 and G3 network Policy on deterministic physical-state sample"};write_json(a.output/"strategic_disagreement.json",strategic)
 cross={"crossplay_unique_states":len(sources.get("CROSS_PLAY",set())),"CROSSPLAY_ONLY_STATES":len(crossonly),"crossplay_only_fraction":len(crossonly)/len(sources.get("CROSS_PLAY",set())),"POOL_NO_CROSS_unique":len(sources.get("G2_G2",set())|sources.get("G3_G3",set())),"POOL_FULL_unique":len(set().union(*sources.values())),"marginal":marg};write_json(a.output/"crossplay_analysis.json",cross);source={s:{"raw_positions":sum(x.metadata["source_type"]==s for x in pool),"unique_states":len(v),"unique_vs_other_pool_sources":len(v-set().union(*(z for k,z in sources.items() if k!=s))),"new_vs_history":len(v-history),"strategic":source_div[s],"terminality":terminal_metrics(pool_rows[s]),"phase_coverage":phase_coverage(pool_rows[s])} for s,v in sources.items()};source["ordered_marginal"]=marg;write_json(a.output/"source_contribution.json",source)
 prov={"CONTROL":dict(Counter(x.metadata["source_type"] for x in control)),"POOL":dict(Counter(x.metadata["source_type"] for x in pool)),"pool_game_sources":pm["sources"],"terminal_z_missing_CONTROL":sum(x.value_target is None for x in control),"terminal_z_missing_POOL":sum(x.value_target is None for x in pool),"missing_z_are_truncated_only":all(x.metadata["status"].startswith("TRUNCATED") for x in control+pool if x.value_target is None),"required_fields_present":all(all(k in x.metadata for k in ("experiment_arm","generation_id","p1_model_id","p2_model_id","p1_role","p2_role","mcts_budget","provenance")) for x in control+pool)};write_json(a.output/"provenance_summary.json",prov)
 eff={"CONTROL":{"games":cm["actual_games"],"positions":len(control),"mcts_simulations":cm["total_mcts_simulations"],"elapsed_s":cm["elapsed_s"],"mean_simulations_game":cm["total_mcts_simulations"]/cm["actual_games"],"unique_per_million_simulations":1e6*len(cs)/cm["total_mcts_simulations"],"new_per_million_simulations":1e6*len(cs-history)/cm["total_mcts_simulations"]},"POOL":{"games":pm["actual_games"],"positions":len(pool),"mcts_simulations":pm["total_mcts_simulations"],"elapsed_s":pm["elapsed_s"],"mean_simulations_game":pm["total_mcts_simulations"]/pm["actual_games"],"unique_per_million_simulations":1e6*len(ps)/pm["total_mcts_simulations"],"new_per_million_simulations":1e6*len(ps-history)/pm["total_mcts_simulations"]}};eff["elapsed_per_sim_ratio_pool_control"]=(eff["POOL"]["elapsed_s"]/eff["POOL"]["mcts_simulations"])/(eff["CONTROL"]["elapsed_s"]/eff["CONTROL"]["mcts_simulations"]);write_json(a.output/"compute_efficiency.json",eff)
 ids=[x.metadata["source_type"] for x in pool];sample=source_balanced_indices(ids,1200,seed=SEED);balanced=Counter(ids[i] for i in sample);replay={"requested":1200,"observed_source_counts":dict(balanced),"source_balancing_valid":max(balanced.values())-min(balanced.values())<=1,"provenance_preserved":all(pool[i].metadata["source_type"]==ids[i] for i in sample),"value_targets_valid":prov["missing_z_are_truncated_only"],"stratification_valid":set(balanced)==set(sources)};write_json(a.output/"replay_validation.json",replay)
 control_trunc=cm["truncated_games"]/cm["actual_games"];pool_trunc=pm["truncated_games"]/pm["actual_games"];coverage_yes=coverage["normalized_gain"]>=THRESHOLDS["material_normalized_gain"];history_yes=hist["normalized_novelty_gain"] is not None and hist["normalized_novelty_gain"]>=THRESHOLDS["material_normalized_gain"];strategic_yes=strategic["relative_top1_disagreement_gain"] is not None and strategic["relative_top1_disagreement_gain"]>=THRESHOLDS["strategic_relative_gain"];cross_yes=cross["crossplay_only_fraction"]>=THRESHOLDS["crossplay_unique_fraction"];pathological=pool_trunc>THRESHOLDS["max_truncation_rate"] or pool_trunc-control_trunc>THRESHOLDS["pathology_pool_minus_control_pp"];cost_ok=eff["elapsed_per_sim_ratio_pool_control"]<=THRESHOLDS["max_elapsed_per_sim_ratio"];pilot_valid=cm["actual_games"]==pm["actual_games"]==400 and prov["required_fields_present"] and prov["missing_z_are_truncated_only"]
 useful=coverage_yes and (history_yes or strategic_yes) and cross_yes and not pathological and cost_ok;status="RETAIN" if useful else "REMOVE" if pilot_valid and not (coverage_yes or history_yes or strategic_yes or cross_yes) else "INCONCLUSIVE";verdict={"PILOT_VALID":"YES" if pilot_valid else "NO","EQUAL_GENERATION_BUDGET":"YES" if cm["actual_games"]==pm["actual_games"]==400 else "NO","POOL_INCREASES_UNIQUE_STATE_COVERAGE":"YES" if coverage_yes else "NO","POOL_INCREASES_HISTORICAL_NOVELTY":"YES" if history_yes else "NO","POOL_INCREASES_STRATEGIC_DIVERSITY":"YES" if strategic_yes else "NO","CROSS_PLAY_ADDS_UNIQUE_STATES":"YES" if cross_yes else "NO","CROSS_PLAY_ADDS_STRATEGIC_DIVERSITY":"YES" if source_div["CROSS_PLAY"]["top1_disagreement_rate"]>source_div["G2_G2"]["top1_disagreement_rate"] else "NO","G3_SELFPLAY_ADDS_USEFUL_DIVERSITY":"YES" if marg["G3_G3"]["marginal_fraction"]>=THRESHOLDS["crossplay_unique_fraction"] and source_div["G3_G3"]["top1_disagreement_rate"]>=source_div["G2_G2"]["top1_disagreement_rate"] else "NO","POOL_DATA_PATHOLOGICAL":"YES" if pathological else "NO","COMPUTE_COST_ACCEPTABLE":"YES" if cost_ok else "NO","SOURCE_BALANCING_VALID":"YES" if replay["source_balancing_valid"] else "NO","REPLAY_STRATIFICATION_VALID":"YES" if replay["stratification_valid"] and replay["provenance_preserved"] else "NO","POOL_DATASET_READY_FOR_TRAINING":"YES" if pilot_valid and replay["value_targets_valid"] else "NO","G3_VALUE_REWORK_GENERATOR_STATUS":status,"OFFICIAL_CHAMPION":"G2","G3_PROMOTED":"NO","NEXT_ACTION":"POOL_GENERATED_G4_TRAINING_DESIGN" if status=="RETAIN" else "GENERATOR_POOL_REASSESSMENT" if status=="REMOVE" else "GENERATOR_POOL_PILOT_EXTENSION"};decision={"thresholds_pre_registered":THRESHOLDS,"criteria":{"coverage":coverage_yes,"historical_novelty":history_yes,"strategic_diversity":strategic_yes,"crossplay_unique":cross_yes,"non_pathological":not pathological,"cost_acceptable":cost_ok},"verdict":verdict};write_json(a.output/"generator_decision.json",decision);write_json(a.output/"report.json",{"lot":31,"training_performed":False,"control_manifest":cm,"pool_manifest":pm,"state_coverage":coverage,"historical_novelty":hist,"trajectory_diversity":traj,"strategic_disagreement":strategic,"crossplay":cross,"source_contribution":source,"compute_efficiency":eff,"replay_validation":replay,"generator_decision":decision,"verdict":verdict});print(json.dumps(verdict,indent=2));return verdict
def main():
 a=args();a.output.mkdir(parents=True,exist_ok=True);config={"seed":SEED,"generation_id":{"CONTROL":"LOT31_CONTROL_V1","POOL":"LOT31_POOL_V1"},"games":{"CONTROL":400,"POOL":400,"pool_sources":{"G2_G2":100,"G3_G3":100,"CROSS_PLAY":200}},"search":{"mcts_budget":64,"c_puct":1.5,"dirichlet_alpha":.3,"dirichlet_epsilon":.25,"root_noise":True,"target_temperature":1.,"action_temperature_early":1.,"drop_ply":30,"late":0.},"thresholds":THRESHOLDS,"canonicalization":False,"training":False,"teacher_labels":False,"minimax_labels":False};write_json(a.output/"configuration.json",config)
 stages=("identity","control","pool","analyze") if a.stage=="all" else (a.stage,)
 for stage in stages:{"identity":identity,"control":lambda x:generate_arm(x,"CONTROL"),"pool":lambda x:generate_arm(x,"POOL"),"analyze":analyze}[stage](a)
if __name__=="__main__":main()
