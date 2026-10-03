#!/usr/bin/env python3
"""Lot 34: generate the two immutable 8k-game causal arms."""
from __future__ import annotations
import argparse,hashlib,json,math,time
from collections import Counter
from pathlib import Path
from songo_ai.dataset import write_d_rl_jsonl
from songo_ai.generation import MatchupQuota,ModelRole,SelfPlayConfig,SelfPlayRunner,deterministic_schedule
from songo_ai.search import MCTSConfig
from run_srn_lot12 import sha256,write_json
from run_srn_lot32 import G2,G3P,V28,fingerprint,models,source_type

OUT=Path("data/experiments/lot34_g4_training");SEED=20263400;SHARD_GAMES=20
ARMS=("CONTROL","POOL")

def cli():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("prepare","control","pool","all"),default="all");p.add_argument("--output",type=Path,default=OUT);return p.parse_args()
def quotas(arm):
 if arm=="CONTROL":return (MatchupQuota("G2","G2",ModelRole.CHAMPION,ModelRole.CHAMPION,8000,"LOT34_CONTROL_G2_SELFPLAY"),)
 return (MatchupQuota("G2","G2",ModelRole.CHAMPION,ModelRole.CHAMPION,800,"LOT34_POOL_G2_SELFPLAY"),MatchupQuota("G3_VALUE_REWORK","G3_VALUE_REWORK",ModelRole.GENERATOR,ModelRole.GENERATOR,800,"LOT34_POOL_G3_SELFPLAY"),MatchupQuota("G2","G3_VALUE_REWORK",ModelRole.CHAMPION,ModelRole.GENERATOR,3200,"LOT34_POOL_CROSS_G2_P1"),MatchupQuota("G3_VALUE_REWORK","G2",ModelRole.GENERATOR,ModelRole.CHAMPION,3200,"LOT34_POOL_CROSS_G3_P1"))
def schedules():return {arm:deterministic_schedule(f"LOT34_{arm}_SEEDS",quotas(arm),seed=SEED+i*100000,mcts_budget=64) for i,arm in enumerate(ARMS)}
def prepare(a):
 g2,p,v,_=models();identity={"G2":{"checkpoint":str(G2),"sha256":sha256(G2),"policy_fingerprint":fingerprint(g2,[k for k in g2.state_dict() if not k.startswith("value_mlp.")]),"value_fingerprint":fingerprint(g2,[k for k in g2.state_dict() if k.startswith("value_mlp.")])},"G3_VALUE_REWORK":{"policy_checkpoint":str(G3P),"policy_sha256":sha256(G3P),"value_checkpoint":str(V28),"value_sha256":sha256(V28),"policy_fingerprint":fingerprint(p,[k for k in p.state_dict() if not k.startswith("value_mlp.")]),"value_fingerprint":fingerprint(v,[k for k in v.state_dict() if k.startswith("value_mlp.")])},"initialization":{"policy":"G3_STRATEGIC","value":"V28_A","identical_between_arms":True},"architecture_fingerprint":hashlib.sha256(Path("packages/songo_ai/model/srn_network.py").read_bytes()+Path("packages/songo_ai/model/srn_graph.py").read_bytes()).hexdigest()};write_json(a.output/"model_identity.json",identity)
 config={"lot":34,"games_per_arm":8000,"pool_ratio":{"G2_G2":.1,"G3_G3":.1,"G2_G3":.8},"training_source_ratio":{"NEW_GENERATION":.7,"HISTORICAL_RL":.2,"AUTONOMOUS_REANALYSIS":.1},"training_updates":32000,"initial_policy":"G3_STRATEGIC","initial_value":"V28_A","policy_objective":"MASKED_SOFT_TARGET_CE_TO_PI_MCTS","value_objective":"MSE_TO_TRUE_TERMINAL_Z_ONLY","replay":"GENERATION_BALANCED_SOURCE_STRATIFIED","search":{"simulations":64,"c_puct":1.5,"dirichlet_alpha":.3,"dirichlet_epsilon":.25,"root_noise":True,"target_temperature":1.,"action_temperature":1.,"temperature_drop_ply":30,"late_temperature":0.,"max_plies":400,"repetition_limit":3},"training_allowed":"PENDING_INTEGRITY","teacher_labels":False,"minimax_labels":False,"engine_modified":False,"srn_modified":False};write_json(a.output/"configuration.json",config)
 ss=schedules();write_json(a.output/"seed_manifest.json",{"created_before_generation":True,"seed_policy":"LOT34_CONTROL_SEEDS / LOT34_POOL_SEEDS","arms":{arm:{"games":len(rows),"generation_id":rows[0].generation_id,"schedule_hash":hashlib.sha256("|".join(f"{x.game_id}:{x.seed}" for x in rows).encode()).hexdigest(),"seeds":[x.seed for x in rows]} for arm,rows in ss.items()}})
def cfg(game,identity,arm):
 def fp(mid):
  x=identity[mid];return x["policy_fingerprint"]+":"+x["value_fingerprint"]
 prov={"arm":arm,"experiment_arm":arm,"generation_id":game.generation_id,"scheduled_game_id":game.game_id,"source_type":source_type(game),"p1_model_id":game.p1_model_id,"p2_model_id":game.p2_model_id,"p1_role":game.p1_role.value,"p2_role":game.p2_role.value,"p1_fingerprint":fp(game.p1_model_id),"p2_fingerprint":fp(game.p2_model_id),"mcts_budget":64,"mcts_config":{"simulations":64,"c_puct":1.5,"dirichlet_alpha":.3,"dirichlet_epsilon":.25},"provenance":game.provenance,"lot":34,"canonicalization":False}
 return SelfPlayConfig(games=1,max_game_plies=400,repetition_limit=3,target_temperature=1.,action_temperature=1.,temperature_drop_ply=30,late_action_temperature=0.,seed=game.seed,generation=34,checkpoint_id=f"{game.p1_model_id}_vs_{game.p2_model_id}",provenance=prov,include_truncated_examples=True,mcts=MCTSConfig(num_simulations=64,c_puct=1.5,dirichlet_alpha=.3,dirichlet_epsilon=.25,add_root_noise=True))
def generate(a,arm):
 identity=json.load((a.output/"model_identity.json").open());g2,_,_,g3=models();registry={"G2":g2,"G3_VALUE_REWORK":g3};schedule=schedules()[arm];root=a.output/("control_data" if arm=="CONTROL" else "pool_data");root.mkdir(exist_ok=True);parts=[]
 for start in range(0,len(schedule),SHARD_GAMES):
  chunk=schedule[start:start+SHARD_GAMES];part=start//SHARD_GAMES;path=root/f"part-{part:04d}.jsonl";meta=root/f"part-{part:04d}.manifest.json"
  if path.exists() and meta.exists():m=json.load(meta.open());print(f"[lot34] {arm} shard {part+1}/400 reused",flush=True)
  else:
   began=time.perf_counter();examples=[];games=[]
   for game in chunk:
    c=cfg(game,identity,arm);run=SelfPlayRunner(registry[game.p1_model_id],c,models_by_player={1:registry[game.p1_model_id],2:registry[game.p2_model_id]}).play_game(0);examples.extend(run.examples);games.append({"scheduled_game_id":game.game_id,"game_id":run.game_id,"p1":game.p1_model_id,"p2":game.p2_model_id,"source_type":source_type(game),"seed":game.seed,"status":run.status.value,"winner":run.winner,"plies":run.num_plies,"total_mcts_simulations":run.total_mcts_simulations,"elapsed_s":run.elapsed_s})
   write_d_rl_jsonl(path,examples);m={"arm":arm,"part":part,"games":len(games),"positions":len(examples),"sha256":sha256(path),"elapsed_s":time.perf_counter()-began,"game_records":games};write_json(meta,m);print(f"[lot34] {arm} shard {part+1}/400: {len(examples)} positions",flush=True)
  parts.append(m)
 manifest={"arm":arm,"planned_games":8000,"actual_games":sum(x["games"] for x in parts),"positions":sum(x["positions"] for x in parts),"sources":dict(Counter(source_type(x) for x in schedule)),"crossplay_sides":{"G2_P1":sum(x.p1_model_id=="G2" and x.p2_model_id=="G3_VALUE_REWORK" for x in schedule),"G3_P1":sum(x.p1_model_id=="G3_VALUE_REWORK" and x.p2_model_id=="G2" for x in schedule)},"shards":len(parts),"mcts_budget":64,"total_mcts_simulations":sum(g["total_mcts_simulations"] for p in parts for g in p["game_records"]),"elapsed_s":sum(p["elapsed_s"] for p in parts),"terminal_games":sum(g["status"] in ("TERMINAL_WIN","TERMINAL_DRAW") for p in parts for g in p["game_records"]),"truncated_games":sum(g["status"].startswith("TRUNCATED") for p in parts for g in p["game_records"])};write_json(a.output/("control_dataset_manifest.json" if arm=="CONTROL" else "pool_dataset_manifest.json"),manifest)
def main():
 a=cli();a.output.mkdir(parents=True,exist_ok=True);stages=("prepare","control","pool") if a.stage=="all" else (a.stage,)
 for stage in stages:{"prepare":prepare,"control":lambda x:generate(x,"CONTROL"),"pool":lambda x:generate(x,"POOL")}[stage](a)
if __name__=="__main__":main()
