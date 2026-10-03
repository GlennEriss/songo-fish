#!/usr/bin/env python3
"""Lot 32: scaling G2/G3 cross-play, generation and measurement only."""
from __future__ import annotations
import argparse, hashlib, json, math, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
import random
import torch

from songo_ai.dataset import read_d_rl_jsonl, write_d_rl_jsonl
from songo_ai.evaluation import action_metrics, physical_state_key, state_coverage
from songo_ai.evaluation.generator_pool import overlap
from songo_ai.evaluation.search_policy_diagnosis import js_divergence
from songo_ai.generation import MatchupQuota, ModelRole, SelfPlayConfig, SelfPlayRunner, deterministic_schedule, source_balanced_indices
from songo_ai.model import SongoGraphBuilder, load_srn_checkpoint, policy_probabilities
from songo_ai.search import MCTSConfig
from songo_ai.songo import SongoLegacyGame
from songo_ai.evaluation.hybrid_evaluator import HybridPolicyValueEvaluator
from run_srn_lot12 import sha256, write_json

G2=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt")
G3P=Path("data/experiments/lot26_g3_scale/checkpoints/g3_strategic_best.pt")
V28=Path("data/experiments/lot28_value_search/checkpoints/v28_a_best.pt")
OUT=Path("data/experiments/lot32_crossplay_scaling")
SEED=20263201
SHARD_GAMES=20
THRESHOLDS={"material_coverage_gain":.02,"material_historical_novelty_gain":.02,"material_strategic_gain":.05,"max_truncation_rate":.05,"pathology_arm_minus_control_pp":.02,"max_elapsed_per_sim_ratio":2.,"high_policy_js":.001,"visitation_shift_tv":.05,"rapid_saturation_ratio":.50}
ARM_NAMES=("CONTROL","ORIGINAL_POOL","CROSSPLAY_ENRICHED")

def cli():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("prepare","control","original_pool","crossplay_enriched","analyze","all"),default="all");p.add_argument("--output",type=Path,default=OUT);return p.parse_args()

def fingerprint(model,keys=None):
 h=hashlib.sha256();state=model.state_dict()
 for key in sorted(keys or state):h.update(key.encode());h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def models():
 g2=load_srn_checkpoint(G2).model;p=load_srn_checkpoint(G3P).model;v=load_srn_checkpoint(V28).model
 for model in (g2,p,v):model.eval()
 return g2,p,v,HybridPolicyValueEvaluator(p,v,name="G3_VALUE_REWORK")

def quotas(arm):
 if arm=="CONTROL":return (MatchupQuota("G2","G2",ModelRole.CHAMPION,ModelRole.CHAMPION,2000,"LOT32_CONTROL_G2_SELFPLAY"),)
 if arm=="ORIGINAL_POOL":counts=(500,500,500,500)
 else:counts=(200,200,800,800)
 return (MatchupQuota("G2","G2",ModelRole.CHAMPION,ModelRole.CHAMPION,counts[0],f"LOT32_{arm}_G2_SELFPLAY"),MatchupQuota("G3_VALUE_REWORK","G3_VALUE_REWORK",ModelRole.GENERATOR,ModelRole.GENERATOR,counts[1],f"LOT32_{arm}_G3_SELFPLAY"),MatchupQuota("G2","G3_VALUE_REWORK",ModelRole.CHAMPION,ModelRole.GENERATOR,counts[2],f"LOT32_{arm}_CROSS_G2_P1"),MatchupQuota("G3_VALUE_REWORK","G2",ModelRole.GENERATOR,ModelRole.CHAMPION,counts[3],f"LOT32_{arm}_CROSS_G3_P1"))

def source_type(game):
 if game.p1_model_id==game.p2_model_id=="G2":return "G2_G2"
 if game.p1_model_id==game.p2_model_id=="G3_VALUE_REWORK":return "G3_G3"
 return "CROSS_PLAY"

def schedules():
 return {arm:deterministic_schedule(f"LOT32_{arm}_V1",quotas(arm),seed=SEED+i*100000,mcts_budget=64) for i,arm in enumerate(ARM_NAMES)}

def prepare(a):
 g2,p,v,_=models();identity={"G2":{"checkpoint":str(G2),"sha256":sha256(G2),"fingerprint":fingerprint(g2),"roles":["OFFICIAL_CHAMPION","AUTHORIZED_GENERATOR"]},"G3_VALUE_REWORK":{"policy_checkpoint":str(G3P),"policy_sha256":sha256(G3P),"value_checkpoint":str(V28),"value_sha256":sha256(V28),"policy_fingerprint":fingerprint(p,[k for k in p.state_dict() if not k.startswith("value_mlp.")]),"value_fingerprint":fingerprint(v,[k for k in v.state_dict() if k.startswith("value_mlp.")]),"roles":["PROBATIONARY_GENERATOR"],"G3_PROMOTED":False},"architecture_fingerprint":hashlib.sha256(Path("packages/songo_ai/model/srn_network.py").read_bytes()+Path("packages/songo_ai/model/srn_graph.py").read_bytes()).hexdigest()}
 write_json(a.output/"model_identity.json",identity)
 config={"lot":32,"seed_set":"LOT32_SEED_SET_V1","base_seed":SEED,"games_per_arm":2000,"arm_ratios":{"CONTROL":{"G2_G2":1.},"ORIGINAL_POOL":{"G2_G2":.25,"G3_G3":.25,"CROSS_PLAY":.5},"CROSSPLAY_ENRICHED":{"G2_G2":.1,"G3_G3":.1,"CROSS_PLAY":.8}},"search":{"mcts_budget":64,"c_puct":1.5,"dirichlet_alpha":.3,"dirichlet_epsilon":.25,"root_noise":True,"target_temperature":1.,"action_temperature_early":1.,"drop_ply":30,"late":0.,"max_game_plies":400,"repetition_limit":3},"thresholds":THRESHOLDS,"canonicalization":False,"training":False,"teacher_labels":False,"minimax_labels":False}
 write_json(a.output/"configuration.json",config)
 ss=schedules();write_json(a.output/"seed_manifest.json",{"seed_set":"LOT32_SEED_SET_V1","independent_from_lot31":True,"arms":{arm:{"generation_id":rows[0].generation_id,"schedule_hash":hashlib.sha256("|".join(f"{g.game_id}:{g.seed}" for g in rows).encode()).hexdigest(),"games":len(rows),"seeds":[g.seed for g in rows]} for arm,rows in ss.items()}})

def game_config(game,identity,arm):
 def fp(mid):
  x=identity[mid];return x["fingerprint"] if mid=="G2" else x["policy_fingerprint"]+":"+x["value_fingerprint"]
 provenance={"experiment_arm":arm,"generation_id":game.generation_id,"scheduled_game_id":game.game_id,"p1_model_id":game.p1_model_id,"p2_model_id":game.p2_model_id,"p1_role":game.p1_role.value,"p2_role":game.p2_role.value,"p1_fingerprint":fp(game.p1_model_id),"p2_fingerprint":fp(game.p2_model_id),"mcts_budget":64,"generator_role":game.provenance,"provenance":game.provenance,"source_type":source_type(game),"lot":32,"canonicalization":False,"seed_set":"LOT32_SEED_SET_V1"}
 return SelfPlayConfig(games=1,max_game_plies=400,repetition_limit=3,target_temperature=1.,action_temperature=1.,temperature_drop_ply=30,late_action_temperature=0.,seed=game.seed,generation=32,checkpoint_id=f"{game.p1_model_id}_vs_{game.p2_model_id}",provenance=provenance,include_truncated_examples=True,mcts=MCTSConfig(num_simulations=64,c_puct=1.5,dirichlet_alpha=.3,dirichlet_epsilon=.25,add_root_noise=True))

def generate(a,arm):
 identity=json.load((a.output/"model_identity.json").open());g2,_,_,g3=models();registry={"G2":g2,"G3_VALUE_REWORK":g3};schedule=schedules()[arm];root=a.output/arm.lower();root.mkdir(exist_ok=True);parts=[]
 for start in range(0,len(schedule),SHARD_GAMES):
  chunk=schedule[start:start+SHARD_GAMES];part=start//SHARD_GAMES;path=root/f"part-{part:04d}.jsonl";meta=root/f"part-{part:04d}.manifest.json"
  if path.exists() and meta.exists():m=json.load(meta.open());print(f"[lot32] {arm} shard {part+1}/100 reused",flush=True)
  else:
   began=time.perf_counter();examples=[];records=[]
   for game in chunk:
    cfg=game_config(game,identity,arm);run=SelfPlayRunner(registry[game.p1_model_id],cfg,models_by_player={1:registry[game.p1_model_id],2:registry[game.p2_model_id]}).play_game(0);examples.extend(run.examples);records.append({"scheduled_game_id":game.game_id,"game_id":run.game_id,"p1":game.p1_model_id,"p2":game.p2_model_id,"source_type":source_type(game),"status":run.status.value,"winner":run.winner,"plies":run.num_plies,"total_mcts_simulations":run.total_mcts_simulations,"network_evaluations":run.total_network_evaluations,"elapsed_s":run.elapsed_s,"action_sequence":list(run.action_sequence)})
   write_d_rl_jsonl(path,examples);m={"arm":arm,"part":part,"games":len(records),"positions":len(examples),"sha256":sha256(path),"elapsed_s":time.perf_counter()-began,"game_records":records};write_json(meta,m);print(f"[lot32] {arm} shard {part+1}/100: {len(examples)} positions",flush=True)
  parts.append(m)
 manifest={"arm":arm,"planned_games":len(schedule),"actual_games":sum(x["games"] for x in parts),"positions":sum(x["positions"] for x in parts),"sources":dict(Counter(source_type(x) for x in schedule)),"shards":len(parts),"mcts_budget":64,"configuration_equal_across_arms":True,"total_mcts_simulations":sum(g["total_mcts_simulations"] for p in parts for g in p["game_records"]),"elapsed_s":sum(p["elapsed_s"] for p in parts),"terminal_games":sum(g["status"] in ("TERMINAL_WIN","TERMINAL_DRAW") for p in parts for g in p["game_records"]),"truncated_games":sum(g["status"].startswith("TRUNCATED") for p in parts for g in p["game_records"])}
 write_json(a.output/{"CONTROL":"control_manifest.json","ORIGINAL_POOL":"original_pool_manifest.json","CROSSPLAY_ENRICHED":"crossplay_enriched_manifest.json"}[arm],manifest)

def read_arm(root):return [x for p in sorted(root.glob("part-*.jsonl")) for x in read_d_rl_jsonl(p)]

def history_states():
 states=set()
 for base in (Path("data/d_scale_v1/d_selfplay_large"),Path("data/d_rl")):
  for p in sorted(base.glob("*.jsonl")):
   try:states.update(physical_state_key(x) for x in read_d_rl_jsonl(p))
   except ValueError:pass
 for p in sorted(Path("data/d_scale_v1/d_reanalysis_large").glob("*.jsonl")):
  for line in p.open():
   r=json.loads(line);s=r["state"];states.add((tuple(s["board"]),int(s["player_to_move"])))
 return states

def group_games(examples):
 out=defaultdict(list)
 for x in examples:out[x.metadata["game_id"]].append(x)
 for rows in out.values():rows.sort(key=lambda x:int(x.metadata["ply"]))
 return out

def trajectory(examples):
 games=group_games(examples);actions=[tuple(int(x.metadata["action_played"]) for x in rows) for rows in games.values()];lengths=[len(x) for x in actions]
 pct=lambda q:sorted(lengths)[min(len(lengths)-1,round((len(lengths)-1)*q))]
 return {"games":len(games),"unique_trajectory_signatures":len(set(actions)),"unique_prefix4":len({x[:4] for x in actions}),"unique_prefix8":len({x[:8] for x in actions}),"unique_prefix12":len({x[:12] for x in actions}),"length_mean":statistics.fmean(lengths),"length_median":statistics.median(lengths),"length_p10":pct(.1),"length_p90":pct(.9),"length_min":min(lengths),"length_max":max(lengths)}

def structural(examples):
 bins={"seeds_in_play":Counter(),"store_difference":Counter(),"legal_moves":Counter(),"game_depth":Counter()}
 for x in examples:
  b=x.state.board;bins["seeds_in_play"][str((sum(b[:14])//10)*10)]+=1;bins["store_difference"][str(max(-7,min(7,(b[14]-b[15])//5)))]+=1;bins["legal_moves"][str(sum(x.legal_mask))]+=1;bins["game_depth"][str(min(9,int(x.metadata["ply"])//20))]+=1
 return {k:dict(v) for k,v in bins.items()}

def policy_metrics(examples,g2,g3,history,limit=50000):
 unique={physical_state_key(x):x for x in examples};rows=sorted(unique.values(),key=lambda x:hashlib.sha256(repr(physical_state_key(x)).encode()).hexdigest())[:limit];builder=SongoGraphBuilder();js=[];top=[];depth=defaultdict(list);novel=[]
 with torch.no_grad():
  for start in range(0,len(rows),512):
   chunk=rows[start:start+512];graph=builder.build_batch([x.state for x in chunk]);mask=torch.tensor([x.legal_mask for x in chunk]);p2=policy_probabilities(g2(graph)[0],mask).tolist();p3=policy_probabilities(g3(graph)[0],mask).tolist()
   for x,a,b in zip(chunk,p2,p3):
    d=js_divergence(a,b);different=max(range(7),key=lambda i:a[i])!=max(range(7),key=lambda i:b[i]);js.append(d);top.append(different);depth[str(min(9,int(x.metadata["ply"])//20))].append(different);novel.append(different and physical_state_key(x) not in history)
 return {"sampled_unique_states":len(rows),"policy_js_mean":statistics.fmean(js),"high_divergence_rate":sum(x>=THRESHOLDS["high_policy_js"] for x in js)/len(js),"top1_disagreement_rate":statistics.fmean(top),"historically_novel_disagreement_rate":statistics.fmean(novel),"disagreement_by_depth":{k:statistics.fmean(v) for k,v in depth.items()}}

def saturation(examples,history):
 games=list(group_games(examples).values());seen=set();rows=[];raw=0
 for idx,game in enumerate(games,1):
  raw+=len(game);seen.update(physical_state_key(x) for x in game)
  if idx%200==0:rows.append({"games":idx,"raw_positions":raw,"unique_states":len(seen),"historical_new":len(seen-history),"marginal_unique_last_200":len(seen)-rows[-1]["unique_states"] if rows else len(seen)})
 return rows

def bootstrap(games,history,replicates=10000,seed=SEED):
 per=[]
 for rows in games.values():
  keys={physical_state_key(x) for x in rows};per.append((len(rows),len(keys),len(keys-history)))
 rng=random.Random(seed);coverage=[];novelty=[];n=len(per)
 for _ in range(replicates):
  sample=[per[rng.randrange(n)] for _ in range(n)];raw=sum(x[0] for x in sample);coverage.append(10000*sum(x[1] for x in sample)/raw);novelty.append(10000*sum(x[2] for x in sample)/raw)
 def ci(values):
  values.sort();return {"mean":statistics.fmean(values),"ci95":[values[249],values[9749]]}
 return {"replicates":replicates,"unit":"game","estimator":"sum of within-game unique states normalized by sampled raw positions","coverage":ci(coverage),"historical_novelty":ci(novelty)}

def bootstrap_difference(left_games,right_games,history,replicates=10000,seed=SEED):
 def vectors(games):
  out=[]
  for rows in games.values():
   keys={physical_state_key(x) for x in rows};out.append((len(rows),len(keys),len(keys-history)))
  return out
 left,right=vectors(left_games),vectors(right_games);rng=random.Random(seed);coverage=[];novelty=[]
 for _ in range(replicates):
  ls=[left[rng.randrange(len(left))] for _ in left];rs=[right[rng.randrange(len(right))] for _ in right]
  lc=10000*sum(x[1] for x in ls)/sum(x[0] for x in ls);rc=10000*sum(x[1] for x in rs)/sum(x[0] for x in rs);ln=10000*sum(x[2] for x in ls)/sum(x[0] for x in ls);rn=10000*sum(x[2] for x in rs)/sum(x[0] for x in rs);coverage.append(rc/lc-1);novelty.append(rn/ln-1)
 def summary(values):
  values.sort();return {"mean":statistics.fmean(values),"ci95":[values[249],values[9749]],"probability_positive":sum(x>0 for x in values)/len(values)}
 return {"replicates":replicates,"unit":"game","coverage_relative_difference":summary(coverage),"historical_novelty_relative_difference":summary(novelty)}

def quality_control(examples):
 games=group_games(examples);conservation=invalid_masks=illegal=transitions=provenance=0;required=("experiment_arm","generation_id","p1_model_id","p2_model_id","p1_role","p2_role","mcts_budget","provenance","source_type")
 for rows in games.values():
  for index,x in enumerate(rows):
   if len(x.state.board)!=16 or sum(x.state.board)!=70 or any(v<0 for v in x.state.board):conservation+=1
   game=SongoLegacyGame.from_board(x.state.board,x.state.player_to_move)
   if tuple(game.legal_mask())!=tuple(x.legal_mask):invalid_masks+=1
   action=int(x.metadata["action_played"])
   if not 0<=action<7 or not x.legal_mask[action]:illegal+=1;continue
   if any(k not in x.metadata for k in required):provenance+=1
   if index+1<len(rows):
    result=game.play_local(action);nxt=rows[index+1]
    if tuple(result.board)!=tuple(nxt.state.board) or result.next_player!=nxt.state.player_to_move:transitions+=1
 return {"games":len(games),"duplicate_game_ids":0 if len(games)==len(set(games)) else len(games)-len(set(games)),"illegal_actions":illegal,"illegal_transitions":transitions,"conservation_failures":conservation,"invalid_masks":invalid_masks,"provenance_failures":provenance,"valid":not any((illegal,transitions,conservation,invalid_masks,provenance))}

def tv_distance(a,b):
 keys=set(a)|set(b);sa=sum(a.values());sb=sum(b.values());return .5*sum(abs(a.get(k,0)/sa-b.get(k,0)/sb) for k in keys)

def analyze(a):
 arms={arm:read_arm(a.output/arm.lower()) for arm in ARM_NAMES};history=history_states();g2,_,_,g3=models();sets={arm:{physical_state_key(x) for x in xs} for arm,xs in arms.items()};coverage={arm:state_coverage(xs) for arm,xs in arms.items()}
 comparisons={f"{left}_VS_{right}":{"normalized_gain":coverage[right]["unique_per_10000_raw"]/coverage[left]["unique_per_10000_raw"]-1,**overlap(sets[left],sets[right])} for left,right in (("CONTROL","ORIGINAL_POOL"),("CONTROL","CROSSPLAY_ENRICHED"),("ORIGINAL_POOL","CROSSPLAY_ENRICHED"))};coverage["comparisons"]=comparisons;write_json(a.output/"state_coverage.json",coverage)
 hist={arm:{"historical_new":len(sets[arm]-history),"new_per_10000_raw":10000*len(sets[arm]-history)/len(arms[arm]),"historical_overlap":len(sets[arm]&history)} for arm in ARM_NAMES};hist["reference_unique_states"]=len(history);hist["reference_frozen_across_arms"]=True;hist["comparisons"]={f"{l}_VS_{r}":hist[r]["new_per_10000_raw"]/hist[l]["new_per_10000_raw"]-1 for l,r in (("CONTROL","ORIGINAL_POOL"),("CONTROL","CROSSPLAY_ENRICHED"),("ORIGINAL_POOL","CROSSPLAY_ENRICHED"))};write_json(a.output/"historical_novelty.json",hist)
 strategic={arm:policy_metrics(xs,g2,g3,history) for arm,xs in arms.items()};strategic["relative_gains_vs_control"]={arm:strategic[arm]["top1_disagreement_rate"]/strategic["CONTROL"]["top1_disagreement_rate"]-1 for arm in ("ORIGINAL_POOL","CROSSPLAY_ENRICHED")};strategic["actions"]={arm:action_metrics(xs) for arm,xs in arms.items()};write_json(a.output/"strategic_diversity.json",strategic)
 source_sets={arm:{src:{physical_state_key(x) for x in xs if x.metadata["source_type"]==src} for src in set(x.metadata["source_type"] for x in xs)} for arm,xs in arms.items()};cross={}
 for arm in ("ORIGINAL_POOL","CROSSPLAY_ENRICHED"):
  cp=source_sets[arm]["CROSS_PLAY"];hom=set().union(source_sets[arm]["G2_G2"],source_sets[arm]["G3_G3"]);raw=sum(x.metadata["source_type"]=="CROSS_PLAY" for x in arms[arm]);cross[arm]={"crossplay_unique_states":len(cp),"crossplay_only_states":len(cp-hom),"crossplay_only_per_10000_crossplay_positions":10000*len(cp-hom)/raw,"crossplay_only_fraction":len(cp-hom)/len(cp)}
 write_json(a.output/"crossplay_only_states.json",cross)
 curves={arm:saturation(xs,history) for arm,xs in arms.items()};write_json(a.output/"saturation_curves.json",curves)
 source_overlap={}
 for arm in ("ORIGINAL_POOL","CROSSPLAY_ENRICHED"):
  source_overlap[arm]={}
  for left in ("G2_G2","G3_G3","CROSS_PLAY"):
   source_overlap[arm][left]={right:overlap(source_sets[arm][left],source_sets[arm][right]) for right in ("G2_G2","G3_G3","CROSS_PLAY")}
 write_json(a.output/"source_overlap.json",source_overlap)
 marginal={}
 for arm in ("ORIGINAL_POOL","CROSSPLAY_ENRICHED"):
  marginal[arm]={}
  for src,current in source_sets[arm].items():
   others=set().union(*(v for k,v in source_sets[arm].items() if k!=src));raw=sum(x.metadata["source_type"]==src for x in arms[arm]);marginal[arm][src]={"raw_positions":raw,"unique_states":len(current),"unique_vs_other_sources":len(current-others),"unique_per_10000_source_positions":10000*len(current)/raw,"historical_new":len(current-history),"historical_new_per_10000_source_positions":10000*len(current-history)/raw}
 write_json(a.output/"source_marginal_value.json",marginal)
 traj={arm:trajectory(xs) for arm,xs in arms.items()};write_json(a.output/"trajectory_diversity.json",traj)
 structs={arm:structural(xs) for arm,xs in arms.items()};structs["tv_vs_control"]={arm:{feature:tv_distance(structs["CONTROL"][feature],structs[arm][feature]) for feature in structs["CONTROL"]} for arm in ("ORIGINAL_POOL","CROSSPLAY_ENRICHED")};write_json(a.output/"structural_coverage.json",structs)
 manifests={"CONTROL":json.load((a.output/"control_manifest.json").open()),"ORIGINAL_POOL":json.load((a.output/"original_pool_manifest.json").open()),"CROSSPLAY_ENRICHED":json.load((a.output/"crossplay_enriched_manifest.json").open())};compute={arm:{"games":m["actual_games"],"positions":len(arms[arm]),"simulations":m["total_mcts_simulations"],"elapsed_s":m["elapsed_s"],"mean_simulations_game":m["total_mcts_simulations"]/m["actual_games"],"unique_per_million_simulations":1e6*len(sets[arm])/m["total_mcts_simulations"],"historical_new_per_million_simulations":1e6*len(sets[arm]-history)/m["total_mcts_simulations"]} for arm,m in manifests.items()};compute["elapsed_per_sim_ratio_vs_control"]={arm:(compute[arm]["elapsed_s"]/compute[arm]["simulations"])/(compute["CONTROL"]["elapsed_s"]/compute["CONTROL"]["simulations"]) for arm in ("ORIGINAL_POOL","CROSSPLAY_ENRICHED")};write_json(a.output/"compute_efficiency.json",compute)
 grouped={arm:group_games(xs) for arm,xs in arms.items()};boot={arm:bootstrap(grouped[arm],history,seed=SEED+i) for i,arm in enumerate(ARM_NAMES)};boot["comparisons"]={f"{left}_VS_{right}":bootstrap_difference(grouped[left],grouped[right],history,seed=SEED+100+i) for i,(left,right) in enumerate((("CONTROL","ORIGINAL_POOL"),("CONTROL","CROSSPLAY_ENRICHED"),("ORIGINAL_POOL","CROSSPLAY_ENRICHED")))};write_json(a.output/"bootstrap_analysis.json",boot)
 quality={arm:quality_control(xs) for arm,xs in arms.items()};quality["all_valid"]=all(x["valid"] for x in quality.values());write_json(a.output/"quality_control.json",quality)
 replay={}
 for arm,xs in arms.items():
  ids=[x.metadata["source_type"] for x in xs];idx=source_balanced_indices(ids,1200,seed=SEED);counts=Counter(ids[i] for i in idx);replay[arm]={"requested":1200,"observed_source_counts":dict(counts),"provenance_preserved":all(xs[i].metadata["source_type"]==ids[i] for i in idx),"value_targets_valid":all(x.value_target is not None or x.metadata["status"].startswith("TRUNCATED") for x in xs)}
 replay["valid"]=all(x["provenance_preserved"] and x["value_targets_valid"] for x in replay.values());write_json(a.output/"replay_validation.json",replay)
 cov_best=max(comparisons["CONTROL_VS_ORIGINAL_POOL"]["normalized_gain"],comparisons["CONTROL_VS_CROSSPLAY_ENRICHED"]["normalized_gain"]);hist_best=max(hist["comparisons"]["CONTROL_VS_ORIGINAL_POOL"],hist["comparisons"]["CONTROL_VS_CROSSPLAY_ENRICHED"]);strat_best=max(strategic["relative_gains_vs_control"].values());material_cov=cov_best>=.02;material_hist=hist_best>=.02;material_strat=strat_best>=.05;cost_ok=max(compute["elapsed_per_sim_ratio_vs_control"].values())<=2.;trunc={arm:manifests[arm]["truncated_games"]/2000 for arm in ARM_NAMES};pathology=any(trunc[x]>.05 or trunc[x]-trunc["CONTROL"]>.02 for x in ARM_NAMES[1:]);vis_shift=max(structs["tv_vs_control"]["CROSSPLAY_ENRICHED"].values())>=THRESHOLDS["visitation_shift_tv"]
 last=curves["CROSSPLAY_ENRICHED"][-1]["marginal_unique_last_200"];first=curves["CROSSPLAY_ENRICHED"][0]["marginal_unique_last_200"];rapid=last/first<THRESHOLDS["rapid_saturation_ratio"]
 cross_marg=marginal["CROSSPLAY_ENRICHED"]["CROSS_PLAY"]["unique_vs_other_sources"]>0;g3_marg=marginal["CROSSPLAY_ENRICHED"]["G3_G3"]["unique_vs_other_sources"]>0;valid=all(m["actual_games"]==2000 and m["configuration_equal_across_arms"] for m in manifests.values()) and replay["valid"] and quality["all_valid"]
 def effect(c,h,s):return "POSITIVE" if c>=.02 or h>=.02 or s>=.05 else "NEGATIVE" if c<0 and h<0 and s<0 else "NEUTRAL"
 ab=effect(comparisons["CONTROL_VS_ORIGINAL_POOL"]["normalized_gain"],hist["comparisons"]["CONTROL_VS_ORIGINAL_POOL"],strategic["relative_gains_vs_control"]["ORIGINAL_POOL"]);ac=effect(comparisons["CONTROL_VS_CROSSPLAY_ENRICHED"]["normalized_gain"],hist["comparisons"]["CONTROL_VS_CROSSPLAY_ENRICHED"],strategic["relative_gains_vs_control"]["CROSSPLAY_ENRICHED"]);bc=effect(comparisons["ORIGINAL_POOL_VS_CROSSPLAY_ENRICHED"]["normalized_gain"],hist["comparisons"]["ORIGINAL_POOL_VS_CROSSPLAY_ENRICHED"],strategic["relative_gains_vs_control"]["CROSSPLAY_ENRICHED"]-strategic["relative_gains_vs_control"]["ORIGINAL_POOL"])
 material=material_cov or material_hist or material_strat;status="RETAIN" if valid and material and cross_marg and not pathology and cost_ok else "REMOVE" if valid and not material else "INCONCLUSIVE";mode="CROSSPLAY_ENRICHED" if status=="RETAIN" and ac=="POSITIVE" else "ORIGINAL_POOL" if status=="RETAIN" and ab=="POSITIVE" else "G2_ONLY" if status=="REMOVE" else "FURTHER_COMPOSITION_STUDY"
 verdict={"EXPERIMENT_VALID":"YES" if valid else "NO","EQUAL_SEARCH_CONFIGURATION":"YES" if all(m["mcts_budget"]==64 for m in manifests.values()) else "NO","ORIGINAL_POOL_SCALE_EFFECT":ab,"CROSSPLAY_ENRICHED_EFFECT":ac,"CROSSPLAY_WEIGHT_EFFECT":bc,"CROSSPLAY_NOVELTY_PERSISTS_AT_SCALE":"YES" if last>0 else "NO","CROSSPLAY_NOVELTY_SATURATES_RAPIDLY":"YES" if rapid else "NO","MATERIAL_COVERAGE_GAIN":"YES" if material_cov else "NO","MATERIAL_HISTORICAL_NOVELTY_GAIN":"YES" if material_hist else "NO","MATERIAL_STRATEGIC_DIVERSITY_GAIN":"YES" if material_strat else "NO","CROSSPLAY_SHIFTS_STATE_VISITATION":"YES" if vis_shift else "NO","G3_SELFPLAY_ADDS_MARGINAL_VALUE":"YES" if g3_marg else "NO","CROSSPLAY_ADDS_MARGINAL_VALUE":"YES" if cross_marg else "NO","DATA_PATHOLOGY":"YES" if pathology else "NO","COMPUTE_COST_ACCEPTABLE":"YES" if cost_ok else "NO","REPLAY_STRATIFICATION_VALID":"YES" if replay["valid"] else "NO","G3_VALUE_REWORK_GENERATOR_STATUS":status,"RECOMMENDED_GENERATION_MODE":mode,"OFFICIAL_CHAMPION":"G2","G3_PROMOTED":"NO","NEXT_ACTION":"POOL_GENERATED_G4_TRAINING_DESIGN" if status=="RETAIN" else "GENERATOR_POOL_REASSESSMENT" if status=="REMOVE" else "CROSSPLAY_SCALING_RETRY"}
 decision={"thresholds_pre_registered":THRESHOLDS,"verdict":verdict};write_json(a.output/"generator_decision.json",decision);write_json(a.output/"report.json",{"lot":32,"training_performed":False,"manifests":manifests,"state_coverage":coverage,"historical_novelty":hist,"strategic_diversity":strategic,"crossplay_only":cross,"saturation":curves,"source_marginal_value":marginal,"trajectory_diversity":traj,"structural_coverage":structs,"compute_efficiency":compute,"bootstrap":boot,"quality_control":quality,"replay":replay,"decision":decision});print(json.dumps(verdict,indent=2))

def main():
 a=cli();a.output.mkdir(parents=True,exist_ok=True);stages=("prepare","control","original_pool","crossplay_enriched","analyze") if a.stage=="all" else (a.stage,)
 for stage in stages:{"prepare":prepare,"control":lambda x:generate(x,"CONTROL"),"original_pool":lambda x:generate(x,"ORIGINAL_POOL"),"crossplay_enriched":lambda x:generate(x,"CROSSPLAY_ENRICHED"),"analyze":analyze}[stage](a)
if __name__=="__main__":main()
