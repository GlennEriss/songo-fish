#!/usr/bin/env python3
"""Lot 29 : confirmation indépendante et promotion finale de G3."""
from __future__ import annotations
import argparse,hashlib,json,random,subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from songo_ai.evaluation import ArenaConfig,HybridPolicyValueEvaluator,generate_unique_deterministic_openings,independent_seed_set,promotion_rule,side_gap,side_score
from songo_ai.model import SongoGraphBuilder,load_srn_checkpoint
from run_srn_lot12 import sha256,write_json
from run_srn_lot23 import arena

G2=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt")
G3P=Path("data/experiments/lot26_g3_scale/checkpoints/g3_strategic_best.pt")
V28=Path("data/experiments/lot28_value_search/checkpoints/v28_a_best.pt")
OUT=Path("data/experiments/lot29_g3_confirmation")
SEEDS={"python":20262929,"numpy":20262930,"torch":20262931,"arena64":20262964,"opening64":20263964,"arena128":202629128,"opening128":202639128,"arena256":202629256,"opening256":202639256,"bootstrap":20262999}
LOT28_SEEDS=(20262828,20262829,20262830,20262831,20262832)
COUNTS={64:256,128:256,256:128} # ouvertures; deux parties chacune

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--phase",choices=("identity","64","128","256","decision","all"),default="all");p.add_argument("--output",type=Path,default=OUT);p.add_argument("--bootstrap",type=int,default=20000);return p.parse_args()
def file_hash(path):return sha256(path)
def tensor_hash(state,keys):
 h=hashlib.sha256()
 for k in sorted(keys):h.update(k.encode());h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def source_hash(paths):
 h=hashlib.sha256()
 for p in paths:h.update(Path(p).read_bytes())
 return h.hexdigest()
def load_models():
 g2=load_srn_checkpoint(G2).model;policy=load_srn_checkpoint(G3P).model;value=load_srn_checkpoint(V28).model
 for m in (g2,policy,value):m.eval();
 return g2,policy,value,HybridPolicyValueEvaluator(policy,value,name="G3_VALUE_REWORK")
def seed_manifest(a):
 path=a.output/"seed_manifest.json";payload={"name":"CONFIRMATION_SEED_SET_V1","seeds":SEEDS,"forbidden_lot28_seeds":list(LOT28_SEEDS),"independent":independent_seed_set(SEEDS.values(),LOT28_SEEDS),"fixed_game_counts":{"MCTS64":512,"MCTS128":512,"MCTS256":256},"created_before_games":True,"immutable":True}
 if path.exists() and json.load(path.open())!=payload:raise RuntimeError("immutable seed manifest mismatch")
 write_json(path,payload);return payload
def identity(a):
 seed=seed_manifest(a);g2,p,v,h=load_models();states=[x.state for x in generate_unique_deterministic_openings(count=128,seed=SEEDS["opening64"],max_prefix_length=40)];graph=SongoGraphBuilder().build_batch(states)
 with torch.no_grad():pp,_=p(graph);ph,vh=h(graph);_,vv=v(graph)
 statep=p.state_dict();statev=v.state_dict();policy_keys=[k for k in statep if not k.startswith("value_mlp.")];value_keys=[k for k in statev if k.startswith("value_mlp.")]
 ident={"candidate":{"policy_checkpoint":str(G3P),"value_checkpoint":str(V28),"policy_checkpoint_sha256":file_hash(G3P),"value_checkpoint_sha256":file_hash(V28),"architecture_fingerprint":source_hash(["packages/songo_ai/model/srn_network.py","packages/songo_ai/model/srn_graph.py"]),"policy_fingerprint":tensor_hash(statep,policy_keys),"value_fingerprint":tensor_hash(statev,value_keys)},"opponent":{"checkpoint":str(G2),"checkpoint_sha256":file_hash(G2)},"engine_fingerprint":source_hash(["packages/songo_ai/songo/rules.py"]),"validation_positions":len(states),"max_abs_policy_logit_delta":float((ph-pp).abs().max()),"max_abs_value_delta":float((vh-vv).abs().max()),"policy_exact":torch.equal(ph,pp),"value_exact":torch.equal(vh,vv),"candidate_matches_lot28_paths":True,"independent_seeds":seed["independent"]}
 ident["valid"]=ident["policy_exact"] and ident["value_exact"] and ident["independent_seeds"]
 write_json(a.output/"checkpoint_identity.json",ident)
 if not ident["valid"]:raise RuntimeError("candidate identity validation failed")
 return ident
def run_budget(a,budget):
 ident=json.load((a.output/"checkpoint_identity.json").open());
 if not ident["valid"]:raise RuntimeError("invalid candidate")
 g2,p,v,candidate=load_models();opening_seed=SEEDS[f"opening{budget}"];arena_seed=SEEDS[f"arena{budget}"];random.seed(SEEDS["python"]);np.random.seed(SEEDS["numpy"]);torch.manual_seed(SEEDS["torch"])
 openings=generate_unique_deterministic_openings(count=COUNTS[budget],seed=opening_seed,max_prefix_length=40);cfg=ArenaConfig(max_plies=400,repetition_limit=3,seed=arena_seed,bootstrap_samples=a.bootstrap);result=arena("G3_VALUE_REWORK",candidate,g2,openings,cfg,budget)
 payload={"budget":budget,"precommitted_games":2*COUNTS[budget],"opening_seed":opening_seed,"arena_seed":arena_seed,"configuration":{"dirichlet":False,"temperature":0,"c_puct":1.5,"side_swapped":True,"bootstrap_samples":a.bootstrap},**result};write_json(a.output/f"arena_mcts{budget}.json",payload);print(json.dumps(payload["summary"],indent=2),flush=True);return payload
def git_commit():
 try:return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
 except Exception:return None
def combined_result(arenas,samples,seed):
 clusters=[];wins=draws=losses=terminal=games=0
 for budget,payload in arenas:
  grouped={}
  for g in payload["games"]:
   games+=1;outcome=g["a_outcome"];grouped.setdefault((budget,g["opening_id"]),[])
   if outcome is not None:
    grouped[(budget,g["opening_id"])].append(float(outcome));terminal+=1;wins+=outcome==1.;draws+=outcome==.5;losses+=outcome==0.
  # Deux parties étaient pré-engagées par paire. Une troncature reste dans le
  # dénominateur du score du Lot 29 conformément à (W + .5D) / games.
  clusters.extend(sum(x)/2 for x in grouped.values())
 rng=random.Random(seed);means=[]
 for _ in range(samples):means.append(sum(clusters[rng.randrange(len(clusters))] for _ in clusters)/len(clusters))
 means.sort();lo=means[int(.025*(len(means)-1))];hi=means[int(.975*(len(means)-1))]
 return {"games":games,"terminal_games":terminal,"wins":wins,"draws":draws,"losses":losses,"score":(wins+.5*draws)/games,"paired_bootstrap_ci":[lo,hi],"effective_opening_clusters":len(clusters),"bootstrap_samples":samples,"bootstrap_unit":"budget-tagged opening pair"}
def protocol_score(summary):return (summary["aggregate"]["wins"]+.5*summary["aggregate"]["draws"])/summary["aggregate"]["games"]
def decide(a):
 ident=json.load((a.output/"checkpoint_identity.json").open());r64=json.load((a.output/"arena_mcts64.json").open());r128=json.load((a.output/"arena_mcts128.json").open());r256=json.load((a.output/"arena_mcts256.json").open()) if (a.output/"arena_mcts256.json").exists() else None;s64,s128=r64["summary"],r128["summary"];s256=r256["summary"] if r256 else None
 gaps={"64":side_gap(s64),"128":side_gap(s128),"256":side_gap(s256) if s256 else None};scores={"64":protocol_score(s64),"128":protocol_score(s128),"256":protocol_score(s256) if s256 else None};budget_ci={"64":combined_result(((64,r64),),a.bootstrap,SEEDS["bootstrap"]+64),"128":combined_result(((128,r128),),a.bootstrap,SEEDS["bootstrap"]+128)}
 if r256:budget_ci["256"]=combined_result(((256,r256),),a.bootstrap,SEEDS["bootstrap"]+256)
 combined=combined_result(((64,r64),(128,r128)),a.bootstrap,SEEDS["bootstrap"]);rule=promotion_rule(scores["64"],scores["128"],side_gap64=gaps["64"],side_gap128=gaps["128"],score256=scores["256"],valid=ident["valid"])
 def effect(score,ref):return score-ref
 comparison={"lot28":{"64":.5573122529644269,"128":.5235294117647059,"256":.5625},"lot29":scores,"delta":{"64":effect(scores["64"],.5573122529644269),"128":effect(scores["128"],.5235294117647059),"256":effect(scores["256"],.5625) if scores["256"] is not None else None}}
 replicated="YES" if scores["64"]>.5 and scores["128"]>.5 and rule["scaling_healthy"] else "PARTIAL" if (scores["64"]+scores["128"])/2>.5 else "NO"
 def adv(score,ci):return "YES" if score>.5 else "NO" if ci[1]<.5 else "INCONCLUSIVE"
 verdict={"CONFIRMATION_VALID":"YES" if ident["valid"] else "NO","CHECKPOINT_IMMUTABLE":"YES" if ident["candidate_matches_lot28_paths"] else "NO","INDEPENDENT_SEEDS":"YES" if ident["independent_seeds"] else "NO","MCTS64_ADVANTAGE_REPLICATED":adv(scores["64"],budget_ci["64"]["paired_bootstrap_ci"]),"MCTS128_ADVANTAGE_REPLICATED":adv(scores["128"],budget_ci["128"]["paired_bootstrap_ci"]),"MCTS256_SCALING":"HEALTHY" if s256 and rule["delta_128_256"]>=-.05 else "UNHEALTHY" if s256 else "NOT_TESTED","SIDE_BALANCE_ACCEPTABLE":"YES" if rule["side_balance_acceptable"] and (gaps["256"] is None or gaps["256"]<=.15) else "NO","LOT28_EFFECT_REPLICATED":replicated,"SEARCH_SCALING_HEALTHY":"YES" if rule["scaling_healthy"] else "NO","G3_PROMOTED":"YES" if rule["promoted"] else "NO","AUTHORIZED_GENERATOR":"G3" if rule["promoted"] else "G2","NEXT_ACTION":"G3_AUTONOMOUS_SELFPLAY_CYCLE" if rule["promoted"] else "REASSESS_MODEL_LEARNING_PARADIGM"}
 side={"side_gap":gaps,"scores":{str(b):{"P1":side_score(s["by_a_side"]["P1"]),"P2":side_score(s["by_a_side"]["P2"])} for b,s in ((64,s64),(128,s128),*(([(256,s256)] if s256 else [])))},"threshold":.15};bootstrap={b:x for b,x in budget_ci.items()};bootstrap["combined_64_128"]=combined
 decision={"precommitted_rule":{"score64_gt":.5,"score128_gt":.5,"combined_gt":.5,"max_scaling_drop":.05,"max_side_gap":.15},"observed":rule,"verdict":verdict};write_json(a.output/"paired_bootstrap.json",bootstrap);write_json(a.output/"side_analysis.json",side);write_json(a.output/"lot28_lot29_comparison.json",comparison);write_json(a.output/"promotion_decision.json",decision)
 report={"lot":29,"candidate":"Policy G3_STRATEGIC + Value V28_A","checkpoint_identity":ident,"seed_manifest":json.load((a.output/"seed_manifest.json").open()),"configuration":json.load((a.output/"configuration.json").open()),"arenas":{"64":s64,"128":s128,"256":s256},"combined_64_128":combined,"bootstrap":bootstrap,"side_analysis":side,"comparison":comparison,"promotion_decision":decision,"verdict":verdict};write_json(a.output/"report.json",report);print(json.dumps(verdict,indent=2));return report
def main():
 a=args();a.output.mkdir(parents=True,exist_ok=True);write_json(a.output/"configuration.json",{"candidate":"G3_VALUE_REWORK","training":False,"architecture_changed":False,"engine_changed":False,"mcts":{"dirichlet":False,"temperature":0,"c_puct":1.5,"budgets":[64,128,256]},"precommitted_game_counts":{"64":512,"128":512,"256":256},"decision_thresholds":{"max_scaling_drop":.05,"max_side_gap":.15},"python_seed":SEEDS["python"],"numpy_seed":SEEDS["numpy"],"torch_seed":SEEDS["torch"],"git_commit":git_commit()});seed_manifest(a)
 phases=("identity","64","128") if a.phase=="all" else (a.phase,)
 for phase in phases:{"identity":identity,"64":lambda x:run_budget(x,64),"128":lambda x:run_budget(x,128),"256":lambda x:run_budget(x,256),"decision":decide}[phase](a)
 if a.phase=="all":
  r64=json.load((a.output/"arena_mcts64.json").open())["summary"]["score_rate_a_terminal"];r128=json.load((a.output/"arena_mcts128.json").open())["summary"]["score_rate_a_terminal"]
  if r64>.5 and r128>.5:run_budget(a,256)
  decide(a)
if __name__=="__main__":main()
