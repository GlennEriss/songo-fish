#!/usr/bin/env python3
"""Lot 27 : diagnostic sans entraînement de l'interaction Policy–MCTS."""
from __future__ import annotations
import argparse,json,math,statistics,time
from collections import Counter
from pathlib import Path
import torch
from songo_ai.dataset import RawSongoState
from songo_ai.evaluation.search_policy_diagnosis import amplification_ratio,first_divergence,js_divergence,l1_distance,regret,rwpm,search_category,search_recovery
from songo_ai.evaluation import generate_unique_deterministic_openings
from songo_ai.model import SongoGraphBuilder,load_srn_checkpoint,policy_probabilities
from songo_ai.search import MCTSConfig,SongoMCTS
from songo_ai.songo.rules import SongoLegacyGame
from run_srn_lot12 import write_json

BUDGETS=(8,16,32,64,128,256)
MODELS={"G2":"data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt","G3-SCALE":"data/experiments/lot26_g3_scale/checkpoints/g3_scale_best.pt","G3-STRATEGIC":"data/experiments/lot26_g3_scale/checkpoints/g3_strategic_best.pt"}
def args():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("data/experiments/lot27_search_policy"));p.add_argument("--positions",type=int,default=2000);p.add_argument("--deep",type=int,default=400);p.add_argument("--seed",type=int,default=20262727);return p.parse_args()
def ph(s):
 import hashlib
 return hashlib.sha256((",".join(map(str,s.board))+f"|{s.player_to_move}").encode()).hexdigest()
def battery(count):
 arena=json.load(Path("data/experiments/lot26_g3_scale/main_arena.json").open());seen={};sources=Counter();openings={x.opening_id:x for x in generate_unique_deterministic_openings(count=128,seed=20262628,max_prefix_length=40)}
 for candidate in ("G3-SCALE","G3-STRATEGIC"):
  for budget in ("64","128"):
   for game_row in arena[candidate][budget]["games"]:
    opening=openings[game_row["opening_id"]];game=SongoLegacyGame.from_state(opening.state.to_engine_state())
    for ply,action in enumerate(game_row["action_sequence"]):
     if game.finished:break
     state=RawSongoState.from_game(game);key=ph(state)
     if key not in seen:seen[key]={"position_hash":key,"state":{"board":list(state.board),"player_to_move":state.player_to_move},"legal_mask":list(game.legal_mask()),"source":"LOT26_ARENA_TRAJECTORY","arena_candidate":candidate,"arena_budget":int(budget),"ply":ply};sources[f"{candidate}@{budget}"]+=1
     game.play_local(action)
    if len(seen)>=count*3:break
  if len(seen)>=count*3:break
 rows=sorted(seen.values(),key=lambda r:r["position_hash"]);step=max(1,len(rows)//count);selected=rows[::step][:count]
 return selected,{"unique_pool":len(rows),"selected":len(selected),"source_counts":dict(sources),"independent_from_checkpoint_selection":True}
def policy_values(rows,models):
 builder=SongoGraphBuilder();out={n:[] for n in models}
 with torch.no_grad():
  for start in range(0,len(rows),512):
   c=rows[start:start+512];g=builder.build_batch([RawSongoState(tuple(r["state"]["board"]),r["state"]["player_to_move"]) for r in c]);mask=torch.tensor([r["legal_mask"] for r in c])
   for n,m in models.items():logits,v=m(g);p=policy_probabilities(logits,mask);out[n].extend({"policy":x,"value":float(y),"logit_std":float(z[r["legal_mask"]].std()) if sum(r["legal_mask"])>1 else 0.} for x,y,z,r in zip(p.tolist(),v.reshape(-1),logits,c))
 return out
def checkpoint(trace,k,legal):
 t=trace[k-1];vis=t["visit_counts_after_backup"];action=max((i for i,x in enumerate(legal) if x),key=lambda i:(vis[i],-i));tot=sum(vis);policy=[x/tot for x in vis];return {"action":action,"visits":vis,"policy":policy,"q":t["root_q_values_after_backup"],"puct":t["root_puct_scores_after_backup"]}
def run_search(a,rows,models):
 a.output.mkdir(parents=True,exist_ok=True);cache=a.output/"search_cache.jsonl";known={}
 if cache.exists():
  for line in cache.open():r=json.loads(line);known[(r["position_hash"],r["model"])]=r
 with cache.open("a") as out:
  for ordinal,row in enumerate(rows,1):
   state=RawSongoState(tuple(row["state"]["board"]),row["state"]["player_to_move"])
   for name,model in models.items():
    key=(row["position_hash"],name)
    if key not in known:
     result=SongoMCTS(model,config=MCTSConfig(num_simulations=256,c_puct=1.5,add_root_noise=False,collect_simulation_trace=True,seed=a.seed)).search(state,policy_temperature=1.);points={str(k):checkpoint(result.simulation_trace,k,row["legal_mask"]) for k in BUDGETS};item={"position_hash":row["position_hash"],"model":name,"root_priors":result.root_priors,"root_value":result.root_value,"points":points,"first_actions":[t["root_action"] for t in result.simulation_trace],"nodes":result.num_nodes,"deep_trace":list(result.simulation_trace) if ordinal<=a.deep else None};out.write(json.dumps(item,sort_keys=True)+"\n");out.flush();known[key]=item
   if ordinal%25==0:print(f"[lot27] search {ordinal}/{len(rows)}",flush=True)
 return known
def dist(v):
 v=sorted(map(float,v));
 if not v:return {"count":0}
 def q(p):x=p*(len(v)-1);i=int(x);j=min(i+1,len(v)-1);return v[i]*(j-x)+v[j]*(x-i)
 return {"count":len(v),"mean":statistics.fmean(v),"median":statistics.median(v),"p75":q(.75),"p90":q(.9),"p95":q(.95),"p99":q(.99),"max":v[-1]}
def analyze(a,rows,pv,search):
 pairs=(("G2","G3-SCALE"),("G2","G3-STRATEGIC"),("G3-SCALE","G3-STRATEGIC"));poldiv={};scaling={n:{str(k):Counter() for k in BUDGETS} for n in pv};amp={};flips={};cases=[]
 for left,right in pairs:
  key=f"{left}_vs_{right}";pjs=[];l1=[];arg=[];top2=[];ents=[];ar={str(k):[] for k in BUDGETS};cats={str(k):Counter() for k in BUDGETS};first=[]
  for i,row in enumerate(rows):
   lp,rp=pv[left][i]["policy"],pv[right][i]["policy"];pj=js_divergence(lp,rp);pjs.append(pj);l1.append(l1_distance(lp,rp));la=max(range(7),key=lambda x:lp[x]);ra=max(range(7),key=lambda x:rp[x]);same=la==ra;arg.append(same);top2.append(len(set(sorted(range(7),key=lambda x:lp[x])[-2:])&set(sorted(range(7),key=lambda x:rp[x])[-2:]))/2);ents.append(sum(-x*math.log(x) for x in lp if x)-sum(-x*math.log(x) for x in rp if x));first.append(first_divergence(search[(row["position_hash"],left)]["first_actions"],search[(row["position_hash"],right)]["first_actions"]))
   for k in BUDGETS:
    ls=search[(row["position_hash"],left)]["points"][str(k)];rs=search[(row["position_hash"],right)]["points"][str(k)];vj=js_divergence(ls["policy"],rs["policy"]);ar[str(k)].append(amplification_ratio(pj,vj));cats[str(k)][search_category(same,ls["action"]==rs["action"])]+=1
   if left=="G2" and right=="G3-STRATEGIC" and search[(row["position_hash"],right)]["points"]["64"]["action"]!=search[(row["position_hash"],right)]["points"]["128"]["action"]:cases.append({"position_hash":row["position_hash"],"state":row["state"],"legal_mask":row["legal_mask"],"P_G2":lp,"P_G3STRAT":rp,"G2":search[(row["position_hash"],left)]["points"],"G3STRAT":search[(row["position_hash"],right)]["points"]})
  poldiv[key]={"policy_js":dist(pjs),"policy_l1":dist(l1),"argmax_agreement":statistics.fmean(arg),"top2_overlap":statistics.fmean(top2),"entropy_delta":dist(ents),"first_root_divergence":dist([x for x in first if x is not None])};amp[key]={k:dist(v) for k,v in ar.items()};flips[key]={k:dict(v) for k,v in cats.items()}
 for n in pv:
  for k in BUDGETS:
   scaling[n][str(k)]={"action_distribution":dict(Counter(str(search[(r["position_hash"],n)]["points"][str(k)]["action"]) for r in rows)),"recovery_rate":statistics.fmean(search_recovery([search[(r["position_hash"],n)]["points"][str(x)]["action"] for x in BUDGETS]) for r in rows)}
 write_json(a.output/"policy_divergence.json",poldiv);write_json(a.output/"amplification.json",amp);write_json(a.output/"search_flips.json",flips);write_json(a.output/"search_scaling.json",scaling);write_json(a.output/"scaling_failure_set.json",{"positions":len(cases),"rows":cases});write_json(a.output/"representative_cases.json",cases[:20]);return poldiv,amp,flips,cases
def qdiag(a,models):
 rows=[json.loads(x) for x in Path("data/d_scale_v1/d_strategic_sample/qdiag256.jsonl").open() if x.strip()];pv=policy_values(rows,models);result={}
 for n in models:
  values=[];high=[]
  for r,x in zip(rows,pv[n]):values.append(rwpm(x["policy"],r["q_values"],r["legal_mask"]));high.append(sum(x["policy"][i] for i,ok in enumerate(r["legal_mask"]) if ok and regret(r["q_values"],i,r["legal_mask"])>.1))
  result[n]={"rwpm":dist(values),"high_regret_mass_gt_0.1":dist(high)}
 write_json(a.output/"rwpm.json",result);write_json(a.output/"regret_tail.json",result);return result
def main():
 a=args();started=time.perf_counter();rows,manifest=battery(a.positions);models={n:load_srn_checkpoint(p).model for n,p in MODELS.items()};pv=policy_values(rows,models);search=run_search(a,rows,models);pol,amp,flips,cases=analyze(a,rows,pv,search);rw=qdiag(a,models);write_json(a.output/"configuration.json",{"positions":len(rows),"deep_positions":a.deep,"budgets":BUDGETS,"c_puct":1.5,"dirichlet":False,"temperature":0,"training_performed":False,"battery":manifest});print(json.dumps({"positions":len(rows),"failure_set":len(cases),"elapsed_s":time.perf_counter()-started},indent=2))
if __name__=="__main__":main()
