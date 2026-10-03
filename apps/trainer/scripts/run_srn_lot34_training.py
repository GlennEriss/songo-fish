#!/usr/bin/env python3
"""Lot 34: integrity gate, common autonomous reanalysis, and controlled training."""
from __future__ import annotations
import argparse,hashlib,json,math,random,statistics,time
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
import torch
import torch.nn.functional as F
from songo_ai.dataset import RawSongoState,read_d_rl_jsonl
from songo_ai.evaluation import physical_state_key,state_coverage
from songo_ai.model import SongoGraphBuilder,SRNTrainingConfig,load_srn_checkpoint,mask_policy_logits
from songo_ai.search import MCTSConfig,SongoMCTS
from run_srn_lot12 import sha256,write_json
from run_srn_lot32 import G2,G3P,V28,models,quality_control

OUT=Path("data/experiments/lot34_g4_training");SEED=20263400
def cli():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("audit","reanalysis","train-control","train-pool","select","all"),default="audit");p.add_argument("--output",type=Path,default=OUT);return p.parse_args()
def read_arm(root):return [x for p in sorted(root.glob("part-*.jsonl")) for x in read_d_rl_jsonl(p)]
def games(examples):
 out=defaultdict(list)
 for x in examples:out[x.metadata["game_id"]].append(x)
 return out
def stats(examples):
 grouped=games(examples);lengths=[len(x) for x in grouped.values()]
 return {**state_coverage(examples),"games":len(grouped),"mean_game_length":statistics.fmean(lengths),"median_game_length":statistics.median(lengths),"terminal_games":sum(not rows[0].metadata["status"].startswith("TRUNCATED") for rows in grouped.values()),"truncated_games":sum(rows[0].metadata["status"].startswith("TRUNCATED") for rows in grouped.values()),"missing_value_examples":sum(x.value_target is None for x in examples)}
def audit(a):
 control=read_arm(a.output/"control_data");pool=read_arm(a.output/"pool_data");cm=json.load((a.output/"control_dataset_manifest.json").open());pm=json.load((a.output/"pool_dataset_manifest.json").open());identity=json.load((a.output/"model_identity.json").open())
 qc={"CONTROL":quality_control(control),"POOL":quality_control(pool)}
 hashes={"G2":sha256(G2)==identity["G2"]["sha256"],"G3_POLICY":sha256(G3P)==identity["G3_VALUE_REWORK"]["policy_sha256"],"G3_VALUE":sha256(V28)==identity["G3_VALUE_REWORK"]["value_sha256"]}
 contracts={"control_exact_games":cm["actual_games"]==8000,"pool_exact_games":pm["actual_games"]==8000,"control_sources":cm["sources"]=={"G2_G2":8000},"pool_sources":pm["sources"]=={"G2_G2":800,"G3_G3":800,"CROSS_PLAY":6400},"crossplay_side_balance":pm["crossplay_sides"]=={"G2_P1":3200,"G3_P1":3200},"checkpoint_identity":all(hashes.values()),"missing_z_truncated_only":all(x.metadata["status"].startswith("TRUNCATED") for x in control+pool if x.value_target is None),"provenance_complete":all(all(k in x.metadata for k in ("arm","generation_id","source_type","p1_model_id","p2_model_id","p1_role","p2_role","mcts_budget","selfplay_seed")) for x in control+pool)}
 allowed=all(contracts.values()) and all(x["valid"] for x in qc.values());report={"CONTROL":stats(control),"POOL":stats(pool),"quality_control":qc,"contracts":contracts,"checkpoint_hashes_valid":hashes,"TRAINING_ALLOWED":"YES" if allowed else "NO"};write_json(a.output/"data_generation_report.json",report)
 config=json.load((a.output/"configuration.json").open());config["training_allowed"]="YES" if allowed else "NO";write_json(a.output/"configuration.json",config)
 if not allowed:raise RuntimeError("Lot34 integrity gate failed; training forbidden")
 print(json.dumps({"TRAINING_ALLOWED":"YES","CONTROL":report["CONTROL"],"POOL":report["POOL"]},indent=2));return report
def existing_reanalysis_rows(limit=40000):
 rows=[]
 for path in sorted(Path("data/d_scale_v1/d_reanalysis_large").glob("part-*.jsonl")):
  for line in path.open():
   if line.strip():rows.append(json.loads(line))
   if len(rows)>=limit:return rows
 raise RuntimeError("insufficient autonomous reanalysis reservoir")
def reanalysis(a):
 gate=json.load((a.output/"data_generation_report.json").open());
 if gate["TRAINING_ALLOWED"]!="YES":raise RuntimeError("integrity gate not open")
 root=a.output/"common_reanalysis";root.mkdir(exist_ok=True);selected_path=root/"g2_selected_40000.jsonl";rows=existing_reanalysis_rows()
 if not selected_path.exists():
  with selected_path.open("w") as f:
   for row in rows:f.write(json.dumps(row,sort_keys=True)+"\n")
 selection_hash=sha256(selected_path);_,_,_,g3=models();parts=[]
 for start in range(0,40000,1000):
  part=start//1000;path=root/f"g3-part-{part:04d}.jsonl";meta=root/f"g3-part-{part:04d}.manifest.json";subset=rows[start:start+1000]
  if path.exists() and meta.exists():m=json.load(meta.open());print(f"[lot34] reanalysis shard {part+1}/40 reused",flush=True)
  else:
   began=time.perf_counter();search=SongoMCTS(g3,config=MCTSConfig(num_simulations=128,c_puct=1.5,add_root_noise=False,seed=SEED+500000+part));sims=0
   with path.open("w") as f:
    for row in subset:
     s=row["state"];result=search.search(RawSongoState(tuple(s["board"]),int(s["player_to_move"])),policy_temperature=1.);record={"state":s,"legal_mask":row["legal_mask"],"visit_counts":list(result.visit_counts),"policy_target":list(result.policy),"source_dataset":"AUTONOMOUS_REANALYSIS","source_position_reservoir":"COMMON_LOT34","position_hash":row.get("position_hash",hashlib.sha256(repr(s).encode()).hexdigest()),"generation_model":"G3_VALUE_REWORK","mcts_budget":128,"c_puct":1.5,"dirichlet":False,"seed":SEED+500000+part,"value_target_present":False};f.write(json.dumps(record,sort_keys=True)+"\n");sims+=result.num_simulations
   m={"part":part,"positions":len(subset),"simulations":sims,"elapsed_s":time.perf_counter()-began,"sha256":sha256(path)};write_json(meta,m);print(f"[lot34] reanalysis shard {part+1}/40",flush=True)
  parts.append(m)
 manifest={"physical_states":40000,"policy_targets":80000,"g2_targets":40000,"g3_targets":40000,"g2_source":"D_REANALYSIS_LARGE MCTS128","g2_selection_path":str(selected_path),"g2_selection_sha256":selection_hash,"g3_shards":40,"mcts_budget":128,"same_states":True,"shared_between_training_arms":True,"teacher_labels":False,"minimax_labels":False,"value_targets":False,"elapsed_s_g3":sum(x["elapsed_s"] for x in parts),"g3_simulations":sum(x["simulations"] for x in parts)};write_json(a.output/"common_reanalysis_manifest.json",manifest);return manifest
def split_code(key):return int(hashlib.sha256(str(key).encode()).hexdigest()[:8],16)%100
def split_rl(rows):
 out={"train":[],"validation":[],"test":[]}
 for x in rows:
  n=split_code(x.metadata["game_id"]);out["train" if n<85 else "validation" if n<95 else "test"].append(x)
 return out
def load_historical():
 large=[x for p in sorted(Path("data/d_scale_v1/d_selfplay_large").glob("*.jsonl")) for x in read_d_rl_jsonl(p)];old=[]
 for p in sorted(Path("data/d_rl").glob("*.jsonl")):
  try:old.extend(read_d_rl_jsonl(p))
  except ValueError:pass
 return large,old
def load_reanalysis_rows(a):
 g2=[json.loads(x) for x in (a.output/"common_reanalysis/g2_selected_40000.jsonl").open() if x.strip()];g3=[]
 for p in sorted((a.output/"common_reanalysis").glob("g3-part-*.jsonl")):g3.extend(json.loads(x) for x in p.open() if x.strip())
 return g2,g3
def re_state(x):
 s=x["state"];return RawSongoState(tuple(s["board"]),int(s["player_to_move"]))
def sample_rows(rng,pool,count):return [pool[rng.randrange(len(pool))] for _ in range(count)]
def make_initial_model():
 policy=load_srn_checkpoint(G3P).model;value=load_srn_checkpoint(V28).model;state=policy.state_dict();vstate=value.state_dict()
 for key in state:
  if key.startswith("value_mlp."):state[key]=vstate[key].detach().clone()
 policy.load_state_dict(state);return policy
def initialization_deltas(left,right):
 return max(float((left.state_dict()[k]-right.state_dict()[k]).abs().max()) for k in left.state_dict())
def batch_tensors(rows,builder):
 states=[];masks=[];targets=[];values=[];vmask=[]
 for kind,x in rows:
  if kind=="REANALYSIS":state=re_state(x);mask=x["legal_mask"];target=x["policy_target"];value=0.;has=False
  else:state=x.state;mask=x.legal_mask;target=x.policy_target;value=float(x.value_target or 0.);has=x.value_target is not None
  states.append(state);masks.append(mask);targets.append(target);values.append(value);vmask.append(has)
 return builder.build_batch(states),torch.tensor(masks,dtype=torch.bool),torch.tensor(targets,dtype=torch.float32),torch.tensor(values,dtype=torch.float32),torch.tensor(vmask,dtype=torch.bool)
def evaluate_model(model,rows,builder):
 model.eval();ce=correct=entropy=n=0.;sq=sign=labeled=0
 with torch.no_grad():
  for start in range(0,len(rows),512):
   graph,mask,target,value,vmask=batch_tensors(rows[start:start+512],builder);logits,pred=model(graph);masked=mask_policy_logits(logits,mask);logp=torch.log_softmax(masked,-1);ce+=float((-(target*logp).sum(-1)).sum());correct+=int((masked.argmax(-1)==target.argmax(-1)).sum());prob=torch.softmax(masked,-1);entropy+=float((-(prob*logp).sum(-1)).sum());n+=len(target)
   if vmask.any():d=pred.reshape(-1)[vmask]-value[vmask];sq+=float(d.square().sum());sign+=int((torch.sign(pred.reshape(-1)[vmask])==torch.sign(value[vmask])).sum());labeled+=int(vmask.sum())
 return {"policy_ce":ce/n,"policy_top1":correct/n,"policy_entropy":entropy/n,"value_mse":sq/labeled if labeled else None,"value_sign_accuracy":sign/labeled if labeled else None,"examples":n,"labeled_values":labeled}
def checkpoint_payload(model,opt,step,seed,metrics,arm):
 cfg=SRNTrainingConfig(epochs=1,batch_size=256,learning_rate=.0003,weight_decay=1e-4,gradient_clip_norm=1.,validation_fraction=.1,seed=seed,device="cpu")
 return {"checkpoint_type":"songo_srn_d_rl_training","checkpoint_version":1,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"srn_config":asdict(model.config),"training_config":asdict(cfg),"epoch":step//4000,"global_step":step,"seed":seed,"dataset_manifest":{"dataset_id":f"LOT34_{arm}","examples":"source-aware replay"},"train_game_ids":[],"validation_game_ids":[],"history":[],"best_epoch":step//4000,"best_validation_loss":metrics["policy_ce"]+(metrics["value_mse"] or 0),"initialization":{"policy":"G3_STRATEGIC","value":"V28_A"},"stopped_early":False,"lineage":{"lot":34,"arm":arm,"metrics":metrics}}
def train_arm(a,arm):
 if json.load((a.output/"configuration.json").open())["training_allowed"]!="YES":raise RuntimeError("training gate closed")
 new=read_arm(a.output/("control_data" if arm=="CONTROL" else "pool_data"));large,old=load_historical();g2re,g3re=load_reanalysis_rows(a);newsp=split_rl(new);largesp=split_rl(large);oldsp=split_rl(old)
 def split_re(rows,name):
  return {part:[x for x in rows if (lambda n: part==("train" if n<85 else "validation" if n<95 else "test"))(split_code(name+str(x.get("position_hash",x["state"]))))] for part in ("train","validation","test")}
 g2sp=split_re(g2re,"g2");g3sp=split_re(g3re,"g3");seed=20263401 if arm=="CONTROL" else 20263402;rng=random.Random(seed);torch.manual_seed(seed);model=make_initial_model();reference=make_initial_model();initial_delta=initialization_deltas(model,reference);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-4);builder=SongoGraphBuilder();out=a.output/"checkpoints"/arm.lower();out.mkdir(parents=True,exist_ok=True);observed=Counter();history=[]
 validation=[]
 validation += [("RL",x) for x in newsp["validation"][:2048]];validation += [("RL",x) for x in largesp["validation"][:1024]];validation += [("RL",x) for x in oldsp["validation"][:512]];validation += [("REANALYSIS",x) for x in g2sp["validation"][:256]];validation += [("REANALYSIS",x) for x in g3sp["validation"][:256]]
 new_buckets={s:[x for x in newsp["train"] if x.metadata["source_type"]==s] for s in ("G2_G2","G3_G3","CROSS_PLAY")} if arm=="POOL" else {}
 for step in range(1,32001):
  rows=[]
  if arm=="CONTROL":selected=sample_rows(rng,newsp["train"],179);rows += [("RL",x) for x in selected];observed["NEW_GENERATION"]+=179;observed["NEW_G2_G2"]+=179
  else:
   for source,count in (("G2_G2",18),("G3_G3",18),("CROSS_PLAY",143)):rows += [("RL",x) for x in sample_rows(rng,new_buckets[source],count)];observed[f"NEW_{source}"]+=count
   observed["NEW_GENERATION"]+=179
  rows += [("RL",x) for x in sample_rows(rng,largesp["train"],26)];rows += [("RL",x) for x in sample_rows(rng,oldsp["train"],25)];observed["HISTORICAL_RL"]+=51
  rows += [("REANALYSIS",x) for x in sample_rows(rng,g2sp["train"],13)];rows += [("REANALYSIS",x) for x in sample_rows(rng,g3sp["train"],13)];observed["AUTONOMOUS_REANALYSIS"]+=26;rng.shuffle(rows)
  graph,mask,target,value,vmask=batch_tensors(rows,builder);model.train();logits,pred=model(graph);ploss=-(target*torch.log_softmax(mask_policy_logits(logits,mask),-1)).sum(-1).mean();vloss=(pred.reshape(-1)[vmask]-value[vmask]).square().mean();loss=ploss+vloss;opt.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step()
  if step%4000==0:
   metrics=evaluate_model(model,validation,builder);metrics.update({"step":step,"train_policy_loss":float(ploss),"train_value_loss":float(vloss),"gradient_norm":grad});history.append(metrics);torch.save(checkpoint_payload(model,opt,step,seed,metrics,arm),out/f"step-{step:05d}.pt");print(f"[lot34] {arm} step {step}/32000 val_ce={metrics['policy_ce']:.5f} val_mse={metrics['value_mse']:.5f}",flush=True)
 report={"arm":arm,"updates":32000,"batch_size":256,"optimizer":"AdamW","learning_rate":.0003,"weight_decay":1e-4,"gradient_clip":1.,"seed":seed,"initialization_parameter_delta_vs_reference":initial_delta,"observed_samples":dict(observed),"requested_ratio":{"NEW_GENERATION":.7,"HISTORICAL_RL":.2,"AUTONOMOUS_REANALYSIS":.1},"observed_ratio":{k:observed[k]/sum(observed[x] for x in ("NEW_GENERATION","HISTORICAL_RL","AUTONOMOUS_REANALYSIS")) for k in ("NEW_GENERATION","HISTORICAL_RL","AUTONOMOUS_REANALYSIS")},"checkpoints":8,"history":history,"split":{"method":"game_id hash 85/10/5","new":{k:len(v) for k,v in newsp.items()},"historical_large":{k:len(v) for k,v in largesp.items()},"historical_old":{k:len(v) for k,v in oldsp.items()},"reanalysis_g2":{k:len(v) for k,v in g2sp.items()},"reanalysis_g3":{k:len(v) for k,v in g3sp.items()}},"teacher_labels":False,"minimax_labels":False};write_json(a.output/("control_training_report.json" if arm=="CONTROL" else "pool_training_report.json"),report);return report
def select(a):
 candidates={};policy=[];value=[]
 for arm in ("CONTROL","POOL"):
  report=json.load((a.output/("control_training_report.json" if arm=="CONTROL" else "pool_training_report.json")).open());rows=report["history"]
  for r in rows:
   path=a.output/"checkpoints"/arm.lower()/f"step-{r['step']:05d}.pt";policy.append({"arm":arm,"step":r["step"],"path":str(path),"policy_ce":r["policy_ce"],"policy_top1":r["policy_top1"],"policy_entropy":r["policy_entropy"]});value.append({"arm":arm,"step":r["step"],"path":str(path),"value_mse":r["value_mse"],"value_sign_accuracy":r["value_sign_accuracy"]})
  candidates[arm]={"policy":sorted([x for x in policy if x["arm"]==arm],key=lambda x:x["policy_ce"])[:2],"value":sorted([x for x in value if x["arm"]==arm],key=lambda x:x["value_mse"])[:2]}
 write_json(a.output/"policy_candidates.json",{"max_per_arm":2,"candidates":{a:candidates[a]["policy"] for a in candidates}});write_json(a.output/"value_candidates.json",{"max_per_arm":2,"candidates":{a:candidates[a]["value"] for a in candidates}})
 matrix={arm:[{"policy":p,"value":v,"combination_id":f"{arm}_P{pi+1}V{vi+1}"} for pi,p in enumerate(candidates[arm]["policy"]) for vi,v in enumerate(candidates[arm]["value"])] for arm in candidates};write_json(a.output/"component_matrix.json",{"max_combinations_per_arm":4,"matrix":matrix});return matrix
def main():
 a=cli();stages=("audit","reanalysis","train-control","train-pool","select") if a.stage=="all" else (a.stage,)
 for stage in stages:
  if stage=="audit":audit(a)
  elif stage=="reanalysis":reanalysis(a)
  elif stage=="train-control":train_arm(a,"CONTROL")
  elif stage=="train-pool":train_arm(a,"POOL")
  elif stage=="select":select(a)
if __name__=="__main__":main()
