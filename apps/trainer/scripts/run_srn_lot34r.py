#!/usr/bin/env python3
"""Lot 34R: forensic analysis and controlled strategic-preservation retry."""
from __future__ import annotations
import argparse,hashlib,json,random
from collections import Counter
from types import SimpleNamespace
from pathlib import Path
import torch
from songo_ai.dataset import iter_reanalysis_jsonl,read_d_rl_jsonl
from songo_ai.model import SongoGraphBuilder,load_srn_checkpoint,mask_policy_logits
from songo_ai.model.correct_preserve import correct_preserve_metrics,correction_loss,preservation_loss
from run_srn_lot12 import sha256,write_json
from run_srn_lot20 import collate
from run_srn_lot23 import reconstruct_d_rank
from run_srn_lot24 import attach_pair_types,cp_batch
from run_srn_lot26 import attach_strategic,strategic_batch
from run_srn_lot32 import G3P
from run_srn_lot34_training import batch_tensors,checkpoint_payload,evaluate_model,load_historical,load_reanalysis_rows,make_initial_model,read_arm,sample_rows,split_code,split_rl,stats

LOT34=Path("data/experiments/lot34_g4_training")
OUT=Path("data/experiments/lot34r_g4_retry")
BATTERY=Path("data/d_scale_v1/d_strategic_sample/qdiag256.jsonl")
GATE=.85

def cli():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("audit","forensics","train-control","train-pool","select","finalize"),default="audit");p.add_argument("--output",type=Path,default=OUT);return p.parse_args()

def tree_hash(root):
 h=hashlib.sha256();files=sorted(root.glob("part-*.jsonl"))
 for path in files:h.update(path.name.encode());h.update(bytes.fromhex(sha256(path)))
 return {"sha256":h.hexdigest(),"files":len(files)}

def audit(a):
 a.output.mkdir(parents=True,exist_ok=True);control=read_arm(LOT34/"control_data");pool=read_arm(LOT34/"pool_data");cm=json.load((LOT34/"control_dataset_manifest.json").open());pm=json.load((LOT34/"pool_dataset_manifest.json").open());cs=stats(control);ps=stats(pool)
 checks={"control_games":cs["games"]==cm["actual_games"]==8000,"control_positions":cs["raw_positions"]==cm["positions"]==704816,"pool_games":ps["games"]==pm["actual_games"]==8000,"pool_positions":ps["raw_positions"]==pm["positions"]==714700,"pool_ratio":pm["sources"]=={"G2_G2":800,"G3_G3":800,"CROSS_PLAY":6400},"pool_side_balance":pm["crossplay_sides"]=={"G2_P1":3200,"G3_P1":3200}}
 identity={"source":"LOT34 immutable generated corpora","CONTROL":{"manifest_sha256":sha256(LOT34/"control_dataset_manifest.json"),"data":tree_hash(LOT34/"control_data"),"games":cs["games"],"positions":cs["raw_positions"]},"POOL":{"manifest_sha256":sha256(LOT34/"pool_dataset_manifest.json"),"data":tree_hash(LOT34/"pool_data"),"games":ps["games"],"positions":ps["raw_positions"]},"checks":checks,"LOT34_DATA_REUSED_EXACTLY":"YES" if all(checks.values()) else "NO","NEW_GENERATION_PERFORMED":"NO"};write_json(a.output/"lot34_dataset_identity.json",identity)
 config={"lot":"34R","retry_of":34,"strategic_preservation_gate":GATE,"new_generation":False,"architecture_modified":False,"engine_modified":False,"mcts_modified":False,"initial_policy":"G3_STRATEGIC","initial_value":"V28_A","source_ratio":{"NEW_GENERATION":.70,"HISTORICAL_RL":.20,"AUTONOMOUS_REANALYSIS":.10},"max_updates":32000,"max_policy_candidates":2,"max_value_candidates":2,"max_combinations":4,"battery":str(BATTERY),"battery_sha256":sha256(BATTERY),"retry_allowed":all(checks.values())};write_json(a.output/"configuration.json",config)
 if not all(checks.values()):raise RuntimeError("RETRY_ALLOWED=NO: Lot34 corpus identity mismatch")
 print(json.dumps(identity,indent=2));return identity

def cp_metrics(model,items):
 parts=[]
 with torch.no_grad():
  for start in range(0,len(items),256):
   chunk=items[start:start+256];logits,corr,pres=strategic_batch(model,chunk);parts.append(correct_preserve_metrics(logits,corr,pres))
 cn=sum(x["correction_pairs"] for x in parts);pn=sum(x["preservation_pairs"] for x in parts)
 return {"correction_pairs":cn,"preservation_pairs":pn,"correction_rate":sum((x["correction_rate"] or 0)*x["correction_pairs"] for x in parts)/cn,"preservation_rate":sum((x["preservation_rate"] or 0)*x["preservation_pairs"] for x in parts)/pn,"strategic_utility":sum(x["strategic_utility"] for x in parts)}

def forensics(a):
 if json.load((a.output/"lot34_dataset_identity.json").open())["LOT34_DATA_REUSED_EXACTLY"]!="YES":raise RuntimeError("retry audit not open")
 raw=[json.loads(x) for x in BATTERY.open() if x.strip()];initial=make_initial_model();items=attach_strategic(raw,load_srn_checkpoint(G3P).model);base=cp_metrics(initial,items);curves={};safe={};crossings={}
 for arm in ("CONTROL","POOL"):
  report=json.load((LOT34/("control_training_report.json" if arm=="CONTROL" else "pool_training_report.json")).open());by_step={x["step"]:x for x in report["history"]};rows=[{"update":0,"policy_ce":None,"policy_top1":None,**base,"parameter_change_nonzero":False,"safe_learning_region":False}]
  for path in sorted((LOT34/"checkpoints"/arm.lower()).glob("step-*.pt")):
   step=int(path.stem.split("-")[-1]);m=cp_metrics(load_srn_checkpoint(path).model,items);r=by_step[step];rows.append({"update":step,"checkpoint":str(path),"checkpoint_sha256":sha256(path),"policy_ce":r["policy_ce"],"policy_top1":r["policy_top1"],**m,"strategic_regression":1-m["preservation_rate"],"parameter_change_nonzero":True,"safe_learning_region":m["preservation_rate"]>=GATE and m["correction_rate"]>base["correction_rate"]})
  curves[arm]=rows;eligible=[x for x in rows if x["safe_learning_region"]];safe[arm]=bool(eligible);above=[x for x in rows if x["preservation_rate"]>=GATE];below=[x for x in rows if x["preservation_rate"]<GATE];crossings[arm]={"last_checkpoint_at_or_above_85":above[-1]["update"] if above else None,"first_checkpoint_below_85":below[0]["update"] if below else None,"safe_nontrivial_updates":[x["update"] for x in eligible]}
 cause="CHECKPOINT_SELECTION" if all(safe.values()) else "POLICY_OBJECTIVE_DRIFT"
 forensic={"valid":True,"gate":GATE,"update0":base,"curves":curves,"crossings":crossings,"SAFE_CHECKPOINT_EXISTS_CONTROL":"YES" if safe["CONTROL"] else "NO","SAFE_CHECKPOINT_EXISTS_POOL":"YES" if safe["POOL"] else "NO"};write_json(a.output/"checkpoint_forensics.json",forensic);write_json(a.output/"preservation_curves.json",curves);write_json(a.output/"safe_learning_region.json",{"definition":"preservation >= .85, non-zero Policy change, correction improvement over update0","gate":GATE,"CONTROL":safe["CONTROL"],"POOL":safe["POOL"],"crossings":crossings});write_json(a.output/"failure_cause.json",{"PRIMARY_LOT34_FAILURE_CAUSE":cause,"evidence":"Existing non-trivial safe checkpoints in both arms" if cause=="CHECKPOINT_SELECTION" else "At least one arm has no saved non-trivial checkpoint in the safe region; one common preservation intervention is required","causal_language":"supports/consistent with"})
 intervention={"required":not all(safe.values()),"name":"NONE" if all(safe.values()) else "CORRECT_AND_PRESERVE","pre_registered_before_retry_training":True,"common_to_both_arms":True,"lambda_correction":None if all(safe.values()) else .1,"lambda_preservation":None if all(safe.values()) else 1.0,"rho":None if all(safe.values()) else .5,"justification":"Reuse Lot24 selected configuration; no sweep","no_lambda_sweep":True,"value_objective":"MSE_TO_TRUE_TERMINAL_Z_ONLY","checkpoint_frequency":2000,"early_stop_rule":"After update 4000, stop after two consecutive checkpoints below .85; same rule both arms","max_updates":32000};write_json(a.output/"retry_intervention.json",intervention);print(json.dumps({"safe":safe,"crossings":crossings,"cause":cause,"intervention":intervention},indent=2));return forensic

def split_re(rows,name):
 out={part:[] for part in ("train","validation","test")}
 for x in rows:
  n=split_code(name+str(x.get("position_hash",x["state"])));out["train" if n<85 else "validation" if n<95 else "test"].append(x)
 return out

def preservation_training_items(parent):
 rl=list(read_d_rl_jsonl(Path("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl")));re=list(iter_reanalysis_jsonl(Path("data/d_reanalysis/lot19_diverse_20k_g2_mcts.jsonl")));args=SimpleNamespace(lot22=Path("data/experiments/lot22_policy_objective"),seed=20262323);items,_,_,_=reconstruct_d_rank(args,rl,re);train=[x for x in items if x["split"]=="train"];attach_pair_types(train,parent);return train

def retry_training_data(arm):
 new=read_arm(LOT34/("control_data" if arm=="CONTROL" else "pool_data"));large,old=load_historical();holder=SimpleNamespace(output=LOT34);g2re,g3re=load_reanalysis_rows(holder);return split_rl(new),split_rl(large),split_rl(old),split_re(g2re,"g2"),split_re(g3re,"g3")

def train(a,arm):
 intervention=json.load((a.output/"retry_intervention.json").open());
 if not intervention["required"]:raise RuntimeError("forensics selected checkpoint-only retry; new training forbidden")
 seed=20263411 if arm=="CONTROL" else 20263412;randomizer=random.Random(seed);torch.manual_seed(seed);model=make_initial_model();parent=make_initial_model();newsp,largesp,oldsp,g2sp,g3sp=retry_training_data(arm);rank=preservation_training_items(parent);raw=[json.loads(x) for x in BATTERY.open() if x.strip()];gate_items=attach_strategic(raw,load_srn_checkpoint(G3P).model);builder=SongoGraphBuilder();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-4);out=a.output/"checkpoints"/arm.lower();out.mkdir(parents=True,exist_ok=True);observed=Counter();history=[];below=0;rank_order=list(range(len(rank)));randomizer.shuffle(rank_order)
 validation=[("RL",x) for x in newsp["validation"][:2048]]+[("RL",x) for x in largesp["validation"][:1024]]+[("RL",x) for x in oldsp["validation"][:512]]+[("REANALYSIS",x) for x in g2sp["validation"][:256]]+[("REANALYSIS",x) for x in g3sp["validation"][:256]];new_buckets={s:[x for x in newsp["train"] if x.metadata["source_type"]==s] for s in ("G2_G2","G3_G3","CROSS_PLAY")} if arm=="POOL" else {}
 for step in range(1,32001):
  rows=[]
  if arm=="CONTROL":rows += [("RL",x) for x in sample_rows(randomizer,newsp["train"],179)];observed["NEW_GENERATION"]+=179
  else:
   for source,count in (("G2_G2",18),("G3_G3",18),("CROSS_PLAY",143)):rows += [("RL",x) for x in sample_rows(randomizer,new_buckets[source],count)]
   observed["NEW_GENERATION"]+=179
  rows += [("RL",x) for x in sample_rows(randomizer,largesp["train"],26)]+[("RL",x) for x in sample_rows(randomizer,oldsp["train"],25)];observed["HISTORICAL_RL"]+=51;rows += [("REANALYSIS",x) for x in sample_rows(randomizer,g2sp["train"],13)]+[("REANALYSIS",x) for x in sample_rows(randomizer,g3sp["train"],13)];observed["AUTONOMOUS_REANALYSIS"]+=26;randomizer.shuffle(rows)
  graph,mask,target,value,vmask=batch_tensors(rows,builder);model.train();logits,pred=model(graph);policy_loss=-(target*torch.log_softmax(mask_policy_logits(logits,mask),-1)).sum(-1).mean();value_loss=(pred.reshape(-1)[vmask]-value[vmask]).square().mean();strategic=[rank[rank_order[(step*32+i)%len(rank_order)]] for i in range(32)];_,slogits,corr,pres=cp_batch(model,strategic);corr_loss=correction_loss(slogits,corr);pres_loss=preservation_loss(slogits,pres,rho=.5);loss=policy_loss+value_loss+.1*corr_loss+pres_loss;optimizer.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));optimizer.step()
  if step%2000==0:
   validation_metrics=evaluate_model(model,validation,builder);strategic_metrics=cp_metrics(model,gate_items);row={"step":step,**validation_metrics,**strategic_metrics,"strategic_regression":1-strategic_metrics["preservation_rate"],"train_policy_loss":float(policy_loss),"train_value_loss":float(value_loss),"train_correction_loss":float(corr_loss),"train_preservation_loss":float(pres_loss),"gradient_norm":grad};history.append(row);payload=checkpoint_payload(model,optimizer,step,seed,row,arm);payload["lineage"].update({"lot":"34R","intervention":"CORRECT_AND_PRESERVE","lambda_correction":.1,"lambda_preservation":1.,"rho":.5});torch.save(payload,out/f"step-{step:05d}.pt");print(f"[lot34r] {arm} step {step}/32000 preservation={strategic_metrics['preservation_rate']:.4f} correction={strategic_metrics['correction_rate']:.4f} CE={validation_metrics['policy_ce']:.5f}",flush=True);below=below+1 if strategic_metrics["preservation_rate"]<GATE else 0
   if step>=4000 and below>=2:break
 report={"arm":arm,"updates":history[-1]["step"],"max_updates":32000,"stopped_early":history[-1]["step"]<32000,"early_stop_rule":intervention["early_stop_rule"],"optimizer":"AdamW","learning_rate":.0003,"weight_decay":1e-4,"gradient_clip":1.,"batch_size":256,"seed":seed,"initialization":{"policy":"G3_STRATEGIC","value":"V28_A","identical_contract":True},"intervention":intervention,"observed_ratio":{k:observed[k]/sum(observed.values()) for k in observed},"history":history,"teacher_labels":False,"minimax_labels":False,"value_true_terminal_z_only":True,"truncated_value_excluded":True};write_json(a.output/("training_report_control.json" if arm=="CONTROL" else "training_report_pool.json"),report);return report

def select(a):
 policies={};values={};matrix={};gate={}
 for arm in ("CONTROL","POOL"):
  report=json.load((a.output/("training_report_control.json" if arm=="CONTROL" else "training_report_pool.json")).open());eligible=[x for x in report["history"] if x["preservation_rate"]>=GATE and x["correction_rate"]>0];ranked=sorted(eligible,key=lambda x:(-x["strategic_utility"],x["policy_ce"]))[:2];vrows=sorted(report["history"],key=lambda x:x["value_mse"])[:2];policies[arm]=[{"arm":arm,"step":x["step"],"path":str(a.output/"checkpoints"/arm.lower()/f"step-{x['step']:05d}.pt"),"policy_ce":x["policy_ce"],"correction_rate":x["correction_rate"],"preservation_rate":x["preservation_rate"],"strategic_utility":x["strategic_utility"]} for x in ranked];values[arm]=[{"arm":arm,"step":x["step"],"path":str(a.output/"checkpoints"/arm.lower()/f"step-{x['step']:05d}.pt"),"value_mse":x["value_mse"],"value_sign_accuracy":x["value_sign_accuracy"]} for x in vrows];matrix[arm]=[{"policy":p,"value":v,"combination_id":f"{arm}_P{pi+1}V{vi+1}"} for pi,p in enumerate(policies[arm]) for vi,v in enumerate(values[arm])];gate[arm]={"pass":bool(policies[arm]),"best_preservation":max((x["preservation_rate"] for x in report["history"]),default=0),"nontrivial_policy_learning":any(x["correction_rate"]>0 for x in eligible)}
 write_json(a.output/"policy_candidates.json",{"max_per_arm":2,"candidates":policies});write_json(a.output/"value_candidates.json",{"max_per_arm":2,"candidates":values});write_json(a.output/"component_matrix.json",{"max_combinations_per_arm":4,"matrix":matrix});write_json(a.output/"strategic_gate.json",{"threshold":GATE,"arms":gate});write_json(a.output/"main_arena_authorization.json",{"MAIN_ARENAS_ALLOWED":"YES" if all(x["pass"] for x in gate.values()) else "NO","condition":"both arms have a non-trivial Policy with preservation >= .85","arms":gate});print(json.dumps(gate,indent=2));return gate

def finalize(a):
 base=json.load((a.output/"report.json").open());identity=json.load((a.output/"lot34_dataset_identity.json").open());forensic=json.load((a.output/"checkpoint_forensics.json").open());cause=json.load((a.output/"failure_cause.json").open());intervention=json.load((a.output/"retry_intervention.json").open());gate=json.load((a.output/"strategic_gate.json").open());finalists=json.load((a.output/"finalist_selection.json").open());authorization=json.load((a.output/"main_arena_authorization.json").open());verdict=base["verdict"]
 verdict.update({"LOT34_DATA_REUSED_EXACTLY":identity["LOT34_DATA_REUSED_EXACTLY"],"NEW_GENERATION_PERFORMED":"NO","CHECKPOINT_FORENSICS_VALID":"YES" if forensic["valid"] else "NO","SAFE_CHECKPOINT_EXISTS_CONTROL":forensic["SAFE_CHECKPOINT_EXISTS_CONTROL"],"SAFE_CHECKPOINT_EXISTS_POOL":forensic["SAFE_CHECKPOINT_EXISTS_POOL"],"PRIMARY_LOT34_FAILURE_CAUSE":cause["PRIMARY_LOT34_FAILURE_CAUSE"],"RETRY_INTERVENTION_REQUIRED":"YES" if intervention["required"] else "NO","RETRY_INTERVENTION":intervention["name"],"CONTROL_STRATEGIC_GATE":"PASS" if gate["arms"]["CONTROL"]["pass"] else "FAIL","POOL_STRATEGIC_GATE":"PASS" if gate["arms"]["POOL"]["pass"] else "FAIL","CONTROL_BEST_PRESERVATION":gate["arms"]["CONTROL"]["best_preservation"],"POOL_BEST_PRESERVATION":gate["arms"]["POOL"]["best_preservation"],"NONTRIVIAL_POLICY_LEARNING_CONTROL":"YES" if gate["arms"]["CONTROL"]["nontrivial_policy_learning"] else "NO","NONTRIVIAL_POLICY_LEARNING_POOL":"YES" if gate["arms"]["POOL"]["nontrivial_policy_learning"] else "NO","MAIN_ARENAS_ALLOWED":authorization["MAIN_ARENAS_ALLOWED"],"POOL_G4R_BEATS_CONTROL_G4R":verdict.pop("POOL_G4_BEATS_CONTROL_G4"),"POOL_G4R_ROBUST_VS_G2":verdict.pop("POOL_G4_ROBUST_VS_G2"),"POOL_G4R_ROBUST_VS_G3":verdict.pop("POOL_G4_ROBUST_VS_G3")})
 result={**base,"lot":"34R","retry_of":34,"status":"COMPLETE","dataset_identity":identity,"checkpoint_forensics":{"safe_control":forensic["SAFE_CHECKPOINT_EXISTS_CONTROL"],"safe_pool":forensic["SAFE_CHECKPOINT_EXISTS_POOL"],"cause":cause["PRIMARY_LOT34_FAILURE_CAUSE"]},"retry_intervention":intervention,"finalists":finalists,"verdict":verdict};write_json(a.output/"report.json",result);print(json.dumps(verdict,indent=2));return result

def main():
 a=cli()
 if a.stage=="audit":audit(a)
 elif a.stage=="forensics":forensics(a)
 elif a.stage=="train-control":train(a,"CONTROL")
 elif a.stage=="train-pool":train(a,"POOL")
 elif a.stage=="select":select(a)
 elif a.stage=="finalize":finalize(a)
if __name__=="__main__":main()
