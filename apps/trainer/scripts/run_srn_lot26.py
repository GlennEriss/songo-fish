#!/usr/bin/env python3
"""Lot 26 : entraînement G3 source-aware sur D_SCALE_V1."""
from __future__ import annotations
import argparse,csv,json,math,random,time
from pathlib import Path
import torch

from songo_ai.dataset import RawSongoState,RLTrainingExample,read_d_rl_jsonl
from songo_ai.evaluation import reanalysis_position_hash
from songo_ai.model import SRNTrainingConfig,load_srn_checkpoint,mask_policy_logits,policy_probabilities,split_bucket
from songo_ai.model.correct_preserve import classify_pairs,correction_loss,preservation_loss
from songo_ai.model.strategic_ranking import build_legal_pairs
from run_srn_lot12 import fixed_batch_outputs,sha256,write_json
from run_srn_lot20 import collate,save_checkpoint

G2_SHA="eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753";MANIFEST_SHA="76015ccd6cd0e49fd506eb7c4d0c795e23e93803e2654401efabe55f0898b1ee"

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--g2",type=Path,default=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt"));p.add_argument("--root",type=Path,default=Path("data/d_scale_v1"));p.add_argument("--manifest",type=Path,default=Path("data/experiments/lot25_scale/corpus_manifest.json"));p.add_argument("--output",type=Path,default=Path("data/experiments/lot26_g3_scale"));p.add_argument("--epochs",type=int,default=8);p.add_argument("--patience",type=int,default=3);p.add_argument("--seed",type=int,default=20262626);return p.parse_args()

def load_data(a):
 sp=[]
 for p in sorted((a.root/"d_selfplay_large").glob("part-*.jsonl")):sp.extend(read_d_rl_jsonl(p))
 re=[]
 for p in sorted((a.root/"d_reanalysis_large").glob("part-*.jsonl")):
  for line in p.open():
   r=json.loads(line);s=r["state"];re.append(RLTrainingExample(RawSongoState(tuple(s["board"]),s["player_to_move"]),tuple(r["legal_mask"]),tuple(r["policy_target"]),None,tuple(r["visit_counts"]),{"game_id":"re:"+r["position_hash"],"source_dataset":"REANALYSIS","position_hash":r["position_hash"]}))
 sr=[json.loads(x) for x in (a.root/"d_strategic_sample/qdiag256.jsonl").open() if x.strip()]
 return sp,re,sr

def split(items,identity,seed):
 tr=[];va=[]
 for x in items:(va if split_bucket(identity(x),seed=seed)==0 else tr).append(x)
 return tr,va

def eval_source(model,items,limit=12000):
 items=items[:limit];ce=correct=js=value_sq=value_sign=labeled=0
 with torch.no_grad():
  for start in range(0,len(items),512):
   b=collate(items[start:start+512]);logits,v=model(b.graph);p=policy_probabilities(logits,b.legal_mask);lp=torch.log_softmax(mask_policy_logits(logits,b.legal_mask),-1);ce+=float((-(b.policy_target*lp).sum(-1)).sum());correct+=int((p.argmax(-1)==b.policy_target.argmax(-1)).sum());m=(p+b.policy_target)/2;js+=float((.5*(p*(torch.log(p.clamp_min(1e-12))-torch.log(m.clamp_min(1e-12)))).sum(-1)+.5*(b.policy_target*(torch.log(b.policy_target.clamp_min(1e-12))-torch.log(m.clamp_min(1e-12)))).sum(-1)).sum());mask=b.value_mask
   if mask.any():d=v.reshape(-1)[mask]-b.value_target[mask];value_sq+=float(d.square().sum());value_sign+=int((torch.sign(v.reshape(-1)[mask])==torch.sign(b.value_target[mask])).sum());labeled+=int(mask.sum())
 return {"positions":len(items),"policy_ce":ce/len(items),"top1_agreement":correct/len(items),"js":js/len(items),"value_mse":value_sq/labeled if labeled else None,"value_sign_accuracy":value_sign/labeled if labeled else None,"labeled_values":labeled}

def attach_strategic(rows,parent):
 examples=[]
 for r in rows:
  s=r["state"];legal=r["legal_mask"];q=r["q_values"];pairs=build_legal_pairs(q,legal,epsilon=.02,scale=1.0);examples.append({"state":RawSongoState(tuple(s["board"]),s["player_to_move"]),"legal":legal,"pairs":pairs,"hash":r["position_hash"]})
 with torch.no_grad():
  for start in range(0,len(examples),256):
   chunk=examples[start:start+256];dummy=[RLTrainingExample(x["state"],tuple(x["legal"]),tuple(1/sum(x["legal"]) if ok else 0 for ok in x["legal"]),None,tuple(1 if ok else 0 for ok in x["legal"]),{"game_id":"st:"+x["hash"]}) for x in chunk];b=collate(dummy);logits,_=parent(b.graph)
   for i,x in enumerate(chunk):x["corr"],x["pres"]=classify_pairs(logits[i].tolist(),x["pairs"])
 return examples

def strategic_batch(model,items):
 dummy=[RLTrainingExample(x["state"],tuple(x["legal"]),tuple(1/sum(x["legal"]) if ok else 0 for ok in x["legal"]),None,tuple(1 if ok else 0 for ok in x["legal"]),{"game_id":"st:"+x["hash"]}) for x in items];b=collate(dummy);logits,_=model(b.graph);return logits,[x["corr"] for x in items],[x["pres"] for x in items]

def train(a,name,strategic,parent,sptr,retr,spv,rev,strat):
 model=load_srn_checkpoint(a.g2).model;cfg=SRNTrainingConfig(epochs=a.epochs,batch_size=256,learning_rate=.0003,weight_decay=1e-4,gradient_clip_norm=1.,validation_fraction=.1,early_stopping_patience=a.patience,early_stopping_min_delta=1e-4,seed=a.seed,device="cpu");opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay);steps=math.ceil(len(retr)/128);best=float("inf");wait=0;history=[];path=a.output/"checkpoints"/f"{name.lower().replace('-','_')}_best.pt"
 for epoch in range(a.epochs+1):
  if epoch:
   model.train();rng=random.Random(a.seed+epoch);si=list(range(len(sptr)));ri=list(range(len(retr)));qi=list(range(len(strat)));rng.shuffle(si);rng.shuffle(ri);rng.shuffle(qi)
   for step in range(steps):
    batch=[sptr[si[(step*128+i)%len(si)]] for i in range(128)]+[retr[ri[(step*128+i)%len(ri)]] for i in range(128)];b=collate(batch);logits,v=model(b.graph);loss=-(b.policy_target*torch.log_softmax(mask_policy_logits(logits,b.legal_mask),-1)).sum(-1).mean();mask=b.value_mask;loss=loss+(v.reshape(-1)[mask]-b.value_target[mask]).square().mean()
    if strategic:
     qs=[strat[qi[(step*32+i)%len(qi)]] for i in range(32)];ql,c,p=strategic_batch(model,qs);loss=loss+.1*correction_loss(ql,c)+preservation_loss(ql,p,rho=.5)
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  sm=eval_source(model,spv);rm=eval_source(model,rev);score=sm["policy_ce"]+rm["policy_ce"]+sm["value_mse"];row={"epoch":epoch,"global_step":epoch*steps,"selection_score":score,**{f"selfplay_{k}":v for k,v in sm.items()},**{f"reanalysis_{k}":v for k,v in rm.items()}};history.append(row);print(f"[lot26] {name} epoch {epoch}: selection={score:.5f}",flush=True)
  if score<best-1e-4:best=score;wait=0;save_checkpoint(path,model,opt,parent.payload,cfg,epoch,history,{"candidate":name,"parent":"G2-best","strategic":strategic,"source_ratio":"128 SELFPLAY : 128 REANALYSIS"},epoch)
  else:wait+=1
  if epoch and wait>=a.patience:break
 with (a.output/f"training_{'strategic' if strategic else 'scale'}.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
 return path,history

def main():
 a=args();a.output.mkdir(parents=True,exist_ok=True);(a.output/"checkpoints").mkdir(exist_ok=True);manifest=json.load(a.manifest.open());integrity=manifest["corpus_sha256_manifest"]==MANIFEST_SHA and sha256(a.g2)==G2_SHA
 if not integrity:raise RuntimeError("D_SCALE_V1/G2 immutable identity mismatch")
 sp,re,srows=load_data(a);sptr,spv=split(sp,lambda x:x.metadata["game_id"],a.seed);retr,rev=split(re,lambda x:x.metadata["position_hash"],a.seed);strat=attach_strategic(srows,load_srn_checkpoint(a.g2).model);strtr,strv=split(strat,lambda x:x["hash"],a.seed)
 parent=load_srn_checkpoint(a.g2);_,fixed,gp,gv=fixed_batch_outputs(parent.model,sp);initial={}
 for n in ("G3-SCALE","G3-STRATEGIC"):
  clone=load_srn_checkpoint(a.g2).model
  with torch.no_grad():p,v=clone(fixed.graph)
  initial[n]={"logits_exact":torch.equal(gp,p),"value_exact":torch.equal(gv,v),"parameters_exact":all(torch.equal(x,y) for x,y in zip(parent.model.state_dict().values(),clone.state_dict().values()))}
 paths={};hist={}
 for n,s in (("G3-SCALE",False),("G3-STRATEGIC",True)):paths[n],hist[n]=train(a,n,s,parent,sptr,retr,spv,rev,strtr)
 models={"G2":parent.model,**{n:load_srn_checkpoint(p).model for n,p in paths.items()}};offline={n:{"selfplay":eval_source(m,spv),"reanalysis":eval_source(m,rev)} for n,m in models.items()};write_json(a.output/"offline_evaluation.json",offline)
 verification={"valid":integrity,"manifest_sha256":MANIFEST_SHA,"G2_sha256":G2_SHA,"selfplay":{"train":len(sptr),"validation":len(spv),"game_leakage":False},"reanalysis":{"train":len(retr),"validation":len(rev),"physical_leakage":False},"strategic":{"train":len(strtr),"validation":len(strv)},"teacher_labels":False,"minimax_labels":False,"initialization":initial};write_json(a.output/"dataset_verification.json",verification);write_json(a.output/"sampling_configuration.json",{"batch":{"SELFPLAY":128,"REANALYSIS":128},"strategic_per_step":32,"value_source":"SELFPLAY only","seed":a.seed});write_json(a.output/"configuration.json",{"epochs":a.epochs,"patience":a.patience,"learning_rate":.0003,"batch_size":256,"calibration":{"historical_lr":.003,"result":"validation degradation; reduced 10x"},"architecture_changed":False,"engine_changed":False});write_json(a.output/"training_report.json",{"checkpoints":{n:str(p) for n,p in paths.items()},"offline":offline,"integrity":verification});print(json.dumps({"checkpoints":{n:str(p) for n,p in paths.items()},"offline":offline},indent=2))
if __name__=="__main__":main()
