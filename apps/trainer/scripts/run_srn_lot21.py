#!/usr/bin/env python3
"""Lot 21 : recalibration Policy et régularisation vers la Policy parent G2."""
from __future__ import annotations

import argparse,csv,json,math,random,statistics,time
from dataclasses import asdict
from pathlib import Path

import torch

from songo_ai.dataset import iter_reanalysis_jsonl,read_d_rl_jsonl
from songo_ai.evaluation import ArenaConfig,SRNMCTSAgent,game_result_to_dict,generate_unique_deterministic_openings,legal_ranking,opening_to_dict,policy_entropy,run_paired_arena,summarize_arena
from songo_ai.model import SRNTrainingConfig,load_srn_checkpoint,mask_policy_logits,policy_probabilities
from songo_ai.model.dual_source_training import bounded_confidence_weights,bounded_regret_weights,parent_policy_kl,regularized_dual_source_loss
from run_srn_lot12 import fixed_batch_outputs,sha256,write_json
from run_srn_lot20 import G2_SHA,RE_SHA,RL_SHA,collate,dist,evaluate_value,load_regrets,per_example_ce,pos_hash,save_checkpoint,split_data


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d-rl",type=Path,default=Path("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl"))
    p.add_argument("--d-re",type=Path,default=Path("data/d_reanalysis/lot19_diverse_20k_g2_mcts.jsonl"))
    p.add_argument("--g2",type=Path,default=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt"))
    p.add_argument("--split-reference",type=Path,default=Path("data/experiments/lot14_g3_seed_20261402/training/best_validation_checkpoint.pt"))
    p.add_argument("--lot20",type=Path,default=Path("data/experiments/lot20_dual_source_policy"))
    p.add_argument("--output",type=Path,default=Path("data/experiments/lot21_policy_recalibration"))
    p.add_argument("--seed",type=int,default=20262020)
    p.add_argument("--arena-seed",type=int,default=20262121)
    p.add_argument("--bootstrap",type=int,default=20000)
    p.add_argument("--skip-arena",action="store_true")
    return p.parse_args()


def policies(model,examples,batch_size=512):
    out=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(examples),batch_size):
            batch=examples[start:start+batch_size];b=collate(batch);logits,_=model(b.graph);out.extend(policy_probabilities(logits,b.legal_mask).tolist())
    return out


def drift_metrics(parent,candidate,examples):
    pp=policies(parent,examples);cp=policies(candidate,examples);js=[];kl=[];arg=[];top2=[];entropy_delta=[];target_delta=[];parent_delta=[]
    for e,p,q in zip(examples,pp,cp):
        legal=[i for i,x in enumerate(e.legal_mask) if x];m=[(p[i]+q[i])/2 for i in range(7)]
        klpq=sum(p[i]*math.log(p[i]/q[i]) for i in legal if p[i]>0 and q[i]>0)
        jsp=sum(.5*p[i]*math.log(p[i]/m[i])+.5*q[i]*math.log(q[i]/m[i]) for i in legal if m[i]>0)
        pr=legal_ranking(p,e.legal_mask);cr=legal_ranking(q,e.legal_mask);tr=legal_ranking(e.policy_target,e.legal_mask)
        js.append(jsp);kl.append(klpq);arg.append(pr[0]!=cr[0]);top2.append(set(pr[:2])!=set(cr[:2]));entropy_delta.append(policy_entropy(q)-policy_entropy(p));target_delta.append(q[tr[0]]-p[tr[0]]);parent_delta.append(q[pr[0]]-p[pr[0]])
    return {"positions":len(examples),"js":dist(js),"kl_parent_candidate":dist(kl),"argmax_disagreement":statistics.fmean(arg),"top2_disagreement":statistics.fmean(top2),"entropy_delta":dist(entropy_delta),"delta_p_new_target":dist(target_delta),"delta_p_parent_top1":dist(parent_delta)}


def regret_rows(cache,examples,val_ids):
    rows=[]
    for line in cache.open():
        row=json.loads(line);example=examples[int(row["index"])]
        if example.metadata["game_id"] in val_ids:rows.append((example,row))
    return rows


def regret_metrics(model,rows):
    probs=policies(model,[e for e,_ in rows]);reg=[]
    for p,(e,row) in zip(probs,rows):reg.append(float(row["regrets"][legal_ranking(p,e.legal_mask)[0]]))
    ordered=sorted(reg,reverse=True);k=max(1,math.ceil(.05*len(ordered)))
    return {"positions":len(reg),"regret":dist(reg),"high_regret_errors_gt_0_25":sum(x>.25 for x in reg),"worst_5_percent_total_cost":sum(ordered[:k])}


def component_gradient_norm(model,loss):
    model.zero_grad(set_to_none=True);loss.backward(retain_graph=True);return math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None))


def representative_losses(model,parent,rl_examples,re_examples,regret_weights,confidence_weights,lambda_re,beta):
    rb=collate(rl_examples);eb=collate(re_examples);rl_logits,rl_v=model(rb.graph);re_logits,_=model(eb.graph)
    with torch.no_grad():parent_rl,_=parent(rb.graph);parent_re,_=parent(eb.graph)
    out=regularized_dual_source_loss(rl_logits,rl_v,rb.legal_mask,rb.policy_target,rb.value_target,regret_weights,re_logits,eb.legal_mask,eb.policy_target,confidence_weights,parent_rl,parent_re,lambda_re=lambda_re,beta=beta)
    components={"L_RL":out["policy_rl"],"lambda_RE_L_RE":lambda_re*out["policy_re"],"beta_L_parent":beta*out["parent"],"L_value":out["value_rl"]}
    return out,{name:{"loss_contribution":float(loss.detach()),"gradient_norm":component_gradient_norm(model,loss)} for name,loss in components.items()}


def beta_calibration(args,config,parent,rl_train,re_train,rw_map,cw_map):
    model=load_srn_checkpoint(args.g2).model;opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay);rng=random.Random(config.seed+21)
    for _ in range(20):
        rs=[rl_train[rng.randrange(len(rl_train))] for _ in range(128)];es=[re_train[rng.randrange(len(re_train))] for _ in range(128)];rb=collate(rs);eb=collate(es);rl_logits,rl_v=model(rb.graph);re_logits,_=model(eb.graph)
        with torch.no_grad():pr,_=parent.model(rb.graph);pe,_=parent.model(eb.graph)
        loss=regularized_dual_source_loss(rl_logits,rl_v,rb.legal_mask,rb.policy_target,rb.value_target,torch.tensor([rw_map[pos_hash(e.state)] for e in rs]),re_logits,eb.legal_mask,eb.policy_target,torch.tensor([cw_map[e.source_position_metadata["position_hash"]] for e in es]),pr,pe,lambda_re=.25,beta=0)
        opt.zero_grad();loss["total"].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm);opt.step()
    rs=rl_train[:128];es=re_train[:128];rw=torch.tensor([rw_map[pos_hash(e.state)] for e in rs]);cw=torch.tensor([cw_map[e.source_position_metadata["position_hash"]] for e in es]);results={}
    for beta in (.05,.10,.25):
        out,grads=representative_losses(model,parent.model,rs,es,rw,cw,.25,beta);den=max(grads["L_RL"]["gradient_norm"]+grads["lambda_RE_L_RE"]["gradient_norm"],1e-12);results[str(beta)]={"losses":{k:float(v.detach()) for k,v in out.items()},"gradient_contributions":grads,"parent_to_policy_gradient_ratio":grads["beta_L_parent"]["gradient_norm"]/den}
    low=min((.05,.10,.25),key=lambda b:abs(results[str(b)]["parent_to_policy_gradient_ratio"]-.05));remaining=[b for b in (.05,.10,.25) if b>low];medium=min(remaining or [.25],key=lambda b:abs(results[str(b)]["parent_to_policy_gradient_ratio"]-.125))
    return low,medium,{"method":"20-step lambda_RE=0.25 probe; choose distinct beta values closest to parent/policy gradient ratios 0.05 (low) and 0.125 (medium)","candidates":results,"selected_low":low,"selected_medium":medium}


def save_candidate(path,model,opt,parent_payload,config,epoch,history,lineage,best_epoch):
    save_checkpoint(path,model,opt,parent_payload,config,epoch,history,lineage,best_epoch)


def train_candidate(args,name,lambda_re,beta,config,parent,rl_train,rl_val,re_train,re_val,rw_map,cw_map):
    model=load_srn_checkpoint(args.g2).model;opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay);out=args.output/"checkpoints";out.mkdir(parents=True,exist_ok=True);best=out/f"{name.lower().replace('-','_')}_best.pt";history=[];steps=math.ceil(len(rl_train)/config.batch_size);global_step=0;best_epoch=0;without=0
    def validation(epoch):
        rl=per_example_ce(model,rl_val);re=per_example_ce(model,re_val);value=evaluate_value(model,rl_val);dr_rl=drift_metrics(parent.model,model,rl_val);dr_re=drift_metrics(parent.model,model,re_val)
        if epoch==0:validation.base=(rl["ce"],re["ce"],value["mse"])
        score=rl["ce"]/validation.base[0]+re["ce"]/validation.base[1]+value["mse"]/validation.base[2]+.5*(dr_rl["kl_parent_candidate"]["mean"]+dr_re["kl_parent_candidate"]["mean"])
        return {"epoch":epoch,"global_step":global_step,"policy_rl_ce":rl["ce"],"policy_rl_top1":rl["top1"],"policy_re_ce":re["ce"],"policy_re_top1":re["top1"],"value_mse":value["mse"],"parent_kl_rl":dr_rl["kl_parent_candidate"]["mean"],"parent_kl_re":dr_re["kl_parent_candidate"]["mean"],"selection_score":score}
    history.append(validation(0));best_score=history[0]["selection_score"];lineage={"candidate":name,"parent":"G2-best","G2_sha256":G2_SHA,"D_RL_sha256":RL_SHA,"D_REANALYSIS_sha256":RE_SHA,"lambda_RE":lambda_re,"beta":beta,"selection_rule":"normalized RL CE + normalized RE CE + normalized Value MSE + 0.5*(parent KL RL + parent KL RE)"};save_candidate(best,model,opt,parent.payload,config,0,history,lineage,0)
    for epoch in range(1,config.epochs+1):
        rng=random.Random(config.seed+epoch);ri=list(range(len(rl_train)));ei=list(range(len(re_train)));rng.shuffle(ri);rng.shuffle(ei);model.train()
        for step in range(steps):
            half=config.batch_size//2;rs=[rl_train[ri[(step*half+i)%len(ri)]] for i in range(half)];es=[re_train[ei[(step*half+i)%len(ei)]] for i in range(half)];rb=collate(rs);eb=collate(es);rl_logits,rl_v=model(rb.graph);re_logits,_=model(eb.graph)
            with torch.no_grad():pr,_=parent.model(rb.graph);pe,_=parent.model(eb.graph)
            loss=regularized_dual_source_loss(rl_logits,rl_v,rb.legal_mask,rb.policy_target,rb.value_target,torch.tensor([rw_map[pos_hash(e.state)] for e in rs]),re_logits,eb.legal_mask,eb.policy_target,torch.tensor([cw_map[e.source_position_metadata["position_hash"]] for e in es]),pr,pe,lambda_re=lambda_re,beta=beta)
            opt.zero_grad();loss["total"].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm);opt.step();global_step+=1
        row=validation(epoch);history.append(row);improved=row["selection_score"]<best_score-config.early_stopping_min_delta
        if improved:best_score=row["selection_score"];best_epoch=epoch;without=0;save_candidate(best,model,opt,parent.payload,config,epoch,history,lineage,best_epoch)
        else:without+=1
        print(f"[lot21] {name} epoch {epoch}: score={row['selection_score']:.5f}",flush=True)
        if without>=config.early_stopping_patience:break
    return best,history,best_epoch


def arena(name,model,parent,openings,config,budget):
    games=run_paired_arena(SRNMCTSAgent(name,model,budget,c_puct=1.5),SRNMCTSAgent("G2",parent,budget,c_puct=1.5),openings,config=config)
    return {"summary":asdict(summarize_arena(games,config=config)),"games":[game_result_to_dict(x) for x in games]}


def main():
    args=parse_args();started=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True);(args.output/"checkpoints").mkdir(exist_ok=True)
    if sha256(args.g2)!=G2_SHA or sha256(args.d_rl)!=RL_SHA or sha256(args.d_re)!=RE_SHA:raise RuntimeError("immutable input hash mismatch")
    parent=load_srn_checkpoint(args.g2);reference=load_srn_checkpoint(args.split_reference);rl=list(read_d_rl_jsonl(args.d_rl));re=list(iter_reanalysis_jsonl(args.d_re));rl_train,rl_val,re_train,re_val=split_data(rl,re,reference,args.seed);config_data=dict(reference.payload["training_config"]);config_data.update({"seed":args.seed,"device":"cpu"});config=SRNTrainingConfig(**config_data)
    _,fixed,g2p,g2v=fixed_batch_outputs(parent.model,rl);init={}
    for name in ("C21-A","C21-B","C21-C","C21-D"):
        clone=load_srn_checkpoint(args.g2).model
        with torch.no_grad():p,v=clone(fixed.graph)
        init[name]={"policy_exact":torch.equal(g2p,p),"value_exact":torch.equal(g2v,v)}
    if not all(all(x.values()) for x in init.values()):raise RuntimeError("epoch0 differs from G2")
    regret_map,_=load_regrets(Path("data/experiments/lot17_policy_regret/regret_search_cache.jsonl"),rl);known=sorted(regret_map.values());q95=known[int(.95*(len(known)-1))] or 1.;rv=bounded_regret_weights([regret_map.get(pos_hash(e.state)) for e in rl],alpha=1,q95=q95);rw_map={pos_hash(e.state):w for e,w in zip(rl,rv)};cv=bounded_confidence_weights([e.search_metadata for e in re],[sum(e.legal_mask) for e in re]);cw_map={e.source_position_metadata["position_hash"]:w for e,w in zip(re,cv)}
    low,medium,beta_report=beta_calibration(args,config,parent,rl_train,re_train,rw_map,cw_map);write_json(args.output/"beta_calibration.json",beta_report)
    specs={"C21-A":(.10,0.),"C21-B":(.25,0.),"C21-C":(.25,low),"C21-D":(.25,medium)};models={"G2":parent.model};training={}
    for name,(lambda_re,beta) in specs.items():
        path,history,best_epoch=train_candidate(args,name,lambda_re,beta,config,parent,rl_train,rl_val,re_train,re_val,rw_map,cw_map);models[name]=load_srn_checkpoint(path).model;training[name]={"best_epoch":best_epoch,"history":history,"checkpoint":str(path)}
        with (args.output/f"training_metrics_{name.lower().replace('-','_')}.csv").open("w",newline="") as stream:w=csv.DictWriter(stream,fieldnames=history[0]);w.writeheader();w.writerows(history)
    offline={};drift={};value={};rows=regret_rows(Path("data/experiments/lot17_policy_regret/regret_search_cache.jsonl"),rl,set(reference.payload["validation_game_ids"]));tails={}
    lot17_examples=[e for e,_ in rows]
    for name,model in models.items():
        offline[name]={"D_RL":{k:v for k,v in per_example_ce(model,rl_val).items() if k!="per_example_ce"},"D_REANALYSIS":{k:v for k,v in per_example_ce(model,re_val).items() if k!="per_example_ce"}};value[name]=evaluate_value(model,rl_val);tails[name]=regret_metrics(model,rows)
        if name!="G2":drift[name]={"D_RL":drift_metrics(parent.model,model,rl_val),"D_REANALYSIS":drift_metrics(parent.model,model,re_val),"LOT17":drift_metrics(parent.model,model,lot17_examples)}
    c20_path=args.lot20/"checkpoints/c20_dual_best.pt"
    if c20_path.exists():
        c20=load_srn_checkpoint(c20_path).model;drift["C20-DUAL"]={"D_RL":drift_metrics(parent.model,c20,rl_val),"D_REANALYSIS":drift_metrics(parent.model,c20,re_val),"LOT17":drift_metrics(parent.model,c20,lot17_examples)};tails["C20-DUAL"]=regret_metrics(c20,rows)
    write_json(args.output/"offline_metrics.json",offline);write_json(args.output/"policy_drift.json",drift);write_json(args.output/"regret_tail.json",tails);write_json(args.output/"value_metrics.json",value)
    gradients={}
    for name,(lambda_re,beta) in specs.items():
        rs=rl_train[:128];es=re_train[:128];_,gradients[name]=representative_losses(models[name],parent.model,rs,es,torch.tensor([rw_map[pos_hash(e.state)] for e in rs]),torch.tensor([cw_map[e.source_position_metadata["position_hash"]] for e in es]),lambda_re,beta)
    write_json(args.output/"gradient_contributions.json",gradients)
    g2tail=tails["G2"];gate={};comparison={}
    for name,(lambda_re,beta) in specs.items():
        learned=offline[name]["D_REANALYSIS"]["ce"]<offline["G2"]["D_REANALYSIS"]["ce"] and drift[name]["D_REANALYSIS"]["delta_p_new_target"]["mean"]>0
        preserved=value[name]["mse"]<=1.10*value["G2"]["mse"]
        tail_status="IMPROVED" if tails[name]["regret"]["p95"]<g2tail["regret"]["p95"] and tails[name]["worst_5_percent_total_cost"]<g2tail["worst_5_percent_total_cost"] else "WORSE" if tails[name]["regret"]["p95"]>1.05*g2tail["regret"]["p95"] else "SIMILAR"
        finite=math.isfinite(drift[name]["D_RL"]["js"]["mean"]+drift[name]["D_REANALYSIS"]["js"]["mean"]);passes=learned and preserved and finite and tail_status!="WORSE";over=drift[name]["D_REANALYSIS"]["js"]["mean"]<1e-6 and offline["G2"]["D_REANALYSIS"]["ce"]-offline[name]["D_REANALYSIS"]["ce"]<1e-4
        gate[name]={"passes":passes,"learns_reanalysis":learned,"value_preserved":preserved,"tail_status":tail_status,"finite_drift":finite};comparison[name]={"lambda_RE":lambda_re,"beta":beta,"offline":offline[name],"drift":drift[name],"regret_tail":tails[name],"value":value[name],"HIGH_REGRET_TAIL_VS_G2":tail_status,"OVER_REGULARIZED":"YES" if over else "NO"}
    short={};main_arena={};finalists=[]
    if not args.skip_arena:
        short_openings=generate_unique_deterministic_openings(count=32,seed=args.arena_seed,max_prefix_length=40);short_config=ArenaConfig(max_plies=400,repetition_limit=3,seed=args.arena_seed,bootstrap_samples=args.bootstrap)
        for name in specs:
            if gate[name]["passes"]:short[name]=arena(name,models[name],parent.model,short_openings,short_config,128);comparison[name]["short_arena_score"]=short[name]["summary"]["score_rate_a_terminal"];print(f"[lot21] short arena {name} done",flush=True)
        eligible=[n for n in short if short[n]["summary"]["score_rate_a_terminal"]>=.40];finalists=sorted(eligible,key=lambda n:(short[n]["summary"]["score_rate_a_terminal"],-tails[n]["regret"]["p95"],-drift[n]["D_REANALYSIS"]["js"]["mean"]),reverse=True)[:2]
        main_openings=generate_unique_deterministic_openings(count=128,seed=args.arena_seed+1,max_prefix_length=40);write_json(args.output/"arena_openings.json",{"short":[opening_to_dict(x) for x in short_openings],"main":[opening_to_dict(x) for x in main_openings]});main_config=ArenaConfig(max_plies=400,repetition_limit=3,seed=args.arena_seed+1,bootstrap_samples=args.bootstrap)
        for name in finalists:
            main_arena[name]={}
            for budget in (64,128):main_arena[name][str(budget)]=arena(name,models[name],parent.model,main_openings,main_config,budget);print(f"[lot21] main arena {name} MCTS{budget} done",flush=True)
            comparison[name]["main_arena"]={b:p["summary"] for b,p in main_arena[name].items()}
    write_json(args.output/"short_arena.json",short);write_json(args.output/"main_arena.json",main_arena)
    def robust(name):return bool(name in main_arena and all(main_arena[name][str(b)]["summary"]["score_rate_a_terminal"]>.5 and main_arena[name][str(b)]["summary"]["paired_bootstrap_ci"][0]>.5 for b in (64,128)))
    winner=next((n for n in finalists if robust(n)),None);lower_scores=[short[n]["summary"]["score_rate_a_terminal"] for n in ("C21-A","C21-B") if n in short];parent_scores=[short[n]["summary"]["score_rate_a_terminal"] for n in ("C21-C","C21-D") if n in short];lower_helps=bool(lower_scores and max(lower_scores)>.40);parent_helps=bool(parent_scores and max(parent_scores)>max(lower_scores or [0])+.05)
    healthy=[]
    for name in finalists:
        a=main_arena[name]["64"]["summary"]["score_rate_a_terminal"];b=main_arena[name]["128"]["summary"]["score_rate_a_terminal"];healthy.append(b>=a-.05)
    safe=winner is not None;all_fail=not winner and not lower_helps and not parent_helps
    verdict={"POLICY_RECALIBRATION_VALID":"YES","LOWER_REANALYSIS_WEIGHT_HELPS":"YES" if lower_helps else "NO" if short else "INCONCLUSIVE","PARENT_REGULARIZATION_HELPS":"YES" if parent_helps else "NO" if short else "INCONCLUSIVE","SAFE_POLICY_UPDATE_REGION_FOUND":"YES" if safe else "NO" if main_arena or short else "INCONCLUSIVE","HIGH_REGRET_TAIL_REDUCED":"YES" if any(gate[n]["tail_status"]=="IMPROVED" for n in specs) else "NO","VALUE_PRESERVED":"YES" if all(gate[n]["value_preserved"] for n in specs) else "NO","SEARCH_SCALING_HEALTHY":"YES" if healthy and all(healthy) else "NO" if healthy else "INCONCLUSIVE","SIMPLE_POLICY_RECALIBRATION_INSUFFICIENT":"YES" if not winner else "NO","G3_CANDIDATE":winner.replace("-","_") if winner else "NONE"}
    verdict["NEXT_ACTION"]="INDEPENDENT_G3_CONFIRMATION" if winner else "PARENT_REGULARIZATION_REFINEMENT" if parent_helps else "CONSERVATIVE_REANALYSIS_REFINEMENT" if lower_helps else "REVISIT_POLICY_OBJECTIVE"
    configuration={"hashes":{"G2":G2_SHA,"D_RL":RL_SHA,"D_REANALYSIS":RE_SHA},"splits":{"rl_train":len(rl_train),"rl_validation":len(rl_val),"re_train":len(re_train),"re_validation":len(re_val)},"seed":args.seed,"arena_seed":args.arena_seed,"training":asdict(config),"candidates":{n:{"lambda_RE":x,"beta":b} for n,(x,b) in specs.items()},"loss":"L_RL + lambda_RE*L_RE + L_value_D_RL + beta*0.5*(KL_parent_RL+KL_parent_RE)","parent_kl":"KL(P_G2 || P_candidate), legal actions only","initialization":init,"checkpoint_selection":"normalized RL CE + normalized RE CE + normalized Value MSE + 0.5*(parent KL RL + parent KL RE)"};write_json(args.output/"configuration.json",configuration)
    for name in comparison:comparison[name]["gate"]=gate[name];comparison[name]["short_arena"]=short.get(name,{}).get("summary")
    write_json(args.output/"candidate_comparison.json",comparison)
    report={"lot":21,"configuration":configuration,"beta_calibration":beta_report,"training":{n:{"best_epoch":x["best_epoch"],"checkpoint":x["checkpoint"]} for n,x in training.items()},"offline":offline,"policy_drift":drift,"regret_tail":tails,"value":value,"gradient_contributions":gradients,"offline_gate":gate,"short_arena":{n:x["summary"] for n,x in short.items()},"finalists":finalists,"main_arena":{n:{b:x["summary"] for b,x in p.items()} for n,p in main_arena.items()},"verdict":verdict,"scientific_question":"YES" if safe else "NO" if short else "INCONCLUSIVE","minimax_benchmark_executed":False,"elapsed_s":time.perf_counter()-started};write_json(args.output/"report.json",report);print(json.dumps({"report":str(args.output/"report.json"),"verdict":verdict,"elapsed_s":report["elapsed_s"]},indent=2),flush=True)


if __name__=="__main__":main()
