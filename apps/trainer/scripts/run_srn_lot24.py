#!/usr/bin/env python3
"""Lot 24 : objectif asymétrique correct-and-preserve."""
from __future__ import annotations

import argparse,csv,hashlib,json,math,random,statistics,time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import torch

from songo_ai.dataset import iter_reanalysis_jsonl,read_d_rl_jsonl
from songo_ai.evaluation import ArenaConfig,generate_unique_deterministic_openings,opening_to_dict,policy_cross_entropy
from songo_ai.model import SRNTrainingConfig,load_srn_checkpoint,mask_policy_logits,policy_probabilities
from songo_ai.model.correct_preserve import classify_pairs,correct_preserve_metrics,correction_loss,preservation_loss
from run_srn_lot12 import fixed_batch_outputs,sha256,write_json
from run_srn_lot20 import G2_SHA,RE_SHA,RL_SHA,collate,dist,evaluate_value,per_example_ce,save_checkpoint,split_data
from run_srn_lot23 import arena,reconstruct_d_rank,strategic_eval


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--d-rl",type=Path,default=Path("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl"));p.add_argument("--d-re",type=Path,default=Path("data/d_reanalysis/lot19_diverse_20k_g2_mcts.jsonl"));p.add_argument("--g2",type=Path,default=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt"));p.add_argument("--split-reference",type=Path,default=Path("data/experiments/lot14_g3_seed_20261402/training/best_validation_checkpoint.pt"));p.add_argument("--lot22",type=Path,default=Path("data/experiments/lot22_policy_objective"));p.add_argument("--lot23",type=Path,default=Path("data/experiments/lot23_strategic_ranking"));p.add_argument("--output",type=Path,default=Path("data/experiments/lot24_correct_preserve"));p.add_argument("--seed",type=int,default=20262323);p.add_argument("--arena-seed",type=int,default=20262424);p.add_argument("--bootstrap",type=int,default=20000);p.add_argument("--skip-arena",action="store_true");return p.parse_args()


def attach_pair_types(items,parent):
    parent.eval()
    with torch.no_grad():
        for start in range(0,len(items),256):
            subset=items[start:start+256];b=collate([x["example"] for x in subset]);logits,_=parent(b.graph)
            for row,item in enumerate(subset):item["correction_pairs"],item["preservation_pairs"]=classify_pairs(logits[row].tolist(),item["pairs"])


def excluded_pair_statistics(qrows, stability_rows, *, seed, epsilon=.02):
    """Compte les paires volontairement absentes de D_RANK, par split immuable."""
    stable={row["position_hash"]:row for row in stability_rows};result={s:{"near_equivalent":0,"unstable_excluded":0} for s in ("train","validation","test")}
    for row in qrows:
        key=row["position_hash"];bucket=int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16],16)%20;split="test" if bucket<3 else "validation" if bucket<6 else "train";q=row["q_values"];legal=[a for a,x in enumerate(q) if x is not None];stable_row=stable.get(key)
        for a,b in combinations(legal,2):
            if abs(q[a]-q[b])<=epsilon:result[split]["near_equivalent"]+=1
            elif stable_row and (stable_row["q256"][a]-stable_row["q256"][b])*(stable_row["q512"][a]-stable_row["q512"][b])<=0:result[split]["unstable_excluded"]+=1
    return result


def cp_batch(model,items):
    b=collate([x["example"] for x in items]);logits,_=model(b.graph);return b,logits,[x["correction_pairs"] for x in items],[x["preservation_pairs"] for x in items]


def cp_eval(model,items):
    totals={"correction_pairs":0,"preservation_pairs":0,"correction_correct":0.,"preservation_correct":0.,"corrected_weight":0.,"damaged_weight":0.}
    with torch.no_grad():
        for start in range(0,len(items),256):
            subset=items[start:start+256];_,logits,corr,pres=cp_batch(model,subset);m=correct_preserve_metrics(logits,corr,pres);totals["correction_pairs"]+=m["correction_pairs"];totals["preservation_pairs"]+=m["preservation_pairs"];totals["correction_correct"]+=m["correction_rate"]*m["correction_pairs"] if m["correction_rate"] is not None else 0;totals["preservation_correct"]+=m["preservation_rate"]*m["preservation_pairs"] if m["preservation_rate"] is not None else 0;totals["corrected_weight"]+=m["corrected_weight"];totals["damaged_weight"]+=m["damaged_weight"]
    cr=totals["correction_correct"]/totals["correction_pairs"] if totals["correction_pairs"] else None;pr=totals["preservation_correct"]/totals["preservation_pairs"] if totals["preservation_pairs"] else None;damaged=totals["preservation_pairs"]-totals["preservation_correct"]
    return {"correction_pairs":totals["correction_pairs"],"preservation_pairs":totals["preservation_pairs"],"correction_rate":cr,"preservation_rate":pr,"damage_rate":1-pr if pr is not None else None,"correction_efficiency":totals["correction_correct"]/damaged if damaged else None,"strategic_utility":totals["corrected_weight"]-totals["damaged_weight"],"corrected_weight":totals["corrected_weight"],"damaged_weight":totals["damaged_weight"]}


def grad_norm(model,loss):
    model.zero_grad(set_to_none=True);loss.backward(retain_graph=True);return math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None))


def calibrate(args,parent,rl_train,rank_train,rank_val,config):
    results={}
    for rho in (.25,.5,.75):
        for lam in (.5,1.,2.):
            rng=random.Random(args.seed+24)
            model=load_srn_checkpoint(args.g2).model;opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
            for step in range(20):
                rs=[rl_train[rng.randrange(len(rl_train))] for _ in range(128)];qs=[rank_train[rng.randrange(len(rank_train))] for _ in range(128)];rb=collate(rs);rl,rv=model(rb.graph);base=-(rb.policy_target*torch.log_softmax(mask_policy_logits(rl,rb.legal_mask),-1)).sum(-1).mean()+(rv-rb.value_target).square().mean();_,ql,c,p=cp_batch(model,qs);loss=base+.1*correction_loss(ql,c)+lam*preservation_loss(ql,p,rho=rho);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm);opt.step()
            metrics=cp_eval(model,rank_val);qs=rank_train[:128];_,ql,c,p=cp_batch(model,qs);corr=.1*correction_loss(ql,c);pres=lam*preservation_loss(ql,p,rho=rho);results[f"rho={rho},lambda={lam}"]={"rho":rho,"lambda_pres":lam,"validation":metrics,"gradient_norm_correction":grad_norm(model,corr),"gradient_norm_preservation":grad_norm(model,pres),"loss_correction":float(corr.detach()),"loss_preservation":float(pres.detach())}
    eligible=[x for x in results.values() if x["validation"]["preservation_rate"]>=.9 and x["validation"]["correction_rate"]>0];chosen=max(eligible or results.values(),key=lambda x:(x["validation"]["preservation_rate"]>=.9,x["validation"]["strategic_utility"],x["validation"]["correction_rate"]));return chosen["rho"],chosen["lambda_pres"],{"grid":results,"selected":{"rho":chosen["rho"],"lambda_pres":chosen["lambda_pres"]},"rule":"preservation>=90%, correction>0, then maximum validation strategic utility; no test or arena"}


def save(path,model,opt,parent,config,epoch,history,lineage,best_epoch):save_checkpoint(path,model,opt,parent.payload,config,epoch,history,lineage,best_epoch)


def train(args,name,mode,parent,config,rl_train,rank_train,rank_val,rho,lambda_pres):
    model=load_srn_checkpoint(args.g2).model;opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay);best=args.output/"checkpoints"/f"{name.lower().replace('-','_')}_best.pt";history=[];steps=math.ceil(len(rl_train)/config.batch_size);global_step=0;without=0;best_score=-1e9;best_epoch=0;ce_items=[x for x in rank_train if x["distribution_target"]=="MCTS128"]
    def metrics(epoch):
        m=cp_eval(model,rank_val);score=(10 if m["preservation_rate"]>=.9 else 0)+m["preservation_rate"]+.5*m["correction_rate"]+.01*m["strategic_utility"]
        return {"epoch":epoch,"global_step":global_step,"selection_score":score,**m}
    history.append(metrics(0));best_score=history[0]["selection_score"];lineage={"candidate":name,"parent":"G2-best","mode":mode,"rho":rho,"lambda_pres":lambda_pres,"lambda_correction":.1,"pair_types":"fixed before training"};save(best,model,opt,parent,config,0,history,lineage,0)
    for epoch in range(1,config.epochs+1):
        rng=random.Random(config.seed+epoch);ro=list(range(len(rl_train)));qo=list(range(len(rank_train)));co=list(range(len(ce_items)));rng.shuffle(ro);rng.shuffle(qo);rng.shuffle(co);model.train()
        for step in range(steps):
            half=config.batch_size//2;rs=[rl_train[ro[(step*half+i)%len(ro)]] for i in range(half)];qs=[rank_train[qo[(step*half+i)%len(qo)]] for i in range(half)];rb=collate(rs);rl,rv=model(rb.graph);total=-(rb.policy_target*torch.log_softmax(mask_policy_logits(rl,rb.legal_mask),-1)).sum(-1).mean()+(rv-rb.value_target).square().mean();_,ql,c,p=cp_batch(model,qs);total=total+.1*correction_loss(ql,c)
            if mode in ("CP","CPH"):total=total+lambda_pres*preservation_loss(ql,p,rho=rho)
            if mode=="CPH" and ce_items:
                cs=[ce_items[co[(step*half+i)%len(co)]] for i in range(half)];cb=collate([x["example"] for x in cs]);cl,_=model(cb.graph);total=total+.25*(-(cb.policy_target*torch.log_softmax(mask_policy_logits(cl,cb.legal_mask),-1)).sum(-1).mean())
            opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm);opt.step();global_step+=1
        row=metrics(epoch);history.append(row);improved=row["selection_score"]>best_score+config.early_stopping_min_delta
        if improved:best_score=row["selection_score"];best_epoch=epoch;without=0;save(best,model,opt,parent,config,epoch,history,lineage,best_epoch)
        else:without+=1
        print(f"[lot24] {name} epoch {epoch}: score={row['selection_score']:.5f} pres={row['preservation_rate']:.3f} corr={row['correction_rate']:.3f}",flush=True)
        if without>=config.early_stopping_patience:break
    return best,history,best_epoch


def main():
    args=parse_args();started=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True);(args.output/"checkpoints").mkdir(exist_ok=True)
    if sha256(args.g2)!=G2_SHA or sha256(args.d_rl)!=RL_SHA or sha256(args.d_re)!=RE_SHA:raise RuntimeError("immutable input mismatch")
    parent=load_srn_checkpoint(args.g2);reference=load_srn_checkpoint(args.split_reference);rl=list(read_d_rl_jsonl(args.d_rl));re=list(iter_reanalysis_jsonl(args.d_re));rl_train,rl_val,_,_=split_data(rl,re,reference,20262020);items,epsilon_report,scale,stable=reconstruct_d_rank(args,rl,re);attach_pair_types(items,parent.model);rank_train=[x for x in items if x["split"]=="train"];rank_val=[x for x in items if x["split"]=="validation"];rank_test=[x for x in items if x["split"]=="test"]
    qrows=[json.loads(x) for x in (args.lot22/"qdiag_cache.jsonl").open()];stability_rows=json.load((args.lot22/"qdiag_stability.json").open())["rows"];excluded=excluded_pair_statistics(qrows,stability_rows,seed=args.seed)
    stats={};conflicts={}
    for split,subset in (("train",rank_train),("validation",rank_val),("test",rank_test)):
        corr=[p for x in subset for p in x["correction_pairs"]];pres=[p for x in subset for p in x["preservation_pairs"]];stats[split]={"positions":len(subset),"correction_pairs":len(corr),"preservation_pairs":len(pres),**excluded[split],"correction_gap":distribution([p[3] for p in corr]),"preservation_gap":distribution([p[3] for p in pres]),"parent_margin":distribution([p[4] for p in pres])};both=0
        for x in subset:
            ca={a for p in x["correction_pairs"] for a in p[:2]};pa={a for p in x["preservation_pairs"] for a in p[:2]};both+=len(ca&pa)
        conflicts[split]={"shared_action_participations":both,"constraint_cycles":0,"reason":"Qdiag scalar order is acyclic"}
    write_json(args.output/"pair_classification.json",stats);write_json(args.output/"pair_conflict_statistics.json",conflicts)
    config_data=dict(reference.payload["training_config"]);config_data.update({"seed":args.seed,"device":"cpu"});config=SRNTrainingConfig(**config_data);rho,lambda_pres,calibration=calibrate(args,parent,rl_train,rank_train,rank_val,config);write_json(args.output/"rho_calibration.json",{"selected_rho":rho,"grid":calibration["grid"],"rule":calibration["rule"]});write_json(args.output/"lambda_pres_calibration.json",{"selected_lambda_pres":lambda_pres,"grid":calibration["grid"],"rule":calibration["rule"]})
    # Micro-overfit des deux termes, sans paire quasi-équivalente active.
    micro_model=load_srn_checkpoint(args.g2).model;opt=torch.optim.AdamW(micro_model.parameters(),lr=config.learning_rate);small=rank_train[:64];rsmall=rl_train[:64];rb_micro=collate(rsmall)
    def micro():_,logits,c,p=cp_batch(micro_model,small);return correction_loss(logits,c),preservation_loss(logits,p,rho=rho),correct_preserve_metrics(logits.detach(),c,p)
    bc,bp,bm=micro()
    for _ in range(20):
        c,p,_=micro();rl_logits,rv=micro_model(rb_micro.graph);base=-(rb_micro.policy_target*torch.log_softmax(mask_policy_logits(rl_logits,rb_micro.legal_mask),-1)).sum(-1).mean()+(rv-rb_micro.value_target).square().mean();loss=base+.1*c+lambda_pres*p;opt.zero_grad();loss.backward();opt.step()
    ac,ap,am=micro();micro_report={"correction_loss_before":float(bc.detach()),"correction_loss_after":float(ac.detach()),"preservation_loss_before":float(bp.detach()),"preservation_loss_after":float(ap.detach()),"correction_rate_before":bm["correction_rate"],"correction_rate_after":am["correction_rate"],"preservation_rate_after":am["preservation_rate"],"near_equivalent_gradient":"ZERO_BY_EXCLUSION","qdiag_value_target":"ABSENT"}
    _,fixed,gp,gv=fixed_batch_outputs(parent.model,rl);specs={"C24-CORR":"CORR","C24-CP":"CP","C24-CPH":"CPH"};models={"G2":parent.model};training={};initialization={}
    for name,mode in specs.items():
        clone=load_srn_checkpoint(args.g2).model
        with torch.no_grad():p,v=clone(fixed.graph)
        initialization[name]={"logits_exact":torch.equal(gp,p),"policy_exact":torch.equal(policy_probabilities(gp,fixed.legal_mask),policy_probabilities(p,fixed.legal_mask)),"value_exact":torch.equal(gv,v)};path,history,best_epoch=train(args,name,mode,parent,config,rl_train,rank_train,rank_val,rho,lambda_pres);models[name]=load_srn_checkpoint(path).model;training[name]={"best_epoch":best_epoch,"checkpoint":str(path),"history":history}
    with (args.output/"training_metrics.json").open("w") as s:json.dump(training,s,indent=2)
    references={"LOT23-RANK":load_srn_checkpoint(args.lot23/"checkpoints/c23_rank_best.pt").model,"LOT23-HYBRID":load_srn_checkpoint(args.lot23/"checkpoints/c23_hybrid_best.pt").model};all_models={**models,**references};validation_eval={n:{"cp":cp_eval(m,rank_val),"strategic":strategic_eval(m,parent.model,rank_val),"value":evaluate_value(m,rl_val)} for n,m in all_models.items()};test_eval={n:{"cp":cp_eval(m,rank_test),"strategic":strategic_eval(m,parent.model,rank_test),"value":evaluate_value(m,rl_val),"D_RL_policy":{k:v for k,v in per_example_ce(m,rl_val).items() if k!="per_example_ce"}} for n,m in all_models.items()};write_json(args.output/"offline_strategic_evaluation.json",validation_eval);write_json(args.output/"strict_test_evaluation.json",test_eval);write_json(args.output/"lot23_comparison.json",{n:test_eval[n] for n in ("G2","LOT23-RANK","LOT23-HYBRID","C24-CORR","C24-CP","C24-CPH")})
    gate={}
    for n in specs:
        v=validation_eval[n];t=test_eval[n];gate[n]={"validation_preservation":v["cp"]["preservation_rate"]>=.9,"validation_correction":v["cp"]["correction_rate"]>0,"test_preservation":t["cp"]["preservation_rate"]>=.9,"test_correction":t["cp"]["correction_rate"]>0,"net_gain":t["cp"]["strategic_utility"]>0,"swi_better":t["strategic"]["swi"]<test_eval["G2"]["strategic"]["swi"],"regret_not_worse":t["strategic"]["decision_regret"]["mean"]<=test_eval["G2"]["strategic"]["decision_regret"]["mean"],"value_preserved":t["value"]["mse"]<=1.10*test_eval["G2"]["value"]["mse"]};gate[n]["passes"]=all(gate[n].values())
    short={};main_arena={};finalists=[]
    if not args.skip_arena:
        so=generate_unique_deterministic_openings(count=32,seed=args.arena_seed,max_prefix_length=40);sc=ArenaConfig(max_plies=400,repetition_limit=3,seed=args.arena_seed,bootstrap_samples=args.bootstrap)
        for n in ("C24-CP","C24-CPH"):
            if gate[n]["passes"]:short[n]=arena(n,models[n],parent.model,so,sc,128);print(f"[lot24] short arena {n} done",flush=True)
        eligible=[n for n in short if short[n]["summary"]["score_rate_a_terminal"]>=.4];finalists=sorted(eligible,key=lambda n:(short[n]["summary"]["score_rate_a_terminal"],test_eval[n]["cp"]["strategic_utility"]),reverse=True)[:2];mo=generate_unique_deterministic_openings(count=128,seed=args.arena_seed+1,max_prefix_length=40);write_json(args.output/"arena_openings.json",{"short":[opening_to_dict(x) for x in so],"main":[opening_to_dict(x) for x in mo]});mc=ArenaConfig(max_plies=400,repetition_limit=3,seed=args.arena_seed+1,bootstrap_samples=args.bootstrap)
        for n in finalists:
            main_arena[n]={}
            for budget in (64,128):main_arena[n][str(budget)]=arena(n,models[n],parent.model,mo,mc,budget);print(f"[lot24] main arena {n} MCTS{budget} done",flush=True)
    write_json(args.output/"short_arena.json",short);write_json(args.output/"main_arena.json",main_arena);scaling={n:{"score64":p["64"]["summary"]["score_rate_a_terminal"],"score128":p["128"]["summary"]["score_rate_a_terminal"],"healthy":p["128"]["summary"]["score_rate_a_terminal"]>=p["64"]["summary"]["score_rate_a_terminal"]-.05} for n,p in main_arena.items()};write_json(args.output/"search_scaling.json",scaling)
    def robust(n):return n in main_arena and all(main_arena[n][str(b)]["summary"]["score_rate_a_terminal"]>.5 and main_arena[n][str(b)]["summary"]["paired_bootstrap_ci"][0]>.5 for b in (64,128))
    winner=next((n for n in finalists if robust(n)),None);safe=any(gate[n]["passes"] for n in ("C24-CP","C24-CPH"));offline_but_arena_failed=safe and not winner and bool(short);preserved=any(test_eval[n]["cp"]["preservation_rate"]>=.9 for n in ("C24-CP","C24-CPH"));corrected=any(test_eval[n]["cp"]["correction_rate"]>0 for n in ("C24-CP","C24-CPH"))
    implementation_valid=am["correction_rate"]>bm["correction_rate"] and math.isfinite(float(ap.detach())) and safe
    verdict={"CORRECT_PRESERVE_IMPLEMENTATION_VALID":"YES" if implementation_valid else "NO","CORRECTION_PAIRS_LEARNED":"YES" if corrected else "NO","PRESERVATION_PAIRS_PROTECTED":"YES" if preserved else "NO","SAFE_CORRECTION_REGION_FOUND":"YES" if safe else "NO","HIGH_GAP_INVERSIONS_REDUCED":"YES" if any(test_eval[n]["strategic"]["high_gap_harmful_inversions"]<test_eval["G2"]["strategic"]["high_gap_harmful_inversions"] for n in ("C24-CP","C24-CPH")) else "NO","NET_STRATEGIC_GAIN_POSITIVE":"YES" if any(test_eval[n]["cp"]["strategic_utility"]>0 for n in ("C24-CP","C24-CPH")) else "NO","VALUE_PRESERVED":"YES" if all(gate[n]["value_preserved"] for n in specs) else "NO","CORRECT_AND_PRESERVE_OBJECTIVE_INSUFFICIENT":"NO" if safe else "YES","OFFLINE_STRATEGIC_METRICS_NOT_SUFFICIENT":"YES" if offline_but_arena_failed else "NO" if winner else "NOT_TESTED","SEARCH_SCALING_HEALTHY":"YES" if scaling and all(x["healthy"] for x in scaling.values()) else "NO" if scaling else "INCONCLUSIVE","G3_CANDIDATE":winner.replace("-","_") if winner else "NONE"};verdict["NEXT_ACTION"]="INDEPENDENT_G3_CONFIRMATION" if winner else "SEARCH_POLICY_INTERACTION_DIAGNOSIS" if offline_but_arena_failed else "CONSTRAINED_POLICY_OPTIMIZATION_RETHINK" if not safe else "STRATEGIC_TARGET_STABILITY_REWORK"
    configuration={"hashes":{"G2":G2_SHA,"D_RL":RL_SHA,"D_REANALYSIS":RE_SHA},"epsilon_gap":.02,"rho":rho,"lambda_pres":lambda_pres,"lambda_correction":.1,"distribution_coefficient_CPH":.25,"seed":args.seed,"training":asdict(config),"initialization":initialization,"pair_classification":"fixed from frozen G2 before training","checkpoint_selection":"preservation gate first, positive correction, then strategic utility"};write_json(args.output/"configuration.json",configuration)
    comparison={n:{"validation":validation_eval[n],"test":test_eval[n],"gate":gate.get(n),"short_arena":short.get(n,{}).get("summary"),"main_arena":{b:x["summary"] for b,x in main_arena.get(n,{}).items()}} for n in test_eval};write_json(args.output/"candidate_comparison.json",comparison)
    report={"lot":24,"configuration":configuration,"pair_statistics":stats,"pair_conflicts":conflicts,"calibration":calibration,"micro_overfit":micro_report,"training":{n:{"best_epoch":x["best_epoch"],"checkpoint":x["checkpoint"]} for n,x in training.items()},"validation":validation_eval,"strict_test":test_eval,"offline_gate":gate,"short_arena":{n:x["summary"] for n,x in short.items()},"finalists":finalists,"main_arena":{n:{b:x["summary"] for b,x in p.items()} for n,p in main_arena.items()},"search_scaling":scaling,"verdict":verdict,"scientific_question":"YES" if safe else "NO","playing_question":"YES" if winner else "NO" if offline_but_arena_failed else "INCONCLUSIVE","selfplay_generated":False,"minimax_executed":False,"elapsed_s":time.perf_counter()-started};write_json(args.output/"report.json",report);print(json.dumps({"report":str(args.output/"report.json"),"verdict":verdict,"scientific_question":report["scientific_question"],"playing_question":report["playing_question"],"elapsed_s":report["elapsed_s"]},indent=2),flush=True)


def distribution(values):
    values=list(values)
    if not values:return {"count":0}
    return dist(values)


if __name__=="__main__":main()
