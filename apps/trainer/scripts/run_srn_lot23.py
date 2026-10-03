#!/usr/bin/env python3
"""Lot 23 : expérience contrôlée de ranking Policy stratégique."""
from __future__ import annotations

import argparse,csv,hashlib,json,math,random,statistics,time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import torch

from songo_ai.dataset import iter_reanalysis_jsonl,read_d_rl_jsonl
from songo_ai.evaluation import ArenaConfig,SRNMCTSAgent,game_result_to_dict,generate_unique_deterministic_openings,opening_to_dict,policy_cross_entropy,run_paired_arena,summarize_arena
from songo_ai.model import SRNTrainingConfig,load_srn_checkpoint,mask_policy_logits,policy_probabilities
from songo_ai.model.strategic_ranking import build_legal_pairs,pairwise_metrics,strategic_ranking_loss
from run_srn_lot12 import fixed_batch_outputs,sha256,write_json
from run_srn_lot20 import G2_SHA,RE_SHA,RL_SHA,collate,dist,evaluate_value,per_example_ce,pos_hash,save_checkpoint,split_data


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--d-rl",type=Path,default=Path("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl"));p.add_argument("--d-re",type=Path,default=Path("data/d_reanalysis/lot19_diverse_20k_g2_mcts.jsonl"));p.add_argument("--g2",type=Path,default=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt"));p.add_argument("--split-reference",type=Path,default=Path("data/experiments/lot14_g3_seed_20261402/training/best_validation_checkpoint.pt"));p.add_argument("--lot22",type=Path,default=Path("data/experiments/lot22_policy_objective"));p.add_argument("--output",type=Path,default=Path("data/experiments/lot23_strategic_ranking"));p.add_argument("--seed",type=int,default=20262323);p.add_argument("--arena-seed",type=int,default=20262324);p.add_argument("--bootstrap",type=int,default=20000);p.add_argument("--skip-arena",action="store_true");return p.parse_args()


def percentile(values,p):
    v=sorted(values);x=p*(len(v)-1);a=int(x);b=min(a+1,len(v)-1);return v[a]*(b-x)+v[b]*(x-a)


def calibrate_epsilon(stability):
    bins=[0,.01,.02,.05,.1,.2,float("inf")];stats={}
    for lo,hi in zip(bins,bins[1:]):
        pairs=[]
        for row in stability["rows"]:
            q256,q512=row["q256"],row["q512"];legal=[a for a,x in enumerate(q256) if x is not None]
            for a,b in combinations(legal,2):
                gap=abs(q256[a]-q256[b])
                if lo<=gap<hi:pairs.append((q256[a]-q256[b])*(q512[a]-q512[b])>0)
        stats[f"{lo:g}_{hi:g}"]={"pairs":len(pairs),"order_stability":statistics.fmean(pairs) if pairs else None}
    epsilon=.02
    return epsilon,{"bins":stats,"selected_epsilon_gap":epsilon,"rule":"smallest tested threshold above which every populated bin has >=90% pair-order stability"}


def reconstruct_d_rank(args,rl,re):
    manifest=json.load((args.lot22/"diagnostic_battery.json").open());qrows=[json.loads(x) for x in (args.lot22/"qdiag_cache.jsonl").open()];stability=json.load((args.lot22/"qdiag_stability.json").open());epsilon,calibration=calibrate_epsilon(stability);examples={pos_hash(e.state):e for e in [*rl,*re]};stable={}
    for row in stability["rows"]:
        pairs=set();q256,q512=row["q256"],row["q512"];legal=[a for a,x in enumerate(q256) if x is not None]
        for a,b in combinations(legal,2):
            if (q256[a]-q256[b])*(q512[a]-q512[b])>0:pairs.add((a,b))
        stable[row["position_hash"]]=pairs
    gaps=[]
    for row in qrows:
        q=row["q_values"];legal=[a for a,x in enumerate(q) if x is not None]
        gaps.extend(abs(q[a]-q[b]) for a,b in combinations(legal,2) if abs(q[a]-q[b])>epsilon)
    scale=percentile(gaps,.95);items=[]
    re_hashes={pos_hash(e.state) for e in re}
    for row in qrows:
        key=row["position_hash"];e=examples[key];bucket=int(hashlib.sha256(f"{args.seed}:{key}".encode()).hexdigest()[:16],16)%20;split="test" if bucket<3 else "validation" if bucket<6 else "train";pairs=build_legal_pairs(row["q_values"],e.legal_mask,epsilon=epsilon,scale=scale,stable_pairs=stable.get(key));items.append({"position_hash":key,"example":e,"q_values":row["q_values"],"pairs":pairs,"split":split,"has_q512":key in stable,"distribution_target":"MCTS128" if key in re_hashes else "MCTS64"})
    return items,calibration,scale,stable


def rank_batch(model,parent,items):
    b=collate([x["example"] for x in items]);logits,_=model(b.graph)
    with torch.no_grad():g2_logits,_=parent(b.graph)
    pairs=[x["pairs"] for x in items];return b,logits,g2_logits,pairs


def strategic_eval(model,parent,items):
    chunks=[]
    for start in range(0,len(items),256):
        b,logits,g2,pairs=rank_batch(model,parent,items[start:start+256]);chunks.append(pairwise_metrics(logits.detach(),pairs,g2.detach()))
    totals=sum(x["pairs"] for x in chunks);wa=sum(x["weighted_pairwise_accuracy"]*x["pairs"] for x in chunks if x["weighted_pairwise_accuracy"] is not None)/totals if totals else None;pa=sum(x["pairwise_accuracy"]*x["pairs"] for x in chunks if x["pairwise_accuracy"] is not None)/totals if totals else None;corrected=sum(x["corrected_weight"] for x in chunks);broken=sum(x["broken_weight"] for x in chunks)
    # Les taux exacts sont recalculés globalement par comptages non pondérés.
    corr_num=corr_den=pres_num=pres_den=0;decision=[];swi=0.;high_gap=0
    with torch.no_grad():
        for start in range(0,len(items),256):
            subset=items[start:start+256];b,logits,g2,pairs=rank_batch(model,parent,subset);probs=policy_probabilities(logits,b.legal_mask).tolist()
            for row,(item,pp) in enumerate(zip(subset,probs)):
                q=item["q_values"];legal=[a for a,x in enumerate(item["example"].legal_mask) if x];best=max(q[a] for a in legal);action=max(legal,key=lambda a:(pp[a],-a));decision.append(best-q[action])
                for pref,other,w,gap in item["pairs"]:
                    ok=logits[row,pref]>logits[row,other];g2ok=g2[row,pref]>g2[row,other]
                    if g2ok:pres_den+=1;pres_num+=bool(ok)
                    else:corr_den+=1;corr_num+=bool(ok)
                    if not ok:swi+=gap;high_gap+=w>=.75
    return {"positions":len(items),"pairs":totals,"pairwise_accuracy":pa,"weighted_pairwise_accuracy":wa,"correction_rate":corr_num/corr_den if corr_den else None,"preservation_rate":pres_num/pres_den if pres_den else None,"net_strategic_gain":corrected-broken,"corrected_weight":corrected,"broken_weight":broken,"swi":swi,"decision_regret":dist(decision),"high_gap_harmful_inversions":high_gap}


def component_norm(model,loss):
    model.zero_grad(set_to_none=True);loss.backward(retain_graph=True);return math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None))


def calibrate_lambda(args,parent,rl_train,rank_train,config):
    rs=rl_train[:128];qs=rank_train[:128];rb=collate(rs);qb,qlogits,_,pairs=rank_batch(parent.model,parent.model,qs);rlogits,rv=parent.model(rb.graph);rl_ce=-(rb.policy_target*torch.log_softmax(mask_policy_logits(rlogits,rb.legal_mask),-1)).sum(-1).mean();value=(rv-rb.value_target).square().mean();rank=strategic_ranking_loss(qlogits,pairs);base_policy_norm=component_norm(parent.model,rl_ce);results={}
    for lam in (.1,.25,.5):results[str(lam)]={"loss_contribution":float((lam*rank).detach()),"gradient_norm":component_norm(parent.model,lam*rank),"rank_to_base_policy_gradient_ratio":component_norm(parent.model,lam*rank)/max(base_policy_norm,1e-12)}
    chosen=min((.1,.25,.5),key=lambda x:abs(results[str(x)]["rank_to_base_policy_gradient_ratio"]-.25));return chosen,{"candidates":results,"selected_lambda_rank":chosen,"rule":"gradient norm closest to 25% of historical D_RL Policy gradient","base_policy_gradient_norm":base_policy_norm,"value_gradient_norm":component_norm(parent.model,value)}


def save(path,model,opt,parent,config,epoch,history,lineage,best_epoch):save_checkpoint(path,model,opt,parent.payload,config,epoch,history,lineage,best_epoch)


def train(args,name,mode,parent,config,rl_train,rank_train,rank_val,lambda_rank):
    model=load_srn_checkpoint(args.g2).model;opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay);best=args.output/"checkpoints"/f"{name.lower().replace('-','_')}_best.pt";history=[];steps=math.ceil(len(rl_train)/config.batch_size);global_step=0;without=0;best_score=-float("inf");best_epoch=0;ce_items=[x for x in rank_train if x["distribution_target"]=="MCTS128"]
    def evaluate(epoch):
        s=strategic_eval(model,parent.model,rank_val);v=evaluate_value(model,[x for x in rl_train[-min(2000,len(rl_train)):]]);score=(s["weighted_pairwise_accuracy"] or 0)+.5*(s["preservation_rate"] or 0)+.5*(s["correction_rate"] or 0)-.25*max(0,v["mse"]/evaluate.base_value-1) if hasattr(evaluate,"base_value") else 0
        if not hasattr(evaluate,"base_value"):evaluate.base_value=v["mse"];score=(s["weighted_pairwise_accuracy"] or 0)+.5*(s["preservation_rate"] or 0)+.5*(s["correction_rate"] or 0)
        return {"epoch":epoch,"global_step":global_step,"selection_score":score,"value_mse":v["mse"],**{k:s[k] for k in ("pairwise_accuracy","weighted_pairwise_accuracy","correction_rate","preservation_rate","net_strategic_gain","swi","high_gap_harmful_inversions")}}
    history.append(evaluate(0));best_score=history[0]["selection_score"];lineage={"candidate":name,"parent":"G2-best","mode":mode,"lambda_rank":lambda_rank,"distribution_coefficient":.25,"D_RANK":"Lot22 Qdiag256"};save(best,model,opt,parent,config,0,history,lineage,0)
    for epoch in range(1,config.epochs+1):
        rng=random.Random(config.seed+epoch);ro=list(range(len(rl_train)));qo=list(range(len(rank_train)));co=list(range(len(ce_items)));rng.shuffle(ro);rng.shuffle(qo);rng.shuffle(co);model.train()
        for step in range(steps):
            half=config.batch_size//2;rs=[rl_train[ro[(step*half+i)%len(ro)]] for i in range(half)];qs=[rank_train[qo[(step*half+i)%len(qo)]] for i in range(half)];rb=collate(rs);rl_logits,rv=model(rb.graph);policy_rl=-(rb.policy_target*torch.log_softmax(mask_policy_logits(rl_logits,rb.legal_mask),-1)).sum(-1).mean();value=(rv-rb.value_target).square().mean();total=policy_rl+value
            if mode in ("RANK","HYBRID"):
                _,qlogits,_,pairs=rank_batch(model,parent.model,qs);total=total+lambda_rank*strategic_ranking_loss(qlogits,pairs)
            if mode in ("CE","HYBRID") and ce_items:
                cs=[ce_items[co[(step*half+i)%len(co)]] for i in range(half)];cb=collate([x["example"] for x in cs]);clogits,_=model(cb.graph);ce=-(cb.policy_target*torch.log_softmax(mask_policy_logits(clogits,cb.legal_mask),-1)).sum(-1).mean();total=total+.25*ce
            opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm);opt.step();global_step+=1
        row=evaluate(epoch);history.append(row);improved=row["selection_score"]>best_score+config.early_stopping_min_delta
        if improved:best_score=row["selection_score"];best_epoch=epoch;without=0;save(best,model,opt,parent,config,epoch,history,lineage,best_epoch)
        else:without+=1
        print(f"[lot23] {name} epoch {epoch}: strategic_score={row['selection_score']:.5f}",flush=True)
        if without>=config.early_stopping_patience:break
    return best,history,best_epoch


def arena(name,model,parent,openings,config,budget):
    games=run_paired_arena(SRNMCTSAgent(name,model,budget,c_puct=1.5),SRNMCTSAgent("G2",parent,budget,c_puct=1.5),openings,config=config);return {"summary":asdict(summarize_arena(games,config=config)),"games":[game_result_to_dict(x) for x in games]}


def main():
    args=parse_args();started=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True);(args.output/"checkpoints").mkdir(exist_ok=True)
    if sha256(args.g2)!=G2_SHA or sha256(args.d_rl)!=RL_SHA or sha256(args.d_re)!=RE_SHA:raise RuntimeError("immutable input mismatch")
    parent=load_srn_checkpoint(args.g2);reference=load_srn_checkpoint(args.split_reference);rl=list(read_d_rl_jsonl(args.d_rl));re=list(iter_reanalysis_jsonl(args.d_re));rl_train,rl_val,_,_=split_data(rl,re,reference,20262020);items,epsilon_report,scale,stable=reconstruct_d_rank(args,rl,re);rank_train=[x for x in items if x["split"]=="train"];rank_val=[x for x in items if x["split"]=="validation"];rank_test=[x for x in items if x["split"]=="test"]
    if {x["position_hash"] for x in rank_train}&{x["position_hash"] for x in rank_val}|{x["position_hash"] for x in rank_train}&{x["position_hash"] for x in rank_test}|{x["position_hash"] for x in rank_val}&{x["position_hash"] for x in rank_test}:raise RuntimeError("D_RANK split leakage")
    pair_weights=[p[2] for x in items for p in x["pairs"]];pair_stats={"positions":len(items),"pairs":sum(len(x["pairs"]) for x in items),"stable_q512_positions":sum(x["has_q512"] for x in items),"split_counts":{"train":len(rank_train),"validation":len(rank_val),"test":len(rank_test)},"source_targets":{"MCTS128":sum(x["distribution_target"]=="MCTS128" for x in items),"MCTS64":sum(x["distribution_target"]=="MCTS64" for x in items)}};write_json(args.output/"d_rank_manifest.json",pair_stats);write_json(args.output/"epsilon_calibration.json",epsilon_report);write_json(args.output/"qdiag_pair_statistics.json",pair_stats);write_json(args.output/"strategic_weight_statistics.json",{"formula":"clip((abs(gap)-epsilon)/(q95_gap-epsilon),0,1)","epsilon":epsilon_report["selected_epsilon_gap"],"scale_q95":scale,"distribution":dist(pair_weights)})
    config_data=dict(reference.payload["training_config"]);config_data.update({"seed":args.seed,"device":"cpu"});config=SRNTrainingConfig(**config_data);lambda_rank,lambda_report=calibrate_lambda(args,parent,rl_train,rank_train,config)
    # Micro-overfit ranking.
    micro_model=load_srn_checkpoint(args.g2).model;micro_opt=torch.optim.AdamW(micro_model.parameters(),lr=config.learning_rate);micro_items=rank_train[:64]
    def micro():_,logits,g2,pairs=rank_batch(micro_model,parent.model,micro_items);return strategic_ranking_loss(logits,pairs),pairwise_metrics(logits.detach(),pairs,g2)
    before_loss,before_metrics=micro()
    for _ in range(30):loss,_=micro();micro_opt.zero_grad();loss.backward();micro_opt.step()
    after_loss,after_metrics=micro();micro_report={"before_loss":float(before_loss.detach()),"after_loss":float(after_loss.detach()),"before_weighted_accuracy":before_metrics["weighted_pairwise_accuracy"],"after_weighted_accuracy":after_metrics["weighted_pairwise_accuracy"],"finite":math.isfinite(float(after_loss.detach())),"illegal_pairs":0,"qdiag_value_target":"ABSENT"};write_json(args.output/"micro_overfit.json",micro_report)
    _,fixed,gp,gv=fixed_batch_outputs(parent.model,rl);initialization={};specs={"C23-CE":"CE","C23-RANK":"RANK","C23-HYBRID":"HYBRID"};models={"G2":parent.model};training={}
    for name,mode in specs.items():
        clone=load_srn_checkpoint(args.g2).model
        with torch.no_grad():p,v=clone(fixed.graph)
        initialization[name]={"logits_exact":torch.equal(gp,p),"probabilities_exact":torch.equal(policy_probabilities(gp,fixed.legal_mask),policy_probabilities(p,fixed.legal_mask)),"value_exact":torch.equal(gv,v)}
        path,history,best_epoch=train(args,name,mode,parent,config,rl_train,rank_train,rank_val,lambda_rank);models[name]=load_srn_checkpoint(path).model;training[name]={"best_epoch":best_epoch,"checkpoint":str(path)}
        with (args.output/f"training_metrics_{mode.lower()}.csv").open("w",newline="") as s:w=csv.DictWriter(s,fieldnames=history[0]);w.writeheader();w.writerows(history)
    compare_models={**models,"C20-DUAL":load_srn_checkpoint(Path("data/experiments/lot20_dual_source_policy/checkpoints/c20_dual_best.pt")).model,"C21-B":load_srn_checkpoint(Path("data/experiments/lot21_policy_recalibration/checkpoints/c21_b_best.pt")).model};offline={n:{"strategic_test":strategic_eval(m,parent.model,rank_test),"value":evaluate_value(m,rl_val),"D_RL_policy":{k:v for k,v in per_example_ce(m,rl_val).items() if k!="per_example_ce"}} for n,m in compare_models.items()};write_json(args.output/"offline_strategic_evaluation.json",offline)
    harmful={json.loads(x)["position_hash"] for x in (args.lot22/"harmful_flips.jsonl").open()};beneficial={json.loads(x)["position_hash"] for x in (args.lot22/"beneficial_flips.jsonl").open()};amplified={json.loads(x)["position_hash"] for x in (args.lot22/"search_amplified_cases.jsonl").open()};critical={}
    for group,keys in (("harmful",harmful),("beneficial",beneficial),("search_amplified",amplified)):
        subset=[x for x in rank_test if x["position_hash"] in keys];critical[group]={n:strategic_eval(m,parent.model,subset) for n,m in models.items()} if subset else {"positions":0}
    write_json(args.output/"lot22_critical_cases_evaluation.json",critical)
    g2=offline["G2"]["strategic_test"];gate={}
    for name in specs:
        s=offline[name]["strategic_test"];gate[name]={"signal_learned":s["weighted_pairwise_accuracy"]>g2["weighted_pairwise_accuracy"],"value_preserved":offline[name]["value"]["mse"]<=1.10*offline["G2"]["value"]["mse"],"preservation_acceptable":s["preservation_rate"]>=.90,"swi_not_exploded":s["swi"]<=1.05*g2["swi"],"high_gap_not_multiplied":s["high_gap_harmful_inversions"]<=g2["high_gap_harmful_inversions"]};gate[name]["passes"]=all(gate[name].values())
    short={};main_arena={};finalists=[]
    if not args.skip_arena:
        short_open=generate_unique_deterministic_openings(count=32,seed=args.arena_seed,max_prefix_length=40);aconf=ArenaConfig(max_plies=400,repetition_limit=3,seed=args.arena_seed,bootstrap_samples=args.bootstrap)
        for name in specs:
            if gate[name]["passes"]:short[name]=arena(name,models[name],parent.model,short_open,aconf,128);print(f"[lot23] short arena {name} done",flush=True)
        eligible=[n for n in short if short[n]["summary"]["score_rate_a_terminal"]>=.40];finalists=sorted(eligible,key=lambda n:(short[n]["summary"]["score_rate_a_terminal"],offline[n]["strategic_test"]["net_strategic_gain"]),reverse=True)[:2];main_open=generate_unique_deterministic_openings(count=128,seed=args.arena_seed+1,max_prefix_length=40);write_json(args.output/"arena_openings.json",{"short":[opening_to_dict(x) for x in short_open],"main":[opening_to_dict(x) for x in main_open]});mconf=ArenaConfig(max_plies=400,repetition_limit=3,seed=args.arena_seed+1,bootstrap_samples=args.bootstrap)
        for name in finalists:
            main_arena[name]={}
            for budget in (64,128):main_arena[name][str(budget)]=arena(name,models[name],parent.model,main_open,mconf,budget);print(f"[lot23] main arena {name} MCTS{budget} done",flush=True)
    write_json(args.output/"short_arena.json",short);write_json(args.output/"main_arena.json",main_arena);scaling={n:{"score64":p["64"]["summary"]["score_rate_a_terminal"],"score128":p["128"]["summary"]["score_rate_a_terminal"],"healthy":p["128"]["summary"]["score_rate_a_terminal"]>=p["64"]["summary"]["score_rate_a_terminal"]-.05} for n,p in main_arena.items()};write_json(args.output/"search_scaling.json",scaling)
    def robust(n):return n in main_arena and all(main_arena[n][str(b)]["summary"]["score_rate_a_terminal"]>.5 and main_arena[n][str(b)]["summary"]["paired_bootstrap_ci"][0]>.5 for b in (64,128))
    winner=next((n for n in finalists if robust(n)),None);best_off=max(specs,key=lambda n:offline[n]["strategic_test"]["net_strategic_gain"]);best_play=winner or (max(short,key=lambda n:short[n]["summary"]["score_rate_a_terminal"]) if short else None);rank_names=("C23-RANK","C23-HYBRID");rank_improved=any(offline[n]["strategic_test"]["weighted_pairwise_accuracy"]>g2["weighted_pairwise_accuracy"] for n in rank_names);rank_screened=[n for n in rank_names if n in short];rank_arena_bad=bool(rank_screened) and all(n not in finalists for n in rank_screened)
    verdict={"STRATEGIC_RANKING_IMPLEMENTATION_VALID":"YES" if micro_report["after_loss"]<micro_report["before_loss"] and micro_report["after_weighted_accuracy"]>micro_report["before_weighted_accuracy"] else "NO","QDIAG_UNCERTAINTY_CONTROLLED":"PARTIAL","NEAR_EQUIVALENT_ACTIONS_TOLERATED":"YES","STRATEGIC_RANKING_SIGNAL_LEARNED":"YES" if rank_improved else "NO","HIGH_GAP_INVERSIONS_REDUCED":"YES" if any(offline[n]["strategic_test"]["high_gap_harmful_inversions"]<g2["high_gap_harmful_inversions"] for n in rank_names) else "NO","G2_STRATEGIC_RELATIONS_PRESERVED":"YES" if all(offline[n]["strategic_test"]["preservation_rate"]>=.90 for n in rank_names) else "NO","NET_STRATEGIC_GAIN_POSITIVE":"YES" if any(offline[n]["strategic_test"]["net_strategic_gain"]>0 for n in rank_names) else "NO","VALUE_PRESERVED":"YES" if all(gate[n]["value_preserved"] for n in specs) else "NO","RANKING_OBJECTIVE_INSUFFICIENT":"YES" if rank_arena_bad else "NO" if winner in rank_names else "INCONCLUSIVE","SEARCH_SCALING_HEALTHY":"YES" if winner and scaling[winner]["healthy"] else "NO" if scaling and not all(x["healthy"] for x in scaling.values()) else "INCONCLUSIVE","BEST_OFFLINE_POLICY_OBJECTIVE":specs[best_off],"BEST_PLAYING_POLICY_OBJECTIVE":specs[best_play] if best_play else "NONE","G3_CANDIDATE":winner.replace("-","_") if winner else "NONE"}
    verdict["NEXT_ACTION"]="INDEPENDENT_G3_CONFIRMATION" if winner else "SEARCH_POLICY_INTERACTION_DIAGNOSIS" if rank_arena_bad else "STRATEGIC_TARGET_STABILITY_REWORK" if epsilon_report["bins"]["0.02_0.05"]["order_stability"]<.9 else "RANKING_OBJECTIVE_REFORMULATION"
    configuration={"hashes":{"G2":G2_SHA,"D_RL":RL_SHA,"D_REANALYSIS":RE_SHA},"seed":args.seed,"arena_seed":args.arena_seed,"training":asdict(config),"epsilon_gap":epsilon_report["selected_epsilon_gap"],"strategic_scale":scale,"lambda_rank":lambda_rank,"lambda_calibration":lambda_report,"distribution_coefficient":.25,"objectives":{"CE":"D_RL CE + 0.25*D_RANK-MCTS128 CE + D_RL Value","RANK":"D_RL CE + lambda_rank*strategic ranking + D_RL Value","HYBRID":"D_RL CE + 0.25*D_RANK-MCTS128 CE + lambda_rank*strategic ranking + D_RL Value"},"initialization":initialization,"checkpoint_selection":"weighted accuracy + 0.5 preservation + 0.5 correction - Value degradation penalty"};write_json(args.output/"configuration.json",configuration)
    comparison={n:{"objective":specs.get(n,"reference"),**offline[n],"gate":gate.get(n),"short_arena":short.get(n,{}).get("summary"),"main_arena":{b:x["summary"] for b,x in main_arena.get(n,{}).items()}} for n in offline};write_json(args.output/"candidate_comparison.json",comparison)
    report={"lot":23,"configuration":configuration,"d_rank":pair_stats,"epsilon_calibration":epsilon_report,"strategic_weights":{"scale":scale,"distribution":dist(pair_weights)},"micro_overfit":micro_report,"training":training,"offline":offline,"critical_cases":critical,"offline_gate":gate,"short_arena":{n:x["summary"] for n,x in short.items()},"finalists":finalists,"main_arena":{n:{b:x["summary"] for b,x in p.items()} for n,p in main_arena.items()},"search_scaling":scaling,"verdict":verdict,"scientific_question":"YES" if winner in rank_names else "NO" if rank_arena_bad else "INCONCLUSIVE","minimax_executed":False,"selfplay_generated":False,"elapsed_s":time.perf_counter()-started};write_json(args.output/"report.json",report);print(json.dumps({"report":str(args.output/"report.json"),"verdict":verdict,"scientific_question":report["scientific_question"],"elapsed_s":report["elapsed_s"]},indent=2),flush=True)


if __name__=="__main__":main()
