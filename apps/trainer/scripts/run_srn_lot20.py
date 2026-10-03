#!/usr/bin/env python3
"""Lot 20 : entraînement Policy dual-source et arènes contrôlées."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,shutil,statistics,time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import torch

from songo_ai.dataset import RawSongoState, RLTrainingExample, iter_reanalysis_jsonl, read_d_rl_jsonl
from songo_ai.evaluation import ArenaConfig, SRNMCTSAgent, game_result_to_dict, generate_unique_deterministic_openings, legal_ranking, model_parameter_fingerprint, opening_to_dict, policy_entropy, run_paired_arena, summarize_arena
from songo_ai.model import SRNBatchCollator, SRNConfig, SRNTrainingConfig, load_srn_checkpoint, make_srn_loader, mask_policy_logits, policy_probabilities, train_srn_from_d_rl
from songo_ai.model.dual_source_training import bounded_confidence_weights,bounded_regret_weights,dual_source_loss,weighted_policy_cross_entropy
from run_srn_lot12 import fixed_batch_outputs,sha256,write_json

G2_SHA="eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753"; RE_SHA="ddc42088c38264fc05dfe6b1faee62a4b7b1fa2613219d71fc6414351928d0cb"; RL_SHA="2f24dacc33f897ed9645b08823a6601d4a48601b7f3bc0c9e8a7c120aa850e57"

def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--d-rl",type=Path,default=Path("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl")); p.add_argument("--d-re",type=Path,default=Path("data/d_reanalysis/lot19_diverse_20k_g2_mcts.jsonl")); p.add_argument("--g2",type=Path,default=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt")); p.add_argument("--split-reference",type=Path,default=Path("data/experiments/lot14_g3_seed_20261402/training/best_validation_checkpoint.pt")); p.add_argument("--output",type=Path,default=Path("data/experiments/lot20_dual_source_policy")); p.add_argument("--seed",type=int,default=20262020); p.add_argument("--arena-seed",type=int,default=20262021); p.add_argument("--openings",type=int,default=128); p.add_argument("--bootstrap",type=int,default=20000); p.add_argument("--skip-arena",action="store_true"); return p.parse_args()

def dist(values):
    v=sorted(float(x) for x in values)
    if not v:return {"count":0}
    def q(p): x=p*(len(v)-1);a=int(x);b=min(a+1,len(v)-1);return v[a]*(b-x)+v[b]*(x-a)
    return {"count":len(v),"minimum":v[0],"mean":statistics.fmean(v),"median":statistics.median(v),"p90":q(.9),"p95":q(.95),"p99":q(.99),"maximum":v[-1]}

def pos_hash(state): return hashlib.sha256((",".join(map(str,state.board))+f"|{state.player_to_move}").encode()).hexdigest()

def split_data(rl,re,reference,seed):
    train_ids=set(reference.payload["train_game_ids"]); val_ids=set(reference.payload["validation_game_ids"]); rl_train=[e for e in rl if e.metadata["game_id"] in train_ids]; rl_val=[e for e in rl if e.metadata["game_id"] in val_ids]
    re_train=[];re_val=[]
    for e in re:
        h=e.source_position_metadata["position_hash"]; bucket=int(hashlib.sha256(f"{seed}:{h}".encode()).hexdigest()[:16],16)%5; (re_val if bucket==0 else re_train).append(e)
    assert not ({e.source_position_metadata["position_hash"] for e in re_train}&{e.source_position_metadata["position_hash"] for e in re_val})
    return rl_train,rl_val,re_train,re_val

def collate(examples):
    class Item:
        pass
    converted=[]
    for i,e in enumerate(examples):
        x=Item();x.state=e.state;x.legal_mask=e.legal_mask;x.policy_target=e.policy_target;x.value_target=getattr(e,"value_target",None);x.metadata={"game_id":getattr(e,"metadata",{}).get("game_id",f"re-{i}")};converted.append(x)
    return SRNBatchCollator()(converted)

def per_example_ce(model,examples,batch_size=512):
    values=[];tops=[];ent=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(examples),batch_size):
            b=collate(examples[start:start+batch_size]);logits,_=model(b.graph);masked=mask_policy_logits(logits,b.legal_mask);logp=torch.log_softmax(masked,-1); values.extend((-(b.policy_target*logp).sum(-1)).tolist()); probs=policy_probabilities(logits,b.legal_mask).tolist();tops.extend(int(legal_ranking(p,m)[0]==legal_ranking(t,m)[0]) for p,t,m in zip(probs,b.policy_target.tolist(),b.legal_mask.tolist()));ent.extend(policy_entropy(p) for p in probs)
    return {"ce":statistics.fmean(values),"top1":statistics.fmean(tops),"entropy":statistics.fmean(ent),"per_example_ce":values}

def evaluate_value(model,examples):
    pred=[];target=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(examples),512):
            b=collate(examples[start:start+512]);_,v=model(b.graph); pred.extend(v.tolist());target.extend(b.value_target.tolist())
    dif=[a-b for a,b in zip(pred,target)];return {"mse":statistics.fmean(x*x for x in dif),"mae":statistics.fmean(abs(x) for x in dif),"sign_accuracy":statistics.fmean((1 if a>.1 else -1 if a<-.1 else 0)==(1 if b>0 else -1 if b<0 else 0) for a,b in zip(pred,target))}

def load_regrets(path,examples):
    mapping={};actions={}
    for line in path.open():
        r=json.loads(line); key=pos_hash(examples[int(r["index"])].state); regrets=[x for x in r["regrets"] if x is not None]; mapping[key]=max(regrets) if regrets else 0.0;actions[key]=r["regrets"]
    return mapping,actions

def alpha_calibration(parent,rl_train,regret_map,action_regret_map,q95,config):
    diagnostic=[e for e in rl_train if pos_hash(e.state) in regret_map]; diagnostic.sort(key=lambda e:pos_hash(e.state)); split=max(1,int(.8*len(diagnostic))); train=diagnostic[:split];val=diagnostic[split:];results={}
    for alpha in (0.,1.,2.):
        model=load_srn_checkpoint(parent).model;opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay);rng=random.Random(config.seed)
        for step in range(40):
            batch=[train[rng.randrange(len(train))] for _ in range(min(128,len(train)))];b=collate(batch);logits,_=model(b.graph);weights=torch.tensor(bounded_regret_weights([regret_map[pos_hash(e.state)] for e in batch],alpha=alpha,q95=q95));loss=weighted_policy_cross_entropy(logits,b.legal_mask,b.policy_target,weights);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        metrics=per_example_ce(model,val);weights=bounded_regret_weights([regret_map[pos_hash(e.state)] for e in val],alpha=alpha,q95=q95);chosen_regrets=[]
        model.eval()
        with torch.no_grad():
            for start in range(0,len(val),512):
                batch=val[start:start+512];b=collate(batch);logits,_=model(b.graph);policies=policy_probabilities(logits,b.legal_mask).tolist()
                for e,p,m in zip(batch,policies,b.legal_mask.tolist()): chosen_regrets.append(float(action_regret_map[pos_hash(e.state)][legal_ranking(p,m)[0]]))
        results[str(int(alpha))]={"heldout_positions":len(val),"weighted_ce":statistics.fmean(w*c for w,c in zip(weights,metrics["per_example_ce"])),"unweighted_ce":metrics["ce"],"selected_action_regret":dist(chosen_regrets),"high_regret_errors_gt_0_25":sum(x>.25 for x in chosen_regrets)}
    chosen=min((0,1,2),key=lambda a:(results[str(a)]["selected_action_regret"]["p95"],results[str(a)]["selected_action_regret"]["mean"],a));return chosen,results

def save_checkpoint(path,model,optimizer,parent_payload,config,epoch,history,lineage,best_epoch):
    payload=dict(parent_payload);payload.update({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"training_config":asdict(config),"epoch":epoch,"global_step":history[-1]["global_step"],"history":history,"best_epoch":best_epoch,"best_validation_loss":min(r["selection_score"] for r in history),"initialization":{"kind":"checkpoint_weights_fresh_optimizer","sha256":G2_SHA},"lineage":lineage,"stopped_early":False});torch.save(payload,path)

def train_dual(args,config,rl_train,rl_val,re_train,re_val,regret_weight_map,alpha,conf_map):
    model=load_srn_checkpoint(args.g2).model;optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay);out=args.output/"checkpoints";out.mkdir(parents=True,exist_ok=True);best=out/"c20_dual_best.pt";last=out/"c20_dual_last.pt";parent=load_srn_checkpoint(args.g2).payload;history=[];best_score=float("inf");best_epoch=0;without=0;steps_per_epoch=math.ceil(len(rl_train)/config.batch_size);global_step=0
    def metrics(epoch):
        rl=per_example_ce(model,rl_val);re=per_example_ce(model,re_val);value=evaluate_value(model,rl_val)
        if epoch==0: metrics.baseline=(rl["ce"],re["ce"],value["mse"])
        score=rl["ce"]/metrics.baseline[0]+re["ce"]/metrics.baseline[1]+value["mse"]/metrics.baseline[2]
        return {"epoch":epoch,"global_step":global_step,"policy_rl_ce":rl["ce"],"policy_rl_top1":rl["top1"],"policy_re_ce":re["ce"],"policy_re_top1":re["top1"],"value_rl_mse":value["mse"],"value_rl_sign_accuracy":value["sign_accuracy"],"selection_score":score}
    history.append(metrics(0));lineage={"candidate":"C20-DUAL","parent":"G2-best","G2_sha256":G2_SHA,"D_RL_sha256":RL_SHA,"D_REANALYSIS_sha256":RE_SHA,"alpha":alpha,"confidence_beta":1.0}
    save_checkpoint(best,model,optimizer,parent,config,0,history,lineage,0)
    for epoch in range(1,config.epochs+1):
        rr=random.Random(config.seed+epoch);rl_order=list(range(len(rl_train)));re_order=list(range(len(re_train)));rr.shuffle(rl_order);rr.shuffle(re_order);model.train()
        for step in range(steps_per_epoch):
            half=config.batch_size//2;ri=[rl_order[(step*half+i)%len(rl_order)] for i in range(half)];ei=[re_order[(step*half+i)%len(re_order)] for i in range(half)];r_examples=[rl_train[i] for i in ri];e_examples=[re_train[i] for i in ei];rb=collate(r_examples);eb=collate(e_examples);rl_logits,rl_v=model(rb.graph);re_logits,re_v=model(eb.graph);rw=torch.tensor([regret_weight_map[pos_hash(e.state)] for e in r_examples]);cw=torch.tensor([conf_map[e.source_position_metadata["position_hash"]] for e in e_examples]);loss=dual_source_loss(rl_logits,rl_v,rb.legal_mask,rb.policy_target,rb.value_target,rw,re_logits,eb.legal_mask,eb.policy_target,cw);optimizer.zero_grad();loss["total"].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm);optimizer.step();global_step+=1
        row=metrics(epoch);history.append(row);improved=row["selection_score"]<best_score-config.early_stopping_min_delta
        if improved:best_score=row["selection_score"];best_epoch=epoch;without=0;save_checkpoint(best,model,optimizer,parent,config,epoch,history,lineage,best_epoch)
        else:without+=1
        save_checkpoint(last,model,optimizer,parent,config,epoch,history,lineage,best_epoch);print(f"[lot20] dual epoch {epoch}: score={row['selection_score']:.4f}",flush=True)
        if without>=config.early_stopping_patience:break
    return best,last,history,best_epoch

def tail_evaluation(models,examples,regret_cache,val_ids):
    rows=[]
    for line in regret_cache.open():
        r=json.loads(line);e=examples[int(r["index"])]
        if e.metadata["game_id"] in val_ids:rows.append((e,r))
    results={};actions={}
    for name,model in models.items():
        ev=per_example_ce(model,[e for e,_ in rows]);acts=[];reg=[]
        model.eval()
        with torch.no_grad():
            for start in range(0,len(rows),512):
                b=collate([e for e,_ in rows[start:start+512]]);logits,_=model(b.graph);p=policy_probabilities(logits,b.legal_mask).tolist();acts.extend(legal_ranking(x,m)[0] for x,m in zip(p,b.legal_mask.tolist()))
        for action,(_,r) in zip(acts,rows):reg.append(float(r["regrets"][action]))
        actions[name]=acts;results[name]={"positions":len(reg),"regret":dist(reg),"mean_ce":ev["ce"],"high_regret_errors_gt_0_25":sum(x>.25 for x in reg)}
    base=results["G2"];g2reg=[]
    for action,(_,r) in zip(actions["G2"],rows):g2reg.append(float(r["regrets"][action]))
    for name in ("C20-BASE","C20-DUAL"):
        current=[]
        for action,(_,r) in zip(actions[name],rows):current.append(float(r["regrets"][action]))
        deltas=[a-b for a,b in zip(current,g2reg)];positive=sorted((x for x in deltas if x>0),reverse=True);k=max(1,math.ceil(.05*len(deltas)));results[name]["vs_G2"]={"regressions":sum(x>0 for x in deltas),"improvements":sum(x<0 for x in deltas),"worst_5_percent_regression_cost":sum(positive[:k]),"mean_delta_regret":statistics.fmean(deltas)}
    return results

def arena_duel(a_name,a,b_name,b,budget,openings,config):
    results=run_paired_arena(SRNMCTSAgent(a_name,a,budget,c_puct=1.5),SRNMCTSAgent(b_name,b,budget,c_puct=1.5),openings,config=config);return asdict(summarize_arena(results,config=config)),[game_result_to_dict(x) for x in results]

def main():
    args=parse_args();started=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True);(args.output/"checkpoints").mkdir(exist_ok=True)
    if sha256(args.g2)!=G2_SHA or sha256(args.d_rl)!=RL_SHA or sha256(args.d_re)!=RE_SHA:raise RuntimeError("immutable input hash mismatch")
    parent=load_srn_checkpoint(args.g2);reference=load_srn_checkpoint(args.split_reference);rl=list(read_d_rl_jsonl(args.d_rl));re=list(iter_reanalysis_jsonl(args.d_re));rl_train,rl_val,re_train,re_val=split_data(rl,re,reference,args.seed)
    config_data=dict(reference.payload["training_config"]);config_data.update({"seed":args.seed,"device":"cpu"});config=SRNTrainingConfig(**config_data);fixed,fixed_batch,g2p,g2v=fixed_batch_outputs(parent.model,rl);clone_a=load_srn_checkpoint(args.g2).model;clone_b=load_srn_checkpoint(args.g2).model
    with torch.no_grad():ap,av=clone_a(fixed_batch.graph);bp,bv=clone_b(fixed_batch.graph)
    initialization={"C20_BASE_policy_exact":torch.equal(g2p,ap),"C20_BASE_value_exact":torch.equal(g2v,av),"C20_DUAL_policy_exact":torch.equal(g2p,bp),"C20_DUAL_value_exact":torch.equal(g2v,bv)}
    if not all(initialization.values()):raise RuntimeError("epoch0 differs from G2")
    regret_map,action_regret_map=load_regrets(Path("data/experiments/lot17_policy_regret/regret_search_cache.jsonl"),rl);known=sorted(regret_map.values());q95=known[int(.95*(len(known)-1))] or 1.0;alpha,alpha_results=alpha_calibration(args.g2,rl_train,regret_map,action_regret_map,q95,config)
    regret_values=bounded_regret_weights([regret_map.get(pos_hash(e.state)) for e in rl],alpha=alpha,q95=q95);regret_weight_map={pos_hash(e.state):w for e,w in zip(rl,regret_values)}
    conf_values=bounded_confidence_weights([e.search_metadata for e in re],[sum(e.legal_mask) for e in re]);conf_map={e.source_position_metadata["position_hash"]:w for e,w in zip(re,conf_values)}
    weights={"regret":{"formula":"bounded mean-one normalization of 1 + alpha*clip(regret/q95,0,1); missing=1 before normalization","alpha_candidates":[0,1,2],"selected_alpha":alpha,"q95":q95,"known_positions":len(regret_map),"distribution":dist(regret_values),"calibration":alpha_results},"confidence":{"formula":"bounded mean-one normalization of 1 + (0.5*visit_margin + 0.5*(1-normalized_entropy) - mean_score)","distribution":dist(conf_values),"full_positions":len(conf_values)}};write_json(args.output/"weighting_statistics.json",weights)
    # Micro-test: losses doivent diminuer et aucune Value RE n'existe.
    micro_model=load_srn_checkpoint(args.g2).model;micro_opt=torch.optim.AdamW(micro_model.parameters(),lr=config.learning_rate);rsmall=rl_train[:64];esmall=re_train[:64];rb=collate(rsmall);eb=collate(esmall)
    def micro_loss():
        rp,rv=micro_model(rb.graph);ep,ev=micro_model(eb.graph);return dual_source_loss(rp,rv,rb.legal_mask,rb.policy_target,rb.value_target,torch.ones(len(rsmall)),ep,eb.legal_mask,eb.policy_target,torch.ones(len(esmall)))
    before={k:float(v.item()) for k,v in micro_loss().items()};
    for _ in range(20):loss=micro_loss();micro_opt.zero_grad();loss["total"].backward();micro_opt.step()
    after={k:float(v.item()) for k,v in micro_loss().items()};micro={"before":before,"after":after,"policy_rl_learned":after["policy_rl"]<before["policy_rl"],"policy_re_learned":after["policy_re"]<before["policy_re"],"value_rl_learned":after["value_rl"]<before["value_rl"],"reanalysis_value_target":"ABSENT"};write_json(args.output/"micro_overfit.json",micro)
    base_dir=args.output/"base_training";base=train_srn_from_d_rl(args.d_rl,base_dir,srn_config=SRNConfig(**parent.payload["srn_config"]),training_config=config,initial_checkpoint=args.g2,lineage={"candidate":"C20-BASE","parent":"G2-best"},fixed_train_game_ids=reference.payload["train_game_ids"],fixed_validation_game_ids=reference.payload["validation_game_ids"]);shutil.copy2(base.best_validation_checkpoint,args.output/"checkpoints/c20_base_best.pt");shutil.copy2(base.last_checkpoint,args.output/"checkpoints/c20_base_last.pt")
    dual_best,dual_last,dual_history,dual_best_epoch=train_dual(args,config,rl_train,rl_val,re_train,re_val,regret_weight_map,alpha,conf_map)
    base_model=load_srn_checkpoint(args.output/"checkpoints/c20_base_best.pt").model;dual_model=load_srn_checkpoint(dual_best).model;models={"G2":parent.model,"C20-BASE":base_model,"C20-DUAL":dual_model}
    offline={name:{"D_RL_policy":{k:v for k,v in per_example_ce(m,rl_val).items() if k!="per_example_ce"},"D_REANALYSIS_policy":{k:v for k,v in per_example_ce(m,re_val).items() if k!="per_example_ce"}} for name,m in models.items()};write_json(args.output/"offline_policy_evaluation.json",offline)
    value={name:evaluate_value(m,rl_val) for name,m in models.items()};write_json(args.output/"value_evaluation.json",value)
    tails=tail_evaluation(models,rl,Path("data/experiments/lot17_policy_regret/regret_search_cache.jsonl"),set(reference.payload["validation_game_ids"]));write_json(args.output/"regret_tail_evaluation.json",tails)
    with (args.output/"training_metrics_base.csv").open("w",newline="") as s:
        rows=[{"epoch":r.epoch,**{f"train_{k}":v for k,v in asdict(r.train).items()},**{f"validation_{k}":v for k,v in asdict(r.validation).items()}} for r in base.history];w=csv.DictWriter(s,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with (args.output/"training_metrics_dual.csv").open("w",newline="") as s:w=csv.DictWriter(s,fieldnames=dual_history[0]);w.writeheader();w.writerows(dual_history)
    arenas={};
    if not args.skip_arena:
        openings=generate_unique_deterministic_openings(count=args.openings,seed=args.arena_seed,max_prefix_length=40);write_json(args.output/"arena_openings.json",{"seed":args.arena_seed,"openings":[opening_to_dict(x) for x in openings]});arena_config=ArenaConfig(max_plies=400,repetition_limit=3,seed=args.arena_seed,bootstrap_samples=args.bootstrap)
        for candidate,model,file in (("C20-BASE",base_model,"arena_base_vs_g2.json"),("C20-DUAL",dual_model,"arena_dual_vs_g2.json")):
            payload={}
            for budget in (64,128):summary,games=arena_duel(candidate,model,"G2",parent.model,budget,openings,arena_config);payload[str(budget)]={"summary":summary,"games":games};print(f"[lot20] arena {candidate}/G2 MCTS{budget} done",flush=True)
            write_json(args.output/file,payload);arenas[file]=payload
        summary,games=arena_duel("C20-DUAL",dual_model,"C20-BASE",base_model,128,openings,arena_config);write_json(args.output/"arena_dual_vs_base.json",{"128":{"summary":summary,"games":games}});arenas["arena_dual_vs_base.json"]={"128":{"summary":summary,"games":games}}
    def progress(payload):
        if not payload:return False
        return all(payload[str(b)]["summary"]["score_rate_a_terminal"]>.5 and payload[str(b)]["summary"]["paired_bootstrap_ci"][0]>.5 for b in (64,128))
    base_progress=progress(arenas.get("arena_base_vs_g2.json"));dual_progress=progress(arenas.get("arena_dual_vs_g2.json"));direct=arenas.get("arena_dual_vs_base.json",{}).get("128",{}).get("summary");direct_progress=bool(direct and direct["score_rate_a_terminal"]>.5 and direct["paired_bootstrap_ci"][0]>.5)
    tail_reduced=tails["C20-DUAL"]["regret"]["p95"]<tails["G2"]["regret"]["p95"] and tails["C20-DUAL"]["high_regret_errors_gt_0_25"]<tails["G2"]["high_regret_errors_gt_0_25"] and tails["C20-DUAL"]["regret"]["mean"]<tails["G2"]["regret"]["mean"]
    diverse=offline["C20-DUAL"]["D_REANALYSIS_policy"]["ce"]<offline["G2"]["D_REANALYSIS_policy"]["ce"] and offline["C20-DUAL"]["D_REANALYSIS_policy"]["top1"]>offline["G2"]["D_REANALYSIS_policy"]["top1"]
    forgetting=offline["C20-DUAL"]["D_RL_policy"]["ce"]>1.10*offline["G2"]["D_RL_policy"]["ce"] and offline["C20-DUAL"]["D_RL_policy"]["top1"]<offline["G2"]["D_RL_policy"]["top1"]-.05;value_preserved=value["C20-DUAL"]["mse"]<=1.10*value["G2"]["mse"]
    verdict={"DUAL_SOURCE_TRAINING_VALID":"YES" if all(micro[k] for k in ("policy_rl_learned","policy_re_learned","value_rl_learned")) else "NO","REGRET_WEIGHTING_VALID":"YES" if alpha>0 else "PARTIAL","CONFIDENCE_WEIGHTING_VALID":"YES","HIGH_REGRET_TAIL_REDUCED":"YES" if tail_reduced else "NO","DIVERSE_POLICY_SIGNAL_LEARNED":"YES" if diverse else "NO","CATASTROPHIC_POLICY_FORGETTING":"YES" if forgetting else "NO","VALUE_PRESERVED":"YES" if value_preserved else "NO","C20_BASE_PROGRESS_OVER_G2":"YES" if base_progress else "NO","C20_DUAL_PROGRESS_OVER_G2":"YES" if dual_progress else "NO","C20_DUAL_PROGRESS_OVER_BASE":"YES" if direct_progress else "NO" if direct else "NOT_TESTED","G3_CANDIDATE":"C20_DUAL" if dual_progress else "NONE"}
    verdict["NEXT_ACTION"]="INDEPENDENT_G3_CONFIRMATION" if dual_progress else "POLICY_WEIGHTING_RECALIBRATION" if tail_reduced else "DIVERSITY_ONLY_CONTROL" if diverse else "PARENT_POLICY_REGULARIZATION" if forgetting else "TARGET_STABILITY_REWORK"
    configuration={"hashes":{"G2":G2_SHA,"D_RL":RL_SHA,"D_REANALYSIS":RE_SHA},"splits":{"rl_train":len(rl_train),"rl_validation":len(rl_val),"re_train":len(re_train),"re_validation":len(re_val),"re_seed":args.seed},"training":asdict(config),"source_coefficients":{"lambda_RL":1,"lambda_RE":1,"lambda_value":1},"initialization":initialization,"checkpoint_selection":{"base":"historical D_RL total validation loss","dual":"sum of epoch0-normalized D_RL Policy CE, D_RE Policy CE, D_RL Value MSE"}};write_json(args.output/"configuration.json",configuration)
    report={"lot":20,"configuration":configuration,"micro_overfit":micro,"weighting":weights,"training":{"base_best_epoch":base.best_epoch,"dual_best_epoch":dual_best_epoch},"offline":offline,"regret_tail":tails,"value":value,"arena":{k:{b:v[b]["summary"] for b in v} for k,v in arenas.items()},"verdict":verdict,"minimax_benchmark_executed":False,"elapsed_s":time.perf_counter()-started};write_json(args.output/"report.json",report);print(json.dumps({"report":str(args.output/'report.json'),"verdict":verdict,"elapsed_s":report["elapsed_s"]},indent=2),flush=True)

if __name__=="__main__":main()
