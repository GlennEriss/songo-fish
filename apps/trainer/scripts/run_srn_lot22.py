#!/usr/bin/env python3
"""Lot 22 : diagnostic sans entraînement de l'objectif Policy."""
from __future__ import annotations

import argparse,hashlib,json,math,random,statistics,time
from collections import Counter,defaultdict
from pathlib import Path

import torch

from songo_ai.dataset import iter_reanalysis_jsonl,read_d_rl_jsonl
from songo_ai.evaluation import HybridPolicyValueEvaluator,correlations,diagnostic_action_values,jensen_shannon,legal_ranking,model_parameter_fingerprint,policy_cross_entropy,policy_entropy,top_margin
from songo_ai.evaluation.policy_objective import classify_flip,decision_regret,kendall_tau_from_rankings,legal_rank_map,pairwise_inversions,search_amplification,strategically_weighted_inversions,top_k_set
from songo_ai.model import load_srn_checkpoint,policy_probabilities
from songo_ai.search import MCTSConfig,SongoMCTS

from run_srn_lot12 import sha256,write_json
from run_srn_lot20 import G2_SHA,RE_SHA,RL_SHA,collate,dist,pos_hash,split_data


MODEL_PATHS={
    "G2":"data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt",
    "G3-A":"data/experiments/lot14_g3_seed_20261402/training/best_validation_checkpoint.pt",
    "G3-B":"data/experiments/lot15_g3b_seed_20261515/training/best_validation_checkpoint.pt",
    "C20-DUAL":"data/experiments/lot20_dual_source_policy/checkpoints/c20_dual_best.pt",
    "C21-B":"data/experiments/lot21_policy_recalibration/checkpoints/c21_b_best.pt",
}


def args_parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--d-rl",type=Path,default=Path("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl"));p.add_argument("--d-re",type=Path,default=Path("data/d_reanalysis/lot19_diverse_20k_g2_mcts.jsonl"));p.add_argument("--split-reference",type=Path,default=Path("data/experiments/lot14_g3_seed_20261402/training/best_validation_checkpoint.pt"));p.add_argument("--output",type=Path,default=Path("data/experiments/lot22_policy_objective"));p.add_argument("--seed",type=int,default=20262222);p.add_argument("--q-count",type=int,default=1200);p.add_argument("--q-budget",type=int,default=256);p.add_argument("--stability-count",type=int,default=200);p.add_argument("--sensitivity-count",type=int,default=100);return p.parse_args()


def policies(models,examples):
    result={n:[] for n in models}
    for m in models.values():m.eval()
    with torch.no_grad():
        for start in range(0,len(examples),512):
            b=collate(examples[start:start+512])
            for name,model in models.items():logits,_=model(b.graph);result[name].extend(policy_probabilities(logits,b.legal_mask).tolist())
    return result


def l1(a,b):return sum(abs(x-y) for x,y in zip(a,b))
def kl(a,b):return sum(x*math.log(x/max(y,1e-12)) for x,y in zip(a,b) if x>0)


def margin_bin(x):return "very_low" if x<.02 else "low" if x<.05 else "medium" if x<.15 else "high"
def seed_bin(n):return "0_15" if n<=15 else "16_30" if n<=30 else "31_50" if n<=50 else "51_plus"


def build_battery(re_val,rl,lot17_cache,model_policies):
    entries={}
    for e in re_val:entries[pos_hash(e.state)]={"example":e,"sources":{"D_REANALYSIS_VALIDATION"},"mcts128_target":True,"lot17_critical":False}
    critical=set()
    critical_path=Path("data/experiments/lot17_policy_regret/regression_cases.jsonl")
    if critical_path.exists():
        for line in critical_path.open():critical.add(int(json.loads(line)["index"]))
    for line in lot17_cache.open():
        row=json.loads(line);idx=int(row["index"]);e=rl[idx];key=pos_hash(e.state)
        if key not in entries:entries[key]={"example":e,"sources":set(),"mcts128_target":False,"lot17_critical":False}
        entries[key]["sources"].add("LOT17_BATTERY");entries[key]["lot17_critical"]|=idx in critical
    battery=list(entries.values());battery.sort(key=lambda x:pos_hash(x["example"].state))
    for i,item in enumerate(battery):
        item["battery_index"]=i;item["position_hash"]=pos_hash(item["example"].state);item["sources"]=sorted(item["sources"])
    return battery


def ranking_and_distribution(battery,pol):
    names=list(pol);rows=[]
    for i,item in enumerate(battery):
        e=item["example"];mask=e.legal_mask;g=pol["G2"][i];gr=legal_ranking(g,mask);seeds=sum(e.state.board[:14]);stores=list(e.state.board[14:16]);row={"battery_index":i,"position_hash":item["position_hash"],"sources":item["sources"],"lot17_critical":item["lot17_critical"],"player_to_move":e.state.player_to_move,"legal_count":sum(mask),"seeds_in_play":seeds,"stores":stores,"nonempty_pits":sum(x>0 for x in e.state.board[:14]),"g2_entropy":policy_entropy(g),"g2_margin":top_margin(g,mask),"g2_margin_bin":margin_bin(top_margin(g,mask)),"seed_bin":seed_bin(seeds),"rankings":{},"comparisons":{}}
        for name in names:
            p=pol[name][i];r=legal_ranking(p,mask);row["rankings"][name]={"ranking":list(r),"top1":r[0],"top2":list(r[:2]),"top1_top2_margin":p[r[0]]-p[r[1]] if len(r)>1 else 1.,"top1_top3_margin":p[r[0]]-p[r[2]] if len(r)>2 else 1.}
        for name in names:
            if name=="G2":continue
            p=pol[name][i];r=legal_ranking(p,mask);inv=pairwise_inversions(g,p,mask);row["comparisons"][name]={"js":jensen_shannon(g,p),"kl_g2_x":kl(g,p),"kl_x_g2":kl(p,g),"l1":l1(g,p),"argmax_changed":r[0]!=gr[0],"top2_set_changed":top_k_set(g,mask,2)!=top_k_set(p,mask,2),"kendall_tau":kendall_tau_from_rankings(gr,r),"pairwise_inversions":len(inv),"g2_top1_rank_in_x":legal_rank_map(p,mask)[gr[0]],"x_top1_rank_in_g2":legal_rank_map(g,mask)[r[0]],"probability_change_x_top1":p[r[0]]-g[r[0]],"margin_change":top_margin(p,mask)-top_margin(g,mask)}
            if item["mcts128_target"]:row["comparisons"][name]["ce_mcts128"]=policy_cross_entropy(e.policy_target,p)
        rows.append(row)
    return rows


def summarize_rows(rows,names):
    distribution_metrics={};ranking_metrics={}
    for name in names:
        c=[r["comparisons"][name] for r in rows];target=[x["ce_mcts128"] for x in c if "ce_mcts128" in x]
        distribution_metrics[name]={"positions":len(c),"mcts128_positions":len(target),"ce_mcts128":dist(target),"js_from_g2":dist([x["js"] for x in c]),"kl_g2_x":dist([x["kl_g2_x"] for x in c]),"kl_x_g2":dist([x["kl_x_g2"] for x in c]),"l1":dist([x["l1"] for x in c])}
        ranking_metrics[name]={"argmax_change_fraction":statistics.fmean(x["argmax_changed"] for x in c),"top2_change_fraction":statistics.fmean(x["top2_set_changed"] for x in c),"kendall_tau":dist([x["kendall_tau"] for x in c]),"pairwise_inversions":dist([x["pairwise_inversions"] for x in c]),"g2_top1_rank_in_x":dict(Counter(str(x["g2_top1_rank_in_x"]) for x in c)),"x_top1_rank_in_g2":dict(Counter(str(x["x_top1_rank_in_g2"]) for x in c))}
    return distribution_metrics,ranking_metrics


def stratified_q_indices(rows,count,seed):
    groups=defaultdict(list)
    for r in rows:
        changed=any(r["comparisons"][n]["argmax_changed"] for n in r["comparisons"]);key=(r["player_to_move"],r["legal_count"],r["g2_margin_bin"],r["sources"][0],changed,r["lot17_critical"]);groups[key].append(r["battery_index"])
    rng=random.Random(seed)
    for v in groups.values():rng.shuffle(v)
    keys=sorted(groups,key=repr);selected=[];cursor=0
    while len(selected)<min(count,len(rows)):
        progress=False
        for key in keys:
            if cursor<len(groups[key]):selected.append(groups[key][cursor]);progress=True
            if len(selected)>=count:break
        if not progress:break
        cursor+=1
    return selected


def qdiag_cache(path,indices,battery,g2,budget,seed):
    cached={}
    if path.exists():
        for line in path.open():row=json.loads(line);cached[row["position_hash"]]=row
    with path.open("a") as stream:
        for ordinal,index in enumerate(indices,1):
            item=battery[index];key=item["position_hash"]
            if key not in cached:
                result=diagnostic_action_values(item["example"].state,g2,num_simulations=budget,seed=seed+index);row={"position_hash":key,"battery_index":index,"budget":budget,**result};stream.write(json.dumps(row)+"\n");stream.flush();cached[key]=row
            if ordinal%50==0 or ordinal==len(indices):print(f"[lot22] Qdiag{budget}: {ordinal}/{len(indices)}",flush=True)
    return {i:cached[battery[i]["position_hash"]] for i in indices}


def concentration(values):
    v=sorted((x for x in values if x>0),reverse=True);total=sum(v)
    return {"positive_count":len(v),"total":total,"top_1_percent_share":sum(v[:max(1,math.ceil(.01*len(v)))])/total if total else 0,"top_5_percent_share":sum(v[:max(1,math.ceil(.05*len(v)))])/total if total else 0,"top_10_percent_share":sum(v[:max(1,math.ceil(.10*len(v)))])/total if total else 0}


def auc(scores,labels):
    pos=[s for s,y in zip(scores,labels) if y];neg=[s for s,y in zip(scores,labels) if not y]
    if not pos or not neg:return None
    return sum(1 if p>n else .5 if p==n else 0 for p in pos for n in neg)/(len(pos)*len(neg))


def q_analysis(indices,battery,rows,pol,qdiag,names,epsilon=.02):
    flips={};harmful_lines=[];beneficial_lines=[];swi={};action_regret={}
    for name in names:
        records=[];all_reg=[];all_swi=[]
        for i in indices:
            e=battery[i]["example"];q=qdiag[i]["q_values"];g=pol["G2"][i];p=pol[name][i];ga=legal_ranking(g,e.legal_mask)[0];xa=legal_ranking(p,e.legal_mask)[0];rg=decision_regret(q,e.legal_mask,ga);rx=decision_regret(q,e.legal_mask,xa);delta=rx-rg;kind=classify_flip(delta,epsilon) if xa!=ga else "NO_FLIP";weighted=strategically_weighted_inversions(g,p,e.legal_mask,q);record={"position_hash":battery[i]["position_hash"],"battery_index":i,"model":name,"g2_action":ga,"candidate_action":xa,"argmax_changed":xa!=ga,"classification":kind,"regret_g2":rg,"regret_candidate":rx,"delta_regret":delta,"q_values":q,"policy_js":rows[i]["comparisons"][name]["js"],"policy_l1":rows[i]["comparisons"][name]["l1"],"top2_changed":rows[i]["comparisons"][name]["top2_set_changed"],"g2_top1_rank_in_x":rows[i]["comparisons"][name]["g2_top1_rank_in_x"],"x_top1_rank_in_g2":rows[i]["comparisons"][name]["x_top1_rank_in_g2"],"pairwise_inversions":rows[i]["comparisons"][name]["pairwise_inversions"],"swi":weighted,"g2_margin":rows[i]["g2_margin"],"legal_count":rows[i]["legal_count"]};records.append(record);all_reg.append(rx);all_swi.append(weighted)
            if kind=="HARMFUL_FLIP":harmful_lines.append(record)
            elif kind=="BENEFICIAL_FLIP":beneficial_lines.append(record)
        changed=[r for r in records if r["argmax_changed"]];harm=[r for r in changed if r["classification"]=="HARMFUL_FLIP"];benef=[r for r in changed if r["classification"]=="BENEFICIAL_FLIP"];neutral=[r for r in changed if r["classification"]=="NEUTRAL_FLIP"];labels=[r["classification"]=="HARMFUL_FLIP" for r in changed]
        metric_scores={"js":[r["policy_js"] for r in changed],"l1":[r["policy_l1"] for r in changed],"argmax_change":[1. for _ in changed],"top2_change":[float(r["top2_changed"]) for r in changed],"rank_drop":[float(r["g2_top1_rank_in_x"]-1) for r in changed],"pairwise_inversions":[r["pairwise_inversions"] for r in changed],"swi":[r["swi"] for r in changed],"g2_margin":[r["g2_margin"] for r in changed]}
        flips[name]={"epsilon":epsilon,"changed":len(changed),"beneficial":len(benef),"neutral":len(neutral),"harmful":len(harm),"fractions":{"beneficial":len(benef)/len(changed) if changed else 0,"neutral":len(neutral)/len(changed) if changed else 0,"harmful":len(harm)/len(changed) if changed else 0},"harmful_delta_regret":dist([r["delta_regret"] for r in harm]),"harmful_cost_concentration":concentration([r["delta_regret"] for r in harm]),"metric_harmful_auc":{k:auc(v,labels) for k,v in metric_scores.items()},"metric_correlations":correlations({**metric_scores,"delta_regret":[r["delta_regret"] for r in changed]}) if changed else {},"by_g2_margin":group_flip(changed,"g2_margin"),"by_legal_count":group_flip(changed,"legal_count"),"top2":{"same_top2":group_values([r for r in changed if not r["top2_changed"]]),"changed_top2":group_values([r for r in changed if r["top2_changed"]])},"records":records}
        swi[name]={"distribution":dist(all_swi),"correlation_with_decision_regret":correlations({"swi":all_swi,"decision_regret":all_reg})};action_regret[name]={"decision_regret":dist(all_reg),"delta_vs_g2":dist([r["delta_regret"] for r in records])}
    return flips,action_regret,swi,harmful_lines,beneficial_lines


def group_values(records):return {"count":len(records),"delta_regret":dist([r["delta_regret"] for r in records]),"harmful_fraction":statistics.fmean(r["classification"]=="HARMFUL_FLIP" for r in records) if records else None}
def group_flip(records,key):
    out=defaultdict(list)
    for r in records:out[margin_bin(r[key]) if key=="g2_margin" else str(r[key])].append(r)
    return {k:group_values(v) for k,v in sorted(out.items())}


def search_sensitivity(indices,battery,models,pol,seed):
    result={};amplified=[];g2=models["G2"]
    for ordinal,i in enumerate(indices,1):
        e=battery[i]["example"];row={"position_hash":battery[i]["position_hash"],"budgets":{}}
        for budget in (8,32,64,128,256):
            searches={}
            for name in ("G2","C21-B"):
                evaluator=HybridPolicyValueEvaluator(models[name],g2,name=f"Policy-{name}+Value-G2");r=SongoMCTS(evaluator,config=MCTSConfig(num_simulations=budget,c_puct=1.5,add_root_noise=False,seed=seed+i)).search(e.state,policy_temperature=1.);searches[name]={"policy":list(r.policy),"selected_action":r.selected_action,"visit_counts":list(r.visit_counts)}
            pjs=jensen_shannon(pol["G2"][i],pol["C21-B"][i]);vjs=jensen_shannon(searches["G2"]["policy"],searches["C21-B"]["policy"]);amp=search_amplification(pjs,vjs,searches["G2"]["selected_action"]!=searches["C21-B"]["selected_action"]);row["budgets"][str(budget)]={"G2":searches["G2"],"C21-B":searches["C21-B"],**amp}
            if pjs<.001 and (vjs>.05 or amp["selected_action_changed"]):amplified.append({"position_hash":row["position_hash"],"budget":budget,**amp})
        result[row["position_hash"]]=row
        if ordinal%20==0 or ordinal==len(indices):print(f"[lot22] search sensitivity: {ordinal}/{len(indices)}",flush=True)
    summary={str(b):{"visit_js":dist([r["budgets"][str(b)]["visit_js"] for r in result.values()]),"selected_change_fraction":statistics.fmean(r["budgets"][str(b)]["selected_action_changed"] for r in result.values()),"amplification_ratio":dist([r["budgets"][str(b)]["ratio"] for r in result.values()])} for b in (8,32,64,128,256)}
    return {"positions":len(indices),"hybrid":"candidate Policy + frozen G2 Value","summary":summary,"rows":list(result.values())},amplified


def stability(indices,battery,g2,q256,seed):
    rows=[]
    for ordinal,i in enumerate(indices,1):
        q512=diagnostic_action_values(battery[i]["example"].state,g2,num_simulations=512,seed=seed+i);mask=battery[i]["example"].legal_mask;legal=[a for a,x in enumerate(mask) if x];r256=sorted(legal,key=lambda a:(-q256[i]["q_values"][a],a));r512=sorted(legal,key=lambda a:(-q512["q_values"][a],a));gaps=[abs(q256[i]["q_values"][a]-q512["q_values"][a]) for a in legal];rows.append({"position_hash":battery[i]["position_hash"],"top1_agreement":r256[0]==r512[0],"kendall_tau":kendall_tau_from_rankings(r256,r512),"max_q_change":max(gaps),"q256":q256[i]["q_values"],"q512":q512["q_values"]})
        if ordinal%20==0 or ordinal==len(indices):print(f"[lot22] Q stability: {ordinal}/{len(indices)}",flush=True)
    return {"positions":len(rows),"top1_agreement":statistics.fmean(r["top1_agreement"] for r in rows),"kendall_tau":dist([r["kendall_tau"] for r in rows]),"max_q_change":dist([r["max_q_change"] for r in rows]),"rows":rows}


def main():
    args=args_parser();started=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True)
    if sha256(args.d_rl)!=RL_SHA or sha256(args.d_re)!=RE_SHA or sha256(Path(MODEL_PATHS["G2"]))!=G2_SHA:raise RuntimeError("immutable input mismatch")
    checkpoints={n:load_srn_checkpoint(Path(p)) for n,p in MODEL_PATHS.items()};models={n:x.model for n,x in checkpoints.items()};before={n:model_parameter_fingerprint(m) for n,m in models.items()};rl=list(read_d_rl_jsonl(args.d_rl));re=list(iter_reanalysis_jsonl(args.d_re));reference=load_srn_checkpoint(args.split_reference);_,_,_,re_val=split_data(rl,re,reference,20262020)
    battery=build_battery(re_val,rl,Path("data/experiments/lot17_policy_regret/regret_search_cache.jsonl"),None);examples=[x["example"] for x in battery];pol=policies(models,examples);rows=ranking_and_distribution(battery,pol);names=[n for n in models if n!="G2"]
    manifest={"positions":len(rows),"unique_positions":len({r["position_hash"] for r in rows}),"source_counts":dict(Counter(s for r in rows for s in r["sources"])),"player_to_move":dict(Counter(str(r["player_to_move"]) for r in rows)),"legal_count":dict(Counter(str(r["legal_count"]) for r in rows)),"seed_bins":dict(Counter(r["seed_bin"] for r in rows)),"g2_margin_bins":dict(Counter(r["g2_margin_bin"] for r in rows)),"model_argmax_changed":{n:sum(r["comparisons"][n]["argmax_changed"] for r in rows) for n in names},"selection":"all fixed D_RE validation + independent Lot17 Q-diagnostic positions, exact-state deduplication","positions_manifest":[{k:r[k] for k in ("battery_index","position_hash","sources","player_to_move","legal_count","seed_bin","g2_margin_bin","lot17_critical")} for r in rows]};write_json(args.output/"diagnostic_battery.json",manifest)
    distribution_metrics,ranking_metrics=summarize_rows(rows,names);write_json(args.output/"distribution_metrics.json",distribution_metrics);write_json(args.output/"ranking_metrics.json",ranking_metrics)
    q_indices=stratified_q_indices(rows,args.q_count,args.seed);qdiag=qdiag_cache(args.output/"qdiag_cache.jsonl",q_indices,battery,models["G2"],args.q_budget,args.seed);flips,action_regret,swi,harmful,beneficial=q_analysis(q_indices,battery,rows,pol,qdiag,names);write_json(args.output/"argmax_flip_analysis.json",{n:{k:v for k,v in x.items() if k!="records"} for n,x in flips.items()});write_json(args.output/"action_regret.json",action_regret);write_json(args.output/"harmful_flip_analysis.json",{n:{"harmful":x["harmful"],"harmful_delta_regret":x["harmful_delta_regret"],"harmful_cost_concentration":x["harmful_cost_concentration"],"metric_harmful_auc":x["metric_harmful_auc"],"by_g2_margin":x["by_g2_margin"],"by_legal_count":x["by_legal_count"],"top2":x["top2"]} for n,x in flips.items()});write_json(args.output/"strategic_weighted_inversions.json",swi)
    for path,data in (("harmful_flips.jsonl",harmful),("beneficial_flips.jsonl",beneficial)):(args.output/path).write_text("".join(json.dumps(x)+"\n" for x in data))
    critical=sorted(q_indices,key=lambda i:max((next((r["delta_regret"] for r in flips[n]["records"] if r["battery_index"]==i),0) for n in names)),reverse=True);sensitivity_indices=critical[:args.sensitivity_count];search,amplified=search_sensitivity(sensitivity_indices,battery,models,pol,args.seed+1);write_json(args.output/"search_sensitivity.json",search);write_json(args.output/"search_amplification.json",{"definition_candidates":["visit JS / Policy JS","selected-action change after MCTS","low Policy JS with high visit JS"],"cases":len(amplified),"summary_by_budget":search["summary"]});(args.output/"search_amplified_cases.jsonl").write_text("".join(json.dumps(x)+"\n" for x in amplified))
    stable_indices=critical[:args.stability_count];stable=stability(stable_indices,battery,models["G2"],qdiag,args.seed+2);write_json(args.output/"qdiag_stability.json",stable)
    # Agrégation explicative : classement qualitatif, sans oracle absolu.
    known_strength={"G3-A":"rejected","G3-B":"rejected","C20-DUAL":"28.13% vs G2 MCTS128","C21-B":"35.16% vs G2 short MCTS128"};comparison={}
    for n in names:comparison[n]={"known_play_strength":known_strength[n],"distribution":distribution_metrics[n],"ranking":ranking_metrics[n],"flips":{k:v for k,v in flips[n].items() if k not in ("records","metric_correlations")},"action_regret":action_regret[n],"swi":swi[n]}
    write_json(args.output/"model_comparison.json",comparison)
    harmful_shares=[flips[n]["harmful_cost_concentration"]["top_10_percent_share"] for n in names if flips[n]["harmful"]];shape="HEAVY_TAIL" if harmful_shares and statistics.median(harmful_shares)>=.5 else "MIXED" if harmful_shares and statistics.median(harmful_shares)>=.3 else "DIFFUSE"
    auc_swi=[flips[n]["metric_harmful_auc"]["swi"] for n in names if flips[n]["metric_harmful_auc"]["swi"] is not None];auc_js=[flips[n]["metric_harmful_auc"]["js"] for n in names if flips[n]["metric_harmful_auc"]["js"] is not None];pairwise_more=statistics.fmean(auc_swi)>statistics.fmean(auc_js)+.05 if auc_swi and auc_js else False;top2_harm=[]
    for n in names:
        same=flips[n]["top2"]["same_top2"];changed=flips[n]["top2"]["changed_top2"]
        if same["harmful_fraction"] is not None and changed["harmful_fraction"] is not None:top2_harm.append(changed["harmful_fraction"]>same["harmful_fraction"]+.05)
    amp256=search["summary"]["256"];amplifies=bool(amplified) and (amp256["selected_change_fraction"]>=.05 or amp256["visit_js"]["p95"]>=.01)
    verdict={"DISTRIBUTIONAL_METRICS_EXPLAIN_FAILURE":"PARTIAL" if auc_js and statistics.fmean(auc_js)>.55 else "NO","ARGMAX_FLIPS_EXPLAIN_FAILURE":"YES" if all(flips[n]["harmful"]>0 for n in names) else "PARTIAL","TOP2_PRESERVATION_MATTERS":"YES" if top2_harm and statistics.fmean(top2_harm)>=.75 else "INCONCLUSIVE","PAIRWISE_RANKING_MORE_INFORMATIVE_THAN_CE":"YES" if pairwise_more else "NO","STRATEGIC_GAP_MATTERS":"YES" if auc_swi and statistics.fmean(auc_swi)>=.65 else "INCONCLUSIVE","SEARCH_AMPLIFIES_POLICY_ERRORS":"YES" if amplifies else "NO","POLICY_FAILURE_SHAPE":shape,"CURRENT_POLICY_OBJECTIVE_ADEQUATE":"NO","RECOMMENDED_POLICY_OBJECTIVE":"STRATEGICALLY_WEIGHTED_RANKING" if pairwise_more and auc_swi and statistics.fmean(auc_swi)>=.65 else "DISTRIBUTION_PLUS_RANKING","NEXT_ACTION":"CONTROLLED_POLICY_OBJECTIVE_EXPERIMENT"}
    recommendation={"recommended":verdict["RECOMMENDED_POLICY_OBJECTIVE"],"not_implemented":True,"requirements":["legal-only pairs","tolerate near-equivalent Qdiag actions","prioritize large strategic gaps","retain a small distributional term only if ranking alone is insufficient","Qdiag remains a noisy autonomous estimate, not an oracle"],"evidence":{"mean_harmful_auc_js":statistics.fmean(auc_js) if auc_js else None,"mean_harmful_auc_swi":statistics.fmean(auc_swi) if auc_swi else None,"qdiag_top1_stability_256_512":stable["top1_agreement"]}};write_json(args.output/"future_loss_recommendation.json",recommendation)
    after={n:model_parameter_fingerprint(m) for n,m in models.items()};integrity={"training_performed":False,"optimizer_created":False,"checkpoints_created":False,"weights_unchanged":before==after,"teacher_used":False,"minimax_used":False,"hashes":{"G2":G2_SHA,"D_RL":RL_SHA,"D_REANALYSIS":RE_SHA},"model_fingerprints":before}
    report={"lot":22,"integrity":integrity,"battery":{k:v for k,v in manifest.items() if k!="positions_manifest"},"qdiag":{"sample_size":len(q_indices),"budget_per_legal_action":args.q_budget,"epsilon":.02,"method":"force legal action with frozen engine, then G2/MCTS on successor with perspective conversion"},"distribution_metrics":distribution_metrics,"ranking_metrics":ranking_metrics,"argmax_flips":{n:{k:v for k,v in x.items() if k not in ("records","metric_correlations")} for n,x in flips.items()},"search_sensitivity":search["summary"],"qdiag_stability":{k:v for k,v in stable.items() if k!="rows"},"verdict":verdict,"scientific_question":"YES" if verdict["ARGMAX_FLIPS_EXPLAIN_FAILURE"]=="YES" and verdict["STRATEGIC_GAP_MATTERS"]=="YES" else "INCONCLUSIVE","elapsed_s":time.perf_counter()-started};write_json(args.output/"report.json",report);print(json.dumps({"report":str(args.output/"report.json"),"verdict":verdict,"scientific_question":report["scientific_question"],"elapsed_s":report["elapsed_s"]},indent=2),flush=True)


if __name__=="__main__":main()
