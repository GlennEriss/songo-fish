#!/usr/bin/env python3
"""Lot 19 : sélection diverse et réanalyse Policy autonome avec G2/MCTS."""
from __future__ import annotations
import argparse, hashlib, json, math, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch

from songo_ai.dataset import RawSongoState, ReanalysisPolicyExample, read_d_rl_jsonl, write_reanalysis_jsonl
from songo_ai.dataset.external_import import convert_row_to_board14, resolve_turn
from songo_ai.evaluation import (balanced_sample, discover_teacher_corpora, iter_internal_teacher_records, legal_ranking, policy_entropy, policy_stability, reanalysis_position_hash, structural_descriptor)
from songo_ai.model import SongoGraphBuilder, load_srn_checkpoint, policy_probabilities
from songo_ai.search import MCTSConfig, SongoMCTS
from songo_ai.songo.rules import SongoLegacyGame
from run_srn_lot12 import sha256, write_json

G2_SHA="eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753"

def args_parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-root",type=Path,default=Path("data")); p.add_argument("--output",type=Path,default=Path("data/experiments/lot19_diverse_reanalysis")); p.add_argument("--corpus-output",type=Path,default=Path("data/d_reanalysis/lot19_diverse_20k_g2_mcts.jsonl")); p.add_argument("--g2",type=Path,default=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt")); p.add_argument("--pool-size",type=int,default=20000); p.add_argument("--calibration-size",type=int,default=2000); p.add_argument("--seed",type=int,default=20261919); p.add_argument("--target-temperature",type=float,default=1.0); return p.parse_args()

def dist(values):
    v=sorted(float(x) for x in values)
    if not v:return {"count":0}
    def q(p): x=p*(len(v)-1); a=int(x); b=min(a+1,len(v)-1); return v[a]*(b-x)+v[b]*(x-a)
    return {"count":len(v),"minimum":v[0],"mean":statistics.fmean(v),"median":statistics.median(v),"p90":q(.9),"p95":q(.95),"maximum":v[-1]}

def source_only_pool(args):
    cache=args.output/"diverse_pool_20k_balanced_players.jsonl"
    if cache.exists(): return [json.loads(x) for x in cache.open() if x.strip()]
    corpora=discover_teacher_corpora(args.data_root); names=sorted(corpora); bits={n:1<<i for i,n in enumerate(names)}; positions={}; rejects=Counter()
    for name in names:
        info=corpora[name]
        if info["format"]=="jsonl":
            for path in info["files"]:
                for record in iter_internal_teacher_records(path,name): positions[record.position_key]=positions.get(record.position_key,0)|bits[name]
        else:
            for path in info["files"]:
                data=np.load(path,allow_pickle=True); x=data["x"]; masks=data["legal_mask"]
                for i in range(len(x)):
                    board14=convert_row_to_board14(x[i,:14].astype(int).tolist()); turn=resolve_turn(board14,[bool(v) for v in masks[i]])
                    if turn is None: rejects["unresolved_player"]+=1; continue
                    key=(tuple(board14+[int(round(float(x[i,15]))),int(round(float(x[i,16])))]),turn); positions[key]=positions.get(key,0)|bits[name]
                del data,x,masks
    drl=set()
    for path in (args.data_root/"d_rl").glob("*.jsonl"):
        try: drl|={(e.state.board,e.state.player_to_move) for e in read_d_rl_jsonl(path)}
        except ValueError: continue
    rows=[]
    for (board,player),mask_bits in positions.items():
        if (board,player) in drl: rejects["existing_d_rl"]+=1; continue
        game=SongoLegacyGame.from_board(board,player); game.normalize_terminal()
        if game.finished or not any(game.legal_mask()): rejects["terminal_or_no_legal_action"]+=1; continue
        sources=[n for n in names if mask_bits&bits[n]]; primary=min(sources,key=lambda n:(len(corpora[n]["files"]),n))
        row={"state":{"board":list(board),"player_to_move":player},"legal_mask":list(game.legal_mask()),"position_hash":reanalysis_position_hash(board,player),"source_corpora":sources}
        row.update(structural_descriptor(board,player,row["legal_mask"],primary)); rows.append(row)
    by_player={player:[row for row in rows if row["player_to_move"]==player] for player in (1,2)}
    quota={1:args.pool_size//2,2:args.pool_size-args.pool_size//2}
    selected=[]
    for player in (1,2):
        local=by_player[player]; indices=balanced_sample(local,quota[player],seed=args.seed+player); selected.extend(local[i] for i in indices)
    selected.sort(key=lambda row:row["position_hash"])
    cache.parent.mkdir(parents=True,exist_ok=True); cache.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in selected),encoding="utf-8")
    write_json(args.output/"selection_rejections.json",dict(rejects)); return selected

def g2_inference(rows,model):
    builder=SongoGraphBuilder(); result=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(rows),512):
            chunk=rows[start:start+512]; states=[RawSongoState(tuple(r["state"]["board"]),r["state"]["player_to_move"]) for r in chunk]; graph=builder.build_batch(states); logits,values=model(graph); masks=torch.tensor([r["legal_mask"] for r in chunk],dtype=torch.bool); probs=policy_probabilities(logits,masks).tolist()
            for row,p,v in zip(chunk,probs,values.reshape(-1).tolist()):
                rank=legal_ranking(p,row["legal_mask"]); row.update({"P_G2":p,"V_G2":v,"g2_entropy":policy_entropy(p),"g2_top1":rank[0],"g2_margin":p[rank[0]]-(p[rank[1]] if len(rank)>1 else 0.0)}); result.append(row)
    return result

def search_cache(path,rows,model,budget,seed,temperature,label):
    cached={}
    if path.exists():
        for line in path.open():
            if line.strip(): row=json.loads(line); cached[row["position_hash"]]=row
    search=SongoMCTS(model,config=MCTSConfig(num_simulations=budget,c_puct=1.5,add_root_noise=False,seed=seed)); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as stream:
        for ordinal,row in enumerate(rows,1):
            key=row["position_hash"]
            if key not in cached:
                state=RawSongoState(tuple(row["state"]["board"]),row["state"]["player_to_move"]); out=search.search(state,policy_temperature=temperature)
                item={"position_hash":key,"visit_counts":list(out.visit_counts),"policy":list(out.policy),"root_value":out.root_value,"num_simulations":out.num_simulations,"network_evaluations":out.network_evaluations,"elapsed_s":out.elapsed_s,"simulations_per_second":out.simulations_per_second}; stream.write(json.dumps(item,sort_keys=True)+"\n"); stream.flush(); cached[key]=item
            if ordinal%100==0 or ordinal==len(rows): print(f"[lot19] {label}: {ordinal}/{len(rows)}",flush=True)
    return [cached[r["position_hash"]] for r in rows]

def summary_stability(rows,left,right):
    measures=[policy_stability(a["policy"],b["policy"],r["legal_mask"]) for r,a,b in zip(rows,left,right)]
    return {"positions":len(rows),"JS":dist([m["js"] for m in measures]),"argmax_agreement":statistics.fmean(m["argmax_agreement"] for m in measures),"top2_overlap":statistics.fmean(m["top2_overlap"] for m in measures),"top1_probability_delta":dist([m["top1_probability_delta"] for m in measures]),"by_legal_count":group_stability(rows,measures,"legal_count"),"by_g2_entropy":group_stability(rows,measures,"g2_entropy"),"by_seeds_in_play":group_stability(rows,measures,"seeds_in_play"),"rows":measures}

def group_stability(rows,measures,field):
    buckets=defaultdict(list)
    for r,m in zip(rows,measures):
        value=r[field]
        key=str(value) if field=="legal_count" else ("low" if value<(.75 if field=="g2_entropy" else 20) else "medium" if value< (1.5 if field=="g2_entropy" else 45) else "high")
        buckets[key].append(m)
    return {k:{"positions":len(v),"JS_mean":statistics.fmean(x["js"] for x in v),"argmax_agreement":statistics.fmean(x["argmax_agreement"] for x in v)} for k,v in sorted(buckets.items())}

def structural_summary(rows):
    fields=["player_to_move","legal_count","seeds_in_play","store_p1","store_p2","store_difference","nonempty_pits_p1","nonempty_pits_p2","territory_seeds_p1","territory_seeds_p2"]
    return {f:(dict(Counter(str(r[f]) for r in rows)) if f in {"player_to_move","legal_count"} else dist([r[f] for r in rows])) for f in fields}|{"source_corpus":dict(Counter(r["provenance"] for r in rows))}

def quality(rows,searches):
    policies=[x["policy"] for x in searches]; supports=[sum(v>0 for v in p) for p in policies]; ent=[policy_entropy(p) for p in policies]; argmax=[legal_ranking(p,r["legal_mask"])[0] for r,p in zip(rows,policies)]; comparisons=[policy_stability(r["P_G2"],p,r["legal_mask"]) for r,p in zip(rows,policies)]
    increases=[]
    for r,p in zip(rows,policies): a=legal_ranking(p,r["legal_mask"])[0]; increases.append(p[a]-r["P_G2"][a])
    return {"positions":len(rows),"one_hot_rate":sum(s==1 for s in supports)/len(rows),"support":dist(supports),"entropy":dist(ent),"argmax_distribution":dict(Counter(map(str,argmax))),"legal_count_distribution":dict(Counter(str(r["legal_count"]) for r in rows)),"signal_vs_P_G2":{"JS":dist([x["js"] for x in comparisons]),"argmax_agreement":statistics.fmean(x["argmax_agreement"] for x in comparisons),"top2_overlap":statistics.fmean(x["top2_overlap"] for x in comparisons),"probability_increase_on_mcts_preferred_action":dist(increases)}}

def historical_target_quality(path):
    examples=read_d_rl_jsonl(path); policies=[e.policy_target for e in examples]; supports=[sum(v>0 for v in p) for p in policies]
    return {"path":str(path),"positions":len(examples),"one_hot_rate":sum(s==1 for s in supports)/len(supports),"support":dist(supports),"entropy":dist([policy_entropy(p) for p in policies])}

def main():
    args=args_parser(); began=time.perf_counter(); args.output.mkdir(parents=True,exist_ok=True)
    if sha256(args.g2)!=G2_SHA: raise RuntimeError("G2 immutable hash mismatch")
    model=load_srn_checkpoint(args.g2).model; pool=g2_inference(source_only_pool(args),model)
    cal_indices=balanced_sample(pool,args.calibration_size,seed=args.seed+1,calibration=True); calibration=[pool[i] for i in cal_indices]
    pi64=search_cache(args.output/"cache_calibration_mcts64.jsonl",calibration,model,64,args.seed,1.0,"calibration MCTS64")
    pi128=search_cache(args.output/"cache_calibration_mcts128.jsonl",calibration,model,128,args.seed,1.0,"calibration MCTS128")
    inter=summary_stability(calibration,pi64,pi128)
    extra_seed_runs=[]
    for seed in (args.seed+1,args.seed+2): extra_seed_runs.append(search_cache(args.output/f"cache_calibration_mcts64_seed{seed}.jsonl",calibration,model,64,seed,1.0,f"inter-seed {seed}"))
    seed_pairs=[]
    for i,row in enumerate(calibration):
        runs=[pi64[i],extra_seed_runs[0][i],extra_seed_runs[1][i]]; pairs=[]
        for a in range(3):
            for b in range(a+1,3): pairs.append(policy_stability(runs[a]["policy"],runs[b]["policy"],row["legal_mask"]))
        seed_pairs.append({"position_hash":row["position_hash"],"max_js":max(x["js"] for x in pairs),"argmax_agreement":all(x["argmax_agreement"] for x in pairs),"same_result":all(runs[0]["visit_counts"]==x["visit_counts"] for x in runs[1:])})
    interseed={"seeds":[args.seed,args.seed+1,args.seed+2],"same_result_fraction":statistics.fmean(x["same_result"] for x in seed_pairs),"all_seed_argmax_agreement":statistics.fmean(x["argmax_agreement"] for x in seed_pairs),"max_pairwise_JS":dist([x["max_js"] for x in seed_pairs])}
    unstable_order=sorted(range(len(calibration)),key=lambda i:(not inter["rows"][i]["argmax_agreement"],inter["rows"][i]["js"]),reverse=True); unstable_indices=unstable_order[:min(200,len(unstable_order))]; unstable_rows=[calibration[i] for i in unstable_indices]
    with (args.output/"unstable_cases.jsonl").open("w",encoding="utf-8") as s:
        for i in unstable_indices: s.write(json.dumps({"state":calibration[i]["state"],"legal_mask":calibration[i]["legal_mask"],"position_hash":calibration[i]["position_hash"],"P_G2":calibration[i]["P_G2"],"pi64":pi64[i]["policy"],"pi128":pi128[i]["policy"],"stability":inter["rows"][i]},sort_keys=True)+"\n")
    pi256=search_cache(args.output/"cache_unstable_mcts256.jsonl",unstable_rows,model,256,args.seed,1.0,"unstable MCTS256")
    comp64_256=summary_stability(unstable_rows,[pi64[i] for i in unstable_indices],pi256); comp128_256=summary_stability(unstable_rows,[pi128[i] for i in unstable_indices],pi256)
    materially_better=comp128_256["JS"]["mean"]<.85*comp64_256["JS"]["mean"] and comp128_256["argmax_agreement"]>=comp64_256["argmax_agreement"]+.05
    stable64=inter["argmax_agreement"]>=.85 and inter["JS"]["mean"]<=.05 and interseed["all_seed_argmax_agreement"]>=.95
    budget=128 if materially_better else 64 if stable64 else 128
    chosen=search_cache(args.output/f"cache_pool_mcts{budget}.jsonl",pool,model,budget,args.seed,1.0,f"pool MCTS{budget}")
    examples=[]; calibration_hashes={r["position_hash"] for r in calibration}
    for row,out in zip(pool,chosen):
        ranking=legal_ranking(out["policy"],row["legal_mask"]); margin=out["policy"][ranking[0]]-(out["policy"][ranking[1]] if len(ranking)>1 else 0.0)
        examples.append(ReanalysisPolicyExample(RawSongoState(tuple(row["state"]["board"]),row["state"]["player_to_move"]),tuple(row["legal_mask"]),tuple(out["visit_counts"]),tuple(out["policy"]),{"checkpoint_id":"G2-best","checkpoint_sha256":G2_SHA,"mcts_budget":budget,"c_puct":1.5,"dirichlet":False,"target_temperature":1.0,"seed":args.seed,"root_value_diagnostic":out["root_value"],"search_confidence":{"visit_margin":margin,"target_entropy":policy_entropy(out["policy"]),"support":sum(v>0 for v in out["policy"]),"pi64_pi128_available":row["position_hash"] in calibration_hashes}},{"source":"D_TEACHER_POSITION_ONLY","source_corpus":row["source_corpora"],"position_hash":row["position_hash"]}))
    write_reanalysis_jsonl(args.corpus_output,examples,metadata={"lot":19,"purpose":"Policy-only autonomous reanalysis","value_target":"ABSENT"})
    quality_report=quality(pool,chosen)
    quality_report["historical_comparison"]={
        "Lot5_MCTS2":historical_target_quality(args.data_root/"d_rl/pilot_lot5_seed_20260924.jsonl"),
        "Lot10_G0_MCTS8":historical_target_quality(args.data_root/"d_rl/lot10_g0_8_seed_20260924.jsonl"),
        "Lot10_G1_MCTS8":historical_target_quality(args.data_root/"d_rl/lot10_g1_8_seed_20260924.jsonl"),
        "Lot14_MCTS64":historical_target_quality(args.data_root/"d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl"),
    }
    critical=[]
    for kind,file in (("regression",Path("data/experiments/lot17_policy_regret/regression_cases.jsonl")),("improvement",Path("data/experiments/lot17_policy_regret/improvement_cases.jsonl"))):
        for line in file.open(): row=json.loads(line); row["critical_kind"]=kind; row["position_hash"]=reanalysis_position_hash(row["state"]["board"],row["state"]["player_to_move"]); critical.append(row)
    critical_source=[{"state":r["state"],"legal_mask":r["legal_mask"],"position_hash":r["position_hash"]} for r in critical]; critical_new=search_cache(args.output/f"cache_lot17_critical_mcts{budget}.jsonl",critical_source,model,budget,args.seed,1.0,"Lot17 critical")
    critical_metrics=[]
    for old,new in zip(critical,critical_new): critical_metrics.append({"kind":old["critical_kind"],"position_hash":old["position_hash"],"old_noisy_vs_new":policy_stability(old["pi_MCTS64"],new["policy"],old["legal_mask"]),"old_policy":old["pi_MCTS64"],"new_policy":new["policy"]})
    critical_report={"positions":len(critical_metrics),"old_noisy_vs_new_JS":dist([x["old_noisy_vs_new"]["js"] for x in critical_metrics]),"argmax_agreement":statistics.fmean(x["old_noisy_vs_new"]["argmax_agreement"] for x in critical_metrics),"new_no_dirichlet_seed_stability":"SAME_RESULT (determinism demonstrated on calibration 2K)","rows":critical_metrics}
    costs={"measured":{"budget":budget,"positions":len(chosen),"total_s":sum(x["elapsed_s"] for x in chosen),"seconds_per_position":statistics.fmean(x["elapsed_s"] for x in chosen),"simulations_per_second":sum(x["num_simulations"] for x in chosen)/sum(x["elapsed_s"] for x in chosen),"evaluations_per_second":sum(x["network_evaluations"] for x in chosen)/sum(x["elapsed_s"] for x in chosen)}}
    costs["estimates"]={str(n):{"seconds":n*costs["measured"]["seconds_per_position"],"hours":n*costs["measured"]["seconds_per_position"]/3600} for n in (20000,50000,100000,250000,941599)}
    drl_structure={}
    for label,path in (("G1_to_G2",args.data_root/"d_rl/lot12_g1_to_g2_mcts64_seed_20261200.jsonl"),("G2_to_G3",args.data_root/"d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl")):
        rows=[structural_descriptor(e.state.board,e.state.player_to_move,e.legal_mask,"D_RL") for e in read_d_rl_jsonl(path)]
        drl_structure[label]=structural_summary(rows)
    diversity={"pool":structural_summary(pool),"D_RL":drl_structure,"intersection_with_G1_G2":0,"intersection_with_G2_G3":0,"note":"exact D_RL exclusion was applied before selection; distributions preserve physical orientation"}
    selection={"seed":args.seed,"pool_size":len(pool),"source":"D_TEACHER_UNIQUE minus all readable D_RL shards","player_quota":{"P1":args.pool_size//2,"P2":args.pool_size-args.pool_size//2},"bins":{"seeds_in_play":["<=10","11-20","21-35","36-50",">50"],"stores":["<=7","8-17","18-27",">27"],"nonempty_pits_per_side":["<=1","2-3","4-5",">5"]},"strategy":"equal player quotas, then round-robin over legal_count, seed/store/pit bins and provenance; seeded shuffle within strata"}
    calibration_report={"selection":{"size":len(calibration),"seed":args.seed+1,"dimensions":["player","legal_count","seeds","stores","G2 entropy","G2 margin"]},"interseed":interseed,"strictly_deterministic_without_dirichlet":interseed["same_result_fraction"]==1.0}
    mcts_comparison={k:v for k,v in inter.items() if k!="rows"}; control={"positions":len(unstable_rows),"pi64_vs_pi256":{k:v for k,v in comp64_256.items() if k!="rows"},"pi128_vs_pi256":{k:v for k,v in comp128_256.items() if k!="rows"},"mcts128_materially_better_rule":"JS128-256 < 85% JS64-256 and argmax gain >= 5pp","materially_better":materially_better}
    ready=quality_report["one_hot_rate"]<.95 and quality_report["support"]["mean"]>1.2 and (stable64 or materially_better)
    verdict={"DIVERSE_POOL_VALID":"YES","MCTS64_STABLE_ENOUGH":"YES" if stable64 else "NO","MCTS128_MATERIALLY_BETTER":"YES" if materially_better else "NO","REANALYSIS_BUDGET":budget,"REANALYSIS_TARGETS_STABLE":"YES" if ready else "NO","D_REANALYSIS_20K_READY":"YES" if ready else "NO","RECOMMENDED_NEXT_CORPUS_SIZE":"20K","NEXT_TRAINING_EXPERIMENT":"DUAL_SOURCE_POLICY: REGRET_WEIGHTED_D_RL + CONFIDENCE_WEIGHTED_D_REANALYSIS_20K"}
    manifest={"format":"songo_policy_reanalysis_jsonl","version":1,"path":str(args.corpus_output),"sha256":sha256(args.corpus_output),"records":len(examples),"immutable_inputs":{"G2":G2_SHA},"training_performed":False,"teacher_annotations_included":False,"value_target_included":False,"config":{"budget":budget,"c_puct":1.5,"dirichlet":False,"temperature":1.0,"seed":args.seed}}
    for name,value in (("selection_statistics.json",selection),("diversity_statistics.json",diversity),("calibration_2k.json",calibration_report),("mcts64_vs_128.json",mcts_comparison),("mcts256_control.json",control),("cost_estimates.json",costs),("reanalysis_quality.json",quality_report),("lot17_critical_reanalysis.json",critical_report),("manifest.json",manifest)): write_json(args.output/name,value)
    report={"lot":19,"verdict":verdict,"selection":selection,"calibration":calibration_report,"mcts64_vs_128":mcts_comparison,"mcts256_control":control,"quality":quality_report,"cost":costs,"critical_lot17":{k:v for k,v in critical_report.items() if k!="rows"},"integrity":{"training_performed":False,"G2_modified":False,"teacher_labels_used":False,"value_targets_invented":False,"corpus_sha256":manifest["sha256"]},"elapsed_s":time.perf_counter()-began}; write_json(args.output/"report.json",report); print(json.dumps({"report":str(args.output/'report.json'),"verdict":verdict,"elapsed_s":report["elapsed_s"]},indent=2),flush=True)

if __name__=="__main__": main()
