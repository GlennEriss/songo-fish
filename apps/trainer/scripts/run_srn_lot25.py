#!/usr/bin/env python3
"""Lot 25 : construction shardée et reprenable de D_SCALE_V1, sans entraînement."""
from __future__ import annotations

import argparse,json,math,statistics,time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from songo_ai.dataset import RawSongoState,read_d_rl_jsonl,write_d_rl_jsonl,validate_reanalysis_record
from songo_ai.dataset.external_import import convert_row_to_board14,resolve_turn
from songo_ai.evaluation import balanced_sample,diagnostic_action_values,discover_teacher_corpora,iter_internal_teacher_records,policy_entropy,policy_stability,reanalysis_position_hash,structural_descriptor
from songo_ai.model import SongoGraphBuilder,load_srn_checkpoint,policy_probabilities
from songo_ai.generation import SelfPlayConfig,SelfPlayRunner
from songo_ai.search import MCTSConfig,SongoMCTS
from songo_ai.songo.rules import SongoLegacyGame
from run_srn_lot12 import sha256,write_json

G2_SHA="eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("selfplay","select","reanalysis","stability","strategic","summarize"),default="selfplay");p.add_argument("--g2",type=Path,default=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt"));p.add_argument("--data-root",type=Path,default=Path("data"));p.add_argument("--output",type=Path,default=Path("data/experiments/lot25_scale"));p.add_argument("--dataset-root",type=Path,default=Path("data/d_scale_v1"));p.add_argument("--games",type=int,default=5000);p.add_argument("--games-per-shard",type=int,default=100);p.add_argument("--reanalysis-positions",type=int,default=100000);p.add_argument("--positions-per-shard",type=int,default=1000);p.add_argument("--stability-positions",type=int,default=1000);p.add_argument("--strategic-positions",type=int,default=2000);p.add_argument("--strategic-stability",type=int,default=200);p.add_argument("--seed",type=int,default=20262525);return p.parse_args()


def config(games,seed):
    return SelfPlayConfig(games=games,max_game_plies=400,repetition_limit=3,target_temperature=1.,action_temperature=1.,temperature_drop_ply=30,late_action_temperature=0.,seed=seed,generation=25,checkpoint_id=f"G2-best-{G2_SHA[:12]}",provenance={"source_dataset":"SELFPLAY","generation_lineage":"G2_to_D_SCALE_V1","generator":"G2-best","generator_checkpoint_sha256":G2_SHA,"lot":25},include_truncated_examples=True,mcts=MCTSConfig(num_simulations=64,c_puct=1.5,dirichlet_alpha=.3,dirichlet_epsilon=.25,add_root_noise=True))


def selfplay(args):
    if sha256(args.g2)!=G2_SHA:raise RuntimeError("G2 immutable hash mismatch")
    model=load_srn_checkpoint(args.g2).model;root=args.dataset_root/"d_selfplay_large";root.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True);planned=(args.games+args.games_per_shard-1)//args.games_per_shard;rows=[]
    for shard in range(planned):
        games=min(args.games_per_shard,args.games-shard*args.games_per_shard);path=root/f"part-{shard:05d}.jsonl";meta_path=root/f"part-{shard:05d}.manifest.json";seed=args.seed+shard
        if path.exists() and meta_path.exists():meta=json.load(meta_path.open());print(f"[lot25] selfplay shard {shard+1}/{planned} reused",flush=True)
        else:
            began=time.perf_counter();run=SelfPlayRunner(model,config(games,seed)).generate();written=write_d_rl_jsonl(path,run.examples);meta={"shard":shard,"games":games,"examples":written,"seed":seed,"sha256":sha256(path),"statistics":asdict(run.statistics),"elapsed_s":time.perf_counter()-began};write_json(meta_path,meta);print(f"[lot25] selfplay shard {shard+1}/{planned}: {games} games, {written} positions",flush=True)
        rows.append(meta)
    manifest={"dataset":"D_SELFPLAY_LARGE","format":"sharded_jsonl","generator":"G2-best","generator_sha256":G2_SHA,"games":sum(x["games"] for x in rows),"positions":sum(x["examples"] for x in rows),"mcts":{"simulations":64,"c_puct":1.5,"dirichlet_alpha":.3,"dirichlet_epsilon":.25,"root_noise":True},"temperature":{"early":1.,"drop_ply":30,"late":0.},"shards":rows,"teacher_labels_used":False,"minimax_used":False};write_json(args.output/"selfplay_manifest.json",manifest);return manifest


def summarize(args):
    paths=sorted((args.dataset_root/"d_selfplay_large").glob("part-*.jsonl"));examples=[e for p in paths for e in read_d_rl_jsonl(p)];keys=[(e.state.board,e.state.player_to_move) for e in examples];lengths={}
    for e in examples:lengths[e.metadata["game_id"]]=max(lengths.get(e.metadata["game_id"],0),int(e.metadata["ply"])+1)
    result={"positions":len(examples),"unique_positions":len(set(keys)),"repeated_positions":len(examples)-len(set(keys)),"duplication_rate":1-len(set(keys))/len(examples),"games":len(lengths),"game_length":{"mean":statistics.fmean(lengths.values()),"median":statistics.median(lengths.values())},"player_distribution":{"P1":sum(e.state.player_to_move==1 for e in examples),"P2":sum(e.state.player_to_move==2 for e in examples)}};write_json(args.output/"selfplay_statistics.json",result);return result


def select_positions(args):
    """Déduplique le réservoir physique avant tout échantillonnage."""
    args.output.mkdir(parents=True,exist_ok=True);cache=args.output/f"selected_positions_{args.reanalysis_positions}.jsonl"
    if cache.exists():return {"selected":sum(1 for _ in cache.open()),"path":str(cache),"reused":True}
    corpora=discover_teacher_corpora(args.data_root);names=sorted(corpora);bits={n:1<<i for i,n in enumerate(names)};positions={};rejects=Counter()
    for name in names:
        info=corpora[name]
        if info["format"]=="jsonl":
            for path in info["files"]:
                for record in iter_internal_teacher_records(path,name):positions[record.position_key]=positions.get(record.position_key,0)|bits[name]
        else:
            for path in info["files"]:
                data=np.load(path,allow_pickle=True);x=data["x"];masks=data["legal_mask"]
                for i in range(len(x)):
                    board14=convert_row_to_board14(x[i,:14].astype(int).tolist());turn=resolve_turn(board14,[bool(v) for v in masks[i]])
                    if turn is None:rejects["unresolved_player"]+=1;continue
                    key=(tuple(board14+[int(round(float(x[i,15]))),int(round(float(x[i,16])))]),turn);positions[key]=positions.get(key,0)|bits[name]
                del data,x,masks
    old=set()
    for path in (args.data_root/"d_rl").glob("*.jsonl"):
        try:old|={(e.state.board,e.state.player_to_move) for e in read_d_rl_jsonl(path)}
        except ValueError:continue
    rows=[]
    for (board,player),mask_bits in positions.items():
        game=SongoLegacyGame.from_board(board,player);game.normalize_terminal()
        if game.finished or not any(game.legal_mask()):rejects["terminal_or_no_legal"]+=1;continue
        sources=[n for n in names if mask_bits&bits[n]];primary=min(sources)
        row={"state":{"board":list(board),"player_to_move":player},"legal_mask":list(game.legal_mask()),"position_hash":reanalysis_position_hash(board,player),"source_corpora":sources,"new_vs_old_d_rl":(board,player) not in old};row.update(structural_descriptor(board,player,row["legal_mask"],primary));rows.append(row)
    if args.reanalysis_positions>len(rows):raise ValueError("requested sample exceeds reservoir")
    selected=[rows[i] for i in balanced_sample(rows,args.reanalysis_positions,seed=args.seed)];selected.sort(key=lambda x:x["position_hash"])
    with cache.open("w") as out:
        for row in selected:out.write(json.dumps(row,sort_keys=True)+"\n")
    report={"raw_records":sum(len(v.get("files",[])) for v in corpora.values()),"unique_physical_reservoir":len(positions),"eligible_positions":len(rows),"selected":len(selected),"new_vs_old_d_rl":sum(x["new_vs_old_d_rl"] for x in selected),"seed":args.seed,"teacher_annotations_used":False,"rejections":dict(rejects),"path":str(cache)};write_json(args.output/"selection_manifest.json",report);return report


def reanalyse(args):
    if sha256(args.g2)!=G2_SHA:raise RuntimeError("G2 immutable hash mismatch")
    selected=args.output/f"selected_positions_{args.reanalysis_positions}.jsonl"
    if not selected.exists():select_positions(args)
    rows=[json.loads(x) for x in selected.open() if x.strip()];model=load_srn_checkpoint(args.g2).model;search=SongoMCTS(model,config=MCTSConfig(num_simulations=128,c_puct=1.5,add_root_noise=False,seed=args.seed));root=args.dataset_root/"d_reanalysis_large";root.mkdir(parents=True,exist_ok=True);parts=[]
    for start in range(0,len(rows),args.positions_per_shard):
        subset=rows[start:start+args.positions_per_shard];part=start//args.positions_per_shard;path=root/f"part-{part:05d}.jsonl";meta_path=root/f"part-{part:05d}.manifest.json"
        if path.exists() and meta_path.exists():meta=json.load(meta_path.open());print(f"[lot25] reanalysis shard {part+1}: reused",flush=True)
        else:
            began=time.perf_counter();simulations=evaluations=0
            with path.open("w") as out:
                for row in subset:
                    s=row["state"];result=search.search(RawSongoState(tuple(s["board"]),s["player_to_move"]),policy_temperature=1.);record={"state":s,"legal_mask":row["legal_mask"],"visit_counts":list(result.visit_counts),"policy_target":list(result.policy),"source_dataset":"REANALYSIS","source_position_reservoir":"D_TEACHER_POSITION_ONLY","source_corpora":row["source_corpora"],"position_hash":row["position_hash"],"generation_model":"G2-best","generation_model_sha256":G2_SHA,"mcts_budget":128,"c_puct":1.5,"dirichlet":False,"seed":args.seed,"value_target_present":False};validate_reanalysis_record(record,expected_visits=128);out.write(json.dumps(record,sort_keys=True)+"\n");simulations+=result.num_simulations;evaluations+=result.network_evaluations
            meta={"shard":part,"positions":len(subset),"sha256":sha256(path),"simulations":simulations,"network_evaluations":evaluations,"elapsed_s":time.perf_counter()-began};write_json(meta_path,meta);print(f"[lot25] reanalysis shard {part+1}: {len(subset)} positions",flush=True)
        parts.append(meta)
    manifest={"dataset":"D_REANALYSIS_LARGE","positions":sum(x["positions"] for x in parts),"policy_only":True,"teacher_labels_used":False,"generator":"G2-best","mcts_budget":128,"shards":parts};write_json(args.output/"reanalysis_manifest.json",manifest);return manifest


def stability(args):
    source=args.dataset_root/"d_reanalysis_large";records=[]
    for path in sorted(source.glob("part-*.jsonl")):
        for line in path.open():records.append(json.loads(line))
        if len(records)>=args.stability_positions:break
    records=records[:args.stability_positions]
    if len(records)<args.stability_positions:raise RuntimeError("insufficient completed reanalysis records")
    model=load_srn_checkpoint(args.g2).model;search=SongoMCTS(model,config=MCTSConfig(num_simulations=256,c_puct=1.5,add_root_noise=False,seed=args.seed+256));cache=args.output/"stability_mcts256.jsonl";known={}
    if cache.exists():
        for line in cache.open():row=json.loads(line);known[row["position_hash"]]=row
    with cache.open("a") as out:
        for i,row in enumerate(records,1):
            key=row["position_hash"]
            if key not in known:
                s=row["state"];r=search.search(RawSongoState(tuple(s["board"]),s["player_to_move"]),policy_temperature=1.);known[key]={"position_hash":key,"policy":list(r.policy),"visit_counts":list(r.visit_counts),"elapsed_s":r.elapsed_s};out.write(json.dumps(known[key],sort_keys=True)+"\n");out.flush()
            if i%100==0:print(f"[lot25] stability {i}/{len(records)}",flush=True)
    measures=[policy_stability(r["policy_target"],known[r["position_hash"]]["policy"],r["legal_mask"]) for r in records]
    report={"positions":len(records),"argmax_agreement":statistics.fmean(x["argmax_agreement"] for x in measures),"top2_agreement":statistics.fmean(x["top2_overlap"] for x in measures),"js_mean":statistics.fmean(x["js"] for x in measures),"visit_entropy_comparison":"reported in target_quality stage","budgets":[128,256]};write_json(args.output/"reanalysis_stability.json",report);return report


def strategic(args):
    """Sélectionne les états ambigus/divergents puis produit Qdiag256 autonome."""
    records=[]
    for path in sorted((args.dataset_root/"d_reanalysis_large").glob("part-*.jsonl")):
        records.extend(json.loads(x) for x in path.open() if x.strip())
    model=load_srn_checkpoint(args.g2).model;builder=SongoGraphBuilder();scored=[];hard_hashes=_hard_hashes(args)
    for start in range(0,len(records),512):
        chunk=records[start:start+512];states=[RawSongoState(tuple(r["state"]["board"]),r["state"]["player_to_move"]) for r in chunk];graph=builder.build_batch(states);masks=np.asarray([r["legal_mask"] for r in chunk],dtype=bool)
        import torch
        with torch.no_grad():logits,_=model(graph);probs=policy_probabilities(logits,torch.tensor(masks)).tolist()
        for row,p in zip(chunk,probs):
            legal=[i for i,x in enumerate(row["legal_mask"]) if x];m=row["policy_target"];g1=max(legal,key=lambda i:p[i]);m1=max(legal,key=lambda i:m[i]);rank=sorted(legal,key=lambda i:m[i],reverse=True);margin=m[rank[0]]-(m[rank[1]] if len(rank)>1 else 0);score=(2 if g1!=m1 else 0)+policy_entropy(m)+(1-margin)+(.5 if row["position_hash"] in hard_hashes else 0);scored.append((score,row))
    scored.sort(key=lambda x:(-x[0],x[1]["position_hash"]));chosen=[r for _,r in scored[:args.strategic_positions]];root=args.dataset_root/"d_strategic_sample";root.mkdir(parents=True,exist_ok=True);path=root/"qdiag256.jsonl";cached={}
    if path.exists():
        for line in path.open():r=json.loads(line);cached[r["position_hash"]]=r
    with path.open("a") as out:
        for i,row in enumerate(chosen,1):
            key=row["position_hash"]
            if key not in cached:
                s=row["state"];state=RawSongoState(tuple(s["board"]),s["player_to_move"]);q=diagnostic_action_values(state,model,num_simulations=256,seed=args.seed+i);item={"state":s,"legal_mask":row["legal_mask"],"position_hash":key,"source_dataset":"STRATEGIC","generation_model":"G2-best","qdiag":True,"qdiag_budget_per_action":256,"dirichlet":False,**q};out.write(json.dumps(item,sort_keys=True)+"\n");out.flush();cached[key]=item
            if i%100==0:print(f"[lot25] strategic Qdiag256 {i}/{len(chosen)}",flush=True)
    manifest={"dataset":"D_STRATEGIC_SAMPLE","positions":len(chosen),"qdiag_budget_per_action":256,"generator":"G2-best","teacher_labels_used":False,"value_target_used":False,"sha256":sha256(path),"path":str(path),"selection":"G2/MCTS divergence + MCTS entropy + low margin + HARD_REANALYSIS"};write_json(args.output/"strategic_manifest.json",manifest);return manifest


def _hard_hashes(args):
    path=args.output/"stability_mcts256.jsonl"
    if not path.exists():return set()
    return {json.loads(x)["position_hash"] for x in path.open() if x.strip()}


def main():
    args=parse_args();result={"selfplay":selfplay,"select":select_positions,"reanalysis":reanalyse,"stability":stability,"strategic":strategic,"summarize":summarize}[args.stage](args);print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
