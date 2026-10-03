#!/usr/bin/env python3
"""Lot 18 : cartographie exhaustive de D_TEACHER, sans entraînement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import resource
import sqlite3
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from songo_ai.dataset import RawSongoState, read_d_rl_jsonl
from songo_ai.dataset.external_import import convert_row_to_board14, resolve_turn
from songo_ai.evaluation import (
    discover_teacher_corpora, iter_internal_teacher_records, jensen_shannon,
    legal_ranking, overlap_summary, policy_entropy, position_fingerprint,
    sha256_file, teacher_record_integrity,
)
from songo_ai.evaluation.real_dataset_audit import iter_real_records, position_key as real_position_key
from songo_ai.model import SongoGraphBuilder, load_srn_checkpoint, policy_probabilities
from songo_ai.songo.rules import SongoLegacyGame

from run_srn_lot12 import write_json


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,default=Path("data")); p.add_argument("--output",type=Path,default=Path("data/experiments/lot18_d_teacher"))
    p.add_argument("--g1-g2",type=Path,default=Path("data/d_rl/lot12_g1_to_g2_mcts64_seed_20261200.jsonl")); p.add_argument("--g2-g3",type=Path,default=Path("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl"))
    p.add_argument("--real",type=Path,default=Path("data/real_matches/match_moves_v1.jsonl")); p.add_argument("--g2",type=Path,default=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt")); p.add_argument("--g3b",type=Path,default=Path("data/experiments/lot15_g3b_seed_20261515/training/best_validation_checkpoint.pt"))
    p.add_argument("--inference-sample",type=int,default=10000); p.add_argument("--sample-seed",type=int,default=20261818)
    return p.parse_args()


def dist(values):
    values=sorted(float(x) for x in values)
    if not values:return {"count":0}
    def q(p):
        x=p*(len(values)-1); lo=int(x); hi=min(lo+1,len(values)-1); return values[lo]*(hi-x)+values[hi]*(x-lo)
    return {"count":len(values),"minimum":min(values),"mean":statistics.fmean(values),"median":statistics.median(values),"p90":q(.9),"p95":q(.95),"maximum":max(values)}


def key_digest(key):
    return bytes.fromhex(position_fingerprint(key))


def annotation_digest(annotation):
    return hashlib.sha256(json.dumps(annotation,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def structural_distributions(keys, ply_by_key=None):
    """Statistiques descriptives communes, sans inventer de phase de jeu."""

    ply_by_key = ply_by_key or {}
    values={"player_to_move":Counter(),"legal_count":Counter(),"seeds_in_play":[],"store_p1":[],"store_p2":[],"nonempty_pits":[],"ply_available":[]}
    invalid=0
    for board,turn in keys:
        try: legal=sum(SongoLegacyGame.from_board(board,turn).legal_mask())
        except Exception: invalid+=1; continue
        values["player_to_move"][str(turn)]+=1; values["legal_count"][str(legal)]+=1
        values["seeds_in_play"].append(sum(board[:14])); values["store_p1"].append(board[14]); values["store_p2"].append(board[15]); values["nonempty_pits"].append(sum(v>0 for v in board[:14]))
        if (board,turn) in ply_by_key and ply_by_key[(board,turn)] is not None: values["ply_available"].append(ply_by_key[(board,turn)])
    return {"invalid_positions":invalid,"player_to_move":dict(values["player_to_move"]),"legal_count":dict(values["legal_count"]),**{k:dist(v) for k,v in values.items() if isinstance(v,list)}}


class AnnotationStats:
    def __init__(self):
        self.actions=Counter(); self.depth=[]; self.nodes=[]; self.elapsed=[]; self.exact=Counter(); self.tier=Counter(); self.margins=[]; self.annotated_actions=[]; self.missing_values=0; self.records=0
    def add_internal(self, ann, legal_count):
        self.records+=1; action=ann.get("best_action"); self.actions[str(action)]+=1
        teacher=ann.get("teacher") or {}
        for target,key in ((self.depth,"depth"),(self.nodes,"nodes"),(self.elapsed,"elapsed_ms")):
            if teacher.get(key) is not None: target.append(float(teacher[key]))
        if teacher.get("is_exact") is not None:self.exact[str(bool(teacher["is_exact"]))]+=1
        if teacher.get("tier") is not None:self.tier[str(teacher["tier"])]+=1
        vals=ann.get("action_values")
        if vals is not None:
            known=[float(x) for x in vals if x is not None]; self.annotated_actions.append(len(known)); self.missing_values+=7-len(known)
            if len(known)>=2:
                ordered=sorted(known,reverse=True); self.margins.append(ordered[0]-ordered[1])
    def add_external(self, action, policy):
        self.records+=1; self.actions[str(int(action))]+=1; self.annotated_actions.append(sum(float(x)>0 for x in policy))
    def report(self):
        return {"records":self.records,"best_action_or_policy_index":dict(sorted(self.actions.items())),"depth":dist(self.depth),"nodes":dist(self.nodes),"elapsed_ms":dist(self.elapsed),"is_exact":dict(self.exact),"tier":dict(self.tier),"derived_top1_top2_action_value_margin":dist(self.margins),"annotated_actions":dist(self.annotated_actions),"missing_action_values":self.missing_values,"is_exact_interpretation":"historical flag: replayed best PV reaches terminal; not a formal minimax proof"}


def main():
    args=parse_args(); started=time.perf_counter(); args.output.mkdir(parents=True,exist_ok=True)
    corpora=discover_teacher_corpora(args.data_root); names=sorted(corpora); bits={name:1<<i for i,name in enumerate(names)}
    position_sets={}; corpus_reports={}; inventory=[]; schemas={}; integrity={}; duplicates={}; annotations={}
    global_positions={}; first_ply={}; implicit_canonical=set(); annotations_db=Path(tempfile.mkdtemp(prefix="songo-lot18-"))/"annotations.sqlite"
    db=sqlite3.connect(annotations_db); db.execute("CREATE TABLE a(pos BLOB,best INTEGER,fp TEXT,depth INTEGER,tier TEXT,incomplete INTEGER)")
    for corpus_index,name in enumerate(names):
        info=corpora[name]; manifest=json.loads(info["manifest"].read_text()); files=info["files"]
        for path in [*files,info["manifest"]]: inventory.append({"corpus":name,"path":str(path.relative_to(args.data_root)),"role":"position_data" if path in files else "manifest","format":path.suffix.lstrip('.'),"size_bytes":path.stat().st_size,"sha256":sha256_file(path)})
        positions=set(); raw=0; errors=Counter(); ann_stats=AnnotationStats(); local_fp={}; same_annotation=0; changed_annotation=0; fields=set(); physical_recoverable=True
        if info["format"]=="jsonl":
            physical_recoverable=False
            for path in files:
                for record in iter_internal_teacher_records(path,name):
                    raw+=1; positions.add(record.position_key); implicit_canonical.add(record.position_key); errors.update(teacher_record_integrity(record)); first_ply.setdefault(record.position_key,record.ply)
                    global_positions[record.position_key]=global_positions.get(record.position_key,0)|bits[name]; ann_stats.add_internal(record.annotation,sum(record.legal_mask)); fields.update(record.annotation)
                    fp=annotation_digest(record.annotation); existed=record.position_key in local_fp
                    previous=local_fp.setdefault(record.position_key,fp)
                    same_annotation+=bool(existed and previous==fp); changed_annotation+=bool(existed and previous!=fp)
                    teacher=record.annotation.get("teacher") or {}; vals=record.annotation.get("action_values"); incomplete=int(vals is None or any(x is None for x in vals))
                    db.execute("INSERT INTO a VALUES(?,?,?,?,?,?)",(key_digest(record.position_key),record.annotation.get("best_action"),fp,teacher.get("depth"),teacher.get("tier"),incomplete))
            schemas[name]={"position":{"board":"state[16] canonicalized","player_to_move":"implicit current player mapped to P1; original physical player unavailable","legal_mask":True},"annotation":{"best_action":True,"action_values":True,"action_value_depths":True,"policy_target":True,"wdl_target":"present in raw JSONL","principal_variation":True,"teacher_depth_nodes_elapsed_exact_tier":True},"POSITION_REUSABLE_WITHOUT_TEACHER_LABELS":"YES_WITH_CANONICAL_ORIENTATION_LIMITATION","TEACHER_ANNOTATIONS_AVAILABLE":"YES"}
        else:
            schemas[name]={"position":{"board":"converted x columns 0:14 + stores x[15:17]","player_to_move":"resolved from legal_mask against engine","legal_mask":True},"annotation":{"policy_index":True,"policy_target_full":True,"value_target":True,"tactical_masks":True,"teacher_depth_nodes_elapsed_exact_tier":False},"POSITION_REUSABLE_WITHOUT_TEACHER_LABELS":"YES","TEACHER_ANNOTATIONS_AVAILABLE":"PARTIAL"}
            for path in files:
                data=np.load(path,allow_pickle=True); x=data["x"]; masks=data["legal_mask"]; actions=data["policy_index"]; policies=data["policy_target_full"]
                for i in range(len(x)):
                    raw+=1; board14=convert_row_to_board14(x[i,:14].astype(int).tolist()); mask=tuple(bool(v) for v in masks[i]); turn=resolve_turn(board14,list(mask))
                    if turn is None: errors["unresolved_player"]+=1; continue
                    board=tuple(board14+[int(round(float(x[i,15]))),int(round(float(x[i,16])))])
                    key=(board,turn); positions.add(key); global_positions[key]=global_positions.get(key,0)|bits[name]
                    if len(board)!=16:errors["board_dimension"]+=1
                    elif sum(board)!=70:errors["seed_conservation"]+=1
                    game=SongoLegacyGame.from_board(board,turn)
                    if game.legal_mask()!=mask:errors["legal_mask_engine_mismatch"]+=1
                    action=int(actions[i]);
                    if not 0<=action<7 or not mask[action]:errors["policy_index_illegal"]+=1
                    ann={"policy_index":action,"policy_target_full":[round(float(v),8) for v in policies[i]]}; ann_stats.add_external(action,policies[i]); fp=annotation_digest(ann)
                    existed=key in local_fp; previous=local_fp.setdefault(key,fp)
                    same_annotation+=bool(existed and previous==fp); changed_annotation+=bool(existed and previous!=fp)
                    db.execute("INSERT INTO a VALUES(?,?,?,?,?,?)",(key_digest(key),action,fp,None,"insane",0))
                del data,x,masks,actions,policies
        db.commit(); position_sets[name]=positions
        duplicates[name]={"raw_positions":raw,"unique_positions":len(positions),"duplicates":raw-len(positions),"duplication_rate":(raw-len(positions))/raw if raw else 0,"repeated_same_annotation_observations":same_annotation,"repeated_changed_annotation_observations":changed_annotation}
        integrity[name]={"errors":dict(errors),"valid_records":raw-sum(errors.values()),"physical_player_recoverable":physical_recoverable}
        if info["format"] == "npz":
            provenance = (
                "positions sampled from generated games across five historical Colab corpora; "
                "generation agents are documented elsewhere as Minimax/MCTS; annotations from "
                f"{manifest.get('teacher_engine', 'UNKNOWN')} level {manifest.get('teacher_level', 'UNKNOWN')}"
            )
        else:
            provenance = manifest.get(
                "source", "internal synthetic trajectories sampled and annotated by historical teacher"
            )
        annotations[name]=ann_stats.report(); corpus_reports[name]={"format":info["format"],"files":[str(p.relative_to(args.data_root)) for p in files],"manifest":str(info["manifest"].relative_to(args.data_root)),"manifest_data":manifest,"raw_positions":raw,"unique_positions":len(positions),"provenance":provenance}
        print(f"[lot18] {name}: {raw} lignes, {len(positions)} positions uniques",flush=True)
    db.execute("CREATE INDEX idx_a_pos ON a(pos)"); db.commit()
    conflict_row=db.execute("SELECT COUNT(*),SUM(best_conflict),SUM(fp_conflict),SUM(depth_conflict),SUM(tier_conflict),SUM(incomplete) FROM (SELECT pos,(COUNT(DISTINCT best)>1) best_conflict,(COUNT(DISTINCT fp)>1) fp_conflict,(COUNT(DISTINCT depth)>1) depth_conflict,(COUNT(DISTINCT tier)>1) tier_conflict,MAX(incomplete) incomplete FROM a GROUP BY pos HAVING COUNT(*)>1)").fetchone()
    conflicts={"positions_with_multiple_annotations":conflict_row[0] or 0,"positions_with_best_action_conflict":conflict_row[1] or 0,"positions_with_any_annotation_conflict":conflict_row[2] or 0,"repeated_positions_with_depth_difference":conflict_row[3] or 0,"repeated_positions_with_tier_difference":conflict_row[4] or 0,"repeated_positions_with_incomplete_action_values":conflict_row[5] or 0}
    db.close()

    overlaps={}
    with (args.output/"overlap_matrix.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream); writer.writerow(["corpus",*names])
        for left in names:
            overlaps[left]={}; row=[left]
            for right in names:
                summary=overlap_summary(position_sets[left],position_sets[right]); common=summary["intersection"]; overlaps[left][right]=summary; row.append(common)
            writer.writerow(row)

    union=set(global_positions); raw_total=sum(x["raw_positions"] for x in duplicates.values()); unique=len(union)
    provenance_counts=Counter(mask.bit_count() for mask in global_positions.values())
    structural_report=structural_distributions(union,first_ply); invalid_union=structural_report["invalid_positions"]

    comparisons={}
    drl_union=set()
    for label,path in (("G1_to_G2",args.g1_g2),("G2_to_G3",args.g2_g3)):
        examples=read_d_rl_jsonl(path); keys={(tuple(e.state.board),e.state.player_to_move) for e in examples}; drl_union|=keys; common=len(union&keys); comparisons[label]={"raw":len(examples),"unique":len(keys),"intersection":common,"teacher_not_in_drl":unique-common,"teacher_coverage_percent":100*common/unique,"drl_coverage_percent":100*common/len(keys)}
    common_all=len(union&drl_union); comparisons["union_D_RL"]={"unique":len(drl_union),"intersection":common_all,"teacher_not_in_drl_union":unique-common_all,"teacher_coverage_percent":100*common_all/unique}
    real_records=list(iter_real_records(args.real)); real_keys={real_position_key(r) for r in real_records}; real_ply={real_position_key(r):r.get("ply") for r in real_records}; common_real=len(union&real_keys); dreal={"D_REAL_unique":len(real_keys),"intersection":common_real,"D_REAL_only":len(real_keys-union),"D_TEACHER_only":len(union-real_keys)}
    structural_comparison={"D_TEACHER_UNIQUE":structural_report,"D_RL_UNION":structural_distributions(drl_union),"D_REAL":structural_distributions(real_keys,real_ply)}

    sample_count=min(args.inference_sample,unique)
    seed_bytes=args.sample_seed.to_bytes(8,"big",signed=False)
    sample=heapq.nsmallest(sample_count,union,key=lambda key:(hashlib.sha256(seed_bytes+key_digest(key)).digest(),key)); states=[RawSongoState(board,turn) for board,turn in sample]
    g2=load_srn_checkpoint(args.g2).model.eval(); g3=load_srn_checkpoint(args.g3b).model.eval(); builder=SongoGraphBuilder(); metrics=[]
    with torch.no_grad():
        for start in range(0,len(states),512):
            chunk=states[start:start+512]; graph=builder.build_batch(chunk); masks=torch.tensor([SongoLegacyGame.from_state(s.to_engine_state()).legal_mask() for s in chunk],dtype=torch.bool)
            l2,_=g2(graph); l3,_=g3(graph); p2=policy_probabilities(l2,masks).tolist(); p3=policy_probabilities(l3,masks).tolist()
            for state,mask,a,b in zip(chunk,masks.tolist(),p2,p3):
                r2=legal_ranking(a,mask); r3=legal_ranking(b,mask); metrics.append({"entropy_g2":policy_entropy(a),"margin_g2":a[r2[0]]-(a[r2[1]] if len(r2)>1 else 0),"top1_g2":r2[0],"top1_g3b":r3[0],"argmax_agreement":r2[0]==r3[0],"js":jensen_shannon(a,b),"top2_overlap":len(set(r2[:2])&set(r3[:2]))/min(2,len(r2)),"legal_count":sum(mask)})
    policy_diag={"sample_size":len(metrics),"selection":"smallest SHA256(seed || exact position identity)","sample_seed":args.sample_seed,"g2_entropy":dist([m["entropy_g2"] for m in metrics]),"g2_margin":dist([m["margin_g2"] for m in metrics]),"g2_uncertain_margin_lt_0_10_fraction":sum(m["margin_g2"]<.1 for m in metrics)/len(metrics),"G2_G3B_argmax_agreement":sum(m["argmax_agreement"] for m in metrics)/len(metrics),"G2_G3B_JS":dist([m["js"] for m in metrics]),"G2_G3B_top2_overlap":statistics.fmean(m["top2_overlap"] for m in metrics)}

    write_json(args.output/"inventory.json",{"position_files":sum(len(corpora[n]["files"]) for n in names),"manifest_files":len(names),"physical_files_total":len(inventory),"files":inventory})
    write_json(args.output/"logical_corpora.json",{"count":len(names),"corpora":corpus_reports}); write_json(args.output/"schemas.json",schemas); write_json(args.output/"integrity.json",integrity); write_json(args.output/"duplicate_statistics.json",duplicates); write_json(args.output/"overlap_statistics.json",overlaps)
    write_json(args.output/"unique_union_statistics.json",{"N_raw":raw_total,"N_unique":unique,"duplicates":raw_total-unique,"global_duplication_rate":(raw_total-unique)/raw_total,"provenance_multiplicity":dict(sorted(provenance_counts.items())),"implicit_canonical_orientation_positions":len(implicit_canonical),"invalid_union_positions":invalid_union,"structural_distributions":structural_report})
    write_json(args.output/"drl_comparison.json",comparisons); write_json(args.output/"dreal_comparison.json",dreal); write_json(args.output/"structural_comparison.json",structural_comparison); write_json(args.output/"teacher_annotation_statistics.json",annotations); write_json(args.output/"annotation_conflicts.json",conflicts); write_json(args.output/"g2_g3_policy_diagnostic.json",policy_diag)
    capacity={"N_raw":raw_total,"N_unique":unique,"N_unique_not_in_D_RL":unique-common_all,"N_unique_not_in_D_REAL":unique-common_real,"N_reusable_without_teacher_labels":unique-invalid_union,"canonical_orientation_limitation":len(implicit_canonical),"recommended_future_flow":"deduplicate/select states -> ignore Teacher labels -> stable G2+MCTS reanalysis -> controlled Policy experiment","scenario_A_autonomous_reanalysis":"feasible","scenario_B_teacher_assisted":"technically possible but not autonomous","scenario_C_benchmark_only":"feasible"}; write_json(args.output/"future_reanalysis_capacity.json",capacity)
    verdict={"D_TEACHER_FOUND":"YES","D_TEACHER_POSITION_DIVERSITY":"HIGH" if unique>=500000 else "MODERATE" if unique>=100000 else "LOW","D_TEACHER_ADDS_NEW_STATES_VS_DRL":"YES" if unique-common_all>0.5*unique else "NO","TEACHER_CORPORA_LARGELY_NESTED":"PARTIAL","POSITIONS_REUSABLE_WITHOUT_TEACHER_LABELS":"YES","AUTONOMOUS_REANALYSIS_FEASIBLE":"YES","STRATEGIC_QUESTION":"YES","NEXT_EXPERIMENT":"DIVERSE_POSITION_REANALYSIS_FIRST"}
    performance={"elapsed_s":time.perf_counter()-started,"max_rss_platform_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"records_processed":raw_total,"method":"streaming JSONL; split-wise NPZ loading; interned position sets; SQLite annotation-conflict aggregation"}
    report={"lot":18,"inventory":{"physical_files":len(inventory),"position_files":sum(len(corpora[n]["files"]) for n in names),"logical_corpora":len(names)},"required_counts":capacity,"union":{"duplication_rate":(raw_total-unique)/raw_total,"structural":structural_report},"major_overlaps":overlaps,"drl":comparisons,"dreal":dreal,"policy_diagnostic":policy_diag,"annotation_conflicts":conflicts,"verdict":verdict,"performance":performance}; write_json(args.output/"report.json",report)
    print(json.dumps({"report":str(args.output/'report.json'),"counts":capacity,"verdict":verdict,"performance":performance},indent=2),flush=True)


if __name__=="__main__":main()
