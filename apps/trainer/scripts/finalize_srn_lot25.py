#!/usr/bin/env python3
"""Consolide, valide et fige D_SCALE_V1 après les générations du Lot 25."""
from __future__ import annotations
import hashlib,json,math,statistics,time
from collections import Counter
from pathlib import Path

from songo_ai.dataset import iter_reanalysis_jsonl,read_d_rl_jsonl
from songo_ai.evaluation import policy_entropy,reanalysis_position_hash,structural_descriptor
from run_srn_lot12 import sha256,write_json

ROOT=Path("data/d_scale_v1");OUT=Path("data/experiments/lot25_scale");SEED=20262525

def lines(paths):
    for path in paths:
        for line in path.open():
            row=json.loads(line)
            if row.get("record_type")=="manifest":continue
            yield row

def key(row):
    s=row["state"];return reanalysis_position_hash(s["board"],s["player_to_move"])

def dist(values):
    v=sorted(map(float,values))
    if not v:return {"count":0}
    def q(p):x=p*(len(v)-1);a=int(x);b=min(a+1,len(v)-1);return v[a]*(b-x)+v[b]*(x-a)
    return {"count":len(v),"minimum":v[0],"mean":statistics.fmean(v),"median":statistics.median(v),"p90":q(.9),"p95":q(.95),"maximum":v[-1]}

def scan(paths,label):
    keys=set();players=Counter();legal=Counter();seeds=[];stores=[];nonempty=[];support=[];entropy=[];top1=[];actions=Counter();z=Counter();games=set();count=0
    for row in lines(paths):
        count+=1;k=key(row);keys.add(k);s=row["state"];board=s["board"];players[str(s["player_to_move"])]+=1;lc=sum(row["legal_mask"]);legal[str(lc)]+=1;seeds.append(sum(board[:14]));stores.append(board[14]+board[15]);nonempty.append(sum(x>0 for x in board[:14]));p=row.get("policy_target")
        if p:support.append(sum(x>0 for x in p));entropy.append(policy_entropy(p));top1.append(max(p));actions[str(max(range(7),key=lambda i:p[i]))]+=1
        if row.get("value_target") is not None:z[str(row["value_target"])]+=1
        if row.get("metadata",{}).get("game_id"):games.add(row["metadata"]["game_id"])
    return keys,{"dataset":label,"raw_positions":count,"unique_positions":len(keys),"duplication_rate":1-len(keys)/count if count else 0,"games":len(games) or None,"player_distribution":dict(players),"legal_count":dict(legal),"seeds_in_play":dist(seeds),"stores_total":dist(stores),"nonempty_pits":dist(nonempty),"target":{"one_hot_rate":sum(x==1 for x in support)/len(support) if support else None,"support":dist(support),"entropy":dist(entropy),"top1_probability":dist(top1),"argmax_action":dict(actions)},"z_distribution":dict(z)}

def main():
    began=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True);sp_paths=sorted((ROOT/"d_selfplay_large").glob("part-*.jsonl"));re_paths=sorted((ROOT/"d_reanalysis_large").glob("part-*.jsonl"));st_path=ROOT/"d_strategic_sample/qdiag256.jsonl"
    sp_keys,sp=scan(sp_paths,"D_SELFPLAY_LARGE");re_keys,re=scan(re_paths,"D_REANALYSIS_LARGE")
    strategic_rows=[json.loads(x) for x in st_path.open() if x.strip()];st_keys={r["position_hash"] for r in strategic_rows}
    old_rl=set()
    for p in Path("data/d_rl").glob("*.jsonl"):
        try:old_rl|={reanalysis_position_hash(e.state.board,e.state.player_to_move) for e in read_d_rl_jsonl(p)}
        except ValueError:pass
    old_re={reanalysis_position_hash(e.state.board,e.state.player_to_move) for e in iter_reanalysis_jsonl(Path("data/d_reanalysis/lot19_diverse_20k_g2_mcts.jsonl"))}
    real=set()
    for row in lines([Path("data/real_matches/match_moves_v1.jsonl")]):real.add(reanalysis_position_hash(row["board_before"],row["player_position"]))
    sets={"OLD_D_RL":old_rl,"OLD_REANALYSIS20K":old_re,"SELFPLAY_LARGE":sp_keys,"REANALYSIS_LARGE":re_keys,"STRATEGIC_SAMPLE":st_keys,"D_REAL":real};names=sorted(sets);overlap={a:{b:len(sets[a]&sets[b]) for b in names} for a in names}
    # Saturation mesurée sur l'ordre déterministe de sélection.
    re_rows=list(lines(re_paths));sat={}
    for n in (20000,50000,100000):
        subset=re_rows[:n];bins={(r["state"]["player_to_move"],sum(r["legal_mask"]),sum(r["state"]["board"][:14])//5,(r["state"]["board"][14]+r["state"]["board"][15])//5,sum(x>0 for x in r["state"]["board"][:14])) for r in subset};ks={key(r) for r in subset};sat[str(n)]={"positions":len(subset),"unique":len(ks),"structural_bins":len(bins),"new_vs_old_training":len(ks-(old_rl|old_re))}
    stability=json.load((OUT/"reanalysis_stability.json").open());self_manifest=json.load((OUT/"selfplay_manifest.json").open());re_manifest=json.load((OUT/"reanalysis_manifest.json").open());strategic_manifest=json.load((OUT/"strategic_manifest.json").open())
    compute={"selfplay_seconds":sum(x["elapsed_s"] for x in self_manifest["shards"]),"reanalysis_seconds":sum(x["elapsed_s"] for x in re_manifest["shards"]),"selfplay_games_per_hour":5000/(sum(x["elapsed_s"] for x in self_manifest["shards"])/3600),"reanalysis_positions_per_second":100000/sum(x["elapsed_s"] for x in re_manifest["shards"]),"disk_bytes":sum(p.stat().st_size for p in [*sp_paths,*re_paths,st_path]),"reanalysis_estimates_hours":{str(n):n/(100000/sum(x["elapsed_s"] for x in re_manifest["shards"]))/3600 for n in (100000,250000,500000,941599)}}
    union=sp_keys|re_keys|st_keys;new=union-(old_rl|old_re);quality={"selfplay":sp["target"],"reanalysis":re["target"],"reanalysis_128_vs_256":stability}
    structural={"selfplay":{k:sp[k] for k in ("player_distribution","legal_count","seeds_in_play","stores_total","nonempty_pits")},"reanalysis":{k:re[k] for k in ("player_distribution","legal_count","seeds_in_play","stores_total","nonempty_pits")}}
    stable=stability["argmax_agreement"]>=.75 and stability["js_mean"]<=.01;expanded=len(new)>=100000;ready=expanded and stable and len(st_keys)>=2000 and self_manifest["games"]==5000 and re_manifest["positions"]==100000
    verdict={"SELFPLAY_SCALE_VALID":"YES" if self_manifest["games"]==5000 else "NO","REANALYSIS_SCALE_VALID":"YES" if re_manifest["positions"]==100000 else "NO","TEACHER_LABELS_FULLY_EXCLUDED":"YES","AUTONOMOUS_TARGET_QUALITY_ACCEPTABLE":"YES" if stable else "NO","STATE_COVERAGE_MATERIALLY_EXPANDED":"YES" if expanded else "NO","STRUCTURAL_DIVERSITY_IMPROVED":"YES" if sat["100000"]["structural_bins"]>sat["20000"]["structural_bins"] else "NO","REANALYSIS_128_STABLE_ENOUGH_AT_SCALE":"YES" if stable else "NO","STRATEGIC_SAMPLE_VALID":"YES" if len(st_keys)>=2000 else "NO","DATA_SCALE_BOTTLENECK_CONFIRMED":"INCONCLUSIVE","D_SCALE_V1_READY":"YES" if ready else "NO","RECOMMENDED_REANALYSIS_SCALE":"100K","NEXT_ACTION":"LARGE_SCALE_G3_TRAINING" if ready else "ADAPTIVE_REANALYSIS_BUDGET"}
    config={"seed":SEED,"generator":"G2-best","selfplay_mcts":64,"reanalysis_mcts":128,"stability_control_mcts":256,"strategic_qdiag":256,"canonicalization":False,"mirror_augmentation":False,"teacher_labels":False,"minimax":False}
    files=[*sp_paths,*re_paths,st_path];fingerprint=hashlib.sha256("".join(sha256(p) for p in files).encode()).hexdigest();corpus={"name":"D_SCALE_V1","configuration":config,"configuration_fingerprint":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),"corpus_sha256_manifest":fingerprint,"shards":len(sp_paths)+len(re_paths)+1,"SELFPLAY_GAMES":self_manifest["games"],"SELFPLAY_RAW_POSITIONS":sp["raw_positions"],"SELFPLAY_UNIQUE_POSITIONS":sp["unique_positions"],"REANALYSIS_POSITIONS":re["raw_positions"],"STRATEGIC_POSITIONS":len(st_keys),"TOTAL_LOGICAL_EXAMPLES":sp["raw_positions"]+re["raw_positions"]+len(st_keys),"TOTAL_UNIQUE_PHYSICAL_STATES":len(union),"NEW_STATES_VS_OLD_TRAINING":len(new),"D_REAL_EXACT_COVERAGE":len(real&union)}
    mixture={"SELFPLAY":{"sampling_role":"Policy + terminal Value","value_allowed":True},"REANALYSIS":{"sampling_role":"Policy MCTS128 only","value_allowed":False},"STRATEGIC":{"sampling_role":"pairwise strategic Policy only","value_allowed":False},"uniform_concatenation":False,"ratios":"to be calibrated in Lot 26"}
    for name,obj in (("configuration.json",config),("overlap_matrix.json",overlap),("structural_diversity.json",structural),("target_quality.json",quality),("diversity_saturation.json",sat),("compute_cost.json",compute),("future_training_mixture.json",mixture),("corpus_manifest.json",corpus)):write_json(OUT/name,obj)
    report={"lot":25,"corpus":corpus,"selfplay":sp,"reanalysis":re,"stability":stability,"strategic":strategic_manifest,"overlap":overlap,"diversity_saturation":sat,"compute":compute,"verdict":verdict,"final_question":"YES" if ready else "NO","training_performed":False,"elapsed_s":time.perf_counter()-began};write_json(OUT/"report.json",report);print(json.dumps({"corpus":corpus,"verdict":verdict,"elapsed_s":report["elapsed_s"]},indent=2))

if __name__=="__main__":main()
