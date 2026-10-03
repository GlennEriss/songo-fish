#!/usr/bin/env python3
"""Lot 28 : intervention Value sous recherche, Policy G3 strictement gelée."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from songo_ai.dataset import RawSongoState, RLTrainingExample, read_d_rl_jsonl
from songo_ai.evaluation import (
    ArenaConfig,
    HybridPolicyValueEvaluator,
    flip_rate,
    generate_unique_deterministic_openings,
    local_value_consistency,
    search_sensitivity,
    value_disagreement,
    value_metrics,
)
from songo_ai.model import SongoGraphBuilder, SRNTrainingConfig, load_srn_checkpoint
from songo_ai.search import MCTSConfig, SongoMCTS
from songo_ai.songo.rules import SongoLegacyGame
from run_srn_lot12 import write_json
from run_srn_lot20 import collate, save_checkpoint
from run_srn_lot23 import arena
from run_srn_lot27 import battery, js_divergence

G2 = Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt")
G3 = Path("data/experiments/lot26_g3_scale/checkpoints/g3_strategic_best.pt")
BUDGETS = (64, 128, 256)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("hybrid", "diagnosis", "train", "mini", "arena", "finalize", "all"), default="all")
    p.add_argument("--output", type=Path, default=Path("data/experiments/lot28_value_search"))
    p.add_argument("--seed", type=int, default=20262828)
    p.add_argument("--diagnostic-positions", type=int, default=400)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bootstrap", type=int, default=20_000)
    return p.parse_args()


def digest_state(state):
    return hashlib.sha256((",".join(map(str, state.board)) + f"|{state.player_to_move}").encode()).hexdigest()


def models():
    g2 = load_srn_checkpoint(G2).model
    g3 = load_srn_checkpoint(G3).model
    g2.eval(); g3.eval()
    return g2, g3, HybridPolicyValueEvaluator(g3, g2, name="G3-HYBRID")


def policy_value_identity(g2, g3, hybrid, rows):
    graph = SongoGraphBuilder().build_batch([RawSongoState(tuple(x["state"]["board"]), x["state"]["player_to_move"]) for x in rows[:128]])
    with torch.no_grad():
        p2, v2 = g2(graph); p3, _ = g3(graph); ph, vh = hybrid(graph)
    return {
        "positions": graph.batch_size,
        "policy_exact": torch.equal(ph, p3),
        "value_exact": torch.equal(vh, v2),
        "max_abs_policy_logit_delta": float((ph-p3).abs().max()),
        "max_abs_value_delta": float((vh-v2).abs().max()),
        "tolerance": 0.0,
    }


def run_hybrid(a):
    rows, manifest = battery(max(128, a.diagnostic_positions))
    g2, g3, hybrid = models()
    identity = policy_value_identity(g2, g3, hybrid, rows)
    openings = generate_unique_deterministic_openings(count=32, seed=a.seed, max_prefix_length=40)
    cfg = ArenaConfig(max_plies=400, repetition_limit=3, seed=a.seed, bootstrap_samples=a.bootstrap)
    short = {}
    for budget in (64, 128):
        short[str(budget)] = arena("G3-HYBRID", hybrid, g2, openings, cfg, budget)
        print(f"[lot28] hybrid short MCTS{budget} done", flush=True)
    weak = any(x["summary"]["score_rate_a_terminal"] < .40 for x in short.values())
    main = {}
    if not weak:
        openings = generate_unique_deterministic_openings(count=128, seed=a.seed+1, max_prefix_length=40)
        cfg = ArenaConfig(max_plies=400, repetition_limit=3, seed=a.seed+1, bootstrap_samples=a.bootstrap)
        for budget in (64, 128):
            main[str(budget)] = arena("G3-HYBRID", hybrid, g2, openings, cfg, budget)
            print(f"[lot28] hybrid main MCTS{budget} done", flush=True)
    result = {"identity": identity, "battery": manifest, "short": short, "main": main, "clear_weakness": weak}
    write_json(a.output/"hybrid_evaluation.json", result)
    write_json(a.output/"short_arena.json", {"G3-HYBRID": {b:x for b,x in short.items()}})
    if main: write_json(a.output/"main_arena.json", {"G3-HYBRID": main})
    return result


def select_diagnostic_rows(count):
    rows, _ = battery(2000)
    failures = {x["position_hash"] for x in json.load(Path("data/experiments/lot27_search_policy/scaling_failure_set.json").open())["rows"]}
    selected = [x for x in rows if x["position_hash"] in failures][:count//2]
    selected += [x for x in rows if x["position_hash"] not in failures][:(count-len(selected))]
    return selected, failures


def model_values(model, states, batch_size=512):
    out=[]; builder=SongoGraphBuilder(); model.eval()
    with torch.no_grad():
        for start in range(0,len(states),batch_size):
            _,v=model(builder.build_batch(states[start:start+batch_size]));out.extend(map(float,v))
    return out


def point(trace, budget, legal):
    t=trace[budget-1]; visits=t["visit_counts_after_backup"]; total=sum(visits)
    action=max((i for i,x in enumerate(legal) if x),key=lambda i:(visits[i],-i))
    return {"action":action,"visits":visits,"policy":[x/total for x in visits],"q":t["root_q_values_after_backup"]}


def run_diagnosis(a):
    rows, failures = select_diagnostic_rows(a.diagnostic_positions)
    g2,g3,hybrid=models(); configs={"G3_VALUE_G2":hybrid,"G3_VALUE_G3":g3}
    cache=a.output/"value_leaf_cache.jsonl"; known={}
    if cache.exists():
        for line in cache.open():
            x=json.loads(line);known[(x["position_hash"],x["config"])]=x
    with cache.open("a") as out:
        for ordinal,row in enumerate(rows,1):
            state=RawSongoState(tuple(row["state"]["board"]),row["state"]["player_to_move"])
            for name,model in configs.items():
                key=(row["position_hash"],name)
                if key not in known:
                    r=SongoMCTS(model,config=MCTSConfig(num_simulations=256,c_puct=1.5,add_root_noise=False,collect_simulation_trace=True,seed=a.seed)).search(state)
                    item={"position_hash":row["position_hash"],"config":name,"failure":row["position_hash"] in failures,"points":{str(b):point(r.simulation_trace,b,row["legal_mask"]) for b in BUDGETS},"trace":list(r.simulation_trace)}
                    out.write(json.dumps(item,sort_keys=True)+"\n");out.flush();known[key]=item
            if ordinal%25==0:print(f"[lot28] diagnosis {ordinal}/{len(rows)}",flush=True)
    # Toutes les feuilles uniques, évaluées par les deux Values sur la même batterie.
    leaf_map={}
    for row in rows:
        for name in configs:
            for t in known[(row["position_hash"],name)]["trace"]:
                s=RawSongoState(tuple(t["leaf_state"]["board"]),t["leaf_state"]["player_to_move"])
                leaf_map.setdefault(digest_state(s),(s,t["leaf_depth"],sum(s.board[:14]),t["leaf_terminal"]))
    leaf=list(leaf_map.values()); states=[x[0] for x in leaf]; vg2=model_values(g2,states);vg3=model_values(g3,states)
    by_depth={};by_seeds={}
    for label,keyfn in ((by_depth,lambda x:str(min(x[1],8))),(by_seeds,lambda x:str((x[2]//10)*10))):
        groups={}
        for i,x in enumerate(leaf):groups.setdefault(keyfn(x),[]).append(i)
        for k,ids in groups.items():label[k]=value_disagreement([vg2[i] for i in ids],[vg3[i] for i in ids])
    # Cohérence parent-enfant sur un coup légal déterministe, non terminal uniquement.
    parents=[];children=[]
    for state,_,_,terminal in leaf[:5000]:
        if terminal:continue
        game=SongoLegacyGame.from_state(state.to_engine_state());game.normalize_terminal()
        if game.finished:continue
        game.play_local(game.legal_local_actions()[0]);game.normalize_terminal()
        if not game.finished:parents.append(state);children.append(RawSongoState.from_game(game))
    pg2,pg3=model_values(g2,parents),model_values(g3,parents);cg2,cg3=model_values(g2,children),model_values(g3,children)
    sensitivity=[]
    for row in rows:
        h=known[(row["position_hash"],"G3_VALUE_G2")];f=known[(row["position_hash"],"G3_VALUE_G3")]
        for b in BUDGETS:
            x,y=h["points"][str(b)],f["points"][str(b)];root_action=x["action"]
            sensitivity.append({"position_hash":row["position_hash"],"budget":b,"delta_value":statistics.fmean(t["leaf_value"] for t in f["trace"][:b])-statistics.fmean(t["leaf_value"] for t in h["trace"][:b]),"delta_q":y["q"][root_action]-x["q"][root_action],"visit_js":js_divergence(x["policy"],y["policy"]),"action_flip":x["action"]!=y["action"]})
    scaling={name:{"flip_rate_64_128":flip_rate([known[(r["position_hash"],name)]["points"]["64"]["action"] for r in rows],[known[(r["position_hash"],name)]["points"]["128"]["action"] for r in rows]),"flip_rate_128_256":flip_rate([known[(r["position_hash"],name)]["points"]["128"]["action"] for r in rows],[known[(r["position_hash"],name)]["points"]["256"]["action"] for r in rows])} for name in configs}
    analysis={"positions":len(rows),"unique_leaves":len(leaf),"value_disagreement":value_disagreement(vg2,vg3),"by_depth":by_depth,"by_seeds_in_play":by_seeds,"local_consistency":{"G2":local_value_consistency(pg2,cg2),"G3":local_value_consistency(pg3,cg3)},"scaling":scaling}
    write_json(a.output/"value_diagnosis.json",analysis);write_json(a.output/"value_leaf_analysis.json",{"leaf_count":len(leaf),"by_depth":by_depth,"by_seeds_in_play":by_seeds,"local_consistency":analysis["local_consistency"]});write_json(a.output/"search_sensitivity.json",{"summary":search_sensitivity(sensitivity),"rows":sensitivity});return analysis


def load_selfplay():
    items=[]
    for path in sorted(Path("data/d_scale_v1/d_selfplay_large").glob("part-*.jsonl")):items.extend(read_d_rl_jsonl(path))
    if any(any(k.lower().startswith(("teacher","minimax")) for k in x.metadata) for x in items):raise RuntimeError("forbidden Teacher/Minimax metadata")
    # Les trajectoires tronquées n'ont légitimement pas de résultat terminal :
    # elles sont exclues, jamais converties artificiellement en z=0.
    return [x for x in items if x.value_target is not None]


def split_selfplay(items,seed):
    train=[];val=[]
    for x in items:
        bucket=int(hashlib.sha256(f"{seed}:{x.metadata['game_id']}".encode()).hexdigest()[:16],16)%10
        (val if bucket==0 else train).append(x)
    return train,val


def evaluate_value_model(model,items,limit=50000):
    sample=items[:limit];pred=[];target=[]
    with torch.no_grad():
        for start in range(0,len(sample),512):
            b=collate(sample[start:start+512]);_,v=model(b.graph);pred.extend(map(float,v));target.extend(map(float,b.value_target))
    return value_metrics(pred,target)


def child_examples(examples):
    result=[]
    for x in examples:
        game=SongoLegacyGame.from_state(x.state.to_engine_state());game.normalize_terminal()
        if game.finished:continue
        game.play_local(game.legal_local_actions()[0]);game.normalize_terminal()
        if game.finished:continue
        legal=tuple(game.legal_mask());n=sum(legal)
        child=RLTrainingExample(RawSongoState.from_game(game),legal,tuple(1/n if z else 0 for z in legal),None,tuple(1 if z else 0 for z in legal),{"game_id":x.metadata["game_id"]})
        result.append((x,child))
    return result


def train_value(a,name,train,val,consistency_weight):
    parent=load_srn_checkpoint(G2);model=load_srn_checkpoint(G2).model
    for p in model.parameters():p.requires_grad=False
    for p in model.value_mlp.parameters():p.requires_grad=True
    cfg=SRNTrainingConfig(epochs=a.epochs,batch_size=256,learning_rate=3e-4,weight_decay=1e-4,lambda_policy=0,lambda_value=1,early_stopping_patience=2,seed=a.seed,device="cpu")
    # Tous les paramètres restent dans le groupe optimiseur pour conserver le
    # contrat de checkpoint historique; seuls ceux de value_mlp ont un gradient.
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay);history=[];best=math.inf;wait=0;path=a.output/"checkpoints"/f"{name.lower()}_best.pt";steps=math.ceil(len(train)/cfg.batch_size)
    policy_reference=load_srn_checkpoint(G3).model;fixed=collate(val[:128]);
    with torch.no_grad():reference_logits,_=policy_reference(fixed.graph)
    for epoch in range(a.epochs+1):
        if epoch:
            order=list(range(len(train)));random.Random(a.seed+epoch).shuffle(order);model.train()
            for step in range(steps):
                batch=[train[order[(step*cfg.batch_size+i)%len(order)]] for i in range(cfg.batch_size)];b=collate(batch);_,v=model(b.graph);loss=(v-b.value_target).square().mean()
                if consistency_weight:
                    pairs=child_examples(batch)
                    if pairs:
                        pb=collate([x[0] for x in pairs]);cb=collate([x[1] for x in pairs]);_,cv=model(cb.graph);_,pv=model(pb.graph);loss=loss+consistency_weight*(pv+cv).square().mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.value_mlp.parameters(),1.);opt.step()
        metric=evaluate_value_model(model,val);row={"epoch":epoch,"global_step":epoch*steps,"selection_score":metric["mse"],**metric};history.append(row);print(f"[lot28] {name} epoch {epoch}: mse={metric['mse']:.6f}",flush=True)
        if metric["mse"]<best-1e-5:
            best=metric["mse"];wait=0;save_checkpoint(path,model,opt,parent.payload,cfg,epoch,history,{"candidate":name,"parent":"G2","objective":"true terminal z only","consistency_weight":consistency_weight,"trainable_parameters":"value_mlp only"},epoch)
        else:wait+=1
        if epoch and wait>=2:break
    trained=load_srn_checkpoint(path).model;hybrid=HybridPolicyValueEvaluator(policy_reference,trained,name=f"G3-{name}")
    with torch.no_grad():logits,_=hybrid(fixed.graph)
    immutable={"max_abs_policy_logit_delta":float((logits-reference_logits).abs().max()),"exact":torch.equal(logits,reference_logits)}
    with (a.output/f"training_{name.lower()}.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    return path,history,immutable


def run_training(a):
    items=load_selfplay();train,val=split_selfplay(items,a.seed);diagnosis=json.load((a.output/"value_diagnosis.json").open());specs={"V28_A":0.0}
    g2c=diagnosis["local_consistency"]["G2"]["mean_abs_residual"];g3c=diagnosis["local_consistency"]["G3"]["mean_abs_residual"]
    if g3c>g2c*1.10:specs["V28_B"]=0.05
    result={"dataset":{"raw_positions":441563,"excluded_without_terminal_z":441563-len(items),"labeled_total":len(items),"train":len(train),"validation":len(val),"targets":"true terminal z only","teacher_labels":False,"minimax_labels":False},"variants":{}}
    for name,weight in specs.items():
        path,history,immutable=train_value(a,name,train,val,weight);result["variants"][name]={"checkpoint":str(path),"history":history,"policy_immutability":immutable,"consistency_weight":weight}
    result["variants_trained"]=len(specs);result["V28_C"]="NOT_JUSTIFIED"
    write_json(a.output/"value_training.json",result);return result


def evaluator_for(path,name):
    g3=load_srn_checkpoint(G3).model;v=load_srn_checkpoint(path).model;return HybridPolicyValueEvaluator(g3,v,name=name)


def run_mini(a):
    rows,_=select_diagnostic_rows(a.diagnostic_positions);g2,g3,hybrid=models();training=json.load((a.output/"value_training.json").open());configs={"G3-HYBRID":hybrid,"G3-STRATEGIC":g3}
    for n,x in training["variants"].items():configs[n]=evaluator_for(x["checkpoint"],n)
    output={}
    for name,model in configs.items():
        acts={str(b):[] for b in BUDGETS};q={str(b):[] for b in BUDGETS};vis={str(b):[] for b in BUDGETS}
        for i,row in enumerate(rows,1):
            state=RawSongoState(tuple(row["state"]["board"]),row["state"]["player_to_move"]);r=SongoMCTS(model,config=MCTSConfig(num_simulations=256,c_puct=1.5,add_root_noise=False,collect_simulation_trace=True,seed=a.seed)).search(state)
            for b in BUDGETS:
                p=point(r.simulation_trace,b,row["legal_mask"]);acts[str(b)].append(p["action"]);q[str(b)].append(p["q"]);vis[str(b)].append(p["policy"])
            if i%50==0:print(f"[lot28] mini {name} {i}/{len(rows)}",flush=True)
        output[name]={"positions":len(rows),"flip_rate_64_128":flip_rate(acts["64"],acts["128"]),"flip_rate_128_256":flip_rate(acts["128"],acts["256"]),"actions":acts}
    write_json(a.output/"mini_search.json",output);return output


def run_candidate_arenas(a):
    training=json.load((a.output/"value_training.json").open());mini=json.load((a.output/"mini_search.json").open());g2,_,hybrid=models()
    candidates={"G3-HYBRID":hybrid}
    ranked=sorted(training["variants"],key=lambda n:(mini[n]["flip_rate_64_128"],mini[n]["flip_rate_128_256"]))
    if ranked:candidates[ranked[0]]=evaluator_for(training["variants"][ranked[0]]["checkpoint"],ranked[0])
    openings=generate_unique_deterministic_openings(count=32,seed=a.seed+2,max_prefix_length=40);cfg=ArenaConfig(max_plies=400,repetition_limit=3,seed=a.seed+2,bootstrap_samples=a.bootstrap);short={}
    # L'hybride a déjà son short; seules les nouvelles combinaisons sont jouées ici.
    for name,model in candidates.items():
        if name=="G3-HYBRID":continue
        short[name]={"64":arena(name,model,g2,openings,cfg,64)};print(f"[lot28] short {name} done",flush=True)
    existing=json.load((a.output/"short_arena.json").open());existing.update(short);write_json(a.output/"short_arena.json",existing)
    finalists={n:m for n,m in candidates.items() if n=="G3-HYBRID" or short.get(n,{}).get("64",{}).get("summary",{}).get("score_rate_a_terminal",0)>=.4}
    main=json.load((a.output/"main_arena.json").open()) if (a.output/"main_arena.json").exists() else {}
    openings=generate_unique_deterministic_openings(count=128,seed=a.seed+3,max_prefix_length=40);cfg=ArenaConfig(max_plies=400,repetition_limit=3,seed=a.seed+3,bootstrap_samples=a.bootstrap)
    for name,model in finalists.items():
        if name in main and {"64","128"}<=set(main[name]):continue
        main[name]={}
        for b in (64,128):main[name][str(b)]=arena(name,model,g2,openings,cfg,b);print(f"[lot28] main {name} MCTS{b} done",flush=True)
    write_json(a.output/"main_arena.json",main)
    # Confirmation 256 ciblée du meilleur score moyen 64/128.
    best=max(finalists,key=lambda n:statistics.fmean(main[n][str(b)]["summary"]["score_rate_a_terminal"] for b in (64,128)))
    scale={n:{"score64":main[n]["64"]["summary"]["score_rate_a_terminal"],"score128":main[n]["128"]["summary"]["score_rate_a_terminal"]} for n in finalists}
    if all(main[best][str(b)]["summary"]["score_rate_a_terminal"]>.5 for b in (64,128)):
        openings=generate_unique_deterministic_openings(count=64,seed=a.seed+4,max_prefix_length=40);cfg=ArenaConfig(max_plies=400,repetition_limit=3,seed=a.seed+4,bootstrap_samples=a.bootstrap);scale[best]["mcts256"]=arena(best,finalists[best],g2,openings,cfg,256);print(f"[lot28] scaling {best} MCTS256 done",flush=True)
    write_json(a.output/"search_scaling.json",scale);return main


def finalize(a):
    hybrid=json.load((a.output/"hybrid_evaluation.json").open());diagnosis=json.load((a.output/"value_diagnosis.json").open());training=json.load((a.output/"value_training.json").open());mini=json.load((a.output/"mini_search.json").open());main=json.load((a.output/"main_arena.json").open());scaling=json.load((a.output/"search_scaling.json").open())
    def robust(name):
        return name in main and all(main[name][str(b)]["summary"]["score_rate_a_terminal"]>.5 for b in (64,128))
    eligible=[n for n in main if robust(n)];best=max(eligible,key=lambda n:statistics.fmean(main[n][str(b)]["summary"]["score_rate_a_terminal"] for b in (64,128))) if eligible else None
    healthy=bool(best) and ("mcts256" not in scaling[best] or scaling[best]["mcts256"]["summary"]["score_rate_a_terminal"]>=.45)
    candidate=best if best and healthy else None;best_value="G2" if candidate=="G3-HYBRID" else candidate.replace("V28_","V28_") if candidate else "NONE"
    verdict={"HYBRID_VALID":"YES" if hybrid["identity"]["policy_exact"] and hybrid["identity"]["value_exact"] else "NO","HYBRID_BEATS_G2":"YES" if robust("G3-HYBRID") else "NO","G3_POLICY_PRESERVED":"YES" if all(x["policy_immutability"]["exact"] for x in training["variants"].values()) else "NO","VALUE_G2_MORE_SEARCH_STABLE_THAN_VALUE_G3":"YES" if diagnosis["scaling"]["G3_VALUE_G2"]["flip_rate_64_128"]<diagnosis["scaling"]["G3_VALUE_G3"]["flip_rate_64_128"] else "NO","VALUE_RETRAINING_VALID":"YES" if training["variants"] else "NOT_NEEDED","VALUE_SEARCH_STABILITY_IMPROVED":"YES" if any(mini[n]["flip_rate_64_128"]<mini["G3-STRATEGIC"]["flip_rate_64_128"] for n in training["variants"]) else "NO","VALUE_REWORK_BEATS_HYBRID":"YES" if candidate and candidate!="G3-HYBRID" else "NO","VALUE_REWORK_INSUFFICIENT":"NO" if candidate else "YES","SEARCH_SCALING_HEALTHY":"YES" if healthy else "NO","BEST_POLICY":"G3_STRATEGIC","BEST_VALUE":best_value,"G3_CANDIDATE":"G3_HYBRID" if candidate=="G3-HYBRID" else "G3_VALUE_REWORK" if candidate else "NONE","NEXT_ACTION":"INDEPENDENT_G3_CONFIRMATION" if candidate else "REASSESS_MODEL_LEARNING_PARADIGM"}
    selection={"selected":candidate,"verdict":verdict,"rule":"advantage at MCTS64 and MCTS128, acceptable scaling, non-degenerate terminal Value"};write_json(a.output/"candidate_selection.json",selection)
    report={"lot":28,"hybrid":hybrid,"value_diagnosis":diagnosis,"value_training":training,"mini_search":mini,"main_arena":{n:{b:x["summary"] for b,x in v.items()} for n,v in main.items()},"search_scaling":scaling,"candidate_selection":selection,"verdict":verdict};write_json(a.output/"report.json",report);return report


def main():
    a=parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/"checkpoints").mkdir(exist_ok=True)
    write_json(a.output/"configuration.json",{"seed":a.seed,"budgets":BUDGETS,"c_puct":1.5,"dirichlet":False,"temperature":0,"bootstrap_samples":a.bootstrap,"policy_checkpoint":str(G3),"value_control_checkpoint":str(G2),"engine_changed":False,"policy_training":False,"max_new_values":3})
    phases=("hybrid","diagnosis","train","mini","arena","finalize") if a.phase=="all" else (a.phase,)
    for phase in phases:
        {"hybrid":run_hybrid,"diagnosis":run_diagnosis,"train":run_training,"mini":run_mini,"arena":run_candidate_arenas,"finalize":finalize}[phase](a)
    if (a.output/"report.json").exists():print(json.dumps(json.load((a.output/"report.json").open())["verdict"],indent=2),flush=True)


if __name__=="__main__":main()
