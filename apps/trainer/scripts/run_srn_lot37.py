#!/usr/bin/env python3
"""Lot 37: courbe high-compute de POOL_G4R contre Minimax-Bidoua."""
from __future__ import annotations
import argparse, concurrent.futures, json, multiprocessing, statistics, time
from dataclasses import asdict
from pathlib import Path

from songo_ai.evaluation import ArenaConfig, MinimaxBidouaReferenceConfig, generate_unique_deterministic_openings, game_result_to_dict, summarize_arena
from run_srn_lot12 import write_json
from run_srn_lot36 import EXPECTED_MINIMAX, _pair, _worker_init, exact_fingerprints

OUT=Path("data/experiments/lot37_g4_high_compute_scaling")
LOT36=Path("data/experiments/lot36_g4_vs_minimax")
BUDGETS=(512,1024,2048,4096); GAMES=64; OPENINGS=32; SEED=20263701
MATERIAL_GAIN=.03

def cli():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("prepare","arena","decide8192","finalize"),required=True);p.add_argument("--budget",type=int,choices=BUDGETS+(8192,));p.add_argument("--workers",type=int,default=8);p.add_argument("--output",type=Path,default=OUT);return p.parse_args()

def prepare(a):
 a.output.mkdir(parents=True,exist_ok=True);ref=MinimaxBidouaReferenceConfig();fps=exact_fingerprints();pool=fps["POOL"]
 if ref.fingerprint!=EXPECTED_MINIMAX:raise RuntimeError("Minimax fingerprint mismatch")
 lot36=json.load((LOT36/"pool_mcts256.json").open());s=lot36["summary"]
 baseline={"BASELINE_SOURCE":"LOT36","budget":256,"score":s["score_rate_a_terminal"],"ci95":s["paired_bootstrap_ci"],"games":s["games"],"artifact":str(LOT36/"pool_mcts256.json"),"replayed":False}
 cfg={"lot":37,"nature":"COMPUTE_SCALING_ONLY","MODEL_PROBE":"POOL_G4R","mandatory_budgets":list(BUDGETS),"conditional_budget":8192,"games_per_budget":GAMES,"opening_pairs":OPENINGS,"paired":True,"side_swapped":True,"common_openings_across_budgets":True,"material_gain_threshold":MATERIAL_GAIN,"plateau_rule":"two consecutive deltas < .03 and uncertainty compatible","mcts8192_rule":"execute iff delta4096 >= .03","no_mcts16384":True,"training_performed":False,"minimax_labels":False,"g5_training_performed":False,"baseline256_replayed":False,"created_before_results":True}
 openings=generate_unique_deterministic_openings(count=OPENINGS,seed=SEED,max_prefix_length=40)
 manifest={"created_before_results":True,"independent_from_lot36":True,"openings_seed":SEED,"arena_seed":SEED+10000,"opening_pairs":OPENINGS,"games":GAMES,"shared_across_all_budgets":True,"openings":[{"opening_id":x.opening_id,"prefix_actions":list(x.prefix_actions)} for x in openings]}
 write_json(a.output/"configuration.json",cfg);write_json(a.output/"fingerprints.json",{"before":{"POOL":pool,"MINIMAX":ref.fingerprint},"after":None,"frozen":True});write_json(a.output/"seed_manifest.json",manifest);write_json(a.output/"baseline_lot36.json",baseline);print(json.dumps({"prepared":True,"baseline":baseline},indent=2))

def arena(a):
 if a.budget is None:raise SystemExit("--budget required")
 if a.budget==8192:
  gate=json.load((a.output/"mcts8192_gate.json").open())
  if not gate["authorized"]:raise RuntimeError("MCTS8192 not authorized")
 fps=exact_fingerprints()["POOL"];ref=MinimaxBidouaReferenceConfig();manifest=json.load((a.output/"seed_manifest.json").open());openings=generate_unique_deterministic_openings(count=OPENINGS,seed=manifest["openings_seed"],max_prefix_length=40);cfg=ArenaConfig(max_plies=400,repetition_limit=3,seed=manifest["arena_seed"],bootstrap_samples=20000);started=time.perf_counter();ctx=multiprocessing.get_context("spawn")
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=ctx,initializer=_worker_init,initargs=(fps["policy_checkpoint"],fps["value_checkpoint"],"POOL_G4R",a.budget,asdict(ref),asdict(cfg))) as ex:pairs=list(ex.map(_pair,openings))
 results=tuple(x for pair in pairs for x in pair);summary=asdict(summarize_arena(results,config=cfg));wall=time.perf_counter()-started
 search_s=sum(sum((g.search_time_s_by_agent or {}).values()) for g in results);pool_s=sum((g.search_time_s_by_agent or {}).get("POOL_G4R",0) for g in results);pool_sims=sum(g.total_mcts_simulations for g in results)
 payload={"budget":a.budget,"summary":summary,"wall_time_s":wall,"compute":{"total_search_s":search_s,"pool_search_s":pool_s,"mean_game_duration_s":wall/len(results),"simulations_per_second":pool_sims/pool_s if pool_s else None,"mean_pool_search_s_per_game":pool_s/len(results)},"games":[game_result_to_dict(g) for g in results]};write_json(a.output/f"mcts{a.budget}.json",payload);print(json.dumps({"budget":a.budget,"score":summary["score_rate_a_terminal"],"ci95":summary["paired_bootstrap_ci"],"wall_time_s":wall},indent=2))

def decide8192(a):
 base=json.load((a.output/"mcts2048.json").open())["summary"]["score_rate_a_terminal"];score=json.load((a.output/"mcts4096.json").open())["summary"]["score_rate_a_terminal"];delta=score-base;gate={"rule":"S4096-S2048 >= .03","delta4096":delta,"threshold":MATERIAL_GAIN,"authorized":delta>=MATERIAL_GAIN,"MCTS8192_EXECUTED":"PENDING" if delta>=MATERIAL_GAIN else "NO"};write_json(a.output/"mcts8192_gate.json",gate);print(json.dumps(gate,indent=2))

def side_score(x):return (x["wins"]+.5*x["draws"])/x["terminal_games"] if x["terminal_games"] else None
def finalize(a):
 base=json.load((a.output/"baseline_lot36.json").open());points={256:base};
 for b in BUDGETS:points[b]=json.load((a.output/f"mcts{b}.json").open())
 gate=json.load((a.output/"mcts8192_gate.json").open());executed=(a.output/"mcts8192.json").exists()
 cancelled=(a.output/"mcts8192_cancellation.json").exists()
 if gate["authorized"] and not executed and not cancelled:raise RuntimeError("authorized MCTS8192 must execute before finalize unless explicitly cancelled")
 if executed:points[8192]=json.load((a.output/"mcts8192.json").open())
 scores={b:(p["score"] if b==256 else p["summary"]["score_rate_a_terminal"]) for b,p in points.items()};ordered=sorted(scores);deltas={str(b):scores[b]-scores[ordered[i-1]] for i,b in enumerate(ordered) if i};relative={str(b):(scores[b]/scores[ordered[i-1]]-1 if scores[ordered[i-1]] else None) for i,b in enumerate(ordered) if i}
 plateau_pairs=[b for b in ordered[2:] if deltas[str(b)]<MATERIAL_GAIN and deltas[str(ordered[ordered.index(b)-1])]<MATERIAL_GAIN];plateau="YES" if plateau_pairs else "NO"
 knee=str(ordered[ordered.index(plateau_pairs[0])-1]) if plateau_pairs else "NOT_REACHED";material=sum(d>=MATERIAL_GAIN for d in deltas.values());trend=scores[ordered[-1]]-scores[256]
 scaling="STRONG" if material>=2 else "POSITIVE" if trend>=MATERIAL_GAIN else "SATURATING" if plateau=="YES" and trend>0 else "FLAT" if abs(trend)<MATERIAL_GAIN else "NEGATIVE" if trend<0 else "INCONCLUSIVE"
 limitation="STRONG_EVIDENCE" if material>=2 and plateau=="NO" else "PARTIAL_EVIDENCE" if trend>=MATERIAL_GAIN else "LITTLE_EVIDENCE" if plateau=="YES" else "INCONCLUSIVE"
 best=max(scores,key=scores.get);bestscore=scores[best];bestci=points[best]["ci95"] if best==256 else points[best]["summary"]["paired_bootstrap_ci"]
 next_action="G4_HIGH_COMPUTE_CONFIRMATION_VS_MINIMAX" if bestscore>=.45 else "G4_EXTREME_COMPUTE_SCALING_DESIGN" if plateau=="NO" and scaling in ("STRONG","POSITIVE") else "MINIMAX_STRATEGIC_GAP_DIAGNOSIS"
 curve={"scores":{str(k):v for k,v in scores.items()},"deltas":deltas,"relative_gains":relative,"gap_to_50":{str(k):.5-v for k,v in scores.items()},"historical_g2":{"score":.015625,"ratios":{str(k):v/.015625 for k,v in scores.items()},"absolute_gains":{str(k):v-.015625 for k,v in scores.items()}}};write_json(a.output/"scaling_curve.json",curve)
 efficiency={str(b):points[b].get("compute") for b in ordered if b!=256};write_json(a.output/"compute_efficiency.json",efficiency);write_json(a.output/"plateau_analysis.json",{"threshold":MATERIAL_GAIN,"two_step_rule":True,"COMPUTE_PLATEAU_DETECTED":plateau,"COMPUTE_KNEE_POINT":knee})
 after=exact_fingerprints()["POOL"];fp=json.load((a.output/"fingerprints.json").open());fp["after"]={"POOL":after,"MINIMAX":MinimaxBidouaReferenceConfig().fingerprint};fp["unchanged"]=fp["before"]==fp["after"];write_json(a.output/"fingerprints.json",fp)
 valid=fp["unchanged"] and all(points[b]["summary"]["games"]==64 for b in points if b!=256)
 decision={"LOT37_VALID":"YES" if valid else "NO","MODEL_PROBE":"POOL_G4R","BASELINE_MCTS256":scores[256],**{f"SCORE_MCTS{b}":scores[b] for b in BUDGETS},"MCTS8192_EXECUTED":"YES" if executed else "NO","MCTS8192_AUTHORIZED_BY_RULE":"YES" if gate["authorized"] else "NO","MCTS8192_CANCELLED_BY_USER":"YES" if cancelled else "NO","SCORE_MCTS8192":scores.get(8192,"N/A"),"HIGH_COMPUTE_SCALING":scaling,"COMPUTE_PLATEAU_DETECTED":plateau,"COMPUTE_KNEE_POINT":knee,"BEST_OBSERVED_BUDGET":best,"BEST_OBSERVED_SCORE":bestscore,"BEST_OBSERVED_IC95":bestci,"BEST_GAP_TO_50":.5-bestscore,"SEARCH_BUDGET_LIMITATION":limitation,"HIGH_COMPUTE_BREAKTHROUGH_CANDIDATE":"YES" if bestscore>=.5 else "NO","G5_TRAINING_PERFORMED":"NO","training_performed":False,"minimax_labels_used":False,"NEXT_ACTION":next_action};write_json(a.output/"final_decision.json",decision);write_json(a.output/"next_action.json",{"NEXT_ACTION":next_action});write_json(a.output/"report.json",{"lot":37,"decision":decision,"curve":curve,"gate8192":gate,"mcts8192_cancelled_by_user":cancelled,"fingerprints_unchanged":fp["unchanged"]});print(json.dumps(decision,indent=2))

if __name__=="__main__":
 a=cli();{"prepare":prepare,"arena":arena,"decide8192":decide8192,"finalize":finalize}[a.stage](a)
