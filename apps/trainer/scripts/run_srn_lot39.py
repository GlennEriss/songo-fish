#!/usr/bin/env python3
"""Lot 39: scaling du MCTS SRN batché sur recherches indépendantes."""
from __future__ import annotations

import argparse,csv,hashlib,json,math,os,platform,resource,statistics,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
import torch

from songo_ai.dataset import RawSongoState
from songo_ai.evaluation import HybridPolicyValueEvaluator,model_parameter_fingerprint
from songo_ai.model import SongoGraphBuilder,load_srn_checkpoint
from songo_ai.search import MCTSConfig,SongoMCTS
from songo_ai.songo.rules import SongoLegacyGame
from run_srn_colab_benchmark import engine_fingerprint,env,git_commit,pool_fingerprints,pool_identity
from run_srn_lot12 import sha256,write_json

OUT=Path("data/experiments/lot39_gpu_parallel_scaling")
CONCURRENCIES=(16,32,64,128,256);POSITIONS=256;BUDGET=4096;SEED=20263901;CPU_REFERENCE=1652.363804632677;WARMUP=10
RESUME_CRITICAL_FILES=("packages/songo_ai/search/mcts.py","packages/songo_ai/songo/rules.py","packages/songo_ai/model/srn_graph.py","packages/songo_ai/model/srn_network.py")
MEASUREMENT_SCHEMA_VERSION=1

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("prepare","correctness","run","finalize","export"),required=True);p.add_argument("--concurrency",type=int,choices=(8,)+CONCURRENCIES);p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto");p.add_argument("--positions-file",type=Path);p.add_argument("--output",type=Path,default=OUT);p.add_argument("--bundle",type=Path,default=Path("lot39_results.tar.gz"));return p.parse_args()

def device(requested):
 if requested=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
 return torch.device("cuda" if requested=="cuda" or requested=="auto" and torch.cuda.is_available() else "cpu")
def fingerprint_state(state):return hashlib.sha256((json.dumps({"board":state.board,"player_to_move":state.player_to_move},separators=(",",":"))).encode()).hexdigest()
def load_positions(path):
 d=json.load(path.open());states=[RawSongoState(tuple(x["board"]),int(x["player_to_move"])) for x in d["states"]]
 if len(states)!=POSITIONS or len({fingerprint_state(x) for x in states})!=POSITIONS:raise RuntimeError("Lot39 requires 256 unique positions")
 for state in states:
  game=SongoLegacyGame.from_state(state.to_engine_state());game.normalize_terminal()
  if game.finished or not game.legal_local_actions():raise RuntimeError("invalid or terminal benchmark position")
 return d,states
def load_model(d):
 item=pool_identity();p=load_srn_checkpoint(item["policy_checkpoint"]).model.to(d).eval();v=load_srn_checkpoint(item["value_checkpoint"]).model.to(d).eval();return HybridPolicyValueEvaluator(p,v,name="POOL_G4R").to(d).eval()
def checksum_payload(path):return {"file":path.name,"sha256":sha256(path),"size":path.stat().st_size}
def resume_code_compatible(previous_commit):
 if previous_commit==git_commit():return True
 try:return subprocess.run(["git","diff","--quiet",previous_commit,"HEAD","--",*RESUME_CRITICAL_FILES],check=False).returncode==0
 except Exception:return False
def write_stage(path,payload):
 write_json(path,payload);write_json(path.with_suffix(path.suffix+".checksum.json"),checksum_payload(path))
def valid_stage(path,concurrency):
 side=path.with_suffix(path.suffix+".checksum.json")
 if not path.is_file() or not side.is_file():return False
 try:d=json.load(path.open());c=json.load(side.open())
 except Exception:return False
 return c.get("sha256")==sha256(path) and d.get("status")=="COMPLETE" and d.get("parallel_searches")==concurrency and d.get("mcts_budget")==BUDGET and d.get("git_commit")==git_commit() and d.get("model_fingerprints")==pool_fingerprints()

def prepare(a):
 if not a.positions_file:raise ValueError("prepare requires --positions-file")
 source,states=load_positions(a.positions_file);a.output.mkdir(parents=True,exist_ok=True)
 positions={"count":POSITIONS,"seed":SEED,"fingerprint":hashlib.sha256("".join(fingerprint_state(s) for s in states).encode()).hexdigest(),"provenance":source.get("provenance",{}),"labels_used":False,"states":[{"board":list(s.board),"player_to_move":s.player_to_move,"state_fingerprint":fingerprint_state(s)} for s in states]}
 write_json(a.output/"benchmark_positions.json",positions);d=device(a.device);write_json(a.output/"environment.json",env(d));write_json(a.output/"configuration.json",{"lot":39,"model":"POOL_G4R","mcts_budget":BUDGET,"parallel_searches":list(CONCURRENCIES),"positions":POSITIONS,"total_simulations_per_stage":POSITIONS*BUDGET,"seed":SEED,"warmup_iterations":WARMUP,"training_performed":False,"optimizer_created":False,"backward_called":False,"teacher_labels":False,"minimax_labels":False});write_json(a.output/"fingerprints.json",{"before":{"model":pool_fingerprints(),"engine":engine_fingerprint()},"after":None});write_json(a.output/"stage_state.json",{"stages":{str(n):"PENDING" for n in CONCURRENCIES}})

def correctness(a):
 d=device(a.device);_,states=load_positions(a.output/"benchmark_positions.json");subset=states[:4];seeds=[SEED+i for i in range(4)];cfg=lambda seed:MCTSConfig(num_simulations=32,c_puct=1.5,add_root_noise=False,seed=seed)
 sequential=[SongoMCTS(load_model(d),config=cfg(seed)).search(state,policy_temperature=0.) for state,seed in zip(subset,seeds)];model=load_model(d);parallel=SongoMCTS(model,config=cfg(SEED)).search_many(subset,policy_temperature=0.,seeds=seeds)
 rows=[]
 for x,y in zip(sequential,parallel):rows.append({"legal_mask_equal":x.legal_mask==y.legal_mask,"root_visits_equal":x.visit_counts==y.visit_counts,"root_policy_equal":np.allclose(x.policy,y.policy),"selected_action_equal":x.selected_action==y.selected_action,"simulation_counts_equal":x.num_simulations==y.num_simulations==32,"terminal_behavior_equal":x.selected_action is None if x.num_simulations==0 else y.selected_action is not None})
 write_json(a.output/"correctness.json",{"positions":4,"simulation_budget":32,"all_coherent":all(all(r.values()) for r in rows),"rows":rows})

def percentile(values,q):
 return float(np.percentile(np.asarray(values,dtype=float),q)) if values else 0.
def gpu_utilization():
 try:
  value=subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"],text=True,stderr=subprocess.DEVNULL,timeout=5).splitlines()[0];return float(value.strip())
 except Exception:return None
def warmup(model,d,states):
 builder=SongoGraphBuilder();sample=states[:min(256,len(states))];graph=builder.build_batch(sample).to(d)
 with torch.inference_mode():
  for _ in range(WARMUP):model(graph)
 if d.type=="cuda":torch.cuda.synchronize(d)

def run(a):
 if a.concurrency is None:raise ValueError("run requires --concurrency")
 path=a.output/f"scaling_{a.concurrency}.json"
 if valid_stage(path,a.concurrency):print(f"SKIP COMPLETE {a.concurrency}");return
 partial_path=a.output/f"scaling_{a.concurrency}.partial.json";d=device(a.device);meta,states=load_positions(a.output/"benchmark_positions.json");model=load_model(d);before=model_parameter_fingerprint(model);warmup(model,d,states)
 if d.type=="cuda":torch.cuda.reset_peak_memory_stats(d)
 results=[];profiles=[];wave_times=[];gpu_samples=[];previous_wall=0.;previous_cpu=0.
 if partial_path.is_file():
  partial=json.load(partial_path.open())
  compatible=resume_code_compatible(partial.get("git_commit")) and partial.get("measurement_schema_version",1)==MEASUREMENT_SCHEMA_VERSION and partial.get("model_fingerprints")==pool_fingerprints() and partial.get("positions_fingerprint")==meta["fingerprint"] and partial.get("parallel_searches")==a.concurrency and partial.get("mcts_budget")==BUDGET
  if not compatible:raise RuntimeError(f"incompatible partial stage: {partial_path}")
  results=partial["results"];profiles=partial["profiles"];wave_times=partial["wave_times"];gpu_samples=partial["gpu_samples"];previous_wall=float(partial["wall_clock_seconds"]);previous_cpu=float(partial["process_cpu_time_s"])
 start_index=len(results)
 if start_index%a.concurrency or start_index>len(states):raise RuntimeError("invalid partial wave boundary")
 total_waves=math.ceil(len(states)/a.concurrency);completed_waves=start_index//a.concurrency
 print(f"[lot39 N={a.concurrency}] resume={completed_waves}/{total_waves} waves; positions={start_index}/{len(states)}",flush=True)
 wall_started=time.perf_counter();cpu_started=time.process_time()
 with torch.inference_mode():
  for wave_index,start in enumerate(range(start_index,len(states),a.concurrency),completed_waves+1):
   wave=states[start:start+a.concurrency];seeds=[SEED+i for i in range(start,start+len(wave))];search=SongoMCTS(model,config=MCTSConfig(num_simulations=BUDGET,c_puct=1.5,add_root_noise=False,seed=seeds[0]));t=time.perf_counter();found=search.search_many(wave,policy_temperature=0.,seeds=seeds);elapsed=time.perf_counter()-t;wave_times.append(elapsed);profiles.append(dict(search.last_profile));gpu_samples.append(gpu_utilization())
   for offset,r in enumerate(found):results.append({"index":start+offset,"simulations":r.num_simulations,"network_evaluations":r.network_evaluations,"nodes":r.num_nodes,"selected_action":r.selected_action,"visit_counts":list(r.visit_counts),"legal_mask":list(r.legal_mask),"wave_elapsed_s":elapsed})
   cumulative_wall=previous_wall+time.perf_counter()-wall_started;cumulative_cpu=previous_cpu+time.process_time()-cpu_started
   partial={"status":"PARTIAL","git_commit":git_commit(),"measurement_schema_version":MEASUREMENT_SCHEMA_VERSION,"resume_critical_files":list(RESUME_CRITICAL_FILES),"model_fingerprints":pool_fingerprints(),"positions_fingerprint":meta["fingerprint"],"parallel_searches":a.concurrency,"mcts_budget":BUDGET,"completed_waves":wave_index,"total_waves":total_waves,"completed_positions":len(results),"wall_clock_seconds":cumulative_wall,"process_cpu_time_s":cumulative_cpu,"results":results,"profiles":profiles,"wave_times":wave_times,"gpu_samples":gpu_samples}
   write_json(partial_path,partial);write_json(a.output/"stage_state.json",{"stages":{str(n):("COMPLETE" if valid_stage(a.output/f"scaling_{n}.json",n) else "RUNNING" if n==a.concurrency else "PENDING") for n in CONCURRENCIES},"active_stage":a.concurrency,"completed_waves":wave_index,"total_waves":total_waves});print(f"[lot39 N={a.concurrency}] wave {wave_index}/{total_waves} complete in {elapsed:.1f}s; positions={len(results)}/{len(states)}; checkpoint={partial_path.name}",flush=True)
 total=previous_wall+time.perf_counter()-wall_started;cpu_time=previous_cpu+time.process_time()-cpu_started;after=model_parameter_fingerprint(model);batches=[int(x) for p in profiles for x in p["effective_batch_sizes"]];profile_keys=("engine_s","tree_selection_s","graph_construction_s","tensor_preparation_s","host_to_device_s","model_forward_wall_s","device_to_host_s","backup_s","batch_coordination_s");profile={k:sum(float(p[k]) for p in profiles) for k in profile_keys};accounted=sum(profile.values());profile["other_s"]=max(0.,total-accounted);profile["percent"]={k:100*v/total for k,v in profile.items() if k.endswith("_s")};total_sims=sum(r["simulations"] for r in results);evals=sum(r["network_evaluations"] for r in results);latencies=[r["wave_elapsed_s"] for r in results]
 payload={"status":"COMPLETE","git_commit":git_commit(),"model_fingerprints":pool_fingerprints(),"positions_fingerprint":meta["fingerprint"],"device":str(d),"parallel_searches":a.concurrency,"positions":len(results),"waves":math.ceil(len(states)/a.concurrency),"mcts_budget":BUDGET,"requested_simulations_per_tree":BUDGET,"actual_simulations_per_tree":{"min":min(r["simulations"] for r in results),"max":max(r["simulations"] for r in results)},"total_simulations":total_sims,"wall_clock_seconds":total,"global_simulations_per_second":total_sims/total,"positions_per_second":len(results)/total,"mean_position_latency":statistics.fmean(latencies),"median_position_latency":statistics.median(latencies),"p95_position_latency":percentile(latencies,95),"network_evaluations":evals,"network_calls":sum(len(p["effective_batch_sizes"]) for p in profiles),"mean_effective_batch_size":statistics.fmean(batches),"median_effective_batch_size":statistics.median(batches),"p95_effective_batch_size":percentile(batches,95),"max_effective_batch_size":max(batches),"mcts_nodes":sum(r["nodes"] for r in results),"srn_states_per_second":evals/total,"gpu_utilization_samples":gpu_samples,"gpu_utilization_mean":statistics.fmean(x for x in gpu_samples if x is not None) if any(x is not None for x in gpu_samples) else None,"gpu_memory_allocated":torch.cuda.memory_allocated(d) if d.type=="cuda" else None,"gpu_memory_reserved":torch.cuda.memory_reserved(d) if d.type=="cuda" else None,"peak_gpu_memory":torch.cuda.max_memory_allocated(d) if d.type=="cuda" else None,"peak_ram_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"process_cpu_time_s":cpu_time,"cpu_utilization_normalized_pct":100*cpu_time/total/max(1,os.cpu_count() or 1),"active_cpu_threads":torch.get_num_threads(),"warmup_performed":True,"warmup_iterations":WARMUP,"profile":profile,"model_weights_changed":before!=after,"results":results}
 if payload["actual_simulations_per_tree"]!={"min":BUDGET,"max":BUDGET}:raise RuntimeError("per-tree simulation budget mismatch")
 write_stage(path,payload);stage={"stages":{str(n):("COMPLETE" if valid_stage(a.output/f"scaling_{n}.json",n) else "PENDING") for n in CONCURRENCIES},"last_completed":a.concurrency};write_json(a.output/"stage_state.json",stage);print(json.dumps({k:payload[k] for k in payload if k not in ("results","profile")},indent=2),flush=True)

def svg_chart(rows,key,title,path):
 width,height,pad=760,420,60;values=[float(r[key]) for r in rows];maximum=max(values) or 1.;points=[]
 for i,value in enumerate(values):
  x=pad+i*(width-2*pad)/(len(values)-1);y=height-pad-value/maximum*(height-2*pad);points.append((x,y))
 circles="".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb"/><text x="{x:.1f}" y="{height-25}" text-anchor="middle">{rows[i]["parallel_searches"]}</text>' for i,(x,y) in enumerate(points));poly=" ".join(f"{x:.1f},{y:.1f}" for x,y in points);labels="".join(f'<text x="{x:.1f}" y="{y-12:.1f}" text-anchor="middle">{values[i]:.1f}</text>' for i,(x,y) in enumerate(points));path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="{width/2}" y="28" text-anchor="middle" font-size="18">{title}</text><line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/><polyline points="{poly}" fill="none" stroke="#2563eb" stroke-width="3"/>{circles}{labels}</svg>')

def write_report(rows,decision):
 docs=Path("docs");docs.mkdir(exist_ok=True);throughput=docs/"lot39_parallel_searches_vs_throughput.svg";batches=docs/"lot39_parallel_searches_vs_batch.svg";svg_chart(rows,"global_simulations_per_second","Lot39 — débit MCTS global",throughput);svg_chart(rows,"mean_effective_batch_size","Lot39 — batch effectif moyen",batches)
 table="\n".join(f'| {r["parallel_searches"]} | {r["global_simulations_per_second"]:.2f} | {r["speedup_vs_n16"]:.3f} | {r["speedup_vs_cpu_lot38"]:.3f} | {r["mean_effective_batch_size"]:.2f} | {r["p95_effective_batch_size"]:.2f} |' for r in rows)
 text=f'''# Lot 39 — GPU Parallel MCTS Scaling

Ce rapport est généré exclusivement depuis les artefacts JSON du Lot 39. Il mesure l'infrastructure, pas la force de G4.

| Recherches | Simulations/s | Speedup N=16 | Speedup CPU Lot38 | Batch moyen | Batch p95 |
|---:|---:|---:|---:|---:|---:|
{table}

![Débit](lot39_parallel_searches_vs_throughput.svg)

![Batch](lot39_parallel_searches_vs_batch.svg)

## Décision

```json
{json.dumps(decision,indent=2)}
```

Aucun entraînement, optimiseur, backward ou changement des poids n'a été effectué.
''';(docs/"srn_gpu_parallel_scaling_lot39.md").write_text(text)

def finalize(a):
 paths=[a.output/f"scaling_{n}.json" for n in CONCURRENCIES]
 if not all(valid_stage(p,n) for p,n in zip(paths,CONCURRENCIES)):raise RuntimeError("all concurrency stages must be COMPLETE and valid")
 rows=[json.load(p.open()) for p in paths];base=rows[0]["global_simulations_per_second"]
 for r in rows:r["speedup_vs_n16"]=r["global_simulations_per_second"]/base;r["scaling_efficiency"]=r["speedup_vs_n16"]/(r["parallel_searches"]/16);r["speedup_vs_cpu_lot38"]=r["global_simulations_per_second"]/CPU_REFERENCE
 raw=[{k:v for k,v in r.items() if k!="results"} for r in rows];write_json(a.output/"raw_scaling_results.json",raw)
 best=max(rows,key=lambda r:r["global_simulations_per_second"]);increments=[rows[i]["global_simulations_per_second"]/rows[i-1]["global_simulations_per_second"]-1 for i in range(1,len(rows))];saturation=next((rows[i]["parallel_searches"] for i,g in enumerate(increments,1) if g<.05),"ABOVE_256")
 pct=best["profile"]["percent"];cpu_pct=100-pct.get("model_forward_wall_s",0.);starved="YES" if cpu_pct>=50 else "NO";categories={"SRN_INFERENCE":pct.get("model_forward_wall_s",0.),"GRAPH_CONSTRUCTION":pct.get("graph_construction_s",0.),"ENGINE":pct.get("engine_s",0.),"TREE_MANAGEMENT":pct.get("tree_selection_s",0.)+pct.get("backup_s",0.),"CPU_COORDINATION":pct.get("batch_coordination_s",0.)+pct.get("other_s",0.),"DATA_TRANSFER":pct.get("host_to_device_s",0.)+pct.get("device_to_host_s",0.)};rank=sorted(categories,key=categories.get,reverse=True)
 decision={"GPU_PARALLEL_SCALING_VALID":"YES","BEST_TESTED_CONCURRENCY":best["parallel_searches"],"BEST_GLOBAL_SIMULATIONS_PER_SECOND":best["global_simulations_per_second"],"BEST_SPEEDUP_VS_CPU_REFERENCE":best["speedup_vs_cpu_lot38"],"GPU_SATURATION_POINT":saturation,"GPU_STARVED_BY_CPU":starved,"PRIMARY_RUNTIME_BOTTLENECK":rank[0],"SECONDARY_RUNTIME_BOTTLENECK":rank[1],"DEEP_MCTS_GPU_INFRASTRUCTURE_READY":"YES","NEXT_CONCURRENCY_TEST":512 if saturation=="ABOVE_256" and increments[-1]>=.15 else None,"NEXT_ACTION":"GPU_PARALLEL_SCALING_EXTENSION_512" if saturation=="ABOVE_256" and increments[-1]>=.15 else "MCTS_CPU_PIPELINE_OPTIMIZATION" if starved=="YES" else "DEEP_MCTS_TARGET_CONVERGENCE_STUDY","TRAINING_PERFORMED":"NO","OPTIMIZER_CREATED":"NO","BACKWARD_CALLED":"NO","MODEL_WEIGHTS_CHANGED":"NO" if all(not r["model_weights_changed"] for r in rows) else "YES"};write_json(a.output/"decision.json",decision)
 write_json(a.output/"profiling.json",{"by_concurrency":{str(r["parallel_searches"]):r["profile"] for r in rows}});write_json(a.output/"gpu_metrics.json",{"by_concurrency":{str(r["parallel_searches"]):{k:r[k] for k in ("gpu_utilization_mean","gpu_memory_allocated","gpu_memory_reserved","peak_gpu_memory")} for r in rows}});write_json(a.output/"cpu_bottleneck.json",{"gpu_starved_by_cpu":starved,"best_concurrency_cpu_time_s":best["process_cpu_time_s"],"best_profile_percent":pct})
 headers=("parallel_searches","positions","mcts_budget","total_simulations","wall_clock_seconds","global_simulations_per_second","positions_per_second","mean_effective_batch_size","p95_effective_batch_size","max_effective_batch_size","gpu_utilization_mean","peak_gpu_memory")
 with (a.output/"scaling_summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows({k:r.get(k) for k in headers} for r in rows)
 after={"model":pool_fingerprints(),"engine":engine_fingerprint()};fp=json.load((a.output/"fingerprints.json").open());fp["after"]=after;fp["unchanged"]=fp["before"]==after;write_json(a.output/"fingerprints.json",fp);write_json(a.output/"microbenchmarks.json",{"phase_b_triggered":cpu_pct>=50,"reason":"targeted microbenchmarks deferred to optimization lot; Phase A profile is authoritative"});write_json(a.output/"report.json",{"lot":39,"decision":decision,"raw_scaling_results":raw,"cpu_reference_lot38":CPU_REFERENCE});print(json.dumps(decision,indent=2))
 write_report(rows,decision)

def export(a):
 required=("decision.json","report.json","raw_scaling_results.json","scaling_summary.csv","correctness.json")
 missing=[x for x in required if not (a.output/x).is_file()]
 if missing:raise RuntimeError(f"cannot export incomplete Lot39: {missing}")
 files=[p for p in sorted(a.output.iterdir()) if p.is_file() and p.name not in ("experiment_manifest.json","checksums.json")];checks={p.name:sha256(p) for p in files};write_json(a.output/"checksums.json",checks);files.append(a.output/"checksums.json");manifest={"experiment_id":"lot39_gpu_parallel_scaling","lot":39,"git_commit":git_commit(),"model_fingerprints":pool_fingerprints(),"engine_fingerprint":engine_fingerprint(),"artifact_checksums":{p.name:sha256(p) for p in files}};write_json(a.output/"experiment_manifest.json",manifest);files.append(a.output/"experiment_manifest.json")
 with tarfile.open(a.bundle,"w:gz") as tf:
  for p in files:tf.add(p,arcname=f"lot39_gpu_parallel_scaling/{p.name}")
 Path(str(a.bundle)+".sha256").write_text(f"{sha256(a.bundle)}  {a.bundle.name}\n");print(a.bundle)

if __name__=="__main__":
 a=args();{"prepare":prepare,"correctness":correctness,"run":run,"finalize":finalize,"export":export}[a.stage](a)
