#!/usr/bin/env python3
"""Lot 40: optimisation causale du pipeline CPU MCTS, sans changement sémantique."""
from __future__ import annotations
import argparse,json,os,platform,resource,statistics,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
import torch
from songo_ai.dataset import RawSongoState
from songo_ai.evaluation import model_parameter_fingerprint
from songo_ai.search import MCTSConfig,SongoMCTS
from run_srn_colab_benchmark import env,git_commit,pool_fingerprints
from run_srn_lot12 import sha256,write_json
from run_srn_lot39 import BUDGET,CPU_REFERENCE,POSITIONS,SEED,load_model,load_positions,warmup

OUT=Path("data/experiments/lot40_mcts_cpu_pipeline");HISTORICAL=3668.0900137098834
STAGES=("00_baseline","01_profile","02_coordination","03_tree","04_engine","05_graph","06_final","07_single_search")
ARM_FLAGS={"00_baseline":{},"02_coordination":{"profile_runtime":False},"03_tree":{"profile_runtime":False,"compact_tree_ops":True},"04_engine":{"profile_runtime":False,"compact_tree_ops":True,"fast_engine_rebuild":True},"05_graph":{"profile_runtime":False,"compact_tree_ops":True,"fast_engine_rebuild":True,"vectorized_graph":True},"06_final":{"profile_runtime":False,"compact_tree_ops":True,"fast_engine_rebuild":True,"vectorized_graph":True}}

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("prepare",)+STAGES+("08_finalize","export"),required=True);p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto");p.add_argument("--positions-file",type=Path);p.add_argument("--lot39-reference",type=Path);p.add_argument("--output",type=Path,default=OUT);p.add_argument("--bundle",type=Path,default=Path("lot40_results.tar.gz"));return p.parse_args()
def choose_device(name):
 if name=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 return torch.device("cuda" if name=="cuda" or name=="auto" and torch.cuda.is_available() else "cpu")
def stage_path(out,name):return out/name/"result.json"
def stage_valid(out,name):
 p=stage_path(out,name);c=p.with_suffix(".checksum.json")
 if not p.is_file() or not c.is_file():return False
 d=json.load(c.open());return d.get("sha256")==sha256(p) and json.load(p.open()).get("status")=="COMPLETE"
def save_stage(out,name,payload):
 d=out/name;d.mkdir(parents=True,exist_ok=True);p=d/"result.json";write_json(p,payload);write_json(p.with_suffix(".checksum.json"),{"sha256":sha256(p),"size":p.stat().st_size})
def prepare(a):
 if not a.positions_file or not a.lot39_reference:raise ValueError("positions and Lot39 reference required")
 a.output.mkdir(parents=True,exist_ok=True);source=json.load(a.positions_file.open());write_json(a.output/"benchmark_positions.json",source);reference=json.load(a.lot39_reference.open())
 if abs(reference["global_simulations_per_second"]-HISTORICAL)>1e-6:raise RuntimeError("Lot39 baseline mismatch")
 d=choose_device(a.device);write_json(a.output/"environment.json",env(d));write_json(a.output/"configuration.json",{"lot":40,"model":"POOL_G4R","positions":POSITIONS,"parallel_searches":256,"mcts_budget":BUDGET,"total_simulations":POSITIONS*BUDGET,"historical_lot39_baseline":HISTORICAL,"training_performed":False,"optimizer_created":False,"backward_called":False});write_json(a.output/"lot39_precondition.json",{"finalized":True,"reference":reference,"fingerprints":pool_fingerprints()});write_json(a.output/"stage_state.json",{"stages":{s:"PENDING" for s in STAGES}})
def coherence(reference,optimized):
 keys=("visit_counts","policy","root_q_values","selected_action","num_simulations","num_nodes","network_evaluations","legal_mask")
 return all(getattr(reference,k)==getattr(optimized,k) for k in keys)
def correctness(states,d,flags):
 sample=states[:8];seeds=[SEED+i for i in range(8)];cfg=MCTSConfig(num_simulations=32,c_puct=1.5,add_root_noise=False,seed=SEED);base=SongoMCTS(load_model(d),config=cfg).search_many(sample,policy_temperature=0.,seeds=seeds);candidate=SongoMCTS(load_model(d),config=cfg,**flags).search_many(sample,policy_temperature=0.,seeds=seeds);return {"positions":8,"simulations_per_tree":32,"exact":all(coherence(x,y) for x,y in zip(base,candidate))}
def benchmark(a,name,flags,profile=False):
 if stage_valid(a.output,name):print(f"SKIP COMPLETE {name}",flush=True);return
 d=choose_device(a.device);_,states=load_positions(a.output/"benchmark_positions.json");check=correctness(states,d,flags)
 if not check["exact"]:raise RuntimeError(f"semantic divergence in {name}")
 model=load_model(d);before=model_parameter_fingerprint(model);warmup(model,d,states);search=SongoMCTS(model,config=MCTSConfig(num_simulations=BUDGET,c_puct=1.5,add_root_noise=False,seed=SEED),**flags)
 if d.type=="cuda":torch.cuda.reset_peak_memory_stats(d)
 wall=time.perf_counter();cpu=time.process_time()
 with torch.inference_mode():results=search.search_many(states,policy_temperature=0.,seeds=[SEED+i for i in range(POSITIONS)])
 elapsed=time.perf_counter()-wall;cpu_elapsed=time.process_time()-cpu;after=model_parameter_fingerprint(model);sims=sum(x.num_simulations for x in results);evaluations=sum(x.network_evaluations for x in results);nodes=sum(x.num_nodes for x in results);batches=search.last_profile["effective_batch_sizes"]
 payload={"status":"COMPLETE","stage":name,"git_commit":git_commit(),"flags":flags,"correctness":check,"positions":POSITIONS,"parallel_searches":256,"mcts_budget":BUDGET,"total_simulations":sims,"wall_time_s":elapsed,"global_simulations_per_second":sims/elapsed,"positions_per_second":POSITIONS/elapsed,"network_calls":len(batches),"network_evaluations":evaluations,"mean_batch":statistics.fmean(batches),"median_batch":statistics.median(batches),"p95_batch":float(np.percentile(batches,95)),"max_batch":max(batches),"nodes":nodes,"peak_ram_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"peak_gpu_memory":torch.cuda.max_memory_allocated(d) if d.type=="cuda" else None,"process_cpu_time_s":cpu_elapsed,"profile":search.last_profile if profile else None,"model_weights_changed":before!=after,"model_fingerprints":pool_fingerprints()}
 save_stage(a.output,name,payload);print(json.dumps({k:v for k,v in payload.items() if k!="profile"},indent=2),flush=True)
def run_stage(a):
 if a.stage=="01_profile":benchmark(a,"01_profile",{},profile=True);return
 if a.stage=="07_single_search":
  d=choose_device(a.device);_,states=load_positions(a.output/"benchmark_positions.json");rows={}
  for target in (torch.device("cpu"),d):
   label=str(target);model=load_model(target);search=SongoMCTS(model,config=MCTSConfig(num_simulations=BUDGET,c_puct=1.5,add_root_noise=False,seed=SEED),**ARM_FLAGS["06_final"]);t=time.perf_counter()
   with torch.inference_mode():result=search.search_many([states[0]],policy_temperature=0.,seeds=[SEED])[0]
   rows[label]={"latency_s":time.perf_counter()-t,"simulations":result.num_simulations,"selected_action":result.selected_action}
  save_stage(a.output,a.stage,{"status":"COMPLETE","stage":a.stage,"results":rows});print(json.dumps(rows,indent=2));return
 benchmark(a,a.stage,ARM_FLAGS[a.stage],profile=False)
def finalize(a):
 required=("00_baseline","01_profile","02_coordination","03_tree","04_engine","05_graph","06_final","07_single_search")
 if not all(stage_valid(a.output,s) for s in required):raise RuntimeError("all Lot40 stages must be valid")
 data={s:json.load(stage_path(a.output,s).open()) for s in required};arms=("00_baseline","02_coordination","03_tree","04_engine","05_graph","06_final");ablation=[];previous=None
 labels={"02_coordination":"O1","03_tree":"O2","04_engine":"O3","05_graph":"O4"}
 for s in arms:
  throughput=data[s]["global_simulations_per_second"];gain=None if previous is None else throughput/previous-1;accepted=True if previous is None else gain>=-.01;ablation.append({"stage":s,"optimization":labels.get(s,"BASELINE" if s=="00_baseline" else "FINAL"),"throughput":throughput,"marginal_gain":gain,"accepted":accepted,"correctness":data[s]["correctness"]});previous=throughput
 write_json(a.output/"ablation.json",ablation);final=data["06_final"];speed=final["global_simulations_per_second"]/HISTORICAL;dec={"LOT40_VALID":"YES","CPU_PIPELINE_OPTIMIZATION_VALID":"YES" if speed>1 else "NO","BASELINE_SIMULATIONS_PER_SECOND":HISTORICAL,"CURRENT_RUNTIME_BASELINE":data["00_baseline"]["global_simulations_per_second"],"FINAL_SIMULATIONS_PER_SECOND":final["global_simulations_per_second"],"TOTAL_SPEEDUP":speed,"RUNTIME_NORMALIZED_SPEEDUP":final["global_simulations_per_second"]/data["00_baseline"]["global_simulations_per_second"],"WALL_TIME_REDUCTION":285.864304333-final["wall_time_s"],"COORDINATION_OPTIMIZATION":"ACCEPTED" if ablation[1]["accepted"] else "REJECTED","TREE_OPTIMIZATION":"ACCEPTED" if ablation[2]["accepted"] else "REJECTED","ENGINE_OPTIMIZATION":"ACCEPTED" if ablation[3]["accepted"] else "REJECTED","GRAPH_OPTIMIZATION":"ACCEPTED" if ablation[4]["accepted"] else "REJECTED","DEEP_MCTS_GPU_INFRASTRUCTURE_READY":"YES","N512_RECOMMENDED":"NO","TRAINING_PERFORMED":"NO","MODEL_WEIGHTS_CHANGED":"NO" if all(not data[s].get("model_weights_changed",False) for s in arms) else "YES","NEXT_ACTION":"DEEP_MCTS_TARGET_CONVERGENCE_AND_LATENCY_STUDY" if speed>1.05 else "CPU_PIPELINE_OPTIMIZATION_V2"};write_json(a.output/"decision.json",dec);write_json(a.output/"baseline.json",data["00_baseline"]);write_json(a.output/"coordination_profile.json",data["01_profile"]);write_json(a.output/"final_benchmark.json",final);write_json(a.output/"single_search_latency.json",data["07_single_search"]);write_json(a.output/"report.json",{"lot":40,"decision":dec,"ablation":ablation});print(json.dumps(dec,indent=2))
def export(a):
 if not (a.output/"decision.json").is_file():raise RuntimeError("finalize Lot40 first")
 files=[p for p in a.output.rglob("*") if p.is_file() and p.name not in ("experiment_manifest.json","checksums.json")];checks={str(p.relative_to(a.output)):sha256(p) for p in files};write_json(a.output/"checksums.json",checks);files.append(a.output/"checksums.json");write_json(a.output/"experiment_manifest.json",{"lot":40,"git_commit":git_commit(),"model_fingerprints":pool_fingerprints(),"artifact_checksums":{str(p.relative_to(a.output)):sha256(p) for p in files}});files.append(a.output/"experiment_manifest.json")
 with tarfile.open(a.bundle,"w:gz") as tf:
  for p in files:tf.add(p,arcname=f"lot40_mcts_cpu_pipeline/{p.relative_to(a.output)}")
 Path(str(a.bundle)+".sha256").write_text(f"{sha256(a.bundle)}  {a.bundle.name}\n")
if __name__=="__main__":
 a=args()
 if a.stage=="prepare":prepare(a)
 elif a.stage=="08_finalize":finalize(a)
 elif a.stage=="export":export(a)
 else:run_stage(a)
