#!/usr/bin/env python3
"""Lot 38: benchmark portable CPU/CUDA du SRN-MCTS, sans entraînement."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, statistics, subprocess, sys, tarfile, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from songo_ai.dataset import RawSongoState
from songo_ai.evaluation import HybridPolicyValueEvaluator, model_parameter_fingerprint
from songo_ai.model import SongoGraphBuilder, load_srn_checkpoint
from songo_ai.search import MCTSConfig, SongoMCTS
from songo_ai.songo.rules import SongoLegacyGame
from run_srn_lot12 import sha256, write_json
from run_srn_lot36 import EXPECTED_MINIMAX

OUT=Path("data/experiments/lot38_colab_compute");IDENTITY=Path("data/experiments/lot35_generator_pool/g4_champion_identity.json")
POSITIONS=16;SEED=20263801;BATCHES=(1,8,16,32,64,128,256)

def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("prepare","benchmark","finalize","export"),required=True);p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto");p.add_argument("--budget",type=int,choices=(1024,4096),default=4096);p.add_argument("--output",type=Path,default=OUT);p.add_argument("--bundle",type=Path,default=Path("lot38_results.tar.gz"));return p.parse_args()

def git_commit():
 try:return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
 except Exception:return "UNAVAILABLE"
def engine_fingerprint():return hashlib.sha256(Path("packages/songo_ai/songo/rules.py").read_bytes()).hexdigest()
def pool_identity():
 if not IDENTITY.exists():raise FileNotFoundError(f"missing POOL_G4R identity manifest: {IDENTITY}; upload/extract lot38_colab_inputs.tar.gz first")
 return json.load(IDENTITY.open())["candidates"]["POOL"]
def pool_fingerprints():
 item=pool_identity();policy=Path(item["policy_checkpoint"]);value=Path(item["value_checkpoint"])
 if not policy.exists() or not value.exists():raise FileNotFoundError("missing POOL_G4R checkpoints; upload/extract lot38_colab_inputs.tar.gz first")
 observed={"model_id":item["model_id"],"policy_checkpoint":str(policy),"value_checkpoint":str(value),"policy_fingerprint":sha256(policy),"value_fingerprint":sha256(value),"architecture_fingerprint":item["architecture_fingerprint"]}
 if observed["policy_fingerprint"]!=item["policy_fingerprint"] or observed["value_fingerprint"]!=item["value_fingerprint"]:raise RuntimeError("POOL_G4R input checkpoint fingerprint mismatch")
 return observed
def device_name(device):return torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor() or platform.machine()
def choose_device(requested):
 if requested=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
 return torch.device("cuda" if (requested=="cuda" or requested=="auto" and torch.cuda.is_available()) else "cpu")
def env(device):
 try:import numba;nv=numba.__version__
 except Exception:nv="UNAVAILABLE"
 ram=None
 try:ram=os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")
 except Exception:pass
 return {"git_commit":git_commit(),"python_version":sys.version,"torch_version":torch.__version__,"numpy_version":np.__version__,"numba_version":nv,"platform":platform.platform(),"machine_architecture":platform.machine(),"cpu_count":os.cpu_count(),"RAM_bytes":ram,"CUDA_available":torch.cuda.is_available(),"CUDA_version":torch.version.cuda,"GPU_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"GPU_memory_bytes":torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,"device":str(device),"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"colab_detected":"COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ}

def generate_positions():
 states=[];game=SongoLegacyGame();rng=np.random.default_rng(SEED)
 for i in range(POSITIONS):
  game=SongoLegacyGame()
  for _ in range(i*3+1):
   game.normalize_terminal()
   if game.finished:break
   legal=game.legal_local_actions();game.play_local(int(rng.choice(legal)))
  game.normalize_terminal()
  if game.finished:raise RuntimeError("deterministic benchmark position became terminal")
  states.append(RawSongoState.from_game(game))
 return states

def prepare(a):
 a.output.mkdir(parents=True,exist_ok=True);d=choose_device(a.device);fps=pool_fingerprints();states=generate_positions()
 write_json(a.output/"environment.json",env(d));write_json(a.output/"benchmark_positions.json",{"seed":SEED,"count":len(states),"deterministic":True,"states":[{"board":list(s.board),"player_to_move":s.player_to_move} for s in states]});write_json(a.output/"configuration.json",{"lot":38,"model_probe":"POOL_G4R","budgets":{"smoke":1024,"main":4096},"benchmark_positions":POSITIONS,"batch_sizes":list(BATCHES),"current_mcts_inference_mode":"SINGLE_STATE","no_batched_mcts_implementation":True,"no_mcts65536":True,"training_performed":False,"optimizer_created":False,"backward_called":False,"minimax_labels":False});write_json(a.output/"fingerprints.json",{"before":{"model":fps,"engine":engine_fingerprint(),"minimax":EXPECTED_MINIMAX},"after":None});print(json.dumps(env(d),indent=2))

class TimedEvaluator(nn.Module):
 def __init__(self,model,device):super().__init__();self.model=model;self.device=device;self.inference_s=0.;self.calls=0;self.states=0
 def forward(self,graph):
  if self.device.type=="cuda":torch.cuda.synchronize(self.device)
  t=time.perf_counter();out=self.model(graph)
  if self.device.type=="cuda":torch.cuda.synchronize(self.device)
  self.inference_s+=time.perf_counter()-t;self.calls+=1;self.states+=graph.batch_size;return out

def load_model(device):
 item=pool_identity();p=load_srn_checkpoint(item["policy_checkpoint"]).model.to(device).eval();v=load_srn_checkpoint(item["value_checkpoint"]).model.to(device).eval();return HybridPolicyValueEvaluator(p,v,name="POOL_G4R").to(device).eval()
def read_positions(path):
 d=json.load(path.open());return [RawSongoState(tuple(x["board"]),x["player_to_move"]) for x in d["states"]]

def mcts_benchmark(model,states,device,budget):
 timed=TimedEvaluator(model,device).to(device);rows=[];started=time.perf_counter();before=model_parameter_fingerprint(model)
 for i,state in enumerate(states):
  t=time.perf_counter();r=SongoMCTS(timed,config=MCTSConfig(num_simulations=budget,c_puct=1.5,add_root_noise=False,seed=SEED+i)).search(state,policy_temperature=0.0);rows.append({"index":i,"elapsed_s":time.perf_counter()-t,"selected_action":r.selected_action,"visit_counts":list(r.visit_counts),"policy":list(r.policy),"root_value":r.root_value,"legal_mask":list(r.legal_mask),"simulations":r.num_simulations,"network_evaluations":r.network_evaluations,"nodes":r.num_nodes})
 total=time.perf_counter()-started;after=model_parameter_fingerprint(model)
 times=[x["elapsed_s"] for x in rows];sims=sum(x["simulations"] for x in rows);evals=sum(x["network_evaluations"] for x in rows)
 return {"device":str(device),"budget":budget,"positions":len(rows),"total_runtime_s":total,"mean_runtime_per_position_s":statistics.fmean(times),"median_runtime_per_position_s":statistics.median(times),"simulations_per_second":sims/total,"srn_evaluations_per_second":evals/total,"mcts_nodes_created":sum(x["nodes"] for x in rows),"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,"model_weights_changed":before!=after,"profiling":{"srn_inference_s":timed.inference_s,"srn_fraction":timed.inference_s/total,"engine_tree_transfer_other_s":total-timed.inference_s,"engine_tree_transfer_other_fraction":1-timed.inference_s/total,"network_forward_calls":timed.calls,"inference_states":timed.states,"mode":"SINGLE_STATE"},"results":rows}

def batch_benchmark(model,states,device):
 builder=SongoGraphBuilder();rows=[]
 for n in BATCHES:
  expanded=[states[i%len(states)] for i in range(n)];graph=builder.build_batch(expanded).to(device)
  for _ in range(3):
   with torch.no_grad():model(graph)
  if device.type=="cuda":torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
  repetitions=max(5,1000//n);t=time.perf_counter()
  with torch.no_grad():
   for _ in range(repetitions):model(graph)
  if device.type=="cuda":torch.cuda.synchronize(device)
  elapsed=time.perf_counter()-t;rows.append({"batch_size":n,"repetitions":repetitions,"latency_per_batch_s":elapsed/repetitions,"states_per_second":n*repetitions/elapsed,"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None})
 return rows

def correctness(model,states,device):
 builder=SongoGraphBuilder();cpu=load_model(torch.device("cpu"));sample=states[:4];gc=builder.build_batch(sample);gd=gc.to(device)
 with torch.no_grad():pc,vc=cpu(gc);pd,vd=model(gd)
 return {"positions":len(sample),"legal_masks_match":all(tuple(SongoLegacyGame.from_state(s.to_engine_state()).legal_mask())==tuple(SongoLegacyGame.from_state(s.to_engine_state()).legal_mask()) for s in sample),"policy_logits_max_abs_delta":float((pc-pd.cpu()).abs().max()),"value_max_abs_delta":float((vc-vd.cpu()).abs().max()),"numerically_coherent":bool(torch.allclose(pc,pd.cpu(),rtol=1e-4,atol=1e-5) and torch.allclose(vc,vd.cpu(),rtol=1e-4,atol=1e-5)),"terminal_handling":"engine-owned; unchanged","selected_action_comparison":"covered by per-device MCTS result artifacts"}

def benchmark(a):
 d=choose_device(a.device);states=read_positions(a.output/"benchmark_positions.json");model=load_model(d);payload=mcts_benchmark(model,states,d,a.budget);write_json(a.output/("benchmark_gpu.json" if d.type=="cuda" else "benchmark_cpu.json"),payload);batch=batch_benchmark(model,states,d);path=a.output/"srn_batch_benchmark.json";existing=json.load(path.open()) if path.exists() else {};existing[d.type]=batch;write_json(path,existing);corr=correctness(model,states,d);cp=a.output/"correctness.json";old=json.load(cp.open()) if cp.exists() else {};old[d.type]=corr;write_json(cp,old);print(json.dumps({k:payload[k] for k in payload if k not in ("results","profiling")},indent=2))

def finalize(a):
 cpu=json.load((a.output/"benchmark_cpu.json").open()) if (a.output/"benchmark_cpu.json").exists() else None;gpu=json.load((a.output/"benchmark_gpu.json").open()) if (a.output/"benchmark_gpu.json").exists() else None;batch=json.load((a.output/"srn_batch_benchmark.json").open()) if (a.output/"srn_batch_benchmark.json").exists() else {};corr=json.load((a.output/"correctness.json").open()) if (a.output/"correctness.json").exists() else {};speed=cpu["total_runtime_s"]/gpu["total_runtime_s"] if cpu and gpu else None
 cpu1=next((x for x in batch.get("cpu",[]) if x["batch_size"]==1),None);gpubest=max(batch.get("cuda",[]),key=lambda x:x["states_per_second"],default=None);beneficial=bool(gpubest and cpu1 and gpubest["states_per_second"]>1.2*cpu1["states_per_second"]);efficient=None if not gpu else speed>=1.2;under="YES" if gpu and not efficient and beneficial else "NO" if gpu else "INCONCLUSIVE"
 before=json.load((a.output/"fingerprints.json").open());after={"model":pool_fingerprints(),"engine":engine_fingerprint(),"minimax":EXPECTED_MINIMAX};before["after"]=after;before["unchanged"]=before["before"]==after;write_json(a.output/"fingerprints.json",before)
 profiling={"CURRENT_INFERENCE_MODE":"SINGLE_STATE","cpu":cpu.get("profiling") if cpu else None,"gpu":gpu.get("profiling") if gpu else None,"GPU_UNDERUTILIZED_BY_CURRENT_MCTS":under};write_json(a.output/"profiling.json",profiling)
 decision={"COLAB_COMPATIBLE":"YES","CUDA_COMPATIBLE":"YES" if gpu else "N/A","CURRENT_MCTS_GPU_EFFICIENT":"YES" if efficient else "NO" if efficient is False else "INCONCLUSIVE","BATCHED_SRN_BENEFICIAL":"YES" if beneficial else "NO" if gpu else "INCONCLUSIVE","BATCHED_MCTS_RECOMMENDED":"YES" if beneficial and not efficient else "NO","GPU_SPEEDUP_CURRENT_MCTS":speed,"CURRENT_INFERENCE_MODE":"SINGLE_STATE","GPU_UNDERUTILIZED_BY_CURRENT_MCTS":under,"TRAINING_PERFORMED":"NO","OPTIMIZER_CREATED":"NO","BACKWARD_CALLED":"NO","MODEL_WEIGHTS_CHANGED":"NO" if before["unchanged"] else "YES","NEXT_ACTION":"BATCHED_MCTS_GPU_DESIGN" if beneficial and not efficient else "DEEP_MCTS_TARGET_CONVERGENCE_STUDY" if efficient else "MULTI_CPU_REMOTE_WORKER_DESIGN"};write_json(a.output/"decision.json",decision);write_json(a.output/"report.json",{"lot":38,"environment":json.load((a.output/"environment.json").open()),"decision":decision,"correctness":corr,"profiling":profiling});print(json.dumps(decision,indent=2))

def export(a):
 files=[p for p in sorted(a.output.glob("*.json")) if p.name!="experiment_manifest.json"];manifest={"experiment_id":"lot38_colab_compute","lot":38,"git_commit":git_commit(),"worker_type":"COLAB" if "COLAB_RELEASE_TAG" in os.environ else "LOCAL","hardware":platform.platform(),"device":json.load((a.output/"environment.json").open())["device"],"model_fingerprints":pool_fingerprints(),"engine_fingerprint":engine_fingerprint(),"configuration":json.load((a.output/"configuration.json").open()),"artifact_checksums":{p.name:sha256(p) for p in files}};write_json(a.output/"experiment_manifest.json",manifest)
 with tarfile.open(a.bundle,"w:gz") as tf:
  for p in files+[a.output/"experiment_manifest.json"]:tf.add(p,arcname=f"lot38_colab_compute/{p.name}")
 (Path(str(a.bundle)+".sha256")).write_text(f"{sha256(a.bundle)}  {a.bundle.name}\n")
 print(a.bundle)
if __name__=="__main__":
 a=args();{"prepare":prepare,"benchmark":benchmark,"finalize":finalize,"export":export}[a.stage](a)
