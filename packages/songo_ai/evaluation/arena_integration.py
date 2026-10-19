"""Stage C0.2 integration layer. Safe by default; dependencies are injected."""
from __future__ import annotations
import hashlib, json, os, sqlite3, tempfile, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from .arena_runner import ArenaStore, Candidate, ExperimentManifest, GameResult, ScheduledGame, canonical, file_sha256, fingerprint, schedule

@dataclass(frozen=True)
class OpeningRecord:
 opening_id:str;state_reference:Mapping[str,Any];state_fingerprint:str;source:str;split_identity:str;order:int
 def validate(self)->None:
  if len(self.state_fingerprint)!=64 or fingerprint(self.state_reference)!=self.state_fingerprint:raise ValueError("opening state fingerprint mismatch")
  if not self.source or not self.split_identity or self.order<0:raise ValueError("incomplete opening provenance")

@dataclass(frozen=True)
class MCTSSettings:
 c_puct:float=1.5;root_noise:bool=False;temperature:float=0.0;search_batching:str="sequential";terminal_handling:str="engine_authoritative";legal_action_masking:bool=True
 def validate(self)->None:
  if self.c_puct<0 or self.root_noise or self.temperature!=0.0:raise ValueError("incompatible deterministic evaluation settings")
  if not self.legal_action_masking or self.terminal_handling!="engine_authoritative":raise ValueError("unsafe MCTS settings")

@dataclass(frozen=True)
class ExecutionAuthorization:
 experiment_id:str;experiment_fingerprint:str;approved:bool;authority_reference:str

@dataclass
class PerformanceMetrics:
 game_duration_s:float=0;search_duration_s:float=0;simulations:int=0;inference_duration_s:float=0;gpu_memory_bytes:int|None=None;peak_gpu_memory_bytes:int|None=None;batch_items:int=0;completed_games:int=0;pending_games:int=0;resume_overhead_s:float=0
 @property
 def simulations_per_second(self)->float|None:return self.simulations/self.search_duration_s if self.search_duration_s>0 else None

class ArenaAdapter:
 def __init__(self,manifest:ExperimentManifest,openings:Mapping[str,OpeningRecord],store:ArenaStore,*,model_loader:Callable[[Candidate],Any],game_executor:Callable[...,Mapping[str,Any]],engine_fingerprint:str,mcts_settings:MCTSSettings):
  self.manifest=manifest;self.openings=dict(openings);self.store=store;self.model_loader=model_loader;self.game_executor=game_executor;self.engine_fingerprint=engine_fingerprint;self.settings=mcts_settings
 def validate(self,checkpoint_root:Path|None=None)->dict:
  self.settings.validate()
  if self.manifest.engine_fingerprint!=self.engine_fingerprint:raise ValueError("engine fingerprint mismatch")
  games=schedule(self.manifest)
  if set(self.manifest.opening_ids)!=set(self.openings):raise ValueError("opening set mismatch")
  for o in self.openings.values():o.validate()
  for c in self.manifest.candidates:c.validate(checkpoint_root)
  self.store.initialize(self.manifest,games)
  return {"dry_run":True,"experiment_fingerprint":self.manifest.fingerprint,"schedule_fingerprint":fingerprint([asdict(g) for g in games]),"planned_games":len(games),"pending_games":len(self.store.pending())}
 def execute_one(self,game:ScheduledGame,authorization:ExecutionAuthorization|None,*,checkpoint_root:Path)->str:
  if authorization is None or not authorization.approved or authorization.experiment_id!=self.manifest.experiment_id or authorization.experiment_fingerprint!=self.manifest.fingerprint or not authorization.authority_reference:raise PermissionError("scientific execution authorization rejected")
  self.validate(checkpoint_root);opening=self.openings[game.opening_id];candidates={c.name:c for c in self.manifest.candidates}
  p1=self.model_loader(candidates[game.candidate_p1]);p2=self.model_loader(candidates[game.candidate_p2])
  raw=self.game_executor(game=game,opening=opening,p1=p1,p2=p2,settings=self.settings)
  result=GameResult(**{**game.__dict__,**raw});return self.store.complete(result)

def cuda_readiness(torch_module:Any,*,requested_device:str="cuda",precision:str="fp32")->dict:
 available=bool(torch_module.cuda.is_available());report={"status":"READY" if available else "GPU_NOT_AVAILABLE","torch_version":str(torch_module.__version__),"cuda_runtime_version":str(getattr(torch_module.version,"cuda",None)),"selected_device":requested_device,"precision":precision,"gpu_name":None,"available_gpu_memory":None}
 if available:
  report["gpu_name"]=torch_module.cuda.get_device_name(0)
  if hasattr(torch_module.cuda,"mem_get_info"):report["available_gpu_memory"]=int(torch_module.cuda.mem_get_info(0)[0])
 if requested_device=="cuda" and not available:report["checkpoint_loading_compatibility"]="BLOCKED"
 else:report["checkpoint_loading_compatibility"]="TO_BE_VALIDATED_WITH_VERIFIED_CHECKPOINT"
 return report

class SQLiteSnapshotManager:
 def __init__(self,directory:Path,experiment_fingerprint:str):self.directory=directory;self.fp=experiment_fingerprint;directory.mkdir(parents=True,exist_ok=True)
 def publish(self,store:ArenaStore,generation:int)->dict:
  if generation<0:raise ValueError("invalid generation")
  target=self.directory/f"arena-{generation:08d}.sqlite";meta=target.with_suffix(".json")
  if target.exists() or meta.exists():raise FileExistsError(target)
  with tempfile.TemporaryDirectory(dir=self.directory) as td:
   tmp=Path(td)/"snapshot.sqlite";dest=sqlite3.connect(tmp);store.db.backup(dest);dest.close()
   check=sqlite3.connect(tmp).execute("PRAGMA integrity_check").fetchone()[0]
   if check!="ok":raise ValueError("snapshot integrity failure")
   digest=file_sha256(tmp);tmp_meta={"schema":1,"generation":generation,"experiment_fingerprint":self.fp,"sha256":digest}
   os.replace(tmp,target);meta.write_bytes(canonical(tmp_meta))
  return tmp_meta|{"path":str(target)}
 def restore(self,generation:int,destination:Path)->None:
  source=self.directory/f"arena-{generation:08d}.sqlite";meta=json.loads(source.with_suffix(".json").read_bytes())
  if meta["experiment_fingerprint"]!=self.fp or meta["generation"]!=generation or file_sha256(source)!=meta["sha256"]:raise ValueError("corrupt, stale, or foreign snapshot")
  if destination.exists():raise FileExistsError(destination)
  src=sqlite3.connect(source);dst=sqlite3.connect(destination);src.backup(dst);dst.close();src.close()
