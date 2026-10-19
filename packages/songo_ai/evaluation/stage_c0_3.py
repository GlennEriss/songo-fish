"""Read-only Stage C0.3 validation; never imports engine or MCTS."""
from __future__ import annotations
import hashlib, json, math, tarfile
from pathlib import Path
from typing import Any
from .arena_runner import ExperimentManifest, file_sha256, schedule

def verify_archive(path:Path,expected_sha256:str)->dict:
 if not path.is_file():raise FileNotFoundError(path)
 actual=file_sha256(path)
 if actual!=expected_sha256:raise ValueError("ARTIFACT_INTEGRITY_FAILURE")
 if not tarfile.is_tarfile(path):raise ValueError("invalid archive")
 with tarfile.open(path,"r:gz") as t:
  files=[m for m in t.getmembers() if m.isfile()];names=[m.name for m in files]
  if len(names)!=len(set(names)) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):raise ValueError("unsafe or duplicate archive member")
 return {"path":str(path),"sha256":actual,"members":len(names),"status":"PASS"}

def validate_model_outputs(policy:Any,value:Any,batch_size:int,actions:int=7)->dict:
 if tuple(policy.shape)!=(batch_size,actions) or tuple(value.shape)!=(batch_size,):raise ValueError("model output shape mismatch")
 finite_policy=bool(policy.isfinite().all().item());finite_value=bool(value.isfinite().all().item())
 if not finite_policy or not finite_value or not bool(((value>=-1)&(value<=1)).all().item()):raise ValueError("invalid numerical model output")
 return {"policy_shape":list(policy.shape),"value_shape":list(value.shape),"finite":True,"value_range_valid":True}

def validate_legal_mask(policy:Any,mask:Any)->None:
 if tuple(policy.shape)!=tuple(mask.shape) or not bool(mask.any(dim=-1).all().item()):raise ValueError("invalid legal mask contract")

def validate_c1_schedule(manifest:ExperimentManifest)->dict:
 games=schedule(manifest);comparisons={g.comparison_id for g in games}
 if len(comparisons)!=6 or len(manifest.opening_ids)!=400 or len(games)!=4800:raise ValueError("frozen C1 schedule mismatch")
 if any(c["budget_a"]!=c["budget_b"] for c in manifest.comparisons):raise ValueError("C1 requires equal budgets")
 by_pair={}
 for g in games:by_pair.setdefault(g.pair_id,[]).append(g)
 if any(len(v)!=2 or v[0].candidate_p1!=v[1].candidate_p2 for v in by_pair.values()):raise ValueError("invalid paired schedule")
 return {"comparisons":6,"paired_openings_per_comparison":400,"games":4800,"unique_game_ids":len({g.game_id for g in games}),"status":"PASS"}

def checkpoint_unchanged(path:Path,before:str)->None:
 if file_sha256(path)!=before:raise ValueError("checkpoint mutated during validation")
