"""Lot46A: pipeline G5 court, reproductible et reprenable.

Le module n'execute aucune arene et ne promeut aucun modele. Les runs longs
restent proteges par ``confirm_full_training`` dans la configuration.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from songo_ai.dataset import read_d_rl_jsonl
from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.model import SongoGraphBuilder, load_srn_checkpoint, mask_policy_logits
from songo_ai.model.correct_preserve import classify_pairs, correction_loss, preservation_loss
from songo_ai.model.strategic_ranking import build_legal_pairs
from songo_ai.training.lot46 import Lot46Error, collate, group_aware_split, reconstruct_policy_target, sha256

FAMILIES = {"CONTROL", "DEEP_POLICY", "DEEP_POLICY_REPLAY", "VALUE_INDEPENDENT"}
MODES = {"HEAD_ONLY", "SHARED_TRUNK_TRAINABLE"}
STATUSES = {"NOT_STARTED", "PREFLIGHT_PASSED", "RUNNING", "INTERRUPTED", "RESUMABLE", "COMPLETED", "FAILED", "INVALID"}
QDIAG256_SHA256 = "e854371afcc162b17e323a062ef0cde96bd7e1041ebe8b794028042b5c0abecc"
QDIAG256_MANIFEST = "data/experiments/lot25_scale/strategic_manifest.json"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def git_commit(root: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _dependency(name: str, path: Path, source: str, required_by: list[str], stage: str,
                expected_sha256: str | None = None) -> dict:
    available=path.is_file(); actual=sha256(path) if available else None
    return {"name":name,"path":str(path),"source":source,"required_by":required_by,
            "required_stage":stage,"expected_sha256":expected_sha256,"actual_sha256":actual,
            "availability":"AVAILABLE" if available else "MISSING",
            "validation_status":"PASS" if available and (expected_sha256 is None or actual==expected_sha256) else "FAIL"}


def _validate_qdiag_schema(path: Path) -> tuple[bool,str,int]:
    required={"state","legal_mask","position_hash","q_values","qdiag","qdiag_budget_per_action",
              "dirichlet","source_dataset","generation_model"}; count=0
    try:
        with path.open() as stream:
            for line_number,line in enumerate(stream,1):
                if not line.strip(): continue
                row=json.loads(line); count+=1
                if not required.issubset(row): return False,f"line {line_number}: missing {sorted(required-set(row))}",count
                state=row["state"]
                if not isinstance(state,dict) or len(state.get("board",[]))!=16 or state.get("player_to_move") not in (1,2):
                    return False,f"line {line_number}: invalid state",count
                if len(row["legal_mask"])!=7 or len(row["q_values"])!=7:
                    return False,f"line {line_number}: policy dimensions must be 7",count
                if row["qdiag"] is not True or row["qdiag_budget_per_action"]!=256 or row["dirichlet"] is not False:
                    return False,f"line {line_number}: incompatible Qdiag protocol",count
                if "value_target" in row or "policy_target" in row:
                    return False,f"line {line_number}: training labels forbidden in preservation battery",count
    except Exception as exc: return False,f"schema read error: {type(exc).__name__}: {exc}",count
    return count==2000,("OK" if count==2000 else f"expected 2000 positions, got {count}"),count


def validate_scientific_inputs(config: "ExperimentConfig", root: Path, *, stage: str,
                               audit_path: Path | None = None) -> dict:
    """Fail-fast, stage-aware audit of immutable scientific inputs."""
    dependencies=[]; full=stage in {"micro-overfit","train","resume","validate","integration-test"}
    for role,value in (("policy_checkpoint",config.policy_checkpoint),("value_checkpoint",config.value_checkpoint)):
        if full:
            path=Path(value);path=path if path.is_absolute() else root/path
            dependencies.append(_dependency(role,path,"G4R frozen checkpoint",["load_initial_model"],stage))
    for source in config.dataset_sources:
        path=Path(source.path);path=path if path.is_absolute() else root/path
        dependencies.append(_dependency(f"dataset:{source.target_source}",path,
            "Lot45" if source.kind=="lot45" else "historical D_RL",["prepare_context","load_sources"],stage))
    if full and config.candidate_family!="VALUE_INDEPENDENT":
        battery=Path(config.objective.get("strategic_battery",""));battery=battery if battery.is_absolute() else root/battery
        dep=_dependency("strategic_preservation_battery",battery,"Lot25 D_STRATEGIC_SAMPLE",
                        ["prepare_strategic_battery","Correct-and-Preserve"],stage,QDIAG256_SHA256)
        if dep["validation_status"]=="PASS":
            valid,detail,count=_validate_qdiag_schema(battery);dep.update({"schema":detail,"positions":count,
                "role":"TRAINING_PRESERVATION_PAIRWISE_ONLY","validation_status":"PASS" if valid else "FAIL"})
        dependencies.append(dep)
        provenance=root/QDIAG256_MANIFEST
        dependencies.append(_dependency("strategic_battery_provenance",provenance,"Lot25 strategic manifest",
                            ["scientific provenance audit"],stage,"337f6eea5613442a0beed2b2bb5998cf9bfacf6af16cf761d5dbd21dc18e19b8"))
    missing=[]
    for dep in dependencies:
        if dep["validation_status"]!="PASS":
            missing.append({"filename":Path(dep["path"]).name,"expected_path":dep["path"],
                "searched_locations":[dep["path"]],"required_by":dep["required_by"],
                "source_archive":"lot46_inputs.tar.gz" if "Lot25" in dep["source"] or "G4R" in dep["source"] or "D_RL" in dep["source"] else "lot45_results.tar.gz",
                "recovery_action":"restore the immutable file from its declared archive and verify SHA256",
                "reason":dep.get("schema") or dep["availability"]})
    report={"stage":stage,"dependencies":dependencies,"MISSING_INPUTS":missing,
            "TRAINING_HOLDOUT_SEPARATION_VALID":"YES",
            "separation_note":"Qdiag256 is optimization-only pairwise preservation; strategic_holdout remains evaluation-only.",
            "SCIENTIFIC_INPUTS_READY":"YES" if not missing else "NO"}
    if audit_path:_write_json(audit_path,report)
    if missing: raise Lot46Error("MISSING_INPUTS="+json.dumps(missing,sort_keys=True))
    return report


def validate_bundle_manifest(manifest: dict, archive_names: set[str]) -> None:
    missing=sorted(set(manifest.get("files",{}))-archive_names)
    if missing: raise Lot46Error(f"incomplete scientific bundle: {missing}")


@dataclass(frozen=True)
class DatasetSource:
    kind: str
    path: str
    weight: float
    target_source: str
    target_budget: int | None

    @classmethod
    def from_dict(cls, value: dict) -> "DatasetSource":
        missing = {"kind", "path", "weight", "target_source", "target_budget"} - set(value)
        if missing: raise Lot46Error(f"dataset source missing {sorted(missing)}")
        result = cls(**{k: value[k] for k in cls.__dataclass_fields__})
        if result.kind not in {"lot45", "reanalysis", "d_rl"}: raise Lot46Error(f"unsupported dataset kind: {result.kind}")
        if not math.isfinite(result.weight) or result.weight <= 0: raise Lot46Error("dataset weight must be positive")
        return result


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    candidate_family: str
    initial_checkpoint: str
    policy_checkpoint: str
    value_checkpoint: str
    dataset_sources: tuple[DatasetSource, ...]
    objective: dict
    optimizer: dict
    scheduler: dict
    learning_rate: float
    batch_size: int
    max_steps: int
    validation_interval: int
    checkpoint_interval: int
    seed: int
    device: str
    output_directory: str
    training_mode: str = "HEAD_ONLY"
    confirm_full_training: bool = False

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        raw = json.loads(path.read_text())
        required = set(cls.__dataclass_fields__) - {"training_mode", "confirm_full_training"}
        missing = required - set(raw)
        if missing: raise Lot46Error(f"training configuration missing {sorted(missing)}")
        raw["dataset_sources"] = tuple(DatasetSource.from_dict(x) for x in raw["dataset_sources"])
        result = cls(**raw); result.validate(); return result

    def validate(self) -> None:
        if not self.experiment_id or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.experiment_id):
            raise Lot46Error("experiment_id must be non-empty uppercase ASCII")
        if self.candidate_family not in FAMILIES: raise Lot46Error(f"invalid candidate_family {self.candidate_family}")
        if self.training_mode not in MODES: raise Lot46Error(f"invalid training_mode {self.training_mode}")
        if self.optimizer.get("name") != "AdamW": raise Lot46Error("Lot46A supports the historically selected AdamW only")
        if self.scheduler.get("name") not in {"constant", "cosine"}: raise Lot46Error("scheduler must be constant or cosine")
        if not (self.learning_rate > 0 and self.batch_size > 0 and self.max_steps > 0): raise Lot46Error("invalid numeric training budget")
        if self.validation_interval <= 0 or self.checkpoint_interval <= 0: raise Lot46Error("intervals must be positive")
        if self.device not in {"cpu", "cuda", "auto"}: raise Lot46Error("device must be cpu/cuda/auto")
        if self.candidate_family == "CONTROL" and any(s.kind == "lot45" for s in self.dataset_sources):
            raise Lot46Error("CONTROL must not use Lot45 deep targets")
        if self.candidate_family in {"DEEP_POLICY", "DEEP_POLICY_REPLAY"} and not any(s.kind == "lot45" for s in self.dataset_sources):
            raise Lot46Error("deep Policy candidates require Lot45")
        if self.candidate_family == "DEEP_POLICY" and len(self.dataset_sources) != 1:
            raise Lot46Error("DEEP_POLICY is the direct Lot45-only intervention")
        if self.candidate_family == "DEEP_POLICY_REPLAY" and len(self.dataset_sources) < 2:
            raise Lot46Error("DEEP_POLICY_REPLAY requires a real replay source")
        if self.objective.get("name") != "CORRECT_AND_PRESERVE" and self.candidate_family != "VALUE_INDEPENDENT":
            raise Lot46Error("Policy candidates require CORRECT_AND_PRESERVE")
        if self.max_steps > 100 and not self.confirm_full_training:
            raise Lot46Error("max_steps > 100 requires confirm_full_training=true")

    def payload(self) -> dict:
        d = asdict(self); d["dataset_sources"] = [asdict(x) for x in self.dataset_sources]; return d


class StatefulSampler:
    """Sampler exact au batch; l'ordre et la position sont checkpointes."""
    def __init__(self, size: int, seed: int, batch_size: int):
        if size <= 0: raise Lot46Error("empty training split")
        self.size, self.seed, self.batch_size = size, seed, batch_size
        self.rng = random.Random(seed); self.epoch = 0; self.position = 0; self.order = list(range(size)); self.rng.shuffle(self.order)

    def next(self) -> list[int]:
        values = []
        while len(values) < self.batch_size:
            remaining = self.size - self.position; take = min(self.batch_size - len(values), remaining)
            values.extend(self.order[self.position:self.position + take]); self.position += take
            if self.position == self.size:
                self.epoch += 1; self.position = 0; self.order = list(range(self.size)); self.rng.shuffle(self.order)
        return values

    def state_dict(self) -> dict:
        return {"size": self.size, "seed": self.seed, "batch_size": self.batch_size, "epoch": self.epoch,
                "position": self.position, "order": self.order, "rng_state": self.rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        for key in ("size", "seed", "batch_size"):
            if state[key] != getattr(self, key): raise Lot46Error(f"sampler {key} incompatibility")
        self.epoch, self.position, self.order = state["epoch"], state["position"], list(state["order"])
        self.rng.setstate(state["rng_state"])


class WeightedStatefulSampler:
    """Echantillonnage source-aware avec remise et reprise exacte."""
    def __init__(self, rows: list[dict], source_weights: dict[str,float], seed: int, batch_size: int):
        self.size=len(rows); self.seed=seed; self.batch_size=batch_size; self.epoch=0; self.position=0; self.samples_seen=0
        if not self.size: raise Lot46Error("empty training split")
        counts={}
        for row in rows: counts[row["target_source"]]=counts.get(row["target_source"],0)+1
        self.weights=[source_weights.get(row["target_source"],0.0)/counts[row["target_source"]] for row in rows]
        if not any(self.weights): raise Lot46Error("dataset weights do not match loaded target sources")
        self.rng=random.Random(seed)

    def next(self) -> list[int]:
        result=self.rng.choices(range(self.size),weights=self.weights,k=self.batch_size); self.samples_seen+=self.batch_size
        self.epoch,self.position=divmod(self.samples_seen,self.size); return result

    def state_dict(self) -> dict:
        return {"kind":"weighted","size":self.size,"seed":self.seed,"batch_size":self.batch_size,"epoch":self.epoch,
                "position":self.position,"samples_seen":self.samples_seen,"weights":self.weights,"rng_state":self.rng.getstate()}

    def load_state_dict(self,state:dict)->None:
        for key in ("size","seed","batch_size"):
            if state[key]!=getattr(self,key): raise Lot46Error(f"sampler {key} incompatibility")
        if state["weights"]!=self.weights: raise Lot46Error("sampler weight incompatibility")
        self.epoch,self.position,self.samples_seen=state["epoch"],state["position"],state["samples_seen"];self.rng.setstate(state["rng_state"])


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def full_rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None: torch.cuda.set_rng_state_all(state["torch_cuda"])


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream: return [json.loads(x) for x in stream if x.strip()]


def _canonical_reanalysis(row: dict, source: DatasetSource) -> dict:
    state = row["state"]; fp = row.get("fingerprint") or row.get("position_hash") or canonical_hash(state)
    visits = list(row["visit_counts"]); legal = list(row["legal_mask"])
    target = reconstruct_policy_target(visits, legal)
    value_available = bool(row.get("value_target_available", row.get("value_target_present", False)))
    value = row.get("z_mean", row.get("value_target")) if value_available else None
    return {"fingerprint": fp, "state": state, "legal_mask": legal, "visit_counts": visits,
            "policy_target": target, "value_target_available": value_available, "z_mean": value,
            "split_group": row.get("split_group") or row.get("source_game_id") or f"state:{fp}",
            "holdout": bool(row.get("holdout", False)), "target_source": source.target_source,
            "target_budget": row.get("simulations", row.get("mcts_budget", source.target_budget)),
            "source_occurrence_count": int(row.get("source_occurrence_count", 1)), "source_kind": source.kind}


def load_sources(config: ExperimentConfig, root: Path) -> tuple[list[dict], dict]:
    by_fp: dict[str, dict] = {}; provenance = {}; conflicts = []
    for source in config.dataset_sources:
        path = Path(source.path); path = path if path.is_absolute() else root / path
        if not path.is_file(): raise Lot46Error(f"dataset source missing: {path}")
        if source.kind in {"lot45", "reanalysis"}:
            rows = [_canonical_reanalysis(x, source) for x in _read_jsonl(path)]
        else:
            rows = []
            for item in read_d_rl_jsonl(path):
                fp = item.metadata.get("position_hash") or canonical_hash({"board": item.state.board, "player_to_move": item.state.player_to_move})
                rows.append({"fingerprint": fp, "state": {"board": list(item.state.board), "player_to_move": item.state.player_to_move},
                    "legal_mask": list(item.legal_mask), "visit_counts": list(item.visit_counts), "policy_target": list(item.policy_target),
                    "value_target_available": item.value_target is not None, "z_mean": item.value_target,
                    "split_group": item.metadata.get("game_id") or f"state:{fp}", "holdout": False,
                    "target_source": source.target_source, "target_budget": item.metadata.get("mcts_budget", source.target_budget),
                    "source_occurrence_count": 1, "source_kind": source.kind})
        provenance[source.path] = {"sha256": sha256(path), "rows": len(rows), "weight": source.weight,
                                   "target_source": source.target_source, "target_budget": source.target_budget}
        for row in rows:
            fp = row["fingerprint"]
            if fp in by_fp:
                old = by_fp[fp]
                if old["state"] != row["state"]: conflicts.append(fp); continue
                old.setdefault("source_variants", []).append({"target_source": row["target_source"], "target_budget": row["target_budget"]})
                old["source_occurrence_count"] += row["source_occurrence_count"]
                old["holdout"] = old["holdout"] or row["holdout"]
            else:
                row["source_variants"] = [{"target_source": row["target_source"], "target_budget": row["target_budget"]}]
                by_fp[fp] = row
    if conflicts: raise Lot46Error(f"physical-state identity conflicts: {conflicts[:5]}")
    rows = list(by_fp.values())
    return rows, {"sources": provenance, "unique_states": len(rows), "logical_occurrences": sum(x["source_occurrence_count"] for x in rows),
                  "duplicate_occurrences": sum(x["source_occurrence_count"] for x in rows) - len(rows)}


def validate_rows(rows: Iterable[dict]) -> dict:
    count = labeled = 0; budgets = {}; sources = {}
    for row in rows:
        count += 1
        target = reconstruct_policy_target(row["visit_counts"], row["legal_mask"])
        if target is None or any(abs(a-b) > 1e-6 for a,b in zip(target,row["policy_target"])): raise Lot46Error("invalid Policy target")
        if row["value_target_available"]:
            z = row["z_mean"]
            if z not in (-1, 0, 1, -1.0, 0.0, 1.0): raise Lot46Error(f"terminal z outside {{-1,0,1}}: {z}")
            labeled += 1
        elif row["z_mean"] is not None: raise Lot46Error("missing Value target contains a numeric label")
        budgets[str(row["target_budget"])] = budgets.get(str(row["target_budget"]), 0) + 1
        sources[row["target_source"]] = sources.get(row["target_source"], 0) + 1
    return {"valid": True, "positions": count, "value_labeled": labeled, "value_unlabeled": count-labeled,
            "target_budgets": budgets, "target_sources": sources}


def configure_trainable(model: torch.nn.Module, family: str, mode: str) -> list[str]:
    for p in model.parameters(): p.requires_grad_(False)
    prefixes = ["value_mlp."] if family == "VALUE_INDEPENDENT" else ["policy_mlp."]
    if mode == "SHARED_TRUNK_TRAINABLE": prefixes += ["node_encoder.", "global_encoder.", "relational_blocks.", "state_fusion."]
    names = []
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes): p.requires_grad_(True); names.append(name)
    if not names: raise Lot46Error("no trainable parameters")
    return names


def make_optimizer_scheduler(model: torch.nn.Module, config: ExperimentConfig):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate,
                            weight_decay=float(config.optimizer.get("weight_decay", 1e-4)),
                            betas=tuple(config.optimizer.get("betas", (.9, .999))))
    if config.scheduler["name"] == "constant": scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    else: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.max_steps,
                                                                  eta_min=float(config.scheduler.get("eta_min", 0.0)))
    return opt, scheduler


def architecture_fingerprint(model) -> str:
    return canonical_hash(model.config.__dict__ if hasattr(model.config, "__dict__") else str(model.config))


def model_fingerprint(model) -> str:
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items(): h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str)+"\n"); os.replace(tmp,path)


class DurableCheckpointStore:
    def __init__(self, local: Path, durable: Path): self.local=local; self.durable=durable

    def publish(self, payload: dict, step: int) -> dict:
        self.local.mkdir(parents=True, exist_ok=True); self.durable.mkdir(parents=True, exist_ok=True)
        name=f"checkpoint-{step:08d}.pt"; local=self.local/name; tmp=local.with_name(f".{name}.tmp-{os.getpid()}")
        torch.save(payload,tmp); torch.load(tmp,map_location="cpu",weights_only=False); os.replace(tmp,local)
        digest=sha256(local); remote=self.durable/name; remote_tmp=self.durable/f".{name}.upload-{os.getpid()}"
        shutil.copyfile(local,remote_tmp)
        if sha256(remote_tmp)!=digest: remote_tmp.unlink(missing_ok=True); raise Lot46Error("durable checkpoint checksum mismatch")
        os.replace(remote_tmp,remote)
        manifest={"experiment_id":payload["experiment_id"],"step":step,"file":name,"sha256":digest,"size":remote.stat().st_size,
                  "code_commit":payload.get("code_commit"),
                  "status":"COMMITTED","published_at":time.time()}
        _write_json(self.durable/f"checkpoint-{step:08d}.manifest.json",manifest)
        return {**manifest,"local_path":str(local),"durable_path":str(remote),"CHECKPOINT_DURABLE":"YES"}

    def latest(self, experiment_id: str) -> tuple[Path,dict] | None:
        valid=[]
        for path in sorted(self.durable.glob("checkpoint-*.manifest.json")):
            try:
                m=json.loads(path.read_text()); data=self.durable/m["file"]
                if m.get("status")=="COMMITTED" and m.get("experiment_id")==experiment_id and data.is_file() and sha256(data)==m["sha256"]: valid.append((m["step"],data,m))
            except Exception: continue
        if not valid:return None
        _,path,manifest=max(valid,key=lambda x:x[0]); return path,manifest


def checkpoint_payload(*, config: ExperimentConfig, model, optimizer, scheduler, sampler, step: int,
                       dataset_fingerprint: str, split_fingerprint: str, initial_fingerprint: str,
                       root: Path, metrics: dict) -> dict:
    return {"checkpoint_type":"songo_lot46a_training","checkpoint_version":1,"experiment_id":config.experiment_id,
        "model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),
        "gradient_scaler_state":None,"global_step":step,"epoch":sampler.epoch,"sampler_state":sampler.state_dict(),
        "rng_state":full_rng_state(),"training_config":config.payload(),"training_config_fingerprint":canonical_hash(config.payload()),
        "dataset_fingerprint":dataset_fingerprint,"split_fingerprint":split_fingerprint,
        "initial_checkpoint_fingerprint":initial_fingerprint,"architecture_fingerprint":architecture_fingerprint(model),
        "code_commit":git_commit(root),"metrics":metrics}


def resume_into(path: Path, *, config: ExperimentConfig, model, optimizer, scheduler, sampler,
                dataset_fingerprint: str, split_fingerprint: str, initial_fingerprint: str,
                code_commit: str | None = None) -> dict:
    payload=torch.load(path,map_location="cpu",weights_only=False)
    checks={"checkpoint_type":payload.get("checkpoint_type")=="songo_lot46a_training",
        "experiment_id":payload.get("experiment_id")==config.experiment_id,
        "training_config":payload.get("training_config_fingerprint")==canonical_hash(config.payload()),
        "dataset":payload.get("dataset_fingerprint")==dataset_fingerprint,"split":payload.get("split_fingerprint")==split_fingerprint,
        "initial":payload.get("initial_checkpoint_fingerprint")==initial_fingerprint,
        "architecture":payload.get("architecture_fingerprint")==architecture_fingerprint(model),
        "code_commit":payload.get("code_commit")==code_commit}
    if not all(checks.values()): raise Lot46Error(f"REFUSE_RESUME incompatible checkpoint: {checks}")
    model.load_state_dict(payload["model_state_dict"]); optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"]); sampler.load_state_dict(payload["sampler_state"]); restore_rng(payload["rng_state"])
    return payload


def evaluate(model, rows: list[dict], device: torch.device, batch_size: int=256) -> dict:
    before=model_fingerprint(model); model.eval(); ce=top=kl=rank=sq=ae=n=labeled=0
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            b=collate(rows[start:start+batch_size],device); logits,value=model(b["graph"]); masked=mask_policy_logits(logits,b["legal_mask"])
            logp=torch.log_softmax(masked,-1); p=torch.softmax(masked,-1); target=b["policy_target"]
            ce+=float((-(target*logp).sum(-1)).sum()); top+=int((p.argmax(-1)==target.argmax(-1)).sum())
            kl+=float((target*(torch.log(target.clamp_min(1e-12))-logp)).sum());
            for x,y,m in zip(p,target,b["legal_mask"]):
                legal=torch.where(m)[0]; rank+=float(torch.equal(legal[torch.argsort(x[legal],descending=True)],legal[torch.argsort(y[legal],descending=True)]))
            vm=b["value_target_available"]
            if vm.any(): d=value[vm]-b["value_target"][vm];sq+=float(d.square().sum());ae+=float(d.abs().sum());labeled+=int(vm.sum())
            n+=len(target)
    after=model_fingerprint(model)
    if before!=after: raise Lot46Error("validation modified model parameters")
    return {"positions":n,"policy_ce":ce/n,"policy_kl":kl/n,"top1_agreement":top/n,"exact_ranking_agreement":rank/n,
            "value_mse":sq/labeled if labeled else None,"value_mae":ae/labeled if labeled else None,"value_labeled":labeled,
            "no_grad_parameter_invariance":True}


def prepare_strategic_battery(path: Path, parent, device: torch.device) -> list[dict]:
    rows=_read_jsonl(path); items=[]; builder=SongoGraphBuilder()
    parent.eval()
    with torch.no_grad():
        for start in range(0,len(rows),256):
            chunk=rows[start:start+256]
            states=[RawSongoState(tuple(x["state"]["board"]),int(x["state"]["player_to_move"])) for x in chunk]
            graph=builder.build_batch_vectorized(states).to(device); logits,_=parent(graph)
            for index,row in enumerate(chunk):
                pairs=build_legal_pairs(row["q_values"],row["legal_mask"],epsilon=.02,scale=1.0)
                corr,pres=classify_pairs(logits[index].detach().cpu().tolist(),pairs)
                items.append({"state":row["state"],"legal_mask":row["legal_mask"],"correction_pairs":corr,
                              "preservation_pairs":pres,"fingerprint":row.get("position_hash")})
    return items


def _strategic_losses(model, rows: list[dict], device: torch.device, rho: float):
    states=[RawSongoState(tuple(x["state"]["board"]),int(x["state"]["player_to_move"])) for x in rows]
    graph=SongoGraphBuilder().build_batch_vectorized(states).to(device); logits,_=model(graph)
    return correction_loss(logits,[x["correction_pairs"] for x in rows]), preservation_loss(logits,[x["preservation_pairs"] for x in rows],rho=rho)


def step_model(model, optimizer, scheduler, rows: list[dict], config: ExperimentConfig, device: torch.device,
               strategic_rows: list[dict] | None=None) -> dict:
    model.train(); b=collate(rows,device); logits,value=model(b["graph"])
    policy=-(b["policy_target"]*torch.log_softmax(mask_policy_logits(logits,b["legal_mask"]),-1)).sum(-1).mean()
    vm=b["value_target_available"]; value_loss=(value[vm]-b["value_target"][vm]).square().mean() if vm.any() else value.sum()*0
    if config.candidate_family != "VALUE_INDEPENDENT":
        if not strategic_rows: raise Lot46Error("Correct-and-Preserve requires a non-empty strategic battery")
        corr,preserve=_strategic_losses(model,strategic_rows,device,float(config.objective.get("rho",.5)))
    else: corr=logits.sum()*0; preserve=logits.sum()*0
    total=value_loss if config.candidate_family=="VALUE_INDEPENDENT" else policy+float(config.objective.get("lambda_correction",.1))*corr+float(config.objective.get("lambda_preservation",1.))*preserve
    optimizer.zero_grad(set_to_none=True); total.backward()
    nonfinite=any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
    if nonfinite: raise Lot46Error("non-finite gradients")
    threshold=float(config.optimizer.get("gradient_clip",1.)); grad=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],threshold))
    optimizer.step(); scheduler.step()
    return {"loss_total":float(total.detach()),"loss_policy":float(policy.detach()),"loss_value":float(value_loss.detach()),
            "loss_correction":float(corr.detach()),"loss_preservation":float(preserve.detach()),"gradient_norm":grad,
            "gradient_clip":threshold,"nonfinite_gradients":False,"learning_rate":optimizer.param_groups[0]["lr"]}


def load_initial_model(config: ExperimentConfig, root: Path, device: torch.device):
    policy_path=Path(config.policy_checkpoint); policy_path=policy_path if policy_path.is_absolute() else root/policy_path
    value_path=Path(config.value_checkpoint); value_path=value_path if value_path.is_absolute() else root/value_path
    policy=load_srn_checkpoint(policy_path,device=str(device)).model; value=load_srn_checkpoint(value_path,device=str(device)).model
    state=policy.state_dict(); value_state=value.state_dict()
    for name in state:
        if name.startswith("value_mlp."): state[name]=value_state[name].detach().clone()
    policy.load_state_dict(state); return policy, canonical_hash({"policy":sha256(policy_path),"value":sha256(value_path)})


def prepare_context(config: ExperimentConfig, root: Path) -> dict:
    config.validate(); output=Path(config.output_directory); output=output if output.is_absolute() else root/output
    if output.name != config.experiment_id: raise Lot46Error("output_directory basename must equal experiment_id")
    rows,source_audit=load_sources(config,root); dataset_audit=validate_rows(rows); splits=group_aware_split(rows,seed=config.seed)
    if not splits["train"] or not splits["validation"]: raise Lot46Error("TRAIN and VALIDATION must be non-empty")
    dataset_fp=canonical_hash({"sources":source_audit["sources"],"states":sorted(x["fingerprint"] for x in rows)})
    split_fp=canonical_hash({k:sorted(x["fingerprint"] for x in v) for k,v in splits.items()})
    return {"output":output,"rows":rows,"splits":splits,"source_audit":source_audit,"dataset_audit":dataset_audit,
            "dataset_fingerprint":dataset_fp,"split_fingerprint":split_fp}


def run_training(config: ExperimentConfig, root: Path, *, resume: bool=False, stop_after: int | None=None,
                 train_limit: int | None=None) -> dict:
    output=Path(config.output_directory);output=output if output.is_absolute() else root/output;output.mkdir(parents=True,exist_ok=True)
    validate_scientific_inputs(config,root,stage="resume" if resume else "train",audit_path=output/"lot46a_input_dependency_audit.json")
    ctx=prepare_context(config,root)
    code_commit=git_commit(root)
    assignment=output/"experiment_assignment.json"
    identity={"experiment_id":config.experiment_id,"output_directory":str(output.resolve()),"config_fingerprint":canonical_hash(config.payload()),
              "code_commit":code_commit}
    if assignment.is_file():
        assigned=json.loads(assignment.read_text())
        if resume and assigned.get("code_commit") != code_commit:
            raise Lot46Error(f"REFUSE_RESUME code_commit mismatch: initial={assigned.get('code_commit')} current={code_commit}")
        if assigned!=identity: raise Lot46Error("experiment output is assigned to a different configuration")
    _write_json(assignment,identity)
    status_path=output/"status.json"
    if status_path.is_file() and not resume:
        previous=json.loads(status_path.read_text()).get("status")
        if previous in {"RUNNING","RESUMABLE","COMPLETED"}: raise Lot46Error(f"experiment already {previous}; use resume or a new experiment_id")
    _write_json(status_path,{"experiment_id":config.experiment_id,"status":"RUNNING","exclusive_assignment":"EXPLICIT_NO_SHARED_WRITER"})
    device=torch.device("cuda" if config.device=="cuda" or config.device=="auto" and torch.cuda.is_available() else "cpu")
    if config.device=="cuda" and not torch.cuda.is_available(): raise Lot46Error("CUDA requested but unavailable")
    set_seed(config.seed); model,initial_fp=load_initial_model(config,root,device); trainable=configure_trainable(model,config.candidate_family,config.training_mode)
    optimizer,scheduler=make_optimizer_scheduler(model,config)
    source_weights={s.target_source:s.weight for s in config.dataset_sources};train_rows=ctx["splits"]["train"][:train_limit] if train_limit else ctx["splits"]["train"]
    sampler=WeightedStatefulSampler(train_rows,source_weights,config.seed,config.batch_size)
    strategic=[]
    if config.candidate_family!="VALUE_INDEPENDENT":
        battery=Path(config.objective.get("strategic_battery", "")); battery=battery if battery.is_absolute() else root/battery
        if not battery.is_file(): raise Lot46Error(f"strategic battery missing: {battery}")
        strategic=prepare_strategic_battery(battery,model,device)
    store=DurableCheckpointStore(output/"local_checkpoints",output/"durable_checkpoints"); step=0; resumed=False
    if resume:
        latest=store.latest(config.experiment_id)
        if latest is None: raise Lot46Error("no valid durable checkpoint to resume")
        payload=resume_into(latest[0],config=config,model=model,optimizer=optimizer,scheduler=scheduler,sampler=sampler,
            dataset_fingerprint=ctx["dataset_fingerprint"],split_fingerprint=ctx["split_fingerprint"],initial_fingerprint=initial_fp,
            code_commit=code_commit)
        step=int(payload["global_step"]); resumed=True
    history=output/"training_history.jsonl"; started=time.perf_counter(); checkpoint_manifest=[]; limit=min(config.max_steps,stop_after or config.max_steps)
    model.train(); before_other=[]
    fixed=ctx["splits"]["validation"][:min(32,len(ctx["splits"]["validation"]))]
    with torch.no_grad():
        b=collate(fixed,device); p0,v0=model(b["graph"]); before_other=(p0.detach().cpu(),v0.detach().cpu())
    while step<limit:
        indices=sampler.next(); batch=[train_rows[i] for i in indices]
        sr=[strategic[(step*32+i)%len(strategic)] for i in range(min(32,len(strategic)))] if strategic else None
        metrics=step_model(model,optimizer,scheduler,batch,config,device,sr); step+=1
        row={"experiment_id":config.experiment_id,"step":step,"epoch":sampler.epoch,"candidate_family":config.candidate_family,
             **metrics,"elapsed_seconds":time.perf_counter()-started}
        with history.open("a") as stream: stream.write(json.dumps(row,sort_keys=True)+"\n")
        if step%config.validation_interval==0 or step==limit: _write_json(output/"validation_latest.json",evaluate(model,ctx["splits"]["validation"],device))
        if step%config.checkpoint_interval==0 or step==limit:
            payload=checkpoint_payload(config=config,model=model,optimizer=optimizer,scheduler=scheduler,sampler=sampler,step=step,
                dataset_fingerprint=ctx["dataset_fingerprint"],split_fingerprint=ctx["split_fingerprint"],initial_fingerprint=initial_fp,
                root=root,metrics=row); checkpoint_manifest.append(store.publish(payload,step)); _write_json(output/"checkpoint_manifest.json",{"checkpoints":checkpoint_manifest})
    with torch.no_grad():
        b=collate(fixed,device); p1,v1=model(b["graph"])
    other_drift=float((v1.cpu()-before_other[1]).abs().max()) if config.candidate_family!="VALUE_INDEPENDENT" else float((p1.cpu()-before_other[0]).abs().max())
    status="COMPLETED" if step==config.max_steps else "RESUMABLE"
    result={"experiment_id":config.experiment_id,"status":status,"global_step":step,"resumed":resumed,"device":str(device),
        "code_commit":code_commit,
        "trainable_parameters":trainable,"training_mode":config.training_mode,"other_head_max_output_drift":other_drift,
        "head_invariance_required":config.training_mode=="HEAD_ONLY","head_invariance_pass":other_drift==0 if config.training_mode=="HEAD_ONLY" else "NOT_APPLICABLE",
        "dataset_fingerprint":ctx["dataset_fingerprint"],"split_fingerprint":ctx["split_fingerprint"],
        "validation":evaluate(model,ctx["splits"]["validation"],device),"checkpoint_durable":bool(store.latest(config.experiment_id)),
        "samples_per_second":step*config.batch_size/max(time.perf_counter()-started,1e-9),"steps_per_second":step/max(time.perf_counter()-started,1e-9)}
    _write_json(status_path,result)
    latest=store.latest(config.experiment_id)
    _write_json(output/"candidate_registry.json",{"candidates":[{"candidate_id":config.experiment_id,"family":config.candidate_family,
        "code_commit":code_commit,
        "initial_checkpoint":config.initial_checkpoint,"training_config":config.payload(),"dataset_sources":[asdict(x) for x in config.dataset_sources],
        "split_fingerprint":ctx["split_fingerprint"],"checkpoint_paths":[str(latest[0])] if latest else [],
        "training_status":status,"validation_status":"COMPLETED"}]})
    return result
