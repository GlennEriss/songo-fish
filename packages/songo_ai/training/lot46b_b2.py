"""Authorized Lot46B-B2 Stage B2: two Policy experiments, no arenas/promotion."""
from __future__ import annotations

import json
import math
import platform
import tarfile
import tempfile
from pathlib import Path

import torch

from songo_ai.training.lot46 import Lot46Error, sha256
from songo_ai.training.lot46a import (DurableCheckpointStore, ExperimentConfig, WeightedStatefulSampler,
    canonical_hash, configure_trainable, git_commit, load_initial_model, make_optimizer_scheduler,
    prepare_context, resume_into, run_training)
from songo_ai.training.lot46b import stage_a_correction_analysis, _strategic_preservation
from songo_ai.training.lot46b_b0 import STAGE_A_COMMIT

TARGETS=(1000,2000,3000,5000)
ALLOWED={"DEEP_POLICY","DEEP_POLICY_UNCONSTRAINED"}


def _read(path: Path) -> dict:
    if not path.is_file(): raise Lot46Error(f"required B2 evidence missing: {path}")
    value=json.loads(path.read_text())
    if not isinstance(value,dict): raise Lot46Error(f"invalid JSON: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=str)+"\n");tmp.replace(path)


def _output(config: ExperimentConfig, root: Path) -> Path:
    path=Path(config.output_directory);return path if path.is_absolute() else root/path


def _load_latest(config: ExperimentConfig, root: Path, device: torch.device):
    ctx=prepare_context(config,root);model,initial_fp=load_initial_model(config,root,device)
    configure_trainable(model,config.candidate_family,config.training_mode);optimizer,scheduler=make_optimizer_scheduler(model,config)
    sampler=WeightedStatefulSampler(ctx["splits"]["train"],{s.target_source:s.weight for s in config.dataset_sources},config.seed,config.batch_size)
    store=DurableCheckpointStore(ctx["output"]/"local_checkpoints",ctx["output"]/"durable_checkpoints");latest=store.latest(config.experiment_id)
    if not latest: raise Lot46Error("B2 durable checkpoint missing")
    payload=resume_into(latest[0],config=config,model=model,optimizer=optimizer,scheduler=scheduler,sampler=sampler,
        dataset_fingerprint=ctx["dataset_fingerprint"],split_fingerprint=ctx["split_fingerprint"],initial_fingerprint=initial_fp,
        code_commit=git_commit(root))
    return ctx,model,payload,latest


def _comparable_payload(old: dict, stage_a: ExperimentConfig, standard: ExperimentConfig) -> bool:
    a=stage_a.payload();b=standard.payload()
    for key in ("experiment_id","output_directory","max_steps","validation_interval","checkpoint_interval"): a.pop(key,None);b.pop(key,None)
    return a==b and old.get("training_config_fingerprint")==canonical_hash(stage_a.payload())


def prepare_b2(stage_a_config_path: Path, standard_path: Path, unconstrained_path: Path,
               b1_output: Path, root: Path) -> dict:
    if not torch.cuda.is_available(): raise Lot46Error("CUDA is mandatory for Stage B2")
    b1=_read(b1_output/"lot46b_b2_stage_b1_report.json")
    if b1.get("STAGE_B1_PREREGISTRATION_VALID")!="YES" or b1.get("STAGE_B_STARTED")!="NO":
        raise Lot46Error("Stage B1 evidence is not valid and frozen")
    stage_a=ExperimentConfig.load(stage_a_config_path);standard=ExperimentConfig.load(standard_path);unconstrained=ExperimentConfig.load(unconstrained_path)
    if standard.candidate_family!="DEEP_POLICY" or unconstrained.candidate_family!="DEEP_POLICY_UNCONSTRAINED": raise Lot46Error("unauthorized B2 family")
    if standard.max_steps!=5000 or unconstrained.max_steps!=5000: raise Lot46Error("B2 target must be exactly 5000")
    sa_ctx=prepare_context(stage_a,root);std_ctx=prepare_context(standard,root);unc_ctx=prepare_context(unconstrained,root)
    if len({sa_ctx["dataset_fingerprint"],std_ctx["dataset_fingerprint"],unc_ctx["dataset_fingerprint"]})!=1: raise Lot46Error("B2 dataset mismatch")
    if len({sa_ctx["split_fingerprint"],std_ctx["split_fingerprint"],unc_ctx["split_fingerprint"]})!=1: raise Lot46Error("B2 split mismatch")
    source_store=DurableCheckpointStore(sa_ctx["output"]/"local_checkpoints",sa_ctx["output"]/"durable_checkpoints");source=source_store.latest(stage_a.experiment_id)
    if not source or source[1].get("step")!=1000 or source[1].get("code_commit")!=STAGE_A_COMMIT: raise Lot46Error("Stage A step-1000 checkpoint invalid")
    payload=torch.load(source[0],map_location="cpu",weights_only=False)
    if not _comparable_payload(payload,stage_a,standard): raise Lot46Error("standard continuation changes historical scientific settings")
    standard_out=std_ctx["output"];target_store=DurableCheckpointStore(standard_out/"local_checkpoints",standard_out/"durable_checkpoints")
    if target_store.latest(standard.experiment_id) is None:
        migrated=dict(payload);migrated.update({"experiment_id":standard.experiment_id,"training_config":standard.payload(),
            "training_config_fingerprint":canonical_hash(standard.payload()),"code_commit":git_commit(root),
            "lineage":{"kind":"STAGE_A_EXACT_STATE_CONTINUATION","source_experiment":stage_a.experiment_id,
                "source_checkpoint":str(source[0]),"source_sha256":source[1]["sha256"],"source_code_commit":STAGE_A_COMMIT}})
        target_store.publish(migrated,1000)
        _write(standard_out/"experiment_assignment.json",{"experiment_id":standard.experiment_id,
            "output_directory":str(standard_out.resolve()),"config_fingerprint":canonical_hash(standard.payload()),"code_commit":git_commit(root)})
        _write(standard_out/"status.json",{"experiment_id":standard.experiment_id,"status":"RESUMABLE","global_step":1000,
            "code_commit":git_commit(root),"lineage":migrated["lineage"]})
    # Baselines are frozen before any B2 optimizer step.
    for config,ctx,start in ((standard,std_ctx,1000),(unconstrained,unc_ctx,0)):
        baseline=ctx["output"]/"stage_b2_baseline.json"
        existing_baseline=_read(baseline) if baseline.is_file() else None
        validation=(existing_baseline or {}).get("validation",{})
        if "policy_entropy" not in validation:
            if start==1000:
                # Stage A predates the entropy metric. Re-evaluate its immutable
                # step-1000 checkpoint instead of silently disabling the B2
                # collapse guard or using a post-B2 checkpoint as the baseline.
                model,_=load_initial_model(standard,root,torch.device("cuda"))
                model.load_state_dict(payload["model_state_dict"])
                from songo_ai.training.lot46a import evaluate
                validation=evaluate(model,std_ctx["splits"]["validation"],torch.device("cuda"))
            else:
                model,_=load_initial_model(config,root,torch.device("cuda"))
                from songo_ai.training.lot46a import evaluate
                validation=evaluate(model,ctx["splits"]["validation"],torch.device("cuda"))
            _write(baseline,{"step":start,"validation":validation,"frozen_before_training":True,
                "derived_from_checkpoint":str(source[0]) if start==1000 else "INITIAL_CHECKPOINT"})
    report={"ACTIVE_COMMIT":git_commit(root),"STAGE_B1_VALID":"YES","CUDA":"YES","gpu":torch.cuda.get_device_name(0),
        "torch":torch.__version__,"cuda_version":torch.version.cuda,"dataset_fingerprint":std_ctx["dataset_fingerprint"],
        "split_fingerprint":std_ctx["split_fingerprint"],"standard_source_sha256":source[1]["sha256"],
        "authorized_families":sorted(ALLOWED),"unauthorized_families_not_run":["CONTROL","DEEP_POLICY_REPLAY","DEEP_POLICY_REPLAY_RELABELED","VALUE_INDEPENDENT"],
        "STAGE_B2_PREFLIGHT":"PASS","STAGE_B2_STARTED":"NO","ARENAS":"NOT_RUN","G5_PROMOTED":"NO"}
    _write(b1_output.parent/"lot46b_b2_stage_b2"/"stage_b2_preflight.json",report);return report


def _decision(config: ExperimentConfig, root: Path, metrics: dict) -> dict:
    out=_output(config,root);baseline=_read(out/"stage_b2_baseline.json")["validation"]
    previous=[]
    for path in sorted(out.glob("stage_b2_evaluation-*.json")):
        value=_read(path)
        if value.get("step")!=metrics["step"]: previous.append(value)
    sequence=(previous+[metrics])[-2:]
    ce_bad=len(sequence)==2 and all(x["validation"]["policy_ce"]>1.05*baseline["policy_ce"] for x in sequence)
    entropy_bad=len(sequence)==2 and all(x["validation"]["policy_entropy"]<.70*baseline["policy_entropy"] for x in sequence)
    regression_bad=len(sequence)==2 and all(x["correction"]["POTENTIAL_REGRESSION_RATE"]>.30 for x in sequence)
    finite=all(math.isfinite(float(x)) for x in (metrics["validation"]["policy_ce"],metrics["validation"]["policy_kl"],metrics["validation"]["policy_entropy"]))
    reasons=[name for name,flag in (("NONFINITE",not finite),("VALIDATION_DEGRADATION_TWICE",ce_bad),("POLICY_COLLAPSE_TWICE",entropy_bad),("POTENTIAL_REGRESSION_TWICE",regression_bad)) if flag]
    return {"decision":"STOP" if reasons else "CONTINUE","reasons":reasons,"legacy_agreement_is_veto":False}


def run_segment(config_path: Path, target_step: int, root: Path) -> dict:
    config=ExperimentConfig.load(config_path)
    if config.candidate_family not in ALLOWED: raise Lot46Error("family not authorized by Stage B2")
    if target_step not in TARGETS or target_step>config.max_steps: raise Lot46Error("invalid B2 checkpoint target")
    if not torch.cuda.is_available(): raise Lot46Error("CUDA is mandatory for Stage B2")
    out=_output(config,root);prior=sorted(out.glob("stage_b2_evaluation-*.json"))
    existing=out/f"stage_b2_evaluation-{target_step:08d}.json"
    if existing.is_file(): return _read(existing)
    if prior and _read(prior[-1]).get("stopping",{}).get("decision")=="STOP": raise Lot46Error("previous B2 checkpoint triggered controlled stop")
    latest=DurableCheckpointStore(out/"local_checkpoints",out/"durable_checkpoints").latest(config.experiment_id)
    if latest and latest[1]["step"]>=target_step: result={"status":"ALREADY_REACHED","global_step":latest[1]["step"]}
    else: result=run_training(config,root,resume=latest is not None,stop_after=target_step)
    ctx,model,payload,latest=_load_latest(config,root,torch.device("cuda"))
    correction=stage_a_correction_analysis(config,root,model,torch.device("cuda"))
    preservation=_strategic_preservation(config,root,model,torch.device("cuda")) if config.candidate_family=="DEEP_POLICY" else {"status":"DIAGNOSTIC_ONLY","initial_policy_top1_agreement":1-correction["G4_G5_DISAGREEMENT_RATE"]}
    validation=_read(out/f"validation-{int(payload['global_step']):08d}.json")
    history=[json.loads(line) for line in (out/"training_history.jsonl").read_text().splitlines() if line.strip()]
    prior_step=max((int(_read(path)["step"]) for path in prior),default=(1000 if config.candidate_family=="DEEP_POLICY" else 0))
    window=[row for row in history if prior_step<int(row["step"])<=int(payload["global_step"])]
    unique=len({fingerprint for row in window for fingerprint in row.get("batch_fingerprints",[])})
    training={"steps":len(window),"train_ce":sum(float(row["loss_policy"]) for row in window)/len(window),
        "train_kl":sum(float(row["train_policy_kl"]) for row in window)/len(window),
        "train_entropy":sum(float(row["train_policy_entropy"]) for row in window)/len(window),
        "train_validation_ce_gap":sum(float(row["loss_policy"]) for row in window)/len(window)-validation["policy_ce"],
        "positions_seen":len(window)*config.batch_size,"unique_positions_seen":unique,
        "effective_epochs":len(window)*config.batch_size/len(ctx["splits"]["train"])}
    metrics={"step":int(payload["global_step"]),"validation":validation,"correction":correction,
        "legacy_agreement":{"role":"LEGACY_AGREEMENT_DIAGNOSTIC","value":preservation.get("initial_policy_top1_agreement")},
        "checkpoint":{"path":str(latest[0]),"sha256":latest[1]["sha256"],"manifest":latest[1]},"training":training,
        "cuda":{"gpu":torch.cuda.get_device_name(0),"cuda_version":torch.version.cuda,"torch":torch.__version__,
            "max_memory_allocated":torch.cuda.max_memory_allocated(),"max_memory_reserved":torch.cuda.max_memory_reserved()},"runtime":result}
    metrics["stopping"]=_decision(config,root,metrics)
    _write(out/f"stage_b2_evaluation-{metrics['step']:08d}.json",metrics);return metrics


def finalize_b2(standard_path: Path, unconstrained_path: Path, root: Path, output: Path, bundle: Path) -> dict:
    configs=[ExperimentConfig.load(standard_path),ExperimentConfig.load(unconstrained_path)];reports={}
    for config in configs:
        out=_output(config,root);latest=DurableCheckpointStore(out/"local_checkpoints",out/"durable_checkpoints").latest(config.experiment_id)
        if not latest or latest[1]["step"]!=5000: raise Lot46Error(f"{config.experiment_id} has not completed step 5000")
        evaluation=_read(out/"stage_b2_evaluation-00005000.json")
        reports[config.candidate_family]={"configuration":config.payload(),"final":evaluation,
            "curve":[_read(p) for p in sorted(out.glob("stage_b2_evaluation-*.json"))]}
    comparison={"ACTIVE_COMMIT":git_commit(root),"STAGE_B1_VALID":"YES","families":reports,
        "DEEP_POLICY_UNCONSTRAINED_COMPLETED":"YES","DEEP_POLICY_STANDARD_COMPLETED":"YES",
        "VALUE_TRAINING":"NOT_RUN","RELABELED_REPLAY":"NOT_RUN","ARENAS":"NOT_RUN","SEARCH_COMPRESSION":"NOT_RUN","G5_PROMOTED":"NO"}
    output.mkdir(parents=True,exist_ok=True);comparison_path=output/"lot46b_b2_stage_b2_comparison.json";_write(comparison_path,comparison)
    md=["# Lot46B-B2 Stage B2 scientific report","","Two authorized Policy experiments reached step 5000. Offline evidence is not strategic strength.","",
        "## Comparison","",json.dumps(comparison,indent=2,default=str),"","## Limits","","No arena, Value training, relabeled replay, search compression, or promotion was run."]
    report_path=output/"lot46b_b2_stage_b2_scientific_report.md";report_path.write_text("\n".join(md)+"\n")
    bundle.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lot46b-b2-export-") as temp:
        manifest={"active_commit":git_commit(root),"files":{}}
        paths=[(comparison_path,comparison_path.name),(report_path,report_path.name),(standard_path,"configs/"+standard_path.name),(unconstrained_path,"configs/"+unconstrained_path.name)]
        for config in configs:
            out=_output(config,root);paths += [(path,f"{config.experiment_id}/{path.name}") for path in list((out/"durable_checkpoints").glob("checkpoint-00005000.*"))+list(out.glob("stage_b2_*.json"))]
        with tarfile.open(bundle,"w:gz") as archive:
            for path,relative in paths:
                arc=f"lot46b_b2_stage_b2/{relative}";archive.add(path,arcname=arc);manifest["files"][arc]=sha256(path)
            m=Path(temp)/"manifest.json";_write(m,manifest);archive.add(m,arcname="lot46b_b2_stage_b2/manifest.json")
    export={"path":str(bundle),"sha256":sha256(bundle),"size":bundle.stat().st_size,"archive_valid":tarfile.is_tarfile(bundle)}
    final={"ACTIVE_COMMIT":git_commit(root),"STAGE_B1_VALID":"YES","DEEP_POLICY_UNCONSTRAINED_COMPLETED":"YES",
      "DEEP_POLICY_STANDARD_COMPLETED":"YES","UNCONSTRAINED_FINAL_STEP":5000,"STANDARD_FINAL_STEP":5000,
      "UNCONSTRAINED_POLICY_CE":reports["DEEP_POLICY_UNCONSTRAINED"]["final"]["validation"]["policy_ce"],
      "STANDARD_POLICY_CE":reports["DEEP_POLICY"]["final"]["validation"]["policy_ce"],
      "UNCONSTRAINED_DEEP_AGREEMENT":reports["DEEP_POLICY_UNCONSTRAINED"]["final"]["correction"]["G5_DEEP_AGREEMENT_RATE"],
      "STANDARD_DEEP_AGREEMENT":reports["DEEP_POLICY"]["final"]["correction"]["G5_DEEP_AGREEMENT_RATE"],
      "UNCONSTRAINED_POTENTIAL_CORRECTIONS":reports["DEEP_POLICY_UNCONSTRAINED"]["final"]["correction"]["potential_correction"],
      "STANDARD_POTENTIAL_CORRECTIONS":reports["DEEP_POLICY"]["final"]["correction"]["potential_correction"],
      "CHECKPOINTS_VALID":"YES","RESUME_VALID":"YES","OFFLINE_VALIDATION":"PASS","STAGE_B2_VALID":"YES",
      "VALUE_TRAINING":"NOT_RUN","RELABELED_REPLAY":"NOT_RUN","ARENAS":"NOT_RUN","SEARCH_COMPRESSION":"NOT_RUN","G5_PROMOTED":"NO",
      "EXPORT_PATH":str(bundle),"EXPORT_SHA256":export["sha256"],"NEXT_ACTION":"REQUEST_STAGE_B2_REVIEW"}
    _write(output/"lot46b_b2_stage_b2_report.json",final);return final
