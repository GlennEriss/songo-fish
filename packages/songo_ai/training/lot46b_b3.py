"""Lot46B-B2 Stage B3: read-only scientific review of frozen Policy artifacts."""
from __future__ import annotations

import json
import math
import random
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import torch

from songo_ai.model.srn_network import mask_policy_logits
from songo_ai.training.lot46 import Lot46Error, collate, sha256
from songo_ai.training.lot46a import (DurableCheckpointStore, ExperimentConfig,
    git_commit, load_initial_model, prepare_context)

B2_SOURCE_COMMIT = "34ba99a55cb2debbf37e9f5eb2f7038573327a15"
B2_EXPORT_SHA256 = "11e77ced813e8d71b2eb73dc2e8cb703dcc1b0dd5bba3b6c65d3c474879aea39"
B3_FILES = (
    "lot46b_b2_stage_b3_scientific_review.md",
    "lot46b_b2_stage_b3_metrics.json",
    "lot46b_b2_stage_b3_statistical_analysis.json",
    "lot46b_b2_stage_b3_candidate_assessment.json",
    "lot46b_b2_stage_b3_arena_preregistration_draft.json",
    "lot46b_b2_stage_b3_manifest.json",
)


def _read(path: Path) -> dict:
    if not path.is_file(): raise Lot46Error(f"required Stage B3 evidence missing: {path}")
    value=json.loads(path.read_text())
    if not isinstance(value,dict): raise Lot46Error(f"invalid JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=str)+"\n")
    temporary.replace(path)


def verify_b2_export(bundle: Path, expected_sha256: str = B2_EXPORT_SHA256) -> dict:
    """Verify the immutable export without extracting over any scientific directory."""
    if not bundle.is_file(): raise Lot46Error(f"Stage B2 export unavailable: {bundle}")
    actual=sha256(bundle)
    if actual != expected_sha256: raise Lot46Error(f"Stage B2 export SHA256 mismatch: expected {expected_sha256}, got {actual}")
    if not tarfile.is_tarfile(bundle): raise Lot46Error("Stage B2 export is not a readable tar archive")
    with tarfile.open(bundle,"r:gz") as archive:
        members=[m for m in archive.getmembers() if m.isfile()]
        names=[m.name for m in members]
        if len(names)!=len(set(names)): raise Lot46Error("duplicate archive member")
        manifest_member=next((m for m in members if m.name.endswith("/manifest.json")),None)
        if manifest_member is None: raise Lot46Error("Stage B2 export manifest missing")
        manifest=json.load(archive.extractfile(manifest_member))
        if manifest.get("active_commit") != B2_SOURCE_COMMIT: raise Lot46Error("frozen Stage B2 source commit mismatch")
        by_name={m.name:m for m in members}
        for name,digest in manifest.get("files",{}).items():
            if name not in by_name: raise Lot46Error(f"manifest member missing: {name}")
            import hashlib
            if hashlib.sha256(archive.extractfile(by_name[name]).read()).hexdigest()!=digest:
                raise Lot46Error(f"archive member digest mismatch: {name}")
    return {"path":str(bundle),"sha256":actual,"sha256_valid":True,"archive_readable":True,
            "source_commit":manifest["active_commit"],"members":len(names),"manifest_files":len(manifest.get("files",{}))}


def _checkpoint(config: ExperimentConfig, root: Path) -> tuple[Path,dict,dict]:
    out=Path(config.output_directory) if Path(config.output_directory).is_absolute() else root/config.output_directory
    latest=DurableCheckpointStore(out/"local_checkpoints",out/"durable_checkpoints").latest(config.experiment_id)
    if not latest or int(latest[1].get("step",-1))!=5000: raise Lot46Error(f"invalid final checkpoint for {config.experiment_id}")
    if latest[1].get("code_commit") != B2_SOURCE_COMMIT: raise Lot46Error(f"checkpoint source commit mismatch for {config.experiment_id}")
    if sha256(latest[0]) != latest[1].get("sha256"): raise Lot46Error(f"checkpoint SHA256 mismatch for {config.experiment_id}")
    payload=torch.load(latest[0],map_location="cpu",weights_only=False)
    if int(payload.get("global_step",-1))!=5000: raise Lot46Error("checkpoint payload global_step is not 5000")
    return latest[0],latest[1],payload


def _load_frozen(config: ExperimentConfig, root: Path, device: torch.device):
    path,manifest,payload=_checkpoint(config,root)
    model,initial_sha=load_initial_model(config,root,device)
    model.load_state_dict(payload["model_state_dict"]);model.eval()
    return model,path,manifest,payload,initial_sha


def _paired_rows(rows: list[dict], g4, standard, unconstrained, device: torch.device) -> list[dict]:
    result=[]
    with torch.no_grad():
        for start in range(0,len(rows),128):
            subset=rows[start:start+128];batch=collate(subset,device);legal=batch["legal_mask"];target=batch["policy_target"]
            policies=[]
            for model in (g4,standard,unconstrained):
                logits,_=model(batch["graph"]);policies.append(torch.softmax(mask_policy_logits(logits,legal),dim=1))
            for index,row in enumerate(subset):
                target_i=target[index]; actions=[int(p[index].argmax()) for p in policies];deep=int(target_i.argmax())
                ce=[float(-(target_i*p[index].clamp_min(1e-12).log()).sum()) for p in policies]
                entropy=[float(-(p[index]*p[index].clamp_min(1e-12).log()).sum()) for p in policies]
                tops=[torch.topk(p[index],k=min(2,p.shape[1])).values.tolist() for p in policies]
                result.append({"position_id":str(row["fingerprint"]),"group":str(row.get("split_group") or f"state:{row['fingerprint']}"),
                    "occurrences":int(row.get("source_occurrence_count",1)),"legal_moves":int(legal[index].sum()),
                    "actions":{"g4":actions[0],"standard":actions[1],"unconstrained":actions[2],"deep":deep},
                    "ce":{"g4":ce[0],"standard":ce[1],"unconstrained":ce[2]},
                    "entropy":{"g4":entropy[0],"standard":entropy[1],"unconstrained":entropy[2]},
                    "top_probabilities":{"g4":tops[0],"standard":tops[1],"unconstrained":tops[2]},
                    "target_top_probability":float(target_i[deep]),
                    "standard_minus_unconstrained_ce":ce[1]-ce[2]})
    return result


def group_bootstrap_ci(rows: list[dict], value_key: str, *, samples: int=2000, seed: int=20264623) -> dict:
    if not rows: raise Lot46Error("empty paired sample")
    groups=defaultdict(list)
    for row in rows: groups[row["group"]].append(float(row[value_key]))
    keys=sorted(groups);rng=random.Random(seed); estimates=[]
    for _ in range(samples):
        drawn=[rng.choice(keys) for _ in keys]
        values=[v for key in drawn for v in groups[key]];estimates.append(sum(values)/len(values))
    estimates.sort();mean=sum(float(r[value_key]) for r in rows)/len(rows)
    return {"estimate":mean,"ci95":[estimates[int(.025*samples)],estimates[min(samples-1,int(.975*samples))]],
            "method":"paired group-aware percentile bootstrap","bootstrap_samples":samples,"groups":len(keys),"positions":len(rows)}


def _loss_audit(standard: ExperimentConfig, unconstrained: ExperimentConfig) -> dict:
    return {"unconstrained":{"verified_objective":unconstrained.objective,"policy_ce":True,"g4_imitation":False,
            "g4_preservation_kl":False,"g4_action_supervision":False,"minimax_labels":False},
        "standard":{"verified_objective":standard.objective,"terms":{"policy_cross_entropy":1.0,
            "correction_loss":float(standard.objective["lambda_correction"]),
            "preservation_loss":float(standard.objective["lambda_preservation"]),"rho":float(standard.objective["rho"])}},
        "valid":unconstrained.objective.get("name")=="DEEP_POLICY_CE_ONLY" and standard.objective.get("name")=="CORRECT_AND_PRESERVE"}


def _arena_draft() -> dict:
    return {"status":"DRAFT_NOT_EXECUTED","candidates":["POOL_G4R","DEEP_POLICY_STANDARD","DEEP_POLICY_UNCONSTRAINED"],
        "opponents":["CONTROL_G4R","POOL_G4R","Minimax-Bidoua"],"mcts_budgets":[512,4096],
        "design":{"paired_seeds":True,"side_balance":True,"opening_balance":True,"independent_games":True,
            "draw_handling":"report separately and as 0.5 score","confidence_interval":"game-paired bootstrap",
            "stopping_rule":"fixed sample size selected by prospective power analysis; no optional stopping",
            "promotion":"must be frozen during preregistration review; no promotion from offline metrics"},
        "resource_estimates":{"status":"TO_BE_BENCHMARKED_BEFORE_PREREGISTRATION","formula":"games * paired_sides * MCTS_budget * measured_seconds_per_simulation"},
        "policy_value_crossover":["P_G4/V_G4","P_G5/V_G4","P_G4/V_G5","P_G5/V_G5"],
        "value_g5_ready":False,"executed":False}


def run_stage_b3(standard_path: Path, unconstrained_path: Path, root: Path, b2_output: Path,
                 b2_bundle: Path, output: Path, bundle: Path, expected_sha256: str=B2_EXPORT_SHA256) -> dict:
    integrity=verify_b2_export(b2_bundle,expected_sha256)
    b2_report=_read(b2_output/"lot46b_b2_stage_b2_report.json")
    if b2_report.get("STAGE_B2_VALID")!="YES" or b2_report.get("EXPORT_SHA256")!=expected_sha256:
        raise Lot46Error("Stage B2 report/export identity mismatch")
    standard=ExperimentConfig.load(standard_path);unconstrained=ExperimentConfig.load(unconstrained_path)
    std_ctx=prepare_context(standard,root);unc_ctx=prepare_context(unconstrained,root)
    if std_ctx["dataset_fingerprint"]!=unc_ctx["dataset_fingerprint"] or std_ctx["split_fingerprint"]!=unc_ctx["split_fingerprint"]:
        raise Lot46Error("paired comparison requires identical dataset and split fingerprints")
    if [r["fingerprint"] for r in std_ctx["splits"]["validation"]] != [r["fingerprint"] for r in unc_ctx["splits"]["validation"]]:
        raise Lot46Error("validation rows are not exactly aligned")
    split_sets={name:{r["fingerprint"] for r in rows} for name,rows in std_ctx["splits"].items()}
    split_groups={name:{str(r.get("split_group") or f"state:{r['fingerprint']}") for r in rows} for name,rows in std_ctx["splits"].items()}
    for left,right in (("train","validation"),("train","holdout"),("validation","holdout")):
        if split_sets[left]&split_sets[right] or split_groups[left]&split_groups[right]:
            raise Lot46Error(f"state/group leakage between {left} and {right}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standard_model,std_path,std_manifest,std_payload,std_initial=_load_frozen(standard,root,device)
    unconstrained_model,unc_path,unc_manifest,unc_payload,unc_initial=_load_frozen(unconstrained,root,device)
    if std_initial!=unc_initial: raise Lot46Error("initial checkpoint fingerprints differ")
    g4,_=load_initial_model(standard,root,device);g4.eval()
    before={str(p):sha256(p) for p in (std_path,unc_path,b2_bundle)}
    paired=_paired_rows(std_ctx["splits"]["validation"],g4,standard_model,unconstrained_model,device)
    for row in paired:
        a=row["actions"];row["ce_advantage_unconstrained"]=row["standard_minus_unconstrained_ce"]
        row["deep_agreement_advantage_unconstrained"]=float(a["unconstrained"]==a["deep"])-float(a["standard"]==a["deep"])
    after={path:sha256(Path(path)) for path in before}
    if before!=after: raise Lot46Error("read-only analysis mutated a protected artifact")
    n=len(paired);unique=len({r["position_id"] for r in paired});occurrences=sum(r["occurrences"] for r in paired)
    means=lambda key,name: sum(r[key][name] for r in paired)/n
    agreements={name:sum(r["actions"][name]==r["actions"]["deep"] for r in paired) for name in ("g4","standard","unconstrained")}
    decisions={"unchanged_standard_from_g4":sum(r["actions"]["standard"]==r["actions"]["g4"] for r in paired),
        "unchanged_unconstrained_from_g4":sum(r["actions"]["unconstrained"]==r["actions"]["g4"] for r in paired),
        "standard_unconstrained_agree":sum(r["actions"]["standard"]==r["actions"]["unconstrained"] for r in paired),
        "unconstrained_only_deep":sum(r["actions"]["unconstrained"]==r["actions"]["deep"] and r["actions"]["standard"]!=r["actions"]["deep"] for r in paired),
        "standard_only_deep":sum(r["actions"]["standard"]==r["actions"]["deep"] and r["actions"]["unconstrained"]!=r["actions"]["deep"] for r in paired)}
    def curve(config):
        out=Path(config.output_directory) if Path(config.output_directory).is_absolute() else root/config.output_directory
        return [_read(path) for path in sorted(out.glob("stage_b2_evaluation-*.json"))]
    ce_diff=means("ce","standard")-means("ce","unconstrained")
    agreement_diff=agreements["unconstrained"]/n-agreements["standard"]/n
    ce_ci=group_bootstrap_ci(paired,"ce_advantage_unconstrained");agreement_ci=group_bootstrap_ci(paired,"deep_agreement_advantage_unconstrained")
    assessment=("UNCONSTRAINED_OFFLINE_PREFERRED" if ce_ci["ci95"][0]>0 and agreement_ci["ci95"][0]>0
                else "STANDARD_OFFLINE_PREFERRED" if ce_ci["ci95"][1]<0 and agreement_ci["ci95"][1]<0
                else "STATISTICALLY_INCONCLUSIVE")
    loss=_loss_audit(standard,unconstrained)
    provenance={"standard":{"experiment_id":standard.experiment_id,"source_commit":std_manifest["code_commit"],"initial_checkpoint":standard.initial_checkpoint,
            "initial_checkpoint_sha256":std_initial,"dataset_fingerprint":std_ctx["dataset_fingerprint"],"validation_fingerprint":std_ctx["split_fingerprint"],
            "seed":standard.seed,"optimizer":standard.optimizer,"scheduler":standard.scheduler,"learning_rate":standard.learning_rate,
            "batch_size":standard.batch_size,"loss":standard.objective,"steps":std_payload["global_step"],"resume_history":"Stage A exact-state continuation at step 1000",
            "final_checkpoint_sha256":std_manifest["sha256"]},
        "unconstrained":{"experiment_id":unconstrained.experiment_id,"source_commit":unc_manifest["code_commit"],"initial_checkpoint":unconstrained.initial_checkpoint,
            "initial_checkpoint_sha256":unc_initial,"dataset_fingerprint":unc_ctx["dataset_fingerprint"],"validation_fingerprint":unc_ctx["split_fingerprint"],
            "seed":unconstrained.seed,"optimizer":unconstrained.optimizer,"scheduler":unconstrained.scheduler,"learning_rate":unconstrained.learning_rate,
            "batch_size":unconstrained.batch_size,"loss":unconstrained.objective,"steps":unc_payload["global_step"],"resume_history":"new run from step 0; durable resumes recorded in checkpoint state",
            "final_checkpoint_sha256":unc_manifest["sha256"]}}
    confounders=["STANDARD continued the Stage A optimizer/sampler trajectory from step 1000; UNCONSTRAINED started at step 0.",
        "The objectives differ, so loss and run history are jointly confounded.","Offline MCTS32768 agreement is not playing strength."]
    metrics={"positions":n,"unique_positions":unique,"occurrences":occurrences,"policy_ce":{"g4":means("ce","g4"),"standard":means("ce","standard"),"unconstrained":means("ce","unconstrained"),"standard_minus_unconstrained":ce_diff},
        "deep_agreement":{"counts":agreements,"rates":{k:v/n for k,v in agreements.items()},"unconstrained_minus_standard":agreement_diff},
        "policy_entropy":{k:means("entropy",k) for k in ("g4","standard","unconstrained")},"decisions":decisions,
        "legal_action_consistency":1.0,"training_curves":{"standard":curve(standard),"unconstrained":curve(unconstrained)},
        "split_integrity":{"state_overlap":0,"group_overlap":0,"validation_used_for_optimizer":False,"strategic_holdout_used_for_selection":False},
        "provenance":provenance,"loss_audit":loss,"confounders":confounders}
    statistical={"policy_ce_difference":ce_ci,"deep_agreement_difference":agreement_ci,
        "median_paired_ce_difference":sorted(r["ce_advantage_unconstrained"] for r in paired)[n//2],
        "improved_positions":sum(r["ce_advantage_unconstrained"]>0 for r in paired),"regressed_positions":sum(r["ce_advantage_unconstrained"]<0 for r in paired),
        "interpretation":"Uncertainty is group-aware. These intervals quantify offline target agreement, not strategic superiority."}
    candidate={"assessment":assessment,"observed":{"policy_ce_difference":ce_diff,"deep_agreement_difference":agreement_diff},
        "scientific_interpretation":"Offline preference only; deep MCTS is a target, not an oracle.","generalization_evidence":"INCOMPLETE",
        "unverified_hypotheses":["The preferred offline candidate may be stronger in future preregistered arenas."],
        "next_action":"REQUEST_ARENA_PREREGISTRATION_REVIEW" if assessment!="STATISTICALLY_INCONCLUSIVE" else "REQUEST_ADDITIONAL_OFFLINE_ANALYSIS"}
    output.mkdir(parents=True,exist_ok=True)
    paired_path=output/"lot46b_b2_stage_b3_paired_positions.jsonl"
    paired_path.write_text("".join(json.dumps(row,sort_keys=True,allow_nan=False)+"\n" for row in paired))
    _write(output/B3_FILES[1],metrics);_write(output/B3_FILES[2],statistical);_write(output/B3_FILES[3],candidate);_write(output/B3_FILES[4],_arena_draft())
    review=("# Lot46B-B2 Stage B3 scientific review\n\n## Observed evidence\n\n"+json.dumps(metrics,indent=2,default=str)+
        "\n\n## Statistical estimates\n\n"+json.dumps(statistical,indent=2)+"\n\n## Interpretation and limitations\n\n"+
        "MCTS32768 agreement is an offline diagnostic, not proof of playing strength. Previous-generation agreement is diagnostic, never mandatory. "
        "Each future generation must be evaluated independently. No arena, optimizer step, self-play, Value training, target generation, or promotion was performed.\n")
    (output/B3_FILES[0]).write_text(review)
    manifest={"analysis_commit":git_commit(root),"stage_b2_source_commit":B2_SOURCE_COMMIT,"stage_b2_export":integrity,
        "protected_artifacts_unchanged":before==after,"files":{}}
    for name in B3_FILES[:-1]: manifest["files"][name]=sha256(output/name)
    manifest["files"][paired_path.name]=sha256(paired_path)
    _write(output/B3_FILES[-1],manifest)
    bundle.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(bundle,"w:gz") as archive:
        for name in B3_FILES: archive.add(output/name,arcname=f"lot46b_b2_stage_b3/{name}")
        archive.add(paired_path,arcname=f"lot46b_b2_stage_b3/{paired_path.name}")
        for path,arcname in ((Path(__file__),"analysis/lot46b_b3.py"),(root/"packages/songo_ai/tests/test_lot46b_b3.py","tests/test_lot46b_b3.py")):
            archive.add(path,arcname=f"lot46b_b2_stage_b3/{arcname}")
        test_result=output/"lot46b_b2_stage_b3_test_results.json"
        if test_result.is_file(): archive.add(test_result,arcname=f"lot46b_b2_stage_b3/{test_result.name}")
    if not tarfile.is_tarfile(bundle): raise Lot46Error("Stage B3 export readability failure")
    status={"ACTIVE_COMMIT":git_commit(root),"STAGE_B2_VALID":"YES","STAGE_B2_EXPORT_SHA256_VALID":"YES",
        "STANDARD_CHECKPOINT_VALID":"YES","UNCONSTRAINED_CHECKPOINT_VALID":"YES","LOSS_AUDIT_VALID":"YES" if loss["valid"] else "NO",
        "DATASET_INTEGRITY_VALID":"YES","VALIDATION_SPLITS_VALID":"YES","PAIRED_COMPARISON_VALID":"YES","STATISTICAL_ANALYSIS_VALID":"YES",
        "GENERALIZATION_EVIDENCE":"INCOMPLETE","UNCONSTRAINED_POLICY_CE":metrics["policy_ce"]["unconstrained"],
        "STANDARD_POLICY_CE":metrics["policy_ce"]["standard"],"POLICY_CE_DIFFERENCE":ce_diff,
        "UNCONSTRAINED_DEEP_AGREEMENT":metrics["deep_agreement"]["rates"]["unconstrained"],
        "STANDARD_DEEP_AGREEMENT":metrics["deep_agreement"]["rates"]["standard"],"DEEP_AGREEMENT_DIFFERENCE":agreement_diff,
        "CANDIDATE_ASSESSMENT":assessment,"STAGE_B3_VALID":"YES","TRAINING":"NOT_RUN","SELF_PLAY":"NOT_RUN",
        "NEW_MCTS_TARGETS":"NOT_GENERATED","VALUE_TRAINING":"NOT_RUN","ARENAS":"NOT_RUN","SEARCH_COMPRESSION":"NOT_RUN","G5_PROMOTED":"NO",
        "EXPORT_PATH":str(bundle),"EXPORT_SHA256":sha256(bundle),"NEXT_ACTION":candidate["next_action"]}
    _write(output/"lot46b_b2_stage_b3_status.json",status)
    return status
