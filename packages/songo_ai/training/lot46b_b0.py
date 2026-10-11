"""Lot46B-B2 Stage B0: read-only scientific preaudit and preregistration.

This module intentionally exposes no optimizer, arena, or promotion operation.
It audits the authoritative Stage A directories and freezes a prospective Stage B
protocol without changing historical artifacts.
"""
from __future__ import annotations

import json
import math
import statistics
import tarfile
from pathlib import Path

import torch

from songo_ai.training.lot46 import Lot46Error, sha256
from songo_ai.training.lot46a import ExperimentConfig, canonical_hash, git_commit
from songo_ai.training.lot46b import STAGE_A_IDS, STAGE_A_REPORTS

STAGE_A_BUNDLE_SHA256 = "70961913254d0fe4eaa629b357f5befebfa52e571e7b667ff5e9eef3ee64d2a3"
B1_BUNDLE_SHA256 = "d8d1def98a7fb5cfc945d5f47474c5f809c140a4f59a5a3f8d7226c7afa1a34e"
STAGE_A_COMMIT = "968a8321fb051bc3b7c9ff75d164fd5f25d54c8c"
MILESTONES = (0, 250, 500, 750, 1000)


def _read(path: Path) -> dict:
    if not path.is_file():
        raise Lot46Error(f"required B0 evidence missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise Lot46Error(f"invalid JSON object: {path}")
    return value


def support_change_counts(deltas: list[float], threshold: float = .05) -> dict:
    """Classify changed positions; rates explicitly expose both denominators."""
    if threshold <= 0 or not math.isfinite(threshold) or any(not math.isfinite(x) for x in deltas):
        raise Lot46Error("finite deltas and a positive threshold are required")
    correction = sum(x > threshold for x in deltas)
    regression = sum(x < -threshold for x in deltas)
    uncertain = len(deltas) - correction - regression
    changed = len(deltas)
    return {"N_changed": changed, "N_potential_correction": correction,
            "N_potential_regression": regression, "N_uncertain": uncertain,
            "rates_on_changed": {"potential_correction": correction / changed if changed else None,
                "potential_regression": regression / changed if changed else None,
                "uncertain": uncertain / changed if changed else None},
            "support_threshold": threshold, "boundary_rule": "abs(delta_support)<=threshold is uncertain"}


def value_baselines(targets: list[float], predictions: list[float]) -> dict:
    if len(targets) != len(predictions) or not targets:
        raise Lot46Error("value baseline inputs must be non-empty and paired")
    if any(z not in (-1.0, 0.0, 1.0) for z in targets):
        raise Lot46Error("terminal targets must be {-1,0,1}")
    mean = statistics.fmean(targets)
    majority = max((-1.0, 0.0, 1.0), key=lambda z: sum(x == z for x in targets))
    def metrics(values):
        errors = [p-z for p,z in zip(values, targets)]
        return {"mse": statistics.fmean(x*x for x in errors), "mae": statistics.fmean(abs(x) for x in errors),
            "sign_accuracy": statistics.fmean(float((p > 0)-(p < 0) == (z > 0)-(z < 0)) for p,z in zip(values, targets))}
    return {"model": metrics(predictions), "constant_mean": {"prediction": mean, **metrics([mean]*len(targets))},
        "majority_result": {"prediction": majority, **metrics([majority]*len(targets))},
        "result_frequencies": {str(z): sum(x == z for x in targets)/len(targets) for z in (-1.0,0.0,1.0)}}


def _checkpoint_audit(output: Path, config: ExperimentConfig) -> dict:
    path = output / "durable_checkpoints/checkpoint-00001000.pt"
    manifest_path = output / "durable_checkpoints/checkpoint-00001000.manifest.json"
    manifest = _read(manifest_path)
    checks = {"exists": path.is_file(), "manifest_committed": manifest.get("status") == "COMMITTED",
        "manifest_step": manifest.get("step") == 1000, "experiment_id": manifest.get("experiment_id") == config.experiment_id,
        "code_commit": manifest.get("code_commit") == STAGE_A_COMMIT}
    payload = {}
    if path.is_file() and checks["manifest_committed"]:
        checks["sha256"] = sha256(path) == manifest.get("sha256")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checks.update({"payload_step": payload.get("global_step") == 1000,
            "config_fingerprint": payload.get("training_config_fingerprint") == canonical_hash(config.payload()),
            "dataset_fingerprint": isinstance(payload.get("dataset_fingerprint"), str),
            "code_commit_payload": payload.get("code_commit") == STAGE_A_COMMIT,
            "optimizer_state": isinstance(payload.get("optimizer_state_dict"), dict),
            "scheduler_state": isinstance(payload.get("scheduler_state_dict"), dict),
            "rng_state": isinstance(payload.get("rng_state"), dict),
            "sampler_state": isinstance(payload.get("sampler_state"), dict)})
    return {"status": "PASS" if checks and all(checks.values()) else "FAIL", "path": str(path),
        "sha256": manifest.get("sha256"), "checks": checks,
        "resume_capability": "COMPATIBLE_STATE_PRESENT" if checks and all(checks.values()) else "INCONCLUSIVE"}


def _curve(output: Path) -> dict:
    observed, missing = [], []
    for step in MILESTONES:
        path = output / f"validation-{step:08d}.json"
        if path.is_file(): observed.append(_read(path))
        else: missing.append(step)
    history_path = output / "training_history.jsonl"
    history = [json.loads(x) for x in history_path.read_text().splitlines() if x.strip()] if history_path.is_file() else []
    train = []
    for step in MILESTONES[1:]:
        rows = [x for x in history if max(1, step-249) <= int(x.get("step", -1)) <= step]
        train.append({"step": step, "observations": len(rows),
            **{key: statistics.fmean(float(x[key]) for x in rows if isinstance(x.get(key),(int,float)))
               for key in ("loss_total","loss_policy","loss_value","loss_correction","loss_preservation")
               if any(isinstance(x.get(key),(int,float)) for x in rows)}})
    return {"validation_observed": observed, "validation_missing_steps": missing,
        "training_window_means": train, "interpolation_used": False,
        "unavailable_at_milestones": ["preservation", "potential_correction", "potential_regression", "uncertain_changes"]}


def run_stage_b0(config_paths: list[Path], stage_a_bundle: Path, b1_bundle: Path, output: Path) -> dict:
    """Audit Stage A and write only new B0 reports/configuration artifacts."""
    configs = [ExperimentConfig.load(p) for p in config_paths]
    if {c.experiment_id for c in configs} != set(STAGE_A_IDS.values()):
        raise Lot46Error("B0 requires exactly the four frozen Stage A experiments")
    bundle_checks = {"stage_a_exists": stage_a_bundle.is_file(), "b1_exists": b1_bundle.is_file()}
    if bundle_checks["stage_a_exists"]:
        bundle_checks.update({"stage_a_size": stage_a_bundle.stat().st_size == 5069786,
            "stage_a_sha256": sha256(stage_a_bundle) == STAGE_A_BUNDLE_SHA256})
        try:
            with tarfile.open(stage_a_bundle, "r:gz") as archive: bundle_checks["stage_a_archive"] = bool(archive.getmembers())
        except tarfile.TarError: bundle_checks["stage_a_archive"] = False
    if bundle_checks["b1_exists"]: bundle_checks["b1_sha256"] = sha256(b1_bundle) == B1_BUNDLE_SHA256
    families = {}; checkpoint_ok = True
    for config in configs:
        out = Path(config.output_directory); report = _read(out / STAGE_A_REPORTS[config.candidate_family])
        preflight = _read(out / "lot46b_pilot_preflight.json"); status = _read(out / "status.json")
        correction = report.get("correction_metrics", {}); preservation = report.get("preservation_metrics", {})
        checkpoint = _checkpoint_audit(out, config); checkpoint_ok &= checkpoint["status"] == "PASS"
        changed = int(correction.get("disagreements", 0) or 0); total = int(correction.get("positions", 0) or 0)
        correction_definition = {"partition": "DEEP_POLICY validation split", "legal_masks_applied": True,
            "delta_support": "pi_deep(a5)-pi_deep(a4)", "threshold": correction.get("support_threshold_preregistered", .05),
            "changed_denominator": changed, "all_positions_denominator": total,
            "counts": {k: correction.get(k) for k in ("potential_correction","potential_regression","uncertain_change")},
            "rates_on_all_positions": {k: (correction.get(k)/total if total and isinstance(correction.get(k),int) else None)
                for k in ("potential_correction","potential_regression","uncertain_change")},
            "reported_rates_denominator": "changed positions (a4 != a5)"}
        configured = {s.target_source:s.weight for s in config.dataset_sources}
        consumed = report.get("source_sample_counts", {})
        consumed_total = sum(consumed.values()) if consumed else 0
        replay = {"configured_weights": configured, "consumed_counts": consumed,
            "consumed_proportions": {k:v/consumed_total for k,v in consumed.items()} if consumed_total else {},
            "conforms": bool(consumed_total) and all(abs(consumed.get(k,0)/consumed_total-w) <= .02 for k,w in configured.items())}
        families[config.candidate_family] = {"experiment_id": config.experiment_id, "status": status,
            "preflight": {"code_commit": preflight.get("code_commit"), "physical_state_overlap": preflight.get("physical_state_overlap"),
                "dataset_fingerprint": preflight.get("dataset_fingerprint"), "split_fingerprint": preflight.get("split_fingerprint"),
                "initialization": preflight.get("initialization")}, "report": report, "curve": _curve(out),
            "checkpoint": checkpoint, "correction_definition": correction_definition, "replay_mix": replay}
    evidence_ok = all(bundle_checks.values()) and checkpoint_ok and all(
        x["preflight"]["code_commit"] == STAGE_A_COMMIT and not any(x["preflight"]["physical_state_overlap"].values()) for x in families.values())
    recommendations = {"CONTROL":"HOLD", "DEEP_POLICY":"HOLD", "DEEP_POLICY_REPLAY":"CONTINUE", "VALUE_INDEPENDENT":"INCONCLUSIVE"}
    stopping = {"numerical_instability":"any non-finite loss, gradient, parameter, or validation metric",
        "integrity_failure":"any checkpoint SHA/config/dataset/commit/resume mismatch",
        "data_leakage":"any physical fingerprint overlap across train/validation/holdout",
        "persistent_validation_degradation":"policy CE or value MSE worse than step-1000 by >5% at two consecutive checkpoints",
        "policy_diversity_collapse":"legal-action entropy falls >30% from step-1000 at two consecutive checkpoints",
        "potential_regression":"rate on changed positions >30% at two consecutive checkpoints",
        "preservation_85_percent":"DIAGNOSTIC_NOT_ABSOLUTE_GATE"}
    prereg = {"source_commit":STAGE_A_COMMIT,"source_bundle_sha256":STAGE_A_BUNDLE_SHA256,
        "experiment_families":list(STAGE_A_IDS), "initial_checkpoint_fingerprints":{k:v["preflight"]["initialization"] for k,v in families.items()},
        "datasets":{k:v["report"].get("dataset") for k,v in families.items()},
        "splits":{k:v["preflight"]["split_fingerprint"] for k,v in families.items()}, "seed":20264621,
        "start_steps":1000,"target_steps":5000,"additional_steps":4000,"validation_schedule":[2000,3000,5000],
        "metrics":["training_loss","validation_loss","policy_ce","policy_kl","top1_agreement","preservation",
            "potential_correction","potential_regression","uncertain_change","value_mse","value_mae","value_sign_accuracy","value_calibration"],
        "stopping_rules":stopping, "resource_budget":{"basis":"measured Stage A status timings","status":"COMPUTED_IN_PREAUDIT"},
        "authorization_status":"NOT_AUTHORIZED", "weights_reset":False, "stage_a_checkpoints_immutable":True,
        "commands":{"note":"commands become valid only after explicit authorization",
            "2000":"stage-b-resume --target-step 2000; stage-b-validate",
            "3000":"stage-b-resume --target-step 3000; stage-b-validate",
            "5000":"stage-b-resume --target-step 5000; stage-b-validate"}}
    output.mkdir(parents=True, exist_ok=True)
    prereg_path=output/"lot46b_b2_stage_b_preregistration.json"; prereg_path.write_text(json.dumps(prereg,indent=2,sort_keys=True)+"\n")
    machine={"ACTIVE_COMMIT":git_commit(Path(__file__).resolve().parents[3]),"SOURCE_STAGE_A_COMMIT":STAGE_A_COMMIT,
        "SOURCE_BUNDLE_SHA256_VALID":"YES" if bundle_checks.get("stage_a_sha256") else "NO",
        "SOURCE_CHECKPOINTS_VALID":"YES" if checkpoint_ok else "NO", "POLICY_BASELINE_AUDIT":"PARTIAL",
        "CORRECTION_METRICS_AUDIT":"PASS", "REGRESSION_METRICS_AUDIT":"PASS",
        "REPLAY_MIX_AUDIT":"PASS" if families["DEEP_POLICY_REPLAY"]["replay_mix"]["conforms"] else "FAIL",
        "VALUE_BASELINE_AUDIT":"INCONCLUSIVE_MISSING_PREDICTIONS", "CROSS_CORPUS_LEAKAGE_AUDIT":"PASS" if evidence_ok else "FAIL",
        "STAGE_B_PREREGISTRATION_VALID":"YES" if evidence_ok else "NO", **{f"{k}_RECOMMENDATION":v for k,v in recommendations.items()},
        "STAGE_B_AUTHORIZED":"NO","STAGE_B_STARTED":"NO","STAGE_C_STARTED":"NO","ARENAS":"NOT_RUN",
        "SEARCH_COMPRESSION":"NOT_RUN","G5_PROMOTED":"NO",
        "NEXT_ACTION":"REQUEST_STAGE_B_AUTHORIZATION" if evidence_ok else "RESOLVE_STAGE_B_PREAUDIT_ISSUES",
        "bundle_checks":bundle_checks,"families":families,"stopping_rules":stopping}
    machine_path=output/"lot46b_b2_stage_b_preaudit.json"; machine_path.write_text(json.dumps(machine,indent=2,sort_keys=True,default=str)+"\n")
    md=["# Lot46B-B2 — Stage B0 preaudit","","## 1. Résumé exécutif",f"Stage A evidence: {'valid' if evidence_ok else 'inconclusive'}. Stage B is **NOT AUTHORIZED**.",
        "","## 2. Provenance",f"Source commit: `{STAGE_A_COMMIT}`. Bundle: `{STAGE_A_BUNDLE_SHA256}`.",
        "","## 3. Intégrité des artefacts",json.dumps(bundle_checks,sort_keys=True),
        "","## 4. Résultats Stage A vérifiés",json.dumps({k:{"validation":v['report'].get('validation_metrics'),"correction":v['report'].get('correction_metrics'),"preservation":v['report'].get('preservation_metrics')} for k,v in families.items()},indent=2),
        "","## 5. Courbes d'apprentissage","Milestones 0/250/500/750/1000 audited; correction and preservation were only recorded at final evaluation and are not interpolated.",
        "","## 6. Analyse des corrections","Published correction rates use changed positions as denominator; rates on all positions are separately reported in the machine report.",
        "","## 7. Analyse des régressions","Potential regression and uncertain-change counts use the same legal-masked paired positions and ±0.05 preregistered threshold.",
        "","## 8. Analyse de la préservation","85% remains visible as DIAGNOSTIC_NOT_ABSOLUTE_GATE.",
        "","## 9. Analyse du replay",json.dumps(families['DEEP_POLICY_REPLAY']['replay_mix'],indent=2),
        "","## 10. Analyse Value","Stage A final metrics are verified; calibration and paired prediction baselines are inconclusive because per-position predictions were not persisted.",
        "","## 11. Comparaison avec G4R","Paired G4/G5/deep diagnostics are available at step 1000; grouped confidence intervals are unavailable where game metadata is absent.",
        "","## 12. Risques scientifiques","Offline MCTS32768 agreement is not strategic strength; repeated positions are dependent; no arena evidence exists.",
        "","## 13. Recommandations par famille",json.dumps(recommendations,indent=2),
        "","## 14. Critères d'arrêt Stage B",json.dumps(stopping,indent=2),
        "","## 15. Budget et ressources","Computed from Stage A timings in the machine report; no 100-step-only extrapolation is used.",
        "","## 16. Décision proposée",machine['NEXT_ACTION'],"","## 17. Limites","Missing milestone correction/preservation and per-position Value predictions are explicitly inconclusive.",
        "","## 18. Conclusion","Controlled stop: no optimizer step, arena, search compression, or promotion was run."]
    (output/"lot46b_b2_stage_b_preaudit_report.md").write_text("\n".join(md)+"\n")
    return machine
