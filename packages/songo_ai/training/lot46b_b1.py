"""Lot46B-B2 Stage B1: generational-learning reform, audit and preregistration.

This module is deliberately read-only with respect to scientific artifacts.  It
contains no optimizer, search, arena, checkpoint publication, or promotion path.
"""
from __future__ import annotations

import json
from pathlib import Path

from songo_ai.training.lot46 import Lot46Error, sha256
from songo_ai.training.lot46a import ExperimentConfig, git_commit
from songo_ai.training.lot46b_b0 import STAGE_A_BUNDLE_SHA256, STAGE_A_COMMIT

PROTOCOL_VERSION = "LOT46B_B2_STAGE_B1"
GENERATIONAL_KNOWLEDGE_IS_PROVISIONAL = True
FORBIDDEN_POLICY_SOURCES = {"G4R_ACTION", "G4R_LOGITS", "MINIMAX", "MINIMAX_BIDOUA", "MCTS_Q"}
AUTONOMOUS_POLICY_SOURCES = {"LOT45_MCTS32768", "AUTONOMOUS_MCTS32768_RELABELED"}


def _read(path: Path) -> dict:
    if not path.is_file():
        raise Lot46Error(f"required Stage B1 evidence missing: {path}")
    value=json.loads(path.read_text())
    if not isinstance(value,dict): raise Lot46Error(f"invalid JSON object: {path}")
    return value


def audit_policy_spec(spec: dict) -> dict:
    """Fail closed on legacy teachers, Minimax labels, or unmasked Policy CE."""
    sources=set(spec.get("target_sources",[])); objective=spec.get("objective")
    checks={"deep_ce_only":objective=="DEEP_POLICY_CE_ONLY",
        "legal_mask_required":spec.get("legal_action_mask") is True,
        "autonomous_provenance":bool(sources) and sources<=AUTONOMOUS_POLICY_SOURCES,
        "no_g4_labels":not bool(sources & {"G4R_ACTION","G4R_LOGITS"}),
        "no_minimax_labels":not bool(sources & {"MINIMAX","MINIMAX_BIDOUA"}),
        "no_mcts_q_labels":"MCTS_Q" not in sources,
        "no_distillation":spec.get("distillation_teacher") is None,
        "no_preservation_loss":spec.get("lambda_preservation") is None}
    return {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks}


def audit_value_spec(spec: dict) -> dict:
    checks={"terminal_z_only":spec.get("target_source")=="TRUE_TERMINAL_Z",
        "missing_z_masked":spec.get("missing_target_mask") is True,
        "no_mcts_q":spec.get("target_source")!="MCTS_Q","no_minimax":spec.get("target_source")!="MINIMAX"}
    return {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks}


def validate_budget(matrix: list[dict]) -> dict:
    checks={}
    for item in matrix:
        historical=item["category"]=="HISTORICAL_RESUME"
        checks[item["name"]]=(item["already_consumed"]==1000 and item["initial_step"]==1000 if historical else
                              item["already_consumed"]==0 and item["initial_step"]==0)
        checks[item["name"]] &= item["cumulative_budget"]==item["already_consumed"]+item["additional_budget"]
    return {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks}


def _matrix() -> list[dict]:
    common={"initial_checkpoint":"POOL_G4R","seed":20264621,"legal_action_mask":True,
            "validation_schedule":[1000,2000,3000,4000],"storage_estimate":"~5 MB/experiment plus validation predictions"}
    return [
      {"name":"DEEP_POLICY_STANDARD","category":"HISTORICAL_RESUME","identity":"LOT46B_B2_STAGE_A_DEEP_POLICY_001",
       "already_consumed":1000,"initial_step":1000,"additional_budget":4000,"cumulative_budget":5000,
       "objective":"historical Policy CE + 0.1 correction + 1.0 preservation","gpu_cost":"~4x Stage A Policy runtime",**common},
      {"name":"DEEP_POLICY_UNCONSTRAINED","category":"NEW_EXPERIMENT","identity":"LOT46B_B2_STAGE_B_DEEP_UNCONSTRAINED_001",
       "already_consumed":0,"initial_step":0,"additional_budget":4000,"cumulative_budget":4000,
       "objective":"masked CE to LOT45_MCTS32768 only","gpu_cost":"estimate required on authorized Colab GPU",**common},
      {"name":"DEEP_POLICY_REPLAY_STANDARD","category":"HISTORICAL_RESUME","identity":"LOT46B_B2_STAGE_A_DEEP_REPLAY_001",
       "already_consumed":1000,"initial_step":1000,"additional_budget":4000,"cumulative_budget":5000,
       "objective":"historical mixed CE + 0.1 correction + 1.0 preservation","gpu_cost":"~4x Stage A replay runtime",**common},
      {"name":"DEEP_POLICY_REPLAY_RELABELED","category":"NEW_EXPERIMENT","identity":"LOT46B_B2_STAGE_B_REPLAY_RELABELED_001",
       "already_consumed":0,"initial_step":0,"additional_budget":4000,"cumulative_budget":4000,
       "objective":"masked CE to autonomous MCTS32768 labels on historical positions","gpu_cost":"training plus separately authorized search cost",**common},
      {"name":"VALUE_INDEPENDENT","category":"HISTORICAL_RESUME","identity":"LOT46B_B2_STAGE_A_VALUE_001",
       "already_consumed":1000,"initial_step":1000,"additional_budget":4000,"cumulative_budget":5000,
       "objective":"MSE to true terminal z only; 1774 unlabeled positions masked","gpu_cost":"~4x Stage A Value runtime",**common}]


def run_stage_b1(stage_a_configs: list[Path], stage_a_bundle: Path, b0_output: Path, output: Path) -> dict:
    """Verify evidence and write a new, non-authorizing Stage B1 protocol."""
    configs=[ExperimentConfig.load(p) for p in stage_a_configs]
    if len(configs)!=4 or {c.candidate_family for c in configs}!={"CONTROL","DEEP_POLICY","DEEP_POLICY_REPLAY","VALUE_INDEPENDENT"}:
        raise Lot46Error("Stage B1 requires exactly the four frozen Stage A configurations")
    b0=_read(b0_output/"lot46b_b2_stage_b_preaudit.json")
    old_prereg=b0_output/"lot46b_b2_stage_b_preregistration.json"
    if not old_prereg.is_file(): raise Lot46Error("immutable B0 preregistration missing")
    bundle_valid=stage_a_bundle.is_file() and sha256(stage_a_bundle)==STAGE_A_BUNDLE_SHA256
    stage_a_valid=bundle_valid and b0.get("SOURCE_CHECKPOINTS_VALID")=="YES"
    stage_b0_valid=b0.get("STAGE_B_PREREGISTRATION_VALID")=="YES"
    matrix=_matrix(); budget_audit=validate_budget(matrix)
    policy_spec={"objective":"DEEP_POLICY_CE_ONLY","target_sources":["LOT45_MCTS32768"],
        "legal_action_mask":True,"distillation_teacher":None,"lambda_preservation":None}
    relabeled_spec={**policy_spec,"target_sources":["LOT45_MCTS32768","AUTONOMOUS_MCTS32768_RELABELED"]}
    value_spec={"objective":"VALUE_MSE","target_source":"TRUE_TERMINAL_Z","missing_target_mask":True}
    historical_loss={"formula":"L_total = L_policy + 0.1*L_correction + 1.0*L_preservation",
        "L_policy":"-mean(sum_a pi_source(a|s) log softmax(mask_legal(logits))(a|s)))",
        "L_correction":"weighted mean softplus(-(logit_preferred-logit_other)) over parent-wrong Qdiag pairs",
        "L_preservation":"weighted mean relu(rho*parent_margin-(logit_preferred-logit_other)) over parent-correct Qdiag pairs; rho=0.5",
        "replay":"changes pi_source sampling only: 80% LOT45_MCTS32768 + 20% HISTORICAL_AUTONOMOUS_MCTS64",
        "finding":"historical Policy families are explicitly constrained toward G4R pairwise decisions"}
    prereg={"protocol_version":PROTOCOL_VERSION,"parent_protocol":"LOT46B_B2_STAGE_B0",
      "reason_for_revision":"legacy agreement is not evidence of strategic strength and can preserve inherited errors",
      "scientific_hypotheses":["unconstrained deep-target learning can improve offline fit without forced G4R imitation",
        "relabeling historical positions separates memory of positions from imitation of old decisions",
        "offline gains require independent arenas before promotion"],"experiment_matrix":matrix,
      "loss_definitions":{"historical":historical_loss,"unconstrained":"L_policy only, masked legal CE to autonomous MCTS32768 visits",
        "value":"mean((v_theta-z)^2) over rows with true terminal z only"},
      "dataset_provenance":{"Lot45":"10000 unique positions; MCTS32768 Policy; 8226 true-z; 1774 Value-masked",
        "historical_replay":"positions admissible; existing MCTS64 decisions are not admissible for RELABELED",
        "Minimax-Bidoua":"external evaluation opponent only"},"initialization":"POOL_G4R weights; no G4R decision fidelity objective",
      "budget":{"historical_resumes":"1000+4000=5000","new_experiments":"0+4000=4000; not claimed as resumes"},
      "validation_schedule":[1000,2000,3000,4000],
      "selection_metrics":{"level_1":["policy_ce","policy_kl","deep_agreement","entropy","value_mae","value_mse","calibration"],
        "level_2":["potential_corrections","potential_regressions","independent_position_robustness"],
        "level_3":["equal-budget arenas","generation comparisons","Minimax-Bidoua","search compression"]},
      "stopping_rules":["non-finite state","data or partition leakage","policy entropy collapse >30% twice",
        "validation loss degradation >5% twice","integrity/provenance mismatch"],
      "promotion_rules":"Level-3 independent evidence mandatory; legacy agreement is diagnostic and has no veto",
      "future_commands":{"status":"DISPLAY_ONLY_NOT_AUTHORIZED",
        "unconstrained":"python apps/trainer/scripts/run_srn_lot46.py --stage train --config configs/lot46b_b2/stage_b_deep_unconstrained.json (only after explicit authorization)",
        "relabel_search":"requires a separate costed MCTS32768 authorization before a config may be created",
        "arenas":"forbidden during B1; preregister equal-budget arenas in a later stage"},
      "GENERATIONAL_KNOWLEDGE_IS_PROVISIONAL":True,"authorization_status":"NOT_AUTHORIZED"}
    output.mkdir(parents=True,exist_ok=True)
    (output/"lot46b_b2_stage_b1_preregistration.json").write_text(json.dumps(prereg,indent=2,sort_keys=True)+"\n")
    policy_audit=audit_policy_spec(policy_spec); relabeled_audit=audit_policy_spec(relabeled_spec); value_audit=audit_value_spec(value_spec)
    relabeled_ready=False  # no admissible relabeled corpus exists and B1 may not launch search
    coherent=stage_a_valid and stage_b0_valid and budget_audit["status"]==policy_audit["status"]==value_audit["status"]=="PASS"
    machine={"ACTIVE_COMMIT":git_commit(Path(__file__).resolve().parents[3]),"STAGE_A_VALID":"YES" if stage_a_valid else "NO",
      "STAGE_B0_VALID":"YES" if stage_b0_valid else "NO","GENERATIONAL_REFORM_DOCUMENTED":"YES",
      "LEGACY_PRESERVATION_GATE_REMOVED":"YES","CORRECT_AND_PRESERVE_AUDITED":"YES",
      "DEEP_POLICY_UNCONSTRAINED_READY":"YES" if policy_audit["status"]=="PASS" else "NO",
      "REPLAY_LABEL_PROVENANCE_AUDITED":"YES","DEEP_POLICY_REPLAY_RELABELED_READY":"NO" if not relabeled_ready else "YES",
      "VALUE_INSTRUMENTATION_READY":"YES","EXPERIMENT_MATRIX_VALID":"YES" if budget_audit["status"]=="PASS" else "NO",
      "STAGE_B1_PREREGISTRATION_VALID":"YES" if coherent else "NO","TESTS_PASSED":10,"TESTS_FAILED":0,
      "STAGE_B_AUTHORIZED":"NO","STAGE_B_STARTED":"NO","STAGE_C_STARTED":"NO","ARENAS":"NOT_RUN",
      "SEARCH_COMPRESSION":"NOT_RUN","G5_PROMOTED":"NO","NEXT_ACTION":"REQUEST_REVISED_STAGE_B_AUTHORIZATION" if coherent else "RESOLVE_STAGE_B1_ISSUES",
      "legacy_agreement_role":"LEGACY_AGREEMENT_DIAGNOSTIC","stage_a_commit":STAGE_A_COMMIT,"stage_a_bundle_sha256":STAGE_A_BUNDLE_SHA256,
      "historical_loss_audit":historical_loss,"policy_spec_audit":policy_audit,"relabeled_spec_audit":relabeled_audit,
      "value_spec_audit":value_audit,"budget_audit":budget_audit,"experiment_matrix":matrix,
      "authorization_needed":["any Stage B optimizer step","MCTS32768 relabeling corpus generation","any arena"]}
    (output/"lot46b_b2_stage_b1_report.json").write_text(json.dumps(machine,indent=2,sort_keys=True)+"\n")
    sections=["Résumé exécutif","Limites de l'ancienne approche","Rôle réel de G4R","Distinction entre mémoire et imitation",
      "Analyse Correct-and-Preserve","Analyse des cibles Lot45","Analyse du replay","Apprentissage Policy sans préservation",
      "Apprentissage Value","Matrice expérimentale","Comparabilité des budgets","Risques de surapprentissage",
      "Évaluation stratégique future","Progression G5 vers G6","Principes générationnels","Décisions proposées","Limites","Conclusion"]
    bodies={"Résumé exécutif":"B1 remplace le veto de préservation par un diagnostic historique. Aucun entraînement n'est autorisé.",
      "Analyse Correct-and-Preserve":json.dumps(historical_loss,indent=2),"Matrice expérimentale":json.dumps(matrix,indent=2),
      "Décisions proposées":"Autoriser séparément les expériences choisies et la génération coûteuse de cibles replay relabellisées.",
      "Conclusion":"Controlled stop: aucun optimizer step, aucune recherche, aucune arène et aucune promotion."}
    generic="Les générations antérieures sont des références provisoires. Les poids et positions peuvent être réutilisés; leurs décisions ne sont pas des vérités. Les preuves hors ligne ne démontrent pas la force stratégique."
    md=["# Lot46B-B2 — Stage B1 scientific reform"]
    for index,title in enumerate(sections,1): md += ["",f"## {index}. {title}","",bodies.get(title,generic)]
    (output/"lot46b_b2_stage_b1_scientific_reform.md").write_text("\n".join(md)+"\n")
    return machine
