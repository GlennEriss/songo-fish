"""Constantes pre-enregistrees du Lot 44.

Tout ce qui est fige avant l'ouverture du TEST vit ici : budgets, definition
du label, groupes de features, regle de seuil et criteres de decision. Ces
dictionnaires sont ecrits tels quels dans ``label_definition.json`` et
``protocol.json`` puis leur empreinte est verifiee a chaque stage.
"""
from __future__ import annotations

LOT = 44
EXPERIMENT_DIR_NAME = "lot44_router_out_of_sample_validation"
SMOKE_DIR_NAME = "lot44_smoke"
BUNDLE_NAME = "lot44_results.tar.gz"
INPUT_BUNDLE_NAME = "lot44_inputs.tar.gz"
CANDIDATES_FILE = "data/colab_bridge/lot44_oos_candidates.json"
ORIGINAL_POSITIONS_FILE = "data/colab_bridge/lot39_benchmark_positions.json"
SOURCE_POOL_DIR = "data/experiments/lot34_g4_training/pool_data"
COUNTEREXAMPLE_FINGERPRINT = "fb248b86b7cb017a4df1c8a3141b81485d1760debb659ef5f2d95de10fc970ab"

# Roles de budget : l1 -> l2 -> l3 forment l'echelle disponible avant routage,
# ``ref`` est la reference profonde qui definit le label.
ROLES = ("l1", "l2", "l3", "ref")
PRE_ROUTING_ROLES = ("l1", "l2", "l3")
PRODUCTION_BUDGETS = {"l1": 4096, "l2": 8192, "l3": 32768, "ref": 65536}
SMOKE_BUDGETS = {"l1": 4, "l2": 8, "l3": 16, "ref": 256}

# Concurrence par budget, bornee par la RAM Colab (Lot41 : 32768 x 256 arbres
# = 11.5 Go de pic, soit ~1.5 Ko par noeud).
DEFAULT_CONCURRENCY = {4096: 256, 8192: 256, 32768: 128, 65536: 64}
BYTES_PER_NODE_ESTIMATE = 1600
BASE_PROCESS_BYTES_ESTIMATE = 1_500_000_000
MAX_RAM_FRACTION = 0.80

C_PUCT = 1.5
POLICY_TEMPERATURE = 1.0
ROOT_NOISE = False
DEFAULT_SEED = 20264401
DEFAULT_MAX_POSITIONS = 768
SMOKE_POSITIONS = 96
SPLIT_FRACTIONS = {"train": 0.60, "calibration": 0.20, "test": 0.20}
PARTITIONS = ("train", "calibration", "test")
MIN_LEGAL_ACTIONS = 2
POSITIONS_PER_GAME = 1

LABEL_DEFINITION = {
    "name": "MATERIAL_LATE_BIFURCATION_32768_65536",
    "version": "V1",
    "inherited_from": "Lot42 ultra_hard definition (run_srn_lot42.finalize: material = flip and (js > 0.05 or regret > 0.10))",
    "shallow_role": "l3",
    "reference_role": "ref",
    "positive_rule": "top1(l3) != top1(ref) AND (JS(pi_l3, pi_ref) > js_threshold OR regret > regret_threshold)",
    "regret_definition": "Q_ref[top1(ref)] - Q_ref[top1(l3)], root perspective, only when both actions have ref visits > 0; otherwise UNAVAILABLE and only the JS branch can fire",
    "js_threshold": 0.05,
    "regret_threshold": 0.10,
    "severity_bins": {"NEAR_TIE": 0.02, "LOW_SEVERITY": 0.05, "MEDIUM_SEVERITY": 0.10},
    "high_severity_rule": "positive AND regret > 0.10",
    "any_top1_change_is_positive": False,
}

PROTOCOL = {
    "version": "LOT44_PROTOCOL_V1",
    "primary_router": "combined",
    "ablation_models": ["classic", "trajectory", "combined"],
    "classifier": "L2-regularized logistic regression, balanced class weights, Newton/IRLS, standardized features",
    "l2_strength": 1.0,
    "cv_folds_max": 5,
    "calibration_method_rule": "PLATT if calibration has >= 5 positives and >= 5 negatives, else IDENTITY (scores used uncalibrated, documented)",
    "calibration_min_class_count": 5,
    "threshold_rule": "threshold = safety_factor * min(calibrated score of development positives), development = out-of-fold TRAIN + CALIBRATION; no development positive -> threshold 0 (route all)",
    "threshold_safety_factor": 0.8,
    "early_stop_8192_allowed": False,
    "conservative_router": "Lot43 CONSERVATIVE_MULTI_SIGNAL 32768 trigger with Lot43 frozen thresholds, no early stop",
    "conservative_js_trigger": 0.025,
    "conservative_q_gap_trigger": 0.035,
    "conservative_margin_trigger": 0.15,
    "min_oos_positives_for_verdict": 15,
    "router_validated_rule": "positives >= 15 AND recall >= 0.90 AND Wilson95 recall low >= 0.70 AND zero high-severity false negatives AND routing rate <= 0.50",
    "router_validated_min_recall": 0.90,
    "router_validated_min_recall_ci_low": 0.70,
    "router_validated_max_routing_rate": 0.50,
    "trajectory_value_rule": "paired bootstrap of AP(combined) - AP(classic) on TEST: YES if CI95 low > 0, NO if CI95 high <= 0, else INCONCLUSIVE; INCONCLUSIVE if positives < 15",
    "restart_viable_rule": "ROUTER_VALIDATED == YES AND classifier restart cost <= 0.75 * uniform 65536 cost",
    "resume_viable_rule": "ROUTER_VALIDATED == YES AND classifier resume cost <= 0.75 * uniform 65536 cost",
    "viability_cost_ratio": 0.75,
    "uniform_32768_defensible_rule": "high-severity late bifurcation rate: YES if Wilson95 high <= 0.03, NO if Wilson95 low > 0.03, else INCONCLUSIVE",
    "uniform_32768_high_severity_rate_limit": 0.03,
    "next_action_rule": "LOT44_VALID=NO -> MORE_ANALYSIS_REQUIRED; ROUTER_VALIDATED=YES and restart viable -> LOT45_ROUTED_AUTONOMOUS_TARGET_GENERATION; YES and only resume viable -> LOT45_MCTS_TREE_RESUME_ENGINEERING; YES otherwise -> LOT45_G5_UNIFORM_32768_TARGET_GENERATION; NO or INCONCLUSIVE -> LOT45_G5_UNIFORM_32768_TARGET_GENERATION (section 107: do not open new analysis lots)",
    "bootstrap_resamples": 2000,
    "ece_bins": 10,
    "test_evaluations_allowed": 1,
}
