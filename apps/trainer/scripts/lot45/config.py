"""Constantes pre-enregistrees du Lot 45."""
from __future__ import annotations

LOT = 45
EXPERIMENT_DIR_NAME = "lot45_g5_uniform_mcts32768_targets"
SMOKE_DIR_NAME = "lot45_smoke"
BUNDLE_NAME = "lot45_results.tar.gz"
INPUT_BUNDLE_NAME = "lot45_inputs.tar.gz"
DATASET_NAME = "G5_DEEP_AUTONOMOUS_REANALYSIS_V1"
DATASET_SCHEMA_VERSION = 1
SELECTION_FILE = "data/colab_bridge/lot45_selection.jsonl.gz"
SELECTION_MANIFEST_FILE = "data/colab_bridge/lot45_selection_manifest.json"
PROBE_POSITIONS_FILE = "data/colab_bridge/lot39_benchmark_positions.json"
TEACHER_MODEL = "POOL_G4R"

BUDGET = 32768
SMOKE_BUDGET = 64
DEFAULT_SEED = 20264501
SHARD_SIZE = 128
SMOKE_SHARD_SIZE = 8
DEFAULT_CONCURRENCY = 128
PILOT_SHARDS = 1
MIN_LEGAL_ACTIONS = 2
MAX_POSITIONS_PER_GAME = 4
HOLDOUT_FRACTION = 0.05
MAX_SELECTION = 100_000
CANDIDATE_TIERS = (10_000, 25_000, 50_000, 100_000)
POLICY_SUM_TOLERANCE = 1e-6
ONE_HOT_LIKE_SHARE = 0.95
HIGH_CONFIDENCE_OLD_SHARE = 0.50

# Part du corpus par famille de sources (priorite : recent, autonome, diversite).
# Une famille sous-dotee redistribue son reliquat aux suivantes, dans cet ordre.
FAMILY_WEIGHTS = {
    "D_REAL_HUMAN": None,              # toutes les positions eligibles (petit corpus)
    "G4_TRAINING_CROSSPLAY": 0.40,
    "G3_CROSSPLAY_SCALING": 0.20,
    "D_TEACHER_STATES_VIA_REANALYSIS": 0.15,
    "G3_GENERATOR_POOL": 0.08,
    "G2_SELFPLAY": 0.07,
    "AUTONOMOUS_REANALYSIS_STATES": 0.05,
    "G1_SELFPLAY": 0.05,
}

# Champs autorises dans une occurrence normalisee. Tout le reste est ignore :
# aucun champ teacher (best_action, action_values, teacher, PV, wdl, regrets...).
OCCURRENCE_FIELDS = ("fingerprint", "state", "source", "family", "game_id", "ply", "z", "generator", "old_target")
FORBIDDEN_TARGET_FIELDS = (
    "best_action", "action_values", "action_value_depths", "teacher", "teacher_score", "principal_variation", "pv",
    "wdl_target", "consequences", "regrets", "q_values", "qdiag", "minimax", "minimax_value",
)

LOCK_HEARTBEAT_S = 60.0
LOCK_STALE_AFTER_S = 900.0
LOCK_SETTLE_S = 20.0
