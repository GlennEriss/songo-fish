"""Schema, shards, validation, manifestes (etape 5 du plan directeur)."""

from .build import DEFAULT_TEACHER_CONFIG, build_dataset
from .merge import merge_releases
from .schema import (
    DATASET_VERSION,
    RULES_VERSION,
    Consequences,
    Observation,
    TeacherMeta,
    annotation_to_observation,
    canonicalize_board,
    score_to_bounded,
)
from .validate import DistributionReport, build_distribution_report, validate_observation
from .selfplay_schema import D_LAB, D_RL, DatasetFamily, RawSongoState, RLTrainingExample
from .selfplay_io import D_RL_FORMAT, D_RL_FORMAT_VERSION, read_d_rl_jsonl, write_d_rl_jsonl
from .reanalysis_schema import ReanalysisPolicyExample
from .scale_data import (TEACHER_FIELDS,deduplicate_physical,deterministic_stratified_sample,
                         intersection_matrix,manifest_hash,physical_state_key,position_only,
                         shard_for,validate_reanalysis_record)
from .reanalysis_io import iter_reanalysis_jsonl, write_reanalysis_jsonl

__all__ = [
    "DEFAULT_TEACHER_CONFIG",
    "build_dataset",
    "merge_releases",
    "DATASET_VERSION",
    "RULES_VERSION",
    "Consequences",
    "Observation",
    "TeacherMeta",
    "annotation_to_observation",
    "canonicalize_board",
    "score_to_bounded",
    "DistributionReport",
    "build_distribution_report",
    "validate_observation",
    "D_LAB",
    "D_RL",
    "DatasetFamily",
    "RawSongoState",
    "RLTrainingExample",
    "D_RL_FORMAT",
    "D_RL_FORMAT_VERSION",
    "read_d_rl_jsonl",
    "write_d_rl_jsonl",
    "ReanalysisPolicyExample",
    "iter_reanalysis_jsonl",
    "write_reanalysis_jsonl",
]
