"""Schema, shards, validation, manifestes (etape 5 du plan directeur)."""

from .build import DEFAULT_TEACHER_CONFIG, build_dataset
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

__all__ = [
    "DEFAULT_TEACHER_CONFIG",
    "build_dataset",
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
]
