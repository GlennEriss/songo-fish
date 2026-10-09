"""Contrats et pipelines d'entrainement generationnels Songo."""

from .g4_design import (DesignError, generator_admission_decision,
                        promotion_decision, validate_g4_design)
from .lot46 import (Lot46Dataset, Lot46Error, atomic_torch_save, checkpoint_diagnostic,
                    group_aware_split, lot46_loss, reconstruct_policy_target)

__all__ = ["DesignError", "validate_g4_design", "promotion_decision",
           "generator_admission_decision", "Lot46Dataset", "Lot46Error",
           "atomic_torch_save", "checkpoint_diagnostic", "group_aware_split", "lot46_loss",
           "reconstruct_policy_target"]
