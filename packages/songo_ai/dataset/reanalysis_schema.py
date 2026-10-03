"""Contrat Policy-only des réanalyses autonomes de positions externes."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Mapping
from .selfplay_schema import RawSongoState

@dataclass(frozen=True)
class ReanalysisPolicyExample:
    """Cible de recherche sans résultat terminal ni annotation Teacher."""
    state: RawSongoState
    legal_mask: tuple[bool, ...]
    visit_counts: tuple[int, ...]
    policy_target: tuple[float, ...]
    search_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_position_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask=tuple(bool(v) for v in self.legal_mask); visits=tuple(int(v) for v in self.visit_counts); policy=tuple(float(v) for v in self.policy_target)
        if len(mask)!=7 or len(visits)!=7 or len(policy)!=7: raise ValueError("mask, visits and policy must contain 7 values")
        if any(v<0 for v in visits): raise ValueError("visit counts must be non-negative")
        if any(not math.isfinite(v) or v<0 for v in policy): raise ValueError("policy must be finite and non-negative")
        if not math.isclose(sum(policy),1.0,abs_tol=1e-6): raise ValueError("policy must sum to 1")
        if any((visits[a] or policy[a]) for a,legal in enumerate(mask) if not legal): raise ValueError("illegal actions must have zero mass")
        forbidden={"best_action","action_values","teacher","value_target","z","winner","minimax_score"}
        if forbidden & (set(self.search_metadata)|set(self.source_position_metadata)): raise ValueError("Teacher/value fields are forbidden")
        if self.source_position_metadata.get("source")!="D_TEACHER_POSITION_ONLY": raise ValueError("invalid source")
        object.__setattr__(self,"legal_mask",mask); object.__setattr__(self,"visit_counts",visits); object.__setattr__(self,"policy_target",policy)
        object.__setattr__(self,"search_metadata",dict(self.search_metadata)); object.__setattr__(self,"source_position_metadata",dict(self.source_position_metadata))
