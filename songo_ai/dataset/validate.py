"""Controles qualite obligatoires avant toute release (section 6.6) :
zero NaN/Inf, policy normalisee, masque legal coherent, aucune position
terminale, plus un rapport de distribution par phase/coups legaux/score."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from songo_ai.songo.rules import BOARD_SIZE, PLAYER_ONE, TOTAL_SEEDS, SongoLegacyGame

from .schema import Observation


def validate_observation(obs: Observation) -> List[str]:
    problems: List[str] = []

    if len(obs.state) != BOARD_SIZE:
        problems.append(f"state must have {BOARD_SIZE} counters, got {len(obs.state)}")
    elif sum(obs.state) != TOTAL_SEEDS:
        problems.append(f"seed conservation violated: sum={sum(obs.state)}")
    elif any(v < 0 for v in obs.state):
        problems.append("negative seed count in state")

    if len(obs.legal_mask) != 7:
        problems.append(f"legal_mask must have 7 entries, got {len(obs.legal_mask)}")
    elif not any(obs.legal_mask):
        problems.append("legal_mask has no legal move: position is terminal, must not appear in dataset")

    if not (0 <= obs.best_action <= 6) or (len(obs.legal_mask) == 7 and not obs.legal_mask[obs.best_action]):
        problems.append(f"best_action {obs.best_action} is not a legal move")

    if len(obs.policy_target) != 7:
        problems.append("policy_target must have 7 entries")
    else:
        if any(math.isnan(p) or math.isinf(p) for p in obs.policy_target):
            problems.append("policy_target contains NaN/Inf")
        if any(p < 0 for p in obs.policy_target):
            problems.append("policy_target has negative probability")
        if abs(sum(obs.policy_target) - 1.0) > 1e-6:
            problems.append(f"policy_target does not sum to 1.0 (sum={sum(obs.policy_target)})")
        for a in range(min(7, len(obs.legal_mask))):
            if not obs.legal_mask[a] and obs.policy_target[a] != 0.0:
                problems.append(f"policy_target has mass {obs.policy_target[a]} on illegal action {a}")

    if len(obs.wdl_target) != 3:
        problems.append("wdl_target must have 3 entries")
    else:
        if any(math.isnan(p) or math.isinf(p) for p in obs.wdl_target):
            problems.append("wdl_target contains NaN/Inf")
        if abs(sum(obs.wdl_target) - 1.0) > 1e-6:
            problems.append(f"wdl_target does not sum to 1.0 (sum={sum(obs.wdl_target)})")

    # obs.state est canonicalise du point de vue du joueur au trait
    # (cf. schema.canonicalize_board) : on peut toujours le relire comme si
    # PLAYER_ONE etait au trait, pas besoin d'un champ "turn" separe.
    try:
        game = SongoLegacyGame.from_board(obs.state, PLAYER_ONE)
    except Exception:
        game = None
    if game is not None and game.finished:
        problems.append("state is already a terminal position: must not appear in dataset")

    return problems


@dataclass
class DistributionReport:
    count: int
    phase_counts: Dict[str, int]
    legal_move_count_histogram: Dict[int, int]
    mean_teacher_depth: float
    mean_teacher_nodes: float
    exact_fraction: float


def build_distribution_report(observations: List[Observation], phases: List[str]) -> DistributionReport:
    phase_counts = Counter(phases)
    legal_counts = Counter(sum(obs.legal_mask) for obs in observations)
    n = max(1, len(observations))
    return DistributionReport(
        count=len(observations),
        phase_counts=dict(phase_counts),
        legal_move_count_histogram=dict(legal_counts),
        mean_teacher_depth=sum(obs.teacher.depth for obs in observations) / n,
        mean_teacher_nodes=sum(obs.teacher.nodes for obs in observations) / n,
        exact_fraction=sum(1 for obs in observations if obs.teacher.is_exact) / n,
    )
