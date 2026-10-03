"""Trajectoires, agents, sampling, deduplication (etape 5 du plan directeur)."""

from .agents import make_mixed_agent, make_shallow_search_agent, random_agent
from .sampling import classify_phase, sample_positions
from .trajectories import TrajectoryPosition, generate_trajectories, generate_trajectory
from .selfplay import (
    PendingSelfPlayStep,
    SelfPlayConfig,
    SelfPlayGameResult,
    SelfPlayRunResult,
    SelfPlayRunner,
    SelfPlayStatistics,
    SelfPlayStatus,
    finalize_selfplay_steps,
    select_action_from_policy,
)

__all__ = [
    "make_mixed_agent",
    "make_shallow_search_agent",
    "random_agent",
    "classify_phase",
    "sample_positions",
    "TrajectoryPosition",
    "generate_trajectories",
    "generate_trajectory",
    "PendingSelfPlayStep",
    "SelfPlayConfig",
    "SelfPlayGameResult",
    "SelfPlayRunResult",
    "SelfPlayRunner",
    "SelfPlayStatistics",
    "SelfPlayStatus",
    "finalize_selfplay_steps",
    "select_action_from_policy",
]
from .population import (
    MatchupQuota,
    ModelRole,
    RegisteredModel,
    REQUIRED_PROVENANCE_FIELDS,
    ScheduledGame,
    deterministic_schedule,
    source_balanced_indices,
    validate_game_provenance,
)
