"""Trajectoires, agents, sampling, deduplication (etape 5 du plan directeur)."""

from .agents import make_mixed_agent, make_shallow_search_agent, random_agent
from .sampling import classify_phase, sample_positions
from .trajectories import TrajectoryPosition, generate_trajectories, generate_trajectory

__all__ = [
    "make_mixed_agent",
    "make_shallow_search_agent",
    "random_agent",
    "classify_phase",
    "sample_positions",
    "TrajectoryPosition",
    "generate_trajectories",
    "generate_trajectory",
]
