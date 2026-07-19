"""Tests de la generation de trajectoires et de l'echantillonnage (etape 5)."""

from __future__ import annotations

import random

from songo_ai.generation import (
    generate_trajectories,
    generate_trajectory,
    make_mixed_agent,
    random_agent,
    sample_positions,
)
from songo_ai.generation.sampling import ENDGAME, MIDGAME, OPENING, TIGHT_ENDGAME, classify_phase
from songo_ai.songo.rules import SongoLegacyGame, assert_invariants


def test_generate_trajectory_never_records_a_terminal_position() -> None:
    rng = random.Random(0)
    positions = generate_trajectory(random_agent, rng, max_moves=400)
    assert len(positions) > 0
    for position in positions:
        game = SongoLegacyGame.from_board(position.state.board, position.state.turn)
        assert not game.finished
        assert_invariants(position.state.board)


def test_generate_trajectory_move_numbers_are_sequential() -> None:
    rng = random.Random(1)
    positions = generate_trajectory(random_agent, rng)
    move_numbers = [p.move_number for p in positions]
    assert move_numbers == list(range(len(positions)))
    assert all(p.total_moves == len(positions) for p in positions)


def test_generate_trajectories_uses_distinct_trajectory_ids() -> None:
    positions = generate_trajectories(lambda rng: make_mixed_agent(rng), num_trajectories=5, seed=0)
    ids = {p.trajectory_id for p in positions}
    assert len(ids) == 5


def test_classify_phase_thresholds() -> None:
    from songo_ai.generation.trajectories import TrajectoryPosition
    from songo_ai.songo.rules import State

    state = State.initial()
    assert classify_phase(TrajectoryPosition(state, "g", 0, 30)) == OPENING
    assert classify_phase(TrajectoryPosition(state, "g", 15, 30)) == MIDGAME
    assert classify_phase(TrajectoryPosition(state, "g", 25, 30)) == ENDGAME


def test_classify_phase_prioritizes_low_seed_count_over_move_ratio() -> None:
    # Une position en debut de partie (par ratio de coups) mais deja
    # assechee (peu de graines en jeu, ex. captures agressives) doit etre
    # classee "fin_serree", pas "ouverture" -- cf. docs/III. Gestion des
    # fins/, positions du type "6 contre 4"/"5 contre 4" (peu importe le
    # numero du coup, ce qui compte est le solde de graines restant).
    from songo_ai.generation.trajectories import TrajectoryPosition
    from songo_ai.songo.rules import State

    board = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 40, 21]  # 9 en jeu, 61 deja captures
    state = State(tuple(board), 2)
    assert classify_phase(TrajectoryPosition(state, "g", 3, 30)) == TIGHT_ENDGAME


def test_classify_phase_ignores_stored_seeds_only_counts_in_play() -> None:
    from songo_ai.generation.trajectories import TrajectoryPosition
    from songo_ai.songo.rules import State

    state = State.initial()  # 70 graines en jeu, aucune capturee
    assert classify_phase(TrajectoryPosition(state, "g", 0, 30)) == OPENING


def test_sample_positions_respects_target_count_and_dedup() -> None:
    positions = generate_trajectories(lambda rng: random_agent, num_trajectories=20, seed=2)
    sampled = sample_positions(positions, target_count=50, seed=2, max_repeats=2)
    assert len(sampled) <= 50
    assert len(sampled) > 0

    from songo_ai.songo.rules import zobrist_hash

    counts: dict[int, int] = {}
    for p in sampled:
        key = zobrist_hash(p.state)
        counts[key] = counts.get(key, 0) + 1
    assert all(c <= 2 for c in counts.values())


def test_sample_positions_favors_midgame_and_endgame_over_opening() -> None:
    positions = generate_trajectories(lambda rng: random_agent, num_trajectories=30, seed=3)
    sampled = sample_positions(positions, target_count=200, seed=3)
    from songo_ai.generation.sampling import classify_phase as cp

    phase_counts = {OPENING: 0, MIDGAME: 0, ENDGAME: 0, TIGHT_ENDGAME: 0}
    for p in sampled:
        phase_counts[cp(p)] += 1
    assert phase_counts[MIDGAME] + phase_counts[ENDGAME] + phase_counts[TIGHT_ENDGAME] >= phase_counts[OPENING]
