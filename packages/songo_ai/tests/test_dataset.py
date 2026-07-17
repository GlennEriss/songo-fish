"""Tests du schema d'observation, de la validation et du pipeline de build
(etape 5 : porte de passage = schema et controles qualite valides)."""

from __future__ import annotations

import json
from pathlib import Path

from songo_ai.dataset import (
    Observation,
    annotation_to_observation,
    build_dataset,
    canonicalize_board,
    validate_observation,
)
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO, SongoLegacyGame
from songo_ai.teachers import STANDARD, DeepTeacher, TeacherConfig


def test_canonicalize_board_is_identity_for_player_one() -> None:
    board = tuple(range(16))
    assert canonicalize_board(board, PLAYER_ONE) == board


def test_canonicalize_board_swaps_sides_for_player_two() -> None:
    board = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100, 200)
    canonical = canonicalize_board(board, PLAYER_TWO)
    assert canonical[0:7] == (7, 8, 9, 10, 11, 12, 13)
    assert canonical[7:14] == (0, 1, 2, 3, 4, 5, 6)
    assert canonical[14] == 200
    assert canonical[15] == 100


def _annotate(game: SongoLegacyGame, tier: str = STANDARD):
    config = TeacherConfig(initial_depth=2, depth_step=2, max_depth=6, max_nodes=20_000, max_time_s=2.0, tier=tier)
    return DeepTeacher(config).annotate(game)


def test_annotation_to_observation_passes_validation_standard_tier() -> None:
    game = SongoLegacyGame()
    annotation = _annotate(game, STANDARD)
    obs = annotation_to_observation(annotation, "traj-1", 0)
    problems = validate_observation(obs)
    assert problems == []


def test_annotation_to_observation_passes_validation_for_player_two() -> None:
    board = [5] * 14 + [0, 0]
    game = SongoLegacyGame.from_board(board, PLAYER_TWO)
    annotation = _annotate(game, STANDARD)
    obs = annotation_to_observation(annotation, "traj-2", 3)
    problems = validate_observation(obs)
    assert problems == []
    assert sum(obs.state) == 70


def test_validate_observation_flags_bad_policy() -> None:
    game = SongoLegacyGame()
    annotation = _annotate(game, STANDARD)
    obs = annotation_to_observation(annotation, "traj-3", 0)
    obs.policy_target[0] = 5.0  # casse la normalisation
    problems = validate_observation(obs)
    assert any("policy_target" in p for p in problems)


def test_build_dataset_end_to_end_small_scale(tmp_path: Path) -> None:
    config = TeacherConfig(
        initial_depth=2, depth_step=2, max_depth=6, max_nodes=20_000, max_time_s=2.0, stability_window=2, min_margin=10.0, tier=STANDARD
    )
    manifest = build_dataset(num_positions=24, out_dir=tmp_path, seed=7, teacher_config=config, trajectory_multiplier=4)

    assert manifest["total_positions"] > 0
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "train.jsonl").exists()

    all_trajectory_ids = {"train": set(), "val": set(), "test": set()}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.jsonl"
        for line in path.read_text().splitlines():
            obs_dict = json.loads(line)
            all_trajectory_ids[split].add(obs_dict["trajectory_id"])

    # Aucune partie ne doit apparaitre dans deux splits differents.
    assert all_trajectory_ids["train"] & all_trajectory_ids["val"] == set()
    assert all_trajectory_ids["train"] & all_trajectory_ids["test"] == set()
    assert all_trajectory_ids["val"] & all_trajectory_ids["test"] == set()
