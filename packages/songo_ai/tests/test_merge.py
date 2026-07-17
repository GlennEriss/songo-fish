"""Tests de la fusion de releases (section 6.6)."""

from __future__ import annotations

import json
from pathlib import Path

from songo_ai.dataset import build_dataset, merge_releases
from songo_ai.teachers import STANDARD, TeacherConfig


def _tiny_config() -> TeacherConfig:
    return TeacherConfig(initial_depth=2, depth_step=2, max_depth=6, max_nodes=20_000, max_time_s=2.0, stability_window=2, min_margin=10.0, tier=STANDARD)


def test_merge_releases_concatenates_and_preserves_counts(tmp_path: Path) -> None:
    release_a = tmp_path / "a"
    release_b = tmp_path / "b"
    build_dataset(num_positions=12, out_dir=release_a, seed=1, teacher_config=_tiny_config())
    build_dataset(num_positions=12, out_dir=release_b, seed=2, teacher_config=_tiny_config())

    merged_dir = tmp_path / "merged"
    manifest = merge_releases([release_a, release_b], merged_dir)

    a_counts = json.loads((release_a / "manifest.json").read_text())["counts_per_split"]
    b_counts = json.loads((release_b / "manifest.json").read_text())["counts_per_split"]

    for split in ("train", "val", "test"):
        expected = a_counts[split] + b_counts[split]
        assert manifest["counts_per_split"][split] == expected
        lines = (merged_dir / f"{split}.jsonl").read_text().splitlines()
        assert len(lines) == expected

    assert manifest["total_positions"] == 24
    assert len(manifest["merged_from"]) == 2


def test_merge_releases_keeps_trajectory_ids_within_their_original_split(tmp_path: Path) -> None:
    release_a = tmp_path / "a"
    build_dataset(num_positions=12, out_dir=release_a, seed=3, teacher_config=_tiny_config())
    merged_dir = tmp_path / "merged"
    merge_releases([release_a], merged_dir)

    def ids_in(path: Path) -> set:
        ids = set()
        for line in path.read_text().splitlines():
            ids.add(json.loads(line)["trajectory_id"])
        return ids

    original_train_ids = ids_in(release_a / "train.jsonl")
    original_val_ids = ids_in(release_a / "val.jsonl")
    merged_train_ids = ids_in(merged_dir / "train.jsonl")
    merged_val_ids = ids_in(merged_dir / "val.jsonl")

    assert merged_train_ids == original_train_ids
    assert merged_val_ids == original_val_ids
    assert original_train_ids.isdisjoint(original_val_ids)
