"""Primitives d'audit pour les corpus historiques annotés par Teacher."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from songo_ai.songo.rules import SongoLegacyGame


@dataclass(frozen=True)
class TeacherAuditRecord:
    board: tuple[int, ...]
    player_to_move: int
    legal_mask: tuple[bool, ...]
    annotation: Mapping
    source_id: str
    ply: int | None = None
    physical_player_recoverable: bool = True

    @property
    def position_key(self) -> tuple[tuple[int, ...], int]:
        return self.board, self.player_to_move


def discover_teacher_corpora(data_root: Path) -> dict[str, dict]:
    """Découvre les corpus par manifeste/schéma, avec exclusions explicites."""

    excluded = {"d_rl", "real_matches", "experiments", "checkpoints"}
    corpora = {}
    for manifest in sorted(data_root.rglob("manifest.json")):
        relative = manifest.relative_to(data_root)
        if relative.parts[0] in excluded or "annotation_cache" in relative.parts:
            continue
        root = manifest.parent
        shards = sorted(path for path in root.iterdir() if path.suffix == ".jsonl")
        if shards:
            corpora[str(root.relative_to(data_root))] = {
                "root": root, "manifest": manifest, "format": "jsonl", "files": shards,
            }
    external = data_root / "dataset_full_matrix_merged_all_colabs"
    metadata = external / "dataset_metadata.json"
    npz = sorted(external.glob("*.npz"))
    if metadata.exists() and npz:
        corpora[str(external.relative_to(data_root))] = {
            "root": external, "manifest": metadata, "format": "npz", "files": npz,
        }
    return corpora


def iter_internal_teacher_records(path: Path, corpus_name: str) -> Iterator[TeacherAuditRecord]:
    """Lit le JSONL canonicalisé historique sans inventer le joueur perdu."""

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            board = tuple(row["state"])
            # Le schéma historique impose le joueur courant dans le camp P1 et
            # ne stocke pas son identité physique originale.
            yield TeacherAuditRecord(
                board=board,
                player_to_move=1,
                legal_mask=tuple(row["legal_mask"]),
                annotation={
                    "best_action": row.get("best_action"),
                    "action_values": row.get("action_values"),
                    "action_value_depths": row.get("action_value_depths"),
                    "policy_target": row.get("policy_target"),
                    "principal_variation": row.get("principal_variation"),
                    "teacher": row.get("teacher"),
                },
                source_id=f"{corpus_name}:{path.name}:{line_number}",
                ply=row.get("move_number"),
                physical_player_recoverable=False,
            )


def teacher_record_integrity(record: TeacherAuditRecord) -> list[str]:
    errors = []
    if len(record.board) != 16:
        return ["board_dimension"]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in record.board):
        errors.append("board_non_integer")
    elif any(value < 0 for value in record.board):
        errors.append("board_negative")
    elif sum(record.board) != 70:
        errors.append("seed_conservation")
    if record.player_to_move not in (1, 2):
        errors.append("player_to_move")
    if len(record.legal_mask) != 7:
        errors.append("legal_mask_dimension")
        return errors
    if not errors:
        game = SongoLegacyGame.from_board(record.board, record.player_to_move)
        if game.legal_mask() != record.legal_mask:
            errors.append("legal_mask_engine_mismatch")
    action = record.annotation.get("best_action")
    if action is not None and (
        not isinstance(action, int) or not 0 <= action < 7 or not record.legal_mask[action]
    ):
        errors.append("best_action_illegal")
    return errors


def annotation_fingerprint(annotation: Mapping) -> str:
    payload = json.dumps(annotation, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def position_fingerprint(position_key: tuple[tuple[int, ...], int]) -> str:
    """Empreinte stable de ``(board[16], player_to_move)``, sans symétrie."""

    board, player_to_move = position_key
    payload = ",".join(map(str, board)) + f"|{player_to_move}"
    return hashlib.sha256(payload.encode()).hexdigest()


def overlap_summary(left: set[tuple], right: set[tuple]) -> dict[str, float | int]:
    """Mesure un chevauchement exact sans modifier ni canonicaliser les états."""

    intersection = len(left & right)
    return {
        "intersection": intersection,
        "percent_left_in_right": 100 * intersection / len(left) if left else 0.0,
        "percent_right_in_left": 100 * intersection / len(right) if right else 0.0,
    }


def annotation_conflict_summary(records: list[TeacherAuditRecord]) -> dict:
    by_position: dict[tuple, list[TeacherAuditRecord]] = {}
    for record in records:
        by_position.setdefault(record.position_key, []).append(record)
    repeated = [values for values in by_position.values() if len(values) > 1]
    best_action_conflicts = 0
    annotation_conflicts = 0
    associated_depth, associated_tier, incomplete_values = 0, 0, 0
    for values in repeated:
        actions = {value.annotation.get("best_action") for value in values}
        fingerprints = {annotation_fingerprint(value.annotation) for value in values}
        if len(fingerprints) > 1:
            annotation_conflicts += 1
        if len(actions) > 1:
            best_action_conflicts += 1
            teachers = [value.annotation.get("teacher") or {} for value in values]
            if len({teacher.get("depth") for teacher in teachers}) > 1:
                associated_depth += 1
            if len({teacher.get("tier") for teacher in teachers}) > 1:
                associated_tier += 1
            if any(
                values_row is None or any(item is None for item in values_row)
                for values_row in (value.annotation.get("action_values") for value in values)
            ):
                incomplete_values += 1
    return {
        "positions_with_multiple_annotations": len(repeated),
        "positions_with_annotation_conflict": annotation_conflicts,
        "positions_with_best_action_conflict": best_action_conflicts,
        "best_action_conflicts_with_depth_difference": associated_depth,
        "best_action_conflicts_with_tier_difference": associated_tier,
        "best_action_conflicts_with_incomplete_action_values": incomplete_values,
    }
