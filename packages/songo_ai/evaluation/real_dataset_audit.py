"""Audit factuel et streaming des exports de vrais matchs Songo."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from songo_ai.songo.rules import SongoLegacyGame, local_action_to_pit


REAL_SCHEMA = "songo-real-move-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_real_records(path: Path) -> Iterator[dict]:
    """Lit les deux sérialisations supportées sans altérer les données."""

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number}: record must be an object")
                    yield value
        return
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{path}: JSON root must be an array")
        for record in value:
            if not isinstance(record, dict):
                raise ValueError(f"{path}: record must be an object")
            yield record
        return
    raise ValueError(f"unsupported real dataset format: {path.suffix}")


def discover_real_dataset_files(data_root: Path) -> list[Path]:
    """Découvre seulement les exports sous ``real_matches``.

    Cette frontière exclut volontairement ``d_rl``, les arènes, checkpoints et
    corpus historiques générés/annotés par teacher.
    """

    root = data_root / "real_matches"
    if not root.exists():
        return []
    candidates = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            try:
                first = next(iter_real_records(path))
            except StopIteration:
                continue
            if first.get("schema_version") == REAL_SCHEMA:
                candidates.append(path)
    return candidates


def position_key(record: Mapping) -> tuple:
    """Identité physique, sans canonicalisation : board[16] + joueur."""

    board = tuple(record["board_before"])
    player = int(record["player_position"])
    return board, player


def position_digest(record: Mapping) -> str:
    board, player = position_key(record)
    payload = bytes(board) + bytes((player,))
    return hashlib.sha256(payload).hexdigest()


def record_integrity(record: Mapping) -> list[str]:
    errors = []
    board = record.get("board_before")
    player = record.get("player_position")
    if not isinstance(board, list) or len(board) != 16:
        return ["board_dimension"]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in board):
        errors.append("board_non_integer")
    elif any(value < 0 for value in board):
        errors.append("board_negative")
    elif sum(board) != 70:
        errors.append("seed_conservation")
    if player not in (1, 2):
        errors.append("player_to_move")
    mask = record.get("legal_mask")
    action = record.get("action_local")
    if not isinstance(mask, list) or len(mask) != 7:
        errors.append("legal_mask_dimension")
    elif not isinstance(action, int) or not 0 <= action < 7 or not bool(mask[action]):
        errors.append("stored_action_illegal")
    if not errors:
        game = SongoLegacyGame.from_board(board, player)
        pit = local_action_to_pit(player, action)
        if not game.is_legal_move(pit):
            errors.append("engine_action_illegal")
        if tuple(bool(value) for value in mask) != game.legal_mask():
            errors.append("legal_mask_engine_mismatch")
    return errors


def audit_real_file(path: Path, *, relative_to: Path | None = None) -> tuple[dict, set, set, set]:
    positions, position_actions, record_ids = set(), set(), set()
    fields, matches, errors = set(), set(), Counter()
    rows = 0
    player_counts, action_counts, legal_counts, winner_counts = Counter(), Counter(), Counter(), Counter()
    plies, stores, seed_counts = [], [], []
    for record in iter_real_records(path):
        rows += 1
        fields.update(record)
        key = position_key(record)
        positions.add(key)
        position_actions.add((key, record.get("action_local")))
        record_ids.add((key, record.get("action_local"), record.get("match_id"), record.get("ply")))
        matches.add(record.get("match_id"))
        errors.update(record_integrity(record))
        player_counts[str(record.get("player_position"))] += 1
        action_counts[str(record.get("action_local"))] += 1
        legal_counts[str(sum(bool(x) for x in record.get("legal_mask", [])))] += 1
        winner_counts[str(record.get("winner_after"))] += 1
        if isinstance(record.get("ply"), int):
            plies.append(record["ply"])
        board = record.get("board_before", [])
        if len(board) == 16:
            stores.extend(board[14:16])
            seed_counts.append(sum(board))
    relative = str(path.relative_to(relative_to)) if relative_to else str(path)
    inventory = {
        "name": path.name,
        "path": relative,
        "format": path.suffix.lstrip("."),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "records": rows,
        "positions": rows,
        "unique_physical_positions": len(positions),
        "duplicate_positions": rows - len(positions),
        "duplication_rate": (rows - len(positions)) / rows if rows else 0.0,
        "unique_position_action": len(position_actions),
        "same_position_different_actions": len(position_actions) - len(positions),
        "matches": len(matches),
        "fields": sorted(fields),
        "integrity_errors": dict(sorted(errors.items())),
        "distributions": {
            "player_to_move": dict(sorted(player_counts.items())),
            "action_local": dict(sorted(action_counts.items())),
            "legal_count": dict(sorted(legal_counts.items())),
            "winner_after": dict(sorted(winner_counts.items())),
            "ply": {"minimum": min(plies), "maximum": max(plies)} if plies else None,
            "stores": {"minimum": min(stores), "maximum": max(stores)} if stores else None,
            "seed_count": dict(Counter(seed_counts)),
        },
    }
    return inventory, positions, position_actions, record_ids


def overlap_report(named_sets: Mapping[str, set]) -> dict:
    matrix = {}
    for left_name, left in named_sets.items():
        matrix[left_name] = {}
        for right_name, right in named_sets.items():
            intersection = len(left & right)
            matrix[left_name][right_name] = {
                "intersection": intersection,
                "percent_of_left": 100.0 * intersection / len(left) if left else 0.0,
                "percent_of_right": 100.0 * intersection / len(right) if right else 0.0,
            }
    return matrix
