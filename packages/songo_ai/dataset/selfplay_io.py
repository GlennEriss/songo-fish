"""Serialisation JSONL versionnee des exemples produits dans ``D_RL``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .selfplay_schema import D_RL, RLTrainingExample, RawSongoState


D_RL_FORMAT = "songo_d_rl_jsonl"
D_RL_FORMAT_VERSION = 1


def _example_to_record(example: RLTrainingExample) -> dict:
    return {
        "record_type": "example",
        "state": {
            "board": list(example.state.board),
            "player_to_move": example.state.player_to_move,
        },
        "legal_mask": list(example.legal_mask),
        "visit_counts": list(example.visit_counts) if example.visit_counts is not None else None,
        "policy_target": list(example.policy_target) if example.policy_target is not None else None,
        "value_target": example.value_target,
        "metadata": dict(example.metadata),
    }


def write_d_rl_jsonl(path: str | Path, examples: Iterable[RLTrainingExample]) -> int:
    """Ecrit un shard local ``D_RL`` et retourne son nombre d'exemples."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "record_type": "manifest",
        "format": D_RL_FORMAT,
        "version": D_RL_FORMAT_VERSION,
        "dataset_family": D_RL.value,
    }
    count = 0
    with destination.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header, sort_keys=True) + "\n")
        for example in examples:
            if example.dataset_family is not D_RL:
                raise ValueError("only D_RL examples can be serialized by this writer")
            stream.write(json.dumps(_example_to_record(example), sort_keys=True) + "\n")
            count += 1
    return count


def read_d_rl_jsonl(path: str | Path) -> tuple[RLTrainingExample, ...]:
    """Charge un shard apres validation stricte de son manifeste versionne."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        lines = [line for line in stream if line.strip()]
    if not lines:
        raise ValueError("D_RL shard is empty")
    header = json.loads(lines[0])
    expected = {
        "record_type": "manifest",
        "format": D_RL_FORMAT,
        "version": D_RL_FORMAT_VERSION,
        "dataset_family": D_RL.value,
    }
    if header != expected:
        raise ValueError(f"unsupported or invalid D_RL manifest: {header!r}")

    examples = []
    for line_number, line in enumerate(lines[1:], start=2):
        record = json.loads(line)
        if record.get("record_type") != "example":
            raise ValueError(f"invalid D_RL record at line {line_number}")
        state_record = record["state"]
        examples.append(
            RLTrainingExample(
                state=RawSongoState(
                    tuple(state_record["board"]),
                    state_record["player_to_move"],
                ),
                legal_mask=tuple(record["legal_mask"]),
                visit_counts=(
                    tuple(record["visit_counts"])
                    if record["visit_counts"] is not None
                    else None
                ),
                policy_target=(
                    tuple(record["policy_target"])
                    if record["policy_target"] is not None
                    else None
                ),
                value_target=record["value_target"],
                metadata=record["metadata"],
            )
        )
    return tuple(examples)
