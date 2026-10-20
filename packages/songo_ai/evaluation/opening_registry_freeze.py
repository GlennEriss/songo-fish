"""Deterministic construction and immutable freeze of an approved opening registry.

This module is deliberately static: it has no model, MCTS, checkpoint, or game
execution dependency.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from songo_ai.evaluation.opening_registry import (
    OpeningRecord,
    canonical,
    physical_state_fingerprint,
    physical_state_payload,
    raw_state_from_record,
    sha256,
    sha256_rank_v1,
    verify_source_bytes,
)

AMENDMENT_SHA256 = "2e4dae3b58beeee4f15b17c72cf2c000729f3ae49e44147cc0a2601b29788662"
SOURCE_SHA256 = "49eb0f83265a763d06f7b45032d994227208f4149ad0c9fb67b6209fd8289667"
ORIGINAL_SOURCE_SHA256 = "17d9ad19921cdbf1591eb54f6c40aa5b710376f6e903491dd39affc605a5ed18"
SELECTION_SEED = 3538971373371044805
REQUIRED_OPENINGS = 400
SOURCE_ID = "LOT45_D_REAL_HUMAN_SUBSET"
REGISTRY_ID = "LOT46B_C0_SHARED_OPENINGS_400_V1"


def legacy_state_fingerprint(state: Mapping[str, Any]) -> str:
    payload = {"board": state["board"], "player_to_move": state["player_to_move"]}
    return sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def load_jsonl_gzip(source: Path, expected_sha256: str = SOURCE_SHA256) -> list[dict[str, Any]]:
    compressed = source.read_bytes()
    verify_source_bytes(compressed, expected_sha256)
    return [json.loads(line) for line in gzip.decompress(compressed).splitlines() if line.strip()]


def state_fingerprints_from_jsonl(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if isinstance(row.get("state"), Mapping):
                result.add(legacy_state_fingerprint(row["state"]))
    return result


def reproduce_candidates(source_rows: Sequence[Mapping[str, Any]], d_rl_fingerprints: set[str]) -> dict[str, Any]:
    human = [row for row in source_rows
             if row.get("family") == "D_REAL_HUMAN"
             and row.get("primary_source") == "D_REAL_HUMAN_MATCHES"]
    if len(human) != 491:
        raise ValueError(f"source record mismatch: {len(human)}")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in human:
        state = raw_state_from_record(row)
        source_record_id = str(row.get("fingerprint", ""))
        if legacy_state_fingerprint(row["state"]) != source_record_id:
            raise ValueError("source physical fingerprint mismatch")
        state_fp = physical_state_fingerprint(state)
        if state_fp in seen:
            raise ValueError("duplicate physical state in source")
        seen.add(state_fp)
        normalized.append({"row": row, "state": state, "state_fingerprint": state_fp,
                           "source_record_id": source_record_id})
    lot45_prefix = {str(row["fingerprint"]) for row in source_rows if int(row["order"]) < 10000}
    excluded_training = {item["source_record_id"] for item in normalized
                         if item["source_record_id"] in d_rl_fingerprints | lot45_prefix}
    eligible = [item for item in normalized if item["source_record_id"] not in excluded_training]
    if len(excluded_training) != 51 or len(eligible) != 440:
        raise ValueError("approved candidate inventory mismatch")
    return {"total_source_records": len(normalized), "unique_physical_states": len(seen),
            "training_exclusions": sorted(excluded_training), "eligible": eligible,
            "lot45_prefix_overlap": sum(item["source_record_id"] in lot45_prefix for item in normalized),
            "d_rl_overlap": sum(item["source_record_id"] in d_rl_fingerprints for item in normalized)}


def select_openings(eligible: Sequence[Mapping[str, Any]], *, seed: int = SELECTION_SEED,
                    count: int = REQUIRED_OPENINGS) -> list[dict[str, Any]]:
    if seed != SELECTION_SEED or count != REQUIRED_OPENINGS:
        raise ValueError("selection parameters differ from approved amendment")
    ranked = []
    for item in eligible:
        rank = sha256_rank_v1(seed, str(item["state_fingerprint"]), str(item["source_record_id"]))
        ranked.append((rank, str(item["state_fingerprint"]),
                       str(item["source_record_id"]).encode("utf-8"), item))
    ranked.sort(key=lambda value: (value[0], value[1], value[2]))
    if len(ranked) < count:
        raise ValueError("insufficient eligible openings")
    selected = []
    for index, (rank, _, _, item) in enumerate(ranked[:count], 1):
        record = OpeningRecord(
            opening_id=f"LOT46B-C0-OPENING-{index:04d}", state=item["state"],
            state_fingerprint=str(item["state_fingerprint"]), source_id=SOURCE_ID,
            source_sha256=SOURCE_SHA256, split="APPROVED_EVALUATION_OPENING",
            source_record_id=str(item["source_record_id"]),
        )
        record.validate()
        selected.append({**record.scientific(), "selection_rank": index,
                         "selection_digest": rank})
    if len({item["state_fingerprint"] for item in selected}) != count:
        raise ValueError("duplicate selected physical state")
    return selected


def candidate_inventory_payload(reproduced: Mapping[str, Any]) -> dict[str, Any]:
    candidates = sorted(({"state_fingerprint": str(item["state_fingerprint"]),
                          "source_record_id": str(item["source_record_id"])}
                         for item in reproduced["eligible"]),
                        key=lambda item: (item["state_fingerprint"], item["source_record_id"].encode()))
    return {"schema": "songo-opening-candidate-inventory-v1", "source_id": SOURCE_ID,
            "source_sha256": SOURCE_SHA256, "training_exclusions": 51,
            "b3_exact_overlaps": 0, "eligible_candidates": 440, "candidates": candidates}


def registry_payload(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(selected) != REQUIRED_OPENINGS:
        raise ValueError("registry must contain exactly 400 openings")
    return {"schema": "songo-opening-registry-v1", "registry_id": REGISTRY_ID,
            "registry_version": 1, "amendment_sha256": AMENDMENT_SHA256,
            "source_id": SOURCE_ID, "source_sha256": SOURCE_SHA256,
            "selection_algorithm": "SHA256_RANK_V1", "selection_seed": SELECTION_SEED,
            "shared_c1_registry": True, "records": list(selected)}


def immutable_atomic_write(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace divergent frozen artifact: {path}")
        return digest
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest


def serialize(value: Any) -> bytes:
    return canonical(value) + b"\n"


def engine_ruleset_fingerprint(paths: Iterable[Path]) -> str:
    parts = [{"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(paths, key=lambda item: item.as_posix())]
    return hashlib.sha256(canonical(parts)).hexdigest()
