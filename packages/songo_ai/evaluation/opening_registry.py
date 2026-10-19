"""Prospective opening-registry contracts; contains no game or MCTS execution."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.songo.rules import SongoLegacyGame

SCHEMA = "songo-opening-registry-v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def physical_state_payload(state: RawSongoState) -> dict[str, Any]:
    return {"schema": SCHEMA, "board": list(state.board),
            "player_to_move": state.player_to_move}


def physical_state_fingerprint(state: RawSongoState) -> str:
    return sha256(canonical(physical_state_payload(state)))


@dataclass(frozen=True)
class OpeningRecord:
    opening_id: str
    state: RawSongoState
    state_fingerprint: str
    source_id: str
    source_sha256: str
    split: str
    source_record_id: str

    def validate(self) -> None:
        if not all((self.opening_id, self.source_id, self.split, self.source_record_id)):
            raise ValueError("missing opening provenance")
        if len(self.source_sha256) != 64:
            raise ValueError("invalid source fingerprint")
        if self.state_fingerprint != physical_state_fingerprint(self.state):
            raise ValueError("physical state fingerprint mismatch")
        game = SongoLegacyGame.from_state(self.state.to_engine_state())
        game.normalize_terminal()
        if game.finished or not game.legal_local_actions():
            raise ValueError("opening must be non-terminal with a legal action")

    def scientific(self) -> dict[str, Any]:
        self.validate()
        return {"opening_id": self.opening_id, "state": physical_state_payload(self.state),
                "state_fingerprint": self.state_fingerprint, "source_id": self.source_id,
                "source_sha256": self.source_sha256, "split": self.split,
                "source_record_id": self.source_record_id}


def opening_record_fingerprint(record: OpeningRecord) -> str:
    return sha256(canonical(record.scientific()))


def registry_payload(records: Sequence[OpeningRecord], provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not records:
        raise ValueError("registry must not be empty")
    if provenance.get("status") not in ("HISTORICAL", "APPROVED_PROSPECTIVE"):
        raise ValueError("missing approved registry provenance")
    ids = [r.opening_id for r in records]
    states = [r.state_fingerprint for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate opening id")
    if len(states) != len(set(states)):
        raise ValueError("duplicate physical state")
    return {"schema": SCHEMA, "provenance": dict(provenance),
            "records": [r.scientific() for r in records]}


def registry_fingerprint(records: Sequence[OpeningRecord], provenance: Mapping[str, Any]) -> str:
    return sha256(canonical(registry_payload(records, provenance)))


def overlap_audit(records: Iterable[OpeningRecord], *, training: set[str],
                  validation: set[str]) -> dict[str, Any]:
    fingerprints = [r.state_fingerprint for r in records]
    return {"exact_duplicates": len(fingerprints) - len(set(fingerprints)),
            "training_overlap": sorted(set(fingerprints) & training),
            "validation_overlap": sorted(set(fingerprints) & validation),
            "known_equivalent_duplicates": "NOT_EVALUATED",
            "unknown_equivalence_risk": "PLAYER_SIDE_CANONICALIZATION_NOT_PROVEN"}


def prospective_selection_spec(*, source_sha256: str, seed: int, count: int = 400) -> dict[str, Any]:
    if len(source_sha256) != 64 or seed < 0 or count != 400:
        raise ValueError("invalid frozen selection specification")
    return {"algorithm": "SHA256_RANK_V1", "source_sha256": source_sha256,
            "selection_seed": seed, "required_openings": count,
            "rank_key": "sha256(seed || physical_state_fingerprint || source_record_id)",
            "candidate_outputs_used": False, "status": "PROPOSED"}
