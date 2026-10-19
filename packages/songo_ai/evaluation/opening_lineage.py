"""Fail-closed helpers for forensic opening-lineage reconstruction.

This module only joins historical metadata and validates static states.  It does
not construct an opening registry and contains no game, MCTS, or model code.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from songo_ai.evaluation.opening_registry import (
    physical_state_fingerprint,
    raw_state_from_record,
    verify_source_bytes,
)

EXACT_VERIFIED = "EXACT_VERIFIED"
PARTIAL_VERIFIED = "PARTIAL_VERIFIED"
AMBIGUOUS = "AMBIGUOUS"
UNRESOLVED = "UNRESOLVED"
REJECTED = "REJECTED"


@dataclass(frozen=True)
class LineageMapping:
    record_id: str
    confidence: str
    state_fingerprint: str | None = None
    source_matches: int = 0
    reason: str | None = None


def require_unique_occurrence_ids(records: Iterable[Mapping[str, Any]], field: str) -> None:
    values = [str(record[field]) for record in records if record.get(field) is not None]
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {field}: {duplicates[:3]}")


def join_historical_identifiers(
    records: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    *,
    record_field: str,
    source_field: str,
    record_namespace: str,
    source_namespace: str,
) -> list[LineageMapping]:
    """Join identifiers only when their historical namespaces are identical.

    Multiple source rows are exact only when they reconstruct the same physical
    state.  Conflicting states are ambiguous and absent/incomplete states remain
    unresolved; no state is guessed.
    """
    if not record_namespace or record_namespace != source_namespace:
        raise ValueError("identifier namespace mismatch")
    index: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for source in sources:
        if source.get(source_field) is not None:
            index[str(source[source_field])].append(source)
    result: list[LineageMapping] = []
    for record in records:
        if record.get(record_field) is None:
            result.append(LineageMapping("", REJECTED, reason="missing historical identifier"))
            continue
        identifier = str(record[record_field])
        matches = index.get(identifier, [])
        if not matches:
            result.append(LineageMapping(identifier, UNRESOLVED, reason="source record absent"))
            continue
        fingerprints: set[str] = set()
        incomplete = False
        for match in matches:
            try:
                fingerprints.add(physical_state_fingerprint(raw_state_from_record(match)))
            except (KeyError, TypeError, ValueError):
                incomplete = True
        if not fingerprints:
            result.append(LineageMapping(identifier, UNRESOLVED, source_matches=len(matches),
                                         reason="complete physical state unavailable"))
        elif len(fingerprints) > 1:
            result.append(LineageMapping(identifier, AMBIGUOUS, source_matches=len(matches),
                                         reason="one identifier maps to multiple physical states"))
        elif incomplete:
            result.append(LineageMapping(identifier, PARTIAL_VERIFIED,
                                         next(iter(fingerprints)), len(matches),
                                         "some source references have no complete state"))
        else:
            result.append(LineageMapping(identifier, EXACT_VERIFIED,
                                         next(iter(fingerprints)), len(matches)))
    return result


def lineage_summary(mappings: Sequence[LineageMapping]) -> dict[str, Any]:
    counts = Counter(mapping.confidence for mapping in mappings)
    exact = [mapping.state_fingerprint for mapping in mappings
             if mapping.confidence == EXACT_VERIFIED and mapping.state_fingerprint]
    return {
        "total_records": len(mappings),
        "exactly_mapped_records": counts[EXACT_VERIFIED],
        "partially_mapped_records": counts[PARTIAL_VERIFIED],
        "ambiguous_records": counts[AMBIGUOUS],
        "unresolved_records": counts[UNRESOLVED],
        "rejected_records": counts[REJECTED],
        "unique_physical_states_recovered": len(set(exact)),
        "physical_coverage_percent": 100.0 * counts[EXACT_VERIFIED] / len(mappings)
        if mappings else 0.0,
        "validation_leakage_status": "CLEAR" if mappings and counts[EXACT_VERIFIED] == len(mappings)
        else "PARTIALLY_VERIFIED" if counts[EXACT_VERIFIED] else "BLOCKED",
    }


def exact_overlap(mappings: Sequence[LineageMapping], candidates: set[str]) -> list[str]:
    return sorted({mapping.state_fingerprint for mapping in mappings
                   if mapping.confidence == EXACT_VERIFIED
                   and mapping.state_fingerprint in candidates})


def classify_provenance(*, derived_bytes: bytes, derived_sha256: str,
                        original_sha256: str | None, original_bytes: bytes | None) -> dict[str, Any]:
    verify_source_bytes(derived_bytes, derived_sha256)
    original_recovered = original_bytes is not None
    if original_recovered:
        if not original_sha256:
            raise ValueError("original bytes lack a historical SHA256")
        verify_source_bytes(original_bytes, original_sha256)
    return {
        "source_content_integrity": "VERIFIED",
        "derived_bytes_verified": True,
        "historical_hash_recorded": bool(original_sha256),
        "original_bytes_recovered": original_recovered,
        "status": "VERIFIED_DERIVATION_CHAIN" if original_recovered
        else "VERIFIED_DERIVATION_CHAIN_ORIGINAL_HASH_RECORDED_ORIGINAL_BYTES_UNAVAILABLE",
    }
