#!/usr/bin/env python3
"""Build the approved Lot46B shared opening registry without executing games."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from songo_ai.evaluation.opening_registry_freeze import *


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approval-record", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(); output = args.output.resolve()
    amendment = repo / "docs/lot46b_stage_c0_4_p5/opening_protocol_amendment_v1.json"
    if hashlib.sha256(amendment.read_bytes()).hexdigest() != AMENDMENT_SHA256:
        raise SystemExit("APPROVAL_INTEGRITY = FAIL")
    original = repo / "data/real_matches/match_moves_v1.jsonl"
    if hashlib.sha256(original.read_bytes()).hexdigest() != ORIGINAL_SOURCE_SHA256:
        raise SystemExit("ORIGINAL_SOURCE_INTEGRITY = FAIL")
    rows = load_jsonl_gzip(repo / "data/colab_bridge/lot45_selection.jsonl.gz")
    d_rl = state_fingerprints_from_jsonl(repo / "data/d_rl/lot11_g1_selected_seed_20260924.jsonl")
    reproduced = reproduce_candidates(rows, d_rl)
    candidate_payload = candidate_inventory_payload(reproduced)
    selected = select_openings(reproduced["eligible"])
    registry = registry_payload(selected)
    candidate_bytes = serialize(candidate_payload); registry_bytes = serialize(registry)
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    registry_sha = hashlib.sha256(registry_bytes).hexdigest()
    approval_sha = hashlib.sha256(args.approval_record.read_bytes()).hexdigest()
    engine_sha = engine_ruleset_fingerprint([
        repo / "packages/songo_ai/songo/rules.py",
        repo / "packages/songo_ai/dataset/selfplay_schema.py",
    ])
    manifest = {
        "REGISTRY_ID": REGISTRY_ID, "REGISTRY_VERSION": 1,
        "AMENDMENT_ID": "LOT46B_STAGE_C0_OPENING_PROTOCOL_AMENDMENT_V1",
        "AMENDMENT_SHA256": AMENDMENT_SHA256, "APPROVAL_RECORD_SHA256": approval_sha,
        "SOURCE_ID": SOURCE_ID, "SOURCE_SHA256": SOURCE_SHA256,
        "ORIGINAL_SOURCE_SHA256": ORIGINAL_SOURCE_SHA256,
        "SELECTION_ALGORITHM": "SHA256_RANK_V1", "SELECTION_SEED": SELECTION_SEED,
        "ELIGIBLE_CANDIDATE_COUNT": 440, "SELECTED_OPENING_COUNT": 400,
        "EXCLUDED_TRAINING_OVERLAPS": 51, "EXCLUDED_B3_OVERLAPS": 0,
        "CANONICAL_STATE_SCHEMA_VERSION": "songo-opening-registry-v1",
        "FINGERPRINT_CONTRACT_VERSION": "SHA256_CANONICAL_JSON_V1",
        "ENGINE_RULESET_FINGERPRINT": engine_sha,
        "OPENING_IDS": [item["opening_id"] for item in selected],
        "OPENING_STATE_FINGERPRINTS": [item["state_fingerprint"] for item in selected],
        "CANDIDATE_INVENTORY_SHA256": candidate_sha, "REGISTRY_CONTENT_SHA256": registry_sha,
        "CREATION_TOOL_VERSION": "LOT46B_C0_4_P6_V1",
        "GIT_COMMIT_REFERENCE": "15edd84d9a73d1324b8a673848ae63f314a738c7",
    }
    audit = {
        "algorithm": "SHA256_RANK_V1", "seed": SELECTION_SEED,
        "total_source_records": reproduced["total_source_records"],
        "unique_physical_states": reproduced["unique_physical_states"],
        "lot45_prefix_overlap": reproduced["lot45_prefix_overlap"],
        "d_rl_overlap": reproduced["d_rl_overlap"],
        "training_overlap_union": len(reproduced["training_exclusions"]),
        "eligible_candidates": len(reproduced["eligible"]), "selected_openings": len(selected),
        "candidate_outputs_used": False, "rng_used": False,
    }
    immutable_atomic_write(output / "opening_candidate_inventory.json", candidate_bytes)
    immutable_atomic_write(output / "opening_registry_400.json", registry_bytes)
    immutable_atomic_write(output / "opening_registry_manifest.json", serialize(manifest))
    immutable_atomic_write(output / "opening_selection_audit.json", serialize(audit))
    immutable_atomic_write(output / "opening_registry_sha256.txt", f"{registry_sha}  opening_registry_400.json\n".encode())
    print(json.dumps({"status": "PASS", "registry_sha256": registry_sha,
                      "candidate_inventory_sha256": candidate_sha, "selected": len(selected)}))


if __name__ == "__main__":
    main()
