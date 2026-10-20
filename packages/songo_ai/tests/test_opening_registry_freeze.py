import json
from pathlib import Path

import pytest

from songo_ai.dataset.selfplay_schema import RawSongoState
from songo_ai.evaluation.arena_runner import Candidate, ExperimentManifest, schedule
from songo_ai.evaluation.opening_amendment import verify_amendment_bytes
from songo_ai.evaluation.opening_registry import physical_state_fingerprint
from songo_ai.evaluation.opening_registry_freeze import *


ROOT = Path(__file__).resolve().parents[3]
P5 = ROOT / "docs/lot46b_stage_c0_4_p5"
P6 = ROOT / "docs/lot46b_stage_c0_4_p6"


@pytest.fixture(scope="module")
def real_inventory():
    amendment = (P5 / "opening_protocol_amendment_v1.json").read_bytes()
    assert verify_amendment_bytes(amendment, AMENDMENT_SHA256) == AMENDMENT_SHA256
    assert hashlib.sha256((ROOT / "data/real_matches/match_moves_v1.jsonl").read_bytes()).hexdigest() == ORIGINAL_SOURCE_SHA256
    rows = load_jsonl_gzip(ROOT / "data/colab_bridge/lot45_selection.jsonl.gz")
    d_rl = state_fingerprints_from_jsonl(ROOT / "data/d_rl/lot11_g1_selected_seed_20260924.jsonl")
    return reproduce_candidates(rows, d_rl)


def test_approved_amendment_source_and_original_hashes(real_inventory):
    assert real_inventory["total_source_records"] == 491


def test_candidate_inventory_and_exact_training_exclusions(real_inventory):
    assert real_inventory["unique_physical_states"] == 491
    assert real_inventory["lot45_prefix_overlap"] == 49
    assert real_inventory["d_rl_overlap"] == 2
    assert len(real_inventory["training_exclusions"]) == 51
    assert len(real_inventory["eligible"]) == 440


def test_sha256_rank_seed_count_ids_and_canonical_registry(real_inventory):
    selected = select_openings(real_inventory["eligible"])
    assert selected == select_openings(tuple(reversed(real_inventory["eligible"])))
    assert len(selected) == 400
    assert [r["opening_id"] for r in selected] == [f"LOT46B-C0-OPENING-{i:04d}" for i in range(1, 401)]
    assert len({r["state_fingerprint"] for r in selected}) == 400
    payload = registry_payload(selected)
    assert serialize(payload) == serialize(json.loads(serialize(payload)))
    with pytest.raises(ValueError, match="approved amendment"):
        select_openings(real_inventory["eligible"], seed=SELECTION_SEED + 1)


def test_all_selected_physical_states_are_valid_and_have_70_seeds(real_inventory):
    for row in select_openings(real_inventory["eligible"]):
        state = RawSongoState(tuple(row["state"]["board"]), row["state"]["player_to_move"])
        assert sum(state.board) == 70
        assert physical_state_fingerprint(state) == row["state_fingerprint"]


def test_selected_set_has_no_training_or_b3_exact_overlap(real_inventory):
    selected = select_openings(real_inventory["eligible"])
    ids = {row["source_record_id"] for row in selected}
    assert not ids.intersection(real_inventory["training_exclusions"])
    # P3 proves all 994 B3 identifiers are in the excluded Lot45 10k prefix.
    assert json.loads((ROOT / "docs/lot46b_stage_c0_4_p3/b3_overlap_analysis.json").read_text())["B3_EXACT_OVERLAP_COUNT"] == 0


def test_side_reversal_shared_registry_and_resume_identity(real_inventory):
    ids = tuple(row["opening_id"] for row in select_openings(real_inventory["eligible"]))
    candidate = lambda name: Candidate(name, "unused.pt", "a" * 64, "c", "srn", "m", "synthetic", 0)
    comparisons = tuple({"comparison_id": f"c{i}", "candidate_a": "A", "candidate_b": "B",
                         "budget_a": 512, "budget_b": 512} for i in range(6))
    manifest = ExperimentManifest("synthetic-registry-check", "c1", "c", "e" * 64,
        (candidate("A"), candidate("B")), comparisons, ids, 46, {"registry_sha256": "f" * 64},
        "cpu", "fp32", {})
    games = schedule(manifest)
    assert len(games) == 4800
    for pair in range(0, len(games), 2):
        first, second = games[pair:pair + 2]
        assert first.opening_id == second.opening_id
        assert (first.candidate_p1, first.candidate_p2) == (second.candidate_p2, second.candidate_p1)
    assert manifest.fingerprint == ExperimentManifest(**manifest.__dict__).fingerprint


def test_atomic_publication_and_immutable_freeze(tmp_path):
    target = tmp_path / "registry.json"
    assert immutable_atomic_write(target, b"frozen\n") == hashlib.sha256(b"frozen\n").hexdigest()
    assert immutable_atomic_write(target, b"frozen\n") == hashlib.sha256(b"frozen\n").hexdigest()
    with pytest.raises(FileExistsError, match="divergent"):
        immutable_atomic_write(target, b"changed\n")


def test_frozen_registry_hash_and_no_execution_surface():
    registry = P6 / "opening_registry_400.json"
    expected = (P6 / "opening_registry_sha256.txt").read_text().split()[0]
    assert hashlib.sha256(registry.read_bytes()).hexdigest() == expected
    import songo_ai.evaluation.opening_registry_freeze as module
    assert all(not hasattr(module, name) for name in
               ("execute", "mcts", "checkpoint", "train", "play_game"))
