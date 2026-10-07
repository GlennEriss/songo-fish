import json
from collections import Counter
from pathlib import Path

import pytest

from lot44.artifacts import Lot44FatalError
from lot45.config import FORBIDDEN_TARGET_FIELDS, OCCURRENCE_FIELDS
from lot45.selection import allocate, coverage_report, deduplication_report, detail_selected, eligibility, holdout, ordered_selection, read_selection, scan, selection_rows, write_selection
from lot45.sources import SOURCES, iter_source
from run_srn_lot39 import fingerprint_state
from songo_ai.dataset import RawSongoState

from lot45_test_support import ROOT

INITIAL = [5] * 14 + [0, 0]


def board(shift: int) -> list[int]:
    b = list(INITIAL)
    taken = 1 + (shift // 14) % 4
    b[shift % 14] -= taken
    b[14] += taken
    return b


def d_rl_row(b, player, game, ply, winner, *, teacher=False):
    row = {"record_type": "example", "state": {"board": b, "player_to_move": player}, "legal_mask": [True] * 7, "policy_target": [1 / 7] * 7, "visit_counts": [10, 9, 9, 9, 9, 9, 9], "value_target": 0.0,
           "metadata": {"game_id": game, "ply": ply, "winner": winner, "status": "TERMINAL_WIN", "mcts_simulations": 64, "root_noise": True, "checkpoint_id": "G2"}}
    if teacher:
        row.update({"best_action": 3, "action_values": [9] * 7, "teacher": {"depth": 12}, "principal_variation": [1, 2], "wdl_target": [1, 0, 0]})
    return row


def source(name):
    return next(s for s in SOURCES if s.name == name)


def write_lines(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def repo(tmp_path):
    pool = tmp_path / "data/experiments/lot34_g4_training/pool_data/part-0000.jsonl"
    rows = [{"record_type": "manifest"}]
    for g in range(6):
        for ply in range(5):
            rows.append(d_rl_row(board(g * 5 + ply), 1 if ply % 2 == 0 else 2, f"pool-g{g}", ply, 1, teacher=(ply == 0)))
    rows.append(d_rl_row(INITIAL, 1, "pool-g0", 9, 2))
    write_lines(pool, rows)
    selfplay = tmp_path / "data/d_scale_v1/d_selfplay_large/part-00000.jsonl"
    write_lines(selfplay, [d_rl_row(board(1), 2, "g2-a", 1, 2), d_rl_row(board(40), 1, "g2-b", 3, 0)])
    rean = tmp_path / "data/d_scale_v1/d_reanalysis_large/part-00000.jsonl"
    write_lines(rean, [{"state": {"board": board(41), "player_to_move": 2}, "visit_counts": [0, 50, 0, 0, 0, 39, 39], "mcts_budget": 128, "dirichlet": False, "best_action": 1, "source_position_reservoir": "D_TEACHER_POSITION_ONLY"}])
    real = tmp_path / "data/real_matches/match_moves_v1.jsonl"
    write_lines(real, [
        {"match_id": "m1", "ply": 3, "player_position": 2, "state": board(42), "finished_after": False, "winner_after": None, "action_local": 4},
        {"match_id": "m1", "ply": 7, "player_position": 1, "state": board(43), "finished_after": True, "winner_after": 2, "action_local": 1},
    ])
    bad = tmp_path / "data/experiments/lot32_crossplay_scaling/control/part-0000.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json}\n" + json.dumps(d_rl_row(board(44), 1, "c-1", 0, 1)) + "\n" + json.dumps({"record_type": "example", "state": {"board": [1] * 16, "player_to_move": 1}, "metadata": {}}) + "\n")
    return tmp_path


def test_occurrences_are_whitelisted_and_never_carry_teacher_labels(repo):
    errors = []
    occs = list(iter_source(repo, source("LOT34_POOL_DATA"), errors.append)) + list(iter_source(repo, source("D_SCALE_REANALYSIS_LARGE"), errors.append))
    assert occs and not errors
    for occ in occs:
        assert set(occ) == set(OCCURRENCE_FIELDS)
        assert not set(occ) & set(FORBIDDEN_TARGET_FIELDS)
        assert "best_action" not in json.dumps(occ) and "principal_variation" not in json.dumps(occ)


def test_invalid_lines_are_reported_not_ignored(repo):
    errors = []
    occs = list(iter_source(repo, source("LOT32_CONTROL"), errors.append))
    assert len(occs) == 1 and len(errors) == 2
    assert {e["line"] for e in errors} == {1, 3} and all(e["source"] == "LOT32_CONTROL" for e in errors)


def test_z_is_player_to_move_perspective_and_real_uses_final_winner(repo):
    occs = list(iter_source(repo, source("LOT34_POOL_DATA"), lambda e: None))
    for occ in occs:
        expected = 1.0 if occ["state"]["player_to_move"] == 1 else -1.0
        if occ["ply"] == 9:
            expected = -1.0  # winner 2, player 1
        assert occ["z"] == expected
    real = list(iter_source(repo, source("D_REAL_HUMAN_MATCHES"), lambda e: None))
    assert [o["z"] for o in real] == [1.0, -1.0]  # winner 2: +1 for player 2, -1 for player 1
    assert [o["old_target"] for o in real] == [None, None]
    rean = list(iter_source(repo, source("D_SCALE_REANALYSIS_LARGE"), lambda e: None))
    assert rean[0]["z"] is None and rean[0]["old_target"]["budget"] == 128


def test_real_corpus_z_matches_value_target():
    path = ROOT / "data/experiments/lot34_g4_training/pool_data/part-0000.jsonl"
    if not path.is_file():
        pytest.skip("Lot34 pool data not available")
    rows = []
    for line in path.read_text().splitlines()[:400]:
        row = json.loads(line)
        if row.get("record_type") != "manifest":
            rows.append(row)
    from lot45.sources import from_d_rl

    for row in rows:
        assert from_d_rl(source("LOT34_POOL_DATA"), row, {})["z"] == row["value_target"]


def test_scan_dedup_counts_occurrences_and_primary_family(repo):
    states, audit = scan(repo, excluded_games=set(), on_error=lambda e: None, log=lambda _: None)
    fp_initial = fingerprint_state(RawSongoState(tuple(INITIAL), 1))
    fp_b1_p2 = fingerprint_state(RawSongoState(tuple(board(1)), 2))
    assert states[fp_b1_p2].occurrences == 2  # pool-g0 ply1 + G2 self-play
    assert SOURCES[states[fp_b1_p2].primary].name == "LOT34_POOL_DATA"
    assert states[fp_initial].occurrences == 1 and states[fp_initial].z_loss == 1
    assert audit["LOT34_POOL_DATA"]["occurrences"] == 31 and audit["LOT34_POOL_DATA"]["games"] == 6
    report = deduplication_report(states)
    assert report["raw_occurrences"] == sum(a["occurrences"] for a in audit.values())
    assert report["duplicate_occurrences"] == report["raw_occurrences"] - report["unique_physical_states"]


def test_eligibility_excludes_diagnostic_states_and_games(repo):
    states, _ = scan(repo, excluded_games={"pool-g3"}, on_error=lambda e: None, log=lambda _: None)
    diag = fingerprint_state(RawSongoState(tuple(board(10)), 1))
    eligible, reasons = eligibility(states, {diag})
    assert diag not in eligible and reasons["lot41_44_diagnostic_state"] == 1
    assert reasons["game_of_lot41_44_diagnostic_state"] == 5


def test_ordered_selection_prefix_game_cap_and_details(repo):
    states, _ = scan(repo, excluded_games=set(), on_error=lambda e: None, log=lambda _: None)
    eligible, _ = eligibility(states, set())
    order, info = ordered_selection(states, eligible, total=20, seed=3)
    assert len(order) == len(set(order)) and len(order) <= 20
    groups = Counter(states[fp].group for fp in order)
    assert max(groups.values()) <= info["max_positions_per_game"]
    assert ordered_selection(states, eligible, total=20, seed=3)[0] == order
    details = detail_selected(repo, set(order), on_error=lambda e: None)
    rows = selection_rows(order, states, eligible, details, seed=3)
    assert [r["order"] for r in rows] == list(range(len(rows)))
    for r in rows:
        assert r["z_perspective"] == "player_to_move" and r["split_group"] and sum(r["sources"].values()) == r["source_occurrence_count"]
        assert "best_action" not in json.dumps(r)
    coverage = coverage_report(rows)
    assert 0 <= coverage["NEW_STATE_RATE_VS_G2_G3_G4_TRAINING"] <= 1


def test_allocation_redistributes_and_respects_availability():
    alloc = allocate({"D_REAL_HUMAN": 5, "G4_TRAINING_CROSSPLAY": 1000, "G3_CROSSPLAY_SCALING": 10, "G2_SELFPLAY": 1000}, 200)
    assert alloc["D_REAL_HUMAN"] == 5 and alloc["G3_CROSSPLAY_SCALING"] == 10
    assert sum(alloc.values()) == 200 and alloc["G4_TRAINING_CROSSPLAY"] > alloc["G2_SELFPLAY"]


def test_selection_roundtrip_and_duplicate_rejection(tmp_path):
    rows = [{"order": i, "fingerprint": f"{i:064x}"} for i in range(5)]
    path = tmp_path / "s.jsonl.gz"
    write_selection(path, rows)
    assert read_selection(path) == rows
    write_selection(path, rows + [{"order": 5, "fingerprint": rows[0]["fingerprint"]}])
    with pytest.raises(Lot44FatalError):
        read_selection(path)


def test_holdout_is_group_deterministic():
    assert holdout("game-1", 1) == holdout("game-1", 1)
    rate = sum(holdout(f"g{i}", 7) for i in range(4000)) / 4000
    assert 0.03 < rate < 0.07
