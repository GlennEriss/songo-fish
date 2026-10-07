import pytest

from lot44.artifacts import Lot44FatalError
from lot44.corpus import build_corpus, eligible, legal_action_count, load_original_reference, select_candidates, state_from_dict
from lot44.splits import group_split, leakage_report
from run_srn_lot39 import fingerprint_state
from songo_ai.dataset import RawSongoState

from lot44_test_support import ORIGINAL_POSITIONS, require_inputs

INITIAL = {"board": [5] * 14 + [0, 0], "player_to_move": 1}


def mid_state(shift: int) -> dict:
    board = [5] * 14 + [0, 0]
    board[0] -= shift
    board[14] += shift
    return {"board": board, "player_to_move": 1}


def record(game: str, ply: int, state: dict, line: int = 1) -> dict:
    return {"state": state, "game_id": game, "ply": ply, "source_shard": "part-0000.jsonl", "source_line": line, "generator": {}}


def test_identity_key_is_board_and_player_to_move():
    a = RawSongoState(tuple(INITIAL["board"]), 1)
    b = RawSongoState(tuple(INITIAL["board"]), 2)
    assert fingerprint_state(a) != fingerprint_state(b)
    assert fingerprint_state(a) == fingerprint_state(state_from_dict(INITIAL))


def test_eligibility_excludes_terminal_and_single_action():
    assert legal_action_count(state_from_dict(INITIAL)) == 7 and eligible(state_from_dict(INITIAL))
    terminal = RawSongoState(tuple([0] * 7 + [5] * 7 + [35, 0]), 1)
    assert legal_action_count(terminal) is None and not eligible(terminal)
    single = RawSongoState(tuple([0] * 6 + [2] + [5] * 7 + [33, 0]), 1)
    assert legal_action_count(single) == 1 and not eligible(single)


def test_select_candidates_one_per_game_dedup_and_exclusion():
    records = [record("g1", 0, INITIAL, 1), record("g1", 1, mid_state(1), 2), record("g2", 0, mid_state(2), 1), record("g3", 0, mid_state(2), 1), record("g4", 3, mid_state(3), 1), record("old", 0, mid_state(4), 1), record(None, 0, mid_state(4), 1)]
    excluded_fp = {fingerprint_state(state_from_dict(INITIAL))}
    out = select_candidates(records, excluded_fps=excluded_fp, excluded_games={"old"}, seed=1)
    rows = out["rows"]
    games = [r["game_id"] for r in rows]
    assert len(games) == len(set(games))
    assert len({r["fingerprint"] for r in rows}) == len(rows)
    assert "old" not in games and all(r["fingerprint"] not in excluded_fp for r in rows)
    assert out["stats"]["cross_game_physical_duplicates_dropped"] == 1  # g2/g3 share a state
    assert out["stats"]["missing_game_id"] == 1 and out["stats"]["excluded_original_game_records"] == 1
    assert all(set(r) >= {"fingerprint", "state", "game_id", "ply"} and "policy_target" not in r for r in rows)
    assert select_candidates(records, excluded_fps=excluded_fp, excluded_games={"old"}, seed=1)["rows"] == rows


def unique_states(n: int) -> list[dict]:
    states, seen = [], set()
    for a in range(5):
        for b in range(5):
            for p in (1, 2):
                board = [5] * 14 + [0, 0]
                board[0] -= a
                board[8] -= b
                board[14] += a
                board[15] += b
                s = {"board": board, "player_to_move": p}
                fp = fingerprint_state(state_from_dict(s))
                if fp not in seen and eligible(state_from_dict(s)):
                    seen.add(fp)
                    states.append(s)
    return states[:n]


def candidate(i: int, s: dict) -> dict:
    return {"fingerprint": fingerprint_state(state_from_dict(s)), "state": s, "game_id": f"g{i}", "ply": 0}


def test_build_corpus_removes_overlap_and_rejects_duplicates():
    states = unique_states(20)
    cands = [candidate(i, s) for i, s in enumerate(states)]
    reference = {"positions_file": "x", "fingerprints": {cands[0]["fingerprint"]}, "games": {"g1"}, "lot41_consistent": None, "lot42_fingerprints": 0}
    out = build_corpus(cands, reference, max_positions=10, seed=3)
    assert len(out["rows"]) == 10
    assert out["overlap"]["OOS_OVERLAP_WITH_ORIGINAL_256"] == 0 and out["overlap"]["OOS_GAME_OVERLAP_WITH_ORIGINAL_256"] == 0
    assert out["overlap"]["candidate_state_overlap_removed"] == 1 and out["overlap"]["candidate_game_overlap_removed"] == 1
    assert out["available"] == 18
    with pytest.raises(Lot44FatalError) as err:
        build_corpus(cands + [dict(cands[3], game_id="other")], reference, max_positions=10, seed=3)
    assert err.value.code == "CORPUS_CORRUPT"
    bad = dict(cands[4], fingerprint="0" * 64)
    with pytest.raises(Lot44FatalError):
        build_corpus([bad], reference, max_positions=10, seed=3)


def test_build_corpus_cap_is_deterministic_and_seeded():
    cands = [candidate(i, s) for i, s in enumerate(unique_states(30))]
    reference = {"positions_file": "x", "fingerprints": set(), "games": set(), "lot41_consistent": None, "lot42_fingerprints": 0}
    a = build_corpus(cands, reference, max_positions=12, seed=1)["rows"]
    b = build_corpus(cands, reference, max_positions=12, seed=1)["rows"]
    c = build_corpus(cands, reference, max_positions=12, seed=2)["rows"]
    assert a == b and a != c


def test_group_split_keeps_games_together_and_is_deterministic():
    rows = [{"fingerprint": f"{i:064x}", "game_id": f"g{i // 3}"} for i in range(90)]
    parts = group_split(rows, seed=5)
    assert {k: len(v) for k, v in parts.items()} == {"train": 54, "calibration": 18, "test": 18}
    report = leakage_report(parts)
    assert report == {"same_game_across_partitions": 0, "same_state_across_partitions": 0, "within_partition_duplicates": 0}
    assert group_split(rows, seed=5) == parts and group_split(rows, seed=6) != parts
    leaked = {k: list(v) for k, v in parts.items()}
    leaked["test"].append(parts["train"][0])
    assert leakage_report(leaked)["same_game_across_partitions"] == 1


def test_original_reference_is_the_256_lot39_states():
    require_inputs()
    ref = load_original_reference(ORIGINAL_POSITIONS)
    assert len(ref["fingerprints"]) == 256 and len(ref["games"]) == 256
