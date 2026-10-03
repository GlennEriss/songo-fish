from pathlib import Path


ROOT=Path(__file__).resolve().parents[3]
SCRIPT=ROOT/"apps/trainer/scripts/run_srn_lot36.py"


def source(): return SCRIPT.read_text()


def test_lot36_protocol_is_exact_and_frozen():
    s=source()
    assert "GAMES = 512" in s and "OPENINGS = 256" in s
    assert "BUDGETS = (64, 128, 256)" in s
    assert "EXPECTED_MINIMAX" in s and "70f3b5632da10cfa9df4c6f28c19f8c8b8927d514b515d1c31984243d61e334c" in s
    assert '"training_performed": False' in s
    assert '"minimax_labels": False' in s
    assert '"no_posthoc_mcts512": True' in s
    assert '"registry_tiebreak": False' in s


def test_lot36_uses_paired_side_swapped_shared_seed_manifests():
    s=source()
    assert '"paired": True' in s and '"side_swapped": True' in s
    assert '"shared_between_control_and_pool": True' in s
    assert "play_arena_game" in s and "a_player=1" in s and "a_player=2" in s


def test_lot36_checks_fingerprints_before_and_after():
    s=source()
    assert "exact_fingerprints()" in s
    assert 'fpdoc["before"] == after' in s
    assert "architecture_fingerprint" in s
    assert "policy_fingerprint" in s and "value_fingerprint" in s


def test_lot36_beat_and_robustness_criteria_are_not_raw_score_only():
    s=source()
    assert "score > .5 and ci and ci[0] > .5" in s
    assert "(beats[0] and beats[1]) or (beats[1] and beats[2])" in s
    assert "OFFICIAL_G4_REPRESENTATION" in s
    assert "G5_TRAINING_PERFORMED" in s
