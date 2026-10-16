import pytest

from songo_ai.evaluation.stage_c0 import ArenaPair, paired_seed, preregistered_pairs, validate_match_manifest


def test_preregistration_has_six_equal_budget_c1_and_one_c2():
    pairs = preregistered_pairs()
    assert len([p for p in pairs if p.phase == "C1"]) == 6
    assert all(p.budget_a == p.budget_b for p in pairs if p.phase == "C1")
    c2 = [p for p in pairs if p.phase == "C2"]
    assert len(c2) == 1 and (c2[0].budget_a, c2[0].budget_b) == (512, 4096)


def test_seed_pairing_is_deterministic_and_side_specific():
    assert paired_seed(20264630, 7, 1) == paired_seed(20264630, 7, 1)
    assert paired_seed(20264630, 7, 1) != paired_seed(20264630, 7, 2)


def test_invalid_pair_and_seed_are_rejected():
    with pytest.raises(ValueError): ArenaPair("C1", "A", "A", 512, 512, 10)
    with pytest.raises(ValueError): paired_seed(1, 0, 3)


def test_manifest_schema_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing match manifest fields"):
        validate_match_manifest({"experiment_id": "x"})


def test_readiness_module_has_no_game_or_search_entrypoint():
    import songo_ai.evaluation.stage_c0 as module
    assert not hasattr(module, "play_arena_game")
    assert not hasattr(module, "SongoMCTS")
