import pytest
from songo_ai.generation.population import *

def test_model_registry_roles_and_retirement_guard():
    m=RegisteredModel("G2","abc",frozenset({ModelRole.CHAMPION,ModelRole.GENERATOR}),"G2","G2");assert ModelRole.CHAMPION in m.roles
    with pytest.raises(ValueError):RegisteredModel("old","x",frozenset({ModelRole.RETIRED,ModelRole.GENERATOR}),"old","old")
def test_deterministic_cross_play_schedule():
    q=(MatchupQuota("G2","G3",ModelRole.CHAMPION,ModelRole.GENERATOR,4,"cross_play"),MatchupQuota("G3","G2",ModelRole.GENERATOR,ModelRole.CHAMPION,4,"cross_play_side_swap"))
    a=deterministic_schedule("pilot-v1",q,seed=30,mcts_budget=64);b=deterministic_schedule("pilot-v1",q,seed=30,mcts_budget=64)
    assert a==b and len(a)==8 and {x.p1_model_id for x in a}=={"G2","G3"} and len({x.seed for x in a})==8
def test_provenance_requires_terminal_result_and_model_fingerprints():
    r={k:"x" for k in REQUIRED_PROVENANCE_FIELDS};r["result"]=1;validate_game_provenance(r)
    del r["p1_fingerprint"]
    with pytest.raises(ValueError):validate_game_provenance(r)
def test_source_balancing_prevents_volume_domination():
    ids=["champion"]*100+["cross"]*3+["historical"]
    selected=source_balanced_indices(ids,12,seed=30);counts={s:sum(ids[i]==s for i in selected) for s in set(ids)}
    assert counts=={"champion":4,"cross":4,"historical":4}
