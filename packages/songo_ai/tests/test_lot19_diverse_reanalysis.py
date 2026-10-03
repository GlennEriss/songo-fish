import json
import pytest

from songo_ai.dataset import RawSongoState, ReanalysisPolicyExample, iter_reanalysis_jsonl, write_reanalysis_jsonl
from songo_ai.evaluation import balanced_sample, diversity_stratum, policy_stability, structural_descriptor


def state(): return RawSongoState(tuple([5]*14+[0,0]),1)


def example():
    return ReanalysisPolicyExample(state(),(True,)*7,(10,9,8,7,6,5,4),tuple(v/49 for v in (10,9,8,7,6,5,4)),{"mcts_budget":64,"dirichlet":False},{"source":"D_TEACHER_POSITION_ONLY","source_corpus":["x"],"position_hash":"abc"})


def test_structural_descriptor_contains_no_teacher_signal():
    d=structural_descriptor(state().board,1,(True,)*7,"x")
    assert d["legal_count"]==7 and d["territory_seeds_p1"]==35
    assert not ({"best_action","action_values","score","margin"}&set(d))
    assert diversity_stratum(d)==diversity_stratum(dict(d))


def test_diverse_selection_is_reproducible_and_balanced():
    rows=[]
    for player in (1,2):
        for legal in (1,7):
            for i in range(10):
                rows.append({"player_to_move":player,"legal_count":legal,"seeds_in_play":10+i,"store_p1":20,"store_p2":20,"nonempty_pits_p1":4,"nonempty_pits_p2":4,"provenance":"x"})
    first=balanced_sample(rows,12,seed=19); second=balanced_sample(rows,12,seed=19)
    assert first==second
    assert {(rows[i]["player_to_move"],rows[i]["legal_count"]) for i in first}=={(1,1),(1,7),(2,1),(2,7)}


def test_reanalysis_round_trip_has_no_fake_z_or_teacher(tmp_path):
    path=tmp_path/"r.jsonl"; write_reanalysis_jsonl(path,[example()],metadata={"value_target":"ABSENT"})
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert "value_target" not in rows[1] and "best_action" not in rows[1]
    assert not ({"teacher","action_values","minimax_score"}&set(rows[1]["source_position_metadata"]))
    assert list(iter_reanalysis_jsonl(path))==[example()]


def test_reanalysis_rejects_teacher_and_value_metadata():
    base=example()
    with pytest.raises(ValueError): ReanalysisPolicyExample(base.state,base.legal_mask,base.visit_counts,base.policy_target,{"value_target":1},{"source":"D_TEACHER_POSITION_ONLY"})
    with pytest.raises(ValueError): ReanalysisPolicyExample(base.state,base.legal_mask,base.visit_counts,base.policy_target,{}, {"source":"D_TEACHER_POSITION_ONLY","best_action":3})


def test_policy_stability_reports_ranking_changes():
    same=policy_stability((.6,.4,0,0,0,0,0),(.55,.45,0,0,0,0,0),(True,True,False,False,False,False,False))
    changed=policy_stability((.6,.4,0,0,0,0,0),(.4,.6,0,0,0,0,0),(True,True,False,False,False,False,False))
    assert same["argmax_agreement"] and not changed["argmax_agreement"]
    assert changed["js"]>same["js"]
