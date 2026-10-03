import json
from pathlib import Path
import pytest

from songo_ai.dataset.scale_data import (deduplicate_physical,deterministic_stratified_sample,
 intersection_matrix,manifest_hash,position_only,shard_for,validate_reanalysis_record)


def state(board=None,player=1,**extra):
    return {"state":{"board":board or [0]*16,"player_to_move":player},**extra}


def test_physical_dedup_handles_nested_corpora_and_ignores_teacher_labels():
    rows=[state(best_action=2),state(best_action=5),state(player=2,action_values=[1])]
    out=deduplicate_physical(rows)
    assert len(out)==2 and all(set(x)=={"state"} for x in out)
    assert "best_action" not in json.dumps(out) and "action_values" not in json.dumps(out)


def test_position_only_rejects_incomplete_state():
    with pytest.raises(ValueError):position_only({"board":[0]*16})


def test_deterministic_stratified_sampling_is_reproducible_and_balanced():
    rows=[{"position_hash":str(i),"player":i%2,"bin":i%3} for i in range(30)]
    a=deterministic_stratified_sample(rows,12,seed=25,stratum_fields=("player","bin"))
    b=deterministic_stratified_sample(rows,12,seed=25,stratum_fields=("player","bin"))
    assert [x["position_hash"] for x in a]==[x["position_hash"] for x in b]
    assert len({(x["player"],x["bin"]) for x in a})==6


def test_reanalysis_contract_is_policy_only_autonomous_and_has_exact_visits():
    row={"state":{"board":[0]*16,"player_to_move":1},"legal_mask":[1]*7,"visit_counts":[128,0,0,0,0,0,0],"policy_target":[1,0,0,0,0,0,0],"source_dataset":"REANALYSIS","generation_model":"G2-best","mcts_budget":128}
    validate_reanalysis_record(row,expected_visits=128)
    for bad in ({**row,"value_target":0.2},{**row,"best_action":0},{**row,"visit_counts":[127,0,0,0,0,0,0]}):
        with pytest.raises(ValueError):validate_reanalysis_record(bad,expected_visits=128)


def test_intersection_matrix_is_symmetric_and_preserves_diagonal():
    m=intersection_matrix({"old":{1,2},"new":{2,3}})
    assert m["old"]["new"]==m["new"]["old"]==1 and m["old"]["old"]==2


def test_sharding_and_manifest_hash_are_deterministic(tmp_path:Path):
    p=tmp_path/"x";p.write_text("immutable")
    assert shard_for("abc",17)==shard_for("abc",17)
    assert manifest_hash([p],{"seed":25})==manifest_hash([p],{"seed":25})
    assert manifest_hash([p],{"seed":25})!=manifest_hash([p],{"seed":26})
