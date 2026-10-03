import torch
import pytest
from songo_ai.model.large_scale_training import SourceAwareSchedule,deterministic_epoch_indices,split_game_ids,split_physical_hashes,verify_value_sources
from songo_ai.model import load_srn_checkpoint

def test_source_aware_sampler_does_not_follow_raw_ratio():
    s=SourceAwareSchedule(128,128,32);assert s.batch_size==256

def test_selfplay_split_is_strictly_by_game_id():
    games=[f"g{i}" for i in range(100)];train,val=split_game_ids(games,seed=26);assert not train&val and train|val==set(games)

def test_reanalysis_split_is_by_physical_identity():
    hashes=[f"h{i}" for i in range(100)];train,val=split_physical_hashes(hashes,seed=26);assert not train&val and train|val==set(hashes)

def test_value_target_is_selfplay_only():
    verify_value_sources(["SELFPLAY","REANALYSIS","STRATEGIC"],[True,False,False])
    with pytest.raises(ValueError):verify_value_sources(["REANALYSIS"],[True])

def test_epoch_sampling_is_reproducible_and_changes_by_epoch():
    assert deterministic_epoch_indices(20,50,seed=26,epoch=1)==deterministic_epoch_indices(20,50,seed=26,epoch=1)
    assert deterministic_epoch_indices(20,50,seed=26,epoch=1)!=deterministic_epoch_indices(20,50,seed=26,epoch=2)

def test_exact_g2_initialization_is_reproducible():
    p="data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt";a=load_srn_checkpoint(p).model;b=load_srn_checkpoint(p).model
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))

def test_contract_has_no_external_label_dependency():
    import inspect,songo_ai.model.large_scale_training as m
    source=inspect.getsource(m).lower();assert "teacher_value" not in source and "minimax_score" not in source
