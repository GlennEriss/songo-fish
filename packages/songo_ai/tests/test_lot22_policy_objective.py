import torch

from songo_ai.evaluation.policy_objective import classify_flip,decision_regret,kendall_tau_from_rankings,legal_rank_map,pairwise_inversions,search_amplification,strategic_gap,strategically_weighted_inversions,top_k_set
from songo_ai.search import convert_value_perspective


def test_ranking_is_legal_only_and_topk_is_exact():
    policy=[.1,.8,.7,.2,0,0,0];mask=[1,0,1,1,0,0,0]
    assert legal_rank_map(policy,mask)=={2:1,3:2,0:3}
    assert top_k_set(policy,mask,2)==frozenset({2,3})


def test_pairwise_inversions_and_kendall_tau():
    mask=[1,1,1,0,0,0,0];parent=[.6,.3,.1,0,0,0,0];child=[.1,.3,.6,0,0,0,0]
    assert set(pairwise_inversions(parent,child,mask))=={(0,1),(0,2),(1,2)}
    assert kendall_tau_from_rankings((0,1,2),(2,1,0))==-1


def test_regret_flip_gap_and_swi_are_consistent():
    mask=[1,1,1,0,0,0,0];q=[.8,.75,-.2,None,None,None,None]
    assert decision_regret(q,mask,0)==0
    assert decision_regret(q,mask,2)>=0
    assert classify_flip(.5)=="HARMFUL_FLIP" and classify_flip(-.5)=="BENEFICIAL_FLIP" and classify_flip(.01)=="NEUTRAL_FLIP"
    assert strategic_gap(q,0,2)==1
    assert strategically_weighted_inversions([.6,.3,.1,0,0,0,0],[.1,.3,.6,0,0,0,0],mask,q)>1


def test_q_perspective_and_search_amplification_are_deterministic():
    assert convert_value_perspective(.4,1,2)==-.4
    first=search_amplification(.001,.1,True);second=search_amplification(.001,.1,True)
    assert first==second and first["ratio"]==100 and first["selected_action_changed"]


def test_diagnostic_module_contains_no_training_dependencies():
    import inspect,songo_ai.evaluation.policy_objective as module
    source=inspect.getsource(module).lower()
    assert "optimizer" not in source and "minimax" not in source and "teacher" not in source
    assert not any(isinstance(value,torch.optim.Optimizer) for value in vars(module).values())
