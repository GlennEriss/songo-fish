import torch

from songo_ai.model.strategic_ranking import build_legal_pairs,pairwise_metrics,strategic_ranking_loss,strategic_weight


def test_legal_pairs_exclude_illegal_and_near_equivalent():
    pairs=build_legal_pairs([.8,.79,-.2,None,None,None,None],[1,1,1,0,0,0,0],epsilon=.02,scale=1.)
    assert all(a<3 and b<3 for a,b,_,_ in pairs)
    assert not any({a,b}=={0,1} for a,b,_,_ in pairs)


def test_unstable_pairs_are_excluded_and_weights_bounded():
    pairs=build_legal_pairs([.8,.2,-.5,None,None,None,None],[1,1,1,0,0,0,0],epsilon=.02,scale=.8,stable_pairs={(0,1)})
    assert len(pairs)==1 and pairs[0][:2]==(0,1)
    assert 0<=pairs[0][2]<=1 and strategic_weight(100,.02,.8)==1


def test_ranking_loss_is_finite_and_gradient_corrects_large_gap():
    logits=torch.tensor([[0.,1.,0.,0.,0.,0.,0.]],requires_grad=True);pairs=[((0,1,1.,1.),)]
    loss=strategic_ranking_loss(logits,pairs);assert torch.isfinite(loss);loss.backward()
    assert logits.grad[0,0]<0 and logits.grad[0,1]>0


def test_correct_pair_has_lower_loss_than_inverted_pair():
    pair=[((0,1,1.,1.),)];good=strategic_ranking_loss(torch.tensor([[2.,0.,0.,0.,0.,0.,0.]]),pair);bad=strategic_ranking_loss(torch.tensor([[0.,2.,0.,0.,0.,0.,0.]]),pair)
    assert good<bad


def test_metrics_measure_correction_preservation_and_net_gain():
    pairs=[((0,1,1.,1.),(2,1,.5,.5))];g2=torch.tensor([[0.,1.,2.,0.,0.,0.,0.]]);candidate=torch.tensor([[2.,0.,1.,0.,0.,0.,0.]])
    m=pairwise_metrics(candidate,pairs,g2);assert m["correction_rate"]==1 and m["preservation_rate"]==1 and m["net_strategic_gain"]>0


def test_qdiag_is_not_a_value_target_contract():
    import inspect,songo_ai.model.strategic_ranking as module
    source=inspect.getsource(module).lower();assert "value_target" not in source and "teacher" not in source and "minimax" not in source
