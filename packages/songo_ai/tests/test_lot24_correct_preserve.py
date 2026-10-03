import torch

from songo_ai.model.correct_preserve import classify_pairs,correct_preserve_metrics,correction_loss,preservation_loss


def test_pair_classification_is_fixed_from_parent():
    pairs=((0,1,1.,.8),(2,1,.5,.4));corr,pres=classify_pairs([0.,1.,2.],pairs)
    assert corr[0][:2]==(0,1) and pres[0][:2]==(2,1)
    assert classify_pairs([0.,1.,2.],pairs)==(corr,pres)


def test_parent_and_safe_margin_are_recorded_and_scaled():
    _,pres=classify_pairs([2.,0.],((0,1,1.,1.),));assert pres[0][4]==2
    good=torch.tensor([[1.,0.]],requires_grad=True);assert preservation_loss(good,[pres],rho=.5)==0


def test_correction_gradient_pushes_preferred_action_up():
    logits=torch.tensor([[0.,1.]],requires_grad=True);corr=(((0,1,1.,1.,-1.),),);loss=correction_loss(logits,corr);loss.backward();assert logits.grad[0,0]<0 and logits.grad[0,1]>0


def test_preservation_gradient_opposes_conflicting_damage():
    logits=torch.tensor([[0.,1.,0.]],requires_grad=True);pres=(((0,1,1.,1.,2.),),);loss=preservation_loss(logits,pres,rho=.5);loss.backward();assert logits.grad[0,0]<0 and logits.grad[0,1]>0


def test_damage_efficiency_and_utility():
    logits=torch.tensor([[2.,0.,1.]]);corr=(((0,1,1.,1.,-1.),),);pres=(((2,1,.5,.5,1.),),);m=correct_preserve_metrics(logits,corr,pres);assert m["correction_rate"]==1 and m["preservation_rate"]==1 and m["damage_rate"]==0 and m["correction_efficiency"] is None and m["strategic_utility"]==1


def test_no_qdiag_value_teacher_or_minimax_contract():
    import inspect,songo_ai.model.correct_preserve as module
    source=inspect.getsource(module).lower();assert "value_target" not in source and "teacher" not in source and "minimax" not in source
