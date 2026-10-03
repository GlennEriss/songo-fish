import torch

from songo_ai.model.dual_source_training import parent_policy_kl, regularized_dual_source_loss


def sample():
    logits=torch.tensor([[1.,0.,9.,0.,0.,0.,0.]],requires_grad=True)
    parent=torch.tensor([[.7,.2,-9.,0.,0.,0.,0.]],requires_grad=True)
    mask=torch.tensor([[1,1,0,0,0,0,0]],dtype=torch.bool)
    target=torch.tensor([[.6,.4,0,0,0,0,0.]])
    value=torch.tensor([.2],requires_grad=True)
    return logits,parent,mask,target,value


def test_parent_kl_is_legal_only_finite_and_zero_at_parent():
    logits,parent,mask,_,_=sample()
    loss=parent_policy_kl(logits,parent,mask)
    assert torch.isfinite(loss) and loss>=0
    changed=parent.detach().clone();changed[0,2]=1e9
    assert torch.allclose(loss,parent_policy_kl(logits,changed,mask))
    assert torch.allclose(parent_policy_kl(parent,parent,mask),torch.tensor(0.),atol=1e-6)


def test_parent_is_frozen_and_candidate_receives_gradient():
    logits,parent,mask,_,_=sample();parent_policy_kl(logits,parent,mask).backward()
    assert logits.grad is not None and logits.grad.abs().sum()>0
    assert parent.grad is None


def test_beta_zero_matches_unregularized_total():
    rl,p_rl,mask,target,value=sample();re,p_re,re_mask,re_target,_=sample()
    out=regularized_dual_source_loss(rl,value,mask,target,torch.tensor([1.]),torch.ones(1),re,re_mask,re_target,torch.ones(1),p_rl,p_re,beta=0)
    expected=out["policy_rl"]+out["policy_re"]+out["value_rl"]
    assert torch.allclose(out["total"],expected)


def test_lambda_re_zero_removes_reanalysis_target_gradient():
    rl,p_rl,mask,target,value=sample();re,p_re,re_mask,re_target,_=sample()
    out=regularized_dual_source_loss(rl,value,mask,target,torch.tensor([1.]),torch.ones(1),re,re_mask,re_target,torch.ones(1),p_rl,p_re,lambda_re=0,beta=0)
    out["total"].backward()
    assert re.grad is not None and torch.equal(re.grad,torch.zeros_like(re.grad))
    assert value.grad is not None and value.grad.abs().sum()>0
    assert p_rl.grad is None and p_re.grad is None
