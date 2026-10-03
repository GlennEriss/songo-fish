import torch
from songo_ai.model.dual_source_training import bounded_confidence_weights, bounded_regret_weights, dual_source_loss, weighted_policy_cross_entropy

def tensors():
    logits=torch.tensor([[1.,0.,-1.,0.,0.,0.,0.]],requires_grad=True); value=torch.tensor([.2],requires_grad=True); mask=torch.tensor([[1,1,0,0,0,0,0]],dtype=torch.bool); target=torch.tensor([[.6,.4,0,0,0,0,0.]])
    return logits,value,mask,target

def test_regret_weights_bounded_and_alpha_zero_is_baseline():
    assert bounded_regret_weights([None,0,2,20],alpha=0,q95=2)==[1,1,1,1]
    weights=bounded_regret_weights([None,0,2,20],alpha=2,q95=2); assert min(weights)>=1/3 and max(weights)<=3 and abs(sum(weights)/len(weights)-1)<1e-9

def test_confidence_weights_are_bounded_and_reproducible():
    metadata=[{"search_confidence":{"visit_margin":.1,"target_entropy":.5}},{"search_confidence":{"visit_margin":.8,"target_entropy":.1}}]; first=bounded_confidence_weights(metadata,[4,4]); assert first==bounded_confidence_weights(metadata,[4,4]); assert min(first)>=.5 and max(first)<=1.5 and abs(sum(first)/len(first)-1)<1e-9

def test_weighted_ce_with_unit_weights_equals_unweighted():
    logits,_,mask,target=tensors(); weighted=weighted_policy_cross_entropy(logits,mask,target,torch.ones(1)); expected=-(target*torch.log_softmax(torch.tensor([[1.,0.,-1e9,-1e9,-1e9,-1e9,-1e9]]),-1)).sum(); assert torch.allclose(weighted,expected)

def test_reanalysis_policy_has_no_value_gradient():
    re_logits,re_value,re_mask,re_target=tensors(); loss=weighted_policy_cross_entropy(re_logits,re_mask,re_target,torch.ones(1)); loss.backward(); assert re_logits.grad is not None and re_logits.grad.abs().sum()>0; assert re_value.grad is None

def test_dual_loss_value_depends_only_on_rl():
    rl_logits,rl_value,mask,target=tensors(); re_logits,re_value,re_mask,re_target=tensors(); out=dual_source_loss(rl_logits,rl_value,mask,target,torch.tensor([1.]),torch.ones(1),re_logits,re_mask,re_target,torch.ones(1)); out["total"].backward(); assert rl_value.grad is not None and rl_value.grad.abs().sum()>0; assert re_value.grad is None
