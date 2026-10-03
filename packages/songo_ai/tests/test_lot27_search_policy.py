import torch
from songo_ai.dataset import RawSongoState
from songo_ai.evaluation.search_policy_diagnosis import *
from songo_ai.model import load_srn_checkpoint
from songo_ai.search import MCTSConfig,SongoMCTS
from songo_ai.evaluation.hybrid_evaluator import DepthPolicyValueEvaluator

def test_visit_js_and_amplification():
    assert js_divergence([.5,.5],[.5,.5])==0 and amplification_ratio(.1,.2)==2
def test_first_divergence_and_categories():
    assert first_divergence([1,1,2],[1,2,2])==2 and search_category(True,False)=="C" and search_category(False,True)=="B"
def test_regret_rwpm_recovery_and_lockin():
    q=[1.,.5,None,None,None,None,None];m=[1,1,0,0,0,0,0];assert regret(q,1,m)==.5 and rwpm([.5,.5,0,0,0,0,0],q,m)==.25
    assert search_recovery([1,1,0]) and early_prior_lockin(1,[1,1,1],q,m)
def test_deterministic_trace_contains_p_n_q_u_and_does_not_train():
    model=load_srn_checkpoint("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt").model;before=[x.clone() for x in model.parameters()];state=RawSongoState((5,)*14+(0,0),1);cfg=MCTSConfig(num_simulations=4,c_puct=1.5,add_root_noise=False,collect_simulation_trace=True,seed=27);a=SongoMCTS(model,config=cfg).search(state);b=SongoMCTS(model,config=cfg).search(state)
    assert a.simulation_trace==b.simulation_trace and {"root_priors_all","visit_counts_after_backup","root_q_values_after_backup","root_puct_scores_after_backup"}<=set(a.simulation_trace[-1])
    assert all(torch.equal(x,y) for x,y in zip(before,model.parameters()))
def test_root_and_descendant_policy_ablation_routes_by_depth():
    g2=load_srn_checkpoint("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt").model;g3=load_srn_checkpoint("data/experiments/lot26_g3_scale/checkpoints/g3_strategic_best.pt").model;hybrid=DepthPolicyValueEvaluator(g3,g2,g2);state=RawSongoState((5,)*14+(0,0),1);r=SongoMCTS(hybrid,config=MCTSConfig(num_simulations=4,seed=27)).search(state)
    assert r.num_simulations==4 and r.network_evaluations>1
