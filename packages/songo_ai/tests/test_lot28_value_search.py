import copy

import torch

from songo_ai.dataset import RawSongoState
from songo_ai.evaluation import (
    ArenaConfig, HybridPolicyValueEvaluator, SRNMCTSAgent, flip_rate,
    generate_unique_deterministic_openings, local_value_consistency,
    run_paired_arena, search_sensitivity, value_metrics,
)
from songo_ai.model import SongoGraphBuilder, load_srn_checkpoint
from songo_ai.search import MCTSConfig, SongoMCTS

G2="data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt"
G3="data/experiments/lot26_g3_scale/checkpoints/g3_strategic_best.pt"


def state(): return RawSongoState((5,)*14+(0,0),1)


def test_hybrid_identity_and_shared_encoder_protection():
    g2=load_srn_checkpoint(G2).model;g3=load_srn_checkpoint(G3).model;h=HybridPolicyValueEvaluator(g3,g2);graph=SongoGraphBuilder().build_batch([state()])
    with torch.no_grad():p2,v2=g2(graph);p3,_=g3(graph);ph,vh=h(graph)
    assert torch.equal(ph,p3) and torch.equal(vh,v2)
    before=copy.deepcopy(g3.state_dict());opt=torch.optim.AdamW(g2.value_mlp.parameters(),lr=1e-4);_,v=h(graph);opt.zero_grad();v.square().mean().backward();opt.step()
    assert all(torch.equal(before[k],g3.state_dict()[k]) for k in before)


def test_only_value_head_is_trainable_and_policy_stays_exact():
    model=load_srn_checkpoint(G2).model
    for p in model.parameters():p.requires_grad=False
    for p in model.value_mlp.parameters():p.requires_grad=True
    assert all(p.requires_grad for p in model.value_mlp.parameters())
    assert not any(p.requires_grad for n,p in model.named_parameters() if not n.startswith("value_mlp."))


def test_true_z_metrics_have_no_pseudo_target_interface():
    assert value_metrics([1.,0.,-1.],[1.,0.,-1.])["mse"]==0
    try:value_metrics([0.],[])
    except ValueError:pass
    else:assert False


def test_leaf_instrumentation_sensitivity_and_flip_rate():
    model=load_srn_checkpoint(G2).model;r=SongoMCTS(model,config=MCTSConfig(num_simulations=2,collect_simulation_trace=True,seed=28)).search(state())
    assert {"leaf_value","leaf_depth","leaf_state","root_q_values_after_backup"}<=set(r.simulation_trace[-1])
    s=search_sensitivity([{"delta_value":.2,"delta_q":.1,"visit_js":.03,"action_flip":True},{"delta_value":.01,"delta_q":.0,"visit_js":.0,"action_flip":False}])
    assert s["flip_rate"]==.5 and flip_rate([1,2],[1,3])==.5
    assert local_value_consistency([.5],[-.5])["mean_abs_residual"]==0


def test_checkpoint_reload_and_arena_reproducibility():
    a=load_srn_checkpoint(G2).model;b=load_srn_checkpoint(G2).model
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))
    openings=generate_unique_deterministic_openings(count=1,seed=28,max_prefix_length=2);cfg=ArenaConfig(max_plies=2,repetition_limit=3,seed=28,bootstrap_samples=20)
    def play():return run_paired_arena(SRNMCTSAgent("A",a,1),SRNMCTSAgent("B",b,1),openings,config=cfg)
    left,right=play(),play();assert [x.action_sequence for x in left]==[x.action_sequence for x in right]
