import copy
import torch
from songo_ai.dataset import RawSongoState
from songo_ai.evaluation import ArenaConfig,HybridPolicyValueEvaluator,SRNMCTSAgent,generate_unique_deterministic_openings,independent_seed_set,promotion_rule,run_paired_arena,side_gap
from songo_ai.model import SongoGraphBuilder,load_srn_checkpoint

G2="data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt";G3="data/experiments/lot26_g3_scale/checkpoints/g3_strategic_best.pt";V="data/experiments/lot28_value_search/checkpoints/v28_a_best.pt"
def test_checkpoint_immutable_and_no_training_guarantee():
 p=load_srn_checkpoint(G3).model;v=load_srn_checkpoint(V).model;h=HybridPolicyValueEvaluator(p,v);before=copy.deepcopy(h.state_dict());g=SongoGraphBuilder().build_batch([RawSongoState((5,)*14+(0,0),1)])
 with torch.no_grad():pp,_=p(g);ph,_=h(g)
 assert torch.equal(pp,ph) and all(torch.equal(before[k],h.state_dict()[k]) for k in before)
def test_independent_seed_set():assert independent_seed_set([29,30,31],[28,32]) and not independent_seed_set([29,29],[28]) and not independent_seed_set([28,29],[28])
def test_paired_arena_side_swap_and_reproducibility():
 a=load_srn_checkpoint(G2).model;b=load_srn_checkpoint(G3).model;o=generate_unique_deterministic_openings(count=2,seed=2900,max_prefix_length=3);cfg=ArenaConfig(max_plies=2,repetition_limit=3,seed=2901,bootstrap_samples=20)
 def play():return run_paired_arena(SRNMCTSAgent("A",a,1),SRNMCTSAgent("B",b,1),o,config=cfg)
 x,y=play(),play();assert len(x)==4 and [z.a_player for z in x]==[1,2,1,2] and [z.action_sequence for z in x]==[z.action_sequence for z in y]
def test_promotion_rule_is_fixed_and_requires_all_gates():
 assert promotion_rule(.55,.52,side_gap64=.03,side_gap128=.04,score256=.54)["promoted"]
 assert not promotion_rule(.55,.49,side_gap64=.03,side_gap128=.04,score256=.54)["promoted"]
 assert not promotion_rule(.56,.51,side_gap64=.16,side_gap128=.04,score256=.54)["promoted"]
def test_side_gap():
 s={"by_a_side":{"P1":{"games":10,"wins":6,"draws":0,"losses":4},"P2":{"games":10,"wins":5,"draws":0,"losses":5}}};assert abs(side_gap(s)-.1)<1e-12
