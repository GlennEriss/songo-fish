import pytest
from songo_ai.dataset import RawSongoState,RLTrainingExample
from songo_ai.evaluation.generator_pool import *
from songo_ai.generation import MatchupQuota,ModelRole,deterministic_schedule,source_balanced_indices
from songo_ai.generation.selfplay import PendingSelfPlayStep,SelfPlayStatus,finalize_selfplay_steps

def ex(board_shift,game,source,action=0):
 b=[5]*14+[0,0];b[0]-=board_shift;b[1]+=board_shift
 return RLTrainingExample(RawSongoState(tuple(b),1),(True,)*7,(1/7,)*7,1.,(1,)*7,{"game_id":game,"source_type":source,"action_played":action})
def test_exact_arm_sizes_schedule_and_crossplay_side_balance():
 q=(MatchupQuota("G2","G2",ModelRole.CHAMPION,ModelRole.CHAMPION,100,"self"),MatchupQuota("G3","G3",ModelRole.GENERATOR,ModelRole.GENERATOR,100,"self"),MatchupQuota("G2","G3",ModelRole.CHAMPION,ModelRole.GENERATOR,100,"cross"),MatchupQuota("G3","G2",ModelRole.GENERATOR,ModelRole.CHAMPION,100,"cross_swap"));s=deterministic_schedule("pool",q,seed=31,mcts_budget=64)
 assert len(s)==400 and sum(x.p1_model_id=="G2" and x.p2_model_id=="G3" for x in s)==100 and sum(x.p1_model_id=="G3" and x.p2_model_id=="G2" for x in s)==100
def test_unique_overlap_crossplay_only_and_marginal_novelty():
 xs=[ex(0,"a","G2_G2"),ex(1,"b","G3_G3"),ex(2,"c","CROSS_PLAY"),ex(0,"d","CROSS_PLAY")];sets=source_state_sets(xs)
 assert state_coverage(xs)["unique_physical_states"]==3 and len(crossplay_only_states(sets))==1 and overlap(sets["G2_G2"],sets["CROSS_PLAY"])["intersection"]==1
 assert marginal_novelty(("G2_G2","G3_G3","CROSS_PLAY"),sets)["CROSS_PLAY"]["marginal_unique_states"]==1
def test_terminal_z_and_truncated_z_contract():
 p=PendingSelfPlayStep(RawSongoState((5,)*14+(0,0),1),(True,)*7,(1,)*7,(1/7,)*7,(1/7,)*7,0,{"game_id":"x","ply":0})
 assert finalize_selfplay_steps([p],status=SelfPlayStatus.TERMINAL_WIN,winner=1,final_score=(40,30),include_truncated_examples=True)[0].value_target==1
 z=finalize_selfplay_steps([p],status=SelfPlayStatus.TRUNCATED_MAX_PLIES,winner=None,final_score=None,include_truncated_examples=True)[0];assert z.value_target is None and z.metadata["terminal_result"] is None
def test_source_balancing_and_action_accounting():
 ids=["G2_G2"]*100+["G3_G3"]*5+["CROSS_PLAY"]*2;idx=source_balanced_indices(ids,30,seed=31);assert {k:sum(ids[i]==k for i in idx) for k in set(ids)}=={"G2_G2":10,"G3_G3":10,"CROSS_PLAY":10}
 assert action_metrics([ex(0,"a","G2_G2",0),ex(1,"b","G2_G2",1)])["counts"]["0"]==1

def test_lot31_configuration_forbids_training_and_model_mutation():
 import ast
 from pathlib import Path
 script=Path("apps/trainer/scripts/run_srn_lot31.py").read_text()
 tree=ast.parse(script)
 forbidden={"backward","step","train","save_checkpoint"}
 calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
 assert forbidden.isdisjoint(calls)
 assert '"training":False' in script and '"teacher_labels":False' in script and '"minimax_labels":False' in script

def test_model_role_and_crossplay_provenance_contract():
 q=(MatchupQuota("G2","G3",ModelRole.CHAMPION,ModelRole.GENERATOR,1,"cross"),)
 game=deterministic_schedule("pool",q,seed=31,mcts_budget=64)[0]
 assert game.p1_role is ModelRole.CHAMPION and game.p2_role is ModelRole.GENERATOR
 assert game.p1_model_id=="G2" and game.p2_model_id=="G3" and game.provenance=="cross"
