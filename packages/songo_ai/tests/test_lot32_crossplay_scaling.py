import ast
import sys
from pathlib import Path
sys.path.insert(0,str(Path("apps/trainer/scripts").resolve()))
from run_srn_lot32 import ARM_NAMES,SEED,bootstrap,bootstrap_difference,quotas,schedules

def test_exact_three_arm_schedule_ratios_and_side_balance():
 rows=schedules();assert set(rows)==set(ARM_NAMES) and all(len(x)==2000 for x in rows.values())
 assert sum(x.p1_model_id==x.p2_model_id=="G2" for x in rows["CONTROL"])==2000
 for arm,g2,g3,cross in (("ORIGINAL_POOL",500,500,1000),("CROSSPLAY_ENRICHED",200,200,1600)):
  xs=rows[arm];assert sum(x.p1_model_id==x.p2_model_id=="G2" for x in xs)==g2;assert sum(x.p1_model_id==x.p2_model_id=="G3_VALUE_REWORK" for x in xs)==g3
  assert sum(x.p1_model_id=="G2" and x.p2_model_id=="G3_VALUE_REWORK" for x in xs)==cross//2
  assert sum(x.p1_model_id=="G3_VALUE_REWORK" and x.p2_model_id=="G2" for x in xs)==cross//2

def test_lot32_seeds_are_independent_and_deterministic():
 first=schedules();second=schedules();assert [[x.seed for x in first[a]] for a in ARM_NAMES]==[[x.seed for x in second[a]] for a in ARM_NAMES]
 lot31_seed=20263131;assert SEED!=lot31_seed and not ({x.seed for a in ARM_NAMES for x in first[a]} & {lot31_seed})

def test_equal_search_configuration_and_no_training_calls():
 script=Path("apps/trainer/scripts/run_srn_lot32.py").read_text();tree=ast.parse(script);calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
 assert {"backward","step","train","save_checkpoint"}.isdisjoint(calls);assert '"training":False' in script
 assert all(q.games>0 for arm in ARM_NAMES for q in quotas(arm))

def test_bootstrap_is_game_level_and_has_10000_replicates():
 class X:
  def __init__(self,key):self.state=type("S",(),{"board":tuple([key]+[0]*15),"player_to_move":1})()
 games={str(i):[X(i)] for i in range(5)};result=bootstrap(games,set(),replicates=10000,seed=1)
 assert result["replicates"]==10000 and result["unit"]=="game" and len(result["coverage"]["ci95"])==2
 diff=bootstrap_difference(games,games,set(),replicates=10000,seed=2)
 assert diff["replicates"]==10000 and len(diff["coverage_relative_difference"]["ci95"])==2
