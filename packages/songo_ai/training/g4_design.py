"""Contrats purs du Lot 33. Ce module ne contient aucune boucle d'entraînement."""
from __future__ import annotations

class DesignError(ValueError):pass

def _ratio100(values,name):
 if abs(sum(values.values())-100.)>1e-9:raise DesignError(f"{name} must sum to 100")
 if any(v<0 for v in values.values()):raise DesignError(f"{name} cannot contain negative ratios")

def validate_g4_design(design):
 _ratio100(design["generation_ratio"],"generation_ratio");_ratio100(design["training_source_ratio"],"training_source_ratio")
 if design["new_generation_games"]<=0:raise DesignError("new_generation_games must be positive")
 if design["policy_objective"]!="MASKED_SOFT_TARGET_CE_TO_PI_MCTS":raise DesignError("unexpected Policy objective")
 if design["value_objective"]!="MSE_TO_TRUE_TERMINAL_Z_ONLY":raise DesignError("unexpected Value objective")
 if design["max_policy_candidates"]*design["max_value_candidates"]>design["max_policy_value_combinations"]:raise DesignError("component matrix is undersized")
 if design["training"]["control_updates"]!=design["training"]["pool_updates"]:raise DesignError("causal arms require equal updates")
 if set(design["opponents"])!={"G2","G3_VALUE_REWORK"}:raise DesignError("fixed opponent battery mismatch")
 if design["official_champion"]!="G2" or design["g3_promoted"]:raise DesignError("official status changed")
 return True

def promotion_decision(metrics):
 """Règle générique préenregistrable pour G4 et générations suivantes."""
 champion=metrics["champion_pooled_score"]>=.52 and metrics["champion_bootstrap_p_gt_50"]>=.95
 population=metrics["population_score"]>=.48 and metrics["population_ci_low"]>=.42
 sides=metrics["max_side_gap"]<=.10
 search=metrics["max_replicated_budget_drop"]<=.07 and not metrics["search_collapse_reproduced"]
 safe=not metrics["data_or_model_pathology"] and metrics["strategic_preservation"]>=.85
 return {"promoted":champion and population and sides and search and safe,"champion_progression":champion,"population_robustness":population,"side_balance":sides,"search_robustness":search,"safety":safe}

def generator_admission_decision(metrics):
 strong_enough=metrics["champion_pooled_score"]>=.42
 diverse=metrics["strategic_diversity_gain"]>=.05 or metrics["marginal_data_gain"]>=.02
 safe=not metrics["data_or_model_pathology"] and metrics["compute_cost_ratio"]<=2.
 return {"admitted":strong_enough and diverse and safe,"sufficient_strength":strong_enough,"useful_diversity":diverse,"safe_and_affordable":safe}
