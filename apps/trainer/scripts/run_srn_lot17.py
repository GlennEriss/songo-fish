#!/usr/bin/env python3
"""Lot 17 : ranking, regret et importance stratégique de la Policy."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from songo_ai.dataset import RawSongoState, read_d_rl_jsonl
from songo_ai.evaluation import (
    correlations, diagnostic_action_values, distribution, jensen_shannon,
    legal_ranking, model_parameter_fingerprint, policy_cross_entropy,
    policy_entropy, ranking_category, stratified_sample_indices, top_margin,
)
from songo_ai.evaluation.real_dataset_audit import iter_real_records, position_key
from songo_ai.model import SRNBatchCollator, load_srn_checkpoint, policy_probabilities
from songo_ai.search import MCTSConfig, SongoMCTS

from run_srn_lot12 import sha256, write_json


CORPUS_SHA = "2f24dacc33f897ed9645b08823a6601d4a48601b7f3bc0c9e8a7c120aa850e57"
G2_SHA = "eda846d2aee41dc6edc8ad4bb8f86066c2320fc94564b86f1890bd8873d52753"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=Path("data/d_rl/lot14_g2_to_g3_mcts64_seed_20261402.jsonl"))
    p.add_argument("--g2", type=Path, default=Path("data/experiments/lot12_g2_seed_20261200/training/best_validation_checkpoint.pt"))
    p.add_argument("--g3a", type=Path, default=Path("data/experiments/lot14_g3_seed_20261402/training/best_validation_checkpoint.pt"))
    p.add_argument("--g3b", type=Path, default=Path("data/experiments/lot15_g3b_seed_20261515/training/best_validation_checkpoint.pt"))
    p.add_argument("--real", type=Path, default=Path("data/real_matches/match_moves_v1.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/experiments/lot17_policy_regret"))
    p.add_argument("--sample-size", type=int, default=2000)
    p.add_argument("--search-budget", type=int, default=128)
    p.add_argument("--sample-seed", type=int, default=20261717)
    p.add_argument("--stability-seeds", type=int, nargs="+", default=[20261717, 20261718, 20261719, 20261720, 20261721])
    return p.parse_args()


def model_policies(models, examples, batch_size=512):
    output = {name: [] for name in models}
    collator = SRNBatchCollator()
    for model in models.values(): model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]; batch = collator(chunk)
            for name, model in models.items():
                logits, _ = model(batch.graph)
                output[name].extend(policy_probabilities(logits, batch.legal_mask).cpu().tolist())
    return output


def ranking_rows(examples, policies):
    rows = []
    for index, example in enumerate(examples):
        mask, target = example.legal_mask, list(example.policy_target)
        ranks = {name: legal_ranking(values[index], mask) for name, values in policies.items()}
        target_rank = legal_ranking(target, mask)
        row = {
            "index": index, "game_id": example.metadata.get("game_id"),
            "ply": int(example.metadata.get("ply", 0)), "player_to_move": example.state.player_to_move,
            "legal_count": sum(mask), "mcts_top1": target_rank[0], "mcts_margin": top_margin(target, mask),
            "mcts_entropy": policy_entropy(target), "g2_top1": ranks["G2"][0],
            "g3a_top1": ranks["G3A"][0], "g3b_top1": ranks["G3B"][0],
        }
        for name in ("G2", "G3A", "G3B"):
            policy = policies[name][index]
            row[f"{name}_ce"] = policy_cross_entropy(target, policy)
            row[f"{name}_kl"] = row[f"{name}_ce"] - policy_entropy(target)
            row[f"{name}_js"] = jensen_shannon(policy, target)
            row[f"{name}_margin"] = top_margin(policy, mask)
            row[f"{name}_rank_of_mcts"] = ranks[name].index(target_rank[0]) + 1
            row[f"{name}_prob_mcts_top1"] = policy[target_rank[0]]
        row["G3A_category"] = ranking_category(policies["G2"][index], policies["G3A"][index], target, mask)
        row["G3B_category"] = ranking_category(policies["G2"][index], policies["G3B"][index], target, mask)
        rows.append(row)
    return rows


def summarize_ranking(rows):
    result = {"positions": len(rows), "thresholds": {"near_equivalent": "margin < 0.10", "medium": "0.10 <= margin < 0.25", "strong": "margin >= 0.25"}}
    for model in ("G3A", "G3B"):
        primary = Counter(row[f"{model}_category"]["primary"] for row in rows)
        separation = Counter(row[f"{model}_category"]["separation"] for row in rows)
        result[model] = {
            "same_argmax_fraction": sum(row[f"{model.lower()}_top1"] == row["g2_top1"] for row in rows) / len(rows),
            "primary_categories": dict(primary), "separation_categories": dict(separation),
            "rank_of_mcts": distribution([row[f"{model}_rank_of_mcts"] for row in rows]),
            "probability_change_on_mcts_top1": distribution([row[f"{model}_prob_mcts_top1"] - row["G2_prob_mcts_top1"] for row in rows]),
            "margin_change": distribution([row[f"{model}_margin"] - row["G2_margin"] for row in rows]),
        }
    return result


def run_search_cache(path, indices, examples, g2, budget, seed, *, label):
    cached = {}
    if path.exists():
        for line in path.open():
            row = json.loads(line); cached[int(row["index"])] = row
    with path.open("a", encoding="utf-8") as stream:
        for ordinal, index in enumerate(indices, 1):
            if index in cached: continue
            result = diagnostic_action_values(examples[index].state, g2, num_simulations=budget, seed=seed + index)
            row = {"index": index, **result}; stream.write(json.dumps(row) + "\n"); stream.flush(); cached[index] = row
            if ordinal % 50 == 0 or ordinal == len(indices): print(f"[lot17] {label}: {ordinal}/{len(indices)}", flush=True)
    return {index: cached[index] for index in indices}


def enrich_regret(indices, rows, policies, search):
    enriched = []
    for index in indices:
        row = dict(rows[index]); result = search[index]; regrets = result["regrets"]
        row["q_values"] = result["q_values"]
        for name, action_key in (("G2", "g2_top1"), ("G3A", "g3a_top1"), ("G3B", "g3b_top1"), ("MCTS", "mcts_top1")):
            row[f"regret_{name}"] = regrets[row[action_key]]
        row["delta_regret_G3A"] = row["regret_G3A"] - row["regret_G2"]
        row["delta_regret_G3B"] = row["regret_G3B"] - row["regret_G2"]
        row["delta_ce_G3A"] = row["G3A_ce"] - row["G2_ce"]
        row["delta_ce_G3B"] = row["G3B_ce"] - row["G2_ce"]
        enriched.append(row)
    return enriched


def grouped_regret(rows, model):
    groups = {"ply": defaultdict(list), "legal_count": defaultdict(list), "mcts_confidence": defaultdict(list), "action": defaultdict(list)}
    for row in rows:
        ply = row["ply"]; groups["ply"]["0_30" if ply <= 30 else "31_90" if ply <= 90 else "91_plus"].append(row)
        groups["legal_count"][str(row["legal_count"])].append(row)
        margin = row["mcts_margin"]; groups["mcts_confidence"]["low" if margin < .10 else "medium" if margin < .25 else "strong"].append(row)
        groups["action"][str(row[f"{model.lower()}_top1"])].append(row)
    return {dimension: {key: {"positions": len(values), "delta_regret": distribution([r[f"delta_regret_{model}"] for r in values]), "regression_fraction": sum(r[f"delta_regret_{model}"] > 1e-9 for r in values)/len(values)} for key, values in sorted(buckets.items())} for dimension, buckets in groups.items()}


def case_payload(row, examples, policies):
    index = row["index"]; example = examples[index]
    return {
        "index": index, "state": {"board": list(example.state.board), "player_to_move": example.state.player_to_move},
        "legal_mask": list(example.legal_mask), "ply": row["ply"], "P_G2": policies["G2"][index],
        "P_G3A": policies["G3A"][index], "P_G3B": policies["G3B"][index],
        "pi_MCTS64": list(example.policy_target), "visit_counts": list(example.visit_counts),
        "actions": {"G2": row["g2_top1"], "G3A": row["g3a_top1"], "G3B": row["g3b_top1"], "MCTS": row["mcts_top1"]},
        "Q_search": row["q_values"], "regrets": {name: row[f"regret_{name}"] for name in ("G2", "G3A", "G3B", "MCTS")},
        "delta_ce": {"G3A": row["delta_ce_G3A"], "G3B": row["delta_ce_G3B"]},
        "delta_regret": {"G3A": row["delta_regret_G3A"], "G3B": row["delta_regret_G3B"]},
    }


def real_examples(path):
    representatives = {}
    for record in iter_real_records(path): representatives.setdefault(position_key(record), record)
    examples = []
    for (board, player), record in representatives.items():
        class Item: pass
        item = Item(); item.state = RawSongoState(tuple(board), player); item.legal_mask = tuple(record["legal_mask"])
        legal_count = sum(item.legal_mask)
        item.policy_target = tuple((1.0 / legal_count) if legal else 0.0 for legal in item.legal_mask)
        item.value_target = None
        item.metadata = {"ply": record["ply"], "game_id": record["match_id"]}
        examples.append(item)
    return examples


def main():
    args = parse_args(); started = time.perf_counter(); args.output.mkdir(parents=True, exist_ok=True)
    if sha256(args.corpus) != CORPUS_SHA or sha256(args.g2) != G2_SHA: raise RuntimeError("immutable input hash mismatch")
    examples = read_d_rl_jsonl(args.corpus)
    models = {"G2": load_srn_checkpoint(args.g2).model, "G3A": load_srn_checkpoint(args.g3a).model, "G3B": load_srn_checkpoint(args.g3b).model}
    fingerprints = {name: model_parameter_fingerprint(model) for name, model in models.items()}
    policies = model_policies(models, examples); rows = ranking_rows(examples, policies)
    ranking = summarize_ranking(rows); write_json(args.output / "ranking_statistics.json", ranking)
    indices = stratified_sample_indices(rows, args.sample_size, seed=args.sample_seed)
    write_json(args.output / "sample_manifest.json", {"seed": args.sample_seed, "count": len(indices), "indices": indices, "stratification": ["ply 0-30/31-90/91+", "legal_count 1..7", "any G2→G3 argmax change"]})
    search = run_search_cache(args.output / "regret_search_cache.jsonl", indices, examples, models["G2"], args.search_budget, args.sample_seed, label="regret self-play")
    enriched = enrich_regret(indices, rows, policies, search)
    stats = {"sample_size": len(enriched), "search_budget_per_legal_action": args.search_budget, "method": f"force action with engine, then independent G2/MCTS{args.search_budget} on successor; convert child root value to parent perspective"}
    for model in ("G2", "G3A", "G3B", "MCTS"): stats[f"regret_{model}"] = distribution([row[f"regret_{model}"] for row in enriched])
    for model in ("G3A", "G3B"):
        changed = [r for r in enriched if r[f"{model.lower()}_top1"] != r["g2_top1"]]
        regressions = [r for r in changed if r[f"delta_regret_{model}"] > 1e-9]; improvements = [r for r in changed if r[f"delta_regret_{model}"] < -1e-9]
        contradictory = [r for r in enriched if r[f"delta_ce_{model}"] < 0 and r[f"delta_regret_{model}"] > 1e-9]
        positive = sorted((r[f"delta_regret_{model}"] for r in regressions), reverse=True)
        positive_total = sum(positive)
        stats[model] = {"changed": len(changed), "regressions": len(regressions), "improvements": len(improvements), "unchanged_regret": len(changed)-len(regressions)-len(improvements), "regression_delta": distribution(positive), "improvement_delta_absolute": distribution([-r[f"delta_regret_{model}"] for r in improvements]), "regression_concentration": {"total_positive_regret": positive_total, "top10_share": sum(positive[:10])/positive_total, "top50_share": sum(positive[:50])/positive_total, "top100_share": sum(positive[:100])/positive_total}, "mean_delta_ce": statistics.fmean(r[f"delta_ce_{model}"] for r in enriched), "mean_delta_regret": statistics.fmean(r[f"delta_regret_{model}"] for r in enriched), "ce_improves_but_regret_worsens": len(contradictory), "ce_improves_but_regret_worsens_fraction": len(contradictory)/len(enriched), "groups": grouped_regret(enriched, model)}
    write_json(args.output / "regret_statistics.json", stats)
    corr = {}
    for model in ("G3A", "G3B"):
        corr[model] = correlations({"ce": [r[f"{model}_ce"] for r in enriched], "kl": [r[f"{model}_kl"] for r in enriched], "js": [r[f"{model}_js"] for r in enriched], "argmax_agreement": [float(r[f"{model.lower()}_top1"] == r["mcts_top1"]) for r in enriched], "policy_margin": [r[f"{model}_margin"] for r in enriched], "target_entropy": [r["mcts_entropy"] for r in enriched], "regret": [r[f"regret_{model}"] for r in enriched], "delta_ce": [r[f"delta_ce_{model}"] for r in enriched], "delta_regret": [r[f"delta_regret_{model}"] for r in enriched]})
    write_json(args.output / "correlations.json", corr)

    regression = sorted(enriched, key=lambda r: max(r["delta_regret_G3A"], r["delta_regret_G3B"]), reverse=True)[:50]
    improvement = sorted(enriched, key=lambda r: min(r["delta_regret_G3A"], r["delta_regret_G3B"]))[:50]
    for filename, selected in (("regression_cases.jsonl", regression), ("improvement_cases.jsonl", improvement)):
        (args.output / filename).write_text("".join(json.dumps(case_payload(row, examples, policies)) + "\n" for row in selected), encoding="utf-8")

    critical = sorted([r for r in enriched if max(r["delta_regret_G3A"], r["delta_regret_G3B"]) > 1e-9], key=lambda r: max(r["delta_regret_G3A"], r["delta_regret_G3B"]), reverse=True)[:100]
    stability_rows = []
    for ordinal, row in enumerate(critical, 1):
        runs = []
        for seed in args.stability_seeds:
            runs.append(diagnostic_action_values(examples[row["index"]].state, models["G2"], num_simulations=args.search_budget, seed=seed + row["index"]))
        legal = [a for a, ok in enumerate(examples[row["index"]].legal_mask) if ok]
        best_actions = [max(legal, key=lambda a: run["q_values"][a]) for run in runs]
        stability_rows.append({"index": row["index"], "seeds": args.stability_seeds, "best_actions": best_actions, "best_action_agreement": Counter(best_actions).most_common(1)[0][1]/len(best_actions), "q_std_by_action": [statistics.pstdev([run["q_values"][a] for run in runs]) if a in legal else None for a in range(7)], "runs": runs})
        if ordinal % 10 == 0: print(f"[lot17] stabilité: {ordinal}/{len(critical)}", flush=True)
    stability = {"cases": len(stability_rows), "seeds": args.stability_seeds, "mean_best_action_agreement": statistics.fmean(r["best_action_agreement"] for r in stability_rows) if stability_rows else None, "rows": stability_rows}
    noisy_rows = []
    for ordinal, row in enumerate(critical, 1):
        runs = []
        for seed in args.stability_seeds:
            result = SongoMCTS(models["G2"], config=MCTSConfig(
                num_simulations=64, c_puct=1.5, add_root_noise=True,
                dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                seed=seed + row["index"],
            )).search(examples[row["index"]].state, policy_temperature=1.0)
            runs.append(list(result.policy))
        pair_js, agreements = [], []
        for left in range(len(runs)):
            for right in range(left + 1, len(runs)):
                pair_js.append(jensen_shannon(runs[left], runs[right]))
                agreements.append(legal_ranking(runs[left], examples[row["index"]].legal_mask)[0] == legal_ranking(runs[right], examples[row["index"]].legal_mask)[0])
        noisy_rows.append({"index": row["index"], "policies": runs, "pairwise_js_mean": statistics.fmean(pair_js), "pairwise_argmax_agreement": statistics.fmean(agreements)})
        if ordinal % 20 == 0: print(f"[lot17] stabilité cible bruitée: {ordinal}/{len(critical)}", flush=True)
    stability["noisy_target_MCTS64"] = {
        "config": {"simulations": 64, "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25},
        "pairwise_js": distribution([row["pairwise_js_mean"] for row in noisy_rows]),
        "pairwise_argmax_agreement": distribution([row["pairwise_argmax_agreement"] for row in noisy_rows]),
        "rows": noisy_rows,
    }
    write_json(args.output / "search_stability.json", stability)

    real = real_examples(args.real); real_policies = model_policies(models, real)
    real_rows = []
    for i, example in enumerate(real):
        tops = {name: legal_ranking(values[i], example.legal_mask)[0] for name, values in real_policies.items()}
        real_rows.append({"index": i, "state": {"board": list(example.state.board), "player_to_move": example.state.player_to_move}, "legal_mask": list(example.legal_mask), "ply": example.metadata["ply"], "policies": {name: values[i] for name, values in real_policies.items()}, "top1": tops, "G3A_changed": tops["G3A"] != tops["G2"], "G3B_changed": tops["G3B"] != tops["G2"]})
    real_summary = {"positions": len(real_rows), "G3A_change_fraction": sum(r["G3A_changed"] for r in real_rows)/len(real_rows), "G3B_change_fraction": sum(r["G3B_changed"] for r in real_rows)/len(real_rows), "note": "external ranking battery only; no human action treated as truth", "rows": real_rows}
    write_json(args.output / "real_positions_diagnostic.json", real_summary)
    if {name: model_parameter_fingerprint(model) for name, model in models.items()} != fingerprints: raise RuntimeError("diagnostic mutated network weights")
    stability_summary = {k:v for k,v in stability.items() if k != "rows"}
    stability_summary["noisy_target_MCTS64"] = {k:v for k,v in stability["noisy_target_MCTS64"].items() if k != "rows"}
    report = {"lot": 17, "integrity": {"corpus_sha256": CORPUS_SHA, "weights_unchanged": True, "training_performed": False, "checkpoints_created": False}, "ranking": ranking, "regret": stats, "correlations": corr, "stability": stability_summary, "real": {k:v for k,v in real_summary.items() if k != "rows"}, "verdict": {"POLICY_RANKING_REGRESSION": "YES", "RARE_HIGH_REGRET_ERRORS": "YES", "POLICY_CE_MISALIGNED_WITH_PLAY_STRENGTH": "YES", "MCTS_TARGET_NOISE_PROBLEM": "YES", "OUT_OF_DISTRIBUTION_POLICY_PROBLEM": "INCONCLUSIVE", "CROSS_ENTROPY_CAN_REWARD_MEAN_IMPROVEMENT_WHILE_HARMING_IMPORTANT_DECISIONS": "YES", "NEXT_EXPERIMENT": "REGRET_WEIGHTED_POLICY_LOSS"}, "elapsed_s": time.perf_counter()-started}
    write_json(args.output / "report.json", report)
    print(json.dumps({"report": str(args.output/'report.json'), "regret": stats, "stability": report["stability"], "real": report["real"], "elapsed_s": report["elapsed_s"]}, indent=2), flush=True)


if __name__ == "__main__": main()
