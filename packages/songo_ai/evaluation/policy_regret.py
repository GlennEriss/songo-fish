"""Ranking Policy et regret de recherche pour le diagnostic Lot 17."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import defaultdict
from typing import Mapping, Sequence

from songo_ai.dataset import RawSongoState
from songo_ai.search import MCTSConfig, SongoMCTS, convert_value_perspective, terminal_value
from songo_ai.songo.rules import SongoLegacyGame


def legal_ranking(policy: Sequence[float], legal_mask: Sequence[bool]) -> tuple[int, ...]:
    legal = [action for action, allowed in enumerate(legal_mask) if allowed]
    if not legal:
        raise ValueError("at least one legal action is required")
    return tuple(sorted(legal, key=lambda action: (-float(policy[action]), action)))


def top_margin(policy: Sequence[float], legal_mask: Sequence[bool]) -> float:
    ranking = legal_ranking(policy, legal_mask)
    return float(policy[ranking[0]] - policy[ranking[1]]) if len(ranking) > 1 else 1.0


def ranking_category(
    parent_policy: Sequence[float], child_policy: Sequence[float],
    target: Sequence[float], legal_mask: Sequence[bool],
) -> dict:
    parent_top = legal_ranking(parent_policy, legal_mask)[0]
    child_top = legal_ranking(child_policy, legal_mask)[0]
    target_rank = legal_ranking(target, legal_mask)
    target_top = target_rank[0]
    if parent_top == child_top:
        primary = "A_same_top1"
    elif child_top == target_top and parent_top != target_top:
        primary = "B_change_to_mcts_top1"
    elif parent_top == target_top and child_top != target_top:
        primary = "C_leave_mcts_top1"
    else:
        primary = "D_neither_is_mcts_top1"
    margin = top_margin(target, legal_mask)
    separation = "E_near_equivalent" if margin < 0.10 else "F_strongly_separated" if margin >= 0.25 else "medium_separation"
    return {
        "primary": primary, "separation": separation, "mcts_margin": margin,
        "parent_top1": parent_top, "child_top1": child_top, "mcts_top1": target_top,
        "parent_rank_of_mcts": target_rank.index(parent_top) + 1,
        "child_rank_of_mcts": target_rank.index(child_top) + 1,
    }


def policy_cross_entropy(target: Sequence[float], policy: Sequence[float]) -> float:
    return -sum(float(t) * math.log(max(float(p), 1e-12)) for t, p in zip(target, policy) if t > 0)


def action_regrets(q_values: Sequence[float], legal_mask: Sequence[bool]) -> tuple[float | None, ...]:
    legal = [action for action, allowed in enumerate(legal_mask) if allowed]
    if not legal:
        raise ValueError("at least one legal action is required")
    best = max(float(q_values[action]) for action in legal)
    return tuple(best - float(q_values[action]) if action in legal else None for action in range(7))


def diagnostic_action_values(
    state: RawSongoState, model, *, num_simulations: int = 128, seed: int = 0,
) -> dict:
    """Évalue chaque action après transition avec un budget MCTS identique."""

    game = SongoLegacyGame.from_state(state.to_engine_state())
    legal = game.legal_local_actions()
    q_values: list[float | None] = [None] * 7
    evaluations = simulations = 0
    for action in legal:
        child = SongoLegacyGame.from_state(state.to_engine_state())
        child.play_local(action); child.normalize_terminal()
        if child.finished:
            q_values[action] = terminal_value(child.winner, state.player_to_move)
            continue
        derived = int.from_bytes(
            hashlib.sha256(f"{seed}:{action}".encode()).digest()[:8], "big"
        )
        result = SongoMCTS(model, config=MCTSConfig(
            num_simulations=num_simulations, c_puct=1.5,
            add_root_noise=False, seed=derived,
        )).search(RawSongoState.from_game(child), policy_temperature=0.0)
        q_values[action] = convert_value_perspective(
            result.root_value, child.turn, state.player_to_move
        )
        evaluations += result.network_evaluations
        simulations += result.num_simulations
    numeric = [float(value) if value is not None else 0.0 for value in q_values]
    return {
        "q_values": q_values,
        "regrets": action_regrets(numeric, tuple(action in legal for action in range(7))),
        "num_simulations_per_nonterminal_action": num_simulations,
        "total_simulations": simulations,
        "network_evaluations": evaluations,
    }


def stratified_sample_indices(rows: Sequence[Mapping], count: int, *, seed: int) -> list[int]:
    """Échantillon reproductible équilibré sur ply, branching et changement."""

    if count <= 0 or count > len(rows):
        raise ValueError("count must be in 1..len(rows)")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        ply = int(row["ply"])
        ply_group = "0_30" if ply <= 30 else "31_90" if ply <= 90 else "91_plus"
        changed = bool(row["g3a_top1"] != row["g2_top1"] or row["g3b_top1"] != row["g2_top1"])
        groups[(ply_group, int(row["legal_count"]), changed)].append(index)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected, keys = [], sorted(groups, key=repr)
    cursor = 0
    while len(selected) < count:
        progressed = False
        for key in keys:
            if cursor < len(groups[key]):
                selected.append(groups[key][cursor]); progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        cursor += 1
    return sorted(selected)


def distribution(values: Sequence[float]) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}
    def q(p):
        pos = p * (len(ordered) - 1); lo = int(pos); hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)
    return {"count": len(ordered), "mean": statistics.fmean(ordered), "median": statistics.median(ordered), "p75": q(.75), "p90": q(.90), "p95": q(.95), "maximum": max(ordered)}


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values); start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]: ranks[index] = rank
        start = end
    return ranks


def pearson(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    mx, my = statistics.fmean(first), statistics.fmean(second)
    dx, dy = [x - mx for x in first], [y - my for y in second]
    denom = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    return sum(x*y for x, y in zip(dx, dy)) / denom if denom else None


def correlations(columns: Mapping[str, Sequence[float]]) -> dict:
    result = {}
    for left, left_values in columns.items():
        result[left] = {}
        for right, right_values in columns.items():
            result[left][right] = {
                "pearson": pearson(left_values, right_values),
                "spearman": pearson(_ranks(left_values), _ranks(right_values)),
            }
    return result
