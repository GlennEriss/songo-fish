"""Negamax/Alpha-Beta avec TT, approfondissement iteratif et PV (etape 3).

Porte de passage (section 13) : TT, ID, ordonnancement et PV fonctionnels.
L'evaluation de feuille est un heuristique provisoire (difference de
magasins + mobilite) : elle sera remplacee par la tete WDL du reseau une
fois l'etape 7 atteinte (section 8.1). Ce module ne depend que de
`songo_ai.songo.rules`, seule implementation des regles du projet
(principe section 12.1).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from songo_ai.songo.rules import SongoLegacyGame, opponent

EXACT = "EXACT"
LOWER = "LOWER"
UPPER = "UPPER"


class SearchAborted(Exception):
    pass


@dataclass
class SearchLimits:
    max_depth: int = 8
    max_nodes: int = 200_000
    max_time_s: float = 2.0


@dataclass
class TTEntry:
    depth: int
    score: float
    flag: str
    best_local_action: Optional[int]


@dataclass
class SearchResult:
    local_action: int
    score: float
    depth_reached: int
    nodes: int
    time_s: float
    pv: List[int] = field(default_factory=list)
    timed_out: bool = False


def default_evaluate(game: SongoLegacyGame, perspective: int) -> float:
    """Heuristique provisoire : diff. de magasins (poids fort) + mobilite."""
    if game.finished:
        score_1, score_2 = game.final_score_with_territory()
        diff = (score_1 - score_2) if perspective == 1 else (score_2 - score_1)
        return diff * 1000.0

    store_1, store_2 = game.score()
    diff = (store_1 - store_2) if perspective == 1 else (store_2 - store_1)
    mobility = len(game.legal_moves(perspective)) - len(game.legal_moves(opponent(perspective)))
    return diff * 10.0 + mobility


def _expand_children(
    game: SongoLegacyGame, legal_actions: List[int], tt_entry: Optional[TTEntry]
) -> List[Tuple[int, SongoLegacyGame]]:
    """Genere chaque enfant une seule fois et l'ordonne (TT move, puis capture
    immediate decroissante) : evite de rejouer deux fois le meme coup
    (une fois pour trier, une fois pour la recherche)."""
    tt_move = tt_entry.best_local_action if tt_entry is not None else None
    own_total_before = sum(game.score())

    pairs: List[Tuple[int, SongoLegacyGame, int]] = []
    for local_action in legal_actions:
        child = game.clone_for_search()
        child.play_local(local_action)
        capture_gain = sum(child.score()) - own_total_before
        pairs.append((local_action, child, capture_gain))

    pairs.sort(key=lambda item: item[2], reverse=True)
    ordered = [(local_action, child) for local_action, child, _ in pairs]
    if tt_move is not None:
        for i, (local_action, _) in enumerate(ordered):
            if local_action == tt_move and i != 0:
                ordered.insert(0, ordered.pop(i))
                break
    return ordered


def negamax_search(
    game: SongoLegacyGame,
    depth: int,
    alpha: float,
    beta: float,
    perspective: int,
    tt: Dict[int, TTEntry],
    node_counter: List[int],
    node_limit: int,
    deadline: float,
) -> Tuple[float, List[int]]:
    node_counter[0] += 1
    if node_counter[0] > node_limit or time.perf_counter() > deadline:
        raise SearchAborted()

    key = game.zobrist_hash()
    tt_entry = tt.get(key)
    original_alpha = alpha
    if tt_entry is not None and tt_entry.depth >= depth:
        if tt_entry.flag == EXACT:
            pv = [tt_entry.best_local_action] if tt_entry.best_local_action is not None else []
            return tt_entry.score, pv
        if tt_entry.flag == LOWER:
            alpha = max(alpha, tt_entry.score)
        elif tt_entry.flag == UPPER:
            beta = min(beta, tt_entry.score)
        if alpha >= beta:
            pv = [tt_entry.best_local_action] if tt_entry.best_local_action is not None else []
            return tt_entry.score, pv

    if game.finished or depth == 0:
        return default_evaluate(game, perspective), []

    legal_actions = game.legal_local_actions()
    if not legal_actions:
        game.normalize_terminal()
        return default_evaluate(game, perspective), []

    ordered_children = _expand_children(game, legal_actions, tt_entry)
    best_score = -math.inf
    best_local: Optional[int] = None
    best_pv: List[int] = []

    for local_action, child in ordered_children:
        score, child_pv = negamax_search(
            child, depth - 1, -beta, -alpha, opponent(perspective), tt, node_counter, node_limit, deadline
        )
        score = -score
        if score > best_score:
            best_score = score
            best_local = local_action
            best_pv = [local_action] + child_pv
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    flag = EXACT
    if best_score <= original_alpha:
        flag = UPPER
    elif best_score >= beta:
        flag = LOWER
    tt[key] = TTEntry(depth, best_score, flag, best_local)
    return best_score, best_pv


def iterative_deepening(game: SongoLegacyGame, limits: SearchLimits) -> SearchResult:
    """Approfondissement iteratif : renvoie toujours la derniere profondeur
    completement terminee (fallback de securite, cf. section 8.2/Annexe B).
    """
    legal_actions = game.legal_local_actions()
    if not legal_actions:
        raise ValueError("no legal move at root: caller must check game.finished first")

    tt: Dict[int, TTEntry] = {}
    node_counter = [0]
    start = time.perf_counter()
    deadline = start + limits.max_time_s
    perspective = game.turn

    best_completed = SearchResult(
        local_action=legal_actions[0], score=0.0, depth_reached=0, nodes=0, time_s=0.0, pv=[], timed_out=False
    )

    for depth in range(1, limits.max_depth + 1):
        try:
            score, pv = negamax_search(game, depth, -math.inf, math.inf, perspective, tt, node_counter, limits.max_nodes, deadline)
        except SearchAborted:
            best_completed.timed_out = True
            break

        best_completed = SearchResult(
            local_action=pv[0] if pv else legal_actions[0],
            score=score,
            depth_reached=depth,
            nodes=node_counter[0],
            time_s=time.perf_counter() - start,
            pv=pv,
            timed_out=False,
        )

        if node_counter[0] >= limits.max_nodes or time.perf_counter() >= deadline:
            break

    return best_completed
