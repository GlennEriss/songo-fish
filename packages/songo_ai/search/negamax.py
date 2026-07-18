"""Negamax/Alpha-Beta avec TT, approfondissement iteratif et PV (etape 3).

Porte de passage (section 13) : TT, ID, ordonnancement et PV fonctionnels.
L'evaluation de feuille est un heuristique provisoire (difference de
magasins + mobilite) : elle sera remplacee par la tete WDL du reseau une
fois l'etape 7 atteinte (section 8.1). Ce module ne depend que de
`songo_ai.songo.rules`, seule implementation des regles du projet
(principe section 12.1).

Ameliorations de recherche (section 8.1 : "coup de TT, policy, captures et
heuristiques" ; 8.2 ; quiescence) : killer moves, history heuristic, PVS et
fenetres d'aspiration sont actives par `iterative_deepening` uniquement --
`negamax_search` appele directement avec ses parametres par defaut (cas du
professeur DeepTeacher) garde un comportement strictement identique a
l'etape 3, pour que les annotations de dataset restent comparables d'un
palier a l'autre. La quiescence est opt-in via
`SearchLimits.quiescence_depth` (0 = desactivee).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from songo_ai.songo.rules import SongoLegacyGame, opponent

EvaluateFn = Callable[[SongoLegacyGame, int], float]
PriorityFn = Callable[[SongoLegacyGame, List[int]], Dict[int, float]]

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
    # Profondeur max de la recherche de quiescence aux feuilles (0 =
    # desactivee, comportement historique). Ne prolonge que les coups de
    # capture : corrige l'effet d'horizon (couper l'evaluation au milieu
    # d'une sequence de recoltes).
    quiescence_depth: int = 0


# Largeur de la fenetre d'aspiration autour du score de l'iteration
# precedente (echelle heuristique : ~5 graines). Fenetre ratee -> nouvelle
# recherche pleine largeur, la correction est donc toujours exacte.
ASPIRATION_DELTA = 50.0

# Les scores etant des flottants (tete WDL du reseau), la "fenetre nulle"
# de PVS a une largeur epsilon plutot que 1.
NULL_WINDOW = 1e-6


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
    game: SongoLegacyGame,
    legal_actions: List[int],
    tt_entry: Optional[TTEntry],
    priority_fn: Optional[PriorityFn] = None,
    killers: Optional[List[int]] = None,
    history: Optional[Dict[int, float]] = None,
) -> List[Tuple[int, SongoLegacyGame]]:
    """Genere chaque enfant une seule fois et l'ordonne : evite de rejouer
    deux fois le meme coup (une fois pour trier, une fois pour la
    recherche). Ordre par defaut : TT move, puis capture immediate
    decroissante. Si `priority_fn` est fourni (section 8.1 : tete policy
    du reseau), il remplace le tri par capture -- reste TT-move-first dans
    tous les cas (la TT est toujours la meilleure information disponible,
    reseau ou pas). `killers` (coups ayant produit une coupure beta a la
    meme profondeur d'arbre) et `history` (frequence des coupures toutes
    profondeurs confondues) ne departagent que les coups a score de base
    egal : mesure sur bench_search.py, les faire passer DEVANT le tri par
    capture degrade l'elagage (~+26% de noeuds) -- au Songo les captures
    sont la tactique dominante, le tri de base est deja tres informatif."""
    tt_move = tt_entry.best_local_action if tt_entry is not None else None

    children: Dict[int, SongoLegacyGame] = {}
    for local_action in legal_actions:
        child = game.clone_for_search()
        child.play_local(local_action)
        children[local_action] = child

    if priority_fn is not None:
        base = priority_fn(game, legal_actions)
    else:
        own_total_before = sum(game.score())
        base = {a: float(sum(children[a].score()) - own_total_before) for a in legal_actions}

    killer_set = frozenset(killers) if killers else frozenset()
    hist = history if history is not None else {}
    ordered_actions = sorted(
        legal_actions,
        key=lambda a: (base.get(a, 0.0), a in killer_set, hist.get(a, 0.0)),
        reverse=True,
    )

    ordered = [(a, children[a]) for a in ordered_actions]
    if tt_move is not None:
        for i, (local_action, _) in enumerate(ordered):
            if local_action == tt_move and i != 0:
                ordered.insert(0, ordered.pop(i))
                break
    return ordered


def _quiescence(
    game: SongoLegacyGame,
    alpha: float,
    beta: float,
    perspective: int,
    node_counter: List[int],
    node_limit: int,
    deadline: float,
    evaluate_fn: EvaluateFn,
    qdepth: int,
) -> float:
    """Recherche de quiescence (section 8.1) : a la feuille, prolonge
    uniquement les coups de capture jusqu'a une position "calme" (ou
    l'epuisement de `qdepth`), pour ne jamais evaluer en pleine sequence
    de recoltes (effet d'horizon). Le stand-pat (evaluation immediate)
    borne la recherche : on ne capture que si ca ameliore."""
    node_counter[0] += 1
    if node_counter[0] > node_limit or time.perf_counter() > deadline:
        raise SearchAborted()

    stand_pat = evaluate_fn(game, perspective)
    if game.finished or qdepth <= 0:
        return stand_pat
    if stand_pat >= beta:
        return stand_pat
    alpha = max(alpha, stand_pat)

    legal_actions = game.legal_local_actions()
    if not legal_actions:
        game.normalize_terminal()
        return evaluate_fn(game, perspective)

    mover_index = game.turn - 1
    store_before = game.score()[mover_index]
    best = stand_pat
    for local_action in legal_actions:
        child = game.clone_for_search()
        child.play_local(local_action)
        if child.score()[mover_index] <= store_before and not child.finished:
            continue  # coup calme : ignore en quiescence
        score = -_quiescence(
            child, -beta, -alpha, opponent(perspective), node_counter, node_limit, deadline, evaluate_fn, qdepth - 1
        )
        if score > best:
            best = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


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
    evaluate_fn: EvaluateFn = default_evaluate,
    priority_fn: Optional[PriorityFn] = None,
    ply: int = 0,
    killers: Optional[Dict[int, List[int]]] = None,
    history: Optional[Dict[int, float]] = None,
    use_pvs: bool = False,
    quiescence_depth: int = 0,
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

    if game.finished:
        return evaluate_fn(game, perspective), []
    if depth == 0:
        if quiescence_depth > 0:
            node_counter[0] -= 1  # _quiescence recompte ce noeud
            score = _quiescence(
                game, alpha, beta, perspective, node_counter, node_limit, deadline, evaluate_fn, quiescence_depth
            )
            return score, []
        return evaluate_fn(game, perspective), []

    legal_actions = game.legal_local_actions()
    if not legal_actions:
        game.normalize_terminal()
        return evaluate_fn(game, perspective), []

    ply_killers = killers.get(ply) if killers is not None else None
    ordered_children = _expand_children(game, legal_actions, tt_entry, priority_fn, ply_killers, history)
    best_score = -math.inf
    best_local: Optional[int] = None
    best_pv: List[int] = []

    def _recurse(child: SongoLegacyGame, child_alpha: float, child_beta: float) -> Tuple[float, List[int]]:
        return negamax_search(
            child,
            depth - 1,
            child_alpha,
            child_beta,
            opponent(perspective),
            tt,
            node_counter,
            node_limit,
            deadline,
            evaluate_fn,
            priority_fn,
            ply + 1,
            killers,
            history,
            use_pvs,
            quiescence_depth,
        )

    for child_index, (local_action, child) in enumerate(ordered_children):
        if use_pvs and child_index > 0 and alpha > -math.inf:
            # PVS : les coups suivants sont d'abord testes en fenetre
            # (quasi) nulle -- juste "bat-il alpha ?". S'il la depasse sans
            # atteindre beta, nouvelle recherche en vraie fenetre pour la
            # valeur exacte. Avec un bon ordre (TT/policy/killers), la
            # re-recherche est rare et le net est un gain.
            score, child_pv = _recurse(child, -alpha - NULL_WINDOW, -alpha)
            score = -score
            if alpha < score < beta:
                score, child_pv = _recurse(child, -beta, -score)
                score = -score
        else:
            score, child_pv = _recurse(child, -beta, -alpha)
            score = -score
        if score > best_score:
            best_score = score
            best_local = local_action
            best_pv = [local_action] + child_pv
        alpha = max(alpha, score)
        if alpha >= beta:
            if killers is not None and local_action not in (killers.get(ply) or []):
                killers.setdefault(ply, []).insert(0, local_action)
                del killers[ply][2:]  # deux killers max par profondeur
            if history is not None:
                history[local_action] = history.get(local_action, 0.0) + depth * depth
            break

    flag = EXACT
    if best_score <= original_alpha:
        flag = UPPER
    elif best_score >= beta:
        flag = LOWER
    tt[key] = TTEntry(depth, best_score, flag, best_local)
    return best_score, best_pv


def iterative_deepening(
    game: SongoLegacyGame,
    limits: SearchLimits,
    evaluate_fn: EvaluateFn = default_evaluate,
    priority_fn: Optional[PriorityFn] = None,
) -> SearchResult:
    """Approfondissement iteratif : renvoie toujours la derniere profondeur
    completement terminee (fallback de securite, cf. section 8.2/Annexe B).
    `evaluate_fn`/`priority_fn` permettent de brancher le reseau entraine
    (section 8.1, cf. songo_ai.hybrid) sans dupliquer cette fonction.
    """
    legal_actions = game.legal_local_actions()
    if not legal_actions:
        raise ValueError("no legal move at root: caller must check game.finished first")

    tt: Dict[int, TTEntry] = {}
    killers: Dict[int, List[int]] = {}
    history: Dict[int, float] = {}
    node_counter = [0]
    start = time.perf_counter()
    deadline = start + limits.max_time_s
    perspective = game.turn

    best_completed = SearchResult(
        local_action=legal_actions[0], score=0.0, depth_reached=0, nodes=0, time_s=0.0, pv=[], timed_out=False
    )

    def _search(depth: int, alpha: float, beta: float) -> Tuple[float, List[int]]:
        return negamax_search(
            game,
            depth,
            alpha,
            beta,
            perspective,
            tt,
            node_counter,
            limits.max_nodes,
            deadline,
            evaluate_fn,
            priority_fn,
            0,
            killers,
            history,
            True,
            limits.quiescence_depth,
        )

    for depth in range(1, limits.max_depth + 1):
        try:
            if best_completed.depth_reached >= 1:
                # Fenetre d'aspiration autour du score precedent ; si le
                # vrai score sort de la fenetre, re-recherche pleine
                # largeur (le resultat final reste exact).
                prev = best_completed.score
                score, pv = _search(depth, prev - ASPIRATION_DELTA, prev + ASPIRATION_DELTA)
                if score <= prev - ASPIRATION_DELTA or score >= prev + ASPIRATION_DELTA:
                    score, pv = _search(depth, -math.inf, math.inf)
            else:
                score, pv = _search(depth, -math.inf, math.inf)
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
