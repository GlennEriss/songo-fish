"""Professeur Alpha-Beta profond : approfondissement adaptatif, arret sur
stabilite, action-values a trois niveaux de cout (etape 4, section 5).

Reutilise le meme coeur de recherche que l'etape 3
(`songo_ai.search.negamax.negamax_search`) : le professeur n'est qu'une
politique d'arret et d'echantillonnage differente autour de la meme
implementation de regles/recherche (principe section 12.1).

Interpretation retenue pour "stable et marge suffisante" (section 5.2) :
le meilleur coup racine doit rester identique sur `stability_window`
profondeurs consecutives ET le score ne doit pas varier de plus de
`min_margin` sur cette meme fenetre. C'est un proxy de stabilite standard
en approfondissement iteratif (moins couteux qu'un vrai calcul de marge
top1/top2 a chaque profondeur), distinct du champ `top1_top2_margin` de
l'annotation, qui lui n'est calcule que pour les niveaux enrichi/premium
(section 5.4) en evaluant explicitement les autres coups racine.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from songo_ai.search.negamax import SearchAborted, TTEntry, negamax_search
from songo_ai.songo.rules import SongoLegacyGame, opponent

STANDARD = "standard"
ENRICHI = "enrichi"
PREMIUM = "premium"


@dataclass
class TeacherConfig:
    initial_depth: int = 4
    depth_step: int = 2
    max_depth: int = 20
    max_nodes: int = 2_000_000
    max_time_s: float = 10.0
    stability_window: int = 3
    min_margin: float = 15.0
    tier: str = STANDARD
    enrichi_depth_reduction: int = 2
    # Ameliorations de recherche (PVS, cf. search/negamax.py) : memes
    # valeurs minimax, elagage plus rapide (~-20% de temps mesure sur le
    # schema d'approfondissement du prof) -- le prof annote plus de
    # positions au meme budget. Active par defaut pour tout NOUVEAU
    # palier ; l'empreinte de cache change avec ce champ, donc les
    # annotations d'anciens paliers (prof legacy) ne sont jamais melangees
    # avec celles-ci.
    use_search_enhancements: bool = True
    # Quiescence aux feuilles (0 = desactivee). ATTENTION : contrairement
    # aux ameliorations ci-dessus, la quiescence CHANGE les valeurs
    # annotees (prolonge les captures a l'horizon) -- a n'activer que
    # deliberement, pour un palier entier, jamais en cours de palier.
    quiescence_depth: int = 0

    def fingerprint(self) -> str:
        base = (
            f"{self.initial_depth}-{self.depth_step}-{self.max_depth}-{self.max_nodes}-"
            f"{self.max_time_s}-{self.stability_window}-{self.min_margin}-{self.tier}-"
            f"{self.enrichi_depth_reduction}"
        )
        # Suffixe seulement si actif : une config entierement legacy garde
        # l'empreinte historique et donc la compatibilite avec les caches
        # d'annotations deja constitues.
        if self.use_search_enhancements or self.quiescence_depth > 0:
            base += f"-enh{int(self.use_search_enhancements)}-q{self.quiescence_depth}"
        return base


@dataclass
class Annotation:
    state_board: Tuple[int, ...]
    state_turn: int
    legal_mask: Tuple[bool, ...]
    best_action: int
    action_values: Dict[int, Optional[float]]
    action_value_depths: Dict[int, int]
    principal_variation: List[int]
    depth: int
    nodes: int
    elapsed_ms: float
    top1_top2_margin: Optional[float]
    is_exact: bool
    tier: str


def _is_stable(history: List[Tuple[int, int, float]], window: int, min_margin: float) -> bool:
    if len(history) < window:
        return False
    recent = history[-window:]
    if len({action for _, action, _ in recent}) != 1:
        return False
    scores = [score for _, _, score in recent]
    return (max(scores) - min(scores)) <= min_margin


def _evaluate_root_actions(
    game: SongoLegacyGame,
    child_depth: int,
    tt: Dict[int, TTEntry],
    node_counter: List[int],
    node_limit: int,
    deadline: float,
    search_kwargs: Optional[dict] = None,
) -> Dict[int, float]:
    """Valeur de CHAQUE coup legal racine, a `child_depth` sur l'enfant.
    Reutilise la TT deja peuplee par la recherche principale : les coups
    deja bien explores coutent nettement moins cher a re-visiter."""
    perspective = game.turn
    values: Dict[int, float] = {}
    for local_action in game.legal_local_actions():
        child = game.clone_for_search()
        child.play_local(local_action)
        try:
            score, _ = negamax_search(
                child,
                child_depth,
                -math.inf,
                math.inf,
                opponent(perspective),
                tt,
                node_counter,
                node_limit,
                deadline,
                **(search_kwargs or {}),
            )
        except SearchAborted:
            break
        values[local_action] = -score
    return values


def _replay_is_terminal(game: SongoLegacyGame, pv: List[int]) -> bool:
    replay = game.clone_for_search()
    for local_action in pv:
        if replay.finished:
            break
        if local_action not in replay.legal_local_actions():
            return False
        replay.play_local(local_action)
    return replay.finished


class DeepTeacher:
    """Section 5.1-5.3 : negamax + ID + TT + ordonnancement (deja construits
    a l'etape 3) pilotes par une politique d'approfondissement adaptative."""

    def __init__(self, config: Optional[TeacherConfig] = None) -> None:
        self.config = config or TeacherConfig()

    def annotate(self, game: SongoLegacyGame) -> Annotation:
        cfg = self.config
        if game.finished:
            raise ValueError("cannot annotate a finished position")
        legal_actions = game.legal_local_actions()
        if not legal_actions:
            raise ValueError("no legal move at root")

        tt: Dict[int, TTEntry] = {}
        node_counter = [0]
        start = time.perf_counter()
        deadline = start + cfg.max_time_s
        perspective = game.turn

        # Kwargs passes a chaque negamax_search : vides pour une config
        # legacy (comportement bit-identique aux paliers deja annotes).
        search_kwargs: dict = {}
        if cfg.use_search_enhancements:
            search_kwargs["use_pvs"] = True
        if cfg.quiescence_depth > 0:
            search_kwargs["quiescence_depth"] = cfg.quiescence_depth

        history: List[Tuple[int, int, float]] = []
        best_action = legal_actions[0]
        best_score = 0.0
        best_pv: List[int] = []
        best_completed_depth = 0

        depth = cfg.initial_depth
        while depth <= cfg.max_depth:
            try:
                score, pv = negamax_search(
                    game, depth, -math.inf, math.inf, perspective, tt, node_counter, cfg.max_nodes, deadline, **search_kwargs
                )
            except SearchAborted:
                break

            best_action = pv[0] if pv else legal_actions[0]
            best_score = score
            best_pv = pv
            best_completed_depth = depth
            history.append((depth, best_action, score))

            if _is_stable(history, cfg.stability_window, cfg.min_margin):
                break
            if node_counter[0] >= cfg.max_nodes or time.perf_counter() >= deadline:
                break

            depth += cfg.depth_step

        action_values: Dict[int, Optional[float]] = {a: None for a in range(7)}
        action_value_depths: Dict[int, int] = {a: 0 for a in range(7)}
        action_values[best_action] = best_score
        action_value_depths[best_action] = best_completed_depth

        top1_top2_margin: Optional[float] = None
        if cfg.tier in (ENRICHI, PREMIUM) and len(legal_actions) > 1:
            child_depth = max(0, best_completed_depth - 1) if cfg.tier == PREMIUM else max(
                0, best_completed_depth - 1 - cfg.enrichi_depth_reduction
            )
            alt_values = _evaluate_root_actions(
                game, child_depth, tt, node_counter, cfg.max_nodes, deadline, search_kwargs
            )
            for action, value in alt_values.items():
                if action == best_action:
                    continue
                action_values[action] = value
                action_value_depths[action] = child_depth + 1
            runner_up = max((v for a, v in action_values.items() if a != best_action and v is not None), default=None)
            if runner_up is not None:
                top1_top2_margin = best_score - runner_up

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        is_exact = _replay_is_terminal(game, best_pv)

        return Annotation(
            state_board=game.to_state().board,
            state_turn=game.turn,
            legal_mask=game.legal_mask(),
            best_action=best_action,
            action_values=action_values,
            action_value_depths=action_value_depths,
            principal_variation=best_pv,
            depth=best_completed_depth,
            nodes=node_counter[0],
            elapsed_ms=elapsed_ms,
            top1_top2_margin=top1_top2_margin,
            is_exact=is_exact,
            tier=cfg.tier,
        )
