"""Schema d'une observation annotee (Annexe A du plan directeur).

policy_target et wdl_target sont derives de l'Annotation du professeur
(section 6.4/6.5). Le calibrage tanh(score/k) de wdl_target est un
PLACEHOLDER non calibre (k choisi arbitrairement) : la vraie calibration
(section 6.5) exige un corpus de resultats reels, qui n'existe pas encore a
l'etape 5. A ne pas prendre pour une probabilite fiable avant calibration.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from songo_ai.songo.rules import PLAYER_ONE, SongoLegacyGame
from songo_ai.teachers import Annotation

RULES_VERSION = "songo-python-v1"
DATASET_VERSION = "v001"

# Echelle provisoire de calibration score -> WDL (section 6.5). Nos scores
# internes sont en unites "diff. de magasins x10 + territoire sur x3 +
# mobilite" (cf. songo_ai.search.negamax.default_evaluate) ; k est choisi
# pour qu'une
# difference de quelques graines produise une inclinaison moderee, pas
# saturee. A recalibrer sur un corpus de resultats reels des que possible.
_WDL_PLACEHOLDER_K = 300.0
_WDL_DRAW_EPSILON = 0.05


@dataclass
class Consequences:
    capture_now: int
    future_mobility: int
    famine_risk: float


@dataclass
class TeacherMeta:
    depth: int
    nodes: int
    elapsed_ms: float
    is_exact: bool
    confidence: float
    tier: str


@dataclass
class Observation:
    state: List[int]
    legal_mask: List[bool]
    best_action: int
    policy_target: List[float]
    wdl_target: List[float]
    action_values: List[Optional[float]]
    action_value_depths: List[int]
    principal_variation: List[int]
    consequences: Consequences
    teacher: TeacherMeta
    trajectory_id: str
    move_number: int
    rules_version: str = RULES_VERSION
    dataset_version: str = DATASET_VERSION

    def to_json_dict(self) -> dict:
        d = asdict(self)
        return d


def canonicalize_board(board: Tuple[int, ...], turn: int) -> Tuple[int, ...]:
    """Plateau vu du joueur au trait (section 6.3) : "mon" camp est toujours
    les pits 0-6 et magasin 14, l'adversaire toujours 7-13 et magasin 15,
    quel que soit le joueur physique. C'est ce qui permet a `state` de ne
    pas avoir besoin d'un champ "turn" separe (coherent avec l'annexe A, qui
    n'en a pas) : legal_mask/best_action/action_values sont deja exprimes en
    action locale 0..6, donc deja relatifs a ce meme repere canonique."""
    if turn == PLAYER_ONE:
        return tuple(board)
    mine, theirs = board[7:14], board[0:7]
    return tuple(mine) + tuple(theirs) + (board[15], board[14])


def score_to_bounded(score: float, k: float = _WDL_PLACEHOLDER_K) -> float:
    """Normalisation placeholder score -> [-1, 1] (section 6.5). Reutilisee
    a la fois pour wdl_target et, cote entrainement, pour les cibles
    action_values : les scores bruts du professeur vont de quelques dizaines
    (milieu de partie) a +/-70000 (positions terminales, diff*1000) et ne
    doivent jamais etre regresses tels quels (MSE explose sur les valeurs
    extremes)."""
    return math.tanh(score / k)


def _score_to_wdl(score: float) -> Tuple[float, float, float]:
    lean = score_to_bounded(score)
    win = max(0.0, lean) * (1.0 - _WDL_DRAW_EPSILON)
    loss = max(0.0, -lean) * (1.0 - _WDL_DRAW_EPSILON)
    draw = 1.0 - win - loss
    total = win + loss + draw
    return win / total, draw / total, loss / total


def _policy_target_from_action_values(
    legal_mask: Tuple[bool, ...], action_values: Dict[int, Optional[float]], best_action: int, temperature: float = 60.0
) -> List[float]:
    """Softmax masque sur les action_values connues (niveau enrichi/premium).
    Les coups legaux sans valeur connue (niveau standard) recoivent une
    petite masse uniforme plutot que zero, pour eviter une politique
    degenerement one-hot des le premier professeur standard."""
    known = {a: v for a, v in action_values.items() if legal_mask[a] and v is not None}
    if not known:
        # Seul best_action est connu (niveau standard) : quasi one-hot avec
        # une masse residuelle repartie sur les autres coups legaux.
        legal_count = sum(1 for m in legal_mask if m)
        target = [0.0] * 7
        residual = 0.05 if legal_count > 1 else 0.0
        for a in range(7):
            if not legal_mask[a]:
                continue
            target[a] = residual / max(1, legal_count - 1) if a != best_action else 1.0 - residual
        return target

    max_value = max(known.values())
    weights = {a: math.exp((v - max_value) / temperature) for a, v in known.items()}
    total = sum(weights.values())
    target = [0.0] * 7
    for a, w in weights.items():
        target[a] = w / total
    return target


def annotation_to_observation(annotation: Annotation, trajectory_id: str, move_number: int) -> Observation:
    consequences = _consequences_from_annotation(annotation)
    wdl = _score_to_wdl(annotation.action_values[annotation.best_action] or 0.0)
    policy = _policy_target_from_action_values(annotation.legal_mask, annotation.action_values, annotation.best_action)

    return Observation(
        state=list(canonicalize_board(annotation.state_board, annotation.state_turn)),
        legal_mask=list(annotation.legal_mask),
        best_action=annotation.best_action,
        policy_target=policy,
        wdl_target=list(wdl),
        action_values=[annotation.action_values.get(a) for a in range(7)],
        action_value_depths=[annotation.action_value_depths.get(a, 0) for a in range(7)],
        principal_variation=list(annotation.principal_variation),
        consequences=consequences,
        teacher=TeacherMeta(
            depth=annotation.depth,
            nodes=annotation.nodes,
            elapsed_ms=annotation.elapsed_ms,
            is_exact=annotation.is_exact,
            confidence=_confidence_from_margin(annotation.top1_top2_margin),
            tier=annotation.tier,
        ),
        trajectory_id=trajectory_id,
        move_number=move_number,
    )


def _confidence_from_margin(margin: Optional[float]) -> float:
    if margin is None:
        return 0.5  # inconnu (niveau standard) : ni confiant ni pas confiant, valeur neutre.
    return max(0.0, min(1.0, 0.5 + margin / 100.0))


def _consequences_from_annotation(annotation: Annotation) -> Consequences:
    game = SongoLegacyGame.from_board(annotation.state_board, annotation.state_turn)
    result = game.play_local(annotation.best_action)
    future_mobility = 0 if game.finished else len(game.legal_local_actions())
    famine_risk = 0.0 if game.finished else (0.0 if game.can_transmit_for_player(game.turn) else 1.0)
    return Consequences(capture_now=result.captured, future_mobility=future_mobility, famine_risk=famine_risk)
