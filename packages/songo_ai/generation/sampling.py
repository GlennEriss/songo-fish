"""Echantillonnage et deduplication (section 6.3) : plus de milieu/fin de
partie que d'ouverture, plafonnement des positions qui reviennent trop
souvent (une ouverture ne doit pas dominer le corpus), sans les eliminer
completement (on garde un signal de frequence)."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

from .trajectories import TrajectoryPosition
from songo_ai.songo.rules import zobrist_hash

OPENING = "ouverture"
MIDGAME = "milieu"
ENDGAME = "fin"
# "Gestion des fins" (docs/III. Gestion des fins/) : positions ou peu de
# graines restent en jeu des deux cotes (ex. combinaisons "6 contre 4",
# "5 contre 4" alignees). Un classement par ratio de coups joues (ci-dessous)
# peut manquer ces cas -- une partie peut atteindre un solde de graines tres
# bas tot (captures agressives) ou au contraire rester chargee tres tard.
# Le nombre de graines encore en jeu est le bon critere directement, pas un
# proxy indirect via le numero du coup.
TIGHT_ENDGAME = "fin_serree"
TIGHT_ENDGAME_SEED_THRESHOLD = 20

DEFAULT_PHASE_WEIGHTS: Dict[str, float] = {
    OPENING: 1.0,
    MIDGAME: 3.0,
    ENDGAME: 3.0,
    # Poids nettement plus eleve : ces positions sont rares en auto-jeu
    # (la plupart des parties gardent plus de 20 graines en jeu jusqu'a
    # une phase tardive), mais sont exactement celles jugees les plus
    # importantes et les plus difficiles a bien gerer (section identifiee
    # via l'audit du manuscrit "Gestion des fins", chapitre explicitement
    # marque "tres important").
    TIGHT_ENDGAME: 8.0,
}


def seeds_in_play(position: TrajectoryPosition) -> int:
    return sum(position.state.board[0:14])


def classify_phase(position: TrajectoryPosition) -> str:
    if seeds_in_play(position) <= TIGHT_ENDGAME_SEED_THRESHOLD:
        return TIGHT_ENDGAME
    if position.total_moves <= 0:
        return OPENING
    ratio = position.move_number / position.total_moves
    if ratio < 1.0 / 3.0:
        return OPENING
    if ratio < 2.0 / 3.0:
        return MIDGAME
    return ENDGAME


def sample_positions(
    positions: List[TrajectoryPosition],
    target_count: int,
    seed: int = 0,
    phase_weights: Dict[str, float] = DEFAULT_PHASE_WEIGHTS,
    max_repeats: int = 3,
) -> List[TrajectoryPosition]:
    """Tire `target_count` positions parmi `positions`, en respectant les
    poids de phase et en plafonnant les repetitions d'une meme position
    (hash Zobrist canonique, deja sensible au joueur au trait)."""

    rng = random.Random(seed)
    by_phase: Dict[str, List[TrajectoryPosition]] = defaultdict(list)
    for position in positions:
        by_phase[classify_phase(position)].append(position)
    for bucket in by_phase.values():
        rng.shuffle(bucket)

    total_weight = sum(phase_weights.get(phase, 1.0) for phase in by_phase if by_phase[phase])
    if total_weight <= 0:
        total_weight = 1.0

    quotas = {
        phase: int(round(target_count * phase_weights.get(phase, 1.0) / total_weight))
        for phase in by_phase
    }

    repeat_counts: Dict[int, int] = defaultdict(int)
    selected: List[TrajectoryPosition] = []
    selected_ids: set = set()  # (trajectory_id, move_number) : cle stable et hashable, cf. eviter O(n^2)

    def try_take(bucket: List[TrajectoryPosition], quota: int) -> None:
        taken = 0
        for position in bucket:
            if taken >= quota or len(selected) >= target_count:
                return
            key = zobrist_hash(position.state)
            if repeat_counts[key] >= max_repeats:
                continue
            repeat_counts[key] += 1
            selected.append(position)
            selected_ids.add((position.trajectory_id, position.move_number))
            taken += 1

    for phase, bucket in by_phase.items():
        try_take(bucket, quotas.get(phase, 0))

    # Deuxieme passe : completer si des quotas de phase n'ont pas pu etre
    # atteints (bucket trop petit ou trop de doublons plafonnes), en piochant
    # dans ce qu'il reste, toutes phases confondues.
    if len(selected) < target_count:
        leftovers = [
            p
            for bucket in by_phase.values()
            for p in bucket
            if (p.trajectory_id, p.move_number) not in selected_ids
        ]
        rng.shuffle(leftovers)
        for position in leftovers:
            if len(selected) >= target_count:
                break
            key = zobrist_hash(position.state)
            if repeat_counts[key] >= max_repeats:
                continue
            repeat_counts[key] += 1
            selected.append(position)
            selected_ids.add((position.trajectory_id, position.move_number))

    rng.shuffle(selected)
    return selected[:target_count]
