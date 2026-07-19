#!/usr/bin/env python3
"""Annote des positions de fin de partie choisies a la main (etape 5+,
suite tactique de la section 9 du plan directeur : "PV et indicateurs
verifies sur cas connus") avec le professeur, budget genereux.

Contexte : docs/III. Gestion des fins/ et docs/II. Contre-Attaques 2/
documentent des combinaisons de fin de partie jugees importantes par des
joueurs experts (manuscrit familial). Verifier chaque affirmation a la
main contre une recherche profonde s'est revele couteux et peu concluant
(memes des positions a 9-10 graines ne se resolvent pas exactement meme a
30-47 coups de profondeur et plusieurs millions de noeuds). Plutot que
d'extraire ces strategies comme des regles codees en dur, ce script les
traite comme une LISTE DE POSITIONS CRITIQUES A ECHANTILLONNER : le
professeur les annote avec sa propre recherche (budget dedie, plus genereux
que le tirage aleatoire standard), et le reseau apprend une verite ancree
dans la recherche, pas une heuristique humaine qu'on peine a verifier
soi-meme.

Convention de plateau (mapping case->pit verifie contre le manuscrit,
cf. apps/table/src/board_view.py::pit_position, section "reference-songo-
rules-docs") :
    A_case_i -> pit(6+i)   (A = rangee du haut, case1=pit7 ... case7=pit13)
    B_case_i -> pit(i-1)   (B = rangee du bas,  case1=pit0 ... case7=pit6)
    pit14 = magasin de B (PLAYER_ONE), pit15 = magasin de A (PLAYER_TWO)

Usage :
    .venv/bin/python apps/trainer/scripts/build_curated_endgames.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from songo_ai.dataset.schema import annotation_to_observation
from songo_ai.dataset.validate import validate_observation
from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO, SongoLegacyGame
from songo_ai.teachers import PREMIUM, DeepTeacher, TeacherConfig

OUT_DIR = Path("data/dataset_curated_endgames")

# Budget genereux et dedie : ce ne sont que quelques dizaines de positions
# (pas 100k+), on peut se permettre une recherche bien plus profonde que
# le tirage standard.
CURATED_CONFIG = TeacherConfig(
    initial_depth=10,
    depth_step=4,
    max_depth=60,
    max_nodes=20_000_000,
    max_time_s=120.0,
    stability_window=4,
    min_margin=10.0,
    tier=PREMIUM,
)


@dataclass
class _CuratedPosition:
    label: str
    board: List[int]
    turn: int
    source: str


def _case(player_row: str, case_i: int) -> int:
    """`player_row` : "A" (rangee du haut, cases 1..7 -> pit 7..13) ou "B"
    (rangee du bas, cases 1..7 -> pit 0..6, sens inverse)."""
    if player_row == "A":
        return 6 + case_i
    if player_row == "B":
        return case_i - 1
    raise ValueError(player_row)


def _board_from_cases(a_cases: dict, b_cases: dict, p1_store: int = 0, p2_store: int = 0) -> List[int]:
    """`p1_store`/`p2_store` : graines deja capturees avant cette etude de
    fin de partie isolee. Le manuscrit ne precise pas ce qu'il y a deja
    dans les magasins (seul le solde "en jeu" l'interesse), mais
    `validate_observation` exige la conservation totale (70 graines,
    plateau+magasins) comme sur toute position reellement atteignable --
    on complete donc arbitrairement, en restant confortablement sous le
    seuil de victoire (>35) pour ne pas terminer la partie prematurement
    et fausser l'etude de la fin en cours."""
    board = [0] * 16
    for case_i, seeds in a_cases.items():
        board[_case("A", case_i)] = seeds
    for case_i, seeds in b_cases.items():
        board[_case("B", case_i)] = seeds
    board[14] = p1_store
    board[15] = p2_store
    return board


# Positions extraites et verifiees (juillet 2026, cf. memoire du projet) :
# le nombre de coups legaux reconstruit correspond exactement a ce que le
# manuscrit decrit a chaque fois (bon signal de fidelite de la
# transcription). D'autres seront ajoutees au fur et a mesure de
# l'extraction du manuscrit -- ce script n'a pas vocation a rester fige a
# ces deux exemples.
CURATED_POSITIONS: List[_CuratedPosition] = [
    _CuratedPosition(
        label="fin_6contre4_b_au_trait",
        # 10 en jeu + 60 deja captures (30/30, arbitraire, sous le seuil de 35)
        board=_board_from_cases({4: 1, 7: 5}, {5: 1, 6: 1, 7: 2}, p1_store=30, p2_store=30),
        turn=PLAYER_ONE,  # B au trait (le manuscrit precise "3 possibilites : B5, B6 ou B7")
        source="docs/III. Gestion des fins/Songo_Fin_01.jpg (2e tableau)",
    ),
    _CuratedPosition(
        label="fin_5contre4_a_au_trait",
        # 9 en jeu + 61 deja captures (30/31, arbitraire, sous le seuil de 35)
        board=_board_from_cases({1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, {1: 1, 2: 1, 3: 1, 4: 1}, p1_store=30, p2_store=31),
        turn=PLAYER_TWO,  # "lorsque le joueur au cote de 5 commence a jouer" -> A commence
        source="docs/III. Gestion des fins/Songo_Fin_03.jpg",
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    teacher = DeepTeacher(CURATED_CONFIG)
    observations = []
    problems_found: List[str] = []

    for i, cp in enumerate(CURATED_POSITIONS):
        game = SongoLegacyGame.from_board(cp.board, cp.turn)
        if game.finished:
            print(f"[{cp.label}] position deja terminee, ignoree")
            continue
        print(f"[{cp.label}] annotation en cours (source: {cp.source})...")
        annotation = teacher.annotate(game)
        obs = annotation_to_observation(annotation, trajectory_id=f"curated-{cp.label}", move_number=0)
        problems = validate_observation(obs)
        if problems:
            problems_found.extend(f"{cp.label}: {p}" for p in problems)
            print(f"[{cp.label}] PROBLEMES DE VALIDATION: {problems}")
            continue
        observations.append(obs)
        print(
            f"[{cp.label}] meilleur coup local={annotation.best_action} profondeur={annotation.depth} "
            f"noeuds={annotation.nodes:,} is_exact={annotation.is_exact} "
            f"elapsed={annotation.elapsed_ms/1000:.1f}s"
        )

    shard_path = OUT_DIR / "curated.jsonl"
    with shard_path.open("w") as f:
        for obs in observations:
            f.write(json.dumps(obs.to_json_dict()) + "\n")

    manifest = {
        "total_positions": len(observations),
        "sources": [cp.source for cp in CURATED_POSITIONS],
        "teacher_config": asdict(CURATED_CONFIG),
        "validation_problems": problems_found,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(observations)} positions annotees -> {shard_path}")
    if problems_found:
        print(f"ATTENTION : {len(problems_found)} position(s) rejetee(s) a la validation, voir manifest.json")


if __name__ == "__main__":
    main()
