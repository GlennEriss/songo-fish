"""Import de datasets externes issus d'un autre pipeline (etape 5+,
section "reutiliser de vraies parties plutot que du seul auto-jeu").

Cas d'usage concret : `songo-model-stockfish-for-google-collab`, un projet
tiers qui a genere de vraies parties (minimax "insane" vs minimax "insane")
et les a exportees en `.npz` avec son propre encodage de plateau -- pas le
notre. Deux ecarts identifies et resolus par audit (juillet 2026) :

1. Format de plateau different : leur "south" (14 premieres colonnes de
   `x`, moitie 0-6) est stocke en ORDRE INVERSE par rapport au sens de
   semis (leur `pit_number_to_index` fait `7 - pit_number` pour ce cote
   seulement, cf. leur `reference_songo/engine.py`). Conversion validee
   sur 15 000 echantillons repartis sur les 3 splits (0 echec) :
   `board_ours = list(reversed(row[:7])) + list(row[7:14])`.
2. Le joueur au trait n'est pas expose directement dans `x` : on le
   retrouve sans ambiguite en comparant `legal_mask` (deja fourni dans
   leur export) au masque que calcule notre propre moteur pour chaque
   camp -- exactement un des deux correspond a chaque fois (0 echec sur
   15 000 tests, ~2% de positions ou les deux masques coincident, ce qui
   ne cree aucune ambiguite reelle puisque la position est alors legale a
   l'identique quel que soit le joueur choisi).
3. Les magasins (graines deja capturees) ne sont PAS dans les 14
   premieres colonnes : colonnes 15 et 16 de `x` (verifie : leur somme +
   les 14 premieres colonnes vaut exactement 70 sur tout l'echantillon
   teste). Ordre suppose south puis north (memes conventions que le
   plateau) -- a defaut d'un moyen de le verifier de façon independante,
   c'est l'hypothese la plus naturelle vu que le reste de leur format
   suit systematiquement cet ordre.

Une regle de capture manquante dans notre propre moteur a ete corrigee au
passage (songo/rules.py::_capture -- une cascade ne peut jamais vider
TOTALEMENT le camp adverse), avant meme cette conversion : sans ce
correctif, une partie des positions externes auraient ete jugees a tort
incompatibles avec nos regles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from songo_ai.songo.rules import PLAYER_ONE, PLAYER_TWO, SongoLegacyGame


@dataclass
class ExternalPosition:
    board: List[int]  # 16 entiers (magasins a 0 : le score externe n'est pas reutilise)
    turn: int
    game_id: str
    sample_id: str


def convert_row_to_board14(row14: List[int]) -> List[int]:
    """Coeur de la conversion (cf. docstring du module) : south invers +
    north tel quel."""
    south, north = row14[:7], row14[7:14]
    return list(reversed(south)) + list(north)


def resolve_turn(board14: List[int], legal_mask_theirs: List[bool]) -> Optional[int]:
    """Retrouve le joueur au trait par correspondance exacte du masque de
    coups legaux. Renvoie None si ni P1 ni P2 ne correspond (n'est jamais
    arrive sur les 15 000 echantillons audites ; garde-fou defensif)."""
    board16 = board14 + [0, 0]
    for player in (PLAYER_ONE, PLAYER_TWO):
        game = SongoLegacyGame.from_board(board16, player)
        if list(game.legal_mask()) == list(legal_mask_theirs):
            return player
    return None


def load_external_npz_split(path: Path) -> List[ExternalPosition]:
    """Charge un split `.npz` du format externe et convertit chaque ligne
    en position exploitable par notre moteur. Les positions dont le
    joueur au trait ne peut pas etre resolu sont ecartees (avec un
    avertissement) plutot que de faire planter tout le chargement --
    aucune n'a ete rencontree lors de l'audit, mais un futur export du
    meme pipeline pourrait differer legerement."""
    data = np.load(path, allow_pickle=True)
    x = data["x"]
    legal_mask = data["legal_mask"]
    game_ids = data["game_ids"]
    sample_ids = data["sample_ids"]

    positions: List[ExternalPosition] = []
    unresolved = 0
    for i in range(len(x)):
        row14 = x[i][:14].astype(int).tolist()
        board14 = convert_row_to_board14(row14)
        mask = [bool(v) for v in legal_mask[i]]
        turn = resolve_turn(board14, mask)
        if turn is None:
            unresolved += 1
            continue
        south_store, north_store = float(x[i][15]), float(x[i][16])
        positions.append(
            ExternalPosition(
                # P1_STORE <- south (le cote qu'on inverse pour obtenir P1),
                # P2_STORE <- north, meme correspondance que pour le plateau.
                board=board14 + [int(round(south_store)), int(round(north_store))],
                turn=turn,
                game_id=str(game_ids[i]),
                sample_id=str(sample_ids[i]),
            )
        )
    if unresolved:
        print(f"[external_import] {unresolved}/{len(x)} positions ecartees (joueur au trait non resolu) dans {path}")
    return positions
