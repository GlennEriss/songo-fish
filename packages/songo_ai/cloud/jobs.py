"""Specs de job independantes du provider (Template Method) : un job decrit
CE qu'il faut calculer ; le provider decide OU (cette machine / une VM) et
via quel `ArtifactStore`.

Trois jobs couvrent le pipeline quotidien :
- `BuildSpec`     : generer + annoter un dataset frais (les "matchs")
- `TrainSpec`     : entrainer une version du reseau from scratch
- `MatchSpec`     : tournoi entre deux agents (champion/challenger, references)

Presets de professeur : un seul endroit pour les configs d'annotation, au
lieu d'une constante recopiee dans chaque script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from songo_ai.dataset import DEFAULT_TEACHER_CONFIG
from songo_ai.teachers import PREMIUM, STANDARD, TeacherConfig

# "deep" : ex-DEEP_CONFIG de build_100k_gcp.py -- "tres profonde mais
# budget-consciente", validee localement (moyenne 6.5 s/position). Le
# palier 100k (etape 7) l'utilise.
DEEP_CONFIG = TeacherConfig(
    initial_depth=6,
    depth_step=2,
    max_depth=28,
    max_nodes=3_000_000,
    max_time_s=45.0,
    stability_window=3,
    min_margin=15.0,
    tier=STANDARD,
)

# "reannotate" : ex-REANNOTATE_CONFIG -- profondeur haute, budget genereux,
# pour re-annoter un corpus existant avec toute la qualite du professeur
# actuel (propagation d'un correctif, ex. bidoua/Yinda).
REANNOTATE_CONFIG = TeacherConfig(
    initial_depth=8,
    depth_step=4,
    max_depth=40,
    max_nodes=10_000_000,
    max_time_s=60.0,
    stability_window=4,
    min_margin=10.0,
    tier=PREMIUM,
)

TEACHER_PRESETS = {
    "default": DEFAULT_TEACHER_CONFIG,  # 5 s/position, palier 10k
    "deep": DEEP_CONFIG,
    "reannotate": REANNOTATE_CONFIG,
}


@dataclass
class BuildSpec:
    num_positions: int
    seed: int = 456
    teacher_preset: str = "deep"
    # nom logique du dataset dans le store (defaut derive du volume + seed)
    dataset_name: Optional[str] = None
    trajectory_multiplier: int = 4
    max_moves: int = 300

    def resolved_name(self) -> str:
        return self.dataset_name or f"datasets/dataset_s{self.seed}_{self.num_positions}"

    def teacher_config(self) -> TeacherConfig:
        if self.teacher_preset not in TEACHER_PRESETS:
            raise ValueError(f"preset inconnu: {self.teacher_preset!r} (attendu {sorted(TEACHER_PRESETS)})")
        return TEACHER_PRESETS[self.teacher_preset]


@dataclass
class TrainSpec:
    version: str
    dataset_name: str  # chemin logique, ex "datasets/dataset_v005_400k"
    epochs: int = 150
    early_stopping_patience: int = 15
    promote: bool = False
    notes: str = ""


@dataclass
class MergeSpec:
    # noms logiques des releases a fusionner (>= 2), ex
    # ["datasets/dataset_s2026_100000", "datasets/dataset_s2027_100000"]
    sources: list
    out_name: str
    # section 6.3 : deux releases distinctes peuvent produire la MEME
    # position par des chemins differents -- dedup par defaut.
    deduplicate: bool = True


@dataclass
class MatchSpec:
    # chaque agent : "vX.Y.Z" (registre), "champion", "random",
    # "minimax:<depth>" (agent de reference)
    agent_a: str
    agent_b: str
    num_games: int = 200
    opening_random_plies: int = 6
    seed: int = 0
    temperature: float = 0.15
