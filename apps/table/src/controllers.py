"""Traduit un choix de controleur ("random", "minimax:4", "songofish:champion")
en Agent jouable (meme protocole que songo_ai.generation.agents.Agent), pour
brancher n'importe quelle combinaison de niveaux/modeles sur la table sans
dupliquer de logique.

Pas de mode "modele seul sans recherche" ici : un reseau sans lookahead n'est
pas un adversaire pertinent a regarder jouer, cf. songo_ai.hybrid (etape 8)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from songo_ai.generation import make_shallow_search_agent, random_agent
from songo_ai.hybrid import SongoFishConfig, make_songofish_agent
from songo_ai.model import get_champion, load_model, load_registry


def _resolve_model_entry(version: str) -> dict:
    entry = get_champion() if version == "champion" else load_registry()["versions"].get(version)
    if entry is None:
        raise ValueError(f"version de modele inconnue: {version} (registre: data/checkpoints/registry.json)")
    return entry


def _load_versioned_model(entry: dict):
    return load_model(
        Path(entry["checkpoint_path"]),
        dropout=entry["architecture"]["dropout"],
        width=entry["architecture"].get("width", 128),
        num_blocks=entry["architecture"].get("num_blocks", 3),
    )


def make_controller(spec: str):
    """`spec` : "random" | "minimax:<profondeur>[:max_time_s]" | "songofish:<version|champion>[:profondeur]"."""
    if spec == "random":
        return random_agent, "Aleatoire"

    if spec.startswith("minimax:"):
        # minimax:<profondeur>[:max_time_s] -- profondeur plafonnee mais le
        # budget de temps (defaut 2s, cf. STANDARD dans songo_ai.teachers ou
        # 5s reprend le budget reel du jeu) reste le vrai facteur limitant en
        # pratique via l'approfondissement iteratif (joue le meilleur coup
        # trouve si le temps est ecoule avant d'atteindre `profondeur`).
        parts = spec.split(":")
        depth = int(parts[1])
        max_time_s = float(parts[2]) if len(parts) > 2 else 2.0
        agent = make_shallow_search_agent(max_depth=depth, max_nodes=300_000, max_time_s=max_time_s)
        return agent, f"Minimax profondeur {depth} (max {max_time_s:g}s)"

    if spec.startswith("songofish:"):
        # reseau (ordonnancement + eval feuilles) + recherche alpha-beta bornee (etape 8)
        parts = spec.split(":")
        version = parts[1]
        depth = int(parts[2]) if len(parts) > 2 else 10
        entry = _resolve_model_entry(version)
        model = _load_versioned_model(entry)
        agent = make_songofish_agent(model, SongoFishConfig(max_depth=depth, max_nodes=300_000, max_time_s=2.0))
        return agent, f"SongoFish v{entry['version']} (recherche profondeur {depth})"

    raise ValueError(
        f"controleur inconnu: {spec!r} (attendu: random | minimax:N | songofish:VERSION[:profondeur])"
    )
