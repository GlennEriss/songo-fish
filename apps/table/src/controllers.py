"""Traduit un choix de controleur ("random", "minimax:4", "model:0.1.0",
"model:champion") en Agent jouable (meme protocole que
songo_ai.generation.agents.Agent), pour brancher n'importe quelle
combinaison de niveaux/modeles sur la table sans dupliquer de logique."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from songo_ai.generation import make_shallow_search_agent, random_agent
from songo_ai.model import get_champion, load_model, make_network_agent, load_registry


def make_controller(spec: str):
    """`spec` : "random" | "minimax:<profondeur>" | "model:<version|champion>"."""
    if spec == "random":
        return random_agent, "Aleatoire"

    if spec.startswith("minimax:"):
        depth = int(spec.split(":", 1)[1])
        agent = make_shallow_search_agent(max_depth=depth, max_nodes=200_000, max_time_s=2.0)
        return agent, f"Minimax profondeur {depth}"

    if spec.startswith("model:"):
        version = spec.split(":", 1)[1]
        entry = get_champion() if version == "champion" else load_registry()["versions"].get(version)
        if entry is None:
            raise ValueError(f"version de modele inconnue: {version} (registre: data/checkpoints/registry.json)")
        model = load_model(
            Path(entry["checkpoint_path"]),
            dropout=entry["architecture"]["dropout"],
            width=entry["architecture"].get("width", 128),
            num_blocks=entry["architecture"].get("num_blocks", 3),
        )
        agent = make_network_agent(model, temperature=0.0)
        return agent, f"Modele v{entry['version']}"

    raise ValueError(f"controleur inconnu: {spec!r} (attendu: random | minimax:N | model:VERSION)")
