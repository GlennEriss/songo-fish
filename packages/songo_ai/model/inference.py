"""Mode "reseau seul" (section 3.2) : une seule inference, sans recherche.
Sert a mesurer ce que le reseau a reellement appris, independamment de tout
budget de recherche -- le point de comparaison le plus simple pour les
tournois de la section 10.2."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch

from songo_ai.dataset import canonicalize_board
from songo_ai.songo.rules import SongoLegacyGame

from .features import observation_features
from .network import SongoNet


def load_model(checkpoint_path: Path, dropout: float = 0.0) -> SongoNet:
    model = SongoNet(dropout=dropout)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def make_network_agent(model: SongoNet):
    """Renvoie un Agent (section generation.agents.Agent) qui joue le coup
    de plus forte probabilite policy, masque aux coups legaux -- aucune
    recherche, une seule passe avant du reseau par coup."""
    model.eval()

    def agent(game: SongoLegacyGame, rng: Optional[random.Random] = None) -> int:
        state = game.to_state()
        canonical_board = canonicalize_board(state.board, state.turn)
        legal_mask = game.legal_mask()
        features = torch.tensor(observation_features(canonical_board, legal_mask), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            policy_logits, _, _ = model(features)
        logits = policy_logits.squeeze(0)
        mask_tensor = torch.tensor(legal_mask, dtype=torch.bool)
        masked = logits.masked_fill(~mask_tensor, float("-inf"))
        return int(torch.argmax(masked).item())

    return agent
