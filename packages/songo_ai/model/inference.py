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


def load_model(checkpoint_path: Path, dropout: float = 0.0, width: int = 128, num_blocks: int = 3) -> SongoNet:
    """`width`/`num_blocks` doivent correspondre a l'architecture enregistree
    dans le manifeste de la version chargee (voir `songo_ai.model.registry`) :
    ils ne sont pas stockes dans le checkpoint lui-meme, seulement les poids."""
    model = SongoNet(width=width, num_blocks=num_blocks, dropout=dropout)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def make_network_agent(model: SongoNet, temperature: float = 0.0):
    """Renvoie un Agent (section generation.agents.Agent). `temperature=0`
    (defaut) : argmax pur, deterministe -- pour mesurer "ce que le reseau a
    appris" (section 3.2). `temperature>0` : echantillonne selon
    softmax(logits/temperature) via le rng fourni par l'appelant (ex.
    songo_ai.evaluation.play_match), utile pour obtenir des parties
    reellement variees en tournoi plutot que la meme partie repetee."""
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

        if temperature <= 0 or rng is None:
            return int(torch.argmax(masked).item())

        probs = torch.softmax(masked / temperature, dim=-1)
        legal_indices = [i for i in range(7) if legal_mask[i]]
        weights = [probs[i].item() for i in legal_indices]
        return rng.choices(legal_indices, weights=weights, k=1)[0]

    return agent
