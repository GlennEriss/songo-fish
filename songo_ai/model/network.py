"""Reseau multi-tetes (section 7.2) : tronc MLP residuel leger, puis trois
tetes policy (7 logits), WDL (3 logits) et Q-actions (7 valeurs). L'option
NNUE (mise a jour incrementale) n'est pas implementee : le plan la reserve
au cas ou le profilage prouverait que l'inference domine reellement le
cout, ce qui n'est pas etabli avec un etat aussi compact (16 compteurs)."""

from __future__ import annotations

import torch
from torch import nn

from .features import FEATURE_SIZE, NUM_ACTIONS


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.act(out + residual)


class SongoNet(nn.Module):
    def __init__(self, input_size: int = FEATURE_SIZE, width: int = 128, num_blocks: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, width), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(num_blocks)])
        self.policy_head = nn.Linear(width, NUM_ACTIONS)
        self.wdl_head = nn.Linear(width, 3)
        self.q_head = nn.Linear(width, NUM_ACTIONS)

    def forward(self, x: torch.Tensor):
        trunk = self.blocks(self.input_layer(x))
        return self.policy_head(trunk), self.wdl_head(trunk), self.q_head(trunk)
