"""Dataset PyTorch charge depuis les shards JSONL de l'etape 5. 10k
positions tiennent largement en memoire : pas besoin de streaming a cette
echelle (a revoir au palier 1M+, section 7.4)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from songo_ai.dataset import score_to_bounded

from .features import NUM_ACTIONS, observation_features


class ObservationDataset(Dataset):
    def __init__(self, shard_path: Path) -> None:
        self.rows: List[dict] = []
        with Path(shard_path).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        features = torch.tensor(observation_features(row["state"], row["legal_mask"]), dtype=torch.float32)
        legal_mask = torch.tensor(row["legal_mask"], dtype=torch.bool)
        policy_target = torch.tensor(row["policy_target"], dtype=torch.float32)
        wdl_target = torch.tensor(row["wdl_target"], dtype=torch.float32)

        action_values = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
        q_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        for a in range(NUM_ACTIONS):
            v = row["action_values"][a]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                # Les scores bruts du professeur vont de quelques dizaines a
                # +/-70000 (positions terminales) : jamais regresses tels
                # quels (cf. songo_ai.dataset.schema.score_to_bounded).
                action_values[a] = score_to_bounded(v)
                q_mask[a] = True

        return {
            "features": features,
            "legal_mask": legal_mask,
            "policy_target": policy_target,
            "wdl_target": wdl_target,
            "action_values": action_values,
            "q_mask": q_mask,
        }
