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
        # Tenseurs precalcules UNE FOIS ici plutot que dans __getitem__ :
        # observation_features() reconstruit des mini-jeux internes
        # (mobilite/transmission adverses, cf. features.py) et coutait donc
        # cher a chaque acces -- avec le DataLoader qui repasse sur les
        # memes lignes a chaque epoque (60 par entrainement), c'etait
        # recalcule 60x pour un resultat pourtant strictement identique a
        # chaque fois (les features ne dependent que de la ligne, jamais de
        # l'epoque). Mesure : ~265s de calcul redondant economises sur un
        # entrainement de 60 epoques / 315k lignes.
        self._features: List[torch.Tensor] = []
        self._legal_mask: List[torch.Tensor] = []
        self._policy_target: List[torch.Tensor] = []
        self._wdl_target: List[torch.Tensor] = []
        self._action_values: List[torch.Tensor] = []
        self._q_mask: List[torch.Tensor] = []

        with Path(shard_path).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._features.append(
                    torch.tensor(observation_features(row["state"], row["legal_mask"]), dtype=torch.float32)
                )
                self._legal_mask.append(torch.tensor(row["legal_mask"], dtype=torch.bool))
                self._policy_target.append(torch.tensor(row["policy_target"], dtype=torch.float32))
                self._wdl_target.append(torch.tensor(row["wdl_target"], dtype=torch.float32))

                action_values = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
                q_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
                for a in range(NUM_ACTIONS):
                    v = row["action_values"][a]
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        # Les scores bruts du professeur vont de quelques
                        # dizaines a +/-70000 (positions terminales) :
                        # jamais regresses tels quels (cf.
                        # songo_ai.dataset.schema.score_to_bounded).
                        action_values[a] = score_to_bounded(v)
                        q_mask[a] = True
                self._action_values.append(action_values)
                self._q_mask.append(q_mask)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int):
        return {
            "features": self._features[idx],
            "legal_mask": self._legal_mask[idx],
            "policy_target": self._policy_target[idx],
            "wdl_target": self._wdl_target[idx],
            "action_values": self._action_values[idx],
            "q_mask": self._q_mask[idx],
        }
