"""Cache d'annotations (section 6.3 : "utiliser un cache d'annotations afin
de ne jamais payer deux fois la meme recherche"). Cle = hash Zobrist de la
position + empreinte de la config du professeur (deux configs differentes
ne doivent pas partager un resultat)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .deep_teacher import Annotation, TeacherConfig


def cache_key(zobrist_hash: int, config: TeacherConfig) -> str:
    raw = f"{zobrist_hash}:{config.fingerprint()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class AnnotationCache:
    """Cache local sur disque (un fichier JSON par entree). Suffisant pour
    le developpement local et les paliers 10k/100k positions (section 11.3 :
    "local d'abord"). Un backend partage (SQLite, Cloud Storage) pourra
    remplacer ce stockage sans changer l'API get/put."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, zobrist_hash: int, config: TeacherConfig) -> Optional[Annotation]:
        path = self._path(cache_key(zobrist_hash, config))
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        data["state_board"] = tuple(data["state_board"])
        data["legal_mask"] = tuple(data["legal_mask"])
        data["principal_variation"] = list(data["principal_variation"])
        data["action_values"] = {int(k): v for k, v in data["action_values"].items()}
        data["action_value_depths"] = {int(k): v for k, v in data["action_value_depths"].items()}
        return Annotation(**data)

    def put(self, zobrist_hash: int, config: TeacherConfig, annotation: Annotation) -> None:
        path = self._path(cache_key(zobrist_hash, config))
        path.write_text(json.dumps(asdict(annotation)))

    def get_or_annotate(self, teacher, game) -> Annotation:
        zhash = game.zobrist_hash()
        cached = self.get(zhash, teacher.config)
        if cached is not None:
            return cached
        annotation = teacher.annotate(game)
        self.put(zhash, teacher.config, annotation)
        return annotation
