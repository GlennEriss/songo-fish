"""Cache d'annotations (section 6.3 : "utiliser un cache d'annotations afin
de ne jamais payer deux fois la meme recherche"). Cle = hash Zobrist de la
position + empreinte de la config du professeur (deux configs differentes
ne doivent pas partager un resultat).

Backend SQLite (WAL) plutot qu'un fichier JSON par position : le run GCP
100k a montre qu'un dossier plat a 100 000 entrees, ecrit en concurrence par
32 processus et re-liste en entier toutes les 5 minutes par le script de
synchronisation GCS, degrade fortement l'efficacite parallele (~45% mesure
au lieu de ~90%+ attendu). SQLite en mode WAL autorise des lectures
concurrentes sans bloquer l'ecrivain, et la synchronisation GCS ne porte
plus que sur un seul fichier (via l'API de sauvegarde a chaud) au lieu de
parcourir des dizaines de milliers de petits fichiers.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .deep_teacher import Annotation, TeacherConfig


def cache_key(zobrist_hash: int, config: TeacherConfig) -> str:
    raw = f"{zobrist_hash}:{config.fingerprint()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class AnnotationCache:
    """Cache SQLite (une base par repertoire `root`, fichier `cache.db`).
    Suffisant pour le developpement local et les paliers 10k/100k/1M
    (section 11.3 : "local d'abord"). Un backend distant (Cloud SQL,
    Firestore) pourra remplacer ce stockage sans changer l'API get/put."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "cache.db"
        # timeout genereux : plusieurs processus peuvent se disputer
        # l'ecrivain unique de SQLite sous forte concurrence (32 workers).
        self._conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS annotations (key TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, zobrist_hash: int, config: TeacherConfig) -> Optional[Annotation]:
        key = cache_key(zobrist_hash, config)
        row = self._conn.execute("SELECT data FROM annotations WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        data["state_board"] = tuple(data["state_board"])
        data["legal_mask"] = tuple(data["legal_mask"])
        data["principal_variation"] = list(data["principal_variation"])
        data["action_values"] = {int(k): v for k, v in data["action_values"].items()}
        data["action_value_depths"] = {int(k): v for k, v in data["action_value_depths"].items()}
        return Annotation(**data)

    def put(self, zobrist_hash: int, config: TeacherConfig, annotation: Annotation) -> None:
        key = cache_key(zobrist_hash, config)
        payload = json.dumps(asdict(annotation))
        self._conn.execute(
            "INSERT OR REPLACE INTO annotations (key, data) VALUES (?, ?)", (key, payload)
        )
        self._conn.commit()

    def get_or_annotate(self, teacher, game) -> Annotation:
        zhash = game.zobrist_hash()
        cached = self.get(zhash, teacher.config)
        if cached is not None:
            return cached
        annotation = teacher.annotate(game)
        self.put(zhash, teacher.config, annotation)
        # Renvoyer la forme RELUE, pas l'objet fraichement calcule : le
        # round-trip JSON (tuple->list->tuple, cles de dict en int, repr des
        # floats) doit etre neutre, et le garantir ici rend une reprise
        # totalement transparente -- la sortie d'un run ne depend pas de
        # savoir quelles positions venaient deja du cache.
        return self.get(zhash, teacher.config)

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]

    def snapshot_to(self, path: Path) -> None:
        """Copie coherente a chaud (API de sauvegarde SQLite) : utilisable
        pendant que d'autres processus ecrivent encore dans la base, sans
        risquer un fichier corrompu ou a moitie ecrit (section 11.3 :
        synchronisation continue vers GCS pendant le calcul)."""
        dest = sqlite3.connect(str(path))
        with dest:
            self._conn.backup(dest)
        dest.close()

    def close(self) -> None:
        self._conn.close()
