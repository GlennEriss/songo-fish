"""Backend de stockage des artefacts -- pattern Strategy (section 11.1 :
"Cloud Storage : shards, cache d'annotations, releases et checkpoints").

Le reste du code ne manipule que des chemins LOGIQUES relatifs
(`"datasets/dataset_v002_100k"`, `"checkpoints/registry.json"`,
`"caches/campaign_x"`). Le backend traduit ce chemin en un `Path` local
utilisable, et sait le rapatrier (`pull`) / le publier (`push`) si la
verite vit ailleurs.

- `LocalStore` : tout est deja sur le disque, `pull`/`push` sont des
  no-op. C'est le cas d'usage principal (machine Windows).
- `GcsStore` : la verite est dans un bucket ; un miroir local sert de
  zone de travail, `pull`/`push` appellent `gcloud storage cp`.

L'`AnnotationCache` (`songo_ai.teachers.cache`) reste inchange : c'est
toujours un fichier SQLite dans le `Path` que lui donne le store. La
"synchronisation continue vers GCS pendant le calcul" (section 11.3) est
une preoccupation du provider GCP, pas du cache lui-meme.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStore(ABC):
    @abstractmethod
    def resolve(self, logical: str) -> Path:
        """Chemin local pour ce chemin logique (cree les dossiers parents)."""

    @abstractmethod
    def pull(self, logical: str) -> Path:
        """Rapatrie l'artefact distant vers son emplacement local et le renvoie.
        No-op pour un backend deja local."""

    @abstractmethod
    def push(self, logical: str) -> None:
        """Publie l'artefact local vers le stockage de verite. No-op pour un
        backend deja local."""

    @abstractmethod
    def exists(self, logical: str) -> bool:
        ...


class LocalStore(ArtifactStore):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def resolve(self, logical: str) -> Path:
        path = self.root / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def pull(self, logical: str) -> Path:
        return self.resolve(logical)

    def push(self, logical: str) -> None:
        return None

    def exists(self, logical: str) -> bool:
        return (self.root / logical).exists()


class GcsStore(ArtifactStore):
    """Miroir local sous `<data_root>/.gcs-mirror/`, verite dans
    `<bucket>/<prefix>/<logical>`. `gcloud` doit etre installe et
    authentifie (`gcloud auth login`)."""

    def __init__(self, bucket: str, prefix: str, mirror_root: Path) -> None:
        self.bucket = bucket.rstrip("/")
        self.prefix = prefix.strip("/")
        self.mirror_root = Path(mirror_root)

    def _uri(self, logical: str) -> str:
        return f"{self.bucket}/{self.prefix}/{logical}".rstrip("/")

    @staticmethod
    def _run(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["gcloud", "storage", *args], check=check, capture_output=True, text=True)

    def resolve(self, logical: str) -> Path:
        path = self.mirror_root / logical
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def pull(self, logical: str) -> Path:
        """Copie `<bucket>/<prefix>/<logical>` (fichier OU dossier) vers le
        miroir local, en le posant exactement a `resolve(logical)`."""
        local = self.resolve(logical)
        local.parent.mkdir(parents=True, exist_ok=True)
        # `cp --recursive gs://.../foo /dest/` -> /dest/foo (fichier ou dossier)
        self._run(["cp", "--recursive", self._uri(logical), str(local.parent) + "/"])
        return local

    def push(self, logical: str) -> None:
        local = self.mirror_root / logical
        parent_uri = self._uri(logical).rsplit("/", 1)[0]
        self._run(["cp", "--recursive", str(local), parent_uri + "/"])

    def exists(self, logical: str) -> bool:
        return self._run(["ls", self._uri(logical)], check=False).returncode == 0


def make_store(config) -> ArtifactStore:
    """`config` : `songo_ai.cloud.config.RuntimeConfig`."""
    if config.provider == "gcp":
        return GcsStore(config.gcp.bucket, config.gcp.prefix, config.data_root / ".gcs-mirror")
    return LocalStore(config.data_root)
