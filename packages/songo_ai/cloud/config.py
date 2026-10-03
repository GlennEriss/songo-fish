"""Configuration d'execution (section 11 : "Locale d'abord ; GCP en option").

Une seule source de verite pour repondre a trois questions, dans cet ordre
de priorite : valeurs par defaut < fichier `songo.toml` (a la racine du
depot) < variables d'environnement `SONGO_*`.

- QUEL provider de calcul : `local` (cette machine) ou `gcp` (VM Compute
  Engine jetable). Defaut `local` -- c'est ce que dit le plan directeur
  (section 11.3, et la porte d'etape "Locale d'abord").
- OU vivent les artefacts : datasets, caches d'annotations, checkpoints,
  registre. En local c'est un dossier du disque ; sur GCP c'est un bucket
  Cloud Storage (voir `songo_ai.cloud.storage`).
- COMBIEN de workers / QUEL device pour l'entrainement.

Aucun import de `torch` ici (dependance optionnelle `train`) : la
resolution du device fait un import paresseux.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    # packages/songo_ai/cloud/config.py -> remonter jusqu'a la racine du depot
    return Path(__file__).resolve().parents[3]


def _default_workers() -> int:
    """Moitie des coeurs logiques, borne a >=1. Volontairement PAS
    `os.cpu_count()` : chaque worker d'annotation instancie son propre
    professeur avec une table de transposition de plusieurs Mo
    (`max_nodes` jusqu'a 10 M) -- sur une machine portable 4 coeurs / 18 Go
    (le cas d'usage cible), saturer tous les threads fait surtout gonfler
    la RAM et la contention SQLite. A relever explicitement dans
    `songo.toml` si la machine a de la marge."""
    return max(1, (os.cpu_count() or 2) // 2)


@dataclass
class GcpConfig:
    bucket: str = "gs://songo-model-ai-vertex-bucket-001"
    prefix: str = "pilots"
    project: str = "songo-model-ai"
    zone: str = "us-central1-a"
    machine_type: str = "c2d-highcpu-32"
    image_family: str = "ubuntu-2404-lts-amd64"
    image_project: str = "ubuntu-os-cloud"
    boot_disk_size: str = "50GB"
    # Filet de securite budget (section 11.3 : "interdire tout job sans
    # maxRunDuration") : la VM est detruite passe ce delai, quoi qu'il
    # arrive. Le timeout dur du startup script est fixe un peu en dessous.
    max_run_duration_s: int = 43200  # 12 h


@dataclass
class RuntimeConfig:
    provider: str = "local"
    data_root: Path = field(default_factory=lambda: _repo_root() / "data")
    num_workers: int = field(default_factory=_default_workers)
    # "auto" -> cuda si dispo, sinon cpu. Rappel projet (notebook Colab,
    # memoire) : ce reseau est si petit que le GPU n'apporte souvent aucun
    # gain -- l'entrainement local sur CPU reste le defaut recommande, le
    # GPU ne se justifie qu'apres un changement d'architecture MAJOR.
    device: str = "cpu"
    gcp: GcpConfig = field(default_factory=GcpConfig)

    def with_overrides(self, **kwargs) -> "RuntimeConfig":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        if "data_root" in clean:
            clean["data_root"] = Path(clean["data_root"]).expanduser()
        updated = replace(self, **clean)
        if updated.provider not in ("local", "gcp"):
            raise ValueError(f"provider inconnu: {updated.provider!r}")
        return updated

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def _load_toml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f).get("runtime", {})


def load_config(config_path: Optional[Path] = None) -> RuntimeConfig:
    """defaults < songo.toml < env `SONGO_*`."""
    cfg = RuntimeConfig()

    toml_path = Path(config_path) if config_path else _repo_root() / "songo.toml"
    table = _load_toml(toml_path)
    gcp_table = table.get("gcp", {})

    if "provider" in table:
        cfg = replace(cfg, provider=str(table["provider"]))
    if "data_root" in table:
        cfg = replace(cfg, data_root=Path(table["data_root"]).expanduser())
    if "num_workers" in table:
        cfg = replace(cfg, num_workers=int(table["num_workers"]))
    if "device" in table:
        cfg = replace(cfg, device=str(table["device"]))
    if gcp_table:
        cfg = replace(cfg, gcp=replace(cfg.gcp, **{
            k: (int(v) if k == "max_run_duration_s" else str(v))
            for k, v in gcp_table.items()
            if k in GcpConfig.__dataclass_fields__
        }))

    env = os.environ
    if env.get("SONGO_PROVIDER"):
        cfg = replace(cfg, provider=env["SONGO_PROVIDER"])
    if env.get("SONGO_DATA_ROOT"):
        cfg = replace(cfg, data_root=Path(env["SONGO_DATA_ROOT"]).expanduser())
    if env.get("SONGO_NUM_WORKERS"):
        cfg = replace(cfg, num_workers=int(env["SONGO_NUM_WORKERS"]))
    if env.get("SONGO_DEVICE"):
        cfg = replace(cfg, device=env["SONGO_DEVICE"])
    if env.get("SONGO_GCS_BUCKET"):
        cfg = replace(cfg, gcp=replace(cfg.gcp, bucket=env["SONGO_GCS_BUCKET"]))
    if env.get("SONGO_GCS_PREFIX"):
        cfg = replace(cfg, gcp=replace(cfg.gcp, prefix=env["SONGO_GCS_PREFIX"]))

    if cfg.provider not in ("local", "gcp"):
        raise ValueError(f"provider inconnu: {cfg.provider!r} (attendu 'local' ou 'gcp')")
    cfg = replace(cfg, data_root=Path(cfg.data_root))
    return cfg
