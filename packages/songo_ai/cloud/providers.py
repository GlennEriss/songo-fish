"""Providers de calcul -- pattern Strategy (section 11 : "Locale d'abord ;
GCP en option apres benchmark").

- `LocalProvider` : execute le job dans ce process, sur cette machine.
  Couvre le pipeline quotidien -- dev, tests, generation <= ~100k
  positions, TOUT l'entrainement, TOUS les tournois. Le cache
  d'annotations est un fichier sur le disque : la "reprise apres crash"
  est automatique (le fichier persiste), aucune boucle de synchronisation.

- `GcpProvider` : provisionne une VM Compute Engine jetable pour UNE
  campagne d'annotation massive (1M+ positions), quand une porte de
  volume a deja ete franchie (section 10.3). Ne gere QUE `run_build` :
  l'entrainement et les tournois n'ont aucune raison de payer du cloud
  (reseau minuscule, quelques minutes de CPU) -- ils restent locaux.
"""

from __future__ import annotations

import json
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .config import RuntimeConfig
from .jobs import BuildSpec, MatchSpec, MergeSpec, TrainSpec
from .storage import ArtifactStore, make_store

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "apps" / "trainer" / "scripts"


class ComputeProvider(ABC):
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.store: ArtifactStore = make_store(config)

    @abstractmethod
    def run_build(self, spec: BuildSpec) -> dict:
        ...

    def run_train(self, spec: TrainSpec) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} ne gere pas l'entrainement -- il tourne toujours en local "
            "(reseau minuscule, quelques minutes de CPU). Utiliser le provider 'local'."
        )

    def run_tournament(self, spec: MatchSpec) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} ne gere pas les tournois -- ils tournent toujours en local. "
            "Utiliser le provider 'local'."
        )

    def run_merge(self, spec: MergeSpec) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} ne gere pas la fusion -- operation fichier, toujours locale. "
            "Utiliser le provider 'local'."
        )


class LocalProvider(ComputeProvider):
    def run_build(self, spec: BuildSpec) -> dict:
        from songo_ai.dataset import build_dataset

        out_dir = self.store.resolve(spec.resolved_name())
        manifest = build_dataset(
            num_positions=spec.num_positions,
            out_dir=out_dir,
            seed=spec.seed,
            teacher_config=spec.teacher_config(),
            trajectory_multiplier=spec.trajectory_multiplier,
            max_moves=spec.max_moves,
            num_workers=self.config.num_workers,
        )
        self.store.push(spec.resolved_name())
        return manifest

    def run_train(self, spec: TrainSpec) -> dict:
        from songo_ai.model import get_champion, train_and_register

        dataset_dir = self.store.pull(spec.dataset_name)
        checkpoint_dir = self.store.resolve("checkpoints")
        registry_path = self.store.resolve("checkpoints/registry.json")

        manifest = train_and_register(
            version=spec.version,
            train_shard=dataset_dir / "train.jsonl",
            val_shard=dataset_dir / "val.jsonl",
            dataset_manifest_path=dataset_dir / "manifest.json",
            checkpoint_dir=checkpoint_dir,
            registry_path=registry_path,
            promote=spec.promote,
            notes=spec.notes,
            epochs=spec.epochs,
            early_stopping_patience=spec.early_stopping_patience,
            device=self.config.resolved_device(),
        )
        self.store.push("checkpoints")
        champion = get_champion(registry_path=registry_path)
        return {
            "version": manifest.version,
            "metrics": manifest.metrics,
            "champion": champion["version"] if champion else None,
            "promoted": spec.promote,
        }

    def run_merge(self, spec: MergeSpec) -> dict:
        from songo_ai.dataset import merge_releases

        if len(spec.sources) < 2:
            raise ValueError("merge : au moins 2 releases sources")
        src_dirs = [self.store.pull(name) for name in spec.sources]
        out_dir = self.store.resolve(spec.out_name)
        manifest = merge_releases(src_dirs, out_dir, deduplicate=spec.deduplicate)
        self.store.push(spec.out_name)
        return manifest

    def run_tournament(self, spec: MatchSpec) -> dict:
        from songo_ai.evaluation import play_match

        self.store.pull("checkpoints")
        registry_path = self.store.resolve("checkpoints/registry.json")
        agent_a = self._make_agent(spec.agent_a, registry_path, spec.temperature)
        agent_b = self._make_agent(spec.agent_b, registry_path, spec.temperature)
        resume_path = self.store.resolve(
            f"tournaments/{spec.agent_a}_vs_{spec.agent_b}_s{spec.seed}_n{spec.num_games}.jsonl".replace(":", "")
        )
        result = play_match(
            agent_a,
            agent_b,
            num_games=spec.num_games,
            seed=spec.seed,
            opening_random_plies=spec.opening_random_plies,
            resume_path=resume_path,
        )
        return {
            "agent_a": spec.agent_a,
            "agent_b": spec.agent_b,
            "games": result.games,
            "distinct_games": result.distinct_games,
            "score_a": result.score_a,
            "win_rate_a": result.win_rate_a,
            "ci95": [result.ci_low, result.ci_high],
        }

    @staticmethod
    def _make_agent(spec: str, registry_path: Path, temperature: float):
        from songo_ai.generation import make_shallow_search_agent, random_agent
        from songo_ai.model import get_champion, load_model, make_network_agent
        from songo_ai.model.registry import load_registry

        if spec == "random":
            return random_agent
        if spec.startswith("minimax:"):
            depth = int(spec.split(":", 1)[1])
            return make_shallow_search_agent(max_depth=depth, max_nodes=50_000, max_time_s=1.0)

        if spec == "champion":
            entry = get_champion(registry_path=registry_path)
            if entry is None:
                raise ValueError("aucun champion dans le registre")
        else:
            version = spec[1:] if spec.startswith("v") else spec
            entry = load_registry(registry_path)["versions"].get(version)
            if entry is None:
                raise ValueError(f"version absente du registre: {version}")
        model = load_model(Path(entry["checkpoint_path"]), dropout=entry["architecture"]["dropout"])
        return make_network_agent(model, temperature=temperature)


class GcpProvider(ComputeProvider):
    """Provisionne une VM jetable qui execute `build_100k_gcp.py` en local
    SUR la VM (donc via `LocalProvider` cote VM), puis pousse la release
    vers GCS et s'auto-detruit. Reutilise tels quels les deux scripts
    shell existants, seul le bucket devient parametrable (env)."""

    def run_build(self, spec: BuildSpec) -> dict:
        gcp = self.config.gcp
        env = {"SONGO_GCS_BUCKET": gcp.bucket, "SONGO_GCS_PREFIX": gcp.prefix}

        self._sh("package_for_gcp.sh", env)

        instance = f"songo-dataset-{spec.num_positions}-{int(time.time())}"
        create = [
            "gcloud", "compute", "instances", "create", instance,
            "--project", gcp.project, "--zone", gcp.zone,
            "--machine-type", gcp.machine_type,
            "--image-family", gcp.image_family, "--image-project", gcp.image_project,
            "--boot-disk-size", gcp.boot_disk_size,
            "--provisioning-model", "STANDARD",
            "--max-run-duration", f"{gcp.max_run_duration_s}s",
            "--instance-termination-action", "DELETE",
            "--scopes", "cloud-platform",
            "--metadata-from-file", f"startup-script={_SCRIPTS_DIR / 'gcp_startup_script.sh'}",
            "--metadata", f"num-positions={spec.num_positions},seed={spec.seed}",
        ]
        subprocess.run(create, check=True)
        return {
            "provider": "gcp",
            "instance": instance,
            "zone": gcp.zone,
            "result_uri": f"{gcp.bucket}/{gcp.prefix}/dataset_v002_{spec.num_positions}/",
            "monitor": f"gcloud compute instances get-serial-port-output {instance} --zone {gcp.zone}",
            "note": "La VM s'auto-detruit a la fin (succes, echec ou maxRunDuration). "
                    "Recuperer la release avec: songo-cloud pull " + spec.resolved_name(),
        }

    @staticmethod
    def _sh(script: str, env: dict) -> None:
        import os

        subprocess.run(["bash", str(_SCRIPTS_DIR / script)], check=True, env={**os.environ, **env})


def make_provider(config: RuntimeConfig) -> ComputeProvider:
    return GcpProvider(config) if config.provider == "gcp" else LocalProvider(config)
