#!/usr/bin/env python3
"""Palier 100k (etape 7, config profonde). Historiquement le point d'entree
GCP ; depuis la migration multi-provider (`songo_ai.cloud`), ce n'est plus
qu'un raccourci vers le runtime unifie, execute en provider `local` -- que
ce "local" soit ta machine Windows ou la VM Compte Engine jetable, le code
est le meme (voir docs/trainer/provider-architecture.md).

Equivalent direct :
    python -m songo_ai.cloud run build --positions 100000 --seed 456 --preset deep

Toujours appele par gcp_startup_script.sh avec (num_positions, seed) ; sur
la VM, SONGO_DATA_ROOT=/tmp fait ecrire dans /tmp/dataset_v002_<N>/ comme
avant, et le startup script upload la release + le cache vers GCS."""

from __future__ import annotations

import json
import os
import sys
import time

from songo_ai.cloud import BuildSpec, load_config
from songo_ai.cloud.providers import LocalProvider


def main() -> None:
    num_positions = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 456

    # Compat startup script GCP : OUT_DIR=/tmp/dataset_v002_<N>, cache dans
    # OUT_DIR/annotation_cache. On force donc data_root=/tmp et un nom sans
    # prefixe "datasets/".
    os.environ.setdefault("SONGO_DATA_ROOT", "/tmp")
    config = load_config().with_overrides(provider="local")
    spec = BuildSpec(
        num_positions=num_positions,
        seed=seed,
        teacher_preset="deep",
        dataset_name=f"dataset_v002_{num_positions}",
    )

    start = time.perf_counter()
    manifest = LocalProvider(config).run_build(spec)
    elapsed = time.perf_counter() - start
    print(f"TERMINE en {elapsed / 3600:.2f}h pour {manifest['total_positions']} positions")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
