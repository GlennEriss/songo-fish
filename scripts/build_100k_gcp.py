#!/usr/bin/env python3
"""Palier 100k (etape 7, config profonde) - destine a tourner sur une VM
GCP (voir scripts/gcp_startup_script.sh). Config "tres profonde mais
budget-consciente" validee localement (moyenne 6.5s/position, mediane
0.4s, ~17% des positions tapent le plafond de noeuds)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from songo_ai.dataset import build_dataset
from songo_ai.teachers import STANDARD, TeacherConfig

DEEP_CONFIG = TeacherConfig(
    initial_depth=6,
    depth_step=2,
    max_depth=28,
    max_nodes=3_000_000,
    max_time_s=45.0,
    stability_window=3,
    min_margin=15.0,
    tier=STANDARD,
)


def main() -> None:
    # Argument optionnel : nombre de positions (defaut 100000). Permet de
    # reutiliser ce meme script pour un test a blanc rapide (ex: 30) avant
    # de lancer le vrai palier 100k (section 11.3 : micro-pilote d'abord).
    num_positions = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    out_dir = Path(f"/tmp/dataset_v002_{num_positions}")
    start = time.perf_counter()
    manifest = build_dataset(
        num_positions=num_positions,
        out_dir=out_dir,
        seed=456,
        teacher_config=DEEP_CONFIG,
        trajectory_multiplier=4,
        max_moves=300,
        # S'adapte a la machine : 8 workers sur un e2-standard-8, 56 sur un
        # c2d-highcpu-56, etc. L'annotation est CPU-bound et embarrassingly
        # parallel, on prend tous les coeurs disponibles.
        num_workers=os.cpu_count() or 8,
    )
    elapsed = time.perf_counter() - start
    print(f"TERMINE en {elapsed / 3600:.2f}h pour {manifest['total_positions']} positions")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
