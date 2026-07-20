#!/usr/bin/env python3
"""Convertit et re-annote avec NOTRE professeur des vraies parties
recues d'un pipeline externe (`songo-model-stockfish-for-google-collab`),
plutot que de garder ses labels ("minimax insane" : profondeur FIXE 22,
seulement 1,2s/position -- notre prof est adaptatif et va nettement plus
loin en pratique sur un budget de temps comparable ou superieur, cf.
session du juillet 2026).

Conversion de plateau et resolution du joueur au trait : voir
`songo_ai.dataset.external_import` (methode et garanties detaillees
la-bas -- validee sur 15 000 echantillons, 0 echec).

Concu pour tourner sur Google Colab (executeur PARALLELE tolerant aux
deconnexions) : le cache d'annotations SQLite (meme mecanisme que le
pipeline GCP habituel) permet de reprendre exactement ou on s'est arrete
si la session Colab se coupe, tant que `--cache-dir` pointe vers un
dossier Google Drive (donc persistant entre deux sessions).

Usage (pilote d'abord, section 11.3 du plan : jamais le volume complet
sans avoir chiffre le cout/temps reel) :
    .venv/bin/python apps/trainer/scripts/reannotate_external_dataset.py \\
        --input-dir data/dataset_full_matrix_merged_all_colabs \\
        --out-dir data/dataset_external_reannotated \\
        --cache-dir data/dataset_external_reannotated/annotation_cache \\
        --limit 300 --num-workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from songo_ai.dataset.external_import import ExternalPosition, load_external_npz_split
from songo_ai.dataset.schema import DATASET_VERSION, RULES_VERSION, annotation_to_observation
from songo_ai.dataset.validate import validate_observation
from songo_ai.songo.fast_rules import FastSongoGame
from songo_ai.songo.rules import zobrist_hash, State
from songo_ai.teachers import PREMIUM, AnnotationCache, DeepTeacher, TeacherConfig

# Generalement pas de limite de temps/profondeur stricte pour ce travail
# (Colab, pas de facture GCP a surveiller au coup par coup) : profondeur
# tres haute + budget de temps genereux, l'arret anticipe (stability_window/
# min_margin) evite de gaspiller du temps sur les positions deja claires.
REANNOTATE_CONFIG = TeacherConfig(
    initial_depth=8,
    depth_step=4,
    max_depth=40,
    max_nodes=10_000_000,
    max_time_s=60.0,
    stability_window=4,
    min_margin=10.0,
    tier=PREMIUM,
)


def _annotate_payload(payload: Tuple[Tuple[int, ...], int, str, int, TeacherConfig, str]):
    board, turn, game_id, sample_index, teacher_config, cache_root = payload
    game = FastSongoGame.from_board(board, turn)
    teacher = DeepTeacher(teacher_config)
    cache = AnnotationCache(Path(cache_root))
    annotation = cache.get_or_annotate(teacher, game)
    return annotation, game_id, sample_index


def _dedupe_global(positions: List[ExternalPosition]) -> List[ExternalPosition]:
    """Deduplication par hash Zobrist canonique, toutes parties/splits
    confondus (contrairement au dataset auto-genere, ici l'origine est
    deja des parties completes distinctes -- pas de plafonnement par
    frequence necessaire, juste eliminer les positions strictement
    identiques, frequentes en debut de partie)."""
    seen = set()
    kept = []
    for p in positions:
        h = zobrist_hash(State(tuple(p.board), p.turn))
        if h in seen:
            continue
        seen.add(h)
        kept.append(p)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="dossier contenant train/validation/test.npz")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True, help="cache d'annotations SQLite (mettre sur Drive pour la reprise entre sessions Colab)")
    parser.add_argument("--limit", type=int, default=None, help="pilote : limite le nombre de positions (avant dedup)")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = str(args.cache_dir)

    all_positions: List[ExternalPosition] = []
    for split_name in ("train", "validation", "test"):
        npz_path = args.input_dir / f"{split_name}.npz"
        if not npz_path.exists():
            continue
        positions = load_external_npz_split(npz_path)
        if args.limit is not None:
            positions = positions[: args.limit]
        print(f"{split_name}: {len(positions)} positions chargees depuis {npz_path.name}")
        all_positions.extend(positions)

    print(f"total avant dedup: {len(all_positions)}")
    all_positions = _dedupe_global(all_positions)
    print(f"total apres dedup (hash Zobrist canonique): {len(all_positions)}")

    payloads = [
        (p.board, p.turn, p.game_id, i, REANNOTATE_CONFIG, cache_root) for i, p in enumerate(all_positions)
    ]

    # Pre-cree le fichier cache.db (WAL) UNE fois dans le process principal :
    # plusieurs workers l'initialisant tous en meme temps au premier lancement
    # peuvent se disputer l'ecrivain SQLite ("database is locked").
    AnnotationCache(args.cache_dir)

    start = time.perf_counter()
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            results = list(pool.map(_annotate_payload, payloads))
    else:
        results = [_annotate_payload(payload) for payload in payloads]
    elapsed = time.perf_counter() - start

    observations = []
    problems_found: List[str] = []
    game_ids_by_obs: List[str] = []
    for annotation, game_id, sample_index in results:
        obs = annotation_to_observation(annotation, trajectory_id=game_id, move_number=sample_index)
        problems = validate_observation(obs)
        if problems:
            problems_found.append(f"{game_id}#{sample_index}: {problems}")
            continue
        observations.append(obs)
        game_ids_by_obs.append(game_id)

    print(f"annote en {elapsed/60:.1f} min ({elapsed/max(1,len(all_positions)):.2f}s/position en moyenne)")
    if problems_found:
        print(f"ATTENTION: {len(problems_found)} observation(s) invalide(s), exemples: {problems_found[:5]}")

    # Split par PARTIE (game_id), pas par ligne isolee (section 6.6) --
    # regroupe les observations d'une meme partie externe dans le meme split.
    by_game: Dict[str, list] = defaultdict(list)
    for obs, gid in zip(observations, game_ids_by_obs):
        by_game[gid].append(obs)

    import random as _random

    game_ids = list(by_game.keys())
    _random.Random(0).shuffle(game_ids)
    n = len(game_ids)
    n_val = max(1, int(n * 0.1)) if n >= 3 else 0
    n_test = max(1, int(n * 0.1)) if n >= 3 else 0
    val_ids = set(game_ids[:n_val])
    test_ids = set(game_ids[n_val : n_val + n_test])

    splits: Dict[str, list] = {"train": [], "val": [], "test": []}
    for gid, obs_list in by_game.items():
        split = "val" if gid in val_ids else ("test" if gid in test_ids else "train")
        splits[split].extend(obs_list)

    checksums = {}
    counts = {}
    for split_name, obs_list in splits.items():
        shard_path = args.out_dir / f"{split_name}.jsonl"
        with shard_path.open("w") as f:
            for obs in obs_list:
                f.write(json.dumps(obs.to_json_dict()) + "\n")
        checksums[split_name] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        counts[split_name] = len(obs_list)

    manifest = {
        "dataset_version": DATASET_VERSION,
        "rules_version": RULES_VERSION,
        "generated_at_unix": time.time(),
        "source": "songo-model-stockfish-for-google-collab (vraies parties, re-annotees avec notre prof)",
        "source_input_dir": str(args.input_dir),
        "total_positions": sum(counts.values()),
        "teacher_config": asdict(REANNOTATE_CONFIG),
        "counts_per_split": counts,
        "checksums_sha256": checksums,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{sum(counts.values())} positions ecrites -> {args.out_dir}")


if __name__ == "__main__":
    main()
