#!/usr/bin/env python3
"""Annote avec NOTRE professeur les positions de VRAIES parties humaines
(dataset `songo-real-move-v1`, produit par build_ordered_real_moves.py a
partir des exports Firebase -- voir data/real_matches/match_moves_v1.jsonl).

Contrairement au dataset externe (`reannotate_external_dataset.py`, parties
d'un pipeline minimax tiers) et a nos propres releases synthetiques
(`reannotate_own_dataset.py`, positions generees par nos agents), ici les
positions viennent de VRAIS joueurs humains -- distribution de jeu reelle,
particulierement precieuse. On ne garde PAS le coup humain comme cible
(un humain n'est pas une reference fiable) : on ne garde que la POSITION,
et notre prof (DeepTeacher, bidoua/Yinda inclus) produit best_action/
action_values/WDL comme pour les autres datasets. Le coup humain reste
tracable via trajectory_id (match_id) + move_number (ply) si besoin.

Format d'entree (une ligne JSON par coup, schema songo-real-move-v1) : on
lit `board_before` (plateau 16 entiers) + `player_position` (joueur au
trait). `state` est deja la canonicalisation de ces deux champs (verifie),
mais on repart de board_before/player_position pour ne dependre que du
plateau brut.

Sharding (--shard-index/--num-shards) : identique aux autres scripts, pour
paralleliser sur plusieurs sessions Colab (cf. colab_reannotate_bidoua.ipynb).

Usage (piloter d'abord) :
    python3 apps/trainer/scripts/annotate_real_matches.py \\
        --input data/real_matches/match_moves_v1.jsonl \\
        --out-dir data/dataset_real_matches_bidoua_v1 \\
        --cache-dir data/dataset_real_matches_bidoua_v1/annotation_cache \\
        --limit 30 --num-workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random as _random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from songo_ai.dataset.schema import DATASET_VERSION, RULES_VERSION, annotation_to_observation
from songo_ai.dataset.validate import validate_observation
from songo_ai.songo.fast_rules import FastSongoGame
from songo_ai.songo.rules import State, zobrist_hash
from songo_ai.teachers import PREMIUM, AnnotationCache, DeepTeacher, TeacherConfig

# Meme config "prof excellent" que les autres campagnes de re-annotation
# (profondeur haute, budget de temps genereux, arret anticipe sur
# stabilite) -- coherence de la verite tactique sur tous les datasets.
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
    board, turn, match_id, ply, teacher_config, cache_root = payload
    game = FastSongoGame.from_board(board, turn)
    teacher = DeepTeacher(teacher_config)
    cache = AnnotationCache(Path(cache_root))
    annotation = cache.get_or_annotate(teacher, game)
    return annotation, match_id, ply


def _load_and_dedupe(input_path: Path) -> List[Tuple[Tuple[int, ...], int, str, int]]:
    """Charge les coups, garde une position par hash Zobrist canonique
    (premiere occurrence -- frequente en debut de partie). Renvoie des
    tuples (board_before, player_position, match_id, ply)."""
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    seen = set()
    kept: List[Tuple[Tuple[int, ...], int, str, int]] = []
    for r in rows:
        board = tuple(r["board_before"])
        turn = r["player_position"]
        h = zobrist_hash(State(board, turn))
        if h in seen:
            continue
        seen.add(h)
        kept.append((board, turn, r["match_id"], r["ply"]))
    print(f"{len(rows)} coups lus, {len(kept)} positions distinctes (hash Zobrist canonique)")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="match_moves_v1.jsonl (schema songo-real-move-v1)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True, help="cache d'annotations SQLite -- DOIT etre frais (cf. autres scripts de re-annotation)")
    parser.add_argument("--limit", type=int, default=None, help="pilote : limite le nombre de positions (apres dedup et sharding)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0, help="index de ce shard, 0..num-shards-1")
    parser.add_argument("--num-shards", type=int, default=1, help="nombre total de shards")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index doit etre dans [0, {args.num_shards})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = str(args.cache_dir)

    positions = _load_and_dedupe(args.input)

    if args.num_shards > 1:
        positions = [p for i, p in enumerate(positions) if i % args.num_shards == args.shard_index]
        print(f"shard {args.shard_index}/{args.num_shards}: {len(positions)} positions a annoter")
    if args.limit is not None:
        positions = positions[: args.limit]

    payloads = [(board, turn, match_id, ply, REANNOTATE_CONFIG, cache_root) for board, turn, match_id, ply in positions]

    # Pre-cree cache.db (WAL) UNE fois dans le process principal : evite la
    # course "database is locked" si tous les workers l'initialisent en meme temps.
    AnnotationCache(args.cache_dir)

    start = time.perf_counter()
    if args.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            results = list(pool.map(_annotate_payload, payloads))
    else:
        results = [_annotate_payload(p) for p in payloads]
    elapsed = time.perf_counter() - start
    print(f"annote en {elapsed/60:.1f} min ({elapsed/max(1,len(payloads)):.2f}s/position en moyenne)")

    observations = []
    match_ids_by_obs: List[str] = []
    problems_found: List[str] = []
    for annotation, match_id, ply in results:
        obs = annotation_to_observation(annotation, trajectory_id=match_id, move_number=ply)
        problems = validate_observation(obs)
        if problems:
            problems_found.append(f"{match_id}#{ply}: {problems}")
            continue
        observations.append(obs)
        match_ids_by_obs.append(match_id)
    if problems_found:
        print(f"ATTENTION: {len(problems_found)} observation(s) invalide(s), exemples: {problems_found[:5]}")

    # Split par PARTIE (match_id), pas par position isolee : toutes les
    # positions d'un meme match vont dans le meme split (evite qu'une
    # partie fuite de train vers val/test).
    by_match: Dict[str, list] = defaultdict(list)
    for obs, mid in zip(observations, match_ids_by_obs):
        by_match[mid].append(obs)

    match_ids = list(by_match.keys())
    _random.Random(0).shuffle(match_ids)
    n = len(match_ids)
    n_val = max(1, int(n * 0.1)) if n >= 3 else 0
    n_test = max(1, int(n * 0.1)) if n >= 3 else 0
    val_ids = set(match_ids[:n_val])
    test_ids = set(match_ids[n_val : n_val + n_test])

    splits: Dict[str, list] = {"train": [], "val": [], "test": []}
    for mid, obs_list in by_match.items():
        split = "val" if mid in val_ids else ("test" if mid in test_ids else "train")
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
        "source": "vraies parties humaines (songo-real-move-v1), annotees avec notre prof (bidoua/Yinda inclus)",
        "source_input": str(args.input),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_positions": sum(counts.values()),
        "teacher_config": asdict(REANNOTATE_CONFIG),
        "counts_per_split": counts,
        "checksums_sha256": checksums,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{sum(counts.values())} positions ecrites -> {args.out_dir}")


if __name__ == "__main__":
    main()
