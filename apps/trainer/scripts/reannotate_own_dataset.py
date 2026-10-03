#!/usr/bin/env python3
"""Re-annote un dataset DEJA au format interne (nos propres releases, ex.
data/dataset_v001_10k, data/dataset_v002_100k, ou une release fusionnee
comme data/dataset_v003_110k) avec le professeur ACTUEL -- typiquement
pour beneficier d'un correctif du professeur applique apres la generation
initiale (ex. le bonus territoire bidoua/Yinda, juillet 2026 :
`default_evaluate` en beneficie automatiquement mais les labels DEJA
ecrits sur disque, eux, restent geles a l'ancienne version tant qu'on ne
relance pas l'annotation).

Contrairement a `reannotate_external_dataset.py` (donnees d'un pipeline
externe, encodage de plateau different a convertir), ici `state` est deja
au format canonique attendu par `SongoLegacyGame` (cf.
songo_ai.dataset.schema.canonicalize_board) -- pas de conversion, juste
une nouvelle annotation. Les splits (train/val/test) du fichier d'entree
sont preserves tels quels (pas de re-melange).

ATTENTION CACHE (piege reel, verifie juillet 2026) : la cle du cache
d'annotations (`AnnotationCache`/`cache_key`) est `hash(zobrist, config.
fingerprint())` -- `TeacherConfig.fingerprint()` ne sait RIEN du
correctif bidoua (ce n'est pas un champ de config, `default_evaluate` est
code en dur cote `negamax_search`). Reutiliser un `--cache-dir` DEJA
peuple avec la MEME config AVANT un futur correctif du professeur
renverrait donc silencieusement les anciennes annotations, sans erreur.
Toujours donner un `--cache-dir` FRAIS et clairement nomme (ex. suffixe
"_bidoua_v1") a chaque campagne de re-annotation qui change le
comportement du professeur, jamais reutiliser un cache d'une campagne
precedente.

Usage (piloter d'abord, meme discipline que reannotate_external_dataset.py) :
    .venv/bin/python apps/trainer/scripts/reannotate_own_dataset.py \\
        --input-dir data/dataset_v003_110k \\
        --out-dir data/dataset_v003_110k_bidoua_v1 \\
        --cache-dir data/dataset_v003_110k_bidoua_v1/annotation_cache \\
        --limit 200 --num-workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

from songo_ai.dataset.schema import DATASET_VERSION, RULES_VERSION, annotation_to_observation
from songo_ai.dataset.validate import validate_observation
from songo_ai.songo.fast_rules import FastSongoGame
from songo_ai.teachers import PREMIUM, AnnotationCache, DeepTeacher, TeacherConfig

# Meme config que reannotate_external_dataset.py (REANNOTATE_CONFIG) :
# profondeur haute + budget de temps genereux, arret anticipe sur
# stabilite -- coherent avec "notre professeur excellent", la meme
# qualite de verite tactique pour toutes les campagnes de re-annotation.
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

SPLIT_NAMES = ("train", "val", "test")


def _annotate_payload(payload: Tuple[Tuple[int, ...], str, int, TeacherConfig, str]):
    board, trajectory_id, move_number, teacher_config, cache_root = payload
    game = FastSongoGame.from_board(board, 1)  # state deja canonique -> toujours "PLAYER_ONE" au trait
    teacher = DeepTeacher(teacher_config)
    cache = AnnotationCache(Path(cache_root))
    annotation = cache.get_or_annotate(teacher, game)
    return annotation, trajectory_id, move_number


def _load_split(path: Path, shard_index: int, num_shards: int, limit: int | None) -> List[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # Sharding deterministe par index absolu dans le fichier (stable d'une
    # execution a l'autre, fichier source statique) : chaque position
    # appartient a EXACTEMENT un shard, jamais retraitee par un autre --
    # essentiel pour paralleliser sur plusieurs sessions Colab en meme
    # temps sans travail en double ni ecriture concurrente sur le meme
    # cache SQLite (chaque shard a son propre --cache-dir/--out-dir).
    shard = [row for i, row in enumerate(rows) if i % num_shards == shard_index]
    return shard[:limit] if limit is not None else shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="dataset existant (train/val/test.jsonl)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, required=True, help="cache d'annotations SQLite -- DOIT etre frais, voir ATTENTION en tete de fichier"
    )
    parser.add_argument("--limit", type=int, default=None, help="pilote : limite le nombre de positions PAR split (applique APRES le sharding)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0, help="index de ce shard, 0..num-shards-1 (ex. plusieurs sessions Colab en parallele)")
    parser.add_argument("--num-shards", type=int, default=1, help="nombre total de shards -- chaque position va a EXACTEMENT un shard")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"--shard-index doit etre dans [0, {args.num_shards})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = str(args.cache_dir)

    # Pre-cree le fichier cache.db (WAL) UNE fois dans le process principal :
    # plusieurs workers l'initialisant tous en meme temps au premier lancement
    # peuvent se disputer l'ecrivain SQLite ("database is locked").
    AnnotationCache(args.cache_dir)

    checksums = {}
    counts = {}
    total_start = time.perf_counter()

    for split_name in SPLIT_NAMES:
        rows = _load_split(args.input_dir / f"{split_name}.jsonl", args.shard_index, args.num_shards, args.limit)
        if not rows:
            continue
        payloads = [
            (tuple(row["state"]), row["trajectory_id"], row["move_number"], REANNOTATE_CONFIG, cache_root)
            for row in rows
        ]

        print(f"{split_name}: {len(payloads)} positions a re-annoter")
        start = time.perf_counter()
        if args.num_workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
                results = list(pool.map(_annotate_payload, payloads))
        else:
            results = [_annotate_payload(p) for p in payloads]
        elapsed = time.perf_counter() - start
        rate = elapsed / max(1, len(payloads))
        print(f"{split_name}: annote en {elapsed/60:.1f} min ({rate:.2f}s/position)")

        observations = []
        problems_found = []
        for annotation, trajectory_id, move_number in results:
            obs = annotation_to_observation(annotation, trajectory_id=trajectory_id, move_number=move_number)
            problems = validate_observation(obs)
            if problems:
                problems_found.append(f"{trajectory_id}#{move_number}: {problems}")
                continue
            observations.append(obs)

        if problems_found:
            print(f"ATTENTION: {len(problems_found)} observation(s) invalide(s), exemples: {problems_found[:5]}")

        shard_path = args.out_dir / f"{split_name}.jsonl"
        with shard_path.open("w") as f:
            for obs in observations:
                f.write(json.dumps(obs.to_json_dict()) + "\n")
        checksums[split_name] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        counts[split_name] = len(observations)

    total_elapsed = time.perf_counter() - total_start
    manifest = {
        "dataset_version": DATASET_VERSION,
        "rules_version": RULES_VERSION,
        "generated_at_unix": time.time(),
        "source": f"re-annotation de {args.input_dir} avec le professeur actuel (correctif bidoua/Yinda inclus)",
        "source_input_dir": str(args.input_dir),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total_positions": sum(counts.values()),
        "teacher_config": asdict(REANNOTATE_CONFIG),
        "counts_per_split": counts,
        "checksums_sha256": checksums,
        "reannotation_elapsed_s": total_elapsed,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{sum(counts.values())} positions ecrites -> {args.out_dir} ({total_elapsed/60:.1f} min au total)")


if __name__ == "__main__":
    main()
