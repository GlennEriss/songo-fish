"""Pipeline de construction d'un dataset annote (etape 5) : genere des
trajectoires legales, echantillonne des positions, les annote avec le
professeur (avec cache), valide chaque observation, puis ecrit une release
immuable (manifeste + checksums + splits PAR PARTIE ENTIERE, section 6.6 :
"jamais par ligne isolee")."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from songo_ai.generation import generate_trajectories, make_mixed_agent, sample_positions
from songo_ai.generation.sampling import classify_phase
from songo_ai.songo.fast_rules import FastSongoGame
from songo_ai.teachers import STANDARD, AnnotationCache, DeepTeacher, TeacherConfig

from .schema import DATASET_VERSION, RULES_VERSION, Observation, annotation_to_observation
from .validate import DistributionReport, build_distribution_report, validate_observation

# Niveau standard par defaut (section 5.4 : "Standard: Majorite du corpus" ;
# "Enrichi: Positions ambigues" est reserve a un sous-ensemble cible, pas a
# la totalite du dataset -- une passe d'enrichissement selective (ex: sur
# les positions a faible top1_top2_margin) pourra etre ajoutee plus tard
# sans regenerer tout le corpus).
DEFAULT_TEACHER_CONFIG = TeacherConfig(
    initial_depth=4,
    depth_step=2,
    max_depth=14,
    max_nodes=300_000,
    max_time_s=5.0,
    stability_window=3,
    min_margin=15.0,
    tier=STANDARD,
)

# Longueur moyenne de partie observee (bench_rules.py, self-play aleatoire) :
# sert uniquement a dimensionner le nombre de trajectoires a generer.
_APPROX_GAME_LENGTH = 100


def _annotate_payload(payload: Tuple[Tuple[int, ...], int, str, int, TeacherConfig, str]):
    """Fonction top-level (picklable) : un worker de process pool recree son
    propre professeur/cache, l'annotation d'une position etant independante
    de toutes les autres (section 4.3 : "distribuer les positions
    independantes entre plusieurs processus CPU")."""
    board, turn, trajectory_id, move_number, teacher_config, cache_root = payload
    game = FastSongoGame.from_board(board, turn)
    teacher = DeepTeacher(teacher_config)
    cache = AnnotationCache(Path(cache_root))
    annotation = cache.get_or_annotate(teacher, game)
    return annotation, trajectory_id, move_number


def _log_progress(fresh: int, fresh_total: int, absolute: int, total: int, start: float) -> None:
    if fresh != fresh_total and fresh % max(1, fresh_total // 100) != 0:
        return
    elapsed = time.perf_counter() - start
    rate = fresh / elapsed if elapsed else 0.0  # cadence du travail restant
    eta = (fresh_total - fresh) / rate if rate else 0.0
    print(
        f"[build] {absolute}/{total} annotees ({absolute / total * 100:.1f}%) "
        f"- {rate:.1f}/s - ETA {eta / 60:.0f} min",
        file=sys.stderr,
        flush=True,
    )


def _run_annotation(payloads: list, num_workers: int, done_offset: int, total: int) -> list:
    """Annote `payloads`, en journalisant la progression. Le cache SQLite
    commit apres CHAQUE position (`AnnotationCache.put`) : si le process est
    tue en cours de route, tout ce qui est deja annote est sur le disque et
    un relancement (memes seed/out_dir) le saute d'entree (voir
    build_dataset)."""
    out: list = []
    fresh_total = len(payloads)
    if fresh_total == 0:
        return out
    start = time.perf_counter()
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for i, res in enumerate(pool.map(_annotate_payload, payloads), 1):
                out.append(res)
                _log_progress(i, fresh_total, done_offset + i, total, start)
    else:
        for i, payload in enumerate(payloads, 1):
            out.append(_annotate_payload(payload))
            _log_progress(i, fresh_total, done_offset + i, total, start)
    return out


def build_dataset(
    num_positions: int,
    out_dir: Path,
    seed: int = 0,
    teacher_config: Optional[TeacherConfig] = None,
    trajectory_multiplier: int = 4,
    max_moves: int = 300,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    num_workers: int = 1,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_config = teacher_config or DEFAULT_TEACHER_CONFIG

    def agent_factory(rng: random.Random):
        return make_mixed_agent(rng, random_weight=0.4)

    num_trajectories = max(4, (num_positions * trajectory_multiplier) // _APPROX_GAME_LENGTH + 1)
    raw_positions = generate_trajectories(agent_factory, num_trajectories, seed=seed, max_moves=max_moves)
    sampled = sample_positions(raw_positions, target_count=num_positions, seed=seed)

    cache_root = str(out_dir / "annotation_cache")
    # Pre-cree le fichier cache.db (WAL) UNE fois dans le process principal :
    # plusieurs workers l'initialisant en meme temps au tout premier
    # lancement se disputent l'ecrivain SQLite ("database is locked"). Meme
    # precaution que apps/trainer/scripts/reannotate_own_dataset.py.
    cache = AnnotationCache(Path(cache_root))
    phase_by_id = {(p.trajectory_id, p.move_number): classify_phase(p) for p in sampled}

    # Reprise : l'echantillonnage est deterministe (meme seed -> memes
    # positions), et le cache d'annotations (SQLite, commit apres chaque
    # position) survit a un arret brutal. On calcule le hash de chaque
    # position et on ne redonne au pool QUE celles qui manquent -- une
    # reprise ne recalcule rien de deja fait, et ne repasse meme pas les
    # positions faites par des workers.
    cached_results: list = []
    todo: list = []
    for p in sampled:
        zhash = FastSongoGame.from_board(p.state.board, p.state.turn).zobrist_hash()
        hit = cache.get(zhash, teacher_config)
        if hit is not None:
            cached_results.append((hit, p.trajectory_id, p.move_number))
        else:
            todo.append(
                (p.state.board, p.state.turn, p.trajectory_id, p.move_number, teacher_config, cache_root)
            )
    cache.close()

    total = len(sampled)
    if cached_results:
        print(
            f"[build] reprise : {len(cached_results)}/{total} deja en cache, {len(todo)} a annoter",
            file=sys.stderr,
            flush=True,
        )
    fresh_results = _run_annotation(todo, num_workers, done_offset=len(cached_results), total=total)
    results = cached_results + fresh_results

    observations: List[Observation] = []
    phases: List[str] = []
    problems_found: List[str] = []

    for annotation, trajectory_id, move_number in results:
        obs = annotation_to_observation(annotation, trajectory_id, move_number)
        problems = validate_observation(obs)
        if problems:
            problems_found.append(f"{trajectory_id}#{move_number}: {problems}")
            continue
        observations.append(obs)
        phases.append(phase_by_id[(trajectory_id, move_number)])

    if problems_found:
        raise RuntimeError(f"{len(problems_found)} observation(s) invalide(s), exemples: {problems_found[:5]}")

    report = build_distribution_report(observations, phases)
    splits = _split_by_game(observations, split_ratios, seed)
    return _write_release(out_dir, splits, report, teacher_config, seed, requested=num_positions)


def _split_by_game(
    observations: List[Observation], ratios: Tuple[float, float, float], seed: int
) -> Dict[str, List[Observation]]:
    by_game: Dict[str, List[Observation]] = defaultdict(list)
    for obs in observations:
        by_game[obs.trajectory_id].append(obs)

    game_ids = list(by_game.keys())
    random.Random(seed).shuffle(game_ids)

    n = len(game_ids)
    # max(1, ...) : avec peu de parties (petits pilotes), un arrondi vers le
    # bas peut sinon donner un split entierement vide alors que sa part est
    # non nulle et qu'il reste assez de parties pour en attribuer une.
    n_val = max(1, int(n * ratios[1])) if ratios[1] > 0 and n >= 3 else int(n * ratios[1])
    n_test_target = max(1, int(n * ratios[2])) if ratios[2] > 0 and n >= 3 else int(n * ratios[2])
    n_train = max(0, n - n_val - n_test_target)
    train_ids = set(game_ids[:n_train])
    val_ids = set(game_ids[n_train : n_train + n_val])

    splits: Dict[str, List[Observation]] = {"train": [], "val": [], "test": []}
    for game_id, obs_list in by_game.items():
        split = "train" if game_id in train_ids else ("val" if game_id in val_ids else "test")
        splits[split].extend(obs_list)
    return splits


def _write_release(
    out_dir: Path,
    splits: Dict[str, List[Observation]],
    report: DistributionReport,
    teacher_config: TeacherConfig,
    seed: int,
    requested: int,
) -> dict:
    checksums: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for split_name, obs_list in splits.items():
        shard_path = out_dir / f"{split_name}.jsonl"
        with shard_path.open("w") as f:
            for obs in obs_list:
                f.write(json.dumps(obs.to_json_dict()) + "\n")
        checksums[split_name] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        counts[split_name] = len(obs_list)

    manifest = {
        "dataset_version": DATASET_VERSION,
        "rules_version": RULES_VERSION,
        "generated_at_unix": time.time(),
        "seed": seed,
        "requested_positions": requested,
        "total_positions": sum(counts.values()),
        "teacher_config": asdict(teacher_config),
        "counts_per_split": counts,
        "checksums_sha256": checksums,
        "distribution_report": asdict(report),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
