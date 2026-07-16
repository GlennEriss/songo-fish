"""Tests du mecanisme de snapshot/reprise du cache SQLite (section 11.3).
Valide localement la logique reutilisee par scripts/gcp_startup_script.sh :
un snapshot a chaud peut etre copie vers un nouveau repertoire de cache et
les entrees deja annotees y sont retrouvees, sans dependre d'un vrai run
GCP pour chaque verification."""

from __future__ import annotations

import shutil
from pathlib import Path

from songo_ai.songo.rules import SongoLegacyGame
from songo_ai.teachers import STANDARD, AnnotationCache, DeepTeacher, TeacherConfig


def _small_config() -> TeacherConfig:
    return TeacherConfig(initial_depth=2, depth_step=2, max_depth=6, max_nodes=20_000, max_time_s=2.0, tier=STANDARD)


def test_snapshot_then_rehydrate_preserves_entries(tmp_path: Path) -> None:
    source_dir = tmp_path / "source_cache"
    cache = AnnotationCache(source_dir)
    teacher = DeepTeacher(_small_config())

    games = [SongoLegacyGame(), SongoLegacyGame.from_board([5] * 14 + [0, 0], 2)]
    for game in games:
        cache.get_or_annotate(teacher, game)
    assert len(cache) == 2

    snapshot_path = tmp_path / "snapshot.db"
    cache.snapshot_to(snapshot_path)
    cache.close()

    # Simule une nouvelle VM : nouveau repertoire de cache, seed le
    # cache.db a partir du snapshot avant tout calcul (comme le fait
    # gcp_startup_script.sh via `gcloud storage cp` avant de lancer le build).
    resumed_dir = tmp_path / "resumed_cache"
    resumed_dir.mkdir()
    shutil.copy(snapshot_path, resumed_dir / "cache.db")

    resumed_cache = AnnotationCache(resumed_dir)
    assert len(resumed_cache) == 2
    for game in games:
        assert resumed_cache.get(game.zobrist_hash(), teacher.config) is not None


def test_snapshot_is_safe_while_more_entries_are_added_after(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = AnnotationCache(cache_dir)
    teacher = DeepTeacher(_small_config())

    cache.get_or_annotate(teacher, SongoLegacyGame())
    snapshot_path = tmp_path / "snap1.db"
    cache.snapshot_to(snapshot_path)

    # Le snapshot pris a ce moment ne doit pas etre affecte par des ecritures
    # ulterieures dans la base source (c'est une copie, pas une reference).
    board = [5] * 14 + [0, 0]
    cache.get_or_annotate(teacher, SongoLegacyGame.from_board(board, 2))
    assert len(cache) == 2

    snapshot_cache = AnnotationCache(tmp_path / "unused")
    snapshot_cache.close()
    import sqlite3

    conn = sqlite3.connect(str(snapshot_path))
    count = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    conn.close()
    assert count == 1
