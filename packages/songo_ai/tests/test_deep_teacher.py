"""Tests du professeur adaptatif (etape 4 : porte de passage = resultats
reproductibles, cache operationnel)."""

from __future__ import annotations

from pathlib import Path

import pytest

from songo_ai.songo.rules import PLAYER_ONE, SongoLegacyGame
from songo_ai.teachers import ENRICHI, STANDARD, AnnotationCache, DeepTeacher, TeacherConfig


def _small_config(**overrides) -> TeacherConfig:
    base = dict(initial_depth=2, depth_step=2, max_depth=8, max_nodes=100_000, max_time_s=3.0, stability_window=2, min_margin=5.0)
    base.update(overrides)
    return TeacherConfig(**base)


def test_annotate_returns_legal_best_action() -> None:
    game = SongoLegacyGame()
    teacher = DeepTeacher(_small_config())
    annotation = teacher.annotate(game)
    assert annotation.best_action in game.legal_local_actions()
    assert annotation.depth >= 1
    assert annotation.nodes > 0
    assert annotation.tier == STANDARD


def test_annotate_is_reproducible() -> None:
    config = _small_config()
    r1 = DeepTeacher(config).annotate(SongoLegacyGame())
    r2 = DeepTeacher(config).annotate(SongoLegacyGame())
    assert r1.best_action == r2.best_action
    assert r1.depth == r2.depth
    assert r1.principal_variation == r2.principal_variation
    assert r1.action_values[r1.best_action] == r2.action_values[r2.best_action]


def test_stability_stops_before_max_depth_on_lopsided_position() -> None:
    # Un seul coup capture beaucoup ; les autres ne rapportent rien. Le
    # meilleur coup doit rester le meme sur plusieurs profondeurs d'affilee
    # et permettre un arret avant max_depth.
    board = [0, 0, 0, 0, 4, 1, 1, 1, 1, 0, 0, 0, 0, 10, 0, 0]
    game = SongoLegacyGame.from_board(board, PLAYER_ONE)
    teacher = DeepTeacher(_small_config(max_depth=12, stability_window=2, min_margin=5.0))
    annotation = teacher.annotate(game)
    assert annotation.depth < 12


def test_enrichi_tier_produces_margin_when_multiple_moves() -> None:
    game = SongoLegacyGame()
    teacher = DeepTeacher(_small_config(tier=ENRICHI))
    annotation = teacher.annotate(game)
    assert annotation.tier == ENRICHI
    assert annotation.top1_top2_margin is not None
    non_null_values = [v for v in annotation.action_values.values() if v is not None]
    assert len(non_null_values) > 1


def test_is_exact_true_when_principal_variation_ends_the_game() -> None:
    # Un seul coup legal (case 6, 2 graines, transmet chez l'adversaire) ;
    # avec seulement 2 graines en jeu, la partie se termine par famine
    # quelques coups plus tard : la PV doit atteindre un etat termine.
    board = [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    game = SongoLegacyGame.from_board(board, PLAYER_ONE)
    assert not game.finished
    teacher = DeepTeacher(_small_config(initial_depth=1, max_depth=4))
    annotation = teacher.annotate(game)
    assert annotation.is_exact is True


def test_annotate_rejects_finished_position() -> None:
    board = [0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 10, 15]
    game = SongoLegacyGame.from_board(board, PLAYER_ONE)
    game.play(0)  # termine par manque de graines cote J1
    assert game.finished
    with pytest.raises(ValueError):
        DeepTeacher(_small_config()).annotate(game)


def test_annotation_cache_round_trip(tmp_path: Path) -> None:
    game = SongoLegacyGame()
    teacher = DeepTeacher(_small_config())
    cache = AnnotationCache(tmp_path)

    first = cache.get_or_annotate(teacher, game)
    assert cache.get(game.zobrist_hash(), teacher.config) is not None

    second = cache.get_or_annotate(teacher, game)
    assert first.best_action == second.best_action
    assert first.principal_variation == second.principal_variation
    assert first.depth == second.depth


def test_annotation_cache_distinguishes_configs(tmp_path: Path) -> None:
    game = SongoLegacyGame()
    cache = AnnotationCache(tmp_path)
    teacher_a = DeepTeacher(_small_config(tier=STANDARD))
    teacher_b = DeepTeacher(_small_config(tier=ENRICHI))

    cache.get_or_annotate(teacher_a, game)
    assert cache.get(game.zobrist_hash(), teacher_b.config) is None
