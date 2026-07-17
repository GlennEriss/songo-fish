"""Tests du registre de versions de modeles (section 10.2/10.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from songo_ai.model import (  # noqa: E402
    EpochMetrics,
    SongoNet,
    build_manifest,
    get_champion,
    list_versions,
    load_registry,
    promote_version,
    register_model,
)


def _fake_history() -> list:
    return [
        EpochMetrics(1, train_loss=2.0, train_policy_top1=0.3, val_loss=2.1, val_policy_top1=0.28),
        EpochMetrics(2, train_loss=1.5, train_policy_top1=0.4, val_loss=1.6, val_policy_top1=0.35),
        EpochMetrics(3, train_loss=1.2, train_policy_top1=0.5, val_loss=1.8, val_policy_top1=0.30),
    ]


def test_build_manifest_picks_best_val_epoch_not_last(tmp_path: Path) -> None:
    dataset_manifest = tmp_path / "manifest.json"
    dataset_manifest.write_text(json.dumps({"total_positions": 1000, "checksums_sha256": {"train": "abc"}}))

    manifest = build_manifest(
        version="0.1.0",
        architecture={"width": 128, "num_blocks": 3},
        dataset_manifest_path=dataset_manifest,
        training_config={"epochs": 3, "lr": 1e-3},
        history=_fake_history(),
    )

    assert manifest.version == "0.1.0"
    assert manifest.metrics["best_epoch"] == 2  # meilleur val_loss, pas la derniere epoque
    assert manifest.dataset["total_positions"] == 1000


def test_register_model_first_version_becomes_champion(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    manifest = build_manifest(
        version="0.0.1",
        architecture={"width": 128},
        dataset_manifest_path=tmp_path / "missing.json",
        training_config={},
        history=_fake_history(),
    )
    registry = register_model(manifest, checkpoint_path=tmp_path / "model.pt", registry_path=registry_path)

    assert registry["champion"] == "0.0.1"
    assert "0.0.1" in registry["versions"]
    assert list_versions(registry_path) == ["0.0.1"]


def test_register_model_second_version_does_not_auto_promote(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    m1 = build_manifest("0.0.1", {}, tmp_path / "x.json", {}, _fake_history())
    m2 = build_manifest("0.1.0", {}, tmp_path / "x.json", {}, _fake_history())

    register_model(m1, tmp_path / "a.pt", registry_path=registry_path)
    registry = register_model(m2, tmp_path / "b.pt", registry_path=registry_path)

    assert registry["champion"] == "0.0.1"  # pas de promotion automatique
    assert set(registry["versions"].keys()) == {"0.0.1", "0.1.0"}


def test_promote_version_changes_champion(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    m1 = build_manifest("0.0.1", {}, tmp_path / "x.json", {}, _fake_history())
    m2 = build_manifest("0.1.0", {}, tmp_path / "x.json", {}, _fake_history())
    register_model(m1, tmp_path / "a.pt", registry_path=registry_path)
    register_model(m2, tmp_path / "b.pt", registry_path=registry_path)

    registry = promote_version("0.1.0", registry_path=registry_path)
    assert registry["champion"] == "0.1.0"
    assert get_champion(registry_path)["version"] == "0.1.0"


def test_promote_unknown_version_raises(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    m1 = build_manifest("0.0.1", {}, tmp_path / "x.json", {}, _fake_history())
    register_model(m1, tmp_path / "a.pt", registry_path=registry_path)

    with pytest.raises(ValueError):
        promote_version("9.9.9", registry_path=registry_path)


def test_get_champion_none_when_registry_empty(tmp_path: Path) -> None:
    assert get_champion(tmp_path / "does_not_exist.json") is None
