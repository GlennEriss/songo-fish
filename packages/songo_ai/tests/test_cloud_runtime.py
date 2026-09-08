"""Runtime multi-provider (`songo_ai.cloud`) : resolution de config,
backend de stockage local, et un build->train->tournoi de bout en bout via
`LocalProvider` sur un tout petit corpus."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from songo_ai.cloud import (
    BuildSpec,
    LocalProvider,
    LocalStore,
    MatchSpec,
    TrainSpec,
    load_config,
    make_provider,
    make_store,
)

_SCRIPTS = Path(__file__).resolve().parents[3] / "apps" / "trainer" / "scripts"


def test_config_defaults_to_local():
    cfg = load_config()
    assert cfg.provider == "local"
    assert cfg.num_workers >= 1


def test_env_overrides_win_over_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("SONGO_PROVIDER", "gcp")
    monkeypatch.setenv("SONGO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SONGO_NUM_WORKERS", "1")
    monkeypatch.setenv("SONGO_GCS_BUCKET", "gs://bucket-de-test")
    cfg = load_config()
    assert cfg.provider == "gcp"
    assert cfg.data_root == tmp_path
    assert cfg.num_workers == 1
    assert cfg.gcp.bucket == "gs://bucket-de-test"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("SONGO_PROVIDER", "aws")
    with pytest.raises(ValueError):
        load_config()


def test_local_store_resolve_and_noop_sync(tmp_path):
    store = LocalStore(tmp_path)
    path = store.resolve("datasets/foo/train.jsonl")
    assert path == tmp_path / "datasets/foo/train.jsonl"
    assert path.parent.is_dir()  # dossiers parents crees
    assert store.push("datasets/foo") is None  # no-op
    assert store.pull("datasets/foo") == tmp_path / "datasets/foo"
    assert not store.exists("datasets/foo/train.jsonl")


def test_make_provider_and_store_follow_config(monkeypatch, tmp_path):
    monkeypatch.setenv("SONGO_DATA_ROOT", str(tmp_path))
    cfg = load_config()
    assert isinstance(make_store(cfg), LocalStore)
    assert isinstance(make_provider(cfg), LocalProvider)


def test_gcp_provider_refuses_train_and_tournament(monkeypatch, tmp_path):
    monkeypatch.setenv("SONGO_PROVIDER", "gcp")
    monkeypatch.setenv("SONGO_DATA_ROOT", str(tmp_path))
    provider = make_provider(load_config())
    with pytest.raises(NotImplementedError):
        provider.run_train(TrainSpec(version="0.0.1", dataset_name="datasets/x"))
    with pytest.raises(NotImplementedError):
        provider.run_tournament(MatchSpec(agent_a="random", agent_b="random"))


def test_local_pipeline_build_train_tournament(monkeypatch, tmp_path):
    monkeypatch.setenv("SONGO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("SONGO_NUM_WORKERS", "2")
    provider = make_provider(load_config())

    manifest = provider.run_build(
        BuildSpec(num_positions=60, seed=3, teacher_preset="default", dataset_name="datasets/pipe")
    )
    assert manifest["total_positions"] > 0
    assert (tmp_path / "datasets/pipe/train.jsonl").exists()

    result = provider.run_train(
        TrainSpec(version="0.0.99", dataset_name="datasets/pipe", epochs=2, early_stopping_patience=2)
    )
    assert result["version"] == "0.0.99"
    assert (tmp_path / "checkpoints/model_v0.0.99.pt").exists()
    assert (tmp_path / "checkpoints/registry.json").exists()

    match = provider.run_tournament(
        MatchSpec(agent_a="v0.0.99", agent_b="random", num_games=4)
    )
    assert match["games"] == 4
    assert 0.0 <= match["win_rate_a"] <= 1.0


def test_multiprocessing_entrypoints_are_guarded():
    """Tout script qui lance un ProcessPoolExecutor doit avoir un
    `if __name__ == "__main__"` : Windows demarre ses workers par spawn
    (re-import du module), un appel non garde au niveau module = fork-bomb."""
    offenders = []
    for path in [*_SCRIPTS.glob("*.py"), Path(__file__).resolve().parents[1] / "dataset" / "build.py"]:
        text = path.read_text()
        if "ProcessPoolExecutor" in text and not re.search(r'__name__\s*==\s*[\'"]__main__[\'"]', text):
            # build.py est une bibliotheque : elle est appelee par un
            # entrypoint garde, jamais executee directement.
            if path.name != "build.py":
                offenders.append(path.name)
    assert not offenders, f"scripts non gardes: {offenders}"
