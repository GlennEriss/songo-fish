import pytest

from lot45_test_support import ROOT


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    # pool_identity()/engine_fingerprint() resolvent des chemins relatifs au depot.
    monkeypatch.chdir(ROOT)
