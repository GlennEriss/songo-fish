import pytest

from lot44_test_support import ROOT


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    # pool_identity()/engine_fingerprint() resolvent des chemins relatifs au depot.
    monkeypatch.chdir(ROOT)


@pytest.fixture(scope="session")
def model():
    import os

    import torch
    from run_srn_lot39 import load_model

    from lot44_test_support import require_inputs

    require_inputs()
    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        return load_model(torch.device("cpu"))
    finally:
        os.chdir(previous)
