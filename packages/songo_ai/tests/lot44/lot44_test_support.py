import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "apps/trainer/scripts"
for entry in (str(SCRIPTS), str(ROOT / "packages")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

ORIGINAL_POSITIONS = ROOT / "data/colab_bridge/lot39_benchmark_positions.json"
IDENTITY = ROOT / "data/experiments/lot35_generator_pool/g4_champion_identity.json"



def require_inputs():
    missing = [str(p) for p in (ORIGINAL_POSITIONS, IDENTITY) if not p.is_file()]
    if missing:
        pytest.skip(f"Lot44 inputs not extracted (lot44_inputs.tar.gz): {missing}")


def drive_experiments() -> Path | None:
    from lot44.paths import resolve_drive_root

    root, _ = resolve_drive_root(None)
    if root is None:
        return None
    path = root / "songo-ai/experiments"
    return path if path.is_dir() else None



def make_context(out: Path, **overrides):
    from lot44.config import DEFAULT_SEED, SMOKE_BUDGETS
    from lot44.pipeline import Context

    values = dict(out=out, budgets=dict(SMOKE_BUDGETS), seed=DEFAULT_SEED, device_name="cpu", max_positions=48, candidates_file=out / "none.json", original_positions=ORIGINAL_POSITIONS, lot41=None, lot42=None, lot43=None, smoke=True, concurrency_override=8, log=lambda _: None)
    values.update(overrides)
    return Context(**values)
