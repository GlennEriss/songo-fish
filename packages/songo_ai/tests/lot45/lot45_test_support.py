import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "apps/trainer/scripts"
for entry in (str(SCRIPTS), str(ROOT / "packages")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

PROBE = ROOT / "data/colab_bridge/lot39_benchmark_positions.json"
IDENTITY = ROOT / "data/experiments/lot35_generator_pool/g4_champion_identity.json"



def require_inputs():
    missing = [str(p) for p in (PROBE, IDENTITY) if not p.is_file()]
    if missing:
        pytest.skip(f"Lot45 inputs not extracted: {missing}")


def make_context(out: Path, **overrides):
    from lot45.config import DEFAULT_SEED, SMOKE_BUDGET, SMOKE_SHARD_SIZE
    from lot45.generation import Context

    values = dict(out=out, seed=DEFAULT_SEED, device_name="cpu", selection_file=out / "selection.jsonl.gz", original_positions=PROBE, lot44=None, budget=SMOKE_BUDGET, shard_size=SMOKE_SHARD_SIZE, concurrency=SMOKE_SHARD_SIZE, smoke=True, lock_settle_s=0.0, log=lambda _: None)
    values.update(overrides)
    return Context(**values)
