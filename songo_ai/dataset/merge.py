"""Fusion de plusieurs releases en une seule (section 6.6 : releases
immuables avec manifeste et checksums). Les splits train/val/test de
chaque release source ont deja ete decides par partie entiere ; fusionner
des releases DIFFERENTES (seeds/trajectory_id distincts par construction)
ne cree pas de fuite train/val/test tant qu'on ne fait que concatener les
memes splits entre eux, sans jamais reordonner une position individuelle
d'un split vers un autre."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List

from .schema import DATASET_VERSION, RULES_VERSION


def merge_releases(release_dirs: List[Path], out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    checksums: Dict[str, str] = {}
    source_manifests = []

    for split in ("train", "val", "test"):
        shard_path = out_dir / f"{split}.jsonl"
        with shard_path.open("w") as out_f:
            for release_dir in release_dirs:
                release_dir = Path(release_dir)
                src = release_dir / f"{split}.jsonl"
                if not src.exists():
                    continue
                with src.open() as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        out_f.write(line + "\n")
                        counts[split] += 1
        checksums[split] = hashlib.sha256(shard_path.read_bytes()).hexdigest()

    for release_dir in release_dirs:
        manifest_path = Path(release_dir) / "manifest.json"
        if manifest_path.exists():
            source_manifests.append(json.loads(manifest_path.read_text()))

    manifest = {
        "dataset_version": DATASET_VERSION,
        "rules_version": RULES_VERSION,
        "generated_at_unix": time.time(),
        "merged_from": [str(d) for d in release_dirs],
        "total_positions": sum(counts.values()),
        "counts_per_split": counts,
        "checksums_sha256": checksums,
        "source_manifests_summary": [
            {
                "path": str(d),
                "total_positions": m.get("total_positions"),
                "teacher_config": m.get("teacher_config"),
            }
            for d, m in zip(release_dirs, source_manifests)
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
