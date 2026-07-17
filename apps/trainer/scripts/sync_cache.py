#!/usr/bin/env python3
"""Snapshot a chaud du cache SQLite (section 11.3 : synchronisation continue
vers GCS pendant le calcul, sans arreter les workers qui ecrivent encore).
Utilise par gcp_startup_script.sh, appele periodiquement en arriere-plan."""

from __future__ import annotations

import sys
from pathlib import Path

from songo_ai.teachers import AnnotationCache


def main() -> None:
    cache_dir = Path(sys.argv[1])
    snapshot_path = Path(sys.argv[2])
    cache = AnnotationCache(cache_dir)
    count = len(cache)
    cache.snapshot_to(snapshot_path)
    cache.close()
    print(f"snapshot: {count} entrees -> {snapshot_path}")


if __name__ == "__main__":
    main()
