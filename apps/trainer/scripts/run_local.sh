#!/bin/bash
# Raccourci macOS/Linux : equivalent de run_local.ps1.
#
#   apps/trainer/scripts/run_local.sh run build --positions 10000 --seed 456
#   apps/trainer/scripts/run_local.sh run tournament --a champion --b minimax:8
#   apps/trainer/scripts/run_local.sh config
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="$REPO/.venv/bin/python"
[ -x "$PYTHON" ] || { echo "venv introuvable : $PYTHON" >&2; exit 1; }

export SONGO_PROVIDER="local"
cd "$REPO"
exec "$PYTHON" -m songo_ai.cloud "$@"
