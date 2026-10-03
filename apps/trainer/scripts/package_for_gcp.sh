#!/bin/bash
# Empaquette le code (package partage + app trainer) et le pousse vers le
# bucket GCS utilise par gcp_startup_script.sh. A lancer depuis la racine
# du repo apres tout changement de code devant tourner sur une VM GCP.
set -euo pipefail

# Parametrable via env (songo_ai.cloud.providers.GcpProvider les passe) ;
# defauts historiques conserves.
BUCKET="${SONGO_GCS_BUCKET:-gs://songo-model-ai-vertex-bucket-001}"
PROJECT="${SONGO_GCP_PROJECT:-songo-model-ai}"
TARBALL="/tmp/songo_ai_code.tar.gz"

cd "$(dirname "$0")/../../.."  # racine du repo (apps/trainer/scripts/.. .. ..)

tar --exclude='.venv' --exclude='.git' --exclude='data' --exclude='__pycache__' \
    --exclude='*.egg-info' --exclude='.pytest_cache' \
    -czf "$TARBALL" packages/songo_ai apps/trainer/scripts pyproject.toml

gcloud storage cp "$TARBALL" "$BUCKET/pilots/songo_ai_code.tar.gz" --project "$PROJECT"

echo "Code empaquete et uploade : $BUCKET/pilots/songo_ai_code.tar.gz"
