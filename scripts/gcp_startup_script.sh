#!/bin/bash
# Startup script VM Compute Engine : palier 100k (etape 7).
# Garde-fous (section 11.3 du plan) :
#  - timeout dur en plus de --max-run-duration de la VM (double filet)
#  - upload du resultat (ou du log d'echec) vers Cloud Storage avant tout
#  - arret automatique de la VM en toute circonstance (succes, echec, timeout)
#
# Point de reprise : le cache d'annotations (une entree par position,
# clef = hash Zobrist + config) est synchronise vers GCS EN CONTINU
# (pas seulement a la fin). L'echantillonnage des positions est deterministe
# (meme seed -> memes positions), donc relancer ce meme script sur une
# nouvelle VM re-hydrate le cache depuis GCS et saute automatiquement tout
# ce qui est deja annote : rien n'est perdu si la VM meurt en cours de route
# (preemption, panne, --max-run-duration atteint).
set -uo pipefail

BUCKET="gs://songo-model-ai-vertex-bucket-001"
LOG_FILE="/tmp/run.log"
NUM_POSITIONS=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/num-positions" || echo "100000")
OUT_DIR="/tmp/dataset_v002_${NUM_POSITIONS}"
CACHE_DIR="${OUT_DIR}/annotation_cache"
CACHE_GCS="${BUCKET}/pilots/annotation_cache_${NUM_POSITIONS}/"
HARD_TIMEOUT_SECONDS=$((40 * 3600))  # 40h : marge de securite au-dessus des ~26h estimees pour 100k
# (le --max-run-duration de la VM elle-meme est fixe a 41h, legerement au-dessus,
# pour laisser ce script finir son upload avant une terminaison forcee)
SYNC_INTERVAL_SECONDS=300  # push du cache vers GCS toutes les 5 minutes

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Debut $(date -u) ==="

apt-get update -y
apt-get install -y python3.12 python3.12-venv python3-pip

gcloud storage cp "$BUCKET/pilots/songo_ai_code.tar.gz" /tmp/songo_ai_code.tar.gz
mkdir -p /opt/songo
tar -xzf /tmp/songo_ai_code.tar.gz -C /opt/songo
cd /opt/songo

python3.12 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[perf]"

echo "=== Rehydratation du cache depuis GCS (reprise eventuelle) $(date -u) ==="
mkdir -p "$CACHE_DIR"
gcloud storage rsync -r "$CACHE_GCS" "$CACHE_DIR" || echo "Aucun cache anterieur (premier lancement) ou rsync vide."
echo "Entrees de cache deja presentes: $(find "$CACHE_DIR" -name '*.json' | wc -l)"

# Boucle de synchronisation en arriere-plan : pousse le cache local vers GCS
# regulierement pendant tout le calcul, pas seulement a la fin.
(
  while true; do
    sleep "$SYNC_INTERVAL_SECONDS"
    gcloud storage rsync -r "$CACHE_DIR" "$CACHE_GCS" 2>>"$LOG_FILE"
    echo "$(date -u) : sync cache -> GCS ($(find "$CACHE_DIR" -name '*.json' | wc -l) entrees)"
  done
) &
SYNC_PID=$!

echo "=== Lancement du build ($NUM_POSITIONS positions) $(date -u) ==="
timeout "$HARD_TIMEOUT_SECONDS" .venv/bin/python scripts/build_100k_gcp.py "$NUM_POSITIONS"
STATUS=$?
echo "=== Fin du build, code=$STATUS $(date -u) ==="

kill "$SYNC_PID" 2>/dev/null || true

echo "=== Sync final du cache (que le job ait reussi, echoue, ou timeout) $(date -u) ==="
gcloud storage rsync -r "$CACHE_DIR" "$CACHE_GCS"

if [ -f "$OUT_DIR/manifest.json" ]; then
  gcloud storage cp "$OUT_DIR"/*.jsonl "$OUT_DIR/manifest.json" "$BUCKET/pilots/dataset_v002_${NUM_POSITIONS}/"
  echo "=== Release complete uploadee ==="
else
  echo "=== Pas de manifest.json : job incomplet, mais le cache est sauvegarde -> relancer reprendra ou il s'est arrete ==="
fi
gcloud storage cp "$LOG_FILE" "$BUCKET/pilots/run_status_${NUM_POSITIONS}_${STATUS}.log"

echo "=== Arret de la VM $(date -u) ==="
shutdown -h now
