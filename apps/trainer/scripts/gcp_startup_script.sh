#!/bin/bash
# Startup script VM Compute Engine : palier 100k+ (etape 7).
# Garde-fous (section 11.3 du plan) :
#  - timeout dur en plus de --max-run-duration de la VM (double filet)
#  - upload du resultat (ou du log d'echec) vers Cloud Storage avant tout
#  - arret automatique de la VM en toute circonstance (succes, echec, timeout)
#
# Point de reprise : le cache d'annotations est un fichier SQLite unique
# (une entree par position, clef = hash Zobrist + config), pas un fichier
# JSON par position. Le premier run 100k (fichier-par-position, dossier
# plat) n'atteignait que ~45% d'efficacite parallele a 32 workers : la
# concurrence d'ecriture sur des dizaines de milliers de petits fichiers,
# plus un `gcloud storage rsync` qui devait relister ce dossier entier
# toutes les 5 minutes, ralentissait tout le monde. Un snapshot SQLite a
# chaud (API de sauvegarde, sans arreter les ecrivains) + upload d'UN SEUL
# fichier resout les deux problemes a la fois. Mesure locale : 5,1x sur 8
# workers (64% d'efficacite) contre lecture/ecriture fichier-par-fichier.
#
# L'echantillonnage des positions est deterministe (meme seed -> memes
# positions), donc relancer ce meme script sur une nouvelle VM re-hydrate
# le cache depuis GCS et saute automatiquement tout ce qui est deja
# annote : rien n'est perdu si la VM meurt en cours de route (panne,
# --max-run-duration atteint).
set -uo pipefail

BUCKET="gs://songo-model-ai-vertex-bucket-001"
LOG_FILE="/tmp/run.log"
NUM_POSITIONS=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/num-positions" || echo "100000")
OUT_DIR="/tmp/dataset_v002_${NUM_POSITIONS}"
CACHE_DIR="${OUT_DIR}/annotation_cache"
CACHE_DB="${CACHE_DIR}/cache.db"
CACHE_SNAPSHOT_GCS="${BUCKET}/pilots/annotation_cache_${NUM_POSITIONS}.db"
HARD_TIMEOUT_SECONDS=$((40 * 3600))  # 40h : marge de securite large
# (le --max-run-duration de la VM elle-meme doit etre fixe legerement
# au-dessus, pour laisser ce script finir son upload avant une terminaison forcee)
SYNC_INTERVAL_SECONDS=300  # snapshot + push du cache vers GCS toutes les 5 minutes

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
gcloud storage cp "$CACHE_SNAPSHOT_GCS" "$CACHE_DB" 2>>"$LOG_FILE" || echo "Aucun cache anterieur (premier lancement)."
if [ -f "$CACHE_DB" ]; then
  echo "Entrees de cache deja presentes: $(.venv/bin/python -c "
from songo_ai.teachers import AnnotationCache
from pathlib import Path
print(len(AnnotationCache(Path('$CACHE_DIR'))))
")"
fi

# Boucle de synchronisation en arriere-plan : snapshot SQLite a chaud + push
# vers GCS regulierement pendant tout le calcul, pas seulement a la fin.
(
  while true; do
    sleep "$SYNC_INTERVAL_SECONDS"
    .venv/bin/python apps/trainer/scripts/sync_cache.py "$CACHE_DIR" /tmp/cache_snapshot.db 2>>"$LOG_FILE" \
      && gcloud storage cp /tmp/cache_snapshot.db "$CACHE_SNAPSHOT_GCS" 2>>"$LOG_FILE" \
      && echo "$(date -u) : sync cache -> GCS"
  done
) &
SYNC_PID=$!

echo "=== Lancement du build ($NUM_POSITIONS positions) $(date -u) ==="
timeout "$HARD_TIMEOUT_SECONDS" .venv/bin/python apps/trainer/scripts/build_100k_gcp.py "$NUM_POSITIONS"
STATUS=$?
echo "=== Fin du build, code=$STATUS $(date -u) ==="

kill "$SYNC_PID" 2>/dev/null || true

echo "=== Sync final du cache (que le job ait reussi, echoue, ou timeout) $(date -u) ==="
.venv/bin/python apps/trainer/scripts/sync_cache.py "$CACHE_DIR" /tmp/cache_snapshot.db
gcloud storage cp /tmp/cache_snapshot.db "$CACHE_SNAPSHOT_GCS"

if [ -f "$OUT_DIR/manifest.json" ]; then
  gcloud storage cp "$OUT_DIR"/*.jsonl "$OUT_DIR/manifest.json" "$BUCKET/pilots/dataset_v002_${NUM_POSITIONS}/"
  echo "=== Release complete uploadee ==="
else
  echo "=== Pas de manifest.json : job incomplet, mais le cache est sauvegarde -> relancer reprendra ou il s'est arrete ==="
fi
gcloud storage cp "$LOG_FILE" "$BUCKET/pilots/run_status_${NUM_POSITIONS}_${STATUS}.log"

echo "=== Arret de la VM $(date -u) ==="
shutdown -h now
